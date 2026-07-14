"""Production orchestration for NEON Stage-2 training (Gap 4).

Composes the tested Stage-2 building blocks into a single ``run_neon_stage2_training``
pipeline. The two infrastructure-dependent steps -- loading the frozen Stage-1
FGNO checkpoint and building the grouped-hydrograph family splits -- are
injected callables, so the orchestration is fully testable with fakes while the
real adapters are validated by a GPU smoke against a concrete checkpoint/dataset.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from neuralop.flood.neon import (
    NEONStage2LossWeights,
    PersistentDirichletParticleControl,
    base_rmse_from_reference,
    calibrate_prior_scale,
    freeze_stage1_model,
    sample_epistemic_indices,
)
from neuralop.flood.train.neon import (
    FrozenFGNOFeatureBatch,
    NEONFamilySample,
    NEONTrainingProgressReporter,
    build_epinet_from_config,
    build_neon_stage2_metadata,
    build_neon_stage2_optimizer,
    collect_frozen_fgno_rollout_features,
    load_neon_stage2_training_state,
    train_neon_stage2_epochs,
)

# load_stage1_fn(checkpoint) -> frozen-able nn.Module
LoadStage1Fn = Callable[[Any], torch.nn.Module]
# build_families_fn(data_root, config) -> (train_families, val_families)
BuildFamiliesFn = Callable[[Any, Any], Tuple[Sequence[NEONFamilySample], Sequence[NEONFamilySample]]]
_DEFAULT_MEMORY_CACHE_LIMIT_BYTES = 64 * 1024**3


def _stable_latent_bank_seed(
    *,
    seed: int,
    family_id: str,
    latent_bank_id: int,
    num_aleatory: int,
    latent_dim: int,
) -> int:
    """Derive deterministic latent-bank seeds independent of collector call order."""

    payload = (
        f"{int(seed)}::{family_id}::{int(latent_bank_id)}::"
        f"{int(num_aleatory)}::{int(latent_dim)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _stable_canonical_latent_seed(*, seed: int, canonical_k: int, latent_dim: int) -> int:
    """Derive one canonical-bank seed shared by every family and evaluation run."""

    payload = f"canonical::{int(seed)}::{int(canonical_k)}::{int(latent_dim)}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _generator_initial_seed(generator: Optional[torch.Generator]) -> int:
    if generator is None:
        return 0
    return int(generator.initial_seed())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _normalizer_physical_scale(normalizer: Any) -> float:
    """Return the affine output scale used to convert normalized WD errors to meters."""

    if normalizer is None:
        return 1.0
    std = getattr(normalizer, "std", None)
    if std is None:
        return 1.0
    values = torch.as_tensor(std).detach().reshape(-1)
    if values.numel() < 1:
        return 1.0
    if not torch.allclose(values, values[0].expand_as(values), rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(
            "a scalar physical prior-spread target requires a uniform affine target "
            "normalizer scale; found spatial/channel-varying std values."
        )
    scale = float(values[0].item())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"target normalizer has invalid physical scale {scale!r}.")
    return scale


def _estimate_cached_feature_bytes(
    probe: FrozenFGNOFeatureBatch,
    *,
    n_families: int,
    latent_bank_count: int,
    cache_dtype: torch.dtype,
) -> int:
    """Estimate RAM required for in-memory frozen-feature cache entries."""

    bytes_per_value = torch.empty((), dtype=cache_dtype).element_size()
    base_bytes = int(probe.base_prediction.numel()) * bytes_per_value
    feature_bytes = int(probe.features.numel()) * bytes_per_value
    latent_bytes = _tensor_nbytes(probe.aleatory_latents)
    canonical_bytes = (
        0
        if probe.canonical_mean_features is None
        else int(probe.canonical_mean_features.numel()) * bytes_per_value
    )
    return (
        base_bytes + feature_bytes + latent_bytes + canonical_bytes
    ) * int(n_families) * int(latent_bank_count)


def make_feature_collector_from_frozen_model(
    stage1_model: torch.nn.Module,
    *,
    feature_source: str,
    n_history: int,
    latent_dim: int,
    generator: Optional[torch.Generator] = None,
    canonical_k: int = 0,
    canonical_seed: int = 123,
    canonical_zero_latent: bool = False,
    canonical_cache_dir: Optional[Any] = None,
    target_normalizer=None,
):
    """Return a ``feature_collector`` that AR-rolls the frozen FGNO per family.

    The returned callable matches the training loop's collector contract:
    ``(family, *, num_aleatory, generator, latent_bank_id) -> FrozenFGNOFeatureBatch``.
    It samples
    ``num_aleatory`` persistent aleatory latents, warm-starts from the family's
    ``initial_histories`` (broadcast to K if a single shared history is given),
    and rolls out to the family's reference horizon ``T``.
    """

    canonical_memory_cache: dict[str, tuple[torch.Tensor, str]] = {}
    canonical_disk_dir = (
        None if canonical_cache_dir is None else Path(canonical_cache_dir)
    )
    if canonical_disk_dir is not None:
        canonical_disk_dir.mkdir(parents=True, exist_ok=True)

    def collector(
        family: NEONFamilySample,
        *,
        num_aleatory: int,
        generator=generator,
        latent_bank_id: int | None = None,
    ):
        if family.initial_histories is None:
            raise ValueError(f"family {family.family_id!r} lacks initial_histories for rollout.")
        # The collector owns device placement: move all inputs to the frozen
        # model's device so the AR rollout is device-consistent on GPU.
        try:
            model_device = next(stage1_model.parameters()).device
        except StopIteration:  # parameter-less model (shouldn't happen for GINO)
            model_device = torch.device("cpu")
        bank = 0 if latent_bank_id is None else int(latent_bank_id)
        latent_generator = torch.Generator(device="cpu")
        latent_generator.manual_seed(
            _stable_latent_bank_seed(
                seed=_generator_initial_seed(generator),
                family_id=str(family.family_id),
                latent_bank_id=bank,
                num_aleatory=int(num_aleatory),
                latent_dim=int(latent_dim),
            )
        )
        latents = torch.randn(
            int(num_aleatory),
            int(latent_dim),
            generator=latent_generator,
        ).to(model_device)
        init = family.initial_histories.to(model_device)
        if init.ndim == 3:  # [n_history, Nv, C] -> broadcast to K members
            init = init.unsqueeze(0).expand(int(num_aleatory), -1, -1, -1).contiguous()
        if target_normalizer is not None:
            target_normalizer.to(model_device)

        def collect(latent_bank: torch.Tensor, histories: torch.Tensor):
            return collect_frozen_fgno_rollout_features(
                stage1_model=stage1_model,
                static=family.static.to(model_device),
                geometry=family.geometry.to(model_device),
                query_points=family.query_points.to(model_device),
                boundary_sequence=family.boundary_sequence.to(model_device),
                initial_histories=histories,
                aleatory_latents=latent_bank,
                rollout_length=int(family.reference.shape[1]),
                n_history=int(n_history),
                feature_source=feature_source,
                structural_dry_mask=(
                    None
                    if family.structural_dry_mask is None
                    else family.structural_dry_mask.to(model_device)
                ),
                target_normalizer=target_normalizer,
            )

        batch = collect(latents, init)
        if int(canonical_k) > 0:
            k0 = int(canonical_k)
            if canonical_zero_latent:
                canonical_latents = torch.zeros(1, int(latent_dim))
                k0 = 1
            else:
                canonical_generator = torch.Generator(device="cpu")
                canonical_generator.manual_seed(
                    _stable_canonical_latent_seed(
                        seed=int(canonical_seed),
                        canonical_k=k0,
                        latent_dim=int(latent_dim),
                    )
                )
                half = k0 // 2
                positive = torch.randn(half, int(latent_dim), generator=canonical_generator)
                canonical_latents = torch.cat([positive, -positive], dim=0)
                if canonical_latents.shape[0] < k0:
                    canonical_latents = torch.cat(
                        [canonical_latents, torch.zeros(1, int(latent_dim))], dim=0
                    )
            canonical_latents = canonical_latents[:k0].to(model_device)
            canonical_bytes = canonical_latents.detach().to("cpu").contiguous().numpy().tobytes()
            canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()
            cache_token = hashlib.sha256(str(family.family_id).encode("utf-8")).hexdigest()[:20]
            canonical_path = (
                None
                if canonical_disk_dir is None
                else canonical_disk_dir / f"{cache_token}_{canonical_hash[:16]}.pt"
            )
            cached = canonical_memory_cache.get(str(family.family_id))
            if cached is None and canonical_path is not None and canonical_path.exists():
                payload = torch.load(canonical_path, map_location="cpu")
                if payload.get("latent_hash") != canonical_hash:
                    raise ValueError("canonical feature-cache latent hash mismatch.")
                cached = (payload["features"], canonical_hash)
            if cached is None:
                canonical_init = family.initial_histories.to(model_device)
                if canonical_init.ndim == 3:
                    canonical_init = canonical_init.unsqueeze(0).expand(
                        k0, -1, -1, -1
                    ).contiguous()
                canonical_batch = collect(canonical_latents, canonical_init)
                canonical_features = canonical_batch.features.mean(dim=1)
                cached_cpu = canonical_features.detach().to(device="cpu", dtype=torch.float16)
                cached = (cached_cpu, canonical_hash)
                if canonical_path is None:
                    canonical_memory_cache[str(family.family_id)] = cached
                else:
                    tmp = canonical_path.with_suffix(f".tmp{os.getpid()}.pt")
                    torch.save(
                        {"features": cached_cpu, "latent_hash": canonical_hash}, tmp
                    )
                    os.replace(tmp, canonical_path)
            batch.canonical_mean_features = cached[0].to(
                device=batch.features.device, dtype=batch.features.dtype
            )
            batch.canonical_latent_hash = canonical_hash
        return batch

    return collector


def make_cached_feature_collector(
    base_collector,
    *,
    cache_device: str | torch.device = "cpu",
    cache_dtype: torch.dtype = torch.float16,
    cache_dir: Optional[Any] = None,
    cache_key: Optional[str] = None,
    prefetch_workers: int = 0,
    prefetch_depth: int = 0,
):
    """Wrap a feature collector so each family's frozen rollout runs only once.

    The frozen-FGNO AR rollout is the dominant per-epoch cost; the epoch loop
    otherwise recollects it every epoch. This wrapper computes each family's
    ``(base_prediction, features)`` on first sight, stashes them on
    ``cache_device`` (CPU by default) in ``cache_dtype`` (fp16 to halve the
    footprint), and on subsequent epochs returns the cached tensors moved back
    to the original compute device/dtype. Multiple ``latent_bank_id`` values
    create distinct cached K-member aleatory feature banks per family, avoiding
    accidental reuse of a single frozen latent bank across all Stage-2 epochs.

    With ``cache_dir`` set, entries are stored as one ``<family_id>.pt`` file
    per family instead of in RAM (written atomically via tmp+rename so
    concurrent jobs can share a cache). Use this when the full family set does
    not fit in memory (e.g. 500 families at T=94 is ~570 GB fp16).
    """
    prefetch_workers = max(0, int(prefetch_workers))
    prefetch_depth = max(0, int(prefetch_depth))
    cache: dict[str, tuple] = {}
    dir_path = None
    if cache_dir is not None:
        dir_path = Path(cache_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
    key_prefix = "" if cache_key is None else f"{str(cache_key)}_"
    executor = (
        ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="neon-cache")
        if dir_path is not None and prefetch_workers > 0 and prefetch_depth > 0
        else None
    )
    prefetch_futures: dict[Path, Future] = {}
    prefetch_lock = threading.Lock()

    def _entry_path(family_id: str, *, num_aleatory: int, latent_bank_id: int) -> Path:
        if dir_path is None:
            raise RuntimeError("disk cache path requested for an in-memory collector")
        safe_key = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in str(family_id)
        )
        return dir_path / (
            f"{key_prefix}{safe_key}_bank{int(latent_bank_id)}_k{int(num_aleatory)}.pt"
        )

    def _load_payload(path: Path):
        return torch.load(path, map_location="cpu")

    def _move_for_compute(tensor: torch.Tensor, *, device: torch.device, dtype=None):
        # Keep the cache dtype during the host-to-device copy, then cast on the
        # accelerator. This avoids a large fp32 CPU intermediate on every hit.
        moved = tensor.to(device=device, non_blocking=True)
        if dtype is not None and moved.dtype != dtype:
            moved = moved.to(dtype=dtype)
        return moved

    def _prefetch(
        family: NEONFamilySample,
        *,
        num_aleatory: int,
        latent_bank_id: int | None = None,
    ) -> bool:
        if executor is None:
            return False
        bank = 0 if latent_bank_id is None else int(latent_bank_id)
        path = _entry_path(
            str(family.family_id),
            num_aleatory=int(num_aleatory),
            latent_bank_id=bank,
        )
        if not path.exists():
            return False
        with prefetch_lock:
            if path not in prefetch_futures:
                prefetch_futures[path] = executor.submit(_load_payload, path)
        return True

    def _take_prefetched_or_load(path: Path):
        with prefetch_lock:
            future = prefetch_futures.pop(path, None)
        return future.result() if future is not None else _load_payload(path)

    def _close() -> None:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with prefetch_lock:
            prefetch_futures.clear()

    def _store(batch) -> None:
        payload = {
            "base": batch.base_prediction.detach().to(device=cache_device, dtype=cache_dtype),
            "features": batch.features.detach().to(device=cache_device, dtype=cache_dtype),
            "latents": batch.aleatory_latents.detach().to(device=cache_device),
            "canonical_mean_features": (
                None
                if batch.canonical_mean_features is None
                else batch.canonical_mean_features.detach().to(
                    device=cache_device, dtype=cache_dtype
                )
            ),
            "canonical_latent_hash": batch.canonical_latent_hash,
            "device": str(batch.features.device),
            "dtype": batch.features.dtype,
        }
        return payload

    def collector(
        family: NEONFamilySample,
        *,
        num_aleatory: int,
        generator=None,
        latent_bank_id: int | None = None,
    ):
        key = str(family.family_id)
        bank = 0 if latent_bank_id is None else int(latent_bank_id)
        cache_key_full = f"{key}|bank{bank}|k{int(num_aleatory)}"
        if dir_path is not None:
            fpath = _entry_path(
                key,
                num_aleatory=int(num_aleatory),
                latent_bank_id=bank,
            )
            if fpath.exists():
                p = _take_prefetched_or_load(fpath)
                dev, dt = torch.device(p["device"]), p["dtype"]
                return FrozenFGNOFeatureBatch(
                    base_prediction=_move_for_compute(p["base"], device=dev, dtype=dt),
                    features=_move_for_compute(p["features"], device=dev, dtype=dt),
                    aleatory_latents=_move_for_compute(p["latents"], device=dev),
                    canonical_mean_features=(
                        None
                        if p.get("canonical_mean_features") is None
                        else _move_for_compute(
                            p["canonical_mean_features"], device=dev, dtype=dt
                        )
                    ),
                    canonical_latent_hash=p.get("canonical_latent_hash"),
                )
            batch = base_collector(
                family,
                num_aleatory=num_aleatory,
                generator=generator,
                latent_bank_id=bank,
            )
            tmp = fpath.with_suffix(f".tmp{os.getpid()}")
            torch.save(_store(batch), tmp)
            os.replace(tmp, fpath)
            return batch
        hit = cache.get(cache_key_full)
        if hit is None:
            batch = base_collector(
                family,
                num_aleatory=num_aleatory,
                generator=generator,
                latent_bank_id=bank,
            )
            p = _store(batch)
            cache[cache_key_full] = (
                p["base"],
                p["features"],
                p["latents"],
                p["canonical_mean_features"],
                p["canonical_latent_hash"],
                batch.features.device,
                batch.features.dtype,
            )
            return batch
        base_c, feat_c, lat_c, canonical_c, canonical_hash, dev, dt = hit
        return FrozenFGNOFeatureBatch(
            base_prediction=base_c.to(device=dev, dtype=dt),
            features=feat_c.to(device=dev, dtype=dt),
            aleatory_latents=lat_c.to(device=dev),
            canonical_mean_features=(
                None if canonical_c is None else canonical_c.to(device=dev, dtype=dt)
            ),
            canonical_latent_hash=canonical_hash,
        )

    collector.prefetch = _prefetch
    collector.close = _close
    collector.prefetch_depth = prefetch_depth
    return collector


def run_neon_stage2_training(
    *,
    config: Any,
    stage1_checkpoint: Any,
    output_dir: Any,
    data_root: Any,
    load_stage1_fn: LoadStage1Fn,
    build_families_fn: BuildFamiliesFn,
    latent_dim: int,
    n_history: int = 3,
    out_channels: int = 1,
    hidden_channels: int = 64,
    generator: Optional[torch.Generator] = None,
    calibrate_prior: bool = True,
    normalizer_fingerprint: Optional[dict] = None,
    structural_dry_policy: Any = None,
    stage1_alias: str = "best_model",
    cache_features: bool = False,
    cache_device: str = "cpu",
    cache_dir: Optional[Any] = None,
    memory_cache_limit_bytes: Optional[int] = _DEFAULT_MEMORY_CACHE_LIMIT_BYTES,
    epistemic_chunk_size: Optional[int] = None,
    val_seed: Optional[int] = None,
    prior_seed: Optional[int] = None,
    latest_checkpoint_path: Optional[Any] = None,
    resume_state_path: Optional[Any] = None,
):
    """End-to-end NEON Stage-2 training orchestration.

    Steps: load + freeze Stage-1, build family splits, build the feature
    collector, probe the frozen feature width, construct the EpiNet, optionally
    auto-calibrate the prior scale, and run the family-level epoch loop saving
    the best checkpoint with structured metadata. Returns the
    :class:`NEONTrainingResult`.
    """
    stage1 = load_stage1_fn(stage1_checkpoint)
    freeze_stage1_model(stage1)

    train_families, val_families = build_families_fn(data_root, config)
    if not train_families:
        raise ValueError("build_families_fn produced no training families.")
    dirichlet_particle_control = None
    if str(getattr(config, "epistemic_index_mode", "continuous")) == "dirichlet_particles":
        dirichlet_particle_control = PersistentDirichletParticleControl.create(
            [family.family_id for family in train_families],
            num_particles=int(getattr(config, "dirichlet_num_particles", 16)),
            seed=int(getattr(config, "dirichlet_particle_seed", 123)),
        )

    prepared = getattr(load_stage1_fn, "last_prepared", {}) or {}
    target_normalizer = (prepared.get("normalizers") or {}).get("target")
    reference_normalizer = (prepared.get("normalizers") or {}).get("dynamic")
    validation_physical_scale = _normalizer_physical_scale(target_normalizer)
    canonical_enabled = bool(getattr(config, "deterministic_head", False)) and str(
        getattr(config, "deterministic_head_feature", "")
    ).strip().lower() in {"canonical_aleatory_mean", "fixed_zero_latent"}
    canonical_zero_latent = str(
        getattr(config, "deterministic_head_feature", "")
    ).strip().lower() == "fixed_zero_latent"
    canonical_cache_dir = None
    if canonical_enabled and cache_features and cache_dir is not None:
        canonical_namespace = hashlib.sha256(
            "|".join(
                [
                    "neon_canonical_feature_v1",
                    str(stage1_checkpoint),
                    str(config.feature_source),
                    str(latent_dim),
                    str(n_history),
                    str(getattr(config, "deterministic_head_canonical_k", 32)),
                    str(getattr(config, "deterministic_head_latent_seed", 123)),
                    str(canonical_zero_latent),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        canonical_cache_dir = Path(cache_dir) / f"canonical_{canonical_namespace}"
    collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=config.feature_source,
        n_history=n_history,
        latent_dim=latent_dim,
        generator=generator,
        canonical_k=(
            int(getattr(config, "deterministic_head_canonical_k", 32))
            if canonical_enabled
            else 0
        ),
        canonical_seed=int(getattr(config, "deterministic_head_latent_seed", 123)),
        canonical_zero_latent=canonical_zero_latent,
        canonical_cache_dir=canonical_cache_dir,
        target_normalizer=target_normalizer,
    )
    # Wrap for caching before the probe so the probe populates the cache and the
    # frozen rollout for family[0] is not recomputed in epoch 0.
    if cache_features:
        cache_payload = "|".join(
            [
                "neon_feature_cache_v3",
                str(stage1_checkpoint),
                str(config.feature_source),
                str(latent_dim),
                str(getattr(config, "k_train", "")),
                str(getattr(config, "latent_bank_count", "")),
                str(n_history),
                str(any(family.structural_dry_mask is not None for family in train_families + val_families)),
                str(getattr(config, "deterministic_head_canonical_k", 0) if canonical_enabled else 0),
                str(getattr(config, "deterministic_head_latent_seed", 0) if canonical_enabled else 0),
                str(canonical_zero_latent),
            ]
        )
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:16]
        collector = make_cached_feature_collector(
            collector,
            cache_device=cache_device,
            cache_dir=cache_dir,
            cache_key=cache_key,
            prefetch_workers=int(getattr(config, "feature_prefetch_workers", 2)),
            prefetch_depth=int(getattr(config, "feature_prefetch_depth", 2)),
        )

    # Probe the frozen feature width from one family to size the EpiNet.
    probe = collector(train_families[0], num_aleatory=max(2, int(config.k_train)), generator=generator)
    feature_channels = int(probe.features.shape[-1])
    if cache_features and cache_dir is None and memory_cache_limit_bytes is not None:
        estimated_cache_bytes = _estimate_cached_feature_bytes(
            probe,
            n_families=len(train_families) + len(val_families),
            latent_bank_count=max(1, int(getattr(config, "latent_bank_count", 1))),
            cache_dtype=torch.float16,
        )
        if estimated_cache_bytes > int(memory_cache_limit_bytes):
            est_gib = estimated_cache_bytes / float(1024**3)
            limit_gib = int(memory_cache_limit_bytes) / float(1024**3)
            raise ValueError(
                "In-memory NEON frozen-feature cache is estimated at "
                f"{est_gib:.1f} GiB, above the configured safety limit "
                f"({limit_gib:.1f} GiB). Pass cache_dir=... so rollout "
                "features are cached on disk instead of in RAM, or explicitly "
                "raise memory_cache_limit_bytes if this is intentional."
            )

    # Seed the EpiNet construction so the randomized prior branch (and the
    # trainable init) is reproducible from the recorded prior_seed.
    if prior_seed is not None:
        torch.manual_seed(int(prior_seed))
    module = build_epinet_from_config(
        config,
        feature_channels=feature_channels,
        out_channels=int(out_channels),
        hidden_channels=int(hidden_channels),
    )
    # Place the trainable EpiNet on the same device as the (frozen) features so
    # forward/backward stay device-consistent on GPU.
    module = module.to(probe.features.device)

    resume_payload = None
    resume_path = Path(resume_state_path) if resume_state_path is not None else None
    if resume_path is not None and resume_path.exists():
        resume_payload = load_neon_stage2_training_state(
            resume_path,
            map_location=probe.features.device,
        )
        module.load_state_dict(resume_payload["state_dict"])
        logging.getLogger(__name__).info(
            "resuming NEON Stage-2 training from %s at epoch %d",
            resume_path,
            int(resume_payload["next_epoch"]),
        )

    if resume_payload is None and calibrate_prior and config.uses_calibrated_prior_scale:
        calibration_m = max(64, int(getattr(config, "calibration_m", 64)), int(config.m_train))
        z_e = (
            dirichlet_particle_control.eval_epistemic_indices(
                device=probe.features.device, dtype=probe.features.dtype
            )
            if dirichlet_particle_control is not None
            else sample_epistemic_indices(
                calibration_m, int(config.d_e),
                device=probe.features.device, dtype=probe.features.dtype, generator=generator,
            )
        )
        n_calib = min(
            max(1, int(getattr(config, "calibration_families", 4))),
            len(train_families),
        )
        rmse_values: list[float] = []
        alpha_values: list[float] = []
        for cal_idx, family in enumerate(train_families[:n_calib]):
            cal_probe = probe if cal_idx == 0 else collector(
                family,
                num_aleatory=max(2, int(config.k_train)),
                generator=generator,
                latent_bank_id=0,
            )
            rmse = base_rmse_from_reference(
                cal_probe.base_prediction,
                family.reference.unsqueeze(0).to(
                    device=cal_probe.features.device, dtype=cal_probe.features.dtype
                ),
            )
            rmse_values.append(float(rmse))
            alpha_values.append(
                calibrate_prior_scale(
                    module=module,
                    features=cal_probe.features,
                    z_e=z_e.to(device=cal_probe.features.device, dtype=cal_probe.features.dtype),
                    node_coords=family.geometry,
                    base_rmse=rmse,
                    target_fraction=(
                        0.10
                        if config.uses_de_spread_prior_scale
                        else float(config.prior_scale_fraction)
                    ),
                    target_std=(
                        float(config.de_spread_target_std_m)
                        / float(validation_physical_scale)
                        if config.uses_de_spread_prior_scale
                        else None
                    ),
                )
            )
        alpha = float(sum(alpha_values) / max(len(alpha_values), 1))
        module.set_prior_scale(alpha)

    optimizer = build_neon_stage2_optimizer(
        module, learning_rate=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=probe.features.device)
    loss_weights = NEONStage2LossWeights(**config.to_loss_weights_dict())
    metadata = build_neon_stage2_metadata(
        stage1_checkpoint_path=str(stage1_checkpoint),
        stage1_checkpoint_alias=stage1_alias,
        normalizer_fingerprint=normalizer_fingerprint or {},
        structural_dry_policy=structural_dry_policy,
        feature_source=config.feature_source,
        dependency=config.dependency,
        d_a=int(latent_dim),
        d_e=int(config.d_e),
        k_train=int(config.k_train),
        m_train=int(config.m_train),
        k_eval=int(config.k_eval),
        m_eval=int(config.m_eval),
        alpha=module.alpha,
        prior_seed=prior_seed,
        loss_weights=config.to_loss_weights_dict(),
        optimizer_settings={
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
        },
        extra={
            "branch_type": str(getattr(config, "branch_type", "projected")),
            "train_hidden_channels": int(getattr(config, "train_hidden_channels", hidden_channels)),
            "prior_hidden_channels": int(getattr(config, "prior_hidden_channels", hidden_channels)),
            "branch_layers": int(getattr(config, "branch_layers", 2)),
            "branch_activation": str(getattr(config, "branch_activation", "gelu")),
            "concat_index": bool(getattr(config, "concat_index", True)),
            "epistemic_basis": str(getattr(config, "epistemic_basis", "identity")),
            "epistemic_linear_terms": bool(
                getattr(config, "epistemic_linear_terms", True)
            ),
            "epistemic_quadratic_terms": int(
                getattr(config, "epistemic_quadratic_terms", 16)
            ),
            "epistemic_basis_seed": int(getattr(config, "epistemic_basis_seed", 123)),
            "epistemic_index_mode": str(
                getattr(config, "epistemic_index_mode", "continuous")
            ),
            "dirichlet_particle_control": (
                None
                if dirichlet_particle_control is None
                else dirichlet_particle_control.to_metadata()
            ),
            "deterministic_head": bool(getattr(config, "deterministic_head", False)),
            "deterministic_head_feature": str(
                getattr(config, "deterministic_head_feature", "canonical_aleatory_mean")
            ),
            "bootstrap": config.to_bootstrap_config_dict()
            if hasattr(config, "to_bootstrap_config_dict")
            else {},
            "member_bootstrap": config.to_member_bootstrap_config_dict()
            if hasattr(config, "to_member_bootstrap_config_dict")
            else {},
            "cancellation_diagnostics": config.to_cancellation_diagnostics_config_dict()
            if hasattr(config, "to_cancellation_diagnostics_config_dict")
            else {},
            "prior_rff_dim": int(getattr(config, "prior_rff_dim", 0)),
            "prior_rff_lengthscale": float(getattr(config, "prior_rff_lengthscale", 0.25)),
            "prior_rff_include_lead": bool(getattr(config, "prior_rff_include_lead", True)),
            "selection_min_retention": float(getattr(config, "selection_min_retention", 0.0)),
            "selection_rmse_margin_m": float(
                getattr(config, "selection_rmse_margin_m", 0.001)
            ),
            "selection_metric": str(getattr(config, "selection_metric", "mixture_crps")),
            "selection_enforce_rmse": bool(
                getattr(config, "selection_enforce_rmse", True)
            ),
            "validation_metrics_inverse_transformed": bool(
                target_normalizer is not None and reference_normalizer is not None
            ),
            "de_spread_normalizer_scale_m_per_normalized_unit": (
                float(validation_physical_scale)
                if config.uses_de_spread_prior_scale
                else None
            ),
            "calibration_families": int(getattr(config, "calibration_families", 1)),
            "calibration_m": int(getattr(config, "calibration_m", int(config.m_train))),
            "auto_prior_calibration_rmse_mean": (
                float(sum(rmse_values) / len(rmse_values))
                if "rmse_values" in locals() and rmse_values
                else None
            ),
            "prior_scale_mode": (
                "de_spread_target"
                if config.uses_de_spread_prior_scale
                else "auto_base_rmse"
                if config.uses_auto_prior_scale
                else "explicit"
            ),
            "prior_scale_target_std_m": config.de_spread_target_std_m,
            "prior_scale_target_std_normalized": (
                float(config.de_spread_target_std_m) / float(validation_physical_scale)
                if config.uses_de_spread_prior_scale
                else None
            ),
            "family_batch_size": int(getattr(config, "family_batch_size", 1)),
            "effective_batch_size": int(getattr(config, "effective_batch_size", 1)),
            "shuffle_families": bool(getattr(config, "shuffle_families", True)),
            "epistemic_resample": str(getattr(config, "epistemic_resample", "epoch")),
            "latent_bank_count": int(getattr(config, "latent_bank_count", 1)),
            "latent_bank_sampling": "isolated_epoch_rng_v1",
            "feature_prefetch_workers": int(
                getattr(config, "feature_prefetch_workers", 0)
            ),
            "feature_prefetch_depth": int(
                getattr(config, "feature_prefetch_depth", 0)
            ),
            "reference_member_subsample": getattr(config, "reference_member_subsample", None),
            "progress_log_interval_effective_batches": int(
                getattr(config, "progress_log_interval_effective_batches", 10)
            ),
            "validation_interval": int(getattr(config, "validation_interval", 1)),
            "feature_cache_schema_version": "neon_feature_cache_v3",
            "structural_dry_feedback_clamp": bool(
                target_normalizer is not None
                and any(
                    family.structural_dry_mask is not None
                    for family in train_families + val_families
                )
            ),
            "deterministic_head_canonical_k": int(
                getattr(config, "deterministic_head_canonical_k", 0)
            ),
            "deterministic_head_latent_seed": int(
                getattr(config, "deterministic_head_latent_seed", 0)
            ),
            "canonical_latent_hash_probe": probe.canonical_latent_hash,
        },
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "neon_stage2_best.pt"
    progress_reporter = NEONTrainingProgressReporter(
        output_dir=output_dir,
        log_interval_effective_batches=int(
            getattr(config, "progress_log_interval_effective_batches", 10)
        ),
        logger=logging.getLogger(__name__),
    )

    try:
        return train_neon_stage2_epochs(
            module=module,
            optimizer=optimizer,
            train_families=train_families,
            val_families=val_families,
            feature_collector=collector,
            n_epochs=int(config.n_epochs),
            m_train=int(config.m_train),
            k_train=int(config.k_train),
            d_e=int(config.d_e),
            loss_weights=loss_weights,
            generator=generator,
            stage1_model=stage1,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=metadata,
            epistemic_chunk_size=epistemic_chunk_size,
            val_seed=val_seed,
            validation_interval=int(getattr(config, "validation_interval", 1)),
            bootstrap_config=config.to_bootstrap_config_dict()
            if hasattr(config, "to_bootstrap_config_dict")
            else None,
            member_bootstrap_config=config.to_member_bootstrap_config_dict()
            if hasattr(config, "to_member_bootstrap_config_dict")
            else None,
            cancellation_config=config.to_cancellation_diagnostics_config_dict()
            if hasattr(config, "to_cancellation_diagnostics_config_dict")
            else None,
            family_batch_size=int(getattr(config, "family_batch_size", 1)),
            effective_batch_size=int(getattr(config, "effective_batch_size", 1)),
            shuffle_families=bool(getattr(config, "shuffle_families", True)),
            epistemic_resample=str(getattr(config, "epistemic_resample", "epoch")),
            latent_bank_count=int(getattr(config, "latent_bank_count", 1)),
            reference_member_subsample=getattr(
                config, "reference_member_subsample", None
            ),
            selection_min_retention=float(
                getattr(config, "selection_min_retention", 0.0)
            ),
            selection_rmse_margin_m=float(
                getattr(config, "selection_rmse_margin_m", 0.001)
            ),
            selection_metric=str(getattr(config, "selection_metric", "mixture_crps")),
            selection_enforce_rmse=bool(
                getattr(config, "selection_enforce_rmse", True)
            ),
            validation_physical_scale=float(validation_physical_scale),
            validation_target_normalizer=target_normalizer,
            validation_reference_normalizer=reference_normalizer,
            dirichlet_particle_control=dirichlet_particle_control,
            progress_reporter=progress_reporter,
            latest_checkpoint_path=latest_checkpoint_path,
            start_epoch=0
            if resume_payload is None
            else int(resume_payload["next_epoch"]),
            initial_history=None
            if resume_payload is None
            else resume_payload.get("history", []),
            initial_best_epoch=-1
            if resume_payload is None
            else int(resume_payload["best_epoch"]),
            initial_best_val_fit=math.inf
            if resume_payload is None
            else float(resume_payload["best_val_fit"]),
        )
    finally:
        close_collector = getattr(collector, "close", None)
        if callable(close_collector):
            close_collector()
