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
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from neuralop.flood.neon import (
    NEONStage2LossWeights,
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


def _generator_initial_seed(generator: Optional[torch.Generator]) -> int:
    if generator is None:
        return 0
    return int(generator.initial_seed())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


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
    return (
        base_bytes + feature_bytes + latent_bytes
    ) * int(n_families) * int(latent_bank_count)


def make_feature_collector_from_frozen_model(
    stage1_model: torch.nn.Module,
    *,
    feature_source: str,
    n_history: int,
    latent_dim: int,
    generator: Optional[torch.Generator] = None,
):
    """Return a ``feature_collector`` that AR-rolls the frozen FGNO per family.

    The returned callable matches the training loop's collector contract:
    ``(family, *, num_aleatory, generator, latent_bank_id) -> FrozenFGNOFeatureBatch``.
    It samples
    ``num_aleatory`` persistent aleatory latents, warm-starts from the family's
    ``initial_histories`` (broadcast to K if a single shared history is given),
    and rolls out to the family's reference horizon ``T``.
    """

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
        return collect_frozen_fgno_rollout_features(
            stage1_model=stage1_model,
            static=family.static.to(model_device),
            geometry=family.geometry.to(model_device),
            query_points=family.query_points.to(model_device),
            boundary_sequence=family.boundary_sequence.to(model_device),
            initial_histories=init,
            aleatory_latents=latents,
            rollout_length=int(family.reference.shape[1]),  # T from the reference ensemble
            n_history=int(n_history),
            feature_source=feature_source,
        )

    return collector


def make_cached_feature_collector(
    base_collector,
    *,
    cache_device: str | torch.device = "cpu",
    cache_dtype: torch.dtype = torch.float16,
    cache_dir: Optional[Any] = None,
    cache_key: Optional[str] = None,
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
    cache: dict[str, tuple] = {}
    dir_path = None
    if cache_dir is not None:
        dir_path = Path(cache_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
    key_prefix = "" if cache_key is None else f"{str(cache_key)}_"

    def _store(batch) -> None:
        payload = {
            "base": batch.base_prediction.detach().to(device=cache_device, dtype=cache_dtype),
            "features": batch.features.detach().to(device=cache_device, dtype=cache_dtype),
            "latents": batch.aleatory_latents.detach().to(device=cache_device),
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
            safe_key = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
            fpath = dir_path / f"{key_prefix}{safe_key}_bank{bank}_k{int(num_aleatory)}.pt"
            if fpath.exists():
                p = torch.load(fpath, map_location="cpu")
                dev, dt = torch.device(p["device"]), p["dtype"]
                return FrozenFGNOFeatureBatch(
                    base_prediction=p["base"].to(device=dev, dtype=dt),
                    features=p["features"].to(device=dev, dtype=dt),
                    aleatory_latents=p["latents"].to(device=dev),
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
                batch.features.device,
                batch.features.dtype,
            )
            return batch
        base_c, feat_c, lat_c, dev, dt = hit
        return FrozenFGNOFeatureBatch(
            base_prediction=base_c.to(device=dev, dtype=dt),
            features=feat_c.to(device=dev, dtype=dt),
            aleatory_latents=lat_c.to(device=dev),
        )

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

    collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=config.feature_source,
        n_history=n_history,
        latent_dim=latent_dim,
        generator=generator,
    )
    # Wrap for caching before the probe so the probe populates the cache and the
    # frozen rollout for family[0] is not recomputed in epoch 0.
    if cache_features:
        cache_payload = "|".join(
            [
                "neon_feature_cache_v2",
                str(stage1_checkpoint),
                str(config.feature_source),
                str(latent_dim),
                str(getattr(config, "k_train", "")),
                str(getattr(config, "latent_bank_count", "")),
                str(n_history),
            ]
        )
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:16]
        collector = make_cached_feature_collector(
            collector, cache_device=cache_device, cache_dir=cache_dir, cache_key=cache_key
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

    if resume_payload is None and calibrate_prior and config.uses_auto_prior_scale:
        z_e = sample_epistemic_indices(
            int(config.m_train), int(config.d_e),
            device=probe.features.device, dtype=probe.features.dtype, generator=generator,
        )
        rmse = base_rmse_from_reference(
            probe.base_prediction,
            train_families[0].reference.unsqueeze(0).to(
                device=probe.features.device, dtype=probe.features.dtype
            ),
        )
        alpha = calibrate_prior_scale(
            module=module, features=probe.features, z_e=z_e,
            base_rmse=rmse, target_fraction=float(config.prior_scale_fraction),
        )
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
            "bootstrap": config.to_bootstrap_config_dict()
            if hasattr(config, "to_bootstrap_config_dict")
            else {},
            "cancellation_diagnostics": config.to_cancellation_diagnostics_config_dict()
            if hasattr(config, "to_cancellation_diagnostics_config_dict")
            else {},
            "family_batch_size": int(getattr(config, "family_batch_size", 1)),
            "effective_batch_size": int(getattr(config, "effective_batch_size", 1)),
            "shuffle_families": bool(getattr(config, "shuffle_families", True)),
            "epistemic_resample": str(getattr(config, "epistemic_resample", "epoch")),
            "latent_bank_count": int(getattr(config, "latent_bank_count", 1)),
            "reference_member_subsample": getattr(config, "reference_member_subsample", None),
            "progress_log_interval_effective_batches": int(
                getattr(config, "progress_log_interval_effective_batches", 10)
            ),
            "feature_cache_schema_version": "neon_feature_cache_v2",
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
        bootstrap_config=config.to_bootstrap_config_dict()
        if hasattr(config, "to_bootstrap_config_dict")
        else None,
        cancellation_config=config.to_cancellation_diagnostics_config_dict()
        if hasattr(config, "to_cancellation_diagnostics_config_dict")
        else None,
        family_batch_size=int(getattr(config, "family_batch_size", 1)),
        effective_batch_size=int(getattr(config, "effective_batch_size", 1)),
        shuffle_families=bool(getattr(config, "shuffle_families", True)),
        epistemic_resample=str(getattr(config, "epistemic_resample", "epoch")),
        latent_bank_count=int(getattr(config, "latent_bank_count", 1)),
        reference_member_subsample=getattr(config, "reference_member_subsample", None),
        progress_reporter=progress_reporter,
        latest_checkpoint_path=latest_checkpoint_path,
        start_epoch=0 if resume_payload is None else int(resume_payload["next_epoch"]),
        initial_history=None if resume_payload is None else resume_payload.get("history", []),
        initial_best_epoch=-1 if resume_payload is None else int(resume_payload["best_epoch"]),
        initial_best_val_fit=math.inf
        if resume_payload is None
        else float(resume_payload["best_val_fit"]),
    )
