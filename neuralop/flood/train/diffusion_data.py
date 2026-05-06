"""Dataset preparation helpers for flood diffusion training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers
from neuralop.flood.data.structural_dry import (
    build_structural_dry_artifact,
    dry_mask_to_wettable_mask,
    load_structural_dry_artifact,
    save_structural_dry_artifact,
    validate_structural_dry_artifact,
)
from neuralop.flood.data.wv import FloodDatasetHDF, NormalizedDatasetOnTheFly
from neuralop.flood.data.wv import (
    build_normalizer_metadata,
    fit_normalizers,
    load_normalizer_metadata,
    normalizer_metadata_matches,
    resolve_normalizer_fit_method,
    resolve_normalizer_metadata_path,
    save_normalizer_metadata,
)
from neuralop.flood.train.diffusion_runtime import (
    DEFAULT_BOUNDARY_CHANNELS,
    TRAIN_FRAC,
    DistContext,
    _dist_barrier,
    _rank0_info,
    _resolve_normalizer_path,
)
from neuralop.flood.utils.diffusion_script_utils import safe_get
from neuralop.flood.utils.runtime import (
    dataloader_worker_init,
    describe_boundary_spec,
    get_dataset_boundary_kwargs,
    get_dataset_hdf_paths,
    get_structural_dry_policy_kwargs,
    make_dataloader_generator,
    make_split_generator,
    wait_for_structural_dry_artifact,
)


def _wait_for_normalizer_artifacts(
    normalizer_path: Path,
    *,
    metadata_path: Path,
    expected_metadata: Dict[str, Any],
    timeout_seconds: float,
    poll_interval_seconds: float = 5.0,
) -> None:
    """Wait for rank-0-produced normalizer and metadata files to appear and match."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if normalizer_path.exists() and metadata_path.exists():
            try:
                metadata = load_normalizer_metadata(metadata_path)
                if metadata is None or not normalizer_metadata_matches(expected_metadata, metadata):
                    raise RuntimeError(
                        f"Normalizer metadata at {metadata_path} does not match the current diffusion run."
                    )
                load_normalizers(normalizer_path, device=None)
                return
            except Exception as exc:  # file may exist but still be mid-write
                last_error = exc
        time.sleep(poll_interval_seconds)
    if last_error is not None:
        raise RuntimeError(
            f"Timed out waiting for readable normalizer artifacts at {normalizer_path}."
        ) from last_error
    raise TimeoutError(
        f"Timed out waiting for normalizer artifacts to appear at {normalizer_path} and {metadata_path}."
    )

def _prepare_datasets(
    config: Any,
    target_variables: list[str],
    logger,
    dist_ctx: DistContext,
) -> Tuple[
    DataLoader,
    DataLoader,
    Optional[DistributedSampler],
    Optional[DistributedSampler],
    Dict[str, Any],
    Path,
    int,
    int,
]:
    data_cfg = safe_get(config, "data", {})
    seed = int(safe_get(safe_get(config, "distributed", {}), "seed", 123))
    static_text_files = list(safe_get(data_cfg, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]))

    dataset_kwargs = dict(
        data_root=str(safe_get(data_cfg, "root", "")),
        n_history=int(safe_get(data_cfg, "n_history", 3)),
        train_txt=str(safe_get(data_cfg, "train_txt", "train.txt")),
        static_text_files=static_text_files,
        hdf_paths=get_dataset_hdf_paths(data_cfg),
        hdf_suffix=".hdf",
        noise_type=str(safe_get(data_cfg, "noise_type", "none")),
        noise_std=safe_get(data_cfg, "noise_std", None),
        skip_before_timestep=int(safe_get(data_cfg, "skip_before_timestep", 0)),
        ar_rollout_steps=max(1, int(safe_get(safe_get(config, "opt", {}), "ar_rollout_steps", 1))),
        target_variables=target_variables,
    )
    dataset_kwargs.update(get_dataset_boundary_kwargs(data_cfg))
    _rank0_info(
        logger,
        dist_ctx,
        "Training dataset boundary=%s",
        describe_boundary_spec(dataset_kwargs["boundary_spec"]),
    )
    write_train_txt = bool(safe_get(data_cfg, "write_train_txt", False))
    dataset_kwargs["write_train_txt"] = write_train_txt and dist_ctx.is_rank0

    # Avoid train.txt write/read races in distributed mode by letting rank 0
    # perform any write-side initialization before other ranks instantiate.
    if dist_ctx.use_distributed and write_train_txt:
        if dist_ctx.is_rank0:
            full_dataset = FloodDatasetHDF(**dataset_kwargs)
        _dist_barrier(dist_ctx)
        if not dist_ctx.is_rank0:
            dataset_kwargs["write_train_txt"] = False
            full_dataset = FloodDatasetHDF(**dataset_kwargs)
    else:
        full_dataset = FloodDatasetHDF(**dataset_kwargs)
    _dist_barrier(dist_ctx)

    structural_dry_policy = get_structural_dry_policy_kwargs(
        config,
        normalizer_path=_resolve_normalizer_path(config),
        allow_data_root_fallback=True,
    )
    if structural_dry_policy["policy"] == "masked_primary":
        artifact_path = structural_dry_policy["artifact_path"]
        artifact = None
        if artifact_path.exists():
            artifact = load_structural_dry_artifact(artifact_path)
        elif dist_ctx.use_distributed and not dist_ctx.is_rank0:
            artifact = wait_for_structural_dry_artifact(artifact_path)
        elif dist_ctx.is_rank0:
            artifact = build_structural_dry_artifact(
                data_root=dataset_kwargs["data_root"],
                run_ids=full_dataset.run_ids,
                train_txt=dataset_kwargs["train_txt"],
                hdf_suffix=dataset_kwargs["hdf_suffix"],
                hdf_paths=full_dataset.hdf_paths,
                cell_point_index=full_dataset.cell_point_index,
                mask_definition=structural_dry_policy["mask_definition"],
            )
            save_structural_dry_artifact(
                artifact,
                artifact_path=artifact_path,
                summary_path=structural_dry_policy["summary_path"],
            )
        _dist_barrier(dist_ctx)
        if artifact is None:
            artifact = load_structural_dry_artifact(artifact_path)
        artifact = validate_structural_dry_artifact(
            artifact,
            expected_cell_count=full_dataset.reference_cell_count,
            expected_run_ids=full_dataset.run_ids,
        )
        full_dataset.set_structural_dry_mask(artifact["dry_mask"])
        _rank0_info(
            logger,
            dist_ctx,
            "Structural-dry policy=%s mask_definition=%s n_dry=%d n_wettable=%d artifact=%s",
            structural_dry_policy["policy"],
            structural_dry_policy["mask_definition"],
            artifact["n_dry"],
            artifact["n_wettable"],
            artifact_path,
        )

    n_samples_max = safe_get(data_cfg, "n_samples_max", None)
    if n_samples_max is not None and int(n_samples_max) > 0:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))

    total_len = len(full_dataset)
    train_sz = max(1, int(TRAIN_FRAC * total_len))
    test_sz = total_len - train_sz
    train_raw, test_raw = random_split(
        full_dataset,
        [train_sz, test_sz],
        generator=make_split_generator(seed),
    )
    _rank0_info(logger, dist_ctx, "Split dataset: total=%d train=%d test=%d", total_len, train_sz, test_sz)

    normalizer_path = _resolve_normalizer_path(config)
    normalizer_fit_method = resolve_normalizer_fit_method(
        train_raw,
        method=str(safe_get(data_cfg, "normalizer_fit_method", "auto")),
        structural_dry_policy=structural_dry_policy["policy"],
    )
    expected_normalizer_metadata = build_normalizer_metadata(
        train_raw,
        structural_dry_policy=structural_dry_policy["policy"],
        fit_method=normalizer_fit_method,
    )
    metadata_path = (
        resolve_normalizer_metadata_path(normalizer_path)
        if normalizer_path is not None
        else None
    )
    can_load_cached_normalizers = (
        normalizer_path is not None
        and normalizer_path.exists()
        and metadata_path is not None
        and metadata_path.exists()
        and normalizer_metadata_matches(
            expected_normalizer_metadata,
            load_normalizer_metadata(metadata_path),
        )
    )
    if can_load_cached_normalizers:
        normalizers = load_normalizers(normalizer_path, device=None)
        _rank0_info(
            logger,
            dist_ctx,
            "Loaded normalizers from %s (method=%s)",
            normalizer_path,
            normalizer_fit_method,
        )
    else:
        normalizers = None
        if dist_ctx.use_distributed and normalizer_path is None:
            raise RuntimeError(
                "Distributed diffusion training requires data.normalizer_path so non-rank0 "
                "processes can wait for rank0-produced normalizers without long-lived NCCL collectives."
            )
        if dist_ctx.is_rank0:
            normalizers, normalizer_fit_method = fit_normalizers(
                train_raw,
                chunk_size=int(safe_get(data_cfg, "normalizer_chunk_size", 10000)),
                expect_target=True,
                structural_dry_policy=structural_dry_policy["policy"],
                method=normalizer_fit_method,
                return_method=True,
            )
            if normalizer_path is not None:
                save_normalizers(normalizers, normalizer_path)
                if metadata_path is not None:
                    save_normalizer_metadata(
                        metadata_path,
                        build_normalizer_metadata(
                            train_raw,
                            structural_dry_policy=structural_dry_policy["policy"],
                            fit_method=normalizer_fit_method,
                        ),
                    )
                logger.info(
                    "Saved normalizers to %s (method=%s)",
                    normalizer_path,
                    normalizer_fit_method,
                )
        if dist_ctx.use_distributed:
            if not dist_ctx.is_rank0:
                wait_timeout_min = float(
                    safe_get(data_cfg, "normalizer_wait_timeout_min", 120)
                )
                _wait_for_normalizer_artifacts(
                    normalizer_path,
                    metadata_path=metadata_path,
                    expected_metadata=expected_normalizer_metadata,
                    timeout_seconds=wait_timeout_min * 60.0,
                )
                normalizers = load_normalizers(normalizer_path, device=None)
                metadata = load_normalizer_metadata(metadata_path) if metadata_path is not None else None
                if metadata is not None:
                    normalizer_fit_method = str(metadata.get("fit_method", normalizer_fit_method))
            _dist_barrier(dist_ctx)
        if normalizers is None:
            raise RuntimeError("Failed to initialize normalizers in distributed setup.")
    _rank0_info(logger, dist_ctx, "normalizer_fit_method=%s", normalizer_fit_method)

    query_res = list(safe_get(data_cfg, "query_res", [48, 48]))
    train_norm = NormalizedDatasetOnTheFly(train_raw, normalizers, query_res=query_res)
    test_norm = NormalizedDatasetOnTheFly(test_raw, normalizers, query_res=query_res)

    batch_size = int(safe_get(data_cfg, "batch_size", 8))
    num_workers = int(safe_get(data_cfg, "num_workers", 0))
    pin_memory = bool(safe_get(data_cfg, "pin_memory", True))
    persistent_workers = bool(safe_get(data_cfg, "persistent_workers", False)) and num_workers > 0
    prefetch_factor = safe_get(data_cfg, "prefetch_factor", 2)

    train_sampler: Optional[DistributedSampler] = None
    test_sampler: Optional[DistributedSampler] = None
    if dist_ctx.use_distributed:
        train_sampler = DistributedSampler(
            train_norm,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            seed=seed,
        )
        test_sampler = DistributedSampler(
            test_norm,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            seed=seed,
        )

    loader_seed = seed
    train_loader_kwargs = dict(
        dataset=train_norm,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=make_dataloader_generator(loader_seed),
    )
    if num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=int(prefetch_factor),
            worker_init_fn=lambda wid: dataloader_worker_init(wid, loader_seed),
        )
    train_loader = DataLoader(**train_loader_kwargs)
    test_loader = DataLoader(
        test_norm,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    n_static = 2 + len(static_text_files)
    n_boundary_channels = DEFAULT_BOUNDARY_CHANNELS
    if len(train_norm) > 0:
        boundary_sample = train_norm[0].get("boundary", None)
        if isinstance(boundary_sample, torch.Tensor) and boundary_sample.ndim >= 1:
            n_boundary_channels = int(boundary_sample.shape[-1])
        else:
            logger.warning(
                "Could not infer boundary channels from dataset sample; "
                "falling back to n_boundary_channels=%d.",
                DEFAULT_BOUNDARY_CHANNELS,
            )
    normalizer_path = normalizer_path or Path("<not_saved>")
    return (
        train_loader,
        test_loader,
        train_sampler,
        test_sampler,
        normalizers,
        normalizer_path,
        n_static,
        n_boundary_channels,
    )


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Normalize batch shape and build diffusion inputs.

    Context channel layout:
    - static features
    - flattened boundary history (all boundary channels)
    - flattened dynamic history (target channels)
    """
    sample: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v.to(device)

    dyn_hist = sample["dynamic"]
    if dyn_hist.ndim == 3:
        dyn_hist = dyn_hist.unsqueeze(0)
    if dyn_hist.ndim != 4:
        raise ValueError(f"dynamic must have shape [B, H, N, C], got {tuple(dyn_hist.shape)}")
    dyn_context = dyn_hist.permute(0, 2, 1, 3)
    bsz, n_cells, n_hist, n_dyn_ch = dyn_context.shape
    dyn_flat = dyn_context.reshape(bsz, n_cells, n_hist * n_dyn_ch)

    bc_hist = sample["boundary"]
    if bc_hist.ndim == 3:
        bc_hist = bc_hist.unsqueeze(0)
    if bc_hist.ndim != 4:
        raise ValueError(f"boundary must have shape [B, H, N, C], got {tuple(bc_hist.shape)}")
    bc_context = bc_hist.permute(0, 2, 1, 3)
    bsz2, n_cells2, n_hist2, n_bc_ch = bc_context.shape
    if (bsz2, n_cells2, n_hist2) != (bsz, n_cells, n_hist):
        raise ValueError(
            "boundary history shape must match dynamic history over batch/cells/history: "
            f"dynamic={(bsz, n_cells, n_hist)} boundary={(bsz2, n_cells2, n_hist2)}"
        )
    bc_flat = bc_context.reshape(bsz2, n_cells2, n_hist2 * n_bc_ch)

    static = sample["static"]
    if static.ndim == 2:
        static = static.unsqueeze(0)

    context = torch.cat([static, bc_flat, dyn_flat], dim=2)

    geom = sample["geometry"]
    if geom.ndim == 2:
        geom = geom.unsqueeze(0)
    geom_shared = geom[0:1] if geom.shape[0] > 1 else geom

    q = sample["query_points"]
    if q.ndim == 3:
        q = q.unsqueeze(0)
    q_shared = q[0:1] if q.shape[0] > 1 else q

    y = sample["target"]
    if y.ndim == 2:
        y = y.unsqueeze(0)

    out = {
        "context": context,
        "target": y,
        "input_geom": geom_shared,
        "latent_queries": q_shared,
        "output_queries": geom_shared.clone(),
        "static": static,
        "dynamic": dyn_hist,
        "boundary": bc_hist,
    }
    if "target_sequence" in sample:
        target_sequence = sample["target_sequence"]
        if target_sequence.ndim == 3:
            target_sequence = target_sequence.unsqueeze(0)
        if target_sequence.ndim == 5 and target_sequence.shape[1] == 1:
            target_sequence = target_sequence.squeeze(1)
        if target_sequence.ndim != 4:
            raise ValueError(
                f"target_sequence must have shape [B, T, N, C], got {tuple(target_sequence.shape)}"
            )
        out["target_sequence"] = target_sequence
    if "boundary_sequence" in sample:
        boundary_sequence = sample["boundary_sequence"]
        if boundary_sequence.ndim == 3:
            boundary_sequence = boundary_sequence.unsqueeze(0)
        if boundary_sequence.ndim == 5 and boundary_sequence.shape[1] == 1:
            boundary_sequence = boundary_sequence.squeeze(1)
        if boundary_sequence.ndim != 4:
            raise ValueError(
                f"boundary_sequence must have shape [B, T, N, C], got {tuple(boundary_sequence.shape)}"
            )
        out["boundary_sequence"] = boundary_sequence
    if "structural_dry_mask" in sample:
        structural_dry_mask = sample["structural_dry_mask"]
        wettable = dry_mask_to_wettable_mask(structural_dry_mask).to(device=device)
        if wettable.ndim == 1:
            wettable = wettable.unsqueeze(0).expand(y.shape[0], -1)
        out["structural_dry_mask"] = structural_dry_mask
        out["point_weights"] = wettable.unsqueeze(-1).to(dtype=y.dtype)
    return out
