"""Dataset preparation helpers for flood diffusion training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers
from neuralop.flood.data.wv import FloodDatasetHDF, NormalizedDatasetOnTheFly, fit_normalizers_streaming
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
    get_dataset_boundary_kwargs,
    make_dataloader_generator,
    make_split_generator,
)


def _wait_for_normalizer_file(
    normalizer_path: Path,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 5.0,
) -> None:
    """Wait for a rank-0-produced normalizer file to appear and become readable."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if normalizer_path.exists():
            try:
                load_normalizers(normalizer_path, device=None)
                return
            except Exception as exc:  # file may exist but still be mid-write
                last_error = exc
        time.sleep(poll_interval_seconds)
    if last_error is not None:
        raise RuntimeError(
            f"Timed out waiting for readable normalizers at {normalizer_path}."
        ) from last_error
    raise TimeoutError(f"Timed out waiting for normalizer file to appear at {normalizer_path}.")

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
        "Training dataset boundary_source=%s%s",
        dataset_kwargs["boundary_source"],
        f", clean_boundary_file={dataset_kwargs['clean_boundary_file']}"
        if dataset_kwargs["boundary_source"] == "clean_family"
        else "",
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
    if normalizer_path is not None and normalizer_path.exists():
        normalizers = load_normalizers(normalizer_path, device=None)
        _rank0_info(logger, dist_ctx, "Loaded normalizers from %s", normalizer_path)
    else:
        normalizers = None
        if dist_ctx.use_distributed and normalizer_path is None:
            raise RuntimeError(
                "Distributed diffusion training requires data.normalizer_path so non-rank0 "
                "processes can wait for rank0-produced normalizers without long-lived NCCL collectives."
            )
        if dist_ctx.is_rank0:
            normalizers = fit_normalizers_streaming(
                train_raw,
                chunk_size=int(safe_get(data_cfg, "normalizer_chunk_size", 10000)),
                expect_target=True,
            )
            if normalizer_path is not None:
                save_normalizers(normalizers, normalizer_path)
                logger.info("Saved normalizers to %s", normalizer_path)
        if dist_ctx.use_distributed:
            if not dist_ctx.is_rank0:
                wait_timeout_min = float(
                    safe_get(data_cfg, "normalizer_wait_timeout_min", 120)
                )
                _wait_for_normalizer_file(
                    normalizer_path,
                    timeout_seconds=wait_timeout_min * 60.0,
                )
                normalizers = load_normalizers(normalizer_path, device=None)
            _dist_barrier(dist_ctx)
        if normalizers is None:
            raise RuntimeError("Failed to initialize normalizers in distributed setup.")

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

    dyn = sample["dynamic"]
    if dyn.ndim == 3:
        dyn = dyn.unsqueeze(0)
    dyn = dyn.permute(0, 2, 1, 3)
    bsz, n_cells, n_hist, n_dyn_ch = dyn.shape
    dyn_flat = dyn.reshape(bsz, n_cells, n_hist * n_dyn_ch)

    bc = sample["boundary"]
    if bc.ndim == 3:
        bc = bc.unsqueeze(0)
    bc = bc.permute(0, 2, 1, 3)
    bsz2, n_cells2, n_hist2, n_bc_ch = bc.shape
    bc_flat = bc.reshape(bsz2, n_cells2, n_hist2 * n_bc_ch)

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

    return {
        "context": context,
        "target": y,
        "input_geom": geom_shared,
        "latent_queries": q_shared,
        "output_queries": geom_shared.clone(),
    }
