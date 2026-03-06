#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train DDO-style conditional diffusion forecaster on WV flood depth-only data."""

from __future__ import annotations

import copy
import datetime as dt
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neuralop import get_model  # noqa: E402
from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers  # noqa: E402
from neuralop.diffusion import (  # noqa: E402
    ConditioningConfig,
    ConditionalDDOForecaster,
    PointRFFGaussianProcessSampler,
)
from scripts.diffusion_script_utils import (  # noqa: E402
    load_checkpoint_bundle,
    save_checkpoint_sidecars,
    safe_wandb_finish,
    safe_get,
    shutdown_dataloader_workers,
    to_builtin,
)
from neuralop.utils import get_wandb_api_key  # noqa: E402
from train_gino_flood_train_rollout_animation_WV import (  # noqa: E402
    FloodDatasetHDF,
    NormalizedDatasetOnTheFly,
    dataloader_worker_init,
    fit_normalizers_streaming,
    make_dataloader_generator,
    make_split_generator,
    parse_target_variables,
    set_seed,
    setup_logging,
)

TRAIN_FRAC = 0.9
DEFAULT_PRINT_EVERY = 100
DEFAULT_MAX_VAL_BATCHES = 64
DEFAULT_BOUNDARY_CHANNELS = 1
TIME_FEATURE_DIM_SINCOS = 2
TIME_FEATURE_DIM_RAW = 1


@dataclass
class DistContext:
    use_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_rank0(self) -> bool:
        return int(self.rank) == 0


def _cfg_dist(config: Any) -> Any:
    return safe_get(config, "distributed", {})


def _should_use_distributed(config: Any) -> bool:
    return bool(safe_get(_cfg_dist(config), "use_distributed", False))


def _init_distributed(config: Any) -> DistContext:
    dist_cfg = _cfg_dist(config)
    requested = _should_use_distributed(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not requested and world_size > 1:
        raise ValueError(
            "WORLD_SIZE>1 detected but distributed.use_distributed=false. "
            "Set --distributed.use_distributed true for torchrun launches."
        )
    if not requested or world_size <= 1:
        return DistContext(use_distributed=False, rank=0, local_rank=0, world_size=1)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_min = int(safe_get(dist_cfg, "ddp_timeout_min", 30))
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=dt.timedelta(minutes=timeout_min),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistContext(
        use_distributed=(world_size > 1),
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=int(world_size),
    )


def _dist_barrier(dist_ctx: DistContext) -> None:
    if dist_ctx.use_distributed and dist.is_initialized():
        dist.barrier()


def _reduce_sum(value: float, *, device: torch.device, dist_ctx: DistContext) -> float:
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist_ctx.use_distributed and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to the target device after resume."""
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_state_dict_compat(module: torch.nn.Module, state_dict: Dict[str, Any], *, name: str) -> None:
    """Load state dict with fallback for legacy DDP `module.` prefixes."""
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            raise RuntimeError(
                f"Could not load {name}: mixed/non-DDP keys detected (first key={key!r})."
            )
    module.load_state_dict(stripped, strict=True)


def _rank0_info(logger, dist_ctx: DistContext, msg: str, *args) -> None:
    if dist_ctx.is_rank0:
        logger.info(msg, *args)


def _load_config(config_default: Path) -> Any:
    import sys as _sys

    config_name = "flood"
    config_path = config_default
    argv = list(_sys.argv[1:])
    for i, a in enumerate(argv):
        if a == "--config_path" and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            if not config_path.is_absolute():
                config_path = (_REPO_ROOT / config_path).resolve()
            idx = _sys.argv.index("--config_path")
            _sys.argv.pop(idx + 1)
            _sys.argv.pop(idx)
            break

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipe = ConfigPipeline(
        [
            YamlConfig(str(config_path), config_name=config_name, config_folder=str(_REPO_ROOT / "config")),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        ]
    )
    config = pipe.read_conf()
    return config


def _resolve_device(config: Any, dist_ctx: DistContext) -> torch.device:
    if dist_ctx.use_distributed and torch.cuda.is_available():
        return torch.device(f"cuda:{dist_ctx.local_rank}")
    configured = str(safe_get(safe_get(config, "distributed", {}), "device", "cuda:0"))
    if configured.startswith("cuda") and torch.cuda.is_available():
        return torch.device(configured)
    return torch.device("cpu")


def _resolve_normalizer_path(config: Any) -> Optional[Path]:
    normalizer_path = safe_get(safe_get(config, "data", {}), "normalizer_path", None)
    if normalizer_path is None:
        return None
    p = Path(str(normalizer_path))
    if not p.is_absolute():
        p = Path(str(safe_get(safe_get(config, "data", {}), "root", "."))) / p
    return p.resolve()


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
    # Backward compatibility: only pass write_train_txt if this FloodDatasetHDF
    # implementation supports it.
    supports_write_train_txt = "write_train_txt" in inspect.signature(FloodDatasetHDF.__init__).parameters
    write_train_txt = bool(safe_get(data_cfg, "write_train_txt", False))
    if supports_write_train_txt:
        dataset_kwargs["write_train_txt"] = write_train_txt and dist_ctx.is_rank0

    # Avoid train.txt write/read races in distributed mode by letting rank 0
    # perform any write-side initialization before other ranks instantiate.
    if dist_ctx.use_distributed and supports_write_train_txt and write_train_txt:
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
            obj = [normalizers]
            dist.broadcast_object_list(obj, src=0)
            normalizers = obj[0]
            if normalizer_path is not None:
                _dist_barrier(dist_ctx)
                if not dist_ctx.is_rank0 and normalizer_path.exists():
                    normalizers = load_normalizers(normalizer_path, device=None)
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


def _configure_denoiser(
    config: Any,
    n_static: int,
    n_boundary_channels: int,
    n_target_channels: int,
) -> Tuple[Dict[str, Any], int, int, str, int]:
    diff_cfg = safe_get(config, "diffusion", {})
    cond_cfg = safe_get(diff_cfg, "conditioning", {})
    n_history = int(safe_get(safe_get(config, "data", {}), "n_history", 3))
    time_injection = str(safe_get(cond_cfg, "time_injection", "channel")).lower()
    if time_injection not in {"channel", "adain"}:
        raise ValueError(
            f"diffusion.conditioning.time_injection must be one of {{'channel', 'adain'}}, got {time_injection!r}"
        )
    time_embedding_dim = int(safe_get(cond_cfg, "time_embedding_dim", 32))

    base_channels = n_static + n_history * n_boundary_channels + n_history * n_target_channels
    extra = 0
    if bool(safe_get(cond_cfg, "add_noisy_target", True)):
        extra += n_target_channels
    if time_injection == "channel" and bool(safe_get(cond_cfg, "add_time_features", True)):
        t_type = str(safe_get(cond_cfg, "time_feature_type", "sincos")).lower()
        extra += TIME_FEATURE_DIM_SINCOS if t_type == "sincos" else TIME_FEATURE_DIM_RAW

    total_in_channels = base_channels + extra

    gino_cfg = copy.deepcopy(to_builtin(safe_get(config, "gino", {})))
    gino_cfg["data_channels"] = int(total_in_channels)
    gino_cfg["out_channels"] = int(n_target_channels)
    gino_cfg["output_distribution"] = "deterministic"
    gino_cfg["use_fgn_noise"] = False
    # Respect config-defined checkpoint/model behavior; no forced AR override here.
    if time_injection == "adain":
        gino_cfg["fno_norm"] = "ada_in"
        gino_cfg["fno_ada_in_dim"] = int(time_embedding_dim)
    elif str(safe_get(gino_cfg, "fno_norm", "")).lower() == "ada_in":
        gino_cfg["fno_norm"] = "instance_norm"
    return gino_cfg, base_channels, total_in_channels, time_injection, time_embedding_dim


def _init_scheduler(config: Any, optimizer: torch.optim.Optimizer):
    opt = safe_get(config, "opt", {})
    name = str(safe_get(opt, "scheduler", "")).lower()
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(safe_get(opt, "scheduler_mode", "min")),
            patience=int(safe_get(opt, "scheduler_patience", 5)),
            threshold=float(safe_get(opt, "scheduler_threshold", 1e-4)),
            threshold_mode=str(safe_get(opt, "scheduler_threshold_mode", "rel")),
            cooldown=int(safe_get(opt, "scheduler_cooldown", 0)),
            min_lr=float(safe_get(opt, "scheduler_min_lr", 0.0)),
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(safe_get(opt, "scheduler_T_max", 200)),
            eta_min=float(safe_get(opt, "scheduler_eta_min", 0.0)),
        )
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(safe_get(opt, "step_size", 50)),
            gamma=float(safe_get(opt, "gamma", 0.5)),
        )
    return None


def _build_wandb_names(config: Any, seed: int) -> Tuple[str, str]:
    diff_cfg = safe_get(config, "diffusion", {})
    gp_cfg = safe_get(diff_cfg, "gp", {})
    sampler_cfg = safe_get(diff_cfg, "sampler", {})
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    gp_type = str(safe_get(gp_cfg, "type", "rff_rbf"))
    ls = float(safe_get(gp_cfg, "length_scale", 0.05))
    n_steps = int(safe_get(sampler_cfg, "num_steps", 40))
    group = f"ddofs_wv_m40_depth_job{job_id}"
    name = f"ddofs_wv_m40_depth_seed{seed}_steps{n_steps}_gp{gp_type}_ls{ls:.4f}"
    return group, name


def _maybe_init_wandb(config: Any, seed: int, logger, *, is_rank0: bool) -> Optional[Any]:
    if not is_rank0:
        return None
    wb_cfg = safe_get(config, "wandb", {})
    if not bool(safe_get(wb_cfg, "log", False)):
        return None

    key = get_wandb_api_key()
    if key:
        wandb.login(key=key, relogin=False)

    group_default, name_default = _build_wandb_names(config, seed)
    run = wandb.init(
        project=str(safe_get(wb_cfg, "project", "Flood_GINO_NoPhysics")),
        entity=safe_get(wb_cfg, "entity", None),
        group=str(safe_get(wb_cfg, "group", group_default) or group_default),
        name=str(safe_get(wb_cfg, "name", name_default) or name_default),
        config=to_builtin(config),
        dir=str(_SCRIPT_DIR / "wandb"),
        reinit=True,
    )
    logger.info("Initialized W&B run: %s", run.name)
    return run


def _evaluate_validation(
    forecaster: ConditionalDDOForecaster,
    loader: DataLoader,
    device: torch.device,
    target_norm: Optional[Any],
    dist_ctx: DistContext,
    max_batches: int = DEFAULT_MAX_VAL_BATCHES,
) -> Dict[str, float]:
    forecaster.eval()
    loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    loss_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_norm_sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_norm_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_phys_sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_phys_count = torch.tensor(0.0, device=device, dtype=torch.float64)

    if target_norm is not None:
        target_norm.to(device)

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break
            sample = _prepare_batch(batch, device)
            loss, _ = forecaster.training_loss(sample)
            bsz = float(sample["target"].shape[0])
            loss_sum += float(loss.item()) * bsz
            loss_count += bsz

            pred = forecaster.sample_next(
                context=sample["context"],
                input_geom=sample["input_geom"],
                latent_queries=sample["latent_queries"],
                output_queries=sample["output_queries"],
                stochastic=False,
                initial_latent=torch.zeros_like(sample["target"]),
            )
            tgt = sample["target"]
            err_norm = pred - tgt
            rmse_norm_sse += float(torch.sum(err_norm.pow(2)).item())
            rmse_norm_count += float(err_norm.numel())

            if target_norm is not None:
                pred_phys = target_norm.inverse_transform(pred)
                tgt_phys = target_norm.inverse_transform(tgt)
                err_phys = pred_phys - tgt_phys
                rmse_phys_sse += float(torch.sum(err_phys.pow(2)).item())
                rmse_phys_count += float(err_phys.numel())

    if dist_ctx.use_distributed and dist.is_initialized():
        for t in (
            loss_sum,
            loss_count,
            rmse_norm_sse,
            rmse_norm_count,
            rmse_phys_sse,
            rmse_phys_count,
        ):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_rmse_norm = torch.sqrt(rmse_norm_sse / torch.clamp(rmse_norm_count, min=1.0))
    if torch.all(rmse_phys_count <= 0):
        val_rmse_phys = torch.tensor(0.0, device=device, dtype=torch.float64)
    else:
        val_rmse_phys = torch.sqrt(rmse_phys_sse / torch.clamp(rmse_phys_count, min=1.0))

    out = {
        "val_loss": float((loss_sum / torch.clamp(loss_count, min=1.0)).item()),
        "val_rmse_norm": float(val_rmse_norm.item()),
        "val_rmse_phys": float(val_rmse_phys.item()),
    }
    forecaster.train()
    return out


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    time_mlp: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    seed: int,
    best_val_loss: float,
    normalizer_path: Path,
    target_variables: list[str],
    gino_cfg: Dict[str, Any],
    forecaster: ConditionalDDOForecaster,
    scheduler: Optional[Any] = None,
) -> None:
    denoiser_state = _unwrap_module(model).state_dict()
    time_mlp_state = _unwrap_module(time_mlp).state_dict() if time_mlp is not None else None

    metadata = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "seed": int(seed),
        "best_val_loss": float(best_val_loss),
        "normalizer_path": str(normalizer_path),
        "target_variables": list(target_variables),
        "gino_config": gino_cfg,
        "diffusion_hparams": forecaster.diffusion_hparams(),
        "has_time_mlp_state_dict": time_mlp_state is not None,
    }
    payload = {
        **metadata,
        "denoiser_state_dict": denoiser_state,
        "time_mlp_state_dict": time_mlp_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    extra_sidecars: Dict[str, Dict[str, Any]] = {}
    if time_mlp_state is not None:
        extra_sidecars["time_mlp_state_dict"] = time_mlp_state
    save_checkpoint_sidecars(
        path,
        denoiser_state_dict=denoiser_state,
        metadata=metadata,
        extra_state_dicts=extra_sidecars,
    )


def _resolve_resume_checkpoint(config: Any) -> Optional[Path]:
    ckpt_cfg = safe_get(config, "checkpoint", {})
    resume_raw = safe_get(ckpt_cfg, "resume_from_dir", None)
    if not resume_raw:
        return None
    p = Path(str(resume_raw))
    if not p.is_absolute():
        p = (_SCRIPT_DIR / p).resolve()
    if p.is_file():
        return p
    if not p.exists():
        return None
    for name in ("checkpoint.pt", "checkpoint_best.pt"):
        cand = p / name
        if cand.exists():
            return cand
    return None


def main() -> int:
    config = _load_config(_REPO_ROOT / "config" / "gino_pluvial_flood_config_WV_depth_only_diffusion.yaml")
    dist_ctx = _init_distributed(config)
    device = _resolve_device(config, dist_ctx=dist_ctx)

    log_file = safe_get(config, "log_file", "train_diffusion.log")
    if not Path(log_file).is_absolute():
        log_file = str((_SCRIPT_DIR / log_file).resolve())
    logger = setup_logging(
        log_level=str(safe_get(config, "log_level", "INFO")),
        log_file=log_file,
        logger_name="flood_diffusion_train",
    )
    if dist_ctx.use_distributed and not dist_ctx.is_rank0:
        logger.setLevel(logging.ERROR)

    base_seed = int(safe_get(safe_get(config, "distributed", {}), "seed", 123))
    seed = int(base_seed + (dist_ctx.rank if dist_ctx.use_distributed else 0))
    deterministic = bool(safe_get(config, "deterministic", True))
    set_seed(seed, deterministic=deterministic)
    _rank0_info(
        logger,
        dist_ctx,
        "Using device=%s base_seed=%d rank_seed=%d deterministic=%s distributed=%s rank=%d/%d",
        device,
        base_seed,
        seed,
        deterministic,
        dist_ctx.use_distributed,
        dist_ctx.rank,
        dist_ctx.world_size,
    )

    target_variables = parse_target_variables(safe_get(safe_get(config, "data", {}), "target_variables", ["wd"]))
    if target_variables != ["wd"]:
        raise ValueError("Diffusion v1 supports depth-only target_variables=['wd'].")

    (
        train_loader,
        test_loader,
        train_sampler,
        _test_sampler,
        normalizers,
        normalizer_path,
        n_static,
        n_boundary_channels,
    ) = _prepare_datasets(
        config,
        target_variables=target_variables,
        logger=logger,
        dist_ctx=dist_ctx,
    )
    _rank0_info(
        logger,
        dist_ctx,
        "Prepared loaders: train_batches=%d test_batches=%d",
        len(train_loader),
        len(test_loader),
    )

    gino_cfg, _, total_in_channels, time_injection, time_embedding_dim = _configure_denoiser(
        config,
        n_static=n_static,
        n_boundary_channels=n_boundary_channels,
        n_target_channels=1,
    )
    model_cfg = {"arch": "gino", "gino": copy.deepcopy(gino_cfg)}
    # get_model() consumes/mutates config keys (e.g., pops data_channels), so
    # pass an isolated model config copy to keep our saved metadata stable.
    model = get_model(model_cfg).to(device)
    _rank0_info(
        logger,
        dist_ctx,
        "Initialized denoiser (GINO) with in_channels=%d (n_boundary_channels=%d)",
        total_in_channels,
        n_boundary_channels,
    )

    diff_cfg = safe_get(config, "diffusion", {})
    gp_cfg = safe_get(diff_cfg, "gp", {})
    cond_cfg_raw = safe_get(diff_cfg, "conditioning", {})
    cond_cfg = ConditioningConfig(
        add_noisy_target=bool(safe_get(cond_cfg_raw, "add_noisy_target", True)),
        add_time_features=bool(safe_get(cond_cfg_raw, "add_time_features", True)),
        time_feature_type=str(safe_get(cond_cfg_raw, "time_feature_type", "sincos")),
        time_injection=str(safe_get(cond_cfg_raw, "time_injection", "channel")).lower(),
        time_embedding_dim=int(safe_get(cond_cfg_raw, "time_embedding_dim", 32)),
        time_embedding_hidden_dim=int(safe_get(cond_cfg_raw, "time_embedding_hidden_dim", 128)),
        time_embedding_scale=float(safe_get(cond_cfg_raw, "time_embedding_scale", 10000.0)),
    )
    _rank0_info(
        logger,
        dist_ctx,
        (
            "Diffusion conditioning: time_injection=%s time_embedding_dim=%d "
            "add_noisy_target=%s add_time_features=%s total_in_channels=%d fno_norm=%s"
        ),
        cond_cfg.time_injection,
        cond_cfg.time_embedding_dim,
        cond_cfg.add_noisy_target,
        cond_cfg.add_time_features,
        total_in_channels,
        str(safe_get(gino_cfg, "fno_norm", "unknown")),
    )
    if cond_cfg.time_injection != time_injection:
        raise ValueError(
            "Mismatch between denoiser wiring and conditioning config: "
            f"time_injection={time_injection!r} vs {cond_cfg.time_injection!r}"
        )
    if cond_cfg.time_injection == "adain" and cond_cfg.time_embedding_dim != time_embedding_dim:
        raise ValueError(
            "Mismatch between denoiser fno_ada_in_dim and conditioning.time_embedding_dim: "
            f"{time_embedding_dim} vs {cond_cfg.time_embedding_dim}"
        )
    gp_sampler = PointRFFGaussianProcessSampler(
        dim=2,
        gp_type=str(safe_get(gp_cfg, "type", "rff_rbf")),
        sigma=float(safe_get(gp_cfg, "sigma", 1.0)),
        length_scale=float(safe_get(gp_cfg, "length_scale", 0.05)),
        rff_features=int(safe_get(gp_cfg, "rff_features", 256)),
        seed=seed,
    ).to(device)

    sched_cfg = safe_get(diff_cfg, "schedule", {})
    sampler_cfg = safe_get(diff_cfg, "sampler", {})
    forecaster = ConditionalDDOForecaster(
        denoiser=model,
        gp_sampler=gp_sampler,
        parameterization=str(safe_get(diff_cfg, "parameterization", "epsilon")),
        timestep_sampler=str(safe_get(diff_cfg, "timestep_sampler", "low_discrepancy")),
        lmbd0=float(safe_get(sched_cfg, "lmbd0", 10.0)),
        lmbd1=float(safe_get(sched_cfg, "lmbd1", -10.0)),
        weight_method=safe_get(sched_cfg, "weight_method", "shifted_sigmoid_2"),
        conditioning=cond_cfg,
        sampler_method=str(safe_get(sampler_cfg, "method", "denoise")),
        sampler_num_steps=int(safe_get(sampler_cfg, "num_steps", 40)),
        sampler_s_min=float(safe_get(sampler_cfg, "s_min", 1e-4)),
        sampler_return_mean_last=bool(safe_get(sampler_cfg, "return_mean_last", True)),
    ).to(device)

    dist_cfg = _cfg_dist(config)
    find_unused_parameters = bool(safe_get(dist_cfg, "find_unused_parameters", False))
    if dist_ctx.use_distributed:
        if device.type == "cuda":
            ddp_kwargs = dict(
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            ddp_kwargs = dict(find_unused_parameters=find_unused_parameters)
        model = DDP(model, **ddp_kwargs)
        forecaster.denoiser = model
        if forecaster.time_mlp is not None:
            forecaster.time_mlp = DDP(forecaster.time_mlp, **ddp_kwargs)
        _rank0_info(
            logger,
            dist_ctx,
            "Enabled DDP: world_size=%d local_rank=%d find_unused_parameters=%s",
            dist_ctx.world_size,
            dist_ctx.local_rank,
            find_unused_parameters,
        )

    opt_cfg = safe_get(config, "opt", {})
    optim_params = list(model.parameters())
    if forecaster.time_mlp is not None:
        optim_params.extend(list(forecaster.time_mlp.parameters()))
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(safe_get(opt_cfg, "learning_rate", 1e-4)),
        weight_decay=float(safe_get(opt_cfg, "weight_decay", 1e-4)),
    )
    scheduler = _init_scheduler(config, optimizer)

    run = _maybe_init_wandb(config, seed=seed, logger=logger, is_rank0=dist_ctx.is_rank0)
    wandb_finish_timeout_seconds = float(
        safe_get(safe_get(config, "wandb", {}), "finish_timeout_seconds", 120.0)
    )

    ckpt_dir = Path(str(safe_get(safe_get(config, "checkpoint", {}), "save_dir", "./checkpoints_WV_depth_only_diffusion")))
    if not ckpt_dir.is_absolute():
        ckpt_dir = (_SCRIPT_DIR / ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    allow_unsafe_legacy_load = bool(
        safe_get(safe_get(config, "checkpoint", {}), "allow_unsafe_legacy_load", True)
    )
    if allow_unsafe_legacy_load and dist_ctx.is_rank0:
        logger.warning(
            "checkpoint.allow_unsafe_legacy_load=true. Legacy pickle checkpoints must be trusted."
        )

    n_epochs = int(safe_get(opt_cfg, "n_epochs", 1))
    print_every = int(safe_get(config, "print_every", DEFAULT_PRINT_EVERY))
    max_val_batches = int(safe_get(config, "max_val_batches", DEFAULT_MAX_VAL_BATCHES))
    grad_clip = safe_get(opt_cfg, "clip_grad_norm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    resume_checkpoint = _resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        resume_bundle = load_checkpoint_bundle(
            resume_checkpoint,
            map_location="cpu",
            allow_unsafe_legacy_load=allow_unsafe_legacy_load,
            logger=logger,
        )
        _load_state_dict_compat(
            _unwrap_module(model),
            resume_bundle["denoiser_state_dict"],
            name="denoiser_state_dict",
        )
        resume_time_mlp = resume_bundle.get("time_mlp_state_dict", None)
        if forecaster.time_mlp is not None:
            if resume_time_mlp is not None:
                _load_state_dict_compat(
                    _unwrap_module(forecaster.time_mlp),
                    resume_time_mlp,
                    name="time_mlp_state_dict",
                )
            else:
                _rank0_info(
                    logger,
                    dist_ctx,
                    "Resume checkpoint %s has no time_mlp_state_dict; continuing with current initialization.",
                    resume_checkpoint,
                )
        if resume_bundle.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_bundle["optimizer_state_dict"])
            _optimizer_to_device(optimizer, device)
        if scheduler is not None and resume_bundle.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_bundle["scheduler_state_dict"])
        best_val_loss = float(resume_bundle.get("best_val_loss", best_val_loss))
        global_step = int(resume_bundle.get("global_step", 0))
        start_epoch = int(resume_bundle.get("epoch", 0)) + 1
        _rank0_info(
            logger,
            dist_ctx,
            "Resumed from %s: start_epoch=%d global_step=%d best_val_loss=%.6e",
            resume_checkpoint,
            start_epoch,
            global_step,
            best_val_loss,
        )

    _dist_barrier(dist_ctx)
    first_batch_checked = False
    _rank0_info(logger, dist_ctx, "Starting diffusion training for %d epochs", n_epochs)

    try:
        for epoch in range(start_epoch, n_epochs + 1):
            forecaster.train()
            epoch_loss = 0.0
            epoch_batches = 0
            t0 = time.time()

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_iter = train_loader
            pbar = None
            if dist_ctx.is_rank0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)
                train_iter = pbar

            for batch in train_iter:
                sample = _prepare_batch(batch, device)
                if not first_batch_checked:
                    batch_size = sample["context"].shape[0]
                    t_probe = torch.full(
                        (batch_size,),
                        0.5,
                        device=device,
                        dtype=sample["context"].dtype,
                    )
                    if cond_cfg.time_injection == "adain":
                        ada_probe = forecaster._build_time_adain(t_probe)
                        expected_shape = (batch_size, cond_cfg.time_embedding_dim)
                        if tuple(ada_probe.shape) != expected_shape:
                            raise ValueError(
                                f"AdaIN timestep embedding shape mismatch: got {tuple(ada_probe.shape)} "
                                f"expected {expected_shape}"
                            )
                    else:
                        if cond_cfg.time_injection != "channel":
                            raise ValueError(
                                f"Unexpected conditioning mode in first-batch check: {cond_cfg.time_injection!r}"
                            )
                    first_batch_checked = True
                optimizer.zero_grad(set_to_none=True)

                loss, stats = forecaster.training_loss(sample)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
                optimizer.step()

                global_step += 1
                epoch_loss += float(loss.item())
                epoch_batches += 1
                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss.item():.4e}")

                if run is not None:
                    wandb.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/mse_eps": stats["mse_eps"],
                            "train/t_mean": stats["t_mean"],
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )

                if global_step % max(1, print_every) == 0:
                    _rank0_info(
                        logger,
                        dist_ctx,
                        "epoch=%d step=%d loss=%.6e lr=%.6e",
                        epoch,
                        global_step,
                        float(loss.item()),
                        float(optimizer.param_groups[0]["lr"]),
                    )

            train_loss_epoch = _reduce_sum(epoch_loss, device=device, dist_ctx=dist_ctx)
            train_batches_global = _reduce_sum(epoch_batches, device=device, dist_ctx=dist_ctx)
            train_loss_epoch = train_loss_epoch / max(1.0, train_batches_global)
            val_stats = _evaluate_validation(
                forecaster=forecaster,
                loader=test_loader,
                device=device,
                target_norm=normalizers.get("target", None),
                dist_ctx=dist_ctx,
                max_batches=max_val_batches,
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_stats["val_loss"])
                else:
                    scheduler.step()

            elapsed = time.time() - t0
            _rank0_info(
                logger,
                dist_ctx,
                "Epoch %d/%d | train_loss=%.6e | val_loss=%.6e | val_rmse_norm=%.6e | val_rmse_phys=%.6e | %.2fs",
                epoch,
                n_epochs,
                train_loss_epoch,
                val_stats["val_loss"],
                val_stats["val_rmse_norm"],
                val_stats["val_rmse_phys"],
                elapsed,
            )

            if run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss_epoch": train_loss_epoch,
                        "val/loss": val_stats["val_loss"],
                        "val/rmse_norm": val_stats["val_rmse_norm"],
                        "val/rmse_phys": val_stats["val_rmse_phys"],
                    },
                    step=global_step,
                )

            if dist_ctx.is_rank0:
                latest_path = ckpt_dir / "checkpoint.pt"
                _save_checkpoint(
                    path=latest_path,
                    model=model,
                    time_mlp=forecaster.time_mlp,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    seed=seed,
                    best_val_loss=best_val_loss,
                    normalizer_path=normalizer_path,
                    target_variables=target_variables,
                    gino_cfg=gino_cfg,
                    forecaster=forecaster,
                    scheduler=scheduler,
                )

            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = float(val_stats["val_loss"])
                if dist_ctx.is_rank0:
                    best_path = ckpt_dir / "checkpoint_best.pt"
                    _save_checkpoint(
                        path=best_path,
                        model=model,
                        time_mlp=forecaster.time_mlp,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        seed=seed,
                        best_val_loss=best_val_loss,
                        normalizer_path=normalizer_path,
                        target_variables=target_variables,
                        gino_cfg=gino_cfg,
                        forecaster=forecaster,
                        scheduler=scheduler,
                    )
                    logger.info("Saved new best checkpoint: %s", best_path)
            _dist_barrier(dist_ctx)

        _rank0_info(
            logger,
            dist_ctx,
            "Training complete. Best val_loss=%.6e | checkpoints=%s",
            best_val_loss,
            ckpt_dir,
        )

        metadata = {
            "checkpoint_dir": str(ckpt_dir),
            "best_val_loss": best_val_loss,
            "normalizer_path": str(normalizer_path),
            "target_variables": target_variables,
            "global_step": global_step,
        }
        if dist_ctx.is_rank0:
            with open(ckpt_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        return 0
    finally:
        shutdown_dataloader_workers(train_loader, logger=logger, name="train_loader")
        shutdown_dataloader_workers(test_loader, logger=logger, name="test_loader")
        if run is not None:
            safe_wandb_finish(run, logger=logger, timeout_seconds=wandb_finish_timeout_seconds)
        if dist_ctx.use_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
