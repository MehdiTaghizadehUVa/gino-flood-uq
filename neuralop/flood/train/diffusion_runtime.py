"""Distributed/runtime helpers for flood diffusion training."""

from __future__ import annotations

import copy
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from neuralop.flood.utils.diffusion_script_utils import safe_get, to_builtin
from neuralop.utils import get_wandb_api_key

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = _REPO_ROOT / 'scripts'
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
