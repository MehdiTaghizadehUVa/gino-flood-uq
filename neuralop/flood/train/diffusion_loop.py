"""Validation/checkpoint loop helpers for flood diffusion training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from neuralop.diffusion import ConditionalDDOForecaster
from neuralop.flood.train.diffusion_data import _prepare_batch
from neuralop.flood.train.diffusion_runtime import (
    DEFAULT_MAX_VAL_BATCHES,
    DistContext,
    _SCRIPT_DIR,
    _unwrap_module,
)
from neuralop.flood.utils.diffusion_script_utils import save_checkpoint_sidecars, safe_get

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
        raise FileNotFoundError(f"checkpoint.resume_from_dir does not exist: {p}")
    for name in ("checkpoint.pt", "checkpoint_best.pt"):
        cand = p / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "checkpoint.resume_from_dir does not contain checkpoint.pt or checkpoint_best.pt: "
        f"{p}"
    )
