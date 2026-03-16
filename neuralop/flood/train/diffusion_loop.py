"""Validation/checkpoint loop helpers for flood diffusion training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from neuralop.diffusion import ConditionalDDOForecaster
from neuralop.flood.losses import dry_falsewet_rate
from neuralop.flood.train.diffusion_data import _prepare_batch
from neuralop.flood.train.diffusion_runtime import (
    DEFAULT_MAX_VAL_BATCHES,
    DistContext,
    _SCRIPT_DIR,
    _unwrap_module,
)
from neuralop.flood.utils.diffusion_script_utils import save_checkpoint_sidecars, safe_get
from neuralop.training.determinism import (
    collect_rng_state_across_ranks,
    deterministic_seed_context,
    stable_seed_from_parts,
)

def _evaluate_validation(
    forecaster: ConditionalDDOForecaster,
    loader: DataLoader,
    device: torch.device,
    target_norm: Optional[Any],
    dist_ctx: DistContext,
    max_batches: int = DEFAULT_MAX_VAL_BATCHES,
    deterministic_eval: bool = False,
    eval_seed: int | None = None,
    epoch: int | None = None,
) -> Dict[str, float]:
    forecaster.eval()
    loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    loss_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_norm_sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_norm_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_phys_sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_phys_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    loss_sum_full = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_norm_sse_full = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_norm_count_full = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_phys_sse_full = torch.tensor(0.0, device=device, dtype=torch.float64)
    rmse_phys_count_full = torch.tensor(0.0, device=device, dtype=torch.float64)
    dry_rmse_sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    dry_rmse_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    dry_mae_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    dry_mae_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    falsewet_001_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    falsewet_005_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    falsewet_count = torch.tensor(0.0, device=device, dtype=torch.float64)

    if target_norm is not None:
        target_norm.to(device)

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break
            sample = _prepare_batch(batch, device)
            full_loss_input = {k: v for k, v in sample.items() if k != "point_weights"}
            batch_seed = None
            if deterministic_eval and eval_seed is not None:
                batch_seed = stable_seed_from_parts(
                    "diffusion_val",
                    int(eval_seed),
                    int(epoch) if epoch is not None else -1,
                    int(bidx),
                )
            with deterministic_seed_context(batch_seed):
                loss, _ = forecaster.training_loss(sample)
            with deterministic_seed_context(batch_seed):
                loss_full, _ = forecaster.training_loss(full_loss_input)
            bsz = float(sample["target"].shape[0])
            loss_sum += float(loss.item()) * bsz
            loss_count += bsz
            loss_sum_full += float(loss_full.item()) * bsz

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
            point_weights = sample.get("point_weights")
            if point_weights is not None:
                weights = point_weights.to(device=device, dtype=err_norm.dtype)
                rmse_norm_sse += float(torch.sum(err_norm.pow(2) * weights).item())
                rmse_norm_count += float(weights.sum().item())
            else:
                rmse_norm_sse += float(torch.sum(err_norm.pow(2)).item())
                rmse_norm_count += float(err_norm.numel())
            rmse_norm_sse_full += float(torch.sum(err_norm.pow(2)).item())
            rmse_norm_count_full += float(err_norm.numel())

            if target_norm is not None:
                pred_phys = target_norm.inverse_transform(pred)
                tgt_phys = target_norm.inverse_transform(tgt)
                err_phys = pred_phys - tgt_phys
                if point_weights is not None:
                    weights_phys = point_weights.to(device=device, dtype=err_phys.dtype)
                    rmse_phys_sse += float(torch.sum(err_phys.pow(2) * weights_phys).item())
                    rmse_phys_count += float(weights_phys.sum().item())
                else:
                    rmse_phys_sse += float(torch.sum(err_phys.pow(2)).item())
                    rmse_phys_count += float(err_phys.numel())
                rmse_phys_sse_full += float(torch.sum(err_phys.pow(2)).item())
                rmse_phys_count_full += float(err_phys.numel())

                structural_dry_mask = sample.get("structural_dry_mask")
                if structural_dry_mask is not None:
                    dry_mask = structural_dry_mask.to(device=device, dtype=torch.bool)
                    if dry_mask.ndim == 1:
                        dry_mask = dry_mask.unsqueeze(0).expand(pred.shape[0], -1)
                    dry_mask_exp = dry_mask.unsqueeze(-1).expand_as(pred_phys)
                    dry_rmse_sse += float(torch.sum((err_phys.pow(2) * dry_mask_exp).double()).item())
                    dry_rmse_count += float(dry_mask_exp.sum().item())
                    dry_mae_sum += float(torch.sum((err_phys.abs() * dry_mask_exp).double()).item())
                    dry_mae_count += float(dry_mask_exp.sum().item())
                    falsewet_001_sum += float(
                        dry_falsewet_rate(
                            pred_phys,
                            structural_dry_mask=dry_mask,
                            threshold=0.01,
                        ).item()
                        * pred.shape[0]
                    )
                    falsewet_005_sum += float(
                        dry_falsewet_rate(
                            pred_phys,
                            structural_dry_mask=dry_mask,
                            threshold=0.05,
                        ).item()
                        * pred.shape[0]
                    )
                    falsewet_count += float(pred.shape[0])

    if dist_ctx.use_distributed and dist.is_initialized():
        for t in (
            loss_sum,
            loss_count,
            loss_sum_full,
            rmse_norm_sse,
            rmse_norm_count,
            rmse_norm_sse_full,
            rmse_norm_count_full,
            rmse_phys_sse,
            rmse_phys_count,
            rmse_phys_sse_full,
            rmse_phys_count_full,
            dry_rmse_sse,
            dry_rmse_count,
            dry_mae_sum,
            dry_mae_count,
            falsewet_001_sum,
            falsewet_005_sum,
            falsewet_count,
        ):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_rmse_norm = torch.sqrt(rmse_norm_sse / torch.clamp(rmse_norm_count, min=1.0))
    if torch.all(rmse_phys_count <= 0):
        val_rmse_phys = torch.tensor(0.0, device=device, dtype=torch.float64)
    else:
        val_rmse_phys = torch.sqrt(rmse_phys_sse / torch.clamp(rmse_phys_count, min=1.0))
    val_rmse_norm_full = torch.sqrt(rmse_norm_sse_full / torch.clamp(rmse_norm_count_full, min=1.0))
    if torch.all(rmse_phys_count_full <= 0):
        val_rmse_phys_full = torch.tensor(0.0, device=device, dtype=torch.float64)
    else:
        val_rmse_phys_full = torch.sqrt(
            rmse_phys_sse_full / torch.clamp(rmse_phys_count_full, min=1.0)
        )

    out = {
        "val_loss": float((loss_sum / torch.clamp(loss_count, min=1.0)).item()),
        "val_rmse_norm": float(val_rmse_norm.item()),
        "val_rmse_phys": float(val_rmse_phys.item()),
        "val_loss_full_domain": float((loss_sum_full / torch.clamp(loss_count, min=1.0)).item()),
        "val_rmse_norm_full_domain": float(val_rmse_norm_full.item()),
        "val_rmse_phys_full_domain": float(val_rmse_phys_full.item()),
    }
    if dry_rmse_count.item() > 0:
        out["val_rmse_dry_background_wd"] = float(
            torch.sqrt(dry_rmse_sse / torch.clamp(dry_rmse_count, min=1.0)).item()
        )
        out["val_mae_dry_background_wd"] = float(
            (dry_mae_sum / torch.clamp(dry_mae_count, min=1.0)).item()
        )
        out["val_falsewet_rate_001_dry_background_wd"] = float(
            (falsewet_001_sum / torch.clamp(falsewet_count, min=1.0)).item()
        )
        out["val_falsewet_rate_005_dry_background_wd"] = float(
            (falsewet_005_sum / torch.clamp(falsewet_count, min=1.0)).item()
        )
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
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
    rng_state_bundle = collect_rng_state_across_ranks()
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
        "has_rng_state": True,
    }
    payload = {
        **metadata,
        "denoiser_state_dict": denoiser_state,
        "time_mlp_state_dict": time_mlp_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state_bundle,
    }
    if is_rank0:
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
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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
