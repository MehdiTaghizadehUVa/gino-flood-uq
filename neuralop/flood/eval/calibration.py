"""Lead-time calibration helpers for operator flood evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from neuralop.flood.eval.metrics import _build_member_model_indices, _sample_from_packed_gaussian
from neuralop.flood.eval.runtime import (
    CHANNEL_INDEX,
    ROLLOUT_INIT_MEMBER_HISTORY,
    UQ_OVERALL_JSON,
    build_rollout_initial_histories,
    normalize_rollout_init_mode,
)
from neuralop.flood.train.operator import (
    get_fgn_rollout_latent,
    sample_fgn_rollout_latent_bank,
    update_fgn_dynamic_members,
)
from neuralop.flood.utils.runtime import (
    normalize_fgn_ar_state_update,
    normalize_fgn_latent_temporal_mode,
)
from neuralop.training.leadtime_affine_calibration import (
    apply_leadtime_affine_to_ensemble,
    fit_leadtime_affine_calibration,
)

CALIBRATION_NPZ = "leadtime_affine_wd.npz"
CALIBRATION_SUMMARY_JSON = "leadtime_affine_wd_summary.json"
CALIBRATION_COMPARISON_JSON = "calibration_comparison.json"

def _apply_wd_calibration_to_prediction_ensemble(
    pred_ens: torch.Tensor,
    target_variables: List[str],
    *,
    a_t: float,
    b_t: float,
    c_t: float,
) -> torch.Tensor:
    """Apply lead-time affine calibration to WD ensemble members (physical space)."""
    if "wd" not in target_variables:
        raise ValueError("WD calibration requested but 'wd' is not in target_variables.")
    wd_idx = target_variables.index("wd")
    wd_pred_np = pred_ens[:, :, wd_idx].detach().cpu().numpy()
    wd_cal_np = apply_leadtime_affine_to_ensemble(
        wd_pred_np, a_t=float(a_t), b_t=float(b_t), c_t=float(c_t)
    )
    pred_cal = pred_ens.clone()
    pred_cal[:, :, wd_idx] = torch.from_numpy(wd_cal_np).to(
        device=pred_ens.device, dtype=pred_ens.dtype
    )
    return pred_cal


def _fit_wd_leadtime_calibration_from_hydrographs(
    models: List[Any],
    hydrograph_samples: List[Dict[str, Any]],
    rollout_length: int,
    history_steps: int,
    dynamic_norm: Any,
    target_norm: Any,
    device: torch.device,
    skip_before_timestep: int,
    target_variables: List[str],
    logger: logging.Logger,
    fgn_noise_dim: Optional[int],
    n_ensemble_samples: int,
    fgn_latent_temporal_mode: str = "stepwise",
    fgn_ar_state_update: str = "mean_feedback",
    gaussian_mode: bool = False,
    gaussian_min_logvar: float = -9.0,
    gaussian_max_logvar: float = 4.0,
    gaussian_state_update: str = "sample",
    rollout_init_mode: str = "mean_history",
    fit_wet_threshold: float = 0.01,
    min_pred_std: float = 1e-4,
    c_clip_min: float = 0.25,
    c_clip_max: float = 4.0,
    smooth_window: int = 5,
) -> Dict[str, Any]:
    """Fit WD lead-time affine calibration from grouped-hydrograph rollout predictions."""
    if "wd" not in target_variables:
        raise ValueError("WD calibration requires 'wd' to be present in target_variables.")
    if not models:
        raise ValueError("No models provided for calibration fitting.")
    if not hydrograph_samples:
        raise ValueError("No grouped hydrograph samples provided for calibration fitting.")

    for model in models:
        model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)

    wd_idx = target_variables.index("wd")
    n_models = len(models)
    n_ens = max(1, int(n_ensemble_samples))
    if n_models > 1 and n_ens < n_models:
        n_ens = n_models
    use_ensemble = n_ens > 1 or n_models > 1
    member_model_indices = _build_member_model_indices(n_models, n_ens)
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
        fgn_latent_temporal_mode
    )
    fgn_ar_state_update = normalize_fgn_ar_state_update(fgn_ar_state_update)
    rollout_init_mode = normalize_rollout_init_mode(rollout_init_mode)
    gaussian_state_update = str(gaussian_state_update).strip().lower()
    if gaussian_mode and gaussian_state_update not in {"sample", "mu"}:
        raise ValueError(
            "rollout.gaussian_state_update must be one of {'sample', 'mu'} "
            f"for gaussian mode, got {gaussian_state_update!r}."
        )

    start_pred_t = skip_before_timestep + history_steps
    end_pred_t = start_pred_t + rollout_length
    mu_pred_by_t: List[List[np.ndarray]] = [[] for _ in range(rollout_length)]
    sigma_pred_by_t: List[List[np.ndarray]] = [[] for _ in range(rollout_length)]
    mu_ref_by_t: List[List[np.ndarray]] = [[] for _ in range(rollout_length)]
    sigma_ref_by_t: List[List[np.ndarray]] = [[] for _ in range(rollout_length)]
    domain_mask_by_t: List[List[np.ndarray]] = [[] for _ in range(rollout_length)]

    logger.info("Calibration rollout initialization mode='%s'", rollout_init_mode)
    member_history_mapping_logged = False
    for sample in tqdm(hydrograph_samples, desc="Calibration rollout (hydrograph)"):
        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = sample["geometry"].to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)
        full_boundary = sample["boundary"].to(device)
        dynamic_ref = sample["dynamic_ref"].to(device)

        gt_rollout_ref = dynamic_ref[:, start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]
        structural_dry_mask = sample.get("structural_dry_mask")
        if structural_dry_mask is not None:
            structural_dry_mask = (
                torch.as_tensor(structural_dry_mask, dtype=torch.bool)
                .reshape(-1)
                .cpu()
                .numpy()
            )
            structural_wettable_mask = ~structural_dry_mask
        else:
            structural_wettable_mask = None

        init_histories, init_ref_indices = build_rollout_initial_histories(
            dynamic_ref,
            skip_before_timestep=skip_before_timestep,
            start_pred_t=start_pred_t,
            n_members=n_ens if use_ensemble else 1,
            rollout_init_mode=rollout_init_mode,
        )
        if (
            rollout_init_mode == ROLLOUT_INIT_MEMBER_HISTORY
            and not member_history_mapping_logged
        ):
            hydro_id = str(sample.get("hydrograph_id", "unknown"))
            reference_run_ids = list(sample.get("reference_run_ids", []))
            if reference_run_ids:
                selected_refs = [
                    reference_run_ids[idx] for idx in init_ref_indices[: min(len(init_ref_indices), n_ens if use_ensemble else 1)]
                ]
                logger.info(
                    "Member-history calibration initialization for hydrograph %s uses reference members=%s",
                    hydro_id,
                    selected_refs,
                )
            else:
                logger.info(
                    "Member-history calibration initialization for hydrograph %s uses reference indices=%s",
                    hydro_id,
                    init_ref_indices[: min(len(init_ref_indices), n_ens if use_ensemble else 1)],
                )
            member_history_mapping_logged = True
        if use_ensemble:
            current_dynamics = [hist.clone() for hist in init_histories]
        else:
            current_dynamic = init_histories[0].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()
        fgn_latent_bank = None
        if fgn_noise_dim is not None:
            fgn_latent_bank = sample_fgn_rollout_latent_bank(
                num_members=n_ens if use_ensemble else 1,
                batch_size=1,
                latent_dim=fgn_noise_dim,
                device=device,
                dtype=dynamic_ref.dtype,
                temporal_mode=fgn_latent_temporal_mode,
            )

        for t in range(rollout_length):
            state_stack: Optional[torch.Tensor] = None
            with torch.no_grad():
                if use_ensemble:
                    pred_members: List[torch.Tensor] = []
                    state_members: List[torch.Tensor] = []
                    for ens_idx in range(n_ens):
                        dyn_hist = current_dynamics[ens_idx]
                        model_idx = member_model_indices[ens_idx]
                        model = models[model_idx]
                        dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(
                            1, current_boundary.shape[1], -1
                        )
                        x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                        if gaussian_mode:
                            out = model(
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
                            )
                            sampled, mu, _ = _sample_from_packed_gaussian(
                                out,
                                n_channels=int(dynamic_ref.shape[-1]),
                                min_logvar=gaussian_min_logvar,
                                max_logvar=gaussian_max_logvar,
                            )
                            pred_members.append(sampled)
                            state_members.append(mu if gaussian_state_update == "mu" else sampled)
                        elif fgn_noise_dim is not None:
                            z = get_fgn_rollout_latent(
                                latent_bank=fgn_latent_bank,
                                member_idx=ens_idx,
                                batch_size=x.shape[0],
                                latent_dim=fgn_noise_dim,
                                device=device,
                                dtype=x.dtype,
                            )
                            pred = model(
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
                                ada_in=z,
                            )
                            pred_members.append(pred)
                            state_members.append(pred)
                        else:
                            pred = model(
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
                            )
                            pred_members.append(pred)
                            state_members.append(pred)
                    pred_stack = torch.stack(pred_members, dim=0)
                    state_stack = torch.stack(state_members, dim=0)
                else:
                    model = models[0]
                    dyn_flat = current_dynamic.permute(1, 0, 2).reshape(
                        1, current_dynamic.shape[1], -1
                    )
                    bc_flat = current_boundary.permute(1, 0, 2).reshape(
                        1, current_boundary.shape[1], -1
                    )
                    x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                    if gaussian_mode:
                        out = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                        )
                        sampled, mu, _ = _sample_from_packed_gaussian(
                            out,
                            n_channels=int(dynamic_ref.shape[-1]),
                            min_logvar=gaussian_min_logvar,
                            max_logvar=gaussian_max_logvar,
                        )
                        pred_stack = sampled.unsqueeze(0)
                        state_stack = (
                            mu if gaussian_state_update == "mu" else sampled
                        ).unsqueeze(0)
                    elif fgn_noise_dim is not None:
                        z = get_fgn_rollout_latent(
                            latent_bank=fgn_latent_bank,
                            member_idx=0,
                            batch_size=x.shape[0],
                            latent_dim=fgn_noise_dim,
                            device=device,
                            dtype=x.dtype,
                        )
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                        )
                        pred_stack = pred.unsqueeze(0)
                        state_stack = pred.unsqueeze(0)
                    else:
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                        )
                        pred_stack = pred.unsqueeze(0)
                        state_stack = pred.unsqueeze(0)

            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))
            inv_gt_ref = dynamic_norm.inverse_transform(gt_rollout_ref[:, t])
            pred_wd = inv_pred_ens[:, :, wd_idx].detach().cpu().numpy()
            gt_wd = inv_gt_ref[:, :, wd_idx].detach().cpu().numpy()
            mu_pred_by_t[t].append(np.mean(pred_wd, axis=0))
            sigma_pred_by_t[t].append(np.std(pred_wd, axis=0, ddof=0))
            mu_ref_by_t[t].append(np.mean(gt_wd, axis=0))
            sigma_ref_by_t[t].append(np.std(gt_wd, axis=0, ddof=0))
            if structural_wettable_mask is not None:
                domain_mask_by_t[t].append(
                    np.asarray(structural_wettable_mask, dtype=bool).copy()
                )

            if use_ensemble:
                if state_stack is None:
                    raise RuntimeError("Internal error: missing state stack in ensemble mode.")
                if fgn_noise_dim is not None and not gaussian_mode:
                    current_dynamics = update_fgn_dynamic_members(
                        dynamic_members=[dyn_hist.unsqueeze(0) for dyn_hist in current_dynamics],
                        pred_samples=state_stack,
                        pred_mean=state_stack.mean(dim=0),
                        n_history=history_steps,
                        state_update_mode=fgn_ar_state_update,
                    )
                    current_dynamics = [dyn_hist.squeeze(0) for dyn_hist in current_dynamics]
                else:
                    for ens_idx in range(n_ens):
                        current_dynamics[ens_idx] = torch.cat(
                            [current_dynamics[ens_idx][1:], state_stack[ens_idx, 0].unsqueeze(0)],
                            dim=0,
                        )
            else:
                if state_stack is None:
                    raise RuntimeError("Internal error: missing state stack in single-member mode.")
                current_dynamic = torch.cat(
                    [current_dynamic[1:], state_stack[0, 0].unsqueeze(0)],
                    dim=0,
                )
            current_boundary = torch.cat(
                [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
            )

    def _flatten_by_lead(values: List[List[np.ndarray]], name: str) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for t in range(rollout_length):
            if not values[t]:
                raise ValueError(
                    f"No calibration samples collected at lead t={t} for {name}."
                )
            out.append(np.concatenate(values[t], axis=0).astype(np.float64, copy=False))
        return out

    coeffs = fit_leadtime_affine_calibration(
        _flatten_by_lead(mu_pred_by_t, "mu_pred"),
        _flatten_by_lead(sigma_pred_by_t, "sigma_pred"),
        _flatten_by_lead(mu_ref_by_t, "mu_ref"),
        _flatten_by_lead(sigma_ref_by_t, "sigma_ref"),
        domain_mask_by_t=(
            _flatten_by_lead(domain_mask_by_t, "domain_mask")
            if any(domain_mask_by_t[t] for t in range(rollout_length))
            else None
        ),
        fit_wet_threshold=float(fit_wet_threshold),
        min_pred_std=float(min_pred_std),
        c_clip_min=float(c_clip_min),
        c_clip_max=float(c_clip_max),
        smooth_window=int(smooth_window),
    )
    logger.info(
        "Fitted WD lead-time affine calibration on %d hydrographs | mean(b)=%.4f mean(c)=%.4f",
        len(hydrograph_samples),
        float(np.mean(coeffs["b"])),
        float(np.mean(coeffs["c"])),
    )
    return coeffs


def _save_calibration_artifacts(
    out_dir: str,
    coeffs: Dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """Persist calibration arrays and a readable summary JSON."""
    calib_dir = Path(out_dir) / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    npz_path = calib_dir / CALIBRATION_NPZ
    np.savez(
        npz_path,
        a=np.asarray(coeffs["a"], dtype=np.float64),
        b=np.asarray(coeffs["b"], dtype=np.float64),
        c=np.asarray(coeffs["c"], dtype=np.float64),
        n_fit=np.asarray(coeffs["n_fit"], dtype=np.int64),
        n_spread=np.asarray(coeffs["n_spread"], dtype=np.int64),
        fit_wet_threshold=float(coeffs["fit_wet_threshold"]),
        min_pred_std=float(coeffs["min_pred_std"]),
        c_clip_min=float(coeffs["c_clip_min"]),
        c_clip_max=float(coeffs["c_clip_max"]),
        smooth_window=int(coeffs["smooth_window"]),
    )
    summary = {
        "n_lead_steps": int(len(coeffs["a"])),
        "a_mean": float(np.mean(coeffs["a"])),
        "b_mean": float(np.mean(coeffs["b"])),
        "c_mean": float(np.mean(coeffs["c"])),
        "a_min": float(np.min(coeffs["a"])),
        "a_max": float(np.max(coeffs["a"])),
        "b_min": float(np.min(coeffs["b"])),
        "b_max": float(np.max(coeffs["b"])),
        "c_min": float(np.min(coeffs["c"])),
        "c_max": float(np.max(coeffs["c"])),
        "fit_wet_threshold": float(coeffs["fit_wet_threshold"]),
        "min_pred_std": float(coeffs["min_pred_std"]),
        "c_clip_min": float(coeffs["c_clip_min"]),
        "c_clip_max": float(coeffs["c_clip_max"]),
        "smooth_window": int(coeffs["smooth_window"]),
        "n_fit_total": int(np.sum(coeffs["n_fit"])),
        "n_spread_total": int(np.sum(coeffs["n_spread"])),
    }
    json_path = calib_dir / CALIBRATION_SUMMARY_JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Saved WD calibration arrays to %s", npz_path)
    logger.info("Saved WD calibration summary to %s", json_path)
    return npz_path


def _build_calibration_comparison(
    raw_metrics_path: Path,
    calibrated_metrics_path: Path,
    out_path: Path,
    logger: logging.Logger,
) -> None:
    """Create side-by-side comparison JSON for raw vs calibrated overall UQ metrics."""
    if not raw_metrics_path.exists():
        raise FileNotFoundError(f"Raw metrics JSON not found: {raw_metrics_path}")
    if not calibrated_metrics_path.exists():
        raise FileNotFoundError(
            f"Calibrated metrics JSON not found: {calibrated_metrics_path}"
        )
    with open(raw_metrics_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(calibrated_metrics_path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for key in sorted(set(raw.keys()).intersection(set(cal.keys()))):
        rv = raw[key]
        cv = cal[key]
        if isinstance(rv, bool) or isinstance(cv, bool):
            continue
        if not isinstance(rv, (int, float)) or not isinstance(cv, (int, float)):
            continue
        rv_f = float(rv)
        cv_f = float(cv)
        if not np.isfinite(rv_f) or not np.isfinite(cv_f):
            continue
        delta = cv_f - rv_f
        pct = None if abs(rv_f) < 1e-12 else 100.0 * delta / abs(rv_f)
        metrics[key] = {
            "raw": rv_f,
            "calibrated": cv_f,
            "delta": float(delta),
            "percent_change": None if pct is None else float(pct),
        }

    payload = {
        "raw_metrics_json": str(raw_metrics_path),
        "calibrated_metrics_json": str(calibrated_metrics_path),
        "n_metrics_compared": int(len(metrics)),
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Saved calibration comparison JSON to %s", out_path)
