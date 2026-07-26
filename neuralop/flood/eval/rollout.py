"""Rollout evaluation helpers for operator-style flood models."""

from __future__ import annotations

import logging
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from neuralop.flood.losses import FloodMaskedRelLpLoss
from neuralop.flood.data.structural_dry import (
    apply_structural_dry_zero_mask,
    clamp_structural_dry_normalized_values,
)
from neuralop.flood.eval.metrics import (
    _build_member_model_indices,
    _compute_csi,
    _crps_ensemble_vs_reference,
    _variance_decomposition_by_model,
    _is_gaussian_mode,
    _pit_rank_counts_from_reference,
)
from neuralop.flood.eval.impact_metrics import (
    compute_flood_impact_crps_metrics,
    normalize_impact_metrics_config,
)
from neuralop.flood.eval.alr_fgn import ALRMemberLayout, forward_alr_rollout_step
from neuralop.flood.train.alr_fgn import clamp_nested_feedback
from neuralop.flood.eval.mc_dropout import (
    enable_mc_dropout_only,
    mc_dropout_seed_context,
)
from neuralop.flood.eval.render import (
    _save_nonspatial_uq_diagnostics,
    _save_generic_rollout_visuals,
    _save_hydrograph_uq_figures_and_animation,
)
from neuralop.flood.eval.scientific_calibration import (
    ARTIFACT_FILE_SUFFIX,
    apply_crps_mbm_to_wd_members,
    save_forecast_artifact,
)
from neuralop.flood.eval.runtime import (
    LEGACY_3CH,
    MIN_EPS,
    PUBLICATION_TIMESTEPS,
    ROLLOUT_INIT_MEMBER_HISTORY,
    ROLLOUT_METRICS_HYDRO_NPZ,
    ROLLOUT_METRICS_NPZ,
    ROLLOUT_SUMMARY_HYDRO_FULL_PNG,
    ROLLOUT_SUMMARY_HYDRO_PNG,
    ROLLOUT_SUMMARY_PNG,
    UQ_EXCEEDANCE_THRESHOLD,
    _opt,
    _opt_float,
    build_rollout_initial_histories,
    normalize_rollout_init_mode,
)
from neuralop.flood.train.operator import (
    FGNTrainer,
    GaussianNLLTrainer,
    get_fgn_rollout_latent,
    sample_fgn_rollout_latent_bank,
    update_fgn_dynamic_members,
)
from neuralop.flood.processing.wv_impl import _sample_from_packed_gaussian
from neuralop.flood.utils.runtime import (
    normalize_fgn_ar_state_update,
    normalize_fgn_latent_temporal_mode,
)
from neuralop.flood.visualization.publication import (
    create_rollout_animation,
    generate_publication_maps,
)
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer


def _structural_masks_from_sample(
    sample: Dict[str, Any],
    *,
    expected_cells: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    dry_mask = sample.get("structural_dry_mask")
    if dry_mask is None:
        return None, None
    dry_mask = torch.as_tensor(dry_mask, dtype=torch.bool, device="cpu").reshape(-1).numpy()
    if dry_mask.size != expected_cells:
        raise ValueError(
            f"structural_dry_mask has {dry_mask.size} cells, expected {expected_cells}."
        )
    return dry_mask, ~dry_mask


def _select_cells(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.asarray(arr)
    return np.asarray(arr)[..., mask]


def _masked_mean(arr: np.ndarray, mask: np.ndarray | None) -> float:
    selected = _select_cells(arr, mask)
    if selected.size == 0:
        return 0.0
    return float(np.mean(selected))


def _masked_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> float:
    diff = np.asarray(pred) - np.asarray(gt)
    return float(np.sqrt(_masked_mean(diff ** 2, mask)))


def _masked_relative_l2(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> float:
    pred_sel = np.asarray(_select_cells(pred, mask), dtype=np.float64).reshape(-1)
    gt_sel = np.asarray(_select_cells(gt, mask), dtype=np.float64).reshape(-1)
    if pred_sel.size == 0 or gt_sel.size == 0:
        return 0.0
    denom = max(float(np.linalg.norm(gt_sel)), MIN_EPS)
    return float(np.linalg.norm(pred_sel - gt_sel) / denom)


def _masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> float:
    return _masked_mean(np.abs(np.asarray(pred) - np.asarray(gt)), mask)


def _masked_csi(
    threshold: float,
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    pred_sel = _select_cells(pred, mask)
    gt_sel = _select_cells(gt, mask)
    if pred_sel.size == 0:
        return 1.0
    return float(_compute_csi(threshold, pred_sel, gt_sel))


def _dry_falsewet_rate(pred: np.ndarray, dry_mask: np.ndarray | None, threshold: float) -> float:
    if dry_mask is None or not np.any(dry_mask):
        return 0.0
    pred_sel = _select_cells(pred, dry_mask)
    if pred_sel.size == 0:
        return 0.0
    return float(np.mean(pred_sel > float(threshold)))


def _dry_pred_std_mean(pred_std: np.ndarray, dry_mask: np.ndarray | None) -> float:
    if dry_mask is None or not np.any(dry_mask):
        return 0.0
    pred_sel = _select_cells(pred_std, dry_mask)
    if pred_sel.size == 0:
        return 0.0
    return float(np.mean(pred_sel))


def _normalize_member_boundary_mode(value: Any) -> str:
    """Normalize optional grouped-rollout boundary source for forecast members."""
    mode = str(value or "shared").strip().lower().replace("-", "_")
    aliases = {
        "shared": "shared",
        "clean": "shared",
        "clean_family": "shared",
        "reference": "reference_member",
        "reference_member": "reference_member",
        "member": "reference_member",
        "member_hdf": "reference_member",
    }
    if mode not in aliases:
        raise ValueError(
            "rollout.member_boundary_mode must be one of {'shared', 'reference_member'}, "
            f"got {value!r}."
        )
    return aliases[mode]


def _broadcast_boundary_series_to_cells(
    series: torch.Tensor,
    *,
    n_cells: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand a [time, bc_dim] boundary series to [time, n_cells, bc_dim]."""
    if series.ndim != 2:
        raise ValueError(
            f"Expected boundary series with shape [time, channels], got {tuple(series.shape)}."
        )
    return series.to(device).unsqueeze(1).expand(-1, int(n_cells), -1).clone()


def _paired_member_rmse(
    pred_members: np.ndarray,
    ref_members: np.ndarray,
    ref_indices: List[int],
    mask: np.ndarray | None,
) -> float:
    """RMSE for forecast members paired with their source reference realization."""
    pred = np.asarray(pred_members, dtype=np.float64)
    ref = np.asarray(ref_members, dtype=np.float64)
    idx = np.asarray(ref_indices, dtype=np.int64)[: pred.shape[0]]
    valid = (idx >= 0) & (idx < ref.shape[0])
    if not np.any(valid):
        return float("nan")
    pred_valid = pred[np.flatnonzero(valid)]
    ref_valid = ref[idx[valid]]
    diff = pred_valid - ref_valid
    if mask is not None:
        diff = diff[:, np.asarray(mask, dtype=bool)]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


def _forward_operator_model(
    model: Any,
    *,
    input_geom: torch.Tensor,
    latent_queries: torch.Tensor,
    output_queries: torch.Tensor,
    x: torch.Tensor,
    ada_in: Optional[torch.Tensor] = None,
    mc_dropout_enabled: bool = False,
    mc_dropout_seed: Optional[int] = None,
    mc_seed_parts: tuple[Any, ...] = (),
) -> torch.Tensor:
    kwargs = {
        "input_geom": input_geom,
        "latent_queries": latent_queries,
        "output_queries": output_queries,
        "x": x,
    }
    if ada_in is not None:
        kwargs["ada_in"] = ada_in
    with mc_dropout_seed_context(mc_dropout_enabled, mc_dropout_seed, *mc_seed_parts):
        return model(**kwargs)


def _rollout_prediction_per_hydrograph(
    models: List[Any],
    hydrograph_samples: List[Dict[str, Any]],
    rollout_length: int,
    history_steps: int,
    dynamic_norm: Any,
    target_norm: Any,
    device: torch.device,
    skip_before_timestep: int,
    dt: float,
    out_dir: str,
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
    mc_dropout_enabled: bool = False,
    mc_dropout_seed: Optional[int] = None,
    visualization_config: Optional[Any] = None,
    impact_metrics_config: Optional[Any] = None,
    calibration_coeffs_wd: Optional[Any] = None,
    forecast_artifact_dir: Optional[str] = None,
    calibration_model: Optional[Mapping[str, Any]] = None,
    calibration_metadata: Optional[Mapping[str, Any]] = None,
    write_visualizations: bool = True,
    member_boundary_mode: str = "shared",
    alr_num_particles: Optional[int] = None,
    alr_aleatory_samples: Optional[int] = None,
    forward_timing_path: Optional[str] = None,
) -> None:
    """
    Evaluate per hydrograph using all reference simulations as ground-truth uncertainty.

    Ground-truth uncertainty: variability across reference simulations (Manning's n).
    Prediction uncertainty: variability across stochastic ensemble rollouts.
    """
    if not models:
        raise ValueError("No models provided for rollout evaluation.")
    if mc_dropout_enabled and (gaussian_mode or fgn_noise_dim is not None):
        raise ValueError("MC-dropout rollout is incompatible with Gaussian or FGN rollout modes.")
    if calibration_coeffs_wd is not None and calibration_model is None:
        raise ValueError(
            "Legacy lead-time affine calibration_coeffs_wd is no longer supported in the "
            "maintained rollout path. Use rollout_calibration.method=crps_member_by_member."
        )
    if calibration_model is not None and "wd" not in target_variables:
        raise ValueError("CRPS MBM calibration requires target_variables to include wd.")
    for model in models:
        model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    os.makedirs(out_dir, exist_ok=True)
    impact_cfg = normalize_impact_metrics_config(impact_metrics_config)
    impact_metrics_enabled = impact_cfg.enabled and "wd" in target_variables

    n_models = len(models)
    alr_layout = None
    if alr_num_particles is not None or alr_aleatory_samples is not None:
        if alr_num_particles is None or alr_aleatory_samples is None:
            raise ValueError(
                "ALR rollout requires both alr_num_particles and alr_aleatory_samples."
            )
        if n_models != 1:
            raise ValueError("ALR rollout requires exactly one shared-backbone model.")
        model = models[0]
        if not bool(getattr(model, "anchored_low_rank_enabled", False)):
            raise ValueError("ALR rollout arguments require an ALR-enabled model.")
        alr_layout = ALRMemberLayout(
            int(alr_num_particles), int(alr_aleatory_samples)
        )
        if int(getattr(model, "anchored_low_rank_num_particles", -1)) != alr_layout.num_particles:
            raise ValueError("Configured ALR particle count does not match the checkpoint.")
        n_ens = alr_layout.n_members
    else:
        n_ens = max(1, int(n_ensemble_samples))
    if n_models > 1 and n_ens < n_models:
        logger.warning(
            "n_ensemble_samples=%d is smaller than number of models=%d. "
            "Raising n_ensemble_samples to %d for paper-style equal model usage.",
            n_ens, n_models, n_models,
        )
        n_ens = n_models
    use_ensemble = n_ens > 1 or n_models > 1
    if alr_layout is None:
        member_model_indices = _build_member_model_indices(n_models, n_ens)
        variance_group_count = n_models
    else:
        member_model_indices = alr_layout.member_epistemic_id.tolist()
        variance_group_count = alr_layout.num_particles
    model_counts = [
        member_model_indices.count(i) for i in range(variance_group_count)
    ]
    if mc_dropout_enabled:
        dropout_counts = [enable_mc_dropout_only(model) for model in models]
        if any(count <= 0 for count in dropout_counts):
            raise ValueError(
                "MC dropout requested for rollout, but one or more models contain no "
                "torch.nn.Dropout modules. Check gino.fno_channel_mlp_dropout."
            )
        logger.info(
            "MC-dropout rollout enabled: members=%d seed=%s dropout_modules_per_model=%s "
            "temporal_mode=stepwise",
            n_ens,
            mc_dropout_seed,
            dropout_counts,
        )
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
        fgn_latent_temporal_mode
    )
    fgn_ar_state_update = normalize_fgn_ar_state_update(fgn_ar_state_update)
    rollout_init_mode = normalize_rollout_init_mode(rollout_init_mode)
    member_boundary_mode = _normalize_member_boundary_mode(member_boundary_mode)
    use_reference_member_boundary = member_boundary_mode == "reference_member"
    if alr_layout is not None and use_reference_member_boundary:
        raise ValueError(
            "The ALR pilot uses one shared clean forcing history; "
            "member_boundary_mode='reference_member' is unsupported."
        )
    if use_reference_member_boundary and rollout_init_mode != ROLLOUT_INIT_MEMBER_HISTORY:
        raise ValueError(
            "rollout.member_boundary_mode='reference_member' requires "
            "rollout.init_mode='member_history' so each forecast member uses a "
            "consistent HEC-RAS realization for both initial state and perturbed forcing."
        )
    gaussian_state_update = str(gaussian_state_update).strip().lower()
    if gaussian_mode and gaussian_state_update not in {"sample", "mu"}:
        raise ValueError(
            "rollout.gaussian_state_update must be one of {'sample', 'mu'} "
            f"for gaussian mode, got {gaussian_state_update!r}."
        )
    logger.info(
        "Hydrograph rollout ensemble members=%d across models=%d with per-model counts=%s",
        n_ens, n_models, model_counts,
    )
    logger.info("Hydrograph rollout initialization mode='%s'", rollout_init_mode)
    logger.info("Hydrograph rollout member boundary mode='%s'", member_boundary_mode)
    if gaussian_mode:
        logger.info(
            "Gaussian rollout state update mode='%s' (predictions are still sampled for ensemble metrics).",
            gaussian_state_update,
        )
    if fgn_noise_dim is not None:
        logger.info(
            "FGN rollout latent temporal mode='%s' with fgn_noise_dim=%d and ar_state_update='%s'",
            fgn_latent_temporal_mode,
            fgn_noise_dim,
            fgn_ar_state_update,
        )
    if alr_layout is not None:
        if fgn_latent_temporal_mode != "persistent":
            raise ValueError("ALR rollout requires persistent aleatory latents.")
        if fgn_ar_state_update != "member_feedback":
            raise ValueError("ALR rollout requires member_feedback state updates.")
        logger.info(
            "ALR nested rollout enabled: particles=%d aleatory_per_particle=%d total=%d",
            alr_layout.num_particles,
            alr_layout.aleatory_samples,
            alr_layout.n_members,
        )
    if not use_ensemble:
        logger.warning(
            "Per-hydrograph UQ run without ensemble spread (single model, single member)."
        )
    if impact_cfg.enabled and "wd" not in target_variables:
        logger.warning("Flood-impact CRPS metrics requested but target_variables does not include 'wd'.")
    elif impact_metrics_enabled:
        logger.info(
            "Flood-impact CRPS metrics enabled: threshold=%.3f m, pooled_radii_m=%s",
            impact_cfg.inundation_threshold_m,
            list(impact_cfg.pooled_radii_m),
        )

    requested_rollout_length = int(rollout_length)
    if requested_rollout_length < -1 or requested_rollout_length == 0:
        raise ValueError(
            "rollout_length must be a positive integer or -1 for the full available horizon "
            f"(got {requested_rollout_length})."
        )
    full_horizon_requested = requested_rollout_length == -1
    full_horizon_log_emitted = False
    start_pred_t = skip_before_timestep + history_steps
    end_pred_t = None if full_horizon_requested else start_pred_t + requested_rollout_length

    per_channel_rmse: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_crps: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_gaussian_nll: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_pred: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_gt: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_ratio: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_within_var: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_between_var: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_total_var: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_between_frac: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_between_to_within: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_rmse_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_crps_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_gaussian_nll_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_pred_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_gt_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_ratio_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_within_var_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_between_var_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_total_var_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_between_frac_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_between_to_within_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    wd_prob_brier: List[np.ndarray] = []
    wd_prob_mae: List[np.ndarray] = []
    wd_wasserstein: List[np.ndarray] = []
    wd_prob_brier_full: List[np.ndarray] = []
    wd_prob_mae_full: List[np.ndarray] = []
    wd_wasserstein_full: List[np.ndarray] = []
    wd_rmse_dry_background: List[np.ndarray] = []
    wd_mae_dry_background: List[np.ndarray] = []
    wd_falsewet_rate_001_dry_background: List[np.ndarray] = []
    wd_falsewet_rate_005_dry_background: List[np.ndarray] = []
    wd_pred_std_mean_dry_background: List[np.ndarray] = []
    paired_member_rmse_wd: List[np.ndarray] = []
    impact_metric_values: Dict[str, List[Any]] = {}

    interval_levels = (0.50, 0.80, 0.90, 0.95)
    wd_interval_coverage: Dict[float, List[np.ndarray]] = {a: [] for a in interval_levels}
    wd_interval_width: Dict[float, List[np.ndarray]] = {a: [] for a in interval_levels}

    rel_n_bins = 12
    rel_edges = np.linspace(0.0, 1.0, rel_n_bins + 1, dtype=np.float64)
    rel_count = np.zeros(rel_n_bins, dtype=np.float64)
    rel_sum_pred = np.zeros(rel_n_bins, dtype=np.float64)
    rel_sum_obs = np.zeros(rel_n_bins, dtype=np.float64)
    rel_brier_sum = 0.0
    rel_brier_count = 0

    pit_edges = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    pit_hist_counts = np.zeros(len(pit_edges) - 1, dtype=np.float64)
    rank_hist_counts = np.zeros(n_ens + 1, dtype=np.float64)
    pit_rank_rng = np.random.default_rng(1234567)

    spread_skill_samples: List[np.ndarray] = []
    max_scatter_points_per_step = 250
    w_quantiles = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    structural_mask_active = False
    forward_timing_events: List[Dict[str, Any]] = []

    member_history_mapping_logged = False
    for sample in tqdm(hydrograph_samples, desc="Hydrograph rollout evaluation"):
        hydro_id = sample["hydrograph_id"]
        geometry = sample["geometry"]
        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = geometry.to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)
        full_boundary = sample["boundary"].to(device)
        boundary_member_series = sample.get("boundary_member_series")
        dynamic_ref = sample["dynamic_ref"].to(device)
        n_ref = int(sample["n_ref_sims"])
        dry_mask_np, wettable_mask_np = _structural_masks_from_sample(
            sample,
            expected_cells=int(dynamic_ref.shape[2]),
        )
        structural_mask_active = structural_mask_active or (wettable_mask_np is not None)

        gt_rollout_ref = dynamic_ref[:, start_pred_t:end_pred_t]
        if use_reference_member_boundary:
            if boundary_member_series is None:
                raise ValueError(
                    "rollout.member_boundary_mode='reference_member' requested, but grouped "
                    f"hydrograph sample {hydro_id!r} does not include boundary_member_series."
                )
            boundary_member_series = torch.as_tensor(
                boundary_member_series, dtype=full_boundary.dtype, device=device
            )
            if boundary_member_series.ndim != 3:
                raise ValueError(
                    "Grouped reference-member boundary series must have shape "
                    f"[n_ref, time, channels], got {tuple(boundary_member_series.shape)}."
                )
            if int(boundary_member_series.shape[0]) != n_ref:
                raise ValueError(
                    "Grouped reference-member boundary count mismatch: "
                    f"boundary_member_series has {boundary_member_series.shape[0]} members, "
                    f"dynamic_ref has {n_ref}."
                )
            gt_boundary_rollout = boundary_member_series[:, start_pred_t:end_pred_t]
            sample_rollout_length = min(
                int(gt_rollout_ref.shape[1]),
                int(gt_boundary_rollout.shape[1]),
            )
        else:
            gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]
            sample_rollout_length = min(
                int(gt_rollout_ref.shape[1]),
                int(gt_boundary_rollout.shape[0]),
            )
        if sample_rollout_length < 1:
            raise ValueError(
                "Hydrograph rollout has no forecast steps after skip/history slicing: "
                f"hydrograph_id={hydro_id!r}, skip_before_timestep={skip_before_timestep}, "
                f"history_steps={history_steps}, requested_rollout_length={requested_rollout_length}."
            )
        if not full_horizon_requested and sample_rollout_length < requested_rollout_length:
            raise ValueError(
                "Hydrograph rollout is shorter than requested: "
                f"hydrograph_id={hydro_id!r}, available={sample_rollout_length}, "
                f"requested={requested_rollout_length}."
            )
        if full_horizon_requested and not full_horizon_log_emitted:
            logger.info(
                "Resolved rollout_length=-1 to full available hydrograph horizon=%d steps "
                "after skip_before_timestep=%d and history_steps=%d.",
                sample_rollout_length,
                skip_before_timestep,
                history_steps,
            )
            full_horizon_log_emitted = True

        run_rmse: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_crps: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_gaussian_nll: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_pred: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_gt: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_ratio: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_within_var: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_between_var: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_total_var: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_between_frac: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_between_to_within: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_rmse_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_crps_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_gaussian_nll_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_pred_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_gt_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_ratio_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_within_var_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_between_var_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_total_var_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_between_frac_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_between_to_within_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_pred_mean_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_pred_std_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_mean_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_std_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_relative_l2: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_wd_pred_prob: List[np.ndarray] = []
        run_wd_gt_prob: List[np.ndarray] = []
        run_wd_crps_map: List[np.ndarray] = []
        run_wd_brier: List[float] = []
        run_wd_mae: List[float] = []
        run_wd_wasserstein: List[float] = []
        run_wd_brier_full: List[float] = []
        run_wd_mae_full: List[float] = []
        run_wd_wasserstein_full: List[float] = []
        run_wd_rmse_dry_background: List[float] = []
        run_wd_mae_dry_background: List[float] = []
        run_wd_falsewet_rate_001_dry_background: List[float] = []
        run_wd_falsewet_rate_005_dry_background: List[float] = []
        run_wd_pred_std_mean_dry_background: List[float] = []
        run_paired_member_rmse_wd: List[float] = []
        run_pred_wd_ens: List[np.ndarray] = []
        run_gt_wd_ref: List[np.ndarray] = []
        run_interval_coverage: Dict[float, List[float]] = {a: [] for a in interval_levels}
        run_interval_width: Dict[float, List[float]] = {a: [] for a in interval_levels}
        artifact_pred_wd_steps: List[np.ndarray] = []
        artifact_ref_wd_steps: List[np.ndarray] = []

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
            reference_run_ids = list(sample.get("reference_run_ids", []))
            if reference_run_ids:
                selected_refs = [
                    reference_run_ids[idx] for idx in init_ref_indices[: min(len(init_ref_indices), n_ens if use_ensemble else 1)]
                ]
                logger.info(
                    "Member-history rollout initialization for hydrograph %s uses reference members=%s",
                    hydro_id,
                    selected_refs,
                )
            else:
                logger.info(
                    "Member-history rollout initialization for hydrograph %s uses reference indices=%s",
                    hydro_id,
                    init_ref_indices[: min(len(init_ref_indices), n_ens if use_ensemble else 1)],
                )
            member_history_mapping_logged = True
        if use_ensemble:
            current_dynamics = [hist.clone() for hist in init_histories]
        else:
            current_dynamic = init_histories[0].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()
        current_boundaries = None
        if use_reference_member_boundary:
            n_cells = int(static_0.shape[1])
            current_boundaries = []
            for ref_idx in init_ref_indices[: n_ens if use_ensemble else 1]:
                if ref_idx < 0:
                    raise ValueError(
                        "Internal error: reference-member boundary mode received an unpaired "
                        f"initial-history index for hydrograph {hydro_id!r}."
                    )
                current_boundaries.append(
                    _broadcast_boundary_series_to_cells(
                        boundary_member_series[ref_idx, skip_before_timestep:start_pred_t],
                        n_cells=n_cells,
                        device=device,
                    )
                )
        fgn_latent_bank = None
        if fgn_noise_dim is not None:
            fgn_latent_bank = sample_fgn_rollout_latent_bank(
                num_members=(
                    alr_layout.aleatory_samples
                    if alr_layout is not None
                    else n_ens if use_ensemble else 1
                ),
                batch_size=1,
                latent_dim=fgn_noise_dim,
                device=device,
                dtype=dynamic_ref.dtype,
                temporal_mode=fgn_latent_temporal_mode,
            )

        event_forward_seconds = 0.0
        for t in range(sample_rollout_length):
            mu_stack: Optional[torch.Tensor] = None
            logvar_stack: Optional[torch.Tensor] = None
            state_stack: Optional[torch.Tensor] = None
            if forward_timing_path is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter() if forward_timing_path is not None else None
            with torch.no_grad():
                if alr_layout is not None:
                    nested_histories = torch.stack(current_dynamics, dim=0).reshape(
                        alr_layout.num_particles,
                        alr_layout.aleatory_samples,
                        *current_dynamics[0].shape,
                    )
                    nested_prediction = forward_alr_rollout_step(
                        models[0],
                        histories=nested_histories,
                        static=static_0,
                        boundary=current_boundary,
                        input_geom=geom_0,
                        latent_queries=query_0,
                        output_queries=geom_0,
                        latent_bank=fgn_latent_bank,
                    )
                    pred_stack = nested_prediction.reshape(
                        alr_layout.n_members, 1, *nested_prediction.shape[-2:]
                    )
                elif use_ensemble:
                    pred_members: List[torch.Tensor] = []
                    mu_members: List[torch.Tensor] = []
                    logvar_members: List[torch.Tensor] = []
                    state_members: List[torch.Tensor] = []
                    for ens_idx in range(n_ens):
                        dyn_hist = current_dynamics[ens_idx]
                        model_idx = member_model_indices[ens_idx]
                        model = models[model_idx]
                        dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                        bc_hist = (
                            current_boundaries[ens_idx]
                            if current_boundaries is not None
                            else current_boundary
                        )
                        bc_flat = bc_hist.permute(1, 0, 2).reshape(1, bc_hist.shape[1], -1)
                        x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                        if gaussian_mode:
                            out = _forward_operator_model(
                                model,
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
                                mc_dropout_enabled=mc_dropout_enabled,
                                mc_dropout_seed=mc_dropout_seed,
                                mc_seed_parts=("rollout_hydro", hydro_id, "ensemble", t, ens_idx, model_idx, "gaussian"),
                            )
                            sampled, mu, logvar = _sample_from_packed_gaussian(
                                out,
                                n_channels=int(dynamic_ref.shape[-1]),
                                min_logvar=gaussian_min_logvar,
                                max_logvar=gaussian_max_logvar,
                            )
                            pred_members.append(sampled)
                            mu_members.append(mu)
                            logvar_members.append(logvar)
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
                            pred_members.append(
                                _forward_operator_model(
                                    model,
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    ada_in=z,
                                    mc_dropout_enabled=mc_dropout_enabled,
                                    mc_dropout_seed=mc_dropout_seed,
                                    mc_seed_parts=("rollout_hydro", hydro_id, "ensemble", t, ens_idx, model_idx, "fgn"),
                                )
                            )
                        else:
                            pred_members.append(
                                _forward_operator_model(
                                    model,
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    mc_dropout_enabled=mc_dropout_enabled,
                                    mc_dropout_seed=mc_dropout_seed,
                                    mc_seed_parts=("rollout_hydro", hydro_id, "ensemble", t, ens_idx, model_idx),
                                )
                            )
                    pred_stack = torch.stack(pred_members, dim=0)  # [n_ens, 1, n_cells, n_target]
                    if gaussian_mode:
                        mu_stack = torch.stack(mu_members, dim=0)
                        logvar_stack = torch.stack(logvar_members, dim=0)
                        state_stack = torch.stack(state_members, dim=0)
                else:
                    model = models[0]
                    dyn_flat = current_dynamic.permute(1, 0, 2).reshape(
                        1, current_dynamic.shape[1], -1
                    )
                    bc_hist = current_boundaries[0] if current_boundaries is not None else current_boundary
                    bc_flat = bc_hist.permute(1, 0, 2).reshape(1, bc_hist.shape[1], -1)
                    x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                    if gaussian_mode:
                        out = _forward_operator_model(
                            model,
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            mc_dropout_enabled=mc_dropout_enabled,
                            mc_dropout_seed=mc_dropout_seed,
                            mc_seed_parts=("rollout_hydro", hydro_id, "single", t, 0, 0, "gaussian"),
                        )
                        sampled, mu, logvar = _sample_from_packed_gaussian(
                            out,
                            n_channels=int(dynamic_ref.shape[-1]),
                            min_logvar=gaussian_min_logvar,
                            max_logvar=gaussian_max_logvar,
                        )
                        pred_stack = sampled.unsqueeze(0)
                        mu_stack = mu.unsqueeze(0)
                        logvar_stack = logvar.unsqueeze(0)
                        state_stack = (mu if gaussian_state_update == "mu" else sampled).unsqueeze(0)
                    elif fgn_noise_dim is not None:
                        z = get_fgn_rollout_latent(
                            latent_bank=fgn_latent_bank,
                            member_idx=0,
                            batch_size=x.shape[0],
                            latent_dim=fgn_noise_dim,
                            device=device,
                            dtype=x.dtype,
                        )
                        pred = _forward_operator_model(
                            model,
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                            mc_dropout_enabled=mc_dropout_enabled,
                            mc_dropout_seed=mc_dropout_seed,
                            mc_seed_parts=("rollout_hydro", hydro_id, "single", t, 0, 0, "fgn"),
                        )
                        pred_stack = pred.unsqueeze(0)

                    else:
                        pred = _forward_operator_model(
                            model,
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            mc_dropout_enabled=mc_dropout_enabled,
                            mc_dropout_seed=mc_dropout_seed,
                            mc_seed_parts=("rollout_hydro", hydro_id, "single", t, 0, 0),
                        )
                        pred_stack = pred.unsqueeze(0)

            if forward_started is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                event_forward_seconds += time.perf_counter() - forward_started

            structural_dry_mask = sample.get("structural_dry_mask")
            if alr_layout is not None:
                nested_prediction = pred_stack.reshape(
                    alr_layout.num_particles,
                    alr_layout.aleatory_samples,
                    1,
                    *pred_stack.shape[-2:],
                )
                nested_prediction = clamp_nested_feedback(
                    nested_prediction,
                    structural_dry_mask=structural_dry_mask,
                    target_normalizer=target_norm,
                    water_depth_index=target_variables.index("wd"),
                )
                pred_stack = nested_prediction.reshape_as(pred_stack)
            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))
            inv_pred_ens = apply_structural_dry_zero_mask(
                inv_pred_ens,
                structural_dry_mask=structural_dry_mask,
            )
            inv_gt_ref = dynamic_norm.inverse_transform(gt_rollout_ref[:, t])
            if calibration_model is not None:
                wd_idx_cal = target_variables.index("wd")
                pred_wd_raw = inv_pred_ens[:, :, wd_idx_cal].detach().cpu().numpy()
                pred_wd_cal = apply_crps_mbm_to_wd_members(
                    pred_wd_raw,
                    lead_time_hour=float((t + 1) * dt / 3600.0),
                    calibration_model=calibration_model,
                    wettable_mask=wettable_mask_np,
                )
                inv_pred_ens[:, :, wd_idx_cal] = torch.as_tensor(
                    pred_wd_cal, device=inv_pred_ens.device, dtype=inv_pred_ens.dtype
                )
            if gaussian_mode and mu_stack is not None and logvar_stack is not None:
                mu_phys_stack = target_norm.inverse_transform(mu_stack.squeeze(1))
                mu_phys_stack = apply_structural_dry_zero_mask(
                    mu_phys_stack,
                    structural_dry_mask=structural_dry_mask,
                )
                std_stat = target_norm.std
                while std_stat.ndim > logvar_stack.squeeze(1).ndim and std_stat.shape[0] == 1:
                    std_stat = std_stat.squeeze(0)
                while std_stat.ndim < logvar_stack.squeeze(1).ndim:
                    std_stat = std_stat.unsqueeze(0)
                std_stat = std_stat.to(logvar_stack.device)
                eps = float(getattr(target_norm, "eps", 1e-7))
                logvar_phys_stack = logvar_stack.squeeze(1) + 2.0 * torch.log(std_stat + eps)
            else:
                mu_phys_stack = None
                logvar_phys_stack = None

            pred_mean_field = inv_pred_ens.mean(dim=0)
            pred_std_field = inv_pred_ens.std(dim=0, unbiased=False)
            gt_mean_field = inv_gt_ref.mean(dim=0)
            gt_std_field = inv_gt_ref.std(dim=0, unbiased=False)

            for ch_idx, ch_name in enumerate(target_variables):
                pred_ens_ch = inv_pred_ens[:, :, ch_idx].detach().cpu().numpy()
                gt_ref_ch = inv_gt_ref[:, :, ch_idx].detach().cpu().numpy()
                pred_mean_ch = pred_mean_field[:, ch_idx].detach().cpu().numpy()
                pred_std_ch = pred_std_field[:, ch_idx].detach().cpu().numpy()
                gt_mean_ch = gt_mean_field[:, ch_idx].detach().cpu().numpy()
                gt_std_ch = gt_std_field[:, ch_idx].detach().cpu().numpy()

                run_pred_mean_by_channel[ch_name].append(pred_mean_ch)
                run_pred_std_by_channel[ch_name].append(pred_std_ch)
                run_gt_mean_by_channel[ch_name].append(gt_mean_ch)
                run_gt_std_by_channel[ch_name].append(gt_std_ch)

                rmse_full_t = _masked_rmse(pred_mean_ch, gt_mean_ch, None)
                rmse_t = _masked_rmse(pred_mean_ch, gt_mean_ch, wettable_mask_np)
                rel_l2_t = _masked_relative_l2(pred_mean_ch, gt_mean_ch, wettable_mask_np)
                crps_map = _crps_ensemble_vs_reference(pred_ens_ch, gt_ref_ch)
                crps_full_t = _masked_mean(crps_map, None)
                crps_t = _masked_mean(crps_map, wettable_mask_np)
                if gaussian_mode and mu_phys_stack is not None and logvar_phys_stack is not None:
                    mu_loc = mu_phys_stack[:, :, ch_idx]
                    var_loc = torch.exp(logvar_phys_stack[:, :, ch_idx])
                    mix_mu = mu_loc.mean(dim=0)
                    second = (var_loc + mu_loc.pow(2)).mean(dim=0)
                    mix_var = torch.clamp(second - mix_mu.pow(2), min=MIN_EPS)
                    gt_loc = gt_mean_field[:, ch_idx]
                    nll_loc = 0.5 * (
                        torch.log(mix_var)
                        + (gt_loc - mix_mu).pow(2) / mix_var
                        + torch.log(
                            torch.tensor(2.0 * torch.pi, device=mix_var.device, dtype=mix_var.dtype)
                        )
                    )
                    nll_loc_np = nll_loc.detach().cpu().numpy()
                    gaussian_nll_full_t = _masked_mean(nll_loc_np, None)
                    gaussian_nll_t = _masked_mean(nll_loc_np, wettable_mask_np)
                else:
                    gaussian_nll_full_t = float("nan")
                    gaussian_nll_t = float("nan")
                pred_spread_loc = np.std(pred_ens_ch, axis=0)
                gt_spread_loc = np.std(gt_ref_ch, axis=0)
                spread_pred_full_t = _masked_mean(pred_spread_loc, None)
                spread_pred_t = _masked_mean(pred_spread_loc, wettable_mask_np)
                spread_gt_full_t = _masked_mean(gt_spread_loc, None)
                spread_gt_t = _masked_mean(gt_spread_loc, wettable_mask_np)
                spread_ratio_full_t = spread_pred_full_t / max(spread_gt_full_t, MIN_EPS)
                spread_ratio_t = spread_pred_t / max(spread_gt_t, MIN_EPS)
                within_loc, between_loc, total_loc = _variance_decomposition_by_model(
                    pred_ens_ch, member_model_indices, variance_group_count
                )
                within_full_t = _masked_mean(within_loc, None)
                between_full_t = _masked_mean(between_loc, None)
                total_full_t = _masked_mean(total_loc, None)
                within_t = _masked_mean(within_loc, wettable_mask_np)
                between_t = _masked_mean(between_loc, wettable_mask_np)
                total_t = _masked_mean(total_loc, wettable_mask_np)
                between_frac_full_t = float(
                    np.clip(between_full_t / max(total_full_t, MIN_EPS), 0.0, 1.0)
                )
                between_frac_t = float(np.clip(between_t / max(total_t, MIN_EPS), 0.0, 1.0))
                between_to_within_full_t = float(
                    np.clip(between_full_t / max(within_full_t, MIN_EPS), 0.0, 100.0)
                )
                between_to_within_t = float(
                    np.clip(between_t / max(within_t, MIN_EPS), 0.0, 100.0)
                )

                run_rmse[ch_name].append(rmse_t)
                run_rmse_full[ch_name].append(rmse_full_t)
                run_relative_l2[ch_name].append(rel_l2_t)
                run_crps[ch_name].append(crps_t)
                run_crps_full[ch_name].append(crps_full_t)
                run_gaussian_nll[ch_name].append(gaussian_nll_t)
                run_gaussian_nll_full[ch_name].append(gaussian_nll_full_t)
                run_spread_pred[ch_name].append(spread_pred_t)
                run_spread_pred_full[ch_name].append(spread_pred_full_t)
                run_spread_gt[ch_name].append(spread_gt_t)
                run_spread_gt_full[ch_name].append(spread_gt_full_t)
                run_spread_ratio[ch_name].append(spread_ratio_t)
                run_spread_ratio_full[ch_name].append(spread_ratio_full_t)
                run_within_var[ch_name].append(within_t)
                run_within_var_full[ch_name].append(within_full_t)
                run_between_var[ch_name].append(between_t)
                run_between_var_full[ch_name].append(between_full_t)
                run_total_var[ch_name].append(total_t)
                run_total_var_full[ch_name].append(total_full_t)
                run_between_frac[ch_name].append(between_frac_t)
                run_between_frac_full[ch_name].append(between_frac_full_t)
                run_between_to_within[ch_name].append(between_to_within_t)
                run_between_to_within_full[ch_name].append(between_to_within_full_t)

                if ch_name == "wd":
                    if forecast_artifact_dir is not None:
                        artifact_pred_wd_steps.append(np.asarray(pred_ens_ch, dtype=np.float32))
                        artifact_ref_wd_steps.append(np.asarray(gt_ref_ch, dtype=np.float32))
                    if impact_metrics_enabled:
                        run_pred_wd_ens.append(np.asarray(pred_ens_ch, dtype=np.float64))
                        run_gt_wd_ref.append(np.asarray(gt_ref_ch, dtype=np.float64))
                    pred_prob = np.mean(pred_ens_ch >= UQ_EXCEEDANCE_THRESHOLD, axis=0)
                    gt_prob = np.mean(gt_ref_ch >= UQ_EXCEEDANCE_THRESHOLD, axis=0)
                    run_wd_pred_prob.append(pred_prob)
                    run_wd_gt_prob.append(gt_prob)
                    run_wd_crps_map.append(crps_map)
                    brier_loc = (pred_prob - gt_prob) ** 2
                    mae_loc = np.abs(pred_prob - gt_prob)
                    run_wd_brier.append(_masked_mean(brier_loc, wettable_mask_np))
                    run_wd_brier_full.append(_masked_mean(brier_loc, None))
                    run_wd_mae.append(_masked_mean(mae_loc, wettable_mask_np))
                    run_wd_mae_full.append(_masked_mean(mae_loc, None))
                    run_wd_rmse_dry_background.append(
                        _masked_rmse(pred_mean_ch, gt_mean_ch, dry_mask_np)
                    )
                    run_wd_mae_dry_background.append(
                        _masked_mae(pred_mean_ch, gt_mean_ch, dry_mask_np)
                    )
                    run_wd_falsewet_rate_001_dry_background.append(
                        _dry_falsewet_rate(pred_mean_ch, dry_mask_np, 0.01)
                    )
                    run_wd_falsewet_rate_005_dry_background.append(
                        _dry_falsewet_rate(pred_mean_ch, dry_mask_np, 0.05)
                    )
                    run_wd_pred_std_mean_dry_background.append(
                        _dry_pred_std_mean(pred_std_ch, dry_mask_np)
                    )
                    if use_reference_member_boundary:
                        run_paired_member_rmse_wd.append(
                            _paired_member_rmse(
                                pred_ens_ch,
                                gt_ref_ch,
                                init_ref_indices[: pred_ens_ch.shape[0]],
                                wettable_mask_np,
                            )
                        )

                    # Reliability bins for event probability calibration.
                    pred_prob_primary = _select_cells(pred_prob, wettable_mask_np)
                    gt_prob_primary = _select_cells(gt_prob, wettable_mask_np)
                    bins = np.clip(
                        np.digitize(pred_prob_primary, rel_edges, right=False) - 1,
                        0,
                        rel_n_bins - 1,
                    )
                    for b in range(rel_n_bins):
                        mask_b = bins == b
                        if not np.any(mask_b):
                            continue
                        rel_count[b] += float(np.sum(mask_b))
                        rel_sum_pred[b] += float(np.sum(pred_prob_primary[mask_b]))
                        rel_sum_obs[b] += float(np.sum(gt_prob_primary[mask_b]))
                    rel_brier_sum += float(np.sum((pred_prob_primary - gt_prob_primary) ** 2))
                    rel_brier_count += int(pred_prob_primary.size)

                    # Interval coverage + sharpness.
                    for alpha in interval_levels:
                        q_lo = 0.5 * (1.0 - alpha)
                        q_hi = 1.0 - q_lo
                        lo = np.quantile(pred_ens_ch, q_lo, axis=0)
                        hi = np.quantile(pred_ens_ch, q_hi, axis=0)
                        cover_mask = (gt_ref_ch >= lo[None, :]) & (gt_ref_ch <= hi[None, :])
                        cover = _masked_mean(cover_mask.astype(np.float64), wettable_mask_np)
                        width = _masked_mean(hi - lo, wettable_mask_np)
                        run_interval_coverage[alpha].append(float(cover))
                        run_interval_width[alpha].append(float(width))

                    # Distribution distance (approximate Wasserstein-1 via quantiles).
                    pred_q = np.quantile(pred_ens_ch, w_quantiles, axis=0)
                    gt_q = np.quantile(gt_ref_ch, w_quantiles, axis=0)
                    wdist = np.abs(pred_q - gt_q)
                    run_wd_wasserstein.append(_masked_mean(wdist, wettable_mask_np))
                    run_wd_wasserstein_full.append(_masked_mean(wdist, None))

                    # Proper PIT/rank: use all reference members as pseudo-observations.
                    pit_counts_t, rank_counts_t = _pit_rank_counts_from_reference(
                        pred_ens=_select_cells(pred_ens_ch, wettable_mask_np),
                        ref_ens=_select_cells(gt_ref_ch, wettable_mask_np),
                        pit_edges=pit_edges,
                        n_ens=n_ens,
                        rng=pit_rank_rng,
                    )
                    pit_hist_counts += pit_counts_t
                    rank_hist_counts += rank_counts_t

                    # Spread-skill diagnostic samples (subsampled for plotting efficiency).
                    spread_loc = np.std(pred_ens_ch, axis=0)
                    abs_err_loc = np.abs(pred_mean_ch - gt_mean_ch)
                    spread_loc = _select_cells(spread_loc, wettable_mask_np)
                    abs_err_loc = _select_cells(abs_err_loc, wettable_mask_np)
                    n_loc = spread_loc.size
                    n_take = min(max_scatter_points_per_step, n_loc)
                    if n_take > 0:
                        idx = np.linspace(0, n_loc - 1, n_take, dtype=np.int64)
                        spread_skill_samples.append(
                            np.stack([spread_loc[idx], abs_err_loc[idx]], axis=1).astype(np.float64)
                        )

            if use_ensemble:
                update_stack = state_stack if (gaussian_mode and state_stack is not None) else pred_stack
                update_stack = clamp_structural_dry_normalized_values(
                    update_stack,
                    structural_dry_mask=structural_dry_mask,
                    normalizer=target_norm,
                )
                if fgn_noise_dim is not None and not gaussian_mode:
                    current_dynamics = update_fgn_dynamic_members(
                        dynamic_members=[dyn_hist.unsqueeze(0) for dyn_hist in current_dynamics],
                        pred_samples=update_stack,
                        pred_mean=update_stack.mean(dim=0),
                        n_history=history_steps,
                        state_update_mode=fgn_ar_state_update,
                    )
                    current_dynamics = [dyn_hist.squeeze(0) for dyn_hist in current_dynamics]
                else:
                    for ens_idx in range(n_ens):
                        current_dynamics[ens_idx] = torch.cat(
                            [current_dynamics[ens_idx][1:], update_stack[ens_idx, 0].unsqueeze(0)],
                            dim=0,
                        )
            else:
                update_stack = state_stack if (gaussian_mode and state_stack is not None) else pred_stack
                update_stack = clamp_structural_dry_normalized_values(
                    update_stack,
                    structural_dry_mask=structural_dry_mask,
                    normalizer=target_norm,
                )
                current_dynamic = torch.cat(
                    [current_dynamic[1:], update_stack[0, 0].unsqueeze(0)],
                    dim=0,
                )
            if use_reference_member_boundary:
                assert current_boundaries is not None
                n_cells = int(static_0.shape[1])
                for ens_idx, ref_idx in enumerate(init_ref_indices[: len(current_boundaries)]):
                    next_boundary = _broadcast_boundary_series_to_cells(
                        gt_boundary_rollout[ref_idx, t].unsqueeze(0),
                        n_cells=n_cells,
                        device=device,
                    )
                    current_boundaries[ens_idx] = torch.cat(
                        [current_boundaries[ens_idx][1:], next_boundary],
                        dim=0,
                    )
            else:
                current_boundary = torch.cat(
                    [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
                )

        pred_mean_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_pred_mean_by_channel.items()
        }
        pred_std_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_pred_std_by_channel.items()
        }
        gt_mean_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_gt_mean_by_channel.items()
        }
        gt_std_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_gt_std_by_channel.items()
        }
        pred_prob_wd = np.stack(run_wd_pred_prob, axis=0) if run_wd_pred_prob else None
        gt_prob_wd = np.stack(run_wd_gt_prob, axis=0) if run_wd_gt_prob else None
        crps_map_wd = np.stack(run_wd_crps_map, axis=0) if run_wd_crps_map else None
        relative_l2_by_channel = {
            k: np.asarray(v, dtype=np.float64) for k, v in run_relative_l2.items()
        }

        if forecast_artifact_dir is not None and artifact_pred_wd_steps:
            pred_art = np.stack(artifact_pred_wd_steps, axis=1)
            ref_art = np.stack(artifact_ref_wd_steps, axis=1)
            wettable_art = (
                wettable_mask_np
                if wettable_mask_np is not None
                else np.ones(pred_art.shape[2], dtype=bool)
            )
            dry_art = (
                dry_mask_np
                if dry_mask_np is not None
                else np.logical_not(np.asarray(wettable_art, dtype=bool))
            )
            safe_hydro_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(hydro_id))
            artifact_path = Path(forecast_artifact_dir) / f"{safe_hydro_id}{ARTIFACT_FILE_SUFFIX}"
            save_forecast_artifact(
                artifact_path,
                hydrograph_id=str(hydro_id),
                pred_members_wd=pred_art,
                ref_members_wd=ref_art,
                wettable_mask=wettable_art,
                structural_dry_mask=dry_art,
                boundary_series_raw=sample.get("boundary_series_raw"),
                boundary_ensemble_series_raw=sample.get("boundary_ensemble_series_raw"),
                boundary_channel_names=sample.get("boundary_channel_names"),
                geometry_raw=sample.get("geometry_raw", sample.get("geometry")),
                elevation_raw=sample.get("elevation_raw"),
                member_model_id=[member_model_indices[i] for i in range(n_ens)],
                member_sample_id=list(range(n_ens)),
                member_epistemic_id=(
                    alr_layout.member_epistemic_id.tolist()
                    if alr_layout is not None
                    else None
                ),
                member_aleatory_id=(
                    alr_layout.member_aleatory_id.tolist()
                    if alr_layout is not None
                    else None
                ),
                time_hours=(np.arange(1, pred_art.shape[1] + 1, dtype=np.float64) * float(dt) / 3600.0),
                metadata={
                    "target_variable": "wd",
                    "rollout_start_index": int(start_pred_t),
                    "history_steps": int(history_steps),
                    "reference_run_ids": list(sample.get("reference_run_ids", [])),
                    "skip_before_timestep": int(skip_before_timestep),
                    "n_models": int(n_models),
                    "alr_num_particles": (
                        int(alr_layout.num_particles) if alr_layout is not None else None
                    ),
                    "alr_aleatory_samples": (
                        int(alr_layout.aleatory_samples) if alr_layout is not None else None
                    ),
                    "gaussian_mode": bool(gaussian_mode),
                    "fgn_noise_dim": None if fgn_noise_dim is None else int(fgn_noise_dim),
                    "calibration_applied": calibration_model is not None,
                    "member_boundary_mode": member_boundary_mode,
                    "member_reference_indices": [int(i) for i in init_ref_indices[:n_ens]],
                    **dict(calibration_metadata or {}),
                },
            )
            logger.info("Saved calibration forecast artifact for hydrograph %s to %s", hydro_id, artifact_path)

        if write_visualizations:
            _save_hydrograph_uq_figures_and_animation(
                geometry=sample.get("geometry_raw", geometry),
                pred_mean_by_channel=pred_mean_by_channel,
                pred_std_by_channel=pred_std_by_channel,
                gt_mean_by_channel=gt_mean_by_channel,
                gt_std_by_channel=gt_std_by_channel,
                target_variables=target_variables,
                out_dir=out_dir,
                hydrograph_id=hydro_id,
                dt_seconds=dt,
                n_ref_sims=n_ref,
                n_ens=n_ens,
                pred_prob_wd=pred_prob_wd,
                gt_prob_wd=gt_prob_wd,
                crps_map_wd=crps_map_wd,
                boundary_series_raw=sample.get("boundary_series_raw"),
                boundary_ensemble_series_raw=sample.get("boundary_ensemble_series_raw"),
                boundary_channel_names=sample.get("boundary_channel_names"),
                relative_l2_by_channel=relative_l2_by_channel,
                rollout_start_index=start_pred_t,
                elevation_raw=sample.get("elevation_raw"),
                visualization_config=visualization_config,
            )

        if impact_metrics_enabled and run_pred_wd_ens and run_gt_wd_ref:
            run_impact_metrics = compute_flood_impact_crps_metrics(
                pred_wd_rollout=np.stack(run_pred_wd_ens, axis=0),
                ref_wd_rollout=np.stack(run_gt_wd_ref, axis=0),
                geometry=sample.get("geometry_raw", geometry),
                static_raw=sample.get("static_raw", sample.get("static")),
                wettable_mask=wettable_mask_np,
                config=impact_cfg,
            )
            for key, value in run_impact_metrics.items():
                impact_metric_values.setdefault(key, []).append(value)

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.asarray(run_rmse[ch_name], dtype=np.float64))
            per_channel_rmse_full[ch_name].append(
                np.asarray(run_rmse_full[ch_name], dtype=np.float64)
            )
            per_channel_crps[ch_name].append(np.asarray(run_crps[ch_name], dtype=np.float64))
            per_channel_crps_full[ch_name].append(
                np.asarray(run_crps_full[ch_name], dtype=np.float64)
            )
            per_channel_gaussian_nll[ch_name].append(
                np.asarray(run_gaussian_nll[ch_name], dtype=np.float64)
            )
            per_channel_gaussian_nll_full[ch_name].append(
                np.asarray(run_gaussian_nll_full[ch_name], dtype=np.float64)
            )
            per_channel_spread_pred[ch_name].append(np.asarray(run_spread_pred[ch_name], dtype=np.float64))
            per_channel_spread_pred_full[ch_name].append(
                np.asarray(run_spread_pred_full[ch_name], dtype=np.float64)
            )
            per_channel_spread_gt[ch_name].append(np.asarray(run_spread_gt[ch_name], dtype=np.float64))
            per_channel_spread_gt_full[ch_name].append(
                np.asarray(run_spread_gt_full[ch_name], dtype=np.float64)
            )
            per_channel_spread_ratio[ch_name].append(np.asarray(run_spread_ratio[ch_name], dtype=np.float64))
            per_channel_spread_ratio_full[ch_name].append(
                np.asarray(run_spread_ratio_full[ch_name], dtype=np.float64)
            )
            per_channel_within_var[ch_name].append(np.asarray(run_within_var[ch_name], dtype=np.float64))
            per_channel_within_var_full[ch_name].append(
                np.asarray(run_within_var_full[ch_name], dtype=np.float64)
            )
            per_channel_between_var[ch_name].append(np.asarray(run_between_var[ch_name], dtype=np.float64))
            per_channel_between_var_full[ch_name].append(
                np.asarray(run_between_var_full[ch_name], dtype=np.float64)
            )
            per_channel_total_var[ch_name].append(np.asarray(run_total_var[ch_name], dtype=np.float64))
            per_channel_total_var_full[ch_name].append(
                np.asarray(run_total_var_full[ch_name], dtype=np.float64)
            )
            per_channel_between_frac[ch_name].append(np.asarray(run_between_frac[ch_name], dtype=np.float64))
            per_channel_between_frac_full[ch_name].append(
                np.asarray(run_between_frac_full[ch_name], dtype=np.float64)
            )
            per_channel_between_to_within[ch_name].append(
                np.asarray(run_between_to_within[ch_name], dtype=np.float64)
            )
            per_channel_between_to_within_full[ch_name].append(
                np.asarray(run_between_to_within_full[ch_name], dtype=np.float64)
            )
        if run_wd_brier:
            wd_prob_brier.append(np.asarray(run_wd_brier, dtype=np.float64))
            wd_prob_brier_full.append(np.asarray(run_wd_brier_full, dtype=np.float64))
            wd_prob_mae.append(np.asarray(run_wd_mae, dtype=np.float64))
            wd_prob_mae_full.append(np.asarray(run_wd_mae_full, dtype=np.float64))
            wd_wasserstein.append(np.asarray(run_wd_wasserstein, dtype=np.float64))
            wd_wasserstein_full.append(np.asarray(run_wd_wasserstein_full, dtype=np.float64))
            wd_rmse_dry_background.append(
                np.asarray(run_wd_rmse_dry_background, dtype=np.float64)
            )
            wd_mae_dry_background.append(
                np.asarray(run_wd_mae_dry_background, dtype=np.float64)
            )
            wd_falsewet_rate_001_dry_background.append(
                np.asarray(run_wd_falsewet_rate_001_dry_background, dtype=np.float64)
            )
            wd_falsewet_rate_005_dry_background.append(
                np.asarray(run_wd_falsewet_rate_005_dry_background, dtype=np.float64)
            )
            wd_pred_std_mean_dry_background.append(
                np.asarray(run_wd_pred_std_mean_dry_background, dtype=np.float64)
            )
            if run_paired_member_rmse_wd:
                paired_member_rmse_wd.append(
                    np.asarray(run_paired_member_rmse_wd, dtype=np.float64)
                )
            for alpha in interval_levels:
                wd_interval_coverage[alpha].append(
                    np.asarray(run_interval_coverage[alpha], dtype=np.float64)
                )
                wd_interval_width[alpha].append(
                    np.asarray(run_interval_width[alpha], dtype=np.float64)
                )

        logger.info("Completed hydrograph %s (n_ref=%d, n_ens=%d)", hydro_id, n_ref, n_ens)
        if forward_timing_path is not None:
            forward_timing_events.append(
                {
                    "hydrograph_id": str(hydro_id),
                    "rollout_steps": int(sample_rollout_length),
                    "ensemble_members": int(n_ens),
                    "forward_rollout_seconds": float(event_forward_seconds),
                    "seconds_per_member_rollout": float(
                        event_forward_seconds / max(1, n_ens)
                    ),
                }
            )

    if forward_timing_path is not None:
        timing_path = Path(str(forward_timing_path)).expanduser()
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_payload = {
            "timing_policy": (
                "synchronized autoregressive prediction section only; excludes inverse "
                "transforms, metrics, artifacts, and visualization"
            ),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "alr_num_particles": (
                int(alr_layout.num_particles) if alr_layout is not None else None
            ),
            "alr_aleatory_samples": (
                int(alr_layout.aleatory_samples) if alr_layout is not None else None
            ),
            "events": forward_timing_events,
        }
        temporary_path = timing_path.with_suffix(timing_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(timing_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_path.replace(timing_path)
        logger.info("Forward-only timing written to %s", timing_path)

    if not any(len(v) > 0 for v in per_channel_rmse.values()):
        logger.warning("No per-hydrograph metrics were produced.")
        return

    metrics: Dict[str, np.ndarray] = {}
    for ch_name in target_variables:
        metrics[f"rmse_{ch_name}"] = np.stack(per_channel_rmse[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"rmse_{ch_name}_full_domain"] = np.stack(
                per_channel_rmse_full[ch_name], axis=0
            )
        metrics[f"crps_{ch_name}"] = np.stack(per_channel_crps[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"crps_{ch_name}_full_domain"] = np.stack(
                per_channel_crps_full[ch_name], axis=0
            )
        if gaussian_mode:
            metrics[f"gaussian_nll_{ch_name}"] = np.stack(
                per_channel_gaussian_nll[ch_name], axis=0
            )
            if structural_mask_active:
                metrics[f"gaussian_nll_{ch_name}_full_domain"] = np.stack(
                    per_channel_gaussian_nll_full[ch_name], axis=0
                )
        metrics[f"spread_pred_{ch_name}"] = np.stack(per_channel_spread_pred[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"spread_pred_{ch_name}_full_domain"] = np.stack(
                per_channel_spread_pred_full[ch_name], axis=0
            )
        metrics[f"spread_gt_{ch_name}"] = np.stack(per_channel_spread_gt[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"spread_gt_{ch_name}_full_domain"] = np.stack(
                per_channel_spread_gt_full[ch_name], axis=0
            )
        metrics[f"spread_ratio_{ch_name}"] = np.stack(per_channel_spread_ratio[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"spread_ratio_{ch_name}_full_domain"] = np.stack(
                per_channel_spread_ratio_full[ch_name], axis=0
            )
        metrics[f"within_var_{ch_name}"] = np.stack(per_channel_within_var[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"within_var_{ch_name}_full_domain"] = np.stack(
                per_channel_within_var_full[ch_name], axis=0
            )
        metrics[f"between_var_{ch_name}"] = np.stack(per_channel_between_var[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"between_var_{ch_name}_full_domain"] = np.stack(
                per_channel_between_var_full[ch_name], axis=0
            )
        metrics[f"total_var_{ch_name}"] = np.stack(per_channel_total_var[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"total_var_{ch_name}_full_domain"] = np.stack(
                per_channel_total_var_full[ch_name], axis=0
            )
        metrics[f"between_frac_{ch_name}"] = np.stack(per_channel_between_frac[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"between_frac_{ch_name}_full_domain"] = np.stack(
                per_channel_between_frac_full[ch_name], axis=0
            )
        metrics[f"between_to_within_{ch_name}"] = np.stack(
            per_channel_between_to_within[ch_name], axis=0
        )
        if structural_mask_active:
            metrics[f"between_to_within_{ch_name}_full_domain"] = np.stack(
                per_channel_between_to_within_full[ch_name], axis=0
            )
    if wd_prob_brier:
        metrics["brier_wd_exceed"] = np.stack(wd_prob_brier, axis=0)
        if structural_mask_active:
            metrics["brier_wd_exceed_full_domain"] = np.stack(wd_prob_brier_full, axis=0)
    if wd_prob_mae:
        metrics["prob_mae_wd_exceed"] = np.stack(wd_prob_mae, axis=0)
        if structural_mask_active:
            metrics["prob_mae_wd_exceed_full_domain"] = np.stack(wd_prob_mae_full, axis=0)
    if wd_wasserstein:
        metrics["wasserstein_wd"] = np.stack(wd_wasserstein, axis=0)
        if structural_mask_active:
            metrics["wasserstein_wd_full_domain"] = np.stack(wd_wasserstein_full, axis=0)
    if structural_mask_active and wd_rmse_dry_background:
        metrics["dry_background_rmse_wd"] = np.stack(wd_rmse_dry_background, axis=0)
        metrics["dry_background_mae_wd"] = np.stack(wd_mae_dry_background, axis=0)
        metrics["dry_background_falsewet_rate_001_wd"] = np.stack(
            wd_falsewet_rate_001_dry_background, axis=0
        )
        metrics["dry_background_falsewet_rate_005_wd"] = np.stack(
            wd_falsewet_rate_005_dry_background, axis=0
        )
        metrics["dry_background_pred_std_mean_wd"] = np.stack(
            wd_pred_std_mean_dry_background, axis=0
        )
    if paired_member_rmse_wd:
        metrics["paired_member_rmse_wd"] = np.stack(paired_member_rmse_wd, axis=0)
    for alpha in interval_levels:
        if wd_interval_coverage[alpha]:
            pct = int(round(alpha * 100))
            metrics[f"coverage_wd_{pct}"] = np.stack(wd_interval_coverage[alpha], axis=0)
            metrics[f"width_wd_{pct}"] = np.stack(wd_interval_width[alpha], axis=0)
    for key, values in sorted(impact_metric_values.items()):
        if not values:
            continue
        first = np.asarray(values[0], dtype=np.float64)
        if first.ndim == 0:
            metrics[key] = np.asarray(
                [float(np.asarray(value, dtype=np.float64)) for value in values],
                dtype=np.float64,
            )
        else:
            metrics[key] = np.stack(
                [np.asarray(value, dtype=np.float64) for value in values],
                axis=0,
            )

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_metric_lengths = [int(v.shape[1]) for v in metrics.values() if getattr(v, "ndim", 0) >= 2]
    if not time_metric_lengths:
        raise ValueError("No time-indexed rollout metrics were produced.")
    if len(set(time_metric_lengths)) != 1:
        raise ValueError(
            "Generic rollout metrics have inconsistent horizons: "
            f"{sorted(set(time_metric_lengths))}."
        )
    effective_rollout_length = time_metric_lengths[0]
    time_hours = (np.arange(1, effective_rollout_length + 1) * dt) / 3600.0
    npz_data: Dict[str, Any] = {"time_hours": time_hours}
    for key, stat_dict in stats.items():
        npz_data[f"{key}_mean"] = stat_dict["mean"]
        npz_data[f"{key}_std"] = stat_dict["std"]
        npz_data[f"{key}_all"] = metrics[key]
    data_path = os.path.join(out_dir, ROLLOUT_METRICS_HYDRO_NPZ)
    np.savez(data_path, **npz_data)
    logger.info("Saved per-hydrograph metrics to %s", data_path)

    # Publication-oriented core summary (clear, lower clutter, key UQ metrics only).
    core_specs = [
        ("rmse_wd", "RMSE (WD)", None),
        ("crps_wd", "CRPS (WD)", None),
        ("gaussian_nll_wd", "Gaussian NLL (WD)", None),
        ("wasserstein_wd", "Wasserstein-1 (WD)", None),
        ("brier_wd_exceed", f"Brier: P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", None),
        ("prob_mae_wd_exceed", f"Prob MAE: P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", None),
        ("spread_ratio_wd", "Spread ratio (pred/gt)", 1.0),
        ("between_frac_wd", "Between-model variance fraction", None),
        ("between_to_within_wd", "Between/within variance ratio", 1.0),
    ]
    core_specs = [spec for spec in core_specs if spec[0] in stats]
    if core_specs:
        n_core = len(core_specs)
        n_cols = 2
        n_rows = int(np.ceil(n_core / n_cols))
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(7.4 * n_cols, 4.3 * n_rows), dpi=280, constrained_layout=True
        )
        axs_flat = np.array(axs).reshape(-1)
        for i, (key, title, ref_line) in enumerate(core_specs):
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            ax = axs_flat[i]
            ax.plot(time_hours, mean, linewidth=1.8, color="#1f77b4")
            ax.fill_between(time_hours, mean - std, mean + std, alpha=0.18, color="#1f77b4")
            ax.set_title(title)
            ax.set_xlabel("Lead time (hour)")
            ax.set_ylabel("Value")
            if ref_line is not None:
                ax.axhline(ref_line, color="gray", linestyle="--", alpha=0.8, linewidth=1.0)
            if key == "spread_ratio_wd":
                lo = float(np.nanquantile(metrics[key], 0.02))
                hi = float(np.nanquantile(metrics[key], 0.98))
                m = 0.1 * max(hi - lo, 0.1)
                ax.set_ylim(max(0.0, lo - m), hi + m)
            elif key == "between_frac_wd":
                ax.set_ylim(0.0, 1.0)
            elif key == "between_to_within_wd":
                lo = float(np.nanquantile(metrics[key], 0.02))
                hi = float(np.nanquantile(metrics[key], 0.98))
                m = 0.1 * max(hi - lo, 0.1)
                ax.set_ylim(max(0.0, lo - m), hi + m)
            ax.grid(True, alpha=0.28)
        for j in range(n_core, len(axs_flat)):
            axs_flat[j].set_visible(False)
        summary_path = os.path.join(out_dir, ROLLOUT_SUMMARY_HYDRO_PNG)
        fig.savefig(summary_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved per-hydrograph core summary figure to %s", summary_path)

    # Full diagnostic summary retained for completeness (all metrics).
    plot_keys = list(stats.keys())
    n_plots = len(plot_keys)
    if n_plots > 0:
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(8.2 * n_cols, 3.8 * n_rows), dpi=220, constrained_layout=True
        )
        axs_flat = np.array(axs).reshape(-1)
        for i, key in enumerate(plot_keys):
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            ax = axs_flat[i]
            if np.ndim(mean) == 0:
                ax.bar([0], [float(mean)], yerr=[float(std)], color="#1f77b4", alpha=0.82)
                ax.set_xticks([])
                ax.set_xlabel("Hydrograph aggregate")
            else:
                ax.plot(time_hours, mean, linewidth=1.35, color="#1f77b4")
                ax.fill_between(time_hours, mean - std, mean + std, alpha=0.18, color="#1f77b4")
                ax.set_xlabel("Lead time (hour)")
            ax.set_title(key)
            if key.startswith("spread_ratio"):
                ax.set_ylabel("Spread ratio")
                ax.axhline(1.0, color="gray", linestyle="--", alpha=0.8, linewidth=0.9)
            elif key.startswith("coverage_"):
                ax.set_ylabel("Coverage")
                if key.endswith("_50"):
                    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, linewidth=0.9)
                elif key.endswith("_80"):
                    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.6, linewidth=0.9)
                elif key.endswith("_90"):
                    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.6, linewidth=0.9)
                elif key.endswith("_95"):
                    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.6, linewidth=0.9)
                ax.set_ylim(0.0, 1.0)
            else:
                ax.set_ylabel("Value")
            ax.grid(True, alpha=0.25)
        for j in range(n_plots, len(axs_flat)):
            axs_flat[j].set_visible(False)
        full_summary_path = os.path.join(out_dir, ROLLOUT_SUMMARY_HYDRO_FULL_PNG)
        fig.savefig(full_summary_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved full per-hydrograph summary figure to %s", full_summary_path)

    rel_mean_pred = np.divide(
        rel_sum_pred, rel_count, out=np.zeros_like(rel_sum_pred), where=rel_count > 0
    )
    rel_mean_obs = np.divide(
        rel_sum_obs, rel_count, out=np.zeros_like(rel_sum_obs), where=rel_count > 0
    )
    reliability_bins = {
        "edges": rel_edges,
        "centers": 0.5 * (rel_edges[:-1] + rel_edges[1:]),
        "count": rel_count,
        "mean_pred": rel_mean_pred,
        "mean_obs": rel_mean_obs,
        "brier_overall": rel_brier_sum / max(float(rel_brier_count), 1.0),
    }
    interval_cov_stack = {
        a: np.stack(wd_interval_coverage[a], axis=0)
        for a in interval_levels
        if wd_interval_coverage[a]
    }
    interval_width_stack = {
        a: np.stack(wd_interval_width[a], axis=0)
        for a in interval_levels
        if wd_interval_width[a]
    }
    spread_skill_arr = (
        np.concatenate(spread_skill_samples, axis=0)
        if spread_skill_samples
        else np.empty((0, 2), dtype=np.float64)
    )
    wasserstein_arr = metrics.get("wasserstein_wd", None)
    _save_nonspatial_uq_diagnostics(
        out_dir=out_dir,
        time_hours=time_hours,
        metrics=metrics,
        reliability_bins=reliability_bins,
        pit_hist_counts=pit_hist_counts,
        pit_edges=pit_edges,
        rank_hist_counts=rank_hist_counts,
        spread_skill_samples=spread_skill_arr,
        interval_coverage=interval_cov_stack,
        interval_width=interval_width_stack,
        wasserstein_wd=wasserstein_arr,
        logger=logger,
    )


def _rollout_prediction_generic(
    models: List[Any],
    rollout_dataset: Any,
    rollout_length: int,
    history_steps: int,
    dynamic_norm: Any,
    target_norm: Any,
    device: torch.device,
    skip_before_timestep: int,
    dt: float,
    out_dir: str,
    target_variables: List[str],
    logger: logging.Logger,
    fgn_noise_dim: Optional[int] = None,
    n_ensemble_samples: int = 1,
    fgn_latent_temporal_mode: str = "stepwise",
    fgn_ar_state_update: str = "mean_feedback",
    gaussian_mode: bool = False,
    gaussian_min_logvar: float = -9.0,
    gaussian_max_logvar: float = 4.0,
    gaussian_state_update: str = "sample",
    mc_dropout_enabled: bool = False,
    mc_dropout_seed: Optional[int] = None,
    visualization_config: Optional[Any] = None,
) -> None:
    """
    Generic rollout mode (single reference trajectory per run).

    Ensemble mode follows paper-style AR updates: each member keeps its own state.
    """
    if not models:
        raise ValueError("No models provided for rollout evaluation.")
    if mc_dropout_enabled and (gaussian_mode or fgn_noise_dim is not None):
        raise ValueError("MC-dropout rollout is incompatible with Gaussian or FGN rollout modes.")
    for model in models:
        model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    os.makedirs(out_dir, exist_ok=True)

    n_models = len(models)
    n_ens = max(1, int(n_ensemble_samples))
    if n_models > 1 and n_ens < n_models:
        logger.warning(
            "n_ensemble_samples=%d is smaller than number of models=%d. "
            "Raising n_ensemble_samples to %d for paper-style equal model usage.",
            n_ens, n_models, n_models,
        )
        n_ens = n_models
    use_ensemble = n_ens > 1 or n_models > 1
    member_model_indices = _build_member_model_indices(n_models, n_ens)
    model_counts = [member_model_indices.count(i) for i in range(n_models)]
    if mc_dropout_enabled:
        dropout_counts = [enable_mc_dropout_only(model) for model in models]
        if any(count <= 0 for count in dropout_counts):
            raise ValueError(
                "MC dropout requested for rollout, but one or more models contain no "
                "torch.nn.Dropout modules. Check gino.fno_channel_mlp_dropout."
            )
        logger.info(
            "MC-dropout rollout enabled: members=%d seed=%s dropout_modules_per_model=%s "
            "temporal_mode=stepwise",
            n_ens,
            mc_dropout_seed,
            dropout_counts,
        )
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
        fgn_latent_temporal_mode
    )
    fgn_ar_state_update = normalize_fgn_ar_state_update(fgn_ar_state_update)
    gaussian_state_update = str(gaussian_state_update).strip().lower()
    if gaussian_mode and gaussian_state_update not in {"sample", "mu"}:
        raise ValueError(
            "rollout.gaussian_state_update must be one of {'sample', 'mu'} "
            f"for gaussian mode, got {gaussian_state_update!r}."
        )
    logger.info(
        "Generic rollout ensemble members=%d across models=%d with per-model counts=%s",
        n_ens, n_models, model_counts,
    )
    if gaussian_mode:
        logger.info(
            "Gaussian rollout state update mode='%s' (predictions are still sampled for ensemble metrics).",
            gaussian_state_update,
        )
    if fgn_noise_dim is not None:
        logger.info(
            "FGN rollout latent temporal mode='%s' with fgn_noise_dim=%d and ar_state_update='%s'",
            fgn_latent_temporal_mode,
            fgn_noise_dim,
            fgn_ar_state_update,
        )

    per_channel_rmse: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_gaussian_nll: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_skill: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_rmse_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_gaussian_nll_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_skill_full: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    wd_csi_005: List[np.ndarray] = []
    wd_csi_03: List[np.ndarray] = []
    wd_csi_005_full: List[np.ndarray] = []
    wd_csi_03_full: List[np.ndarray] = []
    wd_rmse_dry_background: List[np.ndarray] = []
    wd_mae_dry_background: List[np.ndarray] = []
    wd_falsewet_rate_001_dry_background: List[np.ndarray] = []
    wd_falsewet_rate_005_dry_background: List[np.ndarray] = []
    wd_pred_std_mean_dry_background: List[np.ndarray] = []
    structural_mask_active = False

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Rollout evaluation")):
        run_id = sample.get("run_id", f"sample_{idx}")
        full_dynamic = sample["dynamic"].to(device)
        full_boundary = sample["boundary"].to(device)
        geometry = sample["geometry"]
        dry_mask_np, wettable_mask_np = _structural_masks_from_sample(
            sample,
            expected_cells=int(full_dynamic.shape[1]),
        )
        structural_mask_active = structural_mask_active or (wettable_mask_np is not None)
        start_pred_t = skip_before_timestep + history_steps
        end_pred_t = start_pred_t + rollout_length
        gt_rollout = full_dynamic[start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        run_rmse: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_gaussian_nll: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_skill: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_rmse_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_gaussian_nll_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_skill_full: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_csi_005: List[float] = []
        run_csi_03: List[float] = []
        run_csi_005_full: List[float] = []
        run_csi_03_full: List[float] = []
        run_wd_rmse_dry_background: List[float] = []
        run_wd_mae_dry_background: List[float] = []
        run_wd_falsewet_rate_001_dry_background: List[float] = []
        run_wd_falsewet_rate_005_dry_background: List[float] = []
        run_wd_pred_std_mean_dry_background: List[float] = []
        run_pred_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}

        if use_ensemble:
            current_dynamics = [
                full_dynamic[skip_before_timestep:start_pred_t].clone() for _ in range(n_ens)
            ]
        else:
            current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()
        fgn_latent_bank = None
        if fgn_noise_dim is not None:
            fgn_latent_bank = sample_fgn_rollout_latent_bank(
                num_members=n_ens if use_ensemble else 1,
                batch_size=1,
                latent_dim=fgn_noise_dim,
                device=device,
                dtype=full_dynamic.dtype,
                temporal_mode=fgn_latent_temporal_mode,
            )

        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = geometry.to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)

        for t in range(rollout_length):
            mu_stack: Optional[torch.Tensor] = None
            logvar_stack: Optional[torch.Tensor] = None
            state_stack: Optional[torch.Tensor] = None
            with torch.no_grad():
                if use_ensemble:
                    pred_members: List[torch.Tensor] = []
                    mu_members: List[torch.Tensor] = []
                    logvar_members: List[torch.Tensor] = []
                    state_members: List[torch.Tensor] = []
                    for ens_idx in range(n_ens):
                        dyn_hist = current_dynamics[ens_idx]
                        model_idx = member_model_indices[ens_idx]
                        model = models[model_idx]
                        dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                        x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                        if gaussian_mode:
                            out = _forward_operator_model(
                                model,
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
                                mc_dropout_enabled=mc_dropout_enabled,
                                mc_dropout_seed=mc_dropout_seed,
                                mc_seed_parts=("rollout_generic", run_id, "ensemble", t, ens_idx, model_idx, "gaussian"),
                            )
                            sampled, mu, logvar = _sample_from_packed_gaussian(
                                out,
                                n_channels=int(full_dynamic.shape[-1]),
                                min_logvar=gaussian_min_logvar,
                                max_logvar=gaussian_max_logvar,
                            )
                            pred_members.append(sampled)
                            mu_members.append(mu)
                            logvar_members.append(logvar)
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
                            pred_members.append(
                                _forward_operator_model(
                                    model,
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    ada_in=z,
                                    mc_dropout_enabled=mc_dropout_enabled,
                                    mc_dropout_seed=mc_dropout_seed,
                                    mc_seed_parts=("rollout_generic", run_id, "ensemble", t, ens_idx, model_idx, "fgn"),
                                )
                            )
                        else:
                            pred_members.append(
                                _forward_operator_model(
                                    model,
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    mc_dropout_enabled=mc_dropout_enabled,
                                    mc_dropout_seed=mc_dropout_seed,
                                    mc_seed_parts=("rollout_generic", run_id, "ensemble", t, ens_idx, model_idx),
                                )
                            )
                    pred_stack = torch.stack(pred_members, dim=0)
                    if gaussian_mode:
                        mu_stack = torch.stack(mu_members, dim=0)
                        logvar_stack = torch.stack(logvar_members, dim=0)
                        state_stack = torch.stack(state_members, dim=0)
                        pred = mu_stack.mean(dim=0)
                    else:
                        pred = pred_stack.mean(dim=0)
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
                        out = _forward_operator_model(
                            model,
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            mc_dropout_enabled=mc_dropout_enabled,
                            mc_dropout_seed=mc_dropout_seed,
                            mc_seed_parts=("rollout_generic", run_id, "single", t, 0, 0, "gaussian"),
                        )
                        sampled_single, pred, logvar_single = _sample_from_packed_gaussian(
                            out,
                            n_channels=int(full_dynamic.shape[-1]),
                            min_logvar=gaussian_min_logvar,
                            max_logvar=gaussian_max_logvar,
                        )
                        pred_stack = sampled_single.unsqueeze(0)
                        mu_stack = pred.unsqueeze(0)
                        logvar_stack = logvar_single.unsqueeze(0)
                        state_stack = (
                            pred if gaussian_state_update == "mu" else sampled_single
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
                        pred = _forward_operator_model(
                            model,
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                            mc_dropout_enabled=mc_dropout_enabled,
                            mc_dropout_seed=mc_dropout_seed,
                            mc_seed_parts=("rollout_generic", run_id, "single", t, 0, 0, "fgn"),
                        )
                        pred_stack = pred.unsqueeze(0)
                    else:
                        pred = _forward_operator_model(
                            model,
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            mc_dropout_enabled=mc_dropout_enabled,
                            mc_dropout_seed=mc_dropout_seed,
                            mc_seed_parts=("rollout_generic", run_id, "single", t, 0, 0),
                        )
                        pred_stack = pred.unsqueeze(0)

            structural_dry_mask = sample.get("structural_dry_mask")
            inv_pred = target_norm.inverse_transform(pred)
            inv_pred = apply_structural_dry_zero_mask(
                inv_pred,
                structural_dry_mask=structural_dry_mask,
            )
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))
            inv_pred_ens = apply_structural_dry_zero_mask(
                inv_pred_ens,
                structural_dry_mask=structural_dry_mask,
            )
            if gaussian_mode and mu_stack is not None and logvar_stack is not None:
                mu_phys_stack = target_norm.inverse_transform(mu_stack.squeeze(1))
                mu_phys_stack = apply_structural_dry_zero_mask(
                    mu_phys_stack,
                    structural_dry_mask=structural_dry_mask,
                )
                std_stat = target_norm.std
                while std_stat.ndim > logvar_stack.squeeze(1).ndim and std_stat.shape[0] == 1:
                    std_stat = std_stat.squeeze(0)
                while std_stat.ndim < logvar_stack.squeeze(1).ndim:
                    std_stat = std_stat.unsqueeze(0)
                std_stat = std_stat.to(logvar_stack.device)
                eps = float(getattr(target_norm, "eps", 1e-7))
                logvar_phys_stack = logvar_stack.squeeze(1) + 2.0 * torch.log(std_stat + eps)
            else:
                mu_phys_stack = None
                logvar_phys_stack = None

            for ch_idx, ch_name in enumerate(target_variables):
                ch_pred = inv_pred[0, :, ch_idx].detach().cpu().numpy()
                ch_gt = inv_gt[0, :, ch_idx].detach().cpu().numpy()
                rmse_full_t = _masked_rmse(ch_pred, ch_gt, None)
                rmse_t = _masked_rmse(ch_pred, ch_gt, wettable_mask_np)
                run_rmse[ch_name].append(rmse_t)
                run_rmse_full[ch_name].append(rmse_full_t)
                if gaussian_mode and mu_phys_stack is not None and logvar_phys_stack is not None:
                    mu_loc = mu_phys_stack[:, :, ch_idx]
                    var_loc = torch.exp(logvar_phys_stack[:, :, ch_idx])
                    mix_mu = mu_loc.mean(dim=0)
                    second = (var_loc + mu_loc.pow(2)).mean(dim=0)
                    mix_var = torch.clamp(second - mix_mu.pow(2), min=MIN_EPS)
                    gt_loc = inv_gt[0, :, ch_idx]
                    nll_loc = 0.5 * (
                        torch.log(mix_var)
                        + (gt_loc - mix_mu).pow(2) / mix_var
                        + torch.log(
                            torch.tensor(2.0 * torch.pi, device=mix_var.device, dtype=mix_var.dtype)
                        )
                    )
                    nll_loc_np = nll_loc.detach().cpu().numpy()
                    run_gaussian_nll[ch_name].append(_masked_mean(nll_loc_np, wettable_mask_np))
                    run_gaussian_nll_full[ch_name].append(_masked_mean(nll_loc_np, None))
                else:
                    run_gaussian_nll[ch_name].append(float("nan"))
                    run_gaussian_nll_full[ch_name].append(float("nan"))
                run_pred_by_channel[ch_name].append(ch_pred)
                run_gt_by_channel[ch_name].append(ch_gt)
                if use_ensemble:
                    ens_ch = inv_pred_ens[:, :, ch_idx].detach().cpu().numpy()
                    spread_loc = np.std(ens_ch, axis=0)
                    spread_full_t = _masked_mean(spread_loc, None)
                    spread_t = _masked_mean(spread_loc, wettable_mask_np)
                    skill_full_t = _masked_rmse(ch_pred, ch_gt, None)
                    skill_t = _masked_rmse(ch_pred, ch_gt, wettable_mask_np)
                    run_spread[ch_name].append(spread_t)
                    run_spread_full[ch_name].append(spread_full_t)
                    run_spread_skill[ch_name].append(
                        spread_t / skill_t if skill_t > MIN_EPS else 0.0
                    )
                    run_spread_skill_full[ch_name].append(
                        spread_full_t / skill_full_t if skill_full_t > MIN_EPS else 0.0
                    )
                if ch_name == "wd":
                    run_csi_005.append(_masked_csi(0.05, ch_pred, ch_gt, wettable_mask_np))
                    run_csi_03.append(_masked_csi(0.3, ch_pred, ch_gt, wettable_mask_np))
                    run_csi_005_full.append(_masked_csi(0.05, ch_pred, ch_gt, None))
                    run_csi_03_full.append(_masked_csi(0.3, ch_pred, ch_gt, None))
                    run_wd_rmse_dry_background.append(
                        _masked_rmse(ch_pred, ch_gt, dry_mask_np)
                    )
                    run_wd_mae_dry_background.append(
                        _masked_mae(ch_pred, ch_gt, dry_mask_np)
                    )
                    run_wd_falsewet_rate_001_dry_background.append(
                        _dry_falsewet_rate(ch_pred, dry_mask_np, 0.01)
                    )
                    run_wd_falsewet_rate_005_dry_background.append(
                        _dry_falsewet_rate(ch_pred, dry_mask_np, 0.05)
                    )
                    if use_ensemble:
                        run_wd_pred_std_mean_dry_background.append(
                            _dry_pred_std_mean(np.std(ens_ch, axis=0), dry_mask_np)
                        )
                    else:
                        run_wd_pred_std_mean_dry_background.append(0.0)

            if use_ensemble:
                update_stack = state_stack if (gaussian_mode and state_stack is not None) else pred_stack
                update_stack = clamp_structural_dry_normalized_values(
                    update_stack,
                    structural_dry_mask=structural_dry_mask,
                    normalizer=target_norm,
                )
                if fgn_noise_dim is not None and not gaussian_mode:
                    current_dynamics = update_fgn_dynamic_members(
                        dynamic_members=[dyn_hist.unsqueeze(0) for dyn_hist in current_dynamics],
                        pred_samples=update_stack,
                        pred_mean=update_stack.mean(dim=0),
                        n_history=history_steps,
                        state_update_mode=fgn_ar_state_update,
                    )
                    current_dynamics = [dyn_hist.squeeze(0) for dyn_hist in current_dynamics]
                else:
                    for ens_idx in range(n_ens):
                        current_dynamics[ens_idx] = torch.cat(
                            [current_dynamics[ens_idx][1:], update_stack[ens_idx, 0].unsqueeze(0)],
                            dim=0,
                        )
            else:
                update_stack = state_stack if (gaussian_mode and state_stack is not None) else pred_stack
                update_stack = clamp_structural_dry_normalized_values(
                    update_stack,
                    structural_dry_mask=structural_dry_mask,
                    normalizer=target_norm,
                )
                current_dynamic = torch.cat(
                    [current_dynamic[1:], update_stack[0, 0].unsqueeze(0)], dim=0
                )
            current_boundary = torch.cat(
                [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
            )

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.array(run_rmse[ch_name], dtype=np.float64))
            per_channel_rmse_full[ch_name].append(
                np.array(run_rmse_full[ch_name], dtype=np.float64)
            )
            if gaussian_mode:
                per_channel_gaussian_nll[ch_name].append(
                    np.array(run_gaussian_nll[ch_name], dtype=np.float64)
                )
                per_channel_gaussian_nll_full[ch_name].append(
                    np.array(run_gaussian_nll_full[ch_name], dtype=np.float64)
                )
            if use_ensemble:
                per_channel_spread[ch_name].append(np.array(run_spread[ch_name], dtype=np.float64))
                per_channel_spread_skill[ch_name].append(np.array(run_spread_skill[ch_name], dtype=np.float64))
                per_channel_spread_full[ch_name].append(
                    np.array(run_spread_full[ch_name], dtype=np.float64)
                )
                per_channel_spread_skill_full[ch_name].append(
                    np.array(run_spread_skill_full[ch_name], dtype=np.float64)
                )
        if "wd" in target_variables:
            wd_csi_005.append(np.array(run_csi_005, dtype=np.float64))
            wd_csi_03.append(np.array(run_csi_03, dtype=np.float64))
            wd_csi_005_full.append(np.array(run_csi_005_full, dtype=np.float64))
            wd_csi_03_full.append(np.array(run_csi_03_full, dtype=np.float64))
            wd_rmse_dry_background.append(
                np.array(run_wd_rmse_dry_background, dtype=np.float64)
            )
            wd_mae_dry_background.append(
                np.array(run_wd_mae_dry_background, dtype=np.float64)
            )
            wd_falsewet_rate_001_dry_background.append(
                np.array(run_wd_falsewet_rate_001_dry_background, dtype=np.float64)
            )
            wd_falsewet_rate_005_dry_background.append(
                np.array(run_wd_falsewet_rate_005_dry_background, dtype=np.float64)
            )
            wd_pred_std_mean_dry_background.append(
                np.array(run_wd_pred_std_mean_dry_background, dtype=np.float64)
            )

        pred_arr = {k: np.stack(v, axis=0) for k, v in run_pred_by_channel.items()}
        gt_arr = {k: np.stack(v, axis=0) for k, v in run_gt_by_channel.items()}
        if all(k in pred_arr for k in LEGACY_3CH):
            generate_publication_maps(
                geometry=geometry,
                wd_gt_array=gt_arr["wd"],
                wd_pred_array=pred_arr["wd"],
                vx_gt_array=gt_arr["vx"],
                vy_gt_array=gt_arr["vy"],
                vx_pred_array=pred_arr["vx"],
                vy_pred_array=pred_arr["vy"],
                steps=list(PUBLICATION_TIMESTEPS),
                out_dir=os.path.join(out_dir, "figures"),
                run_id=run_id,
                filename_prefix="flood",
            )
            create_rollout_animation(
                geometry=sample.get("geometry_raw", geometry),
                wd_gt=gt_arr["wd"],
                wd_pred=pred_arr["wd"],
                vx_gt=gt_arr["vx"],
                vy_gt=gt_arr["vy"],
                vx_pred=pred_arr["vx"],
                vy_pred=pred_arr["vy"],
                run_id=run_id,
                out_dir=out_dir,
                dt_seconds=dt,
            )
        else:
            _save_generic_rollout_visuals(
                geometry=sample.get("geometry_raw", geometry),
                pred_by_channel=pred_arr,
                gt_by_channel=gt_arr,
                target_variables=target_variables,
                out_dir=out_dir,
                run_id=run_id,
                dt_seconds=dt,
                elevation_raw=sample.get("elevation_raw"),
                visualization_config=visualization_config,
            )
        logger.info("Completed rollout run_id=%s", run_id)

    if not any(len(v) > 0 for v in per_channel_rmse.values()):
        logger.warning("No rollout metrics were produced.")
        return

    metrics: Dict[str, np.ndarray] = {}
    for ch_name in target_variables:
        metrics[f"rmse_{ch_name}"] = np.stack(per_channel_rmse[ch_name], axis=0)
        if structural_mask_active:
            metrics[f"rmse_{ch_name}_full_domain"] = np.stack(
                per_channel_rmse_full[ch_name], axis=0
            )
        if gaussian_mode:
            metrics[f"gaussian_nll_{ch_name}"] = np.stack(
                per_channel_gaussian_nll[ch_name], axis=0
            )
            if structural_mask_active:
                metrics[f"gaussian_nll_{ch_name}_full_domain"] = np.stack(
                    per_channel_gaussian_nll_full[ch_name], axis=0
                )
        if use_ensemble:
            metrics[f"spread_{ch_name}"] = np.stack(per_channel_spread[ch_name], axis=0)
            metrics[f"spread_skill_{ch_name}"] = np.stack(per_channel_spread_skill[ch_name], axis=0)
            if structural_mask_active:
                metrics[f"spread_{ch_name}_full_domain"] = np.stack(
                    per_channel_spread_full[ch_name], axis=0
                )
                metrics[f"spread_skill_{ch_name}_full_domain"] = np.stack(
                    per_channel_spread_skill_full[ch_name], axis=0
                )
    if "wd" in target_variables and wd_csi_005 and wd_csi_03:
        metrics["csi_005"] = np.stack(wd_csi_005, axis=0)
        metrics["csi_03"] = np.stack(wd_csi_03, axis=0)
        if structural_mask_active:
            metrics["csi_005_full_domain"] = np.stack(wd_csi_005_full, axis=0)
            metrics["csi_03_full_domain"] = np.stack(wd_csi_03_full, axis=0)
            metrics["dry_background_rmse_wd"] = np.stack(wd_rmse_dry_background, axis=0)
            metrics["dry_background_mae_wd"] = np.stack(wd_mae_dry_background, axis=0)
            metrics["dry_background_falsewet_rate_001_wd"] = np.stack(
                wd_falsewet_rate_001_dry_background, axis=0
            )
            metrics["dry_background_falsewet_rate_005_wd"] = np.stack(
                wd_falsewet_rate_005_dry_background, axis=0
            )
            metrics["dry_background_pred_std_mean_wd"] = np.stack(
                wd_pred_std_mean_dry_background, axis=0
            )

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_metric_lengths = [int(v.shape[1]) for v in metrics.values() if getattr(v, "ndim", 0) >= 2]
    if not time_metric_lengths:
        raise ValueError("No time-indexed rollout metrics were produced.")
    if len(set(time_metric_lengths)) != 1:
        raise ValueError(
            "Per-hydrograph rollout metrics have inconsistent horizons: "
            f"{sorted(set(time_metric_lengths))}."
        )
    effective_rollout_length = time_metric_lengths[0]
    time_hours = (np.arange(1, effective_rollout_length + 1) * dt) / 3600.0
    npz_data: Dict[str, Any] = {"time_hours": time_hours}
    for key, stat_dict in stats.items():
        npz_data[f"{key}_mean"] = stat_dict["mean"]
        npz_data[f"{key}_std"] = stat_dict["std"]
        npz_data[f"{key}_all"] = metrics[key]
    data_path = os.path.join(out_dir, ROLLOUT_METRICS_NPZ)
    np.savez(data_path, **npz_data)
    logger.info("Saved rollout metrics to %s", data_path)

    plot_keys = list(stats.keys())
    n_plots = len(plot_keys)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), tight_layout=True
    )
    axs_flat = np.array(axs).reshape(-1)
    for i, key in enumerate(plot_keys):
        mean, std = stats[key]["mean"], stats[key]["std"]
        ax = axs_flat[i]
        ax.plot(time_hours, mean, marker="o", label=f"{key} mean")
        ax.fill_between(time_hours, mean - std, mean + std, alpha=0.3, label="±1 std")
        ax.set_title(f"{key} over time")
        ax.set_xlabel("Time (hour)")
        if key.startswith("spread_skill"):
            ax.set_ylabel("Spread-skill ratio")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        elif key.startswith("gaussian_nll"):
            ax.set_ylabel("Gaussian NLL")
        else:
            ax.set_ylabel("RMSE" if key.startswith("rmse_") else "CSI")
        ax.grid(True)
        ax.legend()
    for j in range(n_plots, len(axs_flat)):
        axs_flat[j].set_visible(False)
    summary_path = os.path.join(out_dir, ROLLOUT_SUMMARY_PNG)
    plt.savefig(summary_path)
    plt.close(fig)
    logger.info("Saved rollout summary plot to %s", summary_path)


def _make_trainer(
    config: Any, model: Any, data_processor: Any, device: torch.device, logger: Optional[logging.Logger]
) -> Any:
    """Build FGNTrainer or Trainer from config (no training, eval only)."""
    use_fgn = _opt(config, "gino", "use_fgn_noise", False)
    gaussian_mode = _is_gaussian_mode(config)
    use_progress_bar = _opt(config, None, "use_progress_bar", True)
    scheduler_monitor = _opt(config, "opt", "scheduler_monitor", "train_err")
    eval_interval = _opt(config, "wandb", "eval_interval", 1)
    is_logger = logger is not None
    n_epochs = max(1, int(_opt(config, "opt", "n_epochs", 1)))
    deterministic = bool(_opt(config, None, "deterministic", True))
    eval_seed = int(_opt(config, "distributed", "seed", 123))
    if dist.is_available() and dist.is_initialized():
        eval_seed += int(dist.get_rank())

    common = dict(
        model=model,
        n_epochs=n_epochs,
        data_processor=data_processor,
        device=device,
        wandb_log=False,
        verbose=is_logger,
        logger=logger,
        use_progress_bar=use_progress_bar,
        scheduler_monitor=scheduler_monitor,
        eval_interval=eval_interval,
        deterministic_eval=deterministic,
        eval_seed=eval_seed,
    )
    structural_policy = str(
        _opt(config, "structural_dry", "policy", "legacy_full_domain")
    ).strip().lower()
    rel_l2_loss = FloodMaskedRelLpLoss(
        policy=structural_policy,
        base_loss=LpLoss(d=2, p=2),
        reduction="sum",
    )
    if gaussian_mode:
        return GaussianNLLTrainer(
            **common,
            rel_l2_loss_fn=rel_l2_loss,
            ar_finetune_start_epoch=max(
                0, int(_opt(config, "opt", "ar_finetune_start_epoch", 0))
            ),
            ar_rollout_steps=max(1, int(_opt(config, "opt", "ar_rollout_steps", 1))),
            ar_curriculum_epochs_per_step=max(
                0, int(_opt(config, "opt", "ar_curriculum_epochs_per_step", 0))
            ),
            gaussian_min_logvar=_opt_float(config, "opt", "gaussian_min_logvar", -9.0),
            gaussian_max_logvar=_opt_float(config, "opt", "gaussian_max_logvar", 4.0),
        )
    if use_fgn and _opt(config, "opt", "training_loss", "l2") == "crps":
        return FGNTrainer(
            **common,
            fgn_noise_dim=_opt(config, "gino", "fgn_noise_dim", 32),
            crps_n_samples=max(2, int(_opt(config, "opt", "crps_n_samples", 2))),
            rel_l2_loss_fn=rel_l2_loss,
            crps_l2_weight=float(_opt(config, "opt", "crps_l2_weight", 0.0)),
            ar_finetune_start_epoch=max(
                0, int(_opt(config, "opt", "ar_finetune_start_epoch", 0))
            ),
            ar_rollout_steps=max(1, int(_opt(config, "opt", "ar_rollout_steps", 1))),
            ar_curriculum_epochs_per_step=max(
                0, int(_opt(config, "opt", "ar_curriculum_epochs_per_step", 0))
            ),
            use_flood_crps_spatial_weights=bool(
                _opt(config, "opt", "flood_crps_spatial_weights", False)
            ),
            flood_crps_wet_threshold=float(
                _opt(config, "opt", "wet_threshold", 0.01)
            ),
            flood_crps_wet_smooth_scale=float(
                _opt(config, "opt", "wet_smooth_scale", 0.02)
            ),
            flood_crps_dry_weight_alpha=float(
                _opt(config, "opt", "dry_weight_alpha", 0.1)
            ),
            static_normalizer=None,
            use_hazard_proxy_crps=bool(_opt(config, "opt", "hazard_proxy_crps", False)),
            hazard_proxy_crps_weight=float(
                _opt(config, "opt", "hazard_proxy_crps_weight", 0.15)
            ),
            ar_pooled_crps_gamma=float(
                _opt(config, "opt", "ar_pooled_crps_gamma", 1.0)
            ),
            fgn_latent_temporal_mode=normalize_fgn_latent_temporal_mode(
                _opt(config, "gino", "fgn_latent_temporal_mode", "stepwise")
            ),
            fgn_ar_state_update=normalize_fgn_ar_state_update(
                _opt(config, "opt", "fgn_ar_state_update", "mean_feedback")
            ),
        )
    return Trainer(**common)
