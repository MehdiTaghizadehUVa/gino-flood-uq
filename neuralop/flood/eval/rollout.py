"""Rollout evaluation helpers for operator-style flood models."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from neuralop.flood.eval.metrics import (
    _build_member_model_indices,
    _compute_csi,
    _crps_ensemble_vs_reference,
    _sample_from_packed_gaussian,
    _variance_decomposition_by_model,
    _is_gaussian_mode,
)
from neuralop.flood.eval.render import (
    _save_generic_rollout_visuals,
    _save_hydrograph_uq_figures_and_animation,
)
from neuralop.flood.train.operator import (
    FGNTrainer,
    GaussianNLLTrainer,
    get_fgn_rollout_latent,
    sample_fgn_rollout_latent_bank,
    update_fgn_dynamic_members,
)
from neuralop.flood.utils.runtime import (
    normalize_fgn_ar_state_update,
    normalize_fgn_latent_temporal_mode,
)
from neuralop.training.trainer import Trainer

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
) -> None:
    """
    Evaluate per hydrograph using all reference simulations as ground-truth uncertainty.

    Ground-truth uncertainty: variability across reference simulations (Manning's n).
    Prediction uncertainty: variability across stochastic ensemble rollouts.
    """
    if not models:
        raise ValueError("No models provided for rollout evaluation.")
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
        "Hydrograph rollout ensemble members=%d across models=%d with per-model counts=%s",
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
    if not use_ensemble:
        logger.warning(
            "Per-hydrograph UQ run without ensemble spread (single model, single member)."
        )

    start_pred_t = skip_before_timestep + history_steps
    end_pred_t = start_pred_t + rollout_length

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
    wd_prob_brier: List[np.ndarray] = []
    wd_prob_mae: List[np.ndarray] = []
    wd_wasserstein: List[np.ndarray] = []

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

    for sample in tqdm(hydrograph_samples, desc="Hydrograph rollout evaluation"):
        hydro_id = sample["hydrograph_id"]
        geometry = sample["geometry"]
        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = geometry.to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)
        full_boundary = sample["boundary"].to(device)
        dynamic_ref = sample["dynamic_ref"].to(device)
        n_ref = int(sample["n_ref_sims"])

        gt_rollout_ref = dynamic_ref[:, start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

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
        run_pred_mean_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_pred_std_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_mean_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_std_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_wd_pred_prob: List[np.ndarray] = []
        run_wd_gt_prob: List[np.ndarray] = []
        run_wd_crps_map: List[np.ndarray] = []
        run_wd_brier: List[float] = []
        run_wd_mae: List[float] = []
        run_wd_wasserstein: List[float] = []
        run_interval_coverage: Dict[float, List[float]] = {a: [] for a in interval_levels}
        run_interval_width: Dict[float, List[float]] = {a: [] for a in interval_levels}

        # Do not condition on a specific hidden Manning's-n realization:
        # initialize from the mean history across reference simulations.
        init_history = dynamic_ref[:, skip_before_timestep:start_pred_t].mean(dim=0)
        if use_ensemble:
            current_dynamics = [init_history.clone() for _ in range(n_ens)]
        else:
            current_dynamic = init_history.clone()
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
                            out = model(
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
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
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    ada_in=z,
                                )
                            )
                        else:
                            pred_members.append(
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
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
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                        )
                    else:
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                        )
                        pred_stack = pred.unsqueeze(0)

            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))
            inv_gt_ref = dynamic_norm.inverse_transform(gt_rollout_ref[:, t])
            if gaussian_mode and mu_stack is not None and logvar_stack is not None:
                mu_phys_stack = target_norm.inverse_transform(mu_stack.squeeze(1))
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

                rmse_t = float(np.sqrt(np.mean((pred_mean_ch - gt_mean_ch) ** 2)))
                crps_map = _crps_ensemble_vs_reference(pred_ens_ch, gt_ref_ch)
                crps_t = float(np.mean(crps_map))
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
                    gaussian_nll_t = float(torch.mean(nll_loc).item())
                else:
                    gaussian_nll_t = float("nan")
                spread_pred_t = float(np.mean(np.std(pred_ens_ch, axis=0)))
                spread_gt_t = float(np.mean(np.std(gt_ref_ch, axis=0)))
                spread_ratio_t = spread_pred_t / max(spread_gt_t, MIN_EPS)
                within_loc, between_loc, total_loc = _variance_decomposition_by_model(
                    pred_ens_ch, member_model_indices, n_models
                )
                within_t = float(np.mean(within_loc))
                between_t = float(np.mean(between_loc))
                total_t = float(np.mean(total_loc))
                between_frac_t = float(np.clip(between_t / max(total_t, MIN_EPS), 0.0, 1.0))
                between_to_within_t = float(
                    np.clip(between_t / max(within_t, MIN_EPS), 0.0, 100.0)
                )

                run_rmse[ch_name].append(rmse_t)
                run_crps[ch_name].append(crps_t)
                run_gaussian_nll[ch_name].append(gaussian_nll_t)
                run_spread_pred[ch_name].append(spread_pred_t)
                run_spread_gt[ch_name].append(spread_gt_t)
                run_spread_ratio[ch_name].append(spread_ratio_t)
                run_within_var[ch_name].append(within_t)
                run_between_var[ch_name].append(between_t)
                run_total_var[ch_name].append(total_t)
                run_between_frac[ch_name].append(between_frac_t)
                run_between_to_within[ch_name].append(between_to_within_t)

                if ch_name == "wd":
                    pred_prob = np.mean(pred_ens_ch >= UQ_EXCEEDANCE_THRESHOLD, axis=0)
                    gt_prob = np.mean(gt_ref_ch >= UQ_EXCEEDANCE_THRESHOLD, axis=0)
                    run_wd_pred_prob.append(pred_prob)
                    run_wd_gt_prob.append(gt_prob)
                    run_wd_crps_map.append(crps_map)
                    run_wd_brier.append(float(np.mean((pred_prob - gt_prob) ** 2)))
                    run_wd_mae.append(float(np.mean(np.abs(pred_prob - gt_prob))))

                    # Reliability bins for event probability calibration.
                    bins = np.clip(
                        np.digitize(pred_prob, rel_edges, right=False) - 1, 0, rel_n_bins - 1
                    )
                    for b in range(rel_n_bins):
                        mask_b = bins == b
                        if not np.any(mask_b):
                            continue
                        rel_count[b] += float(np.sum(mask_b))
                        rel_sum_pred[b] += float(np.sum(pred_prob[mask_b]))
                        rel_sum_obs[b] += float(np.sum(gt_prob[mask_b]))
                    rel_brier_sum += float(np.sum((pred_prob - gt_prob) ** 2))
                    rel_brier_count += int(pred_prob.size)

                    # Interval coverage + sharpness.
                    for alpha in interval_levels:
                        q_lo = 0.5 * (1.0 - alpha)
                        q_hi = 1.0 - q_lo
                        lo = np.quantile(pred_ens_ch, q_lo, axis=0)
                        hi = np.quantile(pred_ens_ch, q_hi, axis=0)
                        cover = np.mean((gt_ref_ch >= lo[None, :]) & (gt_ref_ch <= hi[None, :]))
                        width = np.mean(hi - lo)
                        run_interval_coverage[alpha].append(float(cover))
                        run_interval_width[alpha].append(float(width))

                    # Distribution distance (approximate Wasserstein-1 via quantiles).
                    pred_q = np.quantile(pred_ens_ch, w_quantiles, axis=0)
                    gt_q = np.quantile(gt_ref_ch, w_quantiles, axis=0)
                    run_wd_wasserstein.append(float(np.mean(np.abs(pred_q - gt_q))))

                    # Proper PIT/rank: use all reference members as pseudo-observations.
                    pit_counts_t, rank_counts_t = _pit_rank_counts_from_reference(
                        pred_ens=pred_ens_ch,
                        ref_ens=gt_ref_ch,
                        pit_edges=pit_edges,
                        n_ens=n_ens,
                        rng=pit_rank_rng,
                    )
                    pit_hist_counts += pit_counts_t
                    rank_hist_counts += rank_counts_t

                    # Spread-skill diagnostic samples (subsampled for plotting efficiency).
                    spread_loc = np.std(pred_ens_ch, axis=0)
                    abs_err_loc = np.abs(pred_mean_ch - gt_mean_ch)
                    n_loc = spread_loc.size
                    n_take = min(max_scatter_points_per_step, n_loc)
                    if n_take > 0:
                        idx = np.linspace(0, n_loc - 1, n_take, dtype=np.int64)
                        spread_skill_samples.append(
                            np.stack([spread_loc[idx], abs_err_loc[idx]], axis=1).astype(np.float64)
                        )

            if use_ensemble:
                update_stack = state_stack if (gaussian_mode and state_stack is not None) else pred_stack
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
                current_dynamic = torch.cat(
                    [current_dynamic[1:], update_stack[0, 0].unsqueeze(0)],
                    dim=0,
                )
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

        _save_hydrograph_uq_figures_and_animation(
            geometry=geometry,
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
        )

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.asarray(run_rmse[ch_name], dtype=np.float64))
            per_channel_crps[ch_name].append(np.asarray(run_crps[ch_name], dtype=np.float64))
            per_channel_gaussian_nll[ch_name].append(
                np.asarray(run_gaussian_nll[ch_name], dtype=np.float64)
            )
            per_channel_spread_pred[ch_name].append(np.asarray(run_spread_pred[ch_name], dtype=np.float64))
            per_channel_spread_gt[ch_name].append(np.asarray(run_spread_gt[ch_name], dtype=np.float64))
            per_channel_spread_ratio[ch_name].append(np.asarray(run_spread_ratio[ch_name], dtype=np.float64))
            per_channel_within_var[ch_name].append(np.asarray(run_within_var[ch_name], dtype=np.float64))
            per_channel_between_var[ch_name].append(np.asarray(run_between_var[ch_name], dtype=np.float64))
            per_channel_total_var[ch_name].append(np.asarray(run_total_var[ch_name], dtype=np.float64))
            per_channel_between_frac[ch_name].append(np.asarray(run_between_frac[ch_name], dtype=np.float64))
            per_channel_between_to_within[ch_name].append(
                np.asarray(run_between_to_within[ch_name], dtype=np.float64)
            )
        if run_wd_brier:
            wd_prob_brier.append(np.asarray(run_wd_brier, dtype=np.float64))
            wd_prob_mae.append(np.asarray(run_wd_mae, dtype=np.float64))
            wd_wasserstein.append(np.asarray(run_wd_wasserstein, dtype=np.float64))
            for alpha in interval_levels:
                wd_interval_coverage[alpha].append(
                    np.asarray(run_interval_coverage[alpha], dtype=np.float64)
                )
                wd_interval_width[alpha].append(
                    np.asarray(run_interval_width[alpha], dtype=np.float64)
                )

        logger.info("Completed hydrograph %s (n_ref=%d, n_ens=%d)", hydro_id, n_ref, n_ens)

    if not any(len(v) > 0 for v in per_channel_rmse.values()):
        logger.warning("No per-hydrograph metrics were produced.")
        return

    metrics: Dict[str, np.ndarray] = {}
    for ch_name in target_variables:
        metrics[f"rmse_{ch_name}"] = np.stack(per_channel_rmse[ch_name], axis=0)
        metrics[f"crps_{ch_name}"] = np.stack(per_channel_crps[ch_name], axis=0)
        if gaussian_mode:
            metrics[f"gaussian_nll_{ch_name}"] = np.stack(
                per_channel_gaussian_nll[ch_name], axis=0
            )
        metrics[f"spread_pred_{ch_name}"] = np.stack(per_channel_spread_pred[ch_name], axis=0)
        metrics[f"spread_gt_{ch_name}"] = np.stack(per_channel_spread_gt[ch_name], axis=0)
        metrics[f"spread_ratio_{ch_name}"] = np.stack(per_channel_spread_ratio[ch_name], axis=0)
        metrics[f"within_var_{ch_name}"] = np.stack(per_channel_within_var[ch_name], axis=0)
        metrics[f"between_var_{ch_name}"] = np.stack(per_channel_between_var[ch_name], axis=0)
        metrics[f"total_var_{ch_name}"] = np.stack(per_channel_total_var[ch_name], axis=0)
        metrics[f"between_frac_{ch_name}"] = np.stack(per_channel_between_frac[ch_name], axis=0)
        metrics[f"between_to_within_{ch_name}"] = np.stack(
            per_channel_between_to_within[ch_name], axis=0
        )
    if wd_prob_brier:
        metrics["brier_wd_exceed"] = np.stack(wd_prob_brier, axis=0)
    if wd_prob_mae:
        metrics["prob_mae_wd_exceed"] = np.stack(wd_prob_mae, axis=0)
    if wd_wasserstein:
        metrics["wasserstein_wd"] = np.stack(wd_wasserstein, axis=0)
    for alpha in interval_levels:
        if wd_interval_coverage[alpha]:
            pct = int(round(alpha * 100))
            metrics[f"coverage_wd_{pct}"] = np.stack(wd_interval_coverage[alpha], axis=0)
            metrics[f"width_wd_{pct}"] = np.stack(wd_interval_width[alpha], axis=0)

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0
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
            ax.plot(time_hours, mean, linewidth=1.35, color="#1f77b4")
            ax.fill_between(time_hours, mean - std, mean + std, alpha=0.18, color="#1f77b4")
            ax.set_title(key)
            ax.set_xlabel("Lead time (hour)")
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
) -> None:
    """
    Generic rollout mode (single reference trajectory per run).

    Ensemble mode follows paper-style AR updates: each member keeps its own state.
    """
    if not models:
        raise ValueError("No models provided for rollout evaluation.")
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
    wd_csi_005: List[np.ndarray] = []
    wd_csi_03: List[np.ndarray] = []

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Rollout evaluation")):
        run_id = sample.get("run_id", f"sample_{idx}")
        full_dynamic = sample["dynamic"].to(device)
        full_boundary = sample["boundary"].to(device)
        geometry = sample["geometry"]
        start_pred_t = skip_before_timestep + history_steps
        end_pred_t = start_pred_t + rollout_length
        gt_rollout = full_dynamic[start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        run_rmse: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_gaussian_nll: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_skill: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_csi_005: List[float] = []
        run_csi_03: List[float] = []
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
                            out = model(
                                input_geom=geom_0,
                                latent_queries=query_0,
                                output_queries=geom_0,
                                x=x,
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
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    ada_in=z,
                                )
                            )
                        else:
                            pred_members.append(
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
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
                        out = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
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
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                        )
                    else:
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                        )
                        pred_stack = pred.unsqueeze(0)

            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))
            if gaussian_mode and mu_stack is not None and logvar_stack is not None:
                mu_phys_stack = target_norm.inverse_transform(mu_stack.squeeze(1))
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
                run_rmse[ch_name].append(float(np.sqrt(np.mean((ch_pred - ch_gt) ** 2))))
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
                    run_gaussian_nll[ch_name].append(float(torch.mean(nll_loc).item()))
                else:
                    run_gaussian_nll[ch_name].append(float("nan"))
                run_pred_by_channel[ch_name].append(ch_pred)
                run_gt_by_channel[ch_name].append(ch_gt)
                if use_ensemble:
                    ens_ch = inv_pred_ens[:, :, ch_idx].detach().cpu().numpy()
                    spread_t = float(np.mean(np.std(ens_ch, axis=0)))
                    skill_t = float(np.sqrt(np.mean((ch_pred - ch_gt) ** 2)))
                    run_spread[ch_name].append(spread_t)
                    run_spread_skill[ch_name].append(
                        spread_t / skill_t if skill_t > MIN_EPS else 0.0
                    )
                if ch_name == "wd":
                    run_csi_005.append(_compute_csi(0.05, ch_pred, ch_gt))
                    run_csi_03.append(_compute_csi(0.3, ch_pred, ch_gt))

            if use_ensemble:
                update_stack = state_stack if (gaussian_mode and state_stack is not None) else pred_stack
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
                if gaussian_mode:
                    update_stack = state_stack if state_stack is not None else pred_stack
                    current_dynamic = torch.cat(
                        [current_dynamic[1:], update_stack[0, 0].unsqueeze(0)], dim=0
                    )
                else:
                    current_dynamic = torch.cat(
                        [current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0
                    )
            current_boundary = torch.cat(
                [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
            )

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.array(run_rmse[ch_name], dtype=np.float64))
            if gaussian_mode:
                per_channel_gaussian_nll[ch_name].append(
                    np.array(run_gaussian_nll[ch_name], dtype=np.float64)
                )
            if use_ensemble:
                per_channel_spread[ch_name].append(np.array(run_spread[ch_name], dtype=np.float64))
                per_channel_spread_skill[ch_name].append(np.array(run_spread_skill[ch_name], dtype=np.float64))
        if "wd" in target_variables:
            wd_csi_005.append(np.array(run_csi_005, dtype=np.float64))
            wd_csi_03.append(np.array(run_csi_03, dtype=np.float64))

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
                geometry=geometry,
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
                geometry=geometry,
                pred_by_channel=pred_arr,
                gt_by_channel=gt_arr,
                target_variables=target_variables,
                out_dir=out_dir,
                run_id=run_id,
                dt_seconds=dt,
            )
        logger.info("Completed rollout run_id=%s", run_id)

    if not any(len(v) > 0 for v in per_channel_rmse.values()):
        logger.warning("No rollout metrics were produced.")
        return

    metrics: Dict[str, np.ndarray] = {}
    for ch_name in target_variables:
        metrics[f"rmse_{ch_name}"] = np.stack(per_channel_rmse[ch_name], axis=0)
        if gaussian_mode:
            metrics[f"gaussian_nll_{ch_name}"] = np.stack(
                per_channel_gaussian_nll[ch_name], axis=0
            )
        if use_ensemble:
            metrics[f"spread_{ch_name}"] = np.stack(per_channel_spread[ch_name], axis=0)
            metrics[f"spread_skill_{ch_name}"] = np.stack(per_channel_spread_skill[ch_name], axis=0)
    if "wd" in target_variables and wd_csi_005 and wd_csi_03:
        metrics["csi_005"] = np.stack(wd_csi_005, axis=0)
        metrics["csi_03"] = np.stack(wd_csi_03, axis=0)

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0
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
    )
    if gaussian_mode:
        return GaussianNLLTrainer(
            **common,
            rel_l2_loss_fn=LpLoss(d=2, p=2),
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
            rel_l2_loss_fn=LpLoss(d=2, p=2),
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
