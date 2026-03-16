"""Operator rollout helpers used during training and smoke evaluation."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from neuralop.flood.processing.wv_impl import _sample_from_packed_gaussian
from neuralop.flood.train.fgn import get_fgn_rollout_latent, sample_fgn_rollout_latent_bank
from neuralop.flood.utils.runtime_core import normalize_fgn_latent_temporal_mode
from neuralop.flood.visualization.publication import (
    create_rollout_animation,
    generate_publication_maps,
)


def _structural_masks_from_sample(sample, *, expected_cells):
    dry_mask = sample.get("structural_dry_mask")
    if dry_mask is None:
        return None, None
    dry_mask = torch.as_tensor(dry_mask, dtype=torch.bool, device="cpu").reshape(-1).numpy()
    if dry_mask.size != expected_cells:
        raise ValueError(
            f"structural_dry_mask has {dry_mask.size} cells, expected {expected_cells}."
        )
    return dry_mask, ~dry_mask


def _select_cells(arr, mask):
    if mask is None:
        return np.asarray(arr)
    return np.asarray(arr)[..., mask]


def _masked_mean(arr, mask):
    selected = _select_cells(arr, mask)
    if selected.size == 0:
        return 0.0
    return float(np.mean(selected))


def _masked_rmse(pred, gt, mask):
    diff = np.asarray(pred) - np.asarray(gt)
    return float(np.sqrt(_masked_mean(diff ** 2, mask)))


def _masked_mae(pred, gt, mask):
    return _masked_mean(np.abs(np.asarray(pred) - np.asarray(gt)), mask)


def _masked_csi(threshold, pred, gt, mask):
    pred_sel = _select_cells(pred, mask)
    gt_sel = _select_cells(gt, mask)
    event_pred = pred_sel >= threshold
    event_gt = gt_sel >= threshold
    tp = np.sum(event_pred & event_gt)
    fp = np.sum(event_pred & (~event_gt))
    fn = np.sum((~event_pred) & event_gt)
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0


def _dry_falsewet_rate(pred, dry_mask, threshold):
    if dry_mask is None or not np.any(dry_mask):
        return 0.0
    pred_sel = _select_cells(pred, dry_mask)
    if pred_sel.size == 0:
        return 0.0
    return float(np.mean(pred_sel > float(threshold)))


def _dry_pred_std_mean(pred_std, dry_mask):
    if dry_mask is None or not np.any(dry_mask):
        return 0.0
    pred_sel = _select_cells(pred_std, dry_mask)
    if pred_sel.size == 0:
        return 0.0
    return float(np.mean(pred_sel))

def rollout_prediction(
        trainer,
        rollout_dataset,
        rollout_length,
        history_steps,
        dynamic_norm,
        target_norm,
        device,
        skip_before_timestep,
        dt,
        out_dir="./rollout_gifs",
        fgn_noise_dim=None,
        n_ensemble_samples=1,
        fgn_latent_temporal_mode="stepwise",
        gaussian_min_logvar: float = -9.0,
        gaussian_max_logvar: float = 4.0,
):
    """
    Performs autoregressive rollout, computing and plotting metrics for water depth, VX, and VY.
    FGN mode: when fgn_noise_dim is set and n_ensemble_samples > 1, runs an ensemble of
    forwards per step with per-sample noise z shaped [B, D]. In persistent mode each
    member reuses one latent across the full rollout.
    Gaussian mode: samples members from packed [mu, logvar] outputs and propagates sampled
    member states autoregressively.
    """
    model = trainer.model
    model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    output_distribution = str(getattr(model, "output_distribution", "deterministic")).strip().lower()
    use_gaussian = output_distribution == "gaussian"
    n_ens = max(1, int(n_ensemble_samples))
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(fgn_latent_temporal_mode)

    def compute_csi(threshold, pred, gt):
        event_pred = pred >= threshold
        event_gt = gt >= threshold
        TP = np.sum(event_pred & event_gt)
        FP = np.sum(event_pred & (~event_gt))
        FN = np.sum((~event_pred) & event_gt)
        return TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0

    # Containers to aggregate metrics from all rollout samples
    aggregated_rmse, aggregated_csi_005, aggregated_csi_03 = [], [], []
    aggregated_rmse_vx, aggregated_rmse_vy = [], []
    aggregated_rmse_full, aggregated_csi_005_full, aggregated_csi_03_full = [], [], []
    aggregated_rmse_vx_full, aggregated_rmse_vy_full = [], []
    aggregated_spread_wd, aggregated_spread_vx, aggregated_spread_vy = [], [], []
    aggregated_spread_wd_full, aggregated_spread_vx_full, aggregated_spread_vy_full = [], [], []
    aggregated_dry_rmse_wd, aggregated_dry_mae_wd = [], []
    aggregated_dry_falsewet_001_wd, aggregated_dry_falsewet_005_wd = [], []
    aggregated_dry_pred_std_mean_wd = []
    structural_mask_active = False

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Performing rollout evaluation")):
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

        # Containers for per-run data arrays
        wd_pred_list, wd_gt_list, vx_pred_list, vy_pred_list, vx_gt_list, vy_gt_list = [], [], [], [], [], []

        # Containers for per-run metrics
        run_rmse, run_csi_005, run_csi_03 = [], [], []
        run_rmse_vx, run_rmse_vy = [], []
        run_rmse_full, run_csi_005_full, run_csi_03_full = [], [], []
        run_rmse_vx_full, run_rmse_vy_full = [], []
        run_spread_wd, run_spread_vx, run_spread_vy = [], [], []
        run_spread_wd_full, run_spread_vx_full, run_spread_vy_full = [], [], []
        run_dry_rmse_wd, run_dry_mae_wd = [], []
        run_dry_falsewet_001_wd, run_dry_falsewet_005_wd = [], []
        run_dry_pred_std_mean_wd = []

        use_fgn_ensemble = fgn_noise_dim is not None and n_ens > 1
        if (use_gaussian and n_ens > 1) or use_fgn_ensemble:
            current_dynamics = [
                full_dynamic[skip_before_timestep:start_pred_t].clone() for _ in range(n_ens)
            ]
        else:
            current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()
        n_target_channels = int(full_dynamic.shape[-1])
        fgn_latent_bank = None
        if fgn_noise_dim is not None:
            fgn_latent_bank = sample_fgn_rollout_latent_bank(
                num_members=n_ens if use_fgn_ensemble else 1,
                batch_size=1,
                latent_dim=fgn_noise_dim,
                device=device,
                dtype=full_dynamic.dtype,
                temporal_mode=fgn_latent_temporal_mode,
            )

        for t in range(rollout_length):
            with torch.no_grad():
                if use_gaussian:
                    if n_ens > 1:
                        sampled_members = []
                        mean_members = []
                        for ens_idx in range(n_ens):
                            dyn_hist = current_dynamics[ens_idx]
                            dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                            bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                            x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                            out = model(
                                input_geom=geometry.to(device).unsqueeze(0),
                                latent_queries=sample["query_points"].to(device).unsqueeze(0),
                                output_queries=geometry.to(device).unsqueeze(0),
                                x=x,
                            )
                            sampled, mu, _ = _sample_from_packed_gaussian(
                                out,
                                n_channels=n_target_channels,
                                min_logvar=gaussian_min_logvar,
                                max_logvar=gaussian_max_logvar,
                            )
                            sampled_members.append(sampled)
                            mean_members.append(mu)
                        pred_stack = torch.stack(sampled_members, dim=0)  # [E, 1, n_cells, C]
                        pred = torch.stack(mean_members, dim=0).mean(dim=0)
                    else:
                        dyn_flat = current_dynamic.permute(1, 0, 2).reshape(1, current_dynamic.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                        x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                        out = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x,
                        )
                        sampled_single, pred, _ = _sample_from_packed_gaussian(
                            out,
                            n_channels=n_target_channels,
                            min_logvar=gaussian_min_logvar,
                            max_logvar=gaussian_max_logvar,
                        )
                        pred_stack = sampled_single.unsqueeze(0)
                elif use_fgn_ensemble:
                    preds = []
                    for ens_idx in range(n_ens):
                        dyn_hist = current_dynamics[ens_idx]
                        dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                        x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                        z = get_fgn_rollout_latent(
                            latent_bank=fgn_latent_bank,
                            member_idx=ens_idx,
                            batch_size=x.shape[0],
                            latent_dim=fgn_noise_dim,
                            device=device,
                            dtype=x.dtype,
                        )
                        p = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x,
                            ada_in=z,
                        )
                        preds.append(p)
                    pred_stack = torch.stack(preds, dim=0)
                    pred = pred_stack.mean(dim=0)
                else:
                    dyn_flat = current_dynamic.permute(1, 0, 2).reshape(1, current_dynamic.shape[1], -1)
                    bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                    x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                    if fgn_noise_dim is not None:
                        z = get_fgn_rollout_latent(
                            latent_bank=fgn_latent_bank,
                            member_idx=0,
                            batch_size=x.shape[0],
                            latent_dim=fgn_noise_dim,
                            device=device,
                            dtype=x.dtype,
                        )
                        pred = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x,
                            ada_in=z,
                        )
                        pred_stack = pred.unsqueeze(0)
                    else:
                        pred = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x
                        )
                        pred_stack = pred.unsqueeze(0)

            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))

            # Extract all channels and convert to numpy
            wd_pred, vx_pred, vy_pred = [ch.cpu().numpy() for ch in inv_pred[0].T]
            wd_gt, vx_gt, vy_gt = [ch.cpu().numpy() for ch in inv_gt[0].T]
            wd_ens, vx_ens, vy_ens = [ch.cpu().numpy() for ch in inv_pred_ens.permute(2, 0, 1)]

            # Append data to lists
            wd_pred_list.append(wd_pred)
            wd_gt_list.append(wd_gt)
            vx_pred_list.append(vx_pred)
            vx_gt_list.append(vx_gt)
            vy_pred_list.append(vy_pred)
            vy_gt_list.append(vy_gt)

            # --- Compute metrics for the current step ---
            # Water Depth metrics
            run_rmse.append(_masked_rmse(wd_pred, wd_gt, wettable_mask_np))
            run_csi_005.append(_masked_csi(0.05, wd_pred, wd_gt, wettable_mask_np))
            run_csi_03.append(_masked_csi(0.3, wd_pred, wd_gt, wettable_mask_np))
            run_rmse_full.append(_masked_rmse(wd_pred, wd_gt, None))
            run_csi_005_full.append(_masked_csi(0.05, wd_pred, wd_gt, None))
            run_csi_03_full.append(_masked_csi(0.3, wd_pred, wd_gt, None))

            # Velocity metrics
            run_rmse_vx.append(_masked_rmse(vx_pred, vx_gt, wettable_mask_np))
            run_rmse_vy.append(_masked_rmse(vy_pred, vy_gt, wettable_mask_np))
            run_rmse_vx_full.append(_masked_rmse(vx_pred, vx_gt, None))
            run_rmse_vy_full.append(_masked_rmse(vy_pred, vy_gt, None))
            wd_spread = np.std(wd_ens, axis=0)
            vx_spread = np.std(vx_ens, axis=0)
            vy_spread = np.std(vy_ens, axis=0)
            run_spread_wd.append(_masked_mean(wd_spread, wettable_mask_np))
            run_spread_vx.append(_masked_mean(vx_spread, wettable_mask_np))
            run_spread_vy.append(_masked_mean(vy_spread, wettable_mask_np))
            run_spread_wd_full.append(_masked_mean(wd_spread, None))
            run_spread_vx_full.append(_masked_mean(vx_spread, None))
            run_spread_vy_full.append(_masked_mean(vy_spread, None))
            run_dry_rmse_wd.append(_masked_rmse(wd_pred, wd_gt, dry_mask_np))
            run_dry_mae_wd.append(_masked_mae(wd_pred, wd_gt, dry_mask_np))
            run_dry_falsewet_001_wd.append(_dry_falsewet_rate(wd_pred, dry_mask_np, 0.01))
            run_dry_falsewet_005_wd.append(_dry_falsewet_rate(wd_pred, dry_mask_np, 0.05))
            run_dry_pred_std_mean_wd.append(_dry_pred_std_mean(wd_spread, dry_mask_np))

            # Update state for next step
            if use_gaussian:
                if n_ens > 1:
                    for ens_idx in range(n_ens):
                        current_dynamics[ens_idx] = torch.cat(
                            [current_dynamics[ens_idx][1:], pred_stack[ens_idx, 0].unsqueeze(0)], dim=0
                        )
                else:
                    current_dynamic = torch.cat(
                        [current_dynamic[1:], sampled_single.squeeze(0).unsqueeze(0)], dim=0
                    )
            elif use_fgn_ensemble:
                for ens_idx in range(n_ens):
                    current_dynamics[ens_idx] = torch.cat(
                        [current_dynamics[ens_idx][1:], pred_stack[ens_idx, 0].unsqueeze(0)], dim=0
                    )
            else:
                current_dynamic = torch.cat([current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0)
            current_boundary = torch.cat([current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0)

        # Convert lists to numpy arrays
        wd_pred_arr, wd_gt_arr = np.stack(wd_pred_list), np.stack(wd_gt_list)
        vx_pred_arr, vy_pred_arr = np.stack(vx_pred_list), np.stack(vy_pred_list)
        vx_gt_arr, vy_gt_arr = np.stack(vx_gt_list), np.stack(vy_gt_list)

        # Append per-run metrics to aggregated lists
        aggregated_rmse.append(np.array(run_rmse))
        aggregated_csi_005.append(np.array(run_csi_005))
        aggregated_csi_03.append(np.array(run_csi_03))
        aggregated_rmse_vx.append(np.array(run_rmse_vx))
        aggregated_rmse_vy.append(np.array(run_rmse_vy))
        aggregated_rmse_full.append(np.array(run_rmse_full))
        aggregated_csi_005_full.append(np.array(run_csi_005_full))
        aggregated_csi_03_full.append(np.array(run_csi_03_full))
        aggregated_rmse_vx_full.append(np.array(run_rmse_vx_full))
        aggregated_rmse_vy_full.append(np.array(run_rmse_vy_full))
        aggregated_spread_wd.append(np.array(run_spread_wd))
        aggregated_spread_vx.append(np.array(run_spread_vx))
        aggregated_spread_vy.append(np.array(run_spread_vy))
        aggregated_spread_wd_full.append(np.array(run_spread_wd_full))
        aggregated_spread_vx_full.append(np.array(run_spread_vx_full))
        aggregated_spread_vy_full.append(np.array(run_spread_vy_full))
        aggregated_dry_rmse_wd.append(np.array(run_dry_rmse_wd))
        aggregated_dry_mae_wd.append(np.array(run_dry_mae_wd))
        aggregated_dry_falsewet_001_wd.append(np.array(run_dry_falsewet_001_wd))
        aggregated_dry_falsewet_005_wd.append(np.array(run_dry_falsewet_005_wd))
        aggregated_dry_pred_std_mean_wd.append(np.array(run_dry_pred_std_mean_wd))

        generate_publication_maps(
            geometry=geometry,
            wd_gt_array=wd_gt_arr, wd_pred_array=wd_pred_arr,
            vx_gt_array=vx_gt_arr, vy_gt_array=vy_gt_arr,
            vx_pred_array=vx_pred_arr, vy_pred_array=vy_pred_arr,
            steps=[12, 24, 36, 48, 60, 72],
            out_dir=os.path.join(out_dir, "figures"),
            run_id=run_id,
            filename_prefix="flood"
        )

        create_rollout_animation(
            geometry=geometry,
            wd_gt=wd_gt_arr, wd_pred=wd_pred_arr,
            vx_gt=vx_gt_arr, vy_gt=vy_gt_arr,
            vx_pred=vx_pred_arr, vy_pred=vy_pred_arr,
            run_id=run_id, out_dir=out_dir, dt_seconds=dt
        )

    # After all runs, aggregate and plot final metrics
    if aggregated_rmse:
        # Stack and calculate mean/std for all metrics
        metrics = {
            'rmse_wd': np.stack(aggregated_rmse),
            'csi_005': np.stack(aggregated_csi_005),
            'csi_03': np.stack(aggregated_csi_03),
            'rmse_vx': np.stack(aggregated_rmse_vx),
            'rmse_vy': np.stack(aggregated_rmse_vy),
            'spread_wd': np.stack(aggregated_spread_wd),
            'spread_vx': np.stack(aggregated_spread_vx),
            'spread_vy': np.stack(aggregated_spread_vy),
        }
        if structural_mask_active:
            metrics.update({
                'rmse_wd_full_domain': np.stack(aggregated_rmse_full),
                'csi_005_full_domain': np.stack(aggregated_csi_005_full),
                'csi_03_full_domain': np.stack(aggregated_csi_03_full),
                'rmse_vx_full_domain': np.stack(aggregated_rmse_vx_full),
                'rmse_vy_full_domain': np.stack(aggregated_rmse_vy_full),
                'spread_wd_full_domain': np.stack(aggregated_spread_wd_full),
                'spread_vx_full_domain': np.stack(aggregated_spread_vx_full),
                'spread_vy_full_domain': np.stack(aggregated_spread_vy_full),
                'dry_background_rmse_wd': np.stack(aggregated_dry_rmse_wd),
                'dry_background_mae_wd': np.stack(aggregated_dry_mae_wd),
                'dry_background_falsewet_rate_001_wd': np.stack(aggregated_dry_falsewet_001_wd),
                'dry_background_falsewet_rate_005_wd': np.stack(aggregated_dry_falsewet_005_wd),
                'dry_background_pred_std_mean_wd': np.stack(aggregated_dry_pred_std_mean_wd),
            })
        stats = {key: {'mean': arr.mean(axis=0), 'std': arr.std(axis=0)} for key, arr in metrics.items()}

        time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0

        fig, axs = plt.subplots(4, 2, figsize=(16, 24), tight_layout=True)
        axs = axs.flatten()

        plot_info = {
            0: ('rmse_wd', 'RMSE (Depth)', 'RMSE (m)'),
            1: ('rmse_vx', 'RMSE (VX)', 'RMSE (m/s)'),
            2: ('rmse_vy', 'RMSE (VY)', 'RMSE (m/s)'),
            3: ('csi_005', 'CSI (0.05m)', 'CSI'),
            4: ('csi_03', 'CSI (0.3m)', 'CSI'),
            5: ('spread_wd', 'Spread (Depth)', 'Std (m)'),
            6: ('spread_vx', 'Spread (VX)', 'Std (m/s)'),
            7: ('spread_vy', 'Spread (VY)', 'Std (m/s)'),
        }

        for i in range(len(axs)):
            ax = axs[i]
            if i in plot_info:
                key, title, ylabel = plot_info[i]
                mean, std = stats[key]['mean'], stats[key]['std']
                ax.plot(time_hours, mean, label=f'{title} Mean', marker='o')
                ax.fill_between(time_hours, mean - std, mean + std, alpha=0.3, label='±1 Std')
                ax.set_title(title + " over Time")
                ax.set_xlabel("Time (hour)")
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True)
            else:
                ax.set_visible(False)  # Hide unused subplots

        summary_path = os.path.join(out_dir, "rollout_metrics_summary.png")
        plt.savefig(summary_path)
        plt.close(fig)
        logging.getLogger("flood_train").info("Saved aggregated rollout metrics plot to %s", summary_path)

        # Save data for external plotting
        npz_data = {'time_hours': time_hours}
        for key, stat_dict in stats.items():
            npz_data[f'{key}_mean'] = stat_dict['mean']
            npz_data[f'{key}_std'] = stat_dict['std']
            npz_data[f'{key}_all'] = metrics[key]

        data_save_path = os.path.join(out_dir, "rollout_metrics_data.npz")
        np.savez(data_save_path, **npz_data)
        logging.getLogger("flood_train").info("Saved aggregated rollout metrics data to %s", data_save_path)
