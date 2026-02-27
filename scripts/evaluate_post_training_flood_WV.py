#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone post-training evaluation for flood GINO WV pipeline.

This script reproduces evaluation logic that happens after training in
train_gino_flood_train_rollout_animation_WV.py, without retraining:

1) One-step test evaluation on the saved checkpoint
2) Optional rollout evaluation/plots (same rollout block as training script)

Example:
  python scripts/evaluate_post_training_flood_WV.py --config_path config/gino_pluvial_flood_config_WV_depth_only.yaml --checkpoint.save_dir C:/Users/jrj6wm/checkpoints_WV_depth_only_FGN
"""

import argparse
import os
import time
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_gino_flood_train_rollout_animation_WV import (  # noqa: E402
    load_config_and_setup,
    setup_logging,
    set_seed,
    write_train_txt_from_data_root,
    make_split_generator,
    FloodDatasetHDF,
    fit_normalizers_streaming,
    NormalizedDatasetOnTheFly,
    FloodGINODataProcessor,
    FGNTrainer,
    Trainer,
    FloodRolloutTestDatasetHDF,
    collect_all_fields,
    NormalizedRolloutTestDataset,
    create_rollout_animation,
    generate_publication_maps,
    parse_target_variables,
)
from neuralop import get_model  # noqa: E402
from neuralop.losses.data_losses import LpLoss  # noqa: E402
from neuralop.losses.probabilistic_losses import CRPSLoss  # noqa: E402
from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers  # noqa: E402
from neuralop.training.training_state import load_training_state  # noqa: E402


class _PhaseTimer:
    """Lightweight phase timer for structured logs."""
    def __init__(self, logger, phase_name):
        self.logger = logger
        self.phase_name = phase_name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.logger.info(">>> %s", self.phase_name)
        return self

    def __exit__(self, exc_type, exc, _tb):
        dt = time.perf_counter() - self._t0
        if exc is None:
            self.logger.info("<<< %s completed in %.2fs", self.phase_name, dt)
        else:
            self.logger.exception("<<< %s failed after %.2fs: %s", self.phase_name, dt, exc)
        return False


def _build_eval_losses(config, use_fgn):
    l2loss = LpLoss(d=2, p=2)
    if use_fgn and getattr(config.opt, "training_loss", "l2") == "crps":
        crps_n_samples = max(2, int(getattr(config.opt, "crps_n_samples", 2)))
        crps_channel_weights = getattr(config.opt, "crps_channel_weights", None)
        return {
            "l2": l2loss,
            "crps": CRPSLoss(n_samples=crps_n_samples, channel_weights=crps_channel_weights, reduction="mean"),
        }
    return {getattr(config.opt, "testing_loss", "l2"): l2loss}


def _save_generic_rollout_visuals(
    geometry,
    pred_by_channel,
    gt_by_channel,
    target_variables,
    out_dir,
    run_id,
    dt_seconds,
):
    """Create generic maps + animation for any target variable selection."""
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    geom_np = geometry.detach().cpu().numpy() if hasattr(geometry, "detach") else np.asarray(geometry)
    x, y = geom_np[:, 0], geom_np[:, 1]
    rid = run_id or "unknown"
    n_steps = next(iter(gt_by_channel.values())).shape[0]

    # Publication maps for selected timesteps
    default_steps = [12, 24, 36, 48, 60, 72]
    steps = [s for s in default_steps if 0 <= s < n_steps]
    if not steps:
        steps = [min(n_steps - 1, 0)]
    for t in steps:
        n_rows = len(target_variables)
        fig, axs = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), dpi=250, constrained_layout=True)
        axs = np.atleast_2d(axs)
        for r, ch in enumerate(target_variables):
            gt_t = gt_by_channel[ch][t]
            pred_t = pred_by_channel[ch][t]
            err_t = np.abs(pred_t - gt_t)
            if ch == "wd":
                vmax = max(float(np.nanmax(gt_t)), float(np.nanmax(pred_t)), 1e-9)
                cmap, vmin = "viridis", 0.0
            else:
                vmax = max(float(np.nanmax(np.abs(gt_t))), float(np.nanmax(np.abs(pred_t))), 1e-9)
                cmap, vmin = "coolwarm", -vmax
            emax = max(float(np.nanmax(err_t)), 1e-9)

            panels = [
                (f"{ch.upper()} Ground Truth", gt_t, cmap, vmin, vmax, ch),
                (f"{ch.upper()} Prediction", pred_t, cmap, vmin, vmax, ch),
                (f"{ch.upper()} Abs Error", err_t, "magma", 0.0, emax, "error"),
            ]
            for c, (title, arr, pcmap, pvmin, pvmax, cblabel) in enumerate(panels):
                ax = axs[r, c]
                sc = ax.scatter(x, y, c=arr, s=6, marker="s", linewidths=0, cmap=pcmap, vmin=pvmin, vmax=pvmax, rasterized=True)
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label(cblabel)
        fig.savefig(os.path.join(fig_dir, f"rollout_{rid}_t{t}.png"), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # Generic animation (rows = channels, cols = GT/Pred)
    n_rows = len(target_variables)
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), constrained_layout=True)
    axs = np.atleast_2d(axs)
    scatters = []
    for r, ch in enumerate(target_variables):
        gt0 = gt_by_channel[ch][0]
        pred0 = pred_by_channel[ch][0]
        if ch == "wd":
            vmax = max(float(np.nanmax(gt_by_channel[ch])), float(np.nanmax(pred_by_channel[ch])), 1e-9)
            cmap, vmin = "viridis", 0.0
        else:
            vmax = max(float(np.nanmax(np.abs(gt_by_channel[ch]))), float(np.nanmax(np.abs(pred_by_channel[ch]))), 1e-9)
            cmap, vmin = "coolwarm", -vmax
        s_gt = axs[r, 0].scatter(x, y, c=gt0, s=12, cmap=cmap, vmin=vmin, vmax=vmax)
        s_pr = axs[r, 1].scatter(x, y, c=pred0, s=12, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[r, 0].set_title(f"{ch.upper()} Ground Truth")
        axs[r, 1].set_title(f"{ch.upper()} Prediction")
        axs[r, 0].axis("off")
        axs[r, 1].axis("off")
        fig.colorbar(s_gt, ax=axs[r, 0], fraction=0.046, pad=0.03)
        fig.colorbar(s_pr, ax=axs[r, 1], fraction=0.046, pad=0.03)
        scatters.append((ch, s_gt, s_pr))

    def _animate(frame_idx):
        time_hours = (frame_idx + 1) * dt_seconds / 3600.0
        fig.suptitle(f"Rollout Comparison (Run: {rid}) - Time: {time_hours:.2f} hrs", fontsize=16)
        artists = []
        for ch, s_gt, s_pr in scatters:
            s_gt.set_array(gt_by_channel[ch][frame_idx])
            s_pr.set_array(pred_by_channel[ch][frame_idx])
            artists.extend([s_gt, s_pr])
        return artists

    ani = animation.FuncAnimation(fig, _animate, frames=n_steps, interval=200, blit=False)
    ani.save(os.path.join(out_dir, f"rollout_{rid}.gif"), writer="pillow", fps=5)
    plt.close(fig)


def _rollout_prediction_generic(
    trainer,
    rollout_dataset,
    rollout_length,
    history_steps,
    dynamic_norm,
    target_norm,
    device,
    skip_before_timestep,
    dt,
    out_dir,
    target_variables,
    logger,
    fgn_noise_dim=None,
    n_ensemble_samples=1,
):
    """
    Generic rollout evaluator supporting any target variable subset/order.
    Saves aggregate metrics plot + npz at out_dir.
    """
    model = trainer.model
    model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    os.makedirs(out_dir, exist_ok=True)

    def compute_csi(threshold, pred, gt):
        event_pred = pred >= threshold
        event_gt = gt >= threshold
        tp = np.sum(event_pred & event_gt)
        fp = np.sum(event_pred & (~event_gt))
        fn = np.sum((~event_pred) & event_gt)
        denom = tp + fp + fn
        return float(tp / denom) if denom > 0 else 1.0

    # Per-channel containers
    per_channel_rmse = {name: [] for name in target_variables}
    wd_csi_005 = []
    wd_csi_03 = []

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Performing rollout evaluation")):
        run_id = sample.get("run_id", f"sample_{idx}")
        full_dynamic = sample["dynamic"].to(device)
        full_boundary = sample["boundary"].to(device)
        geometry = sample["geometry"]

        start_pred_t = skip_before_timestep + history_steps
        end_pred_t = start_pred_t + rollout_length
        gt_rollout = full_dynamic[start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        run_rmse = {name: [] for name in target_variables}
        run_csi_005 = []
        run_csi_03 = []
        run_pred_by_channel = {name: [] for name in target_variables}
        run_gt_by_channel = {name: [] for name in target_variables}

        current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()

        for t in range(rollout_length):
            dyn_flat = current_dynamic.permute(1, 0, 2).reshape(1, current_dynamic.shape[1], -1)
            bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
            x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)

            with torch.no_grad():
                if fgn_noise_dim is not None and n_ensemble_samples > 1:
                    preds = []
                    for _ in range(n_ensemble_samples):
                        z = torch.randn(fgn_noise_dim, device=device, dtype=x.dtype)
                        p = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x,
                            ada_in=z,
                        )
                        preds.append(p)
                    pred = torch.stack(preds, dim=0).mean(dim=0)
                else:
                    pred = model(
                        input_geom=geometry.to(device).unsqueeze(0),
                        latent_queries=sample["query_points"].to(device).unsqueeze(0),
                        output_queries=geometry.to(device).unsqueeze(0),
                        x=x,
                    )

            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))

            for ch_idx, ch_name in enumerate(target_variables):
                ch_pred = inv_pred[0, :, ch_idx].detach().cpu().numpy()
                ch_gt = inv_gt[0, :, ch_idx].detach().cpu().numpy()
                rmse_val = float(np.sqrt(np.mean((ch_pred - ch_gt) ** 2)))
                run_rmse[ch_name].append(rmse_val)
                run_pred_by_channel[ch_name].append(ch_pred)
                run_gt_by_channel[ch_name].append(ch_gt)

                if ch_name == "wd":
                    run_csi_005.append(compute_csi(0.05, ch_pred, ch_gt))
                    run_csi_03.append(compute_csi(0.3, ch_pred, ch_gt))

            # Autoregressive state update in normalized space
            current_dynamic = torch.cat([current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0)
            current_boundary = torch.cat([current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0)

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.array(run_rmse[ch_name], dtype=np.float64))
        if "wd" in target_variables:
            wd_csi_005.append(np.array(run_csi_005, dtype=np.float64))
            wd_csi_03.append(np.array(run_csi_03, dtype=np.float64))

        # Visual outputs per run
        pred_arr = {k: np.stack(v, axis=0) for k, v in run_pred_by_channel.items()}
        gt_arr = {k: np.stack(v, axis=0) for k, v in run_gt_by_channel.items()}
        if all(k in pred_arr for k in ("wd", "vx", "vy")):
            # Preserve legacy high-quality 3-channel visual style when available.
            generate_publication_maps(
                geometry=geometry,
                wd_gt_array=gt_arr["wd"],
                wd_pred_array=pred_arr["wd"],
                vx_gt_array=gt_arr["vx"],
                vy_gt_array=gt_arr["vy"],
                vx_pred_array=pred_arr["vx"],
                vy_pred_array=pred_arr["vy"],
                steps=[12, 24, 36, 48, 60, 72],
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

    metrics = {}
    for ch_name in target_variables:
        stacked = np.stack(per_channel_rmse[ch_name], axis=0)
        metrics[f"rmse_{ch_name}"] = stacked
    if "wd" in target_variables and wd_csi_005 and wd_csi_03:
        metrics["csi_005"] = np.stack(wd_csi_005, axis=0)
        metrics["csi_03"] = np.stack(wd_csi_03, axis=0)

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0

    # Save NPZ
    npz_data = {"time_hours": time_hours}
    for key, stat_dict in stats.items():
        npz_data[f"{key}_mean"] = stat_dict["mean"]
        npz_data[f"{key}_std"] = stat_dict["std"]
        npz_data[f"{key}_all"] = metrics[key]
    data_save_path = os.path.join(out_dir, "rollout_metrics_data.npz")
    np.savez(data_save_path, **npz_data)
    logger.info("Saved aggregated rollout metrics data to %s", data_save_path)

    # Plot summary
    plot_keys = list(stats.keys())
    n_plots = len(plot_keys)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), tight_layout=True)
    axs = np.array(axs).reshape(-1)
    for i, key in enumerate(plot_keys):
        mean = stats[key]["mean"]
        std = stats[key]["std"]
        ax = axs[i]
        ax.plot(time_hours, mean, marker="o", label=f"{key} mean")
        ax.fill_between(time_hours, mean - std, mean + std, alpha=0.3, label="±1 std")
        ax.set_title(f"{key} over time")
        ax.set_xlabel("Time (hour)")
        ax.set_ylabel("RMSE" if key.startswith("rmse_") else "CSI")
        ax.grid(True)
        ax.legend()
    for j in range(n_plots, len(axs)):
        axs[j].set_visible(False)

    summary_path = os.path.join(out_dir, "rollout_metrics_summary.png")
    plt.savefig(summary_path)
    plt.close(fig)
    logger.info("Saved aggregated rollout metrics plot to %s", summary_path)


def _make_trainer(config, model, data_processor, device, logger):
    use_fgn = getattr(config.gino, "use_fgn_noise", False)
    use_progress_bar = getattr(config, "use_progress_bar", True)
    scheduler_monitor = getattr(config.opt, "scheduler_monitor", "train_err")
    eval_interval = getattr(config.wandb, "eval_interval", 1)
    is_logger = logger is not None

    if use_fgn and getattr(config.opt, "training_loss", "l2") == "crps":
        trainer = FGNTrainer(
            model=model,
            n_epochs=max(1, int(getattr(config.opt, "n_epochs", 1))),
            data_processor=data_processor,
            device=device,
            wandb_log=False,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
            fgn_noise_dim=getattr(config.gino, "fgn_noise_dim", 32),
            crps_n_samples=max(2, int(getattr(config.opt, "crps_n_samples", 2))),
            rel_l2_loss_fn=LpLoss(d=2, p=2),
            crps_l2_weight=float(getattr(config.opt, "crps_l2_weight", 0.0)),
            ar_finetune_start_epoch=max(0, int(getattr(config.opt, "ar_finetune_start_epoch", 0))),
            ar_rollout_steps=max(1, int(getattr(config.opt, "ar_rollout_steps", 1))),
            ar_curriculum_epochs_per_step=max(0, int(getattr(config.opt, "ar_curriculum_epochs_per_step", 0))),
            use_flood_crps_spatial_weights=bool(getattr(config.opt, "flood_crps_spatial_weights", False)),
            flood_crps_wet_threshold=float(getattr(config.opt, "wet_threshold", 0.01)),
            flood_crps_wet_smooth_scale=float(getattr(config.opt, "wet_smooth_scale", 0.02)),
            flood_crps_dry_weight_alpha=float(getattr(config.opt, "dry_weight_alpha", 0.1)),
            static_normalizer=None,
            use_hazard_proxy_crps=bool(getattr(config.opt, "hazard_proxy_crps", False)),
            hazard_proxy_crps_weight=float(getattr(config.opt, "hazard_proxy_crps_weight", 0.15)),
            ar_pooled_crps_gamma=float(getattr(config.opt, "ar_pooled_crps_gamma", 1.0)),
        )
    else:
        trainer = Trainer(
            model=model,
            n_epochs=max(1, int(getattr(config.opt, "n_epochs", 1))),
            data_processor=data_processor,
            device=device,
            wandb_log=False,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
        )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Post-training evaluation for flood WV script.")
    parser.add_argument("--eval_log_file", type=str, default="eval_post_training.log",
                        help="Evaluation log file path (absolute, or relative to checkpoint dir).")
    parser.add_argument("--run_single_step", action="store_true",
                        help="Run one-step test-set evaluation (default: false).")
    parser.add_argument("--skip_single_step", action="store_true",
                        help="Skip one-step test-set evaluation.")
    parser.add_argument("--run_rollout", action="store_true",
                        help="Force running rollout evaluation regardless of config.rollout.run_after_training.")
    parser.add_argument("--skip_rollout", action="store_true",
                        help="Skip rollout evaluation.")
    args, remaining_argv = parser.parse_known_args()
    # Keep only non-local args for ConfigPipeline (ArgparseConfig) to parse.
    # Without this, local flags like --skip_rollout are rejected as unknown.
    sys.argv = [sys.argv[0]] + remaining_argv

    config, device, is_logger = load_config_and_setup()
    if isinstance(device, str) and "cuda" in device and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    seed = getattr(config.distributed, "seed", 123)
    deterministic = getattr(config, "deterministic", True)
    set_seed(seed, deterministic=deterministic)

    save_dir = Path(getattr(config.checkpoint, "save_dir", "."))
    if not save_dir.is_absolute():
        # Match training behavior: relative paths resolve from current working directory.
        save_dir = save_dir.resolve()
    if not save_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

    eval_log_file = Path(args.eval_log_file)
    if not eval_log_file.is_absolute():
        eval_log_file = save_dir / eval_log_file
    logger = setup_logging(
        log_level=getattr(config, "log_level", "INFO"),
        log_file=str(eval_log_file),
        logger_name="flood_eval",
    )
    if is_logger:
        logger.info("Post-training evaluation started")
        logger.info("Device=%s | Seed=%s | Deterministic=%s", device, seed, deterministic)
        logger.info("Checkpoint directory: %s", save_dir)
        logger.info("Config data.root: %s", getattr(config.data, "root", "N/A"))

    if (save_dir / "best_model_state_dict.pt").exists():
        save_name = "best_model"
    elif (save_dir / "model_state_dict.pt").exists():
        save_name = "model"
    else:
        children = [p.name for p in save_dir.iterdir()]
        raise FileNotFoundError(
            f"No model checkpoint found in: {save_dir}. "
            f"Expected best_model_state_dict.pt or model_state_dict.pt. "
            f"Found: {children}"
        )
    logger.info("Selected checkpoint alias: %s", save_name)

    if getattr(config.data, "write_train_txt", False):
        with _PhaseTimer(logger, "Refreshing train.txt from available HDF runs"):
            run_ids = write_train_txt_from_data_root(
                config.data.root,
                train_txt=getattr(config.data, "train_txt", "train.txt"),
                hdf_suffix=".hdf",
            )
        logger.info("train.txt refreshed with %d run IDs", len(run_ids))

    static_text_files = getattr(config.data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"])
    ar_rollout_steps = max(1, int(getattr(config.opt, "ar_rollout_steps", 1)))
    n_history = config.data.n_history
    target_variables = parse_target_variables(getattr(config.data, "target_variables", ["wd", "vx", "vy"]))
    n_target_channels = len(target_variables)
    n_static = 2 + len(static_text_files)
    data_channels = n_static + n_history * 1 + n_history * n_target_channels
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target_channels)

    with _PhaseTimer(logger, "Building one-step dataset"):
        full_dataset = FloodDatasetHDF(
            data_root=config.data.root,
            n_history=config.data.n_history,
            query_res=getattr(config.data, "query_res", [48, 48]),
            run_ids=None,
            train_txt=getattr(config.data, "train_txt", "train.txt"),
            static_text_files=static_text_files,
            hdf_suffix=".hdf",
            raise_on_smaller=True,
            skip_before_timestep=getattr(config.data, "skip_before_timestep", 0),
            noise_type=getattr(config.data, "noise_type", "none"),
            noise_std=getattr(config.data, "noise_std", None),
            ar_rollout_steps=ar_rollout_steps,
            target_variables=target_variables,
        )

    n_samples_max = getattr(config.data, "n_samples_max", None)
    if n_samples_max is not None and int(n_samples_max) > 0:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))

    total_len = len(full_dataset)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, test_data_raw = random_split(
        full_dataset, [train_sz, test_sz], generator=make_split_generator(seed)
    )
    logger.info(
        "Dataset split: total=%d train=%d test=%d target_variables=%s",
        total_len, len(train_data_raw), len(test_data_raw), target_variables
    )

    normalizer_path = getattr(config.data, "normalizer_path", None)
    if normalizer_path is not None:
        normalizer_path = Path(normalizer_path)
        if not normalizer_path.is_absolute():
            normalizer_path = Path(config.data.root) / normalizer_path
    if normalizer_path is not None and normalizer_path.exists():
        with _PhaseTimer(logger, f"Loading normalizers from {normalizer_path}"):
            normalizers = load_normalizers(normalizer_path, device=None)
    else:
        with _PhaseTimer(logger, "Fitting normalizers on training split"):
            normalizers = fit_normalizers_streaming(
                train_data_raw,
                chunk_size=getattr(config.data, "normalizer_chunk_size", 10000),
                expect_target=True,
            )
        if normalizer_path is not None:
            save_normalizers(normalizers, normalizer_path)
            logger.info("Saved fitted normalizers to %s", normalizer_path)

    with _PhaseTimer(logger, "Wrapping normalized datasets"):
        train_norm = NormalizedDatasetOnTheFly(train_data_raw, normalizers, query_res=config.data.query_res)
        test_norm = NormalizedDatasetOnTheFly(test_data_raw, normalizers, query_res=config.data.query_res)
    test_loader = DataLoader(
        test_norm,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )
    logger.info("Test DataLoader: batch_size=%s num_workers=0 batches=%d", config.data.batch_size, len(test_loader))

    with _PhaseTimer(logger, "Loading model + checkpoint state"):
        model = get_model(config)
        load_training_state(save_dir=save_dir, save_name=save_name, model=model)
        model = model.to(device).eval()

    inverse_test = getattr(config, "inverse_test", True)
    if normalizers.get("target") is not None:
        normalizers["target"] = normalizers["target"].to(device)
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target", None),
        inverse_test=inverse_test,
    )
    data_processor.wrap(model)

    trainer = _make_trainer(config, model, data_processor, device, logger)
    eval_losses = _build_eval_losses(config, use_fgn=getattr(config.gino, "use_fgn_noise", False))
    logger.info("Eval losses configured: %s | inverse_test=%s", list(eval_losses.keys()), inverse_test)

    # One-step eval is opt-in; rollout-focused usage skips it by default.
    run_single_step = bool(args.run_single_step) and (not args.skip_single_step)
    if run_single_step:
        with _PhaseTimer(logger, "Running one-step evaluation"):
            metrics = trainer.evaluate(eval_losses, test_loader, log_prefix="test")
        clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items() if not k.endswith("_outputs")}
        logger.info("Post-training one-step TEST metrics:")
        print("Post-training one-step TEST metrics:")
        for k, v in clean.items():
            logger.info("  %s: %.6e", k, v)
            print(f"  {k}: {v:.6e}")

    run_rollout_cfg = bool(getattr(config.rollout, "run_after_training", False))
    run_rollout = (run_rollout_cfg or args.run_rollout) and (not args.skip_rollout)
    if not run_rollout:
        logger.info("Skipping rollout evaluation (config.rollout.run_after_training=false and --run_rollout not set).")
        return 0

    rollout_length = config.data.rollout_length
    history_steps = config.data.n_history
    rollout_skip_before_timestep = getattr(config.data, "skip_before_timestep", 0)
    ch_to_idx = {"wd": 0, "vx": 1, "vy": 2}
    target_indices = [ch_to_idx[v] for v in target_variables]
    with _PhaseTimer(logger, "Building rollout test dataset"):
        rollout_test_dataset = FloodRolloutTestDatasetHDF(
            rollout_data_root=config.rollout_data.root,
            n_history=history_steps,
            rollout_length=rollout_length,
            run_ids=None,
            test_txt=getattr(config.rollout_data, "test_txt", "test.txt"),
            static_text_files=getattr(config.rollout_data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]),
            hdf_suffix=".hdf",
            raise_on_smaller=True,
            skip_before_timestep=rollout_skip_before_timestep,
        )
    logger.info("Rollout runs available: %d", len(rollout_test_dataset))
    with _PhaseTimer(logger, "Collecting rollout fields"):
        rollout_geom_list, rollout_static_list, rollout_boundary_list, rollout_dyn_list, _ = collect_all_fields(
            rollout_test_dataset, expect_target=False
        )

    ref_device = None
    for key in ("dynamic", "target", "geometry"):
        if key in normalizers and normalizers[key] is not None and hasattr(normalizers[key], "mean"):
            ref_device = normalizers[key].mean.device
            break
    if ref_device is None:
        ref_device = torch.device("cpu")
    for key in ["geometry", "static", "boundary", "dynamic", "target"]:
        if key in normalizers and normalizers[key] is not None:
            normalizers[key].to(ref_device)
    geometry_big = torch.stack(rollout_geom_list, dim=0) if rollout_geom_list else None
    static_big = torch.stack(rollout_static_list, dim=0) if rollout_static_list else None
    boundary_big = torch.stack(rollout_boundary_list, dim=0) if rollout_boundary_list else None
    dynamic_big = torch.stack(rollout_dyn_list, dim=0) if rollout_dyn_list else None
    if dynamic_big is not None:
        # Align rollout dynamic channels with training target_variables order/subset.
        dynamic_big = dynamic_big[..., target_indices]
    if geometry_big is not None and "geometry" in normalizers:
        geometry_big = normalizers["geometry"].transform(geometry_big.to(ref_device))
    if static_big is not None and "static" in normalizers:
        static_big = normalizers["static"].transform(static_big.to(ref_device))
    if boundary_big is not None and "boundary" in normalizers:
        boundary_big = normalizers["boundary"].transform(boundary_big.to(ref_device))
    if dynamic_big is not None and "dynamic" in normalizers:
        dynamic_big = normalizers["dynamic"].transform(dynamic_big.to(ref_device))
    logger.info("Rollout tensors transformed on device=%s", ref_device)

    normalized_rollout_samples = []
    for i in tqdm(range(len(rollout_test_dataset)), desc="Preparing normalized rollout samples"):
        normalized_rollout_samples.append({
            "run_id": rollout_test_dataset.valid_run_ids[i],
            "geometry": geometry_big[i],
            "static": static_big[i],
            "boundary": boundary_big[i],
            "dynamic": dynamic_big[i],
        })
    rollout_normalized_dataset = NormalizedRolloutTestDataset(
        normalized_samples=normalized_rollout_samples,
        query_res=config.data.query_res,
    )
    logger.info("Prepared rollout dataset with %d runs", len(rollout_normalized_dataset))

    with _PhaseTimer(logger, "Running rollout evaluation + plotting"):
        _rollout_prediction_generic(
            trainer=trainer,
            rollout_dataset=rollout_normalized_dataset,
            rollout_length=rollout_length,
            history_steps=history_steps,
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            device=device,
            skip_before_timestep=rollout_skip_before_timestep,
            dt=config.data.dt,
            out_dir=config.rollout.out_dir,
            target_variables=target_variables,
            logger=logger,
            fgn_noise_dim=getattr(config.gino, "fgn_noise_dim", 32) if getattr(config.gino, "use_fgn_noise", False) else None,
            n_ensemble_samples=getattr(config.rollout, "n_ensemble_samples", 1),
        )
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

