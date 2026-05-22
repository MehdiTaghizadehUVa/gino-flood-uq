"""Publication-quality rollout figures and animations."""

from __future__ import annotations

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

def create_rollout_animation(
        geometry,
        wd_gt, wd_pred,
        vx_gt, vy_gt,
        vx_pred, vy_pred,
        run_id=None,
        out_dir=".",
        filename_prefix="rollout",
        dt_seconds: float = 900.0
):
    """
    Creates an animation comparing Ground Truth and Predictions in a 3x2 grid.
    - Row 1: Ground Truth Depth vs. Predicted Depth
    - Row 2: Ground Truth VX vs. Predicted VX
    - Row 3: Ground Truth VY vs. Predicted VY
    """
    # Convert inputs to numpy arrays
    if isinstance(geometry, torch.Tensor):
        geometry = geometry.cpu().numpy()
    x_coords, y_coords = geometry[:, 0], geometry[:, 1]

    wd_gt, wd_pred = np.asarray(wd_gt), np.asarray(wd_pred)
    vx_gt, vy_gt = np.asarray(vx_gt), np.asarray(vy_gt)
    vx_pred, vy_pred = np.asarray(vx_pred), np.asarray(vy_pred)
    rollout_length = wd_gt.shape[0]

    # Prepare figure with a 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
    fig.suptitle(f"Rollout Comparison (Run: {run_id or 'unknown'})", fontsize=20)
    (ax_gt_wd, ax_pred_wd), (ax_gt_vx, ax_pred_vx), (ax_gt_vy, ax_pred_vy) = axes

    # --- Set Color Limits ---
    depth_max = max(np.nanmax(wd_gt), np.nanmax(wd_pred))
    # For velocities, find the max absolute value for a symmetric color scale
    vx_abs_max = np.max([np.abs(vx_gt), np.abs(vx_pred)])
    vy_abs_max = np.max([np.abs(vy_gt), np.abs(vy_pred)])

    # --- Row 1: Water Depth ---
    sc_gt_wd = ax_gt_wd.scatter(x_coords, y_coords, c=wd_gt[0], vmin=0, vmax=depth_max, s=15, cmap='viridis')
    ax_gt_wd.set_title("Ground Truth Depth", pad=10)
    ax_gt_wd.axis('off')
    fig.colorbar(sc_gt_wd, ax=ax_gt_wd, fraction=0.046, pad=0.04).set_label("Depth (m)")

    sc_pred_wd = ax_pred_wd.scatter(x_coords, y_coords, c=wd_pred[0], vmin=0, vmax=depth_max, s=15, cmap='viridis')
    ax_pred_wd.set_title("Predicted Depth", pad=10)
    ax_pred_wd.axis('off')
    fig.colorbar(sc_pred_wd, ax=ax_pred_wd, fraction=0.046, pad=0.04).set_label("Depth (m)")

    # --- Row 2: X-Velocity (VX) ---
    sc_gt_vx = ax_gt_vx.scatter(x_coords, y_coords, c=vx_gt[0], vmin=-vx_abs_max, vmax=vx_abs_max, s=15,
                                cmap='coolwarm')
    ax_gt_vx.set_title("Ground Truth VX", pad=10)
    ax_gt_vx.axis('off')
    fig.colorbar(sc_gt_vx, ax=ax_gt_vx, fraction=0.046, pad=0.04).set_label("VX (m/s)")

    sc_pred_vx = ax_pred_vx.scatter(x_coords, y_coords, c=vx_pred[0], vmin=-vx_abs_max, vmax=vx_abs_max, s=15,
                                    cmap='coolwarm')
    ax_pred_vx.set_title("Predicted VX", pad=10)
    ax_pred_vx.axis('off')
    fig.colorbar(sc_pred_vx, ax=ax_pred_vx, fraction=0.046, pad=0.04).set_label("VX (m/s)")

    # --- Row 3: Y-Velocity (VY) ---
    sc_gt_vy = ax_gt_vy.scatter(x_coords, y_coords, c=vy_gt[0], vmin=-vy_abs_max, vmax=vy_abs_max, s=15,
                                cmap='coolwarm')
    ax_gt_vy.set_title("Ground Truth VY", pad=10)
    ax_gt_vy.axis('off')
    fig.colorbar(sc_gt_vy, ax=ax_gt_vy, fraction=0.046, pad=0.04).set_label("VY (m/s)")

    sc_pred_vy = ax_pred_vy.scatter(x_coords, y_coords, c=vy_pred[0], vmin=-vy_abs_max, vmax=vy_abs_max, s=15,
                                    cmap='coolwarm')
    ax_pred_vy.set_title("Predicted VY", pad=10)
    ax_pred_vy.axis('off')
    fig.colorbar(sc_pred_vy, ax=ax_pred_vy, fraction=0.046, pad=0.04).set_label("VY (m/s)")

    # Animation update function
    def animate(frame_idx):
        time_hours = (frame_idx + 1) * dt_seconds / 3600.0
        fig.suptitle(f"Rollout Comparison (Run: {run_id or 'unknown'}) - Time: {time_hours:.2f} hrs", fontsize=20)
        sc_gt_wd.set_array(wd_gt[frame_idx])
        sc_pred_wd.set_array(wd_pred[frame_idx])
        sc_gt_vx.set_array(vx_gt[frame_idx])
        sc_pred_vx.set_array(vx_pred[frame_idx])
        sc_gt_vy.set_array(vy_gt[frame_idx])
        sc_pred_vy.set_array(vy_pred[frame_idx])
        return sc_gt_wd, sc_pred_wd, sc_gt_vx, sc_pred_vx, sc_gt_vy, sc_pred_vy

    ani = animation.FuncAnimation(fig, animate, frames=rollout_length, interval=200, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{run_id or 'unknown'}.gif")
    ani.save(out_path, writer='pillow', fps=5)
    plt.close(fig)
    print(f"Saved rollout animation to: {out_path}")


def generate_publication_maps(
        geometry,
        wd_gt_array: np.ndarray, wd_pred_array: np.ndarray,
        vx_gt_array: np.ndarray, vy_gt_array: np.ndarray,
        vx_pred_array: np.ndarray, vy_pred_array: np.ndarray,
        steps,
        out_dir: str = ".",
        run_id: str = None,
        filename_prefix: str = "step"
):
    """
    Generates high-quality 3x3 comparison maps for specific timesteps.
    - Row 1: Ground Truth Depth, Predicted Depth, Absolute Depth Error.
    - Row 2: Ground Truth VX, Predicted VX, Absolute VX Error.
    - Row 3: Ground Truth VY, Predicted VY, Absolute VY Error.
    """
    if isinstance(steps, int):
        steps = [steps]
    geo_np = geometry.cpu().numpy() if hasattr(geometry, "cpu") else np.asarray(geometry)
    x, y = geo_np[:, 0], geo_np[:, 1]
    rid = run_id or "unknown"
    os.makedirs(out_dir, exist_ok=True)
    plt.rc("font", family="serif", size=12)

    for t in steps:
        if t < 0 or t >= wd_gt_array.shape[0]:
            print(f"  Skipping invalid step {t}")
            continue

        # Extract data for the specific timestep
        wd_gt, wd_pred = wd_gt_array[t], wd_pred_array[t]
        vx_gt, vy_gt = vx_gt_array[t], vy_gt_array[t]
        vx_pred, vy_pred = vx_pred_array[t], vy_pred_array[t]

        # Calculate errors
        err_wd = np.abs(wd_pred - wd_gt)
        err_vx = np.abs(vx_pred - vx_gt)
        err_vy = np.abs(vy_pred - vy_gt)

        # Determine robust color limits
        dmax = max(np.nanmax(wd_gt), np.nanmax(wd_pred))
        emax_wd = np.nanmax(err_wd)
        vx_abs_max = np.max([np.abs(vx_gt), np.abs(vx_pred)])
        vy_abs_max = np.max([np.abs(vy_gt), np.abs(vy_pred)])
        emax_vx = np.nanmax(err_vx)
        emax_vy = np.nanmax(err_vy)

        fig, axs = plt.subplots(3, 3, figsize=(18, 17), dpi=300, constrained_layout=True)
        panels = [
            ("(a) Ground Truth Depth", wd_gt, "viridis", 0.0, dmax, "Depth (m)"),
            ("(b) Predicted Depth", wd_pred, "viridis", 0.0, dmax, "Depth (m)"),
            ("(c) Depth Abs. Error", err_wd, "magma", 0.0, emax_wd, "Error (m)"),
            ("(d) Ground Truth VX", vx_gt, "coolwarm", -vx_abs_max, vx_abs_max, "VX (m/s)"),
            ("(e) Predicted VX", vx_pred, "coolwarm", -vx_abs_max, vx_abs_max, "VX (m/s)"),
            ("(f) VX Abs. Error", err_vx, "magma", 0.0, emax_vx, "Error (m/s)"),
            ("(g) Ground Truth VY", vy_gt, "coolwarm", -vy_abs_max, vy_abs_max, "VY (m/s)"),
            ("(h) Predicted VY", vy_pred, "coolwarm", -vy_abs_max, vy_abs_max, "VY (m/s)"),
            ("(i) VY Abs. Error", err_vy, "magma", 0.0, emax_vy, "Error (m/s)"),
        ]

        for ax, (title, data, cmap, vmin, vmax, cblabel) in zip(axs.flatten(), panels):
            sc = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, s=6, marker="s", linewidths=0,
                            rasterized=True)
            ax.set_title(title, pad=8, fontsize=14)
            ax.set_aspect("equal")
            ax.axis("off")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cblabel, labelpad=10, fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        fname = f"{filename_prefix}_{rid}_t{t}.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved publication map for t={t} -> {out_path}")


###############################################################################
# 8b) Training verification (gradient flow + overfit sanity check)
###############################################################################
