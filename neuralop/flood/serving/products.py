"""Build user-facing forecast products from raw and calibrated ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np

if TYPE_CHECKING:
    from neuralop.flood.serving.calibration import CalibrationAdapter


@dataclass(frozen=True)
class ForecastResult:
    """Physical-space forecast members for one serving run.

    members_wd shape is [n_members, n_time, n_cells] in meters.
    """

    members_wd: np.ndarray
    lead_time_hours: np.ndarray
    wettable_mask: np.ndarray | None = None
    metadata: Dict[str, object] | None = None

    def validate(self) -> "ForecastResult":
        arr = np.asarray(self.members_wd)
        if arr.ndim != 3:
            raise ValueError(f"members_wd must have shape [members,time,cells], got {arr.shape}.")
        if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1:
            raise ValueError("members_wd dimensions must all be positive.")
        if np.asarray(self.lead_time_hours).shape[0] != arr.shape[1]:
            raise ValueError("lead_time_hours length must match forecast time dimension.")
        if self.wettable_mask is not None and np.asarray(self.wettable_mask).shape[0] != arr.shape[2]:
            raise ValueError("wettable_mask length must match forecast cell dimension.")
        return self


class ForecastProductBuilder:
    """Convert ensembles into dashboard/download summaries without ground truth."""

    def __init__(self, *, thresholds_m: Sequence[float] = (0.01, 0.05, 0.10, 0.30, 0.50)) -> None:
        self.thresholds_m = tuple(float(x) for x in thresholds_m)

    def build_summary(
        self,
        forecast: ForecastResult,
        *,
        label: str = "calibrated",
        calibration_adapter: "CalibrationAdapter | None" = None,
    ) -> Dict[str, object]:
        forecast = forecast.validate()
        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        mask = np.asarray(forecast.wettable_mask, dtype=bool) if forecast.wettable_mask is not None else None
        members_eval = members[:, :, mask] if mask is not None else members
        mean = members_eval.mean(axis=0)
        std = members_eval.std(axis=0)
        q05, q50, q95 = np.quantile(members_eval, [0.05, 0.50, 0.95], axis=0)
        use_iso = (
            calibration_adapter is not None
            and calibration_adapter.has_isotonic_curves()
        )
        lead_arr = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        exceedance: Dict[str, object] = {}
        for thr in self.thresholds_m:
            prob_map = (members_eval > thr).mean(axis=0)
            if use_iso:
                prob_map = calibration_adapter.apply_isotonic_exceedance(
                    prob_map,
                    threshold_m=thr,
                    lead_time_hour=lead_arr,
                    wettable_mask=mask,
                )
            exceedance[f"p_wd_gt_{thr:g}m_mean"] = float(prob_map.mean())
        peak_by_time = mean.max(axis=1)
        inundated_by_time = (mean > 0.05).sum(axis=1)
        arrival = np.argmax(mean > 0.05, axis=0) if mean.size else np.array([], dtype=int)
        any_wet = np.any(mean > 0.05, axis=0) if mean.size else np.array([], dtype=bool)
        arrival_hours = np.asarray(forecast.lead_time_hours)[arrival[any_wet]] if np.any(any_wet) else np.array([])
        return {
            "label": label,
            "n_members": int(members.shape[0]),
            "n_time": int(members.shape[1]),
            "n_cells": int(members.shape[2]),
            "lead_time_hours": [float(x) for x in np.asarray(forecast.lead_time_hours)],
            "mean_wd_overall_m": float(mean.mean()),
            "max_mean_wd_m": float(mean.max()),
            "mean_spread_wd_m": float(std.mean()),
            "mean_q05_wd_m": float(q05.mean()),
            "mean_q50_wd_m": float(q50.mean()),
            "mean_q95_wd_m": float(q95.mean()),
            "peak_mean_wd_by_time_m": [float(x) for x in peak_by_time],
            "inundated_cells_by_time_gt_0.05m": [int(x) for x in inundated_by_time],
            "max_inundated_cells_gt_0.05m": int(inundated_by_time.max()) if inundated_by_time.size else 0,
            "mean_arrival_time_hours_gt_0.05m": float(arrival_hours.mean()) if arrival_hours.size else None,
            "isotonic_calibration_applied": bool(use_iso),
            **exceedance,
        }

    def write_map_pngs(
        self,
        forecast: ForecastResult,
        *,
        output_dir: str | Path,
        label: str = "calibrated",
        max_times: int = 3,
        calibration_adapter: "CalibrationAdapter | None" = None,
    ) -> list[Path]:
        """Write lightweight UTM scatter-map PNG products for dashboard/download use.

        This intentionally avoids requiring ground truth. It produces forecast-only
        hydraulic products: ensemble mean, spread, p95, and exceedance probability.
        When ``calibration_adapter`` provides isotonic curves, exceedance probability
        maps are isotonic-calibrated on the wettable cells.
        """
        forecast = forecast.validate()
        metadata = forecast.metadata or {}
        geometry = metadata.get("geometry_xy")
        if geometry is None:
            return []
        xy = np.asarray(geometry, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != forecast.members_wd.shape[2]:
            raise ValueError("forecast.metadata['geometry_xy'] must have shape [n_cells,2].")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        lead = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        n_time = members.shape[1]
        if n_time <= 0:
            return []
        if n_time <= max_times:
            time_indices = list(range(n_time))
        else:
            time_indices = sorted(set(np.linspace(0, n_time - 1, max_times, dtype=int).tolist()))
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else None
        )
        use_iso = (
            calibration_adapter is not None
            and calibration_adapter.has_isotonic_curves()
            and wettable is not None
        )
        product_specs = [
            ("mean", "Mean WD (m)", lambda arr: arr.mean(axis=0), "viridis", 0.0, None, None),
            ("spread", "WD spread (m)", lambda arr: arr.std(axis=0), "magma", 0.0, None, None),
            ("p95", "WD p95 (m)", lambda arr: np.quantile(arr, 0.95, axis=0), "viridis", 0.0, None, None),
            ("p_gt_0p30m", "P(WD > 0.30 m)", lambda arr: (arr > 0.30).mean(axis=0), "cividis", 0.0, 1.0, 0.30),
        ]
        for t_idx in time_indices:
            arr_t = members[:, t_idx, :]
            for key, title, reducer, cmap, vmin, vmax, threshold_m in product_specs:
                values = np.asarray(reducer(arr_t), dtype=np.float64)
                if use_iso and threshold_m is not None:
                    raw_wet = values[wettable]
                    cal_wet = calibration_adapter.apply_isotonic_exceedance(
                        raw_wet,
                        threshold_m=float(threshold_m),
                        lead_time_hour=float(lead[t_idx]),
                        wettable_mask=wettable,
                    )
                    values = values.copy()
                    values[wettable] = cal_wet
                    values[~wettable] = 0.0
                fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
                sc = ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=values,
                    s=3.0,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0,
                )
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{label} {title} | lead {lead[t_idx]:.2f} h")
                ax.set_xlabel("UTM Easting (m)")
                ax.set_ylabel("UTM Northing (m)")
                cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
                cbar.set_label(title)
                fig.tight_layout()
                path = out_dir / f"{label}_{key}_t{t_idx + 1:03d}.png"
                fig.savefig(path)
                plt.close(fig)
                paths.append(path)
        return paths

    def write_animation_gif(
        self,
        forecast: ForecastResult,
        *,
        output_path: str | Path,
        label: str = "calibrated",
        fps: int = 4,
    ) -> Path | None:
        """Render an animated GIF showing ensemble-mean WD evolving over the forecast.

        Returns the written path, or None if the forecast does not provide
        ``geometry_xy`` metadata (matching ``write_map_pngs``'s contract).
        """
        forecast = forecast.validate()
        metadata = forecast.metadata or {}
        geometry = metadata.get("geometry_xy")
        if geometry is None:
            return None
        xy = np.asarray(geometry, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != forecast.members_wd.shape[2]:
            raise ValueError("forecast.metadata['geometry_xy'] must have shape [n_cells,2].")
        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        lead = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        n_time = members.shape[1]
        if n_time <= 0:
            return None
        mean_by_time = members.mean(axis=0)
        vmax = float(mean_by_time.max()) if mean_by_time.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter

        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=120)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=mean_by_time[0],
            s=3.0,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            linewidths=0,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label("Mean WD (m)")
        title = ax.set_title(f"{label} mean WD | lead {lead[0]:.2f} h")
        try:
            writer = PillowWriter(fps=int(max(1, fps)))
            with writer.saving(fig, str(out_path), dpi=120):
                for t_idx in range(n_time):
                    scatter.set_array(mean_by_time[t_idx])
                    title.set_text(f"{label} mean WD | lead {lead[t_idx]:.2f} h")
                    writer.grab_frame()
        finally:
            plt.close(fig)
        return out_path
