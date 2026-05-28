"""Python-generated dashboard figures for FGN serving products."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from neuralop.flood.serving.forcing import ForcingInput


class ForecastFigureBuilder:
    """Render publication-style SVG figures from serving product summaries."""

    def __init__(
        self,
        *,
        dt_seconds: int,
        skip_before_timestep: int,
        n_history: int,
        thresholds_m: Sequence[float],
    ) -> None:
        self.dt_seconds = int(dt_seconds)
        self.skip_before_timestep = int(skip_before_timestep)
        self.n_history = int(n_history)
        self.thresholds_m = tuple(float(x) for x in thresholds_m)

    def write_figures(
        self,
        *,
        forcing: ForcingInput,
        raw_summary: Mapping[str, object],
        calibrated_summary: Mapping[str, object],
        comparison_summary: Mapping[str, object],
        output_dir: str | Path,
    ) -> list[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib as mpl

        # rc_context isolates every figure in this run from the worker's
        # global matplotlib state. Required to keep rcParams corruption
        # contained — see _plot_spatial_field in eval.render for context.
        with mpl.rc_context(self._SERVING_RC_OVERRIDES):
            return [
                self._write_forcing_hydrograph(forcing, out_dir / "forcing_hydrograph.svg"),
                self._write_extent_by_time(calibrated_summary, out_dir / "uq_extent_by_time.svg"),
                self._write_exceedance_bars(calibrated_summary, out_dir / "uq_exceedance_bars.svg"),
                self._write_uncertainty_width(calibrated_summary, out_dir / "uq_uncertainty_width.svg"),
                self._write_calibration_effect(raw_summary, calibrated_summary, comparison_summary, out_dir / "calibration_effect.svg"),
            ]

    # Style overrides applied inside ``mpl.rc_context`` in ``write_figures``
    # so they cannot leak across serving runs in long-running Celery workers
    # (see the ``_plot_spatial_field`` comment in eval.render for context).
    _SERVING_RC_OVERRIDES = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#b7c9d0",
        "axes.labelcolor": "#102027",
        "xtick.color": "#465b63",
        "ytick.color": "#465b63",
        "grid.color": "#dfe7ea",
        "grid.linewidth": 0.8,
        "svg.hashsalt": "fgn-serving-uq",
    }

    @classmethod
    def _setup_matplotlib(cls):
        """Return ``plt`` with the Agg backend ensured.

        Called inside an ``mpl.rc_context`` scope established by
        ``write_figures``; individual ``_write_*`` methods inherit the
        scoped rcParams from that context.
        """
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt

    @staticmethod
    def _save_svg(fig, path: Path) -> Path:
        fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.05, metadata={"Date": None})
        return path

    @staticmethod
    def _lead_hours(summary: Mapping[str, object]) -> np.ndarray:
        return np.asarray(summary.get("lead_time_hours", []), dtype=np.float64)

    @staticmethod
    def _series(summary: Mapping[str, object], key: str) -> np.ndarray:
        return np.asarray(summary.get(key, []), dtype=np.float64)

    @staticmethod
    def _threshold_payload(summary: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
        raw = summary.get("exceedance_by_threshold_m", {})
        return raw if isinstance(raw, dict) else {}

    def _write_forcing_hydrograph(self, forcing: ForcingInput, path: Path) -> Path:
        plt = self._setup_matplotlib()
        t_h = np.arange(forcing.n_rows, dtype=np.float64) * float(forcing.dt_seconds) / 3600.0
        forecast_start = (self.skip_before_timestep + self.n_history) * float(forcing.dt_seconds) / 3600.0
        forecast_end = forecast_start + max(0, int(forcing.forecast_steps) - 1) * float(forcing.dt_seconds) / 3600.0
        fig, (ax_stage, ax_rain) = plt.subplots(
            2,
            1,
            figsize=(8.2, 4.8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.2, 1.0]},
        )
        try:
            for ax in (ax_stage, ax_rain):
                ax.axvspan(forecast_start, forecast_end, color="#e8f6f4", alpha=0.8, label="forecast window")
                ax.grid(True, axis="y")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            ax_stage.plot(t_h, forcing.stage, color="#0b766d", linewidth=2.0)
            ax_stage.set_title("Uploaded forcing time series", loc="left")
            ax_stage.set_ylabel("Stage (m)")
            bar_width = max(float(forcing.dt_seconds) / 3600.0 * 0.82, 0.02)
            ax_rain.bar(t_h, forcing.precipitation, width=bar_width, color="#2f6db3", edgecolor="#1e4f88", linewidth=0.3)
            ax_rain.set_ylabel("Precip. (mm/step)")
            ax_rain.set_xlabel("Lead time from CSV start (hours)")
            handles, labels = ax_stage.get_legend_handles_labels()
            if handles:
                ax_stage.legend(handles[:1], labels[:1], loc="upper right", frameon=False)
            return self._save_svg(fig, path)
        finally:
            plt.close(fig)

    def _write_extent_by_time(self, summary: Mapping[str, object], path: Path) -> Path:
        plt = self._setup_matplotlib()
        lead = self._lead_hours(summary)
        payload = self._threshold_payload(summary)
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 3.8), constrained_layout=True)
        try:
            ax.grid(True, axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            colors = ["#0b766d", "#2f6db3", "#c77700", "#8b1e3f", "#5b5fc7"]
            plotted = False
            for idx, threshold in enumerate(self.thresholds_m):
                item = payload.get(f"{threshold:g}")
                if not item:
                    continue
                series = np.asarray(item.get("expected_area_fraction_wettable_by_time", []), dtype=np.float64)
                if lead.size != series.size or series.size == 0:
                    continue
                ax.plot(lead, series * 100.0, linewidth=2.0, color=colors[idx % len(colors)], label=f">{threshold:g} m")
                plotted = True
            if not plotted:
                series = self._series(summary, "expected_flooded_area_fraction_wettable_by_time_gt_0.05m")
                if lead.size == series.size and series.size:
                    ax.plot(lead, series * 100.0, linewidth=2.0, color=colors[0], label=">0.05 m")
            ax.set_title("Expected flooded area fraction by threshold", loc="left")
            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel("Wettable domain (%)")
            ax.set_ylim(bottom=0.0)
            ax.legend(loc="upper left", ncols=3, frameon=False)
            return self._save_svg(fig, path)
        finally:
            plt.close(fig)

    def _write_exceedance_bars(self, summary: Mapping[str, object], path: Path) -> Path:
        plt = self._setup_matplotlib()
        payload = self._threshold_payload(summary)
        rows: list[tuple[float, float, float]] = []
        for threshold in self.thresholds_m:
            item = payload.get(f"{threshold:g}")
            if not item:
                continue
            rows.append(
                (
                    threshold,
                    100.0 * float(item.get("peak_expected_area_fraction_wettable", 0.0) or 0.0),
                    100.0 * float(item.get("peak_high_confidence_area_fraction_wettable", 0.0) or 0.0),
                )
            )
        fig, ax = plt.subplots(1, 1, figsize=(7.2, max(3.0, 0.55 * max(1, len(rows)) + 1.0)), constrained_layout=True)
        try:
            ax.grid(True, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if rows:
                y = np.arange(len(rows))
                peak = np.asarray([row[1] for row in rows])
                high = np.asarray([row[2] for row in rows])
                ax.barh(y + 0.16, peak, height=0.30, color="#0b766d", label="expected footprint")
                ax.barh(y - 0.16, high, height=0.30, color="#c77700", label="P >= 50% footprint")
                ax.set_yticks(y, [f">{row[0]:g} m" for row in rows])
            ax.set_title("Peak exceedance footprint", loc="left")
            ax.set_xlabel("Wettable domain (%)")
            ax.set_xlim(left=0.0)
            ax.legend(loc="lower right", frameon=False)
            return self._save_svg(fig, path)
        finally:
            plt.close(fig)

    def _write_uncertainty_width(self, summary: Mapping[str, object], path: Path) -> Path:
        plt = self._setup_matplotlib()
        lead = self._lead_hours(summary)
        iqr = self._series(summary, "area_weighted_iqr_wd_m_by_time")
        central90 = self._series(summary, "area_weighted_central_90_wd_m_by_time")
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 3.8), constrained_layout=True)
        try:
            ax.grid(True, axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if lead.size == iqr.size and iqr.size:
                ax.plot(lead, iqr, color="#5b5fc7", linewidth=2.0, label="p75-p25")
                ax.fill_between(lead, 0.0, iqr, color="#5b5fc7", alpha=0.13)
            if lead.size == central90.size and central90.size:
                ax.plot(lead, central90, color="#8b1e3f", linewidth=1.8, linestyle="--", label="p95-p05")
            ax.set_title("Area-weighted uncertainty width", loc="left")
            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel("Water depth width (m)")
            ax.set_ylim(bottom=0.0)
            ax.legend(loc="upper left", frameon=False)
            return self._save_svg(fig, path)
        finally:
            plt.close(fig)

    def _write_calibration_effect(
        self,
        raw_summary: Mapping[str, object],
        calibrated_summary: Mapping[str, object],
        comparison_summary: Mapping[str, object],
        path: Path,
    ) -> Path:
        plt = self._setup_matplotlib()
        raw_payload = self._threshold_payload(raw_summary)
        cal_payload = self._threshold_payload(calibrated_summary)
        rows: list[tuple[str, float]] = []
        for threshold in self.thresholds_m:
            raw_item = raw_payload.get(f"{threshold:g}")
            cal_item = cal_payload.get(f"{threshold:g}")
            if not raw_item or not cal_item:
                continue
            delta = 100.0 * (
                float(cal_item.get("peak_expected_area_fraction_wettable", 0.0) or 0.0)
                - float(raw_item.get("peak_expected_area_fraction_wettable", 0.0) or 0.0)
            )
            rows.append((f">{threshold:g} m footprint", delta))
        iqr_delta = comparison_summary.get("delta_peak_area_weighted_iqr_wd_m")
        if isinstance(iqr_delta, (int, float)):
            rows.append(("Peak IQR width (m)", float(iqr_delta)))
        fig, ax = plt.subplots(1, 1, figsize=(8.2, max(3.0, 0.48 * max(1, len(rows)) + 1.0)), constrained_layout=True)
        try:
            ax.axvline(0.0, color="#465b63", linewidth=1.0)
            ax.grid(True, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if rows:
                y = np.arange(len(rows))
                vals = np.asarray([row[1] for row in rows], dtype=np.float64)
                colors = ["#0b766d" if v >= 0 else "#8b1e3f" for v in vals]
                ax.barh(y, vals, color=colors, alpha=0.9)
                ax.set_yticks(y, [row[0] for row in rows])
            ax.set_title("Calibration shift relative to raw FGN", loc="left")
            ax.set_xlabel("Change: percentage points for footprint, meters for IQR")
            return self._save_svg(fig, path)
        finally:
            plt.close(fig)
