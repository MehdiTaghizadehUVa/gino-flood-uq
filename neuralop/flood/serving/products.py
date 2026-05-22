"""Build user-facing forecast products from raw and calibrated ensembles."""

from __future__ import annotations

import json
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

    EXTENT_THRESHOLD_M = 0.05

    def __init__(self, *, thresholds_m: Sequence[float] = (0.01, 0.05, 0.10, 0.30, 0.50)) -> None:
        self.thresholds_m = tuple(float(x) for x in thresholds_m)

    @staticmethod
    def _wettable_area(forecast: ForecastResult, mask: np.ndarray | None) -> np.ndarray:
        n_cells = int(forecast.members_wd.shape[2])
        raw = (forecast.metadata or {}).get("cell_area_m2")
        if raw is None:
            area = np.ones(n_cells, dtype=np.float64)
        else:
            area = np.asarray(raw, dtype=np.float64).reshape(-1)
            if area.shape[0] != n_cells:
                raise ValueError("forecast.metadata['cell_area_m2'] must have length matching n_cells.")
            area = np.where(np.isfinite(area) & (area > 0.0), area, 0.0)
            if float(area.sum()) <= 0.0:
                area = np.ones(n_cells, dtype=np.float64)
        return area[mask] if mask is not None else area

    @staticmethod
    def _area_by_time(probability_by_time: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
        return np.asarray(probability_by_time, dtype=np.float64) @ np.asarray(area_m2, dtype=np.float64)

    @staticmethod
    def _peak_value_and_time(values: np.ndarray, lead_hours: np.ndarray) -> tuple[float, float | None]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0 or float(np.nanmax(arr)) <= 0.0:
            return 0.0, None
        idx = int(np.nanargmax(arr))
        return float(arr[idx]), float(lead_hours[idx])

    @staticmethod
    def _first_time_at_or_above(values: np.ndarray, lead_hours: np.ndarray, threshold: float) -> float | None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        hits = np.flatnonzero(arr >= float(threshold))
        if hits.size == 0:
            return None
        return float(lead_hours[int(hits[0])])

    @staticmethod
    def _threshold_key(threshold_m: float) -> str:
        return f"{float(threshold_m):.2f}".replace(".", "p").replace("-", "m")

    def _probability_by_time(
        self,
        members_eval: np.ndarray,
        *,
        threshold_m: float,
        lead_hours: np.ndarray,
        wettable_mask: np.ndarray | None,
        calibration_adapter: "CalibrationAdapter | None",
    ) -> np.ndarray:
        prob = (members_eval > float(threshold_m)).mean(axis=0)
        if (
            calibration_adapter is not None
            and calibration_adapter.has_isotonic_curves()
            and wettable_mask is not None
        ):
            prob = calibration_adapter.apply_isotonic_exceedance(
                prob,
                threshold_m=float(threshold_m),
                lead_time_hour=lead_hours,
                wettable_mask=wettable_mask,
            )
        return np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)

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
        area_m2 = self._wettable_area(forecast, mask)
        wettable_area_m2 = float(area_m2.sum())
        if wettable_area_m2 <= 0.0:
            wettable_area_m2 = float(max(1, members_eval.shape[2]))
            area_m2 = np.ones(members_eval.shape[2], dtype=np.float64)
        mean = members_eval.mean(axis=0)
        std = members_eval.std(axis=0)
        q05, q25, q50, q75, q95 = np.quantile(members_eval, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        iqr = q75 - q25
        central_90 = q95 - q05
        area_weighted_iqr_by_time = (iqr @ area_m2) / wettable_area_m2
        area_weighted_central90_by_time = (central_90 @ area_m2) / wettable_area_m2
        use_iso = calibration_adapter is not None and calibration_adapter.has_isotonic_curves()
        lead_arr = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        exceedance: Dict[str, object] = {}
        exceedance_nested: Dict[str, object] = {}
        for thr in self.thresholds_m:
            prob_map = self._probability_by_time(
                members_eval,
                threshold_m=thr,
                lead_hours=lead_arr,
                wettable_mask=mask,
                calibration_adapter=calibration_adapter,
            )
            area_by_time = self._area_by_time(prob_map, area_m2)
            fraction_by_time = area_by_time / wettable_area_m2
            high_conf_by_time = self._area_by_time(prob_map >= 0.5, area_m2)
            high_conf_fraction_by_time = high_conf_by_time / wettable_area_m2
            peak_area_m2, peak_lead = self._peak_value_and_time(area_by_time, lead_arr)
            peak_high_conf_m2, peak_high_conf_lead = self._peak_value_and_time(high_conf_by_time, lead_arr)
            exceedance[f"p_wd_gt_{thr:g}m_mean"] = float(prob_map.mean())
            exceedance_nested[f"{thr:g}"] = {
                "threshold_m": float(thr),
                "mean_probability": float(prob_map.mean()),
                "expected_area_by_time_m2": [float(x) for x in area_by_time],
                "expected_area_fraction_wettable_by_time": [float(x) for x in fraction_by_time],
                "peak_expected_area_m2": peak_area_m2,
                "peak_expected_area_km2": peak_area_m2 / 1_000_000.0,
                "peak_expected_area_fraction_wettable": float(fraction_by_time.max()) if fraction_by_time.size else 0.0,
                "peak_expected_area_lead_hours": peak_lead,
                "high_confidence_area_by_time_m2": [float(x) for x in high_conf_by_time],
                "high_confidence_area_fraction_wettable_by_time": [float(x) for x in high_conf_fraction_by_time],
                "peak_high_confidence_area_m2": peak_high_conf_m2,
                "peak_high_confidence_area_km2": peak_high_conf_m2 / 1_000_000.0,
                "peak_high_confidence_area_fraction_wettable": float(high_conf_fraction_by_time.max()) if high_conf_fraction_by_time.size else 0.0,
                "peak_high_confidence_area_lead_hours": peak_high_conf_lead,
            }
        extent_prob = self._probability_by_time(
            members_eval,
            threshold_m=self.EXTENT_THRESHOLD_M,
            lead_hours=lead_arr,
            wettable_mask=mask,
            calibration_adapter=calibration_adapter,
        )
        expected_area_by_time_m2 = self._area_by_time(extent_prob, area_m2)
        expected_area_fraction_by_time = expected_area_by_time_m2 / wettable_area_m2
        peak_expected_area_m2, peak_expected_area_lead = self._peak_value_and_time(expected_area_by_time_m2, lead_arr)
        onset_lead = self._first_time_at_or_above(expected_area_fraction_by_time, lead_arr, 0.01)
        p95_peak_by_time = q95.max(axis=1)
        _, peak_p95_lead = self._peak_value_and_time(p95_peak_by_time, lead_arr)
        peak_by_time = mean.max(axis=1)
        checkpoint_disagreement = self.checkpoint_disagreement_summary_by_time(
            members_eval=members_eval,
            member_model_id=(forecast.metadata or {}).get("member_model_id"),
            area_m2=area_m2,
            wettable_area_m2=wettable_area_m2,
            lead_hours=lead_arr,
        )
        inundated_by_time = (mean > 0.05).sum(axis=1)
        arrival = np.argmax(mean > 0.05, axis=0) if mean.size else np.array([], dtype=int)
        any_wet = np.any(mean > 0.05, axis=0) if mean.size else np.array([], dtype=bool)
        arrival_hours = np.asarray(forecast.lead_time_hours)[arrival[any_wet]] if np.any(any_wet) else np.array([])
        max_mean_wd_m = float(mean.max())
        peak_iqr = float(area_weighted_iqr_by_time.max()) if area_weighted_iqr_by_time.size else 0.0
        return {
            "label": label,
            "n_members": int(members.shape[0]),
            "n_time": int(members.shape[1]),
            "n_cells": int(members.shape[2]),
            "lead_time_hours": [float(x) for x in np.asarray(forecast.lead_time_hours)],
            "mean_wd_overall_m": float(mean.mean()),
            "max_mean_wd_m": max_mean_wd_m,
            "mean_spread_wd_m": float(std.mean()),
            "mean_q05_wd_m": float(q05.mean()),
            "mean_q50_wd_m": float(q50.mean()),
            "mean_q95_wd_m": float(q95.mean()),
            "peak_mean_wd_by_time_m": [float(x) for x in peak_by_time],
            "p95_wd_peak_by_time_m": [float(x) for x in p95_peak_by_time],
            "inundated_cells_by_time_gt_0.05m": [int(x) for x in inundated_by_time],
            "max_inundated_cells_gt_0.05m": int(inundated_by_time.max()) if inundated_by_time.size else 0,
            "mean_arrival_time_hours_gt_0.05m": float(arrival_hours.mean()) if arrival_hours.size else None,
            "wettable_area_m2": wettable_area_m2,
            "wettable_area_km2": wettable_area_m2 / 1_000_000.0,
            "expected_flooded_area_by_time_m2_gt_0.05m": [float(x) for x in expected_area_by_time_m2],
            "expected_flooded_area_fraction_wettable_by_time_gt_0.05m": [
                float(x) for x in expected_area_fraction_by_time
            ],
            "peak_expected_flooded_area_m2_gt_0.05m": peak_expected_area_m2,
            "peak_expected_flooded_area_km2_gt_0.05m": peak_expected_area_m2 / 1_000_000.0,
            "peak_expected_flooded_area_fraction_wettable_gt_0.05m": (
                float(expected_area_fraction_by_time.max()) if expected_area_fraction_by_time.size else 0.0
            ),
            "peak_expected_flooded_area_lead_hours_gt_0.05m": peak_expected_area_lead,
            "onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m": onset_lead,
            "peak_p95_wd_lead_hours": peak_p95_lead,
            "area_weighted_iqr_wd_m_by_time": [float(x) for x in area_weighted_iqr_by_time],
            "area_weighted_central_90_wd_m_by_time": [float(x) for x in area_weighted_central90_by_time],
            "peak_area_weighted_iqr_wd_m": peak_iqr,
            "peak_area_weighted_central_90_wd_m": (
                float(area_weighted_central90_by_time.max()) if area_weighted_central90_by_time.size else 0.0
            ),
            "uncertainty_to_signal_ratio": (peak_iqr / max_mean_wd_m) if max_mean_wd_m > 1.0e-9 else None,
            "exceedance_by_threshold_m": exceedance_nested,
            "isotonic_calibration_applied": bool(use_iso),
            **checkpoint_disagreement,
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
        """Write Rivanna-style UTM map PNG products for dashboard/download use.

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
        from neuralop.flood.serving.map_rendering import write_rivanna_style_map_png

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
        elevation_raw = metadata.get("elevation_raw")
        visualization_config = metadata.get("visualization_config")
        product_specs = []
        for threshold_m in self.thresholds_m:
            product_specs.append(
                (
                    f"p_gt_{self._threshold_key(threshold_m)}m",
                    f"P(WD > {threshold_m:g} m)",
                    lambda arr, thr=threshold_m: (arr > thr).mean(axis=0),
                    "viridis",
                    0.0,
                    1.0,
                    float(threshold_m),
                    False,
                    True,
                )
            )
        product_specs.extend(
            [
                ("iqr", "WD IQR (m)", lambda arr: np.quantile(arr, 0.75, axis=0) - np.quantile(arr, 0.25, axis=0), "spread_violet", 0.0, None, None, False, True),
                ("p95", "WD p95 (m)", lambda arr: np.quantile(arr, 0.95, axis=0), "viridis", 0.0, None, None, True, False),
                ("mean", "Mean WD (m)", lambda arr: arr.mean(axis=0), "viridis", 0.0, None, None, True, False),
                ("spread", "WD spread (m)", lambda arr: arr.std(axis=0), "spread_violet", 0.0, None, None, False, True),
            ]
        )
        for t_idx in time_indices:
            arr_t = members[:, t_idx, :]
            for key, title, reducer, cmap, vmin, vmax, threshold_m, is_wd_depth, zero_transparent in product_specs:
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
                path = write_rivanna_style_map_png(
                    values=values,
                    geometry_xy=xy,
                    output_path=out_dir / f"{label}_{key}_t{t_idx + 1:03d}.png",
                    title=f"{label} {title} | lead {lead[t_idx]:.2f} h",
                    colorbar_label=title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
                    is_wd_depth=is_wd_depth,
                    zero_transparent=zero_transparent,
                    visualization_config=visualization_config,
                )
                paths.append(path)
        return paths

    @staticmethod
    def empirical_crps_per_cell(ensemble: np.ndarray) -> np.ndarray:
        """Mean absolute pairwise difference per cell — a ground-truth-free
        CRPS-like measure of intra-ensemble spread.

        Computed via the order-statistic identity::

            E[|X_i - X_j|] = (2/M^2) * Σ_k (2k - M - 1) * x_(k)

        where ``x_(k)`` is the k-th order statistic (1-indexed). This avoids
        the O(M^2 C) intermediate that brute-forces the pairwise differences
        and stays well under memory pressure for the full 5904-cell domain.

        ``ensemble`` must have shape ``[n_members, n_cells]`` (single time
        slice). Returns float32 array of shape ``[n_cells]``.
        """
        sorted_e = np.sort(ensemble.astype(np.float64, copy=False), axis=0)
        m = sorted_e.shape[0]
        if m < 2:
            return np.zeros(sorted_e.shape[1], dtype=np.float32)
        # 0-indexed k -> 1-indexed (k+1); weight = 2(k+1) - m - 1 = 2k + 1 - m,
        # times the normalising factor 2 / m^2 (mean over m^2 pairs).
        idx = np.arange(m, dtype=np.float64)
        weights = (2.0 * idx + 1.0 - m) * (2.0 / (m * m))
        return (weights[:, None] * sorted_e).sum(axis=0).astype(np.float32, copy=False)

    @staticmethod
    def spread_decomposition_per_cell(
        ensemble: np.ndarray,
        member_model_id: Sequence[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, dict] | None:
        """Per-cell decomposition of ensemble variance into
        between-checkpoint vs within-checkpoint components.

        Bundles produce 60 members as 3 checkpoints × 20 latent samples each.
        ``member_model_id`` is the per-member checkpoint label
        (``["model_0", ..., "model_2"]``). With it, total spread can be split:

        - **between**: variance across the per-checkpoint mean depths.
          Reflects disagreement *between* the trained models — closer to
          structural / epistemic uncertainty.
        - **within**: mean of the per-checkpoint variances.
          Reflects latent / aleatoric variability the model itself encodes.

        Returns ``(between_var, within_var, summary)`` or ``None`` when
        ``member_model_id`` is missing or has fewer than two distinct groups
        (no decomposition possible). Variances are population variance
        (``ddof=0``) for additivity.
        """
        if member_model_id is None:
            return None
        ids = list(member_model_id)
        if len(ids) != ensemble.shape[0]:
            return None
        unique = sorted(set(ids))
        if len(unique) < 2:
            return None
        ensemble_f = ensemble.astype(np.float64, copy=False)
        group_means: list[np.ndarray] = []
        within_vars: list[np.ndarray] = []
        for model in unique:
            mask = np.array([m == model for m in ids], dtype=bool)
            group = ensemble_f[mask]
            if group.shape[0] < 1:
                continue
            group_means.append(group.mean(axis=0))
            within_vars.append(group.var(axis=0, ddof=0))
        means_arr = np.stack(group_means, axis=0)  # [n_groups, n_cells]
        between = means_arr.var(axis=0, ddof=0).astype(np.float32, copy=False)
        within = np.stack(within_vars, axis=0).mean(axis=0).astype(np.float32, copy=False)
        # Global share-of-total: useful headline for "is uncertainty mostly
        # structural or aleatoric?" Numerically robust to all-zero domains.
        between_total = float(np.nansum(between))
        within_total = float(np.nansum(within))
        denom = max(between_total + within_total, 1e-12)
        summary = {
            "n_groups": int(len(unique)),
            "groups": list(unique),
            "between_share": between_total / denom,
            "within_share": within_total / denom,
        }
        return between, within, summary

    @staticmethod
    def checkpoint_disagreement_summary_by_time(
        *,
        members_eval: np.ndarray,
        member_model_id: object,
        area_m2: np.ndarray,
        wettable_area_m2: float,
        lead_hours: np.ndarray,
        high_share_threshold: float = 0.50,
        min_total_spread_m: float = EXTENT_THRESHOLD_M,
    ) -> Dict[str, object]:
        """Summarise checkpoint-vs-latent disagreement over the full horizon.

        The serving ensemble is structured as checkpoints × latent samples.
        This helper decomposes variance at each lead time into:

        - between-checkpoint variance: disagreement among trained checkpoints;
        - within-checkpoint variance: latent variability inside each checkpoint.

        A "high checkpoint-disagreement footprint" is the wettable area where
        checkpoint disagreement explains at least half of total ensemble
        variance and the total ensemble spread is at least 5 cm. The spread
        floor avoids treating numerically tiny differences in dry areas as a
        meaningful disagreement signal.
        """
        empty: Dict[str, object] = {
            "checkpoint_disagreement_available": False,
            "checkpoint_disagreement_groups": [],
            "checkpoint_disagreement_high_share_threshold": float(high_share_threshold),
            "checkpoint_disagreement_min_total_spread_m": float(min_total_spread_m),
            "area_weighted_total_ensemble_variance_wd_m2_by_time": [],
            "area_weighted_between_checkpoint_variance_wd_m2_by_time": [],
            "area_weighted_within_checkpoint_variance_wd_m2_by_time": [],
            "area_weighted_between_checkpoint_variance_share_by_time": [],
            "high_checkpoint_disagreement_area_fraction_wettable_by_time": [],
            "peak_area_weighted_total_ensemble_variance_wd_m2": None,
            "peak_area_weighted_total_ensemble_spread_wd_m": None,
            "peak_area_weighted_between_checkpoint_variance_wd_m2": None,
            "peak_area_weighted_between_checkpoint_spread_wd_m": None,
            "peak_area_weighted_within_checkpoint_variance_wd_m2": None,
            "peak_area_weighted_within_checkpoint_spread_wd_m": None,
            "peak_area_weighted_between_checkpoint_variance_share": None,
            "peak_between_checkpoint_disagreement_lead_hours": None,
            "peak_high_checkpoint_disagreement_area_fraction_wettable": None,
            "peak_high_checkpoint_disagreement_lead_hours": None,
        }
        if member_model_id is None or isinstance(member_model_id, (str, bytes)):
            return empty
        try:
            ids = list(member_model_id)  # type: ignore[arg-type]
        except TypeError:
            return empty
        members = np.asarray(members_eval, dtype=np.float64)
        if members.ndim != 3 or len(ids) != members.shape[0]:
            return empty
        groups = sorted(set(str(item) for item in ids))
        if len(groups) < 2:
            return empty

        group_means: list[np.ndarray] = []
        group_vars: list[np.ndarray] = []
        for group_id in groups:
            mask = np.array([str(item) == group_id for item in ids], dtype=bool)
            group = members[mask]
            if group.shape[0] < 1:
                continue
            group_means.append(group.mean(axis=0))
            group_vars.append(group.var(axis=0, ddof=0))
        if len(group_means) < 2:
            return empty

        area = np.asarray(area_m2, dtype=np.float64).reshape(-1)
        if area.shape[0] != members.shape[2]:
            return empty
        denom_area = float(wettable_area_m2)
        if denom_area <= 0.0:
            return empty

        total = members.var(axis=0, ddof=0)  # [time, cells]
        between = np.stack(group_means, axis=0).var(axis=0, ddof=0)
        within = np.stack(group_vars, axis=0).mean(axis=0)
        denom = between + within
        share = np.divide(
            between,
            denom,
            out=np.zeros_like(between, dtype=np.float64),
            where=denom > 1.0e-12,
        )

        def area_weight(values: np.ndarray) -> np.ndarray:
            return (np.asarray(values, dtype=np.float64) @ area) / denom_area

        total_by_time = area_weight(total)
        between_by_time = area_weight(between)
        within_by_time = area_weight(within)
        share_by_time = area_weight(share)
        total_spread = np.sqrt(np.maximum(total, 0.0))
        high_disagreement = (share >= float(high_share_threshold)) & (
            total_spread >= float(min_total_spread_m)
        )
        high_area_fraction_by_time = area_weight(high_disagreement.astype(np.float64))
        peak_between, peak_between_lead = ForecastProductBuilder._peak_value_and_time(
            between_by_time,
            lead_hours,
        )
        peak_high_fraction, peak_high_lead = ForecastProductBuilder._peak_value_and_time(
            high_area_fraction_by_time,
            lead_hours,
        )

        return {
            "checkpoint_disagreement_available": True,
            "checkpoint_disagreement_groups": groups,
            "checkpoint_disagreement_high_share_threshold": float(high_share_threshold),
            "checkpoint_disagreement_min_total_spread_m": float(min_total_spread_m),
            "area_weighted_total_ensemble_variance_wd_m2_by_time": [float(x) for x in total_by_time],
            "area_weighted_between_checkpoint_variance_wd_m2_by_time": [float(x) for x in between_by_time],
            "area_weighted_within_checkpoint_variance_wd_m2_by_time": [float(x) for x in within_by_time],
            "area_weighted_between_checkpoint_variance_share_by_time": [float(x) for x in share_by_time],
            "high_checkpoint_disagreement_area_fraction_wettable_by_time": [
                float(x) for x in high_area_fraction_by_time
            ],
            "peak_area_weighted_total_ensemble_variance_wd_m2": float(total_by_time.max()) if total_by_time.size else 0.0,
            "peak_area_weighted_total_ensemble_spread_wd_m": (
                float(np.sqrt(max(float(total_by_time.max()), 0.0))) if total_by_time.size else 0.0
            ),
            "peak_area_weighted_between_checkpoint_variance_wd_m2": peak_between,
            "peak_area_weighted_between_checkpoint_spread_wd_m": float(np.sqrt(max(peak_between, 0.0))),
            "peak_area_weighted_within_checkpoint_variance_wd_m2": (
                float(within_by_time.max()) if within_by_time.size else 0.0
            ),
            "peak_area_weighted_within_checkpoint_spread_wd_m": (
                float(np.sqrt(max(float(within_by_time.max()), 0.0))) if within_by_time.size else 0.0
            ),
            "peak_area_weighted_between_checkpoint_variance_share": (
                float(share_by_time.max()) if share_by_time.size else 0.0
            ),
            "peak_between_checkpoint_disagreement_lead_hours": peak_between_lead,
            "peak_high_checkpoint_disagreement_area_fraction_wettable": peak_high_fraction,
            "peak_high_checkpoint_disagreement_lead_hours": peak_high_lead,
        }

    @staticmethod
    def reliability_curves_payload(
        members: np.ndarray,
        wettable_mask: np.ndarray | None,
        lead_time_hours: np.ndarray,
        peak_time_idx: int,
        thresholds_m: Sequence[float],
        calibration_adapter: "CalibrationAdapter | None",
    ) -> dict:
        """Build the reliability JSON the Uncertainty tab renders.

        For each threshold this produces a binned mapping from raw to
        isotonic-calibrated exceedance probability across all wettable cells
        at the run's global peak time, plus per-bin counts for the two
        histograms shown above the curve. The per-bin "calibrated mean"
        line is the empirical isotonic curve **as it acted on this run** —
        easier for stakeholders to interpret than the raw bin coefficients.

        When the adapter has no isotonic curves, the mapping is the identity
        and ``applied`` is False; the chart still draws so the user sees
        "calibration was a no-op for this run".
        """
        wettable = (
            np.asarray(wettable_mask, dtype=bool)
            if wettable_mask is not None
            else np.ones(members.shape[2], dtype=bool)
        )
        members_at_peak = members[:, peak_time_idx, :]  # [n_members, n_cells]
        applied = (
            calibration_adapter is not None
            and calibration_adapter.has_isotonic_curves()
        )
        bin_edges = np.linspace(0.0, 1.0, 21)  # 20 bins
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        curves: dict[str, dict] = {}
        for thr in thresholds_m:
            thr_float = float(thr)
            raw_full = (members_at_peak > thr_float).mean(axis=0)  # [n_cells]
            raw_wet = raw_full[wettable]
            if applied:
                cal_wet = calibration_adapter.apply_isotonic_exceedance(
                    raw_wet,
                    threshold_m=thr_float,
                    lead_time_hour=float(lead_time_hours[peak_time_idx]),
                    wettable_mask=wettable,
                )
            else:
                cal_wet = raw_wet.copy()
            # Bin raw probabilities, compute mean calibrated probability per bin.
            digitized = np.clip(np.digitize(raw_wet, bin_edges) - 1, 0, len(bin_centers) - 1)
            cal_means: list[float | None] = []
            counts_raw: list[int] = []
            counts_cal: list[int] = []
            for b in range(len(bin_centers)):
                mask = digitized == b
                if mask.any():
                    cal_means.append(float(cal_wet[mask].mean()))
                else:
                    cal_means.append(None)
                counts_raw.append(int(mask.sum()))
            cal_digitized = np.clip(np.digitize(cal_wet, bin_edges) - 1, 0, len(bin_centers) - 1)
            for b in range(len(bin_centers)):
                counts_cal.append(int((cal_digitized == b).sum()))
            curves[f"{thr_float:g}"] = {
                "threshold_m": thr_float,
                "raw_bin_centers": [float(c) for c in bin_centers],
                "calibrated_means_per_bin": cal_means,
                "raw_distribution_counts": counts_raw,
                "calibrated_distribution_counts": counts_cal,
                "raw_mean": float(raw_wet.mean()) if raw_wet.size else 0.0,
                "calibrated_mean": float(cal_wet.mean()) if cal_wet.size else 0.0,
            }
        return {
            "applied": bool(applied),
            "n_wettable_cells": int(wettable.sum()),
            "peak_time_idx": int(peak_time_idx),
            "peak_lead_hours": float(lead_time_hours[peak_time_idx]),
            "thresholds_m": [float(t) for t in thresholds_m],
            "bin_edges": [float(e) for e in bin_edges],
            "curves": curves,
        }

    @staticmethod
    def _simple_histogram_svg(
        *,
        title: str,
        x_label: str,
        y_label: str,
        bin_edges: Sequence[float],
        counts: Sequence[int],
        color: str = "#0f766e",
    ) -> str:
        width, height = 760, 360
        left, right, top, bottom = 64, 24, 42, 58
        inner_w = width - left - right
        inner_h = height - top - bottom
        max_count = max([int(x) for x in counts] + [1])
        n = max(1, len(counts))
        bar_w = inner_w / n
        bars = []
        for idx, count in enumerate(counts):
            h = (float(count) / max_count) * inner_h
            x = left + idx * bar_w
            y = top + inner_h - h
            bars.append(
                f'<rect x="{x + 1:.2f}" y="{y:.2f}" width="{max(1.0, bar_w - 2):.2f}" '
                f'height="{max(0.0, h):.2f}" fill="{color}" opacity="0.82" />'
            )
        x_min = float(bin_edges[0]) if len(bin_edges) else 0.0
        x_max = float(bin_edges[-1]) if len(bin_edges) else 1.0
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
                '<rect width="100%" height="100%" fill="#ffffff" />',
                f'<text x="{left}" y="24" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{title}</text>',
                f'<rect x="{left}" y="{top}" width="{inner_w}" height="{inner_h}" fill="#f8fafc" stroke="#dbe3ea" />',
                *bars,
                f'<line x1="{left}" y1="{top + inner_h}" x2="{left + inner_w}" y2="{top + inner_h}" stroke="#334155" />',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_h}" stroke="#334155" />',
                f'<text x="{left}" y="{height - 24}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#475569">{x_min:.2f}</text>',
                f'<text x="{left + inner_w}" y="{height - 24}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="#475569">{x_max:.2f}</text>',
                f'<text x="{left + inner_w / 2}" y="{height - 12}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">{x_label}</text>',
                f'<text x="18" y="{top + inner_h / 2}" transform="rotate(-90 18 {top + inner_h / 2})" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">{y_label}</text>',
                f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="11" fill="#64748b">{max_count}</text>',
                "</svg>",
            ]
        )

    def write_spread_to_peak_histogram(
        self,
        forecast: ForecastResult,
        *,
        output_dir: str | Path,
        max_ratio: float = 2.0,
        bins: int = 24,
    ) -> dict[str, Path]:
        """Write the relative uncertainty diagnostic for the Uncertainty tab.

        The ratio is computed per wettable cell at that cell's local
        ensemble-mean peak: ``std_at_peak / mean_depth_at_peak``. It is
        ground-truth-free and communicates where ensemble disagreement is
        large relative to the forecast signal itself.
        """
        forecast = forecast.validate()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        n_members, n_time, n_cells = members.shape
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else np.ones(n_cells, dtype=bool)
        )
        mean_tc = members.mean(axis=0)
        std_tc = members.std(axis=0)
        local_peak_idx = mean_tc.argmax(axis=0)
        cells = np.arange(n_cells)
        peak_mean = mean_tc[local_peak_idx, cells]
        peak_std = std_tc[local_peak_idx, cells]
        valid = wettable & np.isfinite(peak_mean) & np.isfinite(peak_std) & (peak_mean > 1.0e-6)
        ratios = peak_std[valid] / peak_mean[valid]
        bin_edges = np.linspace(0.0, float(max_ratio), int(bins) + 1)
        clipped = np.clip(ratios, 0.0, float(max_ratio))
        counts, _ = np.histogram(clipped, bins=bin_edges)
        payload = {
            "metric": "spread_to_peak_ratio",
            "description": "Per wettable cell std_at_local_peak / ensemble_mean_depth_at_local_peak.",
            "n_members": int(n_members),
            "n_time": int(n_time),
            "n_wettable_cells": int(wettable.sum()),
            "n_signal_cells": int(valid.sum()),
            "max_ratio": float(max_ratio),
            "bin_edges": [float(x) for x in bin_edges],
            "counts": [int(x) for x in counts],
            "mean_ratio": float(ratios.mean()) if ratios.size else None,
            "median_ratio": float(np.median(ratios)) if ratios.size else None,
            "p95_ratio": float(np.quantile(ratios, 0.95)) if ratios.size else None,
        }
        json_path = out_dir / "spread_to_peak_histogram.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        svg_path = out_dir / "spread_to_peak_histogram.svg"
        svg_path.write_text(
            self._simple_histogram_svg(
                title="Spread-to-peak ratio across wettable cells",
                x_label="std at local peak / mean depth at local peak",
                y_label="Wettable cell count",
                bin_edges=bin_edges,
                counts=counts,
                color="#7c3aed",
            )
        )
        return {json_path.name: json_path, svg_path.name: svg_path}

    def build_cell_contribution_leaderboard(
        self,
        forecast: ForecastResult,
        *,
        threshold_m: float = 0.30,
        top_n: int = 40,
        calibration_adapter: "CalibrationAdapter | None" = None,
    ) -> dict:
        """Rank cells by expected area contribution to exceedance footprint."""
        forecast = forecast.validate()
        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        n_members, n_time, n_cells = members.shape
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else np.ones(n_cells, dtype=bool)
        )
        area = self._wettable_area(forecast, None)
        lead = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        probability = self._probability_by_time(
            members,
            threshold_m=float(threshold_m),
            lead_hours=lead,
            wettable_mask=wettable,
            calibration_adapter=calibration_adapter,
        )
        mean = members.mean(axis=0)
        expected_area = probability * area[None, :]
        expected_area[:, ~wettable] = 0.0
        contribution_m2 = expected_area.max(axis=0)
        total = float(contribution_m2.sum())
        peak_idx = expected_area.argmax(axis=0)
        peak_prob = probability[peak_idx, np.arange(n_cells)]
        peak_depth = mean[peak_idx, np.arange(n_cells)]
        geometry = (forecast.metadata or {}).get("geometry_xy")
        xy = np.asarray(geometry, dtype=np.float64) if geometry is not None else None
        order = np.argsort(contribution_m2)[::-1]
        rows = []
        for idx in order:
            if len(rows) >= int(top_n):
                break
            if not wettable[idx] or contribution_m2[idx] <= 0.0:
                continue
            row = {
                "cell_index": int(idx),
                "peak_expected_area_m2": float(contribution_m2[idx]),
                "contribution_share": float(contribution_m2[idx] / total) if total > 0 else 0.0,
                "peak_probability": float(peak_prob[idx]),
                "peak_mean_wd_m": float(peak_depth[idx]),
                "peak_lead_hours": float(lead[int(peak_idx[idx])]) if lead.size else None,
                "cell_area_m2": float(area[idx]),
            }
            if xy is not None and xy.ndim == 2 and xy.shape[0] == n_cells:
                row["x"] = float(xy[idx, 0])
                row["y"] = float(xy[idx, 1])
            rows.append(row)
        return {
            "threshold_m": float(threshold_m),
            "n_members": int(n_members),
            "n_time": int(n_time),
            "n_cells": int(n_cells),
            "n_wettable_cells": int(wettable.sum()),
            "total_peak_expected_area_m2": total,
            "total_peak_expected_area_km2": total / 1_000_000.0,
            "rows": rows,
        }

    def write_uncertainty_diagnostics(
        self,
        forecast: ForecastResult,
        *,
        output_dir: str | Path,
        calibration_adapter: "CalibrationAdapter | None" = None,
        thresholds_m: Sequence[float] = (0.05, 0.30),
        render_pngs: bool = False,
    ) -> dict[str, Path]:
        """Compute and persist the Uncertainty-tab diagnostics:

        - ``empirical_crps_map.npy`` — per-cell mean pairwise abs difference
          across the ensemble at the global peak time. Ground-truth-free
          spread map. Optionally rendered as PNG.
        - ``spread_decomposition.npz`` — ``between_var`` + ``within_var``
          per cell, computed from the bundle's checkpoint × latent layout.
          Plus paired PNG renderings when ``render_pngs=True``.
        - ``spread_decomposition_summary.json`` — global between/within
          shares for the headline.
        - ``reliability_curves.json`` — raw → isotonic-calibrated binned
          mapping per threshold for the reliability card.
        """
        forecast = forecast.validate()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        lead = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        n_members, n_time, n_cells = members.shape
        if n_time < 1 or n_cells < 1:
            return {}
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else np.ones(n_cells, dtype=bool)
        )
        # Pick the global peak time as the focal slice for these diagnostics:
        # it's the moment that defines the headline, so spread reported here
        # answers "how confident are we *at peak*?"
        mean_tc = members.mean(axis=0)
        peak_t = int(mean_tc.max(axis=1).argmax())

        written: dict[str, Path] = {}

        # 1. Empirical CRPS map (peak time slice).
        crps = self.empirical_crps_per_cell(members[:, peak_t, :])
        crps_masked = crps.copy()
        crps_masked[~wettable] = np.nan
        crps_path = out_dir / "empirical_crps_map.npy"
        np.save(crps_path, crps_masked.astype(np.float32, copy=False))
        written["empirical_crps_map.npy"] = crps_path

        # 2. Spread decomposition.
        decomp = None
        metadata = forecast.metadata or {}
        member_model_id = metadata.get("member_model_id")
        if member_model_id is not None:
            decomp = self.spread_decomposition_per_cell(
                members[:, peak_t, :], member_model_id
            )
        if decomp is not None:
            between_var, within_var, summary = decomp
            between_masked = between_var.copy()
            between_masked[~wettable] = np.nan
            within_masked = within_var.copy()
            within_masked[~wettable] = np.nan
            decomp_path = out_dir / "spread_decomposition.npz"
            np.savez(decomp_path, between_var=between_masked, within_var=within_masked)
            written["spread_decomposition.npz"] = decomp_path
            summary_path = out_dir / "spread_decomposition_summary.json"
            summary_path.write_text(json.dumps(summary))
            written["spread_decomposition_summary.json"] = summary_path

        # 3. Reliability curves.
        reliability = self.reliability_curves_payload(
            members=members,
            wettable_mask=wettable,
            lead_time_hours=lead,
            peak_time_idx=peak_t,
            thresholds_m=thresholds_m,
            calibration_adapter=calibration_adapter,
        )
        reliability_path = out_dir / "reliability_curves.json"
        reliability_path.write_text(json.dumps(reliability))
        written["reliability_curves.json"] = reliability_path

        written.update(
            self.write_spread_to_peak_histogram(
                forecast,
                output_dir=out_dir,
            )
        )

        if render_pngs:
            self._render_uncertainty_pngs(
                forecast=forecast,
                arrays_path=written,
                output_dir=out_dir,
                wettable=wettable,
            )

        return written

    def _render_uncertainty_pngs(
        self,
        *,
        forecast: ForecastResult,
        arrays_path: "dict[str, Path]",
        output_dir: Path,
        wettable: np.ndarray,
    ) -> None:
        """Render the three uncertainty maps with a consistent palette so
        the Uncertainty tab reads at a glance: empirical CRPS in magma
        (uncertainty palette), between in cividis (model variability),
        within in viridis (latent variability)."""
        metadata = forecast.metadata or {}
        geometry = metadata.get("geometry_xy")
        if geometry is None:
            return
        xy = np.asarray(geometry, dtype=np.float64)
        elevation_raw = metadata.get("elevation_raw")
        visualization_config = metadata.get("visualization_config")
        from neuralop.flood.serving.map_rendering import write_rivanna_style_map_png

        def _render(npy_name: str, *, title: str, cbar: str, cmap: str) -> None:
            path = arrays_path.get(npy_name)
            if path is None:
                return
            data = np.load(path) if path.suffix == ".npy" else None
            if data is None:
                return
            values = np.where(np.isfinite(data), data, 0.0)
            png_name = npy_name.replace(".npy", ".png")
            write_rivanna_style_map_png(
                values=values,
                geometry_xy=xy,
                output_path=output_dir / png_name,
                title=title,
                colorbar_label=cbar,
                cmap=cmap,
                vmin=0.0,
                vmax=None,
                elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
                is_wd_depth=False,
                zero_transparent=True,
                visualization_config=visualization_config,
                dpi=180,
            )

        _render(
            "empirical_crps_map.npy",
            title="Empirical CRPS at peak time (intra-ensemble spread)",
            cbar="Empirical CRPS (m)",
            cmap="magma",
        )
        # spread_decomposition.npz isn't a single .npy, render its halves directly.
        decomp_path = arrays_path.get("spread_decomposition.npz")
        if decomp_path is not None:
            with np.load(decomp_path) as data:
                for key, cmap, title, cbar, png_name in (
                    ("between_var", "cividis", "Between-checkpoint variance at peak",
                     "Var across checkpoint means (m²)", "between_var_map.png"),
                    ("within_var", "viridis", "Within-checkpoint (latent) variance at peak",
                     "Mean of per-checkpoint variances (m²)", "within_var_map.png"),
                ):
                    arr = np.asarray(data[key])
                    values = np.where(np.isfinite(arr), arr, 0.0)
                    write_rivanna_style_map_png(
                        values=values,
                        geometry_xy=xy,
                        output_path=output_dir / png_name,
                        title=title,
                        colorbar_label=cbar,
                        cmap=cmap,
                        vmin=0.0,
                        vmax=None,
                        elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
                        is_wd_depth=False,
                        zero_transparent=True,
                        visualization_config=visualization_config,
                        dpi=180,
                    )

    @staticmethod
    def summarize_cell_timeseries(
        *,
        h5_bytes: bytes,
        cell_index: int,
        thresholds_m: Sequence[float] = (0.05, 0.30),
        calibration_adapter: "CalibrationAdapter | None" = None,
    ) -> dict:
        """Read one cell's full ensemble trace from a ``forecast_members.h5``
        artifact and compute the panel of statistics the per-cell inspector
        renders.

        The HDF5 file is gzip-compressed and ~5-20 MB for a typical run; for
        an interactive inspector we open it from a ``BytesIO`` (so the
        artifact store stays the source of truth) and slice the third axis
        (cells) at the requested index. The slice is tiny — 60 members x
        n_time floats — so the wire payload is small and we can return raw
        traces plus derived stats without paginating.

        Returns a JSON-friendly dict with::

            {
              cell_index, n_members, n_time,
              lead_time_hours: [...],
              raw_members_wd:        [[60][n_time]],
              calibrated_members_wd: [[60][n_time]],
              calibrated_mean_wd:    [n_time],   # convenience
              calibrated_q05_wd:     [n_time],
              calibrated_q50_wd:     [n_time],
              calibrated_q95_wd:     [n_time],
              calibrated_exceedance_prob: {        # P(WD > thr) at each time
                  "0.05": [n_time],
                  "0.30": [n_time],
              },
              calibrated_member_arrival_hours: {    # first-wet per member
                  "0.05": [60],   # NaN where the member never crosses
                  "0.30": [60],
              },
              peak_calibrated_mean_wd_m: float,
              peak_calibrated_lead_hours: float,
              wettable: bool,
            }

        Raises ``KeyError`` if either members dataset is missing.
        Raises ``IndexError`` if ``cell_index`` is out of bounds.
        """
        import io

        try:
            import h5py
        except Exception as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "Per-cell inspection requires h5py in the serving worker environment."
            ) from exc

        cell_index = int(cell_index)
        with h5py.File(io.BytesIO(h5_bytes), "r") as h5:
            if "raw_members_wd" not in h5 or "calibrated_members_wd" not in h5:
                raise KeyError("forecast_members.h5 is missing raw or calibrated ensemble datasets.")
            n_members, n_time, n_cells = h5["calibrated_members_wd"].shape
            if cell_index < 0 or cell_index >= int(n_cells):
                raise IndexError(f"cell_index {cell_index} out of range [0, {n_cells}).")
            # Single-cell slice — small and fast.
            raw_cell = np.asarray(h5["raw_members_wd"][:, :, cell_index], dtype=np.float32)
            cal_cell = np.asarray(h5["calibrated_members_wd"][:, :, cell_index], dtype=np.float32)
            lead = np.asarray(h5["lead_time_hours"][...], dtype=np.float32)
            wettable_mask = np.asarray(h5["wettable_mask"][...], dtype=bool)
            wettable = bool(wettable_mask[cell_index]) if wettable_mask.shape[0] > cell_index else True

        # Numerical clipping mirrors ForecastResult.validate's contract: depth
        # is non-negative even when calibration returns small negatives.
        cal_cell = np.clip(cal_cell, 0.0, None)
        raw_cell = np.clip(raw_cell, 0.0, None)

        mean_t = cal_cell.mean(axis=0)
        q05 = np.quantile(cal_cell, 0.05, axis=0)
        q50 = np.quantile(cal_cell, 0.50, axis=0)
        q95 = np.quantile(cal_cell, 0.95, axis=0)
        peak_idx = int(np.argmax(mean_t))
        peak_mean = float(mean_t[peak_idx])
        peak_lead = float(lead[peak_idx]) if lead.shape[0] > peak_idx else 0.0

        exceedance: dict[str, list[float]] = {}
        arrival: dict[str, list[float | None]] = {}
        for thr in thresholds_m:
            thr_float = float(thr)
            thr_key = f"{thr_float:g}"
            above = cal_cell > thr_float  # [n_members, n_time]
            raw_probability = above.mean(axis=0).astype(np.float64)
            if calibration_adapter is not None and calibration_adapter.has_isotonic_curves():
                probability = calibration_adapter.apply_isotonic_exceedance_for_cell(
                    raw_probability,
                    threshold_m=thr_float,
                    lead_time_hour=lead,
                    cell_index=cell_index,
                    wettable_mask=wettable_mask,
                )
            else:
                probability = raw_probability
            exceedance[thr_key] = np.clip(probability, 0.0, 1.0).astype(np.float32).tolist()
            # Per-member arrival time: first lead-hour where that member exceeds
            # the threshold. NaN if the member never crosses.
            ever = above.any(axis=1)
            first_idx = above.argmax(axis=1)
            member_arrivals: list[float | None] = []
            for m in range(int(n_members)):
                if ever[m]:
                    member_arrivals.append(float(lead[first_idx[m]]))
                else:
                    member_arrivals.append(None)
            arrival[thr_key] = member_arrivals

        return {
            "cell_index": cell_index,
            "n_members": int(n_members),
            "n_time": int(n_time),
            "lead_time_hours": lead.astype(np.float32).tolist(),
            "raw_members_wd": raw_cell.tolist(),
            "calibrated_members_wd": cal_cell.tolist(),
            "calibrated_mean_wd": mean_t.astype(np.float32).tolist(),
            "calibrated_q05_wd": q05.astype(np.float32).tolist(),
            "calibrated_q50_wd": q50.astype(np.float32).tolist(),
            "calibrated_q95_wd": q95.astype(np.float32).tolist(),
            "calibrated_exceedance_prob": exceedance,
            "calibrated_member_arrival_hours": arrival,
            "peak_calibrated_mean_wd_m": peak_mean,
            "peak_calibrated_lead_hours": peak_lead,
            "wettable": wettable,
        }

    def build_geometry_meta(self, forecast: ForecastResult) -> dict | None:
        """Compact spatial sidecar for the click-to-inspect overlay.

        The web inspector positions an SVG hit-target layer over the rendered
        Hazard map and snaps clicks to the nearest cell. To do that it needs
        the UTM coordinates of every cell plus the data bounds — but NOT the
        full ensemble, normalizer, or anything else from the model bundle.
        This builds the minimum payload (cells xy + extent + wettable flag)
        as plain Python so it serializes cleanly through ``put_json``.

        Returns None if geometry_xy isn't attached to the forecast — caller
        treats that as "no inspector this run" rather than failing the run.
        """
        forecast = forecast.validate()
        metadata = forecast.metadata or {}
        geometry = metadata.get("geometry_xy")
        if geometry is None:
            return None
        xy = np.asarray(geometry, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != forecast.members_wd.shape[2]:
            return None
        n_cells = int(xy.shape[0])
        x = xy[:, 0]
        y = xy[:, 1]
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else np.ones(n_cells, dtype=bool)
        )
        payload = {
            "n_cells": n_cells,
            "bounds": {
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
            },
            # Round to one decimetre — UTM units are metres, so 0.1 m is a
            # generous floor on the cell-position precision and trims JSON
            # size by ~30% versus dumping the full float64.
            "x": [round(float(v), 1) for v in x],
            "y": [round(float(v), 1) for v in y],
            "wettable": [bool(b) for b in wettable.tolist()],
        }
        try:
            from neuralop.flood.serving.map_rendering import compute_rivanna_style_data_viewport

            elevation_raw = metadata.get("elevation_raw")
            viewport = compute_rivanna_style_data_viewport(
                geometry_xy=xy,
                elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
                visualization_config=metadata.get("visualization_config"),
            )
            data_bounds = viewport.pop("data_bounds", None)
            payload["image_data_viewport"] = viewport
            if data_bounds is not None:
                payload["image_data_bounds"] = data_bounds
        except Exception:
            # Older/minimal environments may lack the optional Matplotlib stack.
            # The frontend falls back to the full-image transform, but new
            # production runs should carry this field for accurate hit testing.
            pass
        for metadata_key, payload_key in (
            ("elevation_raw", "elevation_m"),
            ("slope_raw", "slope"),
            ("slope", "slope"),
            ("flow_accumulation_raw", "flow_accumulation"),
            ("flow_accumulation", "flow_accumulation"),
        ):
            values = metadata.get(metadata_key)
            if values is None or payload_key in payload:
                continue
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if arr.shape[0] == n_cells:
                payload[payload_key] = [
                    (round(float(v), 6) if np.isfinite(v) else None) for v in arr
                ]
        return payload

    def write_envelope_maps(
        self,
        forecast: ForecastResult,
        *,
        output_dir: str | Path,
        thresholds_m: Sequence[float] = (0.05, 0.30),
        render_pngs: bool = False,
    ) -> dict[str, Path]:
        """Compute and persist per-cell envelope (temporal-summary) maps as float32 .npy files.

        These four single-frame summaries are the spine of the new Hazard tab.
        Unlike the time-indexed scrub frames they collapse the time dimension,
        so the user sees the *envelope* of the forecast in one glance:

        - ``peak_depth_map.npy`` — per-cell maximum of the ensemble-mean depth.
          The single most important "where will it be deep?" map.
        - ``arrival_time_map_gt_{thr}m.npy`` — for each cell, the lead time at
          which the ensemble-mean depth first exceeds ``thr``. NaN where the
          cell never wets above the threshold; values are in hours.
        - ``duration_map_gt_{thr}m.npy`` — total hours during the forecast for
          which the ensemble-mean depth exceeds ``thr``.
        - ``quantile_envelope_at_peak.npy`` — at each cell's local peak time
          (argmax of ensemble-mean depth over time), the width of the
          ensemble 5%-95% envelope. Communicates *where uncertainty is large
          at the moment that cell matters*, which is more interpretable than
          the global-peak spread map.

        Structurally-dry cells are NaN in every array so the renderer can mask
        them cleanly. The wettable mask is folded into the contract here so
        downstream callers don't have to re-apply it.

        Threshold strings use the ``0p05`` convention shared with the static
        scrub-frame names (``calibrated_p_gt_0p30m_t001.png``).
        """
        forecast = forecast.validate()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        lead = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        n_members, n_time, n_cells = members.shape
        if n_time < 1 or n_cells < 1:
            return {}
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else np.ones(n_cells, dtype=bool)
        )
        dt_hours = float(lead[0]) if n_time >= 1 and lead[0] > 0 else (
            float(lead[1] - lead[0]) if n_time >= 2 else 1.0
        )

        mean_tc = members.mean(axis=0)  # [n_time, n_cells]
        peak_depth = mean_tc.max(axis=0).astype(np.float32, copy=False)  # [n_cells]
        peak_depth[~wettable] = np.nan

        # Per-cell argmax(time) of the ensemble mean — the "local peak time".
        peak_time_idx = mean_tc.argmax(axis=0)  # [n_cells]
        cell_idx = np.arange(n_cells)
        peak_members = members[:, peak_time_idx, cell_idx]  # [n_members, n_cells]
        q05 = np.quantile(peak_members, 0.05, axis=0)
        q95 = np.quantile(peak_members, 0.95, axis=0)
        envelope_width = (q95 - q05).astype(np.float32, copy=False)
        envelope_width[~wettable] = np.nan

        written: dict[str, Path] = {}

        def _save(name: str, array: np.ndarray) -> None:
            path = out_dir / name
            np.save(path, np.asarray(array, dtype=np.float32))
            written[name] = path

        _save("peak_depth_map.npy", peak_depth)
        _save("quantile_envelope_at_peak.npy", envelope_width)

        for thr in thresholds_m:
            thr_label = f"{float(thr):g}".replace(".", "p")  # 0.05 -> "0p05"
            above = mean_tc > float(thr)  # [n_time, n_cells]
            ever_wet = above.any(axis=0)  # [n_cells]

            # Arrival time: lead hour of the first crossing.
            first_idx = above.argmax(axis=0)  # 0 if never crossed (masked below)
            arrival = lead[first_idx].astype(np.float32, copy=False).copy()
            arrival[~ever_wet] = np.nan
            arrival[~wettable] = np.nan
            _save(f"arrival_time_map_gt_{thr_label}m.npy", arrival)

            # Duration: integer count of timesteps above threshold × dt.
            duration = (above.sum(axis=0).astype(np.float32, copy=False) * dt_hours)
            duration[~wettable] = np.nan
            _save(f"duration_map_gt_{thr_label}m.npy", duration)

        if render_pngs:
            # Side-products: PNG renderings of the .npy arrays so the Hazard
            # tab can <img src=...> them directly without a client-side numpy
            # decoder. Kept gated so tests can skip the slow matplotlib pass.
            self._render_envelope_pngs(
                forecast=forecast,
                arrays=written,
                output_dir=out_dir,
                thresholds_m=thresholds_m,
                total_horizon_h=float(lead[-1]) if n_time >= 1 else 0.0,
                wettable=wettable,
            )

        return written

    def _render_envelope_pngs(
        self,
        *,
        forecast: ForecastResult,
        arrays: "dict[str, Path]",
        output_dir: Path,
        thresholds_m: Sequence[float],
        total_horizon_h: float,
        wettable: np.ndarray,
    ) -> None:
        """Render each envelope .npy as a Rivanna-styled PNG side-product.

        Co-located with ``write_envelope_maps`` so naming and metadata stay in
        lockstep — every ``foo.npy`` produces a ``foo.png``. PNGs use the same
        DEM-context renderer as the static map gallery for visual consistency.
        """
        metadata = forecast.metadata or {}
        geometry = metadata.get("geometry_xy")
        if geometry is None:
            return
        xy = np.asarray(geometry, dtype=np.float64)
        elevation_raw = metadata.get("elevation_raw")
        visualization_config = metadata.get("visualization_config")
        from neuralop.flood.serving.map_rendering import write_rivanna_style_map_png

        def _render(npy_name: str, *, title: str, cbar: str, cmap: str, vmin: float, vmax: float | None,
                    is_wd_depth: bool = False, zero_transparent: bool = True) -> None:
            data = np.load(arrays[npy_name])
            # write_rivanna_style_map_png handles NaN cells via its zero_transparent
            # path when we feed zeros in — but here we keep NaN to communicate
            # "no data" explicitly. Replace NaN with -inf-like sentinel so the
            # cmap treats them as masked.
            values = np.where(np.isfinite(data), data, 0.0)
            png_name = npy_name.replace(".npy", ".png")
            write_rivanna_style_map_png(
                values=values,
                geometry_xy=xy,
                output_path=output_dir / png_name,
                title=title,
                colorbar_label=cbar,
                cmap=cmap,
                vmin=float(vmin),
                vmax=vmax,
                elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
                is_wd_depth=is_wd_depth,
                zero_transparent=zero_transparent,
                visualization_config=visualization_config,
                dpi=180,
            )

        _render(
            "peak_depth_map.npy",
            title="Peak ensemble-mean depth (envelope)",
            cbar="Peak WD (m)",
            cmap="viridis",
            vmin=0.0,
            vmax=None,
            is_wd_depth=True,
            zero_transparent=False,
        )
        _render(
            "quantile_envelope_at_peak.npy",
            title="Envelope width (q95 – q05) at local peak time",
            cbar="Spread at peak (m)",
            cmap="magma",
            vmin=0.0,
            vmax=None,
            zero_transparent=True,
        )
        horizon = max(total_horizon_h, 1e-3)
        for thr in thresholds_m:
            thr_label = f"{float(thr):g}".replace(".", "p")
            _render(
                f"arrival_time_map_gt_{thr_label}m.npy",
                title=f"Arrival time at WD > {float(thr):.2f} m",
                cbar="Lead time at first wetting (h)",
                cmap="cividis",
                vmin=0.0,
                vmax=horizon,
                zero_transparent=True,
            )
            _render(
                f"duration_map_gt_{thr_label}m.npy",
                title=f"Duration at WD > {float(thr):.2f} m",
                cbar="Wet duration (h)",
                cmap="viridis",
                vmin=0.0,
                vmax=horizon,
                zero_transparent=True,
            )

    # Product key contract — keep in lockstep with the frontend's
    # `buildScrubFrames(product=...)` regex matcher. Adding a product here
    # requires a frontend update; removing one orphans existing artifacts.
    SCRUB_PRODUCT_MEAN = "mean"
    SCRUB_PRODUCT_SPREAD = "spread"
    SCRUB_PRODUCT_P95 = "p95"
    SCRUB_PRODUCT_IQR = "iqr"
    SCRUB_PRODUCT_P_GT_0P30M = "p_gt_0p30m"
    SCRUB_PRODUCTS = (
        SCRUB_PRODUCT_MEAN,
        SCRUB_PRODUCT_SPREAD,
        SCRUB_PRODUCT_P95,
        SCRUB_PRODUCT_IQR,
        SCRUB_PRODUCT_P_GT_0P30M,
    )

    def write_compare_delta_maps_from_hdf5(
        self,
        *,
        run_a_h5_bytes: bytes,
        run_b_h5_bytes: bytes,
        geometry_meta: dict,
        output_dir: str | Path,
        label: str,
        threshold_m: float = 0.30,
        max_frames: int = 120,
    ) -> list[Path]:
        """Render B-minus-A delta maps for Compare mode from ensemble HDF5s."""
        import io

        try:
            import h5py
        except Exception as exc:  # pragma: no cover - environment guard
            raise RuntimeError("Compare delta maps require h5py.") from exc
        x = np.asarray(geometry_meta.get("x"), dtype=np.float64)
        y = np.asarray(geometry_meta.get("y"), dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("geometry_meta must contain x and y arrays of equal length.")
        xy = np.stack([x, y], axis=1)
        elevation = geometry_meta.get("elevation_m")
        elevation_raw = (
            np.asarray(elevation, dtype=np.float64)
            if isinstance(elevation, list) and len(elevation) == xy.shape[0]
            else None
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        from neuralop.flood.serving.map_rendering import write_rivanna_style_map_png

        with h5py.File(io.BytesIO(run_a_h5_bytes), "r") as a_h5, h5py.File(io.BytesIO(run_b_h5_bytes), "r") as b_h5:
            a_ds = a_h5["calibrated_members_wd"]
            b_ds = b_h5["calibrated_members_wd"]
            n_time = min(int(a_ds.shape[1]), int(b_ds.shape[1]))
            n_cells = min(int(a_ds.shape[2]), int(b_ds.shape[2]), int(xy.shape[0]))
            if n_time <= 0 or n_cells <= 0:
                return []
            if n_time > int(max_frames):
                time_indices = sorted(set(np.linspace(0, n_time - 1, int(max_frames), dtype=int).tolist()))
            else:
                time_indices = list(range(n_time))
            for t_idx in time_indices:
                a = np.clip(np.asarray(a_ds[:, t_idx, :n_cells], dtype=np.float64), 0.0, None)
                b = np.clip(np.asarray(b_ds[:, t_idx, :n_cells], dtype=np.float64), 0.0, None)
                deltas = {
                    "mean": (b.mean(axis=0) - a.mean(axis=0), "Mean WD delta (B - A)", "Δ mean WD (m)"),
                    "spread": (b.std(axis=0) - a.std(axis=0), "Spread delta (B - A)", "Δ spread (m)"),
                    "p_gt_0p30m": (
                        100.0 * ((b > float(threshold_m)).mean(axis=0) - (a > float(threshold_m)).mean(axis=0)),
                        f"P(WD > {threshold_m:.2f} m) delta (B - A)",
                        "Δ probability (percentage points)",
                    ),
                }
                for product, (values, title, cbar) in deltas.items():
                    vmax = float(np.nanmax(np.abs(values))) if values.size else 0.0
                    if not np.isfinite(vmax) or vmax <= 0.0:
                        vmax = 1.0
                    path = write_rivanna_style_map_png(
                        values=values,
                        geometry_xy=xy[:n_cells],
                        output_path=out_dir / f"compare_{label}_{product}_delta_t{t_idx + 1:03d}.png",
                        title=f"{title} | lead slot {t_idx + 1}",
                        colorbar_label=cbar,
                        cmap="coolwarm",
                        vmin=-vmax,
                        vmax=vmax,
                        elevation_raw=elevation_raw[:n_cells] if elevation_raw is not None else None,
                        is_wd_depth=False,
                        zero_transparent=False,
                        dpi=180,
                    )
                    written.append(path)
        return written

    def write_scrub_frames(
        self,
        forecast: ForecastResult,
        *,
        output_dir: str | Path,
        label: str = "calibrated",
        product: str = "mean",
        threshold_m: float = 0.30,
        calibration_adapter: "CalibrationAdapter | None" = None,
        dpi: int = 180,
    ) -> list[Path]:
        """Render one per-timestep PNG per requested product for the scrubber.

        The web Time Player switches its source frames via this contract:
        for any supported ``product`` it discovers ``{label}_{product}_scrub_t{NNN}.png``.
        One vmax/colorbar is shared across the time sequence so the user sees
        the *quantity* changing, not the legend shifting under their feet.

        Products:
        - ``mean`` — ensemble-mean depth per cell, ``viridis`` cmap.
        - ``spread`` — ensemble standard deviation per cell, ``spread_violet``
          cmap (uncertainty palette already registered by eval.render).
        - ``p_gt_0p30m`` — calibrated probability of exceeding ``threshold_m``
          (default 0.30 m). When ``calibration_adapter`` provides isotonic
          curves, raw frequencies are corrected per the same per-lead-time
          binned mapping used by ``write_map_pngs``; otherwise raw mean
          indicator is returned. Range fixed at [0, 1].

        DPI 180 keeps the result pane crisp at a manageable per-frame size.
        Returns the written paths in time order. The forecast must carry
        ``geometry_xy`` metadata; otherwise this is a no-op.
        """
        if product not in self.SCRUB_PRODUCTS:
            raise ValueError(
                f"Unsupported scrub product {product!r}; expected one of {self.SCRUB_PRODUCTS}."
            )
        forecast = forecast.validate()
        metadata = forecast.metadata or {}
        geometry = metadata.get("geometry_xy")
        if geometry is None:
            return []
        xy = np.asarray(geometry, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != forecast.members_wd.shape[2]:
            raise ValueError("forecast.metadata['geometry_xy'] must have shape [n_cells,2].")
        from neuralop.flood.eval import render as eval_render
        from neuralop.flood.serving.map_rendering import write_rivanna_style_map_png

        members = np.clip(np.asarray(forecast.members_wd, dtype=np.float64), 0.0, None)
        lead = np.asarray(forecast.lead_time_hours, dtype=np.float64)
        n_time = members.shape[1]
        if n_time <= 0:
            return []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        elevation_raw = metadata.get("elevation_raw")
        visualization_config = metadata.get("visualization_config")
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else None
        )

        # Pre-compute the per-timestep value tensor + per-product render config.
        if product == self.SCRUB_PRODUCT_MEAN:
            value_by_time = members.mean(axis=0)  # [n_time, n_cells]
            vmax = eval_render._wd_spatial_vmax(value_by_time)
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            render_cfg = {
                "title_prefix": "mean WD",
                "colorbar_label": "Mean WD (m)",
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": float(vmax),
                "is_wd_depth": True,
                "zero_transparent": False,
            }
        elif product == self.SCRUB_PRODUCT_SPREAD:
            value_by_time = members.std(axis=0)  # [n_time, n_cells]
            vmax = float(eval_render._robust_nonnegative_vmax(value_by_time))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            render_cfg = {
                "title_prefix": "ensemble spread",
                "colorbar_label": "Spread (m)",
                "cmap": "spread_violet",
                "vmin": 0.0,
                "vmax": vmax,
                "is_wd_depth": False,
                "zero_transparent": True,
            }
        elif product == self.SCRUB_PRODUCT_P95:
            # 95th percentile of depth across the ensemble at each (time, cell).
            # Same depth scale + cmap as `mean` so users can A/B-toggle the two
            # without their eyes recalibrating to a different palette.
            value_by_time = np.quantile(members, 0.95, axis=0)
            vmax = eval_render._wd_spatial_vmax(value_by_time)
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            render_cfg = {
                "title_prefix": "p95 WD",
                "colorbar_label": "WD p95 (m)",
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": float(vmax),
                "is_wd_depth": True,
                "zero_transparent": False,
            }
        elif product == self.SCRUB_PRODUCT_IQR:
            # Interquartile range = q75 - q25 — robust spread metric, useful
            # alongside the std-based `spread` map because it ignores the tails.
            q75 = np.quantile(members, 0.75, axis=0)
            q25 = np.quantile(members, 0.25, axis=0)
            value_by_time = q75 - q25
            vmax = float(eval_render._robust_nonnegative_vmax(value_by_time))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            render_cfg = {
                "title_prefix": "WD IQR",
                "colorbar_label": "WD IQR (m)",
                "cmap": "magma",
                "vmin": 0.0,
                "vmax": vmax,
                "is_wd_depth": False,
                "zero_transparent": True,
            }
        else:  # SCRUB_PRODUCT_P_GT_0P30M
            raw_prob = (members > float(threshold_m)).mean(axis=0)  # [n_time, n_cells]
            value_by_time = raw_prob.copy()
            use_iso = (
                calibration_adapter is not None
                and calibration_adapter.has_isotonic_curves()
                and wettable is not None
            )
            if use_iso:
                # Mirror write_map_pngs's per-timestep calibration: feed only
                # wettable cells through the isotonic curve, then re-embed.
                for t_idx in range(n_time):
                    raw_wet = raw_prob[t_idx][wettable]
                    cal_wet = calibration_adapter.apply_isotonic_exceedance(
                        raw_wet,
                        threshold_m=float(threshold_m),
                        lead_time_hour=float(lead[t_idx]),
                        wettable_mask=wettable,
                    )
                    value_by_time[t_idx] = raw_prob[t_idx].copy()
                    value_by_time[t_idx][wettable] = cal_wet
                    value_by_time[t_idx][~wettable] = 0.0
            render_cfg = {
                "title_prefix": f"P(WD > {float(threshold_m):.2f} m)",
                "colorbar_label": f"P(WD > {float(threshold_m):.2f} m)",
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": 1.0,
                "is_wd_depth": False,
                "zero_transparent": True,
            }

        paths: list[Path] = []
        for t_idx in range(n_time):
            path = write_rivanna_style_map_png(
                values=value_by_time[t_idx],
                geometry_xy=xy,
                output_path=out_dir / f"{label}_{product}_scrub_t{t_idx + 1:03d}.png",
                title=f"{label} {render_cfg['title_prefix']} | lead {lead[t_idx]:.2f} h",
                colorbar_label=render_cfg["colorbar_label"],
                cmap=render_cfg["cmap"],
                vmin=render_cfg["vmin"],
                vmax=render_cfg["vmax"],
                elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
                is_wd_depth=render_cfg["is_wd_depth"],
                zero_transparent=render_cfg["zero_transparent"],
                visualization_config=visualization_config,
                dpi=int(dpi),
            )
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
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
        from neuralop.flood.eval import render as eval_render

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.titleweight": "semibold",
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )

        x = xy[:, 0]
        y = xy[:, 1]
        fig_size = (7.2, 6.1)
        dpi = 180
        renderer = eval_render._build_spatial_renderer(x, y, figsize=fig_size, dpi=dpi, n_rows=1, n_cols=1)
        elevation_raw = metadata.get("elevation_raw")
        visualization_config = metadata.get("visualization_config")
        context = eval_render._cartographic_context(
            x=x,
            y=y,
            elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
            out_dir=str(out_path.parent),
            visualization_config=visualization_config,
        )
        vmax = eval_render._wd_spatial_vmax(mean_by_time)
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        options = context.get("options", {})
        wet_threshold = float(options.get("wet_edge_threshold_m", 0.05))
        initial_idx = next(
            (idx for idx in range(n_time) if float(np.nanmax(mean_by_time[idx])) > wet_threshold),
            0,
        )
        fig, ax = plt.subplots(
            1,
            1,
            figsize=fig_size,
            dpi=dpi,
            constrained_layout=True,
            facecolor="#f7fafb",
        )
        try:
            ax.set_facecolor("#f7fafb")
            artist, _ = eval_render._plot_spatial_panel(
                ax=ax,
                x=x,
                y=y,
                arr=mean_by_time[initial_idx],
                renderer=renderer,
                context=context,
                cmap="viridis",
                vmin=0.0,
                vmax=float(vmax),
                is_wd_depth=True,
                zero_transparent=False,
                annotate=False,
            )
            ax.set_aspect("equal")
            ax.axis("off")
            title = ax.set_title(
                f"{label} mean WD | lead {lead[initial_idx]:.2f} h",
                loc="left",
                color="#102027",
                pad=8,
            )
            cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8, colors="#3f535b")
            cbar.outline.set_edgecolor("#aebfc7")
            cbar.set_label("Mean WD (m)", color="#102027", fontsize=8.5)
            writer = PillowWriter(fps=int(max(1, fps)))
            with writer.saving(fig, str(out_path), dpi=dpi):
                for t_idx in range(n_time):
                    frame = eval_render._mask_wd_dry_for_overlay(mean_by_time[t_idx], wet_threshold)
                    eval_render._update_spatial_artist(artist, frame, renderer)
                    title.set_text(f"{label} mean WD | lead {lead[t_idx]:.2f} h")
                    writer.grab_frame()
        finally:
            plt.close(fig)
        return out_path

    def write_exceedance_animation_gif(
        self,
        forecast: ForecastResult,
        *,
        output_path: str | Path,
        threshold_m: float,
        label: str = "calibrated",
        fps: int = 4,
        calibration_adapter: "CalibrationAdapter | None" = None,
    ) -> Path | None:
        """Render an animated GIF of exceedance probability over lead time."""
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
        wettable = (
            np.asarray(forecast.wettable_mask, dtype=bool)
            if forecast.wettable_mask is not None
            else np.ones(members.shape[2], dtype=bool)
        )
        prob_wet = self._probability_by_time(
            members[:, :, wettable],
            threshold_m=float(threshold_m),
            lead_hours=lead,
            wettable_mask=wettable,
            calibration_adapter=calibration_adapter,
        )
        prob_full = np.zeros((n_time, members.shape[2]), dtype=np.float64)
        prob_full[:, wettable] = prob_wet
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
        from neuralop.flood.eval import render as eval_render

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.titleweight": "semibold",
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )

        x = xy[:, 0]
        y = xy[:, 1]
        fig_size = (7.2, 6.1)
        dpi = 180
        renderer = eval_render._build_spatial_renderer(x, y, figsize=fig_size, dpi=dpi, n_rows=1, n_cols=1)
        elevation_raw = metadata.get("elevation_raw")
        visualization_config = metadata.get("visualization_config")
        context = eval_render._cartographic_context(
            x=x,
            y=y,
            elevation_raw=np.asarray(elevation_raw, dtype=np.float64) if elevation_raw is not None else None,
            out_dir=str(out_path.parent),
            visualization_config=visualization_config,
        )
        initial_idx = next((idx for idx in range(n_time) if float(np.nanmax(prob_full[idx])) > 0.0), 0)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=fig_size,
            dpi=dpi,
            constrained_layout=True,
            facecolor="#f7fafb",
        )
        try:
            ax.set_facecolor("#f7fafb")
            artist, _ = eval_render._plot_spatial_panel(
                ax=ax,
                x=x,
                y=y,
                arr=prob_full[initial_idx],
                renderer=renderer,
                context=context,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                is_wd_depth=False,
                zero_transparent=True,
                annotate=False,
            )
            ax.set_aspect("equal")
            ax.axis("off")
            title = ax.set_title(
                f"{label} P(WD > {threshold_m:g} m) | lead {lead[initial_idx]:.2f} h",
                loc="left",
                color="#102027",
                pad=8,
            )
            cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8, colors="#3f535b")
            cbar.outline.set_edgecolor("#aebfc7")
            cbar.set_label(f"P(WD > {threshold_m:g} m)", color="#102027", fontsize=8.5)
            writer = PillowWriter(fps=int(max(1, fps)))
            with writer.saving(fig, str(out_path), dpi=dpi):
                for t_idx in range(n_time):
                    eval_render._update_spatial_artist(artist, prob_full[t_idx], renderer)
                    title.set_text(f"{label} P(WD > {threshold_m:g} m) | lead {lead[t_idx]:.2f} h")
                    writer.grab_frame()
        finally:
            plt.close(fig)
        return out_path
