"""Offline artifact-only calibration ablations for flood UQ artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from neuralop.flood.eval.impact_metrics import (
    DEFAULT_IMPACT_THRESHOLD_M,
    DEFAULT_POOLED_RADII_M,
    compute_flood_impact_crps_metrics,
)
from neuralop.flood.eval.scientific_calibration import (
    COEFFICIENTS_JSON,
    DEFAULT_LEAD_BINS_H,
    DEFAULT_THRESHOLDS_M,
    DEFAULT_WET_FREQ_EDGES,
    ISOTONIC_JSON,
    CalibrationBins,
    apply_crps_mbm_to_wd_members,
    build_calibration_comparison,
    compute_artifact_uq_metrics,
    empirical_crps_mean,
    fit_crps_member_by_member_from_artifacts,
    fit_exceedance_isotonic_from_artifacts,
    list_forecast_artifacts,
    load_forecast_artifact,
    save_crps_mbm_coefficients,
    save_exceedance_isotonic,
    save_fit_diagnostics,
    save_metrics_json,
)


SCOREBOARD_CSV = "calibration_ablation_scoreboard.csv"
SCOREBOARD_JSON = "calibration_ablation_scoreboard.json"
SELECTED_VARIANT_JSON = "selected_calibration_variant.json"
REGIME_DIAGNOSTICS_JSON = "regime_diagnostics.json"
IMPACT_METRICS_JSON = "impact_metrics.json"


@dataclass(frozen=True)
class CalibrationAblationVariant:
    """One MBM calibration objective variant evaluated from saved artifacts."""

    name: str
    objective: str
    mean_rmse_weight: float = 0.0
    spread_ratio_weight: float = 0.0
    target_spread_ratio: float = 1.0
    tail_threshold_m: float = 0.30
    tail_weight: float = 4.0


def default_ablation_variants(
    *,
    mean_rmse_weight: float = 0.5,
    spread_ratio_weight: float = 0.5,
    target_spread_ratio: float = 1.0,
    tail_threshold_m: float = 0.30,
    tail_weight: float = 4.0,
) -> List[CalibrationAblationVariant]:
    """Return the standard E1-E5 CRPS-MBM ablation matrix."""
    return [
        CalibrationAblationVariant("E1_empirical_crps_mbm", "empirical_crps"),
        CalibrationAblationVariant(
            "E2_tail_weighted_crps_mbm",
            "tail_weighted_crps",
            tail_threshold_m=float(tail_threshold_m),
            tail_weight=float(tail_weight),
        ),
        CalibrationAblationVariant(
            "E3_mean_regularized_crps_mbm",
            "mean_regularized_crps",
            mean_rmse_weight=float(mean_rmse_weight),
        ),
        CalibrationAblationVariant(
            "E4_spread_regularized_crps_mbm",
            "spread_regularized_crps",
            spread_ratio_weight=float(spread_ratio_weight),
            target_spread_ratio=float(target_spread_ratio),
        ),
        CalibrationAblationVariant(
            "E5_combined_regularized_crps_mbm",
            "combined_regularized_crps",
            mean_rmse_weight=float(mean_rmse_weight),
            spread_ratio_weight=float(spread_ratio_weight),
            target_spread_ratio=float(target_spread_ratio),
            tail_threshold_m=float(tail_threshold_m),
            tail_weight=float(tail_weight),
        ),
    ]


def collect_artifact_paths(paths: Sequence[str | Path]) -> List[Path]:
    """Resolve files/directories into a sorted, duplicate-free artifact list."""
    out: List[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        candidates = list_forecast_artifacts(path) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            key = str(resolved)
            if key not in seen:
                out.append(resolved)
                seen.add(key)
    if not out:
        raise ValueError("No artifact paths were supplied.")
    return sorted(out)


def _nested_get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            if key not in cur:
                return default
            cur = cur[key]
            continue
        try:
            cur = getattr(cur, key)
        except (AttributeError, KeyError, TypeError):
            return default
    return cur


def _load_json_or_yaml(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Calibration ablation config not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError(
                f"PyYAML is required to read non-JSON config files: {cfg_path}"
            ) from exc
        payload = yaml.safe_load(text) or {}
    if isinstance(payload, Mapping) and "flood" in payload and isinstance(payload["flood"], Mapping):
        payload = payload["flood"]
    return dict(payload)


def _bins_from_config(config: Mapping[str, Any] | None) -> CalibrationBins:
    if config:
        return CalibrationBins.from_config(config)
    return CalibrationBins(
        lead_time_hours=tuple(DEFAULT_LEAD_BINS_H),
        wet_frequency_edges=tuple(DEFAULT_WET_FREQ_EDGES),
        wet_threshold_m=0.01,
    )


def _threshold_key(threshold_m: float, *, prefix: str = "brier_wd_exceed") -> str:
    return f"{prefix}_{float(threshold_m):.2f}m_overall_mean".replace(".", "p")


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _pct_change(raw_value: Any, new_value: Any) -> float:
    raw = _finite_float(raw_value)
    new = _finite_float(new_value)
    if not np.isfinite(raw) or abs(raw) < 1e-12 or not np.isfinite(new):
        return math.nan
    return float(100.0 * (new - raw) / abs(raw))


def _calibrated_pred_for_artifact(
    artifact: Mapping[str, Any],
    calibration_model: Mapping[str, Any] | None,
) -> np.ndarray:
    pred = np.asarray(artifact["pred_members_wd"], dtype=np.float64)
    if calibration_model is None:
        return pred
    wettable = np.asarray(
        artifact.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)),
        dtype=bool,
    )
    time_hours = artifact.get("time_hours")
    if time_hours is None:
        time_hours = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
    out = np.empty_like(pred, dtype=np.float64)
    for t in range(pred.shape[1]):
        out[:, t, :] = apply_crps_mbm_to_wd_members(
            pred[:, t, :],
            lead_time_hour=float(time_hours[t]),
            calibration_model=calibration_model,
            wettable_mask=wettable,
        )
    return out


def _point_metrics_for_mask(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    thresholds_m: Sequence[float],
) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return {}
    pred_sel = np.asarray(pred, dtype=np.float64)[:, mask]
    ref_sel = np.asarray(ref, dtype=np.float64)[:, mask]
    if pred_sel.size == 0 or ref_sel.size == 0:
        return {}
    pred_mean = np.mean(pred_sel, axis=0)
    ref_mean = np.mean(ref_sel, axis=0)
    pred_spread = float(np.mean(np.std(pred_sel, axis=0)))
    ref_spread = float(np.mean(np.std(ref_sel, axis=0)))
    out = {
        "n_cells": int(np.sum(mask)),
        "crps_wd_mean": empirical_crps_mean(pred_sel, ref_sel),
        "rmse_ens_mean_wd_mean": float(np.sqrt(np.mean((pred_mean - ref_mean) ** 2))),
        "spread_pred_wd_mean": pred_spread,
        "spread_gt_wd_mean": ref_spread,
        "spread_ratio_wd_mean": pred_spread / max(ref_spread, 1e-12),
    }
    for alpha in (0.90, 0.95):
        lo_q = 0.5 * (1.0 - alpha)
        hi_q = 1.0 - lo_q
        lo = np.quantile(pred_sel, lo_q, axis=0)
        hi = np.quantile(pred_sel, hi_q, axis=0)
        out[f"coverage_wd_{int(round(alpha * 100))}_mean"] = float(
            np.mean((ref_sel >= lo[None, :]) & (ref_sel <= hi[None, :]))
        )
    for threshold in thresholds_m:
        raw_prob = np.mean(pred_sel > float(threshold), axis=0)
        obs_prob = np.mean(ref_sel > float(threshold), axis=0)
        out[f"brier_wd_exceed_{float(threshold):.2f}m_mean".replace(".", "p")] = float(
            np.mean((raw_prob - obs_prob) ** 2)
        )
        out[f"reliability_abs_wd_exceed_{float(threshold):.2f}m_mean".replace(".", "p")] = float(
            abs(np.mean(raw_prob) - np.mean(obs_prob))
        )
    return out


def _append_metric_dict(accum: Dict[str, Dict[str, List[float]]], group: str, values: Mapping[str, Any]) -> None:
    bucket = accum.setdefault(group, {})
    for key, value in values.items():
        if key == "n_cells":
            bucket.setdefault(key, []).append(float(value))
            continue
        val = _finite_float(value)
        if np.isfinite(val):
            bucket.setdefault(key, []).append(val)


def compute_artifact_regime_diagnostics(
    artifact_paths: Sequence[str | Path],
    *,
    bins: CalibrationBins,
    calibration_model: Mapping[str, Any] | None = None,
    wet_frequency_by_cell: np.ndarray | None = None,
    thresholds_m: Sequence[float] = DEFAULT_THRESHOLDS_M,
) -> Dict[str, Dict[str, float]]:
    """Compute CRPS/Brier diagnostics by lead-time, wet-frequency, and depth regime."""
    accum: Dict[str, Dict[str, List[float]]] = {}
    if wet_frequency_by_cell is not None:
        wet_frequency = np.asarray(wet_frequency_by_cell, dtype=np.float64)
        wet_bin_by_cell = bins.wet_index(wet_frequency)
    else:
        wet_bin_by_cell = None
    for path in artifact_paths:
        artifact = load_forecast_artifact(path, load_members=True)
        pred = _calibrated_pred_for_artifact(artifact, calibration_model)
        ref = np.asarray(artifact["ref_members_wd"], dtype=np.float64)
        wettable = np.asarray(
            artifact.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)),
            dtype=bool,
        )
        time_hours = artifact.get("time_hours")
        if time_hours is None:
            time_hours = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
        if wet_bin_by_cell is not None and wet_bin_by_cell.size != pred.shape[2]:
            raise ValueError(
                f"wet_frequency_by_cell length {wet_bin_by_cell.size} does not match artifact cells {pred.shape[2]}."
            )
        for t in range(pred.shape[1]):
            lead_idx = bins.lead_index(float(time_hours[t]))
            ref_mean = np.mean(ref[:, t, :], axis=0)
            base = wettable
            _append_metric_dict(
                accum,
                f"lead_bin_{lead_idx}",
                _point_metrics_for_mask(pred[:, t, :], ref[:, t, :], base, thresholds_m),
            )
            if wet_bin_by_cell is not None:
                for wet_idx in range(bins.n_wet_bins):
                    _append_metric_dict(
                        accum,
                        f"wet_frequency_bin_{wet_idx}",
                        _point_metrics_for_mask(
                            pred[:, t, :],
                            ref[:, t, :],
                            base & (wet_bin_by_cell == wet_idx),
                            thresholds_m,
                        ),
                    )
            regimes = {
                "depth_dry": ref_mean <= 0.01,
                "depth_transition": (ref_mean > 0.01) & (ref_mean < 0.10),
                "depth_shallow": (ref_mean >= 0.10) & (ref_mean < 0.30),
                "depth_deep": ref_mean >= 0.30,
            }
            for name, mask in regimes.items():
                _append_metric_dict(
                    accum,
                    name,
                    _point_metrics_for_mask(pred[:, t, :], ref[:, t, :], base & mask, thresholds_m),
                )
    out: Dict[str, Dict[str, float]] = {}
    for group, values in accum.items():
        out[group] = {}
        for key, vals in values.items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size:
                out[group][f"{key}_mean"] = float(np.mean(arr))
                out[group][f"{key}_std"] = float(np.std(arr))
    return out


def compute_artifact_impact_metrics(
    artifact_paths: Sequence[str | Path],
    *,
    calibration_model: Mapping[str, Any] | None = None,
    impact_config: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    """Compute pooled/impact CRPS summaries from saved artifacts when geometry is present."""
    accum: Dict[str, List[float]] = {}
    final_accum: Dict[str, List[float]] = {}
    skipped = 0
    for path in artifact_paths:
        artifact = load_forecast_artifact(path, load_members=True)
        geometry = artifact.get("geometry_raw")
        if geometry is None:
            skipped += 1
            continue
        pred = _calibrated_pred_for_artifact(artifact, calibration_model)
        ref = np.asarray(artifact["ref_members_wd"], dtype=np.float64)
        wettable = np.asarray(
            artifact.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)),
            dtype=bool,
        )
        metrics = compute_flood_impact_crps_metrics(
            np.transpose(pred, (1, 0, 2)),
            np.transpose(ref, (1, 0, 2)),
            geometry=geometry,
            static_raw=None,
            wettable_mask=wettable,
            config=impact_config,
        )
        for key, value in metrics.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                if np.isfinite(float(arr)):
                    accum.setdefault(key, []).append(float(arr))
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size:
                accum.setdefault(f"{key}_mean", []).append(float(np.mean(finite)))
                final_accum.setdefault(f"{key}_final", []).append(float(finite[-1]))
    out: Dict[str, float] = {"n_artifacts_skipped_no_geometry": float(skipped)}
    for key, values in accum.items():
        out[key] = float(np.mean(values)) if values else math.nan
    for key, values in final_accum.items():
        out[key] = float(np.mean(values)) if values else math.nan
    return out


def _scoreboard_row(
    *,
    name: str,
    metrics: Mapping[str, Any],
    raw_metrics: Mapping[str, Any],
    variant: CalibrationAblationVariant | None = None,
    current_spread_ratio: float | None = None,
    impact_metrics: Mapping[str, Any] | None = None,
    current_impact_metrics: Mapping[str, Any] | None = None,
    isotonic: bool = False,
) -> Dict[str, Any]:
    thresholds = (0.05, 0.10, 0.30, 0.50)
    row: Dict[str, Any] = {
        "variant": name,
        "objective": variant.objective if variant else ("raw" if not isotonic else "best_mbm_plus_isotonic"),
        "mean_rmse_weight": variant.mean_rmse_weight if variant else 0.0,
        "spread_ratio_weight": variant.spread_ratio_weight if variant else 0.0,
        "target_spread_ratio": variant.target_spread_ratio if variant else 1.0,
        "tail_threshold_m": variant.tail_threshold_m if variant else 0.30,
        "tail_weight": variant.tail_weight if variant else 4.0,
        "crps_wd": _finite_float(metrics.get("crps_wd_overall_mean")),
        "crps_improvement_pct_vs_raw": -_pct_change(
            raw_metrics.get("crps_wd_overall_mean"),
            metrics.get("crps_wd_overall_mean"),
        ),
        "rmse_ens_mean_wd": _finite_float(metrics.get("rmse_ens_mean_wd_overall_mean")),
        "rmse_degradation_pct_vs_raw": _pct_change(
            raw_metrics.get("rmse_ens_mean_wd_overall_mean"),
            metrics.get("rmse_ens_mean_wd_overall_mean"),
        ),
        "spread_ratio_wd": _finite_float(metrics.get("spread_ratio_wd_overall_mean")),
        "coverage_wd_90": _finite_float(metrics.get("coverage_wd_90_overall_mean")),
        "coverage_wd_95": _finite_float(metrics.get("coverage_wd_95_overall_mean")),
        "rank_histogram_l1_wd": _finite_float(metrics.get("rank_histogram_l1_wd_overall_mean")),
        "isotonic_probability_row": bool(isotonic),
    }
    row["spread_ratio_improves_current"] = (
        bool(np.isfinite(row["spread_ratio_wd"]) and current_spread_ratio is not None and row["spread_ratio_wd"] < current_spread_ratio)
        if name != "E0_raw"
        else False
    )
    row["spread_ratio_below_2p0"] = bool(np.isfinite(row["spread_ratio_wd"]) and row["spread_ratio_wd"] < 2.0)
    brier_ok = True
    for threshold in thresholds:
        key = _threshold_key(threshold)
        iso_key = _threshold_key(threshold, prefix="brier_isotonic_wd_exceed")
        row[key] = _finite_float(metrics.get(key))
        row[f"{key}_improvement_pct_vs_raw"] = -_pct_change(raw_metrics.get(key), metrics.get(key))
        row[iso_key] = _finite_float(metrics.get(iso_key))
        if not np.isfinite(row[key]) or not np.isfinite(_finite_float(raw_metrics.get(key))) or row[key] > _finite_float(raw_metrics.get(key)) + 1e-15:
            brier_ok = False
    impact_ok = True
    for key, value in (impact_metrics or {}).items():
        if key.startswith("crps_") or key.startswith("pooled_") or "crps" in key:
            metric_value = _finite_float(value)
            row[f"impact_{key}"] = metric_value
            current_value = _finite_float((current_impact_metrics or {}).get(key))
            if np.isfinite(metric_value) and np.isfinite(current_value) and metric_value > current_value + 1e-15:
                impact_ok = False
    row["brier_improves_required_thresholds"] = brier_ok
    row["impact_metrics_available"] = bool(impact_metrics)
    row["impact_not_worse_current"] = impact_ok
    coverage_ok = (
        np.isfinite(row["coverage_wd_90"])
        and np.isfinite(row["coverage_wd_95"])
        and 0.88 <= row["coverage_wd_90"] <= 0.97
        and 0.93 <= row["coverage_wd_95"] <= 0.985
    )
    row["coverage_close_to_nominal"] = bool(coverage_ok)
    row["accepted"] = bool(
        name != "E0_raw"
        and not isotonic
        and row["crps_improvement_pct_vs_raw"] >= 5.0
        and row["rmse_degradation_pct_vs_raw"] <= 0.5
        and row["spread_ratio_improves_current"]
        and brier_ok
        and coverage_ok
        and impact_ok
    )
    return row


def _write_scoreboard(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    json_path = out_dir / SCOREBOARD_JSON
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, sort_keys=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(out_dir / SCOREBOARD_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _plot_scoreboard(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    plot_rows = [row for row in rows if row.get("variant") != "E0_raw" and not row.get("isotonic_probability_row")]
    if not plot_rows:
        return
    names = [str(row["variant"]).replace("_crps_mbm", "").replace("_", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].bar(x, [_finite_float(row.get("crps_improvement_pct_vs_raw")) for row in plot_rows])
    axes[0, 0].axhline(5.0, color="0.3", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("CRPS improvement vs raw (%)")
    axes[0, 1].bar(x, [_finite_float(row.get("rmse_degradation_pct_vs_raw")) for row in plot_rows])
    axes[0, 1].axhline(0.5, color="0.3", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("RMSE degradation vs raw (%)")
    axes[1, 0].bar(x, [_finite_float(row.get("spread_ratio_wd")) for row in plot_rows])
    axes[1, 0].axhline(2.0, color="0.3", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Spread ratio")
    axes[1, 1].bar(x, [_finite_float(row.get(_threshold_key(0.10))) for row in plot_rows])
    axes[1, 1].set_title("Brier wd > 0.10 m")
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_dir / "calibration_ablation_scoreboard.png", dpi=180)
    plt.close(fig)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_utc_now()}] {message}", flush=True)


def _write_progress(out_dir: Path, payload: Mapping[str, Any]) -> None:
    progress = {"updated_at_utc": _utc_now(), **dict(payload)}
    with open(out_dir / "progress.json", "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2, sort_keys=True)


def _select_best_variant(rows: Sequence[Mapping[str, Any]]) -> str:
    candidates = [
        row
        for row in rows
        if row.get("variant") != "E0_raw" and not row.get("isotonic_probability_row")
    ]
    accepted = [row for row in candidates if bool(row.get("accepted"))]
    pool = accepted or candidates
    if not pool:
        raise ValueError("No calibration variants were evaluated.")
    return str(min(pool, key=lambda row: _finite_float(row.get("crps_wd"), math.inf))["variant"])


def run_calibration_ablation(
    calibration_artifacts: Sequence[str | Path],
    heldout_artifacts: Sequence[str | Path],
    out_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    variants: Sequence[CalibrationAblationVariant] | None = None,
    thresholds_m: Sequence[float] = DEFAULT_THRESHOLDS_M,
    include_impact_metrics: bool = True,
    progress_interval: int = 50,
) -> Dict[str, Any]:
    """Fit/evaluate artifact-only calibration variants and write a ranked scoreboard."""
    out_path = Path(out_dir).expanduser().resolve(strict=False)
    out_path.mkdir(parents=True, exist_ok=True)
    calibration_paths = collect_artifact_paths(calibration_artifacts)
    heldout_paths = collect_artifact_paths(heldout_artifacts)
    _log(
        "resolved artifacts: "
        f"calibration={len(calibration_paths)} heldout={len(heldout_paths)} out_dir={out_path}"
    )
    _write_progress(out_path, {"stage": "resolved_artifacts", "variant": None})
    bins = _bins_from_config(config)
    bins.validate()

    optimizer_cfg = _nested_get(config, "rollout_calibration", "optimizer", default={}) or {}
    impact_cfg = {
        "enabled": True,
        "inundation_threshold_m": float(
            _nested_get(
                config,
                "rollout",
                "impact_metrics",
                "inundation_threshold_m",
                default=DEFAULT_IMPACT_THRESHOLD_M,
            )
        ),
        "pooled_radii_m": list(
            _nested_get(
                config,
                "rollout",
                "impact_metrics",
                "pooled_radii_m",
                default=DEFAULT_POOLED_RADII_M,
            )
        ),
    }
    bounds = _nested_get(optimizer_cfg, "bounds", default=None) or {
        "a_m": [-2.0, 2.0],
        "beta": [0.2, 2.5],
        "gamma": [0.05, 3.0],
    }
    variants = list(variants or default_ablation_variants(
        mean_rmse_weight=float(_nested_get(optimizer_cfg, "mean_rmse_weight", default=0.5)),
        spread_ratio_weight=float(_nested_get(optimizer_cfg, "spread_ratio_weight", default=0.5)),
        target_spread_ratio=float(_nested_get(optimizer_cfg, "target_spread_ratio", default=1.0)),
        tail_threshold_m=float(_nested_get(optimizer_cfg, "tail_threshold_m", default=0.30)),
        tail_weight=float(_nested_get(optimizer_cfg, "tail_weight", default=4.0)),
    ))

    raw_dir = out_path / "E0_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _log("E0_raw metrics start")
    _write_progress(out_path, {"stage": "raw_metrics_start", "variant": "E0_raw"})
    raw_metrics = compute_artifact_uq_metrics(heldout_paths, thresholds_m=thresholds_m)
    save_metrics_json(raw_metrics, raw_dir / "metrics.json")
    _log(
        "E0_raw metrics done "
        f"crps={raw_metrics.get('crps_wd_overall_mean')} "
        f"rmse={raw_metrics.get('rmse_wd_overall_mean')} "
        f"spread_ratio={raw_metrics.get('spread_ratio_wd_overall_mean')}"
    )
    raw_impact = (
        compute_artifact_impact_metrics(heldout_paths, impact_config=impact_cfg)
        if include_impact_metrics
        else {}
    )
    save_metrics_json(raw_impact, raw_dir / IMPACT_METRICS_JSON)
    _write_progress(out_path, {"stage": "raw_metrics_done", "variant": "E0_raw"})

    rows: List[Dict[str, Any]] = [
        _scoreboard_row(
            name="E0_raw",
            metrics=raw_metrics,
            raw_metrics=raw_metrics,
            impact_metrics=raw_impact,
        )
    ]
    models: Dict[str, Dict[str, Any]] = {}
    variant_metrics: Dict[str, Dict[str, float]] = {}
    variant_impacts: Dict[str, Dict[str, float]] = {}

    for variant in variants:
        variant_dir = out_path / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _log(f"{variant.name} fit start objective={variant.objective}")
        _write_progress(out_path, {"stage": "fit_start", "variant": variant.name})

        def variant_progress(message: str, *, variant_name: str = variant.name) -> None:
            _log(f"{variant_name} {message}")
            _write_progress(out_path, {"stage": "fit_progress", "variant": variant_name, "message": message})

        model = fit_crps_member_by_member_from_artifacts(
            calibration_paths,
            bins=bins,
            max_fit_points_per_bin=int(_nested_get(optimizer_cfg, "max_fit_points_per_bin", default=1_000_000)),
            min_fit_points_per_bin=int(_nested_get(optimizer_cfg, "min_fit_points_per_bin", default=1000)),
            seed=int(_nested_get(optimizer_cfg, "seed", default=123)),
            bounds=bounds,
            objective=variant.objective,
            tail_threshold_m=variant.tail_threshold_m,
            tail_weight=variant.tail_weight,
            mean_rmse_weight=variant.mean_rmse_weight,
            spread_ratio_weight=variant.spread_ratio_weight,
            target_spread_ratio=variant.target_spread_ratio,
            multistart=bool(_nested_get(optimizer_cfg, "multistart", default=True)),
            progress_callback=variant_progress,
            progress_interval=int(progress_interval),
        )
        _log(f"{variant.name} fit done")
        _write_progress(out_path, {"stage": "fit_done", "variant": variant.name})
        save_crps_mbm_coefficients(model, variant_dir)
        save_fit_diagnostics(model, variant_dir)
        _log(f"{variant.name} heldout metrics start")
        _write_progress(out_path, {"stage": "heldout_metrics_start", "variant": variant.name})
        metrics = compute_artifact_uq_metrics(
            heldout_paths,
            calibration_model=model,
            thresholds_m=thresholds_m,
        )
        save_metrics_json(metrics, variant_dir / "metrics.json")
        save_metrics_json(build_calibration_comparison(raw_metrics, metrics), variant_dir / "comparison_vs_raw.json")
        _log(
            f"{variant.name} heldout metrics done "
            f"crps={metrics.get('crps_wd_overall_mean')} "
            f"rmse={metrics.get('rmse_wd_overall_mean')} "
            f"spread_ratio={metrics.get('spread_ratio_wd_overall_mean')}"
        )
        regime = compute_artifact_regime_diagnostics(
            heldout_paths,
            bins=bins,
            calibration_model=model,
            wet_frequency_by_cell=model.get("wet_frequency_by_cell"),
            thresholds_m=thresholds_m,
        )
        save_metrics_json(regime, variant_dir / REGIME_DIAGNOSTICS_JSON)
        _log(f"{variant.name} impact metrics start")
        _write_progress(out_path, {"stage": "impact_metrics_start", "variant": variant.name})
        impact = (
            compute_artifact_impact_metrics(
                heldout_paths,
                calibration_model=model,
                impact_config=impact_cfg,
            )
            if include_impact_metrics
            else {}
        )
        save_metrics_json(impact, variant_dir / IMPACT_METRICS_JSON)
        _log(f"{variant.name} done")
        _write_progress(out_path, {"stage": "variant_done", "variant": variant.name})
        models[variant.name] = model
        variant_metrics[variant.name] = metrics
        variant_impacts[variant.name] = impact

    current_spread_ratio = _finite_float(
        variant_metrics.get("E1_empirical_crps_mbm", {}).get("spread_ratio_wd_overall_mean"),
        default=2.39,
    )
    for variant in variants:
        rows.append(
            _scoreboard_row(
                name=variant.name,
                metrics=variant_metrics[variant.name],
                raw_metrics=raw_metrics,
                variant=variant,
                current_spread_ratio=current_spread_ratio,
                impact_metrics=variant_impacts.get(variant.name, {}),
                current_impact_metrics=variant_impacts.get("E1_empirical_crps_mbm", {}),
            )
        )

    selected_name = _select_best_variant(rows)
    _log(f"selected MBM variant for isotonic evaluation: {selected_name}")
    _write_progress(out_path, {"stage": "isotonic_start", "variant": selected_name})
    selected_model = models[selected_name]
    iso_dir = out_path / f"E6_{selected_name}_plus_isotonic"
    iso_dir.mkdir(parents=True, exist_ok=True)
    isotonic = fit_exceedance_isotonic_from_artifacts(
        calibration_paths,
        bins=bins,
        wet_frequency_by_cell=selected_model["wet_frequency_by_cell"],
        thresholds_m=thresholds_m,
        min_fit_points_per_bin=int(
            _nested_get(config, "rollout_calibration", "exceedance", "min_fit_points_per_bin", default=128)
        ),
        calibration_model=selected_model,
    )
    save_exceedance_isotonic(isotonic, iso_dir)
    iso_metrics = compute_artifact_uq_metrics(
        heldout_paths,
        calibration_model=selected_model,
        isotonic_model=isotonic,
        apply_isotonic=True,
        thresholds_m=thresholds_m,
    )
    save_metrics_json(iso_metrics, iso_dir / "metrics.json")
    _log(f"E6 isotonic metrics done for {selected_name}")
    rows.append(
        _scoreboard_row(
            name=f"E6_{selected_name}_plus_isotonic",
            metrics=iso_metrics,
            raw_metrics=raw_metrics,
            current_spread_ratio=current_spread_ratio,
            impact_metrics=variant_impacts.get(selected_name, {}),
            current_impact_metrics=variant_impacts.get("E1_empirical_crps_mbm", {}),
            isotonic=True,
        )
    )

    selected_payload = {
        "selected_variant": selected_name,
        "selection_rule": "accepted variant with lowest held-out CRPS, otherwise lowest held-out CRPS",
        "coefficients_json": str((out_path / selected_name / COEFFICIENTS_JSON).resolve(strict=False)),
        "isotonic_json": str((iso_dir / ISOTONIC_JSON).resolve(strict=False)),
        "accepted": bool(next((row.get("accepted") for row in rows if row.get("variant") == selected_name), False)),
        "calibration_artifacts": [str(path) for path in calibration_paths],
        "heldout_artifacts": [str(path) for path in heldout_paths],
    }
    save_metrics_json(selected_payload, out_path / SELECTED_VARIANT_JSON)
    _write_scoreboard(rows, out_path)
    _plot_scoreboard(rows, out_path)
    _write_progress(out_path, {"stage": "done", "variant": selected_name})
    _log(f"ablation done selected={selected_name}")
    return {"rows": rows, "selected": selected_payload, "out_dir": str(out_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline CRPS-MBM calibration ablations from saved HDF5 forecast artifacts."
    )
    parser.add_argument("--calibration-artifacts", nargs="+", required=True, help="Calibration artifact files or directories.")
    parser.add_argument("--heldout-artifacts", nargs="+", required=True, help="Held-out artifact files or directories.")
    parser.add_argument("--out-dir", required=True, help="Output directory for coefficients, metrics, and scoreboard.")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML eval config for bins and optimizer defaults.")
    parser.add_argument("--thresholds-m", nargs="*", type=float, default=list(DEFAULT_THRESHOLDS_M))
    parser.add_argument("--mean-rmse-weight", type=float, default=None)
    parser.add_argument("--spread-ratio-weight", type=float, default=None)
    parser.add_argument("--target-spread-ratio", type=float, default=None)
    parser.add_argument("--tail-threshold-m", type=float, default=None)
    parser.add_argument("--tail-weight", type=float, default=None)
    parser.add_argument("--progress-interval", type=int, default=50, help="Optimizer score-call interval for progress logs.")
    parser.add_argument("--skip-impact-metrics", action="store_true", help="Skip pooled/impact CRPS diagnostics.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _load_json_or_yaml(args.config)
    optimizer_cfg = config.setdefault("rollout_calibration", {}).setdefault("optimizer", {})
    for attr, key in (
        ("mean_rmse_weight", "mean_rmse_weight"),
        ("spread_ratio_weight", "spread_ratio_weight"),
        ("target_spread_ratio", "target_spread_ratio"),
        ("tail_threshold_m", "tail_threshold_m"),
        ("tail_weight", "tail_weight"),
    ):
        value = getattr(args, attr)
        if value is not None:
            optimizer_cfg[key] = value
    result = run_calibration_ablation(
        args.calibration_artifacts,
        args.heldout_artifacts,
        args.out_dir,
        config=config,
        thresholds_m=args.thresholds_m,
        include_impact_metrics=not bool(args.skip_impact_metrics),
        progress_interval=int(args.progress_interval),
    )
    print(json.dumps(result["selected"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
