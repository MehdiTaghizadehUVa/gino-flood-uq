"""Lead-time affine post-hoc calibration utilities for rollout ensembles."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _to_1d_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.size == 0:
        return out
    return out


def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving average smoothing for 1D coefficient curves."""
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    w = int(window)
    if w <= 1:
        return x.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    y = np.convolve(x_pad, kernel, mode="valid")
    return y.astype(np.float64, copy=False)


def fit_leadtime_affine_calibration(
    mu_pred_by_t: Sequence[np.ndarray],
    sigma_pred_by_t: Sequence[np.ndarray],
    mu_ref_by_t: Sequence[np.ndarray],
    sigma_ref_by_t: Sequence[np.ndarray],
    *,
    fit_wet_threshold: float = 0.01,
    min_pred_std: float = 1e-4,
    c_clip_min: float = 0.25,
    c_clip_max: float = 4.0,
    smooth_window: int = 5,
) -> Dict[str, Any]:
    """
    Fit per-lead affine calibration coefficients for ensemble forecasts.

    For each lead time t:
      mu_ref ~= a_t + b_t * mu_pred
      spread calibration c_t from median(sigma_ref / max(sigma_pred, min_pred_std))

    Inputs are sequences of flattened arrays (one array per lead time).
    """
    n_steps = len(mu_pred_by_t)
    if not (
        len(sigma_pred_by_t) == n_steps
        and len(mu_ref_by_t) == n_steps
        and len(sigma_ref_by_t) == n_steps
    ):
        raise ValueError("All input sequences must have the same length.")
    if n_steps <= 0:
        raise ValueError("Expected at least one lead-time step for calibration fit.")

    wet_thr = float(fit_wet_threshold)
    std_floor = max(float(min_pred_std), 1e-12)
    cmin = float(c_clip_min)
    cmax = float(c_clip_max)
    if cmin <= 0 or cmax < cmin:
        raise ValueError("Invalid c_clip bounds.")

    a = np.zeros(n_steps, dtype=np.float64)
    b = np.ones(n_steps, dtype=np.float64)
    c = np.ones(n_steps, dtype=np.float64)
    n_fit = np.zeros(n_steps, dtype=np.int64)
    n_spread = np.zeros(n_steps, dtype=np.int64)

    for t in range(n_steps):
        x = _to_1d_float(mu_pred_by_t[t])
        y = _to_1d_float(mu_ref_by_t[t])
        sp = _to_1d_float(sigma_pred_by_t[t])
        sr = _to_1d_float(sigma_ref_by_t[t])

        if not (x.size == y.size == sp.size == sr.size):
            raise ValueError(f"Lead-time {t}: inconsistent flattened sizes.")

        finite_mu = np.isfinite(x) & np.isfinite(y)
        active = np.maximum(x, y) >= wet_thr
        fit_mask = finite_mu & active
        n_fit[t] = int(np.sum(fit_mask))

        if n_fit[t] > 0:
            xx = x[fit_mask]
            yy = y[fit_mask]
            x_mean = float(np.mean(xx))
            y_mean = float(np.mean(yy))
            var_x = float(np.mean((xx - x_mean) ** 2))
            if var_x > 1e-12:
                cov_xy = float(np.mean((xx - x_mean) * (yy - y_mean)))
                b[t] = cov_xy / var_x
            else:
                b[t] = 1.0
            a[t] = y_mean - b[t] * x_mean

        finite_spread = np.isfinite(sp) & np.isfinite(sr)
        spread_mask = finite_spread & active
        n_spread[t] = int(np.sum(spread_mask))
        if n_spread[t] > 0:
            ratio = sr[spread_mask] / np.maximum(sp[spread_mask], std_floor)
            ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
            if ratio.size > 0:
                c[t] = float(np.median(ratio))
        c[t] = float(np.clip(c[t], cmin, cmax))

    w = int(smooth_window)
    if w > 1:
        a = moving_average_1d(a, w)
        b = moving_average_1d(b, w)
        c = moving_average_1d(c, w)
        c = np.clip(c, cmin, cmax)

    return {
        "a": a,
        "b": b,
        "c": c,
        "n_fit": n_fit,
        "n_spread": n_spread,
        "fit_wet_threshold": wet_thr,
        "min_pred_std": std_floor,
        "c_clip_min": cmin,
        "c_clip_max": cmax,
        "smooth_window": int(w),
    }


def apply_leadtime_affine_to_ensemble(
    pred_ens: np.ndarray,
    *,
    a_t: float,
    b_t: float,
    c_t: float,
) -> np.ndarray:
    """
    Apply calibrated affine location/scale transform to one lead-time ensemble.

    pred_ens shape: [n_members, n_locations]
    """
    x = np.asarray(pred_ens, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("pred_ens must be 2D [n_members, n_locations].")
    mu = np.mean(x, axis=0, keepdims=True)
    anom = x - mu
    out = float(a_t) + float(b_t) * mu + float(c_t) * anom
    return out.astype(np.float64, copy=False)


def validate_split_leakage_guard(
    calib_txt: str,
    test_txt: str,
    *,
    allow_same_split: bool = False,
) -> None:
    """Validate calibration/eval split paths for leakage-safe usage."""
    c = str(calib_txt).strip()
    t = str(test_txt).strip()
    if not c:
        raise ValueError("rollout_calibration.calib_txt must be non-empty.")
    if not t:
        raise ValueError("rollout_data.test_txt must be non-empty.")
    if (not allow_same_split) and c == t:
        raise ValueError(
            "Calibration split equals evaluation split; set rollout_calibration.allow_same_split=true "
            "only for debug/non-publishable runs."
        )
