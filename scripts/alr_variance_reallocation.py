"""Post-hoc variance reallocation on saved ALR-FGNO forecasts.

Motivation
----------
The dispersion audit showed the FGNO predictive spread is ~1.39x wider than the
HEC-RAS simulator spread on every held-out event, while the epistemic channel
covers the reference mean only ~17% of the time at nominal 95%.  The anchor /
contrast sweep separately showed the adapter subspace responds *linearly* --
scaling inter-particle contrast by x5/x10 scales epistemic spread by 5.07/10.1
with no saturation, and improves wet-front error localization (0.28 -> 0.44),
while degrading CRPS only because total spread grows.

Together those say the binding constraint is the variance *budget*: the aleatory
channel has absorbed the model error, leaving nothing for the epistemic channel.

This script tests the implication directly, with no retraining and no new
inference.  It decomposes each saved nested forecast

    X[m,k] = mu + (Xbar[m] - mu) + (X[m,k] - Xbar[m])
           = grand mean + epistemic deviation + aleatory deviation

and rescales the two deviation channels

    X'[m,k] = mu + alpha * epistemic_dev + beta * aleatory_dev

with (alpha, beta) chosen so the TOTAL spread matches the reference-ensemble
spread while a fraction ``f`` of that variance sits in the epistemic channel.
The ensemble mean is invariant by construction, so RMSE is unchanged and the
comparison isolates the decomposition.

Reported per reallocation setting: design-aware crossed fair CRPS, predictive
coverage at 50/80/90/95, coverage of the reference MEAN by the epistemic
interval, and the wet-front epistemic-spread / error association.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

NOMINAL = (0.50, 0.80, 0.90, 0.95)
# f = share of the (calibrated) total variance assigned to the epistemic channel
F_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)


def _pairsum(arr: np.ndarray, axis: int) -> np.ndarray:
    xs = np.sort(arr, axis=axis)
    n = arr.shape[axis]
    coef = (2 * np.arange(1, n + 1) - 1 - n).astype(np.float64)
    shape = [1] * arr.ndim
    shape[axis] = n
    return 2.0 * np.sum(xs * coef.reshape(shape), axis=axis)


def _cross_term(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xs = np.sort(x, axis=0)
    prefix = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(xs, axis=0)], axis=0)
    total = prefix[-1]
    c = (xs[:, None, :] < y[None, :, :]).sum(axis=0)
    pc = np.take_along_axis(prefix, c, axis=0)
    return ((2.0 * c - n) * y - 2.0 * pc + total[None, :]).sum(axis=0) / (n * y.shape[0])


def fair_crps_crossed(nested: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Fair CRPS whose self-distance excludes pairs sharing an m or a k."""
    M, K, nc = nested.shape
    x = nested.reshape(M * K, nc)
    s = _pairsum(x, 0) - _pairsum(nested, 1).sum(axis=0) - _pairsum(nested, 0).sum(axis=0)
    return _cross_term(x, ref) - s / (2.0 * M * (M - 1) * K * (K - 1))


def _nested_view(pred, epi_id, ale_id):
    m_vals, k_vals = np.unique(epi_id), np.unique(ale_id)
    M, K = len(m_vals), len(k_vals)
    out = np.empty((M, K) + pred.shape[1:], dtype=np.float64)
    mi = {v: i for i, v in enumerate(m_vals)}
    ki = {v: i for i, v in enumerate(k_vals)}
    for n in range(pred.shape[0]):
        out[mi[epi_id[n]], ki[ale_id[n]]] = pred[n]
    return out, M, K


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def audit_event(path: Path, f_grid=F_GRID) -> dict:
    with h5py.File(path, "r") as fh:
        pred = np.asarray(fh["pred_members_wd"], dtype=np.float64)
        ref = np.asarray(fh["ref_members_wd"], dtype=np.float64)
        wettable = np.asarray(fh["wettable_mask"], dtype=bool)
        dry = np.asarray(fh["structural_dry_mask"], dtype=bool)
        epi_id = np.asarray(fh["member_epistemic_id"])
        ale_id = np.asarray(fh["member_aleatory_id"])
        event = str(fh.attrs.get("hydrograph_id", path.stem))

    nested, M, K = _nested_view(pred, epi_id, ale_id)
    T, Nv = pred.shape[1], pred.shape[2]
    active = wettable & (~dry)
    if not active.any():
        active = wettable

    nested = nested[:, :, :, active]                 # [M,K,T,nc]
    ref = ref[:, :, active]                          # [R,T,nc]
    nc = int(active.sum())

    grand = nested.mean(axis=(0, 1))                 # [T,nc]
    part_mean = nested.mean(axis=1)                  # [M,T,nc]
    epi_dev = part_mean - grand[None]                # [M,T,nc]
    ale_dev = nested - part_mean[:, None]            # [M,K,T,nc]

    var_epi = part_mean.var(axis=0, ddof=1)          # [T,nc]
    var_ale = nested.var(axis=1, ddof=1).mean(axis=0)
    var_ref = ref.var(axis=0, ddof=1)

    S_epi = float(np.sqrt(np.mean(var_epi)))
    S_ale = float(np.sqrt(np.mean(var_ale)))
    S_ref = float(np.sqrt(np.mean(var_ref)))

    ref_mean = ref.mean(axis=0)                      # [T,nc]
    abs_err = np.abs(grand - ref_mean)               # ensemble mean is scale-invariant
    front = (ref_mean > 0.01) & (ref_mean <= 0.10)

    # Total predictive spread currently realised (near the CRPS-optimal scale,
    # which for a forecast with mean error is ~sqrt(sigma_ref^2 + RMSE^2), not
    # sigma_ref).  Two families of settings are compared:
    #   refmatch_* : total shrunk to the simulator spread  (magnitude change)
    #   retain_*   : total held at the current value        (pure re-attribution)
    S_tot = float(np.sqrt(np.mean(var_epi + var_ale)))
    settings = [("current", 1.0, 1.0)]
    for f in f_grid:
        alpha = (S_ref * np.sqrt(f) / S_epi) if (f > 0 and S_epi > 0) else 0.0
        beta = S_ref * np.sqrt(1.0 - f) / S_ale if S_ale > 0 else 1.0
        settings.append((f"refmatch_f{f:g}", float(alpha), float(beta)))
    for f in f_grid:
        alpha = (S_tot * np.sqrt(f) / S_epi) if (f > 0 and S_epi > 0) else 0.0
        beta = S_tot * np.sqrt(1.0 - f) / S_ale if S_ale > 0 else 1.0
        settings.append((f"retain_f{f:g}", float(alpha), float(beta)))

    results = {}
    for label, alpha, beta in settings:
        x = grand[None, None] + alpha * epi_dev[:, None] + beta * ale_dev   # [M,K,T,nc]

        crps = float(np.mean(np.concatenate(
            [fair_crps_crossed(x[:, :, t, :], ref[:, t, :]) for t in range(T)])))

        flat = x.reshape(M * K, T, nc)
        cov, epi_cov = {}, {}
        pm = x.mean(axis=1)                          # [M,T,nc] rescaled particle means
        for lvl in NOMINAL:
            lo_q, hi_q = (1 - lvl) / 2, 1 - (1 - lvl) / 2
            lo, hi = np.quantile(flat, lo_q, axis=0), np.quantile(flat, hi_q, axis=0)
            cov[f"{int(lvl*100)}"] = float(np.mean((ref >= lo[None]) & (ref <= hi[None])))
            elo, ehi = np.quantile(pm, lo_q, axis=0), np.quantile(pm, hi_q, axis=0)
            epi_cov[f"{int(lvl*100)}"] = float(np.mean((ref_mean >= elo) & (ref_mean <= ehi)))

        sd_epi_new = np.sqrt(pm.var(axis=0, ddof=1))
        corr_front = _pearson(sd_epi_new[front], abs_err[front])

        results[label] = {
            "alpha_epistemic": alpha, "beta_aleatory": beta,
            "epistemic_spread_m": float(np.sqrt(np.mean(pm.var(axis=0, ddof=1)))),
            "aleatory_spread_m": float(np.sqrt(np.mean(x.var(axis=1, ddof=1).mean(axis=0)))),
            "total_spread_m": float(np.sqrt(np.mean(flat.var(axis=0, ddof=1)))),
            "crossed_fair_crps_m": crps,
            "coverage": cov,
            "epistemic_covers_reference_mean": epi_cov,
            "wetfront_epistemic_error_corr": corr_front,
        }

    return {"event": event, "M": M, "K": K, "R": int(ref.shape[0]),
            "sigma_ref_m": S_ref, "sigma_ale_m": S_ale, "sigma_epi_m": S_epi,
            "rmse_m": float(np.sqrt(np.mean(abs_err ** 2))), "settings": results}


def main() -> int:
    art_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    files = sorted(art_dir.glob("*.h5"))
    if limit:
        files = files[:limit]
    print(f"reallocation study over {len(files)} events", flush=True)

    rows = []
    for i, f in enumerate(files, 1):
        rows.append(audit_event(f))
        cur = rows[-1]["settings"]["current"]
        best = rows[-1]["settings"]["retain_f0.2"]
        print(f"  [{i}/{len(files)}] {rows[-1]['event']}: "
              f"crps {cur['crossed_fair_crps_m']:.5f} -> {best['crossed_fair_crps_m']:.5f} | "
              f"cov50 {cur['coverage']['50']:.3f} -> {best['coverage']['50']:.3f} | "
              f"front corr {cur['wetfront_epistemic_error_corr']:.3f} -> "
              f"{best['wetfront_epistemic_error_corr']:.3f}", flush=True)

    labels = list(rows[0]["settings"].keys())
    summary = {"n_events": len(rows),
               "rmse_m_invariant": float(np.mean([r["rmse_m"] for r in rows])),
               "sigma_ref_m": float(np.mean([r["sigma_ref_m"] for r in rows])),
               "settings": {}}
    for lab in labels:
        g = [r["settings"][lab] for r in rows]
        summary["settings"][lab] = {
            "alpha_epistemic": float(np.mean([x["alpha_epistemic"] for x in g])),
            "beta_aleatory": float(np.mean([x["beta_aleatory"] for x in g])),
            "epistemic_spread_m": float(np.mean([x["epistemic_spread_m"] for x in g])),
            "aleatory_spread_m": float(np.mean([x["aleatory_spread_m"] for x in g])),
            "total_spread_m": float(np.mean([x["total_spread_m"] for x in g])),
            "crossed_fair_crps_m": float(np.mean([x["crossed_fair_crps_m"] for x in g])),
            "coverage": {k: float(np.mean([x["coverage"][k] for x in g])) for k in ("50", "80", "90", "95")},
            "epistemic_covers_reference_mean": {
                k: float(np.mean([x["epistemic_covers_reference_mean"][k] for x in g]))
                for k in ("50", "80", "90", "95")},
            "wetfront_epistemic_error_corr": float(
                np.nanmean([x["wetfront_epistemic_error_corr"] for x in g])),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"summary": summary, "per_event": rows}, fh, indent=2)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
