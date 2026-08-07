"""ALR-FGNO dispersion / calibration / scoring audit.

Answers four questions from the saved held-out forecast artifacts, with no
retraining and no new inference:

1. OVER-DISPERSION: is the FGNO predictive spread wider than the HEC-RAS
   simulator spread it is supposed to reproduce?  Reports sigma_pred/sigma_ref
   on the wettable domain, stratified by wetness regime and lead time, plus the
   implied "excess" standard deviation in metres (compared against the
   deep-ensemble epistemic scale ~0.046 m).
2. CALIBRATION: coverage of nominal 50/80/90/95% predictive intervals against
   reference members -- computed BOTH over all cells (dry-inflated, the way it
   was previously reported) and over the wettable mask.
3. EPISTEMIC TARGET: coverage of the reference-ensemble MEAN by the epistemic
   interval (the quantity the methodology chapter specifies for epistemic
   validity, never previously computed).
4. SCORING: flattened fair CRPS (what the 60-member comparison used) versus the
   design-aware crossed estimator, to size the duplicate-member artifact.

Run inside the pytorch container with PYTHONPATH=<repo>.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


# --- self-contained fair-CRPS estimators (repo-independent) ---------------
# Sum_{i!=j} |x_i - x_j| via order statistics: 2 * sum_i (2i-1-n) x_(i).
def _pairsum(arr: np.ndarray, axis: int) -> np.ndarray:
    xs = np.sort(arr, axis=axis)
    n = arr.shape[axis]
    coef = (2 * np.arange(1, n + 1) - 1 - n).astype(np.float64)
    shape = [1] * arr.ndim
    shape[axis] = n
    return 2.0 * np.sum(xs * coef.reshape(shape), axis=axis)


def _cross_term(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(1/(N R)) sum_n sum_r |x_n - y_r| per column, via order statistics."""
    n = x.shape[0]
    xs = np.sort(x, axis=0)
    prefix = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(xs, axis=0)], axis=0)
    total = prefix[-1]
    c = (xs[:, None, :] < y[None, :, :]).sum(axis=0)              # [R, nc]
    pc = np.take_along_axis(prefix, c, axis=0)                    # [R, nc]
    contrib = (2.0 * c - n) * y - 2.0 * pc + total[None, :]
    return contrib.sum(axis=0) / (n * y.shape[0])


def fair_crps_flat(nested: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Ordinary fair CRPS on the flattened M*K ensemble (treats members iid)."""
    M, K, nc = nested.shape
    x = nested.reshape(M * K, nc)
    n = M * K
    return _cross_term(x, ref) - _pairsum(x, 0) / (2.0 * n * (n - 1))


def fair_crps_crossed(nested: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Design-aware fair CRPS: self-distance excludes pairs sharing m or k."""
    M, K, nc = nested.shape
    x = nested.reshape(M * K, nc)
    s_all = _pairsum(x, 0)
    s_same_m = _pairsum(nested, 1).sum(axis=0)   # pairs within a particle
    s_same_k = _pairsum(nested, 0).sum(axis=0)   # pairs sharing an aleatory draw
    s_crossed = s_all - s_same_m - s_same_k
    return _cross_term(x, ref) - s_crossed / (2.0 * M * (M - 1) * K * (K - 1))


def _selftest() -> None:
    """A collapsed epistemic axis must reproduce the K-member fair CRPS."""
    rng = np.random.default_rng(0)
    K, nc, R = 15, 40, 100
    base = rng.normal(size=(K, nc))
    nested = np.repeat(base[None, :, :], 4, axis=0)       # 4 identical particles
    ref = rng.normal(size=(R, nc))
    single = base[None, :, :]                              # M=1 view is undefined; use direct
    n = K
    xs = base
    direct = _cross_term(xs, ref) - _pairsum(xs, 0) / (2.0 * n * (n - 1))
    crossed = fair_crps_crossed(nested, ref)
    assert np.allclose(direct, crossed, rtol=1e-10, atol=1e-12), "crossed estimator failed collapse identity"
    flat = fair_crps_flat(nested, ref)
    assert np.all(flat > crossed), "flattened estimator should be inflated under duplication"
    print(f"[selftest] crossed==K-member fair CRPS OK; flattened inflated by "
          f"{100*float(np.mean(flat-crossed)/np.mean(crossed)):.2f}% on duplicated fixture", flush=True)


DE_EPISTEMIC_SPREAD_M = 0.046  # 3-checkpoint deep-ensemble between-model scale
NOMINAL = (0.50, 0.80, 0.90, 0.95)


def _nested_view(pred: np.ndarray, epi_id: np.ndarray, ale_id: np.ndarray):
    """Reorder [N,T,Nv] members into [M,K,T,Nv] using stored nested IDs."""
    m_vals = np.unique(epi_id)
    k_vals = np.unique(ale_id)
    M, K = len(m_vals), len(k_vals)
    if M * K != pred.shape[0]:
        raise ValueError(f"nested ids do not tile members: M={M} K={K} N={pred.shape[0]}")
    out = np.empty((M, K) + pred.shape[1:], dtype=pred.dtype)
    m_index = {v: i for i, v in enumerate(m_vals)}
    k_index = {v: i for i, v in enumerate(k_vals)}
    for n in range(pred.shape[0]):
        out[m_index[epi_id[n]], k_index[ale_id[n]]] = pred[n]
    return out, M, K


def _masked_rms(x: np.ndarray, mask: np.ndarray) -> float:
    """RMS of x[..., cells] restricted to mask cells."""
    sel = x[..., mask]
    return float(np.sqrt(np.mean(sel.astype(np.float64) ** 2)))


def _coverage(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, level: float) -> float:
    """Fraction of reference members inside the central predictive interval."""
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lo = np.quantile(pred[:, :, mask], lo_q, axis=0)
    hi = np.quantile(pred[:, :, mask], hi_q, axis=0)
    r = ref[:, :, mask]
    inside = (r >= lo[None]) & (r <= hi[None])
    return float(np.mean(inside))


def _epistemic_mean_coverage(nested: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                             level: float) -> float:
    """Coverage of the reference-ensemble MEAN by the epistemic interval.

    The epistemic interval is formed from the M particle means (each averaged
    over its K aleatory draws) -- this is the interval that should bracket the
    conditional mean if the epistemic component is calibrated.
    """
    particle_means = nested.mean(axis=1)  # [M, T, Nv]
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lo = np.quantile(particle_means[:, :, mask], lo_q, axis=0)
    hi = np.quantile(particle_means[:, :, mask], hi_q, axis=0)
    ref_mean = ref[:, :, mask].mean(axis=0)
    return float(np.mean((ref_mean >= lo) & (ref_mean <= hi)))


def audit_event(path: Path) -> dict:
    with h5py.File(path, "r") as fh:
        pred = np.asarray(fh["pred_members_wd"], dtype=np.float32)   # [N,T,Nv]
        ref = np.asarray(fh["ref_members_wd"], dtype=np.float32)     # [R,T,Nv]
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

    # ---- 1. dispersion: predictive spread vs simulator spread -------------
    sd_pred = pred.std(axis=0, ddof=1)   # [T,Nv]
    sd_ref = ref.std(axis=0, ddof=1)     # [T,Nv]
    ref_mean_depth = ref.mean(axis=0)    # [T,Nv]

    rms_pred = _masked_rms(sd_pred, active)
    rms_ref = _masked_rms(sd_ref, active)
    excess_var = max(0.0, rms_pred ** 2 - rms_ref ** 2)

    # wetness-stratified dispersion (cell-time selection)
    strata = {}
    active_2d = np.broadcast_to(active[None, :], (T, Nv))
    for name, sel in (
        ("wet_gt_0p01m", active_2d & (ref_mean_depth > 0.01)),
        ("wet_front_0p01_0p10m", active_2d & (ref_mean_depth > 0.01) & (ref_mean_depth <= 0.10)),
        ("deep_gt_0p10m", active_2d & (ref_mean_depth > 0.10)),
    ):
        if sel.any():
            p = float(np.sqrt(np.mean(sd_pred[sel].astype(np.float64) ** 2)))
            r = float(np.sqrt(np.mean(sd_ref[sel].astype(np.float64) ** 2)))
            strata[name] = {
                "sigma_pred_m": p,
                "sigma_ref_m": r,
                "ratio": p / r if r > 0 else float("nan"),
                "n_cells_times": int(sel.sum()),
            }

    # lead-time thirds
    lead = {}
    edges = [(0, T // 3, "early"), (T // 3, 2 * T // 3, "mid"), (2 * T // 3, T, "late")]
    for a, b, name in edges:
        p = float(np.sqrt(np.mean(sd_pred[a:b][:, active].astype(np.float64) ** 2)))
        r = float(np.sqrt(np.mean(sd_ref[a:b][:, active].astype(np.float64) ** 2)))
        lead[name] = {"sigma_pred_m": p, "sigma_ref_m": r, "ratio": p / r if r > 0 else float("nan")}

    # ---- 2/3. calibration --------------------------------------------------
    all_cells = np.ones(Nv, dtype=bool)
    coverage_all = {f"{int(l*100)}": _coverage(pred, ref, all_cells, l) for l in NOMINAL}
    coverage_wet = {f"{int(l*100)}": _coverage(pred, ref, active, l) for l in NOMINAL}
    epi_cov = {f"{int(l*100)}": _epistemic_mean_coverage(nested, ref, active, l) for l in NOMINAL}

    # ---- 4. scoring: flattened vs crossed fair CRPS ------------------------
    flat_acc, crossed_acc = [], []
    for t in range(T):
        xt = nested[:, :, t, :][:, :, active].astype(np.float64)      # [M,K,nc]
        yt = ref[:, t, :][:, active].astype(np.float64)               # [R,nc]
        flat_acc.append(fair_crps_flat(xt, yt))
        crossed_acc.append(fair_crps_crossed(xt, yt))
    crps_flat = float(np.mean(np.concatenate(flat_acc)))
    crps_crossed = float(np.mean(np.concatenate(crossed_acc)))

    # Nested decomposition, on the same active mask and the same RMS reduction
    # as sigma_ref above, so gate GC compares like with like.
    particle_mean = nested.mean(axis=1)                       # [M, T, Nv]
    sigma_epi = float(np.sqrt(np.mean(
        particle_mean.var(axis=0, ddof=1)[..., active]))) if M > 1 else 0.0
    sigma_ale = float(np.sqrt(np.mean(
        nested.var(axis=1, ddof=1).mean(axis=0)[..., active])))
    rmse_m = float(np.sqrt(np.mean(
        (pred.mean(axis=0)[..., active] - ref.mean(axis=0)[..., active]) ** 2)))

    return {
        "event": event,
        "M": M, "K": K, "R": int(ref.shape[0]), "T": T, "n_active_cells": int(active.sum()),
        "decomposition": {
            "epistemic_spread_m": sigma_epi,
            "aleatory_spread_m": sigma_ale,
            "aleatory_over_reference": sigma_ale / rms_ref if rms_ref > 0 else float("nan"),
            "epistemic_variance_fraction": (
                sigma_epi ** 2 / (sigma_epi ** 2 + sigma_ale ** 2)
                if (sigma_epi ** 2 + sigma_ale ** 2) > 0 else float("nan")
            ),
            "rmse_m": rmse_m,
        },
        "dispersion": {
            "sigma_pred_m": rms_pred,
            "sigma_ref_m": rms_ref,
            "ratio_pred_over_ref": rms_pred / rms_ref if rms_ref > 0 else float("nan"),
            "excess_sigma_m": float(np.sqrt(excess_var)),
            "excess_vs_deep_ensemble_ratio": float(np.sqrt(excess_var) / DE_EPISTEMIC_SPREAD_M),
            "by_wetness": strata,
            "by_lead_third": lead,
        },
        "coverage_all_cells": coverage_all,
        "coverage_wettable": coverage_wet,
        "epistemic_interval_covers_reference_mean": epi_cov,
        "crps": {
            "flattened_fair_crps_m": crps_flat,
            "crossed_fair_crps_m": crps_crossed,
            "flattened_minus_crossed_m": crps_flat - crps_crossed,
            "flattened_inflation_percent": 100.0 * (crps_flat - crps_crossed) / crps_crossed
            if crps_crossed > 0 else float("nan"),
        },
    }


def main() -> int:
    artifact_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    _selftest()
    files = sorted(artifact_dir.glob("*.h5"))
    if limit:
        files = files[:limit]
    print(f"auditing {len(files)} events from {artifact_dir}", flush=True)

    rows = []
    for i, f in enumerate(files, 1):
        rows.append(audit_event(f))
        d = rows[-1]["dispersion"]
        print(f"  [{i}/{len(files)}] {rows[-1]['event']}: "
              f"sigma_pred/sigma_ref={d['ratio_pred_over_ref']:.3f} "
              f"crps_flat={rows[-1]['crps']['flattened_fair_crps_m']:.6f} "
              f"crps_crossed={rows[-1]['crps']['crossed_fair_crps_m']:.6f}", flush=True)

    def agg(fn):
        return float(np.mean([fn(r) for r in rows]))

    summary = {
        "n_events": len(rows),
        "dispersion": {
            "sigma_pred_m": agg(lambda r: r["dispersion"]["sigma_pred_m"]),
            "sigma_ref_m": agg(lambda r: r["dispersion"]["sigma_ref_m"]),
            "ratio_pred_over_ref_mean": agg(lambda r: r["dispersion"]["ratio_pred_over_ref"]),
            "excess_sigma_m_mean": agg(lambda r: r["dispersion"]["excess_sigma_m"]),
            "excess_vs_deep_ensemble_ratio_mean": agg(
                lambda r: r["dispersion"]["excess_vs_deep_ensemble_ratio"]),
        },
        "decomposition": {
            "epistemic_spread_m": agg(lambda r: r["decomposition"]["epistemic_spread_m"]),
            "aleatory_spread_m": agg(lambda r: r["decomposition"]["aleatory_spread_m"]),
            "aleatory_over_reference": agg(lambda r: r["decomposition"]["aleatory_over_reference"]),
            "epistemic_variance_fraction": agg(
                lambda r: r["decomposition"]["epistemic_variance_fraction"]),
            "rmse_m": agg(lambda r: r["decomposition"]["rmse_m"]),
        },
        "coverage_all_cells": {k: agg(lambda r, k=k: r["coverage_all_cells"][k]) for k in ("50", "80", "90", "95")},
        "coverage_wettable": {k: agg(lambda r, k=k: r["coverage_wettable"][k]) for k in ("50", "80", "90", "95")},
        "epistemic_interval_covers_reference_mean": {
            k: agg(lambda r, k=k: r["epistemic_interval_covers_reference_mean"][k])
            for k in ("50", "80", "90", "95")},
        "crps": {
            "flattened_fair_crps_m": agg(lambda r: r["crps"]["flattened_fair_crps_m"]),
            "crossed_fair_crps_m": agg(lambda r: r["crps"]["crossed_fair_crps_m"]),
            "flattened_inflation_percent_mean": agg(lambda r: r["crps"]["flattened_inflation_percent"]),
        },
    }
    for name in ("wet_gt_0p01m", "wet_front_0p01_0p10m", "deep_gt_0p10m"):
        vals = [r["dispersion"]["by_wetness"][name]["ratio"] for r in rows
                if name in r["dispersion"]["by_wetness"]]
        if vals:
            summary["dispersion"][f"ratio_{name}"] = float(np.mean(vals))
    for name in ("early", "mid", "late"):
        summary["dispersion"][f"ratio_lead_{name}"] = agg(
            lambda r, n=name: r["dispersion"]["by_lead_third"][n]["ratio"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"summary": summary, "per_event": rows}, fh, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
