"""Calibrate the dispersion-pinning weight from saved artifacts (Phase C4).

Reading the penalty out of a training log turned out not to work: the trainer's
per-batch metrics dict never reaches the epoch summary line, and the validation
path does not compute the penalty at all.  Rather than iterate on logging, the
same quantity is reconstructed here from the held-out artifacts, which carry
both the nested forecasts and the 100-member references -- everything the
penalty needs -- on 50 real events instead of a 2-family smoke.

Computes exactly what ``dispersion_pinning_penalty`` computes:

    per stratum: mean over cells of (E|X - X'|_within-particle  -  E|H - H'|)
    penalty    : sum over strata of that mean, squared

then reports the lambda that puts the penalty at a target share of the training
objective.

Two differences from the training-time value, both quantified and small:
  * K: artifacts have 15 aleatory draws per particle, training has 2.  Both give
    an unbiased E|X - X'|, so this shifts variance, not the mean.
  * population: this is measured on held-out families; training families carry
    1.070x the reference dispersion (probe A5).  Reported both ways below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

TRAIN_OVER_TEST_DISPERSION = 1.070
WET_THRESHOLDS = (0.01, 0.10)


def _pairsum(arr: np.ndarray, axis: int) -> np.ndarray:
    xs = np.sort(arr, axis=axis)
    n = arr.shape[axis]
    coef = (2 * np.arange(1, n + 1) - 1 - n).astype(np.float64)
    shape = [1] * arr.ndim
    shape[axis] = n
    return 2.0 * np.sum(xs * coef.reshape(shape), axis=axis)


def _mean_abs_pairwise(x: np.ndarray, axis: int) -> np.ndarray:
    """Unbiased E|X - X'| along ``axis``."""
    n = x.shape[axis]
    return _pairsum(x, axis) / (n * (n - 1))


def _decode(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])
    return a.astype(str)


def event_penalty(path: Path) -> dict:
    with h5py.File(path, "r") as fh:
        pred = np.asarray(fh["pred_members_wd"], dtype=np.float64)
        ref = np.asarray(fh["ref_members_wd"], dtype=np.float64)
        wettable = np.asarray(fh["wettable_mask"], dtype=bool)
        dry = np.asarray(fh["structural_dry_mask"], dtype=bool)
        epi = _decode(fh["member_epistemic_id"])
        event = str(fh.attrs.get("hydrograph_id", path.stem))

    active = wettable & (~dry)
    if not active.any():
        active = wettable
    pred, ref = pred[:, :, active], ref[:, :, active]

    m_vals = list(dict.fromkeys(epi))
    nested = np.stack([pred[epi == v] for v in m_vals])          # [M, K, T, nc]

    model_disp = np.stack([_mean_abs_pairwise(nested[m], axis=0)
                           for m in range(nested.shape[0])])     # [M, T, nc]
    ref_disp = _mean_abs_pairwise(ref, axis=0)                   # [T, nc]
    ref_mean = ref.mean(axis=0)                                  # [T, nc]

    deviation = model_disp - ref_disp[None]
    lo, hi = WET_THRESHOLDS
    strata = {"dry": ref_mean <= lo,
              "front": (ref_mean > lo) & (ref_mean <= hi),
              "deep": ref_mean > hi}

    penalty, means, counts = 0.0, {}, {}
    for name, sel in strata.items():
        w = np.broadcast_to(sel[None], deviation.shape)
        n = int(w.sum())
        mean_dev = float((deviation * w).sum() / max(n, 1))
        penalty += mean_dev ** 2
        means[name] = mean_dev
        counts[name] = n

    return {"event": event, "penalty_m2": penalty,
            "per_stratum_mean_deviation_m": means, "per_stratum_count": counts,
            "model_dispersion_m": float(np.sqrt(np.mean(model_disp ** 2))),
            "reference_dispersion_m": float(np.sqrt(np.mean(ref_disp ** 2)))}


def main() -> int:
    art_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    train_loss = float(sys.argv[3])
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    files = sorted(art_dir.glob("*.h5"))
    if limit:
        files = files[:limit]
    rows = []
    for i, f in enumerate(files, 1):
        rows.append(event_penalty(f))
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {rows[-1]['event']}: penalty {rows[-1]['penalty_m2']:.3e}",
                  flush=True)

    penalty = float(np.mean([r["penalty_m2"] for r in rows]))
    # in-sample dispersion is 1.070x, so the deviation the trainer sees is smaller
    ref_scale = float(np.mean([r["reference_dispersion_m"] for r in rows]))
    model_scale = float(np.mean([r["model_dispersion_m"] for r in rows]))
    dev_train = model_scale - ref_scale * TRAIN_OVER_TEST_DISPERSION
    penalty_train = penalty * (dev_train / max(model_scale - ref_scale, 1e-12)) ** 2

    lambdas = {f"{int(frac*100)}pct": frac * train_loss / penalty
               for frac in (0.05, 0.10, 0.15, 0.20)}

    summary = {
        "n_events": len(rows),
        "training_objective_used": train_loss,
        "penalty_m2_heldout": penalty,
        "penalty_m2_train_scale_estimate": penalty_train,
        "model_dispersion_m": model_scale,
        "reference_dispersion_m_heldout": ref_scale,
        "reference_dispersion_m_train_scale": ref_scale * TRAIN_OVER_TEST_DISPERSION,
        "per_stratum_mean_deviation_m": {
            k: float(np.mean([r["per_stratum_mean_deviation_m"][k] for r in rows]))
            for k in ("dry", "front", "deep")},
        "lambda_for_target_share": lambdas,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_event": rows}, indent=2))

    print("\n===== LAMBDA CALIBRATION =====")
    print(f"model within-particle E|X-X'| : {model_scale:.5f} m")
    print(f"reference E|H-H'| (held-out)  : {ref_scale:.5f} m")
    print(f"reference E|H-H'| (train scl) : {ref_scale*TRAIN_OVER_TEST_DISPERSION:.5f} m")
    print("per-stratum mean deviation    : " + ", ".join(
        f"{k}={summary['per_stratum_mean_deviation_m'][k]:+.5f}" for k in ("dry", "front", "deep")))
    print(f"penalty (held-out)            : {penalty:.4e} m^2")
    print(f"penalty (train-scale est.)    : {penalty_train:.4e} m^2")
    print(f"training objective            : {train_loss:.6f}")
    print("\nlambda for a target penalty share of the objective:")
    for k, v in lambdas.items():
        print(f"   {k:>6} -> lambda = {v:.4g}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
