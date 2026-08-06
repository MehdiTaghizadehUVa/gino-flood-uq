"""Confound-controlled reanalysis of the ALR particle-contrast probe (Phase A4).

The probe reported that scaling inter-particle contrast improves the wet-front
association between epistemic spread and error (0.284 -> 0.435 -> 0.563 -> 0.748
at x1/x5/x10/x20).  That was read as evidence the epistemic direction is
informative.  It cannot support that reading as computed, because over the same
sweep the RMSE degrades 0.0609 -> 0.0619 -> 0.0722 -> 0.1401: amplification
progressively breaks the model, and the breakage is concentrated exactly where
the dynamics are most sensitive -- the wet front.  A spread field that *causes*
error will correlate with that error no matter how uninformative it is.

The discriminating question is whether the amplified spread predicts the error
the model would have made ANYWAY.  So we correlate each setting's epistemic
spread field against two targets:

  own_error       |mean_s - ref_mean|   (what the probe did; confounded)
  baseline_error  |mean_x1 - ref_mean|  (the x1 model's error field; clean)

If `baseline_error` correlation is flat across scales while `own_error` climbs,
the reported gain is self-inflicted and the epistemic direction carries no
information about the deployed model's error.  If both climb, amplification is
genuinely resolving error structure that small perturbations leave in the noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/"
            "anchor_diversity_sensitivity_20260728_154015_58ae4f7")
SETTINGS = ["anchor_5pct", "contrast_x5", "contrast_x10", "contrast_x20"]
BASELINE = "anchor_5pct"          # the x1 reference: contrast_x* reuse this checkpoint
FRONT_LO, FRONT_HI = 0.01, 0.10   # wet-front band, matching the original probe


def _decode(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])
    return a.astype(str)


def load(path: Path):
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
    part_mean = np.stack([pred[epi == v].mean(axis=0) for v in m_vals])   # [M,T,nc]
    return {
        "event": event,
        "sd_epi": part_mean.std(axis=0, ddof=1),        # [T,nc]
        "ens_mean": pred.mean(axis=0),                  # [T,nc]
        "ref_mean": ref.mean(axis=0),                   # [T,nc]
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main() -> int:
    out_path = Path(sys.argv[1])

    # index every setting by event id
    data = {}
    for s in SETTINGS:
        d = ROOT / s / "forecast_artifacts" / "heldout"
        if not d.is_dir():
            print(f"!! missing {s}: {d}", flush=True)
            continue
        data[s] = {}
        for f in sorted(d.glob("*.h5")):
            rec = load(f)
            data[s][rec["event"]] = rec
        print(f"{s}: {len(data[s])} events {sorted(data[s])}", flush=True)

    if BASELINE not in data:
        raise SystemExit(f"baseline {BASELINE} not found")
    events = sorted(set.intersection(*[set(v) for v in data.values()]))
    print(f"\ncommon events: {events}\n", flush=True)

    rows = {}
    for s in data:
        per_event = []
        for e in events:
            cur, base = data[s][e], data[BASELINE][e]
            front = (base["ref_mean"] > FRONT_LO) & (base["ref_mean"] <= FRONT_HI)
            own_err = np.abs(cur["ens_mean"] - cur["ref_mean"])
            base_err = np.abs(base["ens_mean"] - base["ref_mean"])
            per_event.append({
                "event": e,
                "epistemic_spread_m": float(np.sqrt(np.mean(cur["sd_epi"] ** 2))),
                "rmse_m": float(np.sqrt(np.mean(own_err ** 2))),
                "corr_vs_own_error": _pearson(cur["sd_epi"][front], own_err[front]),
                "corr_vs_baseline_error": _pearson(cur["sd_epi"][front], base_err[front]),
                "n_front_cells": int(front.sum()),
            })
        rows[s] = {
            "per_event": per_event,
            "epistemic_spread_m": float(np.mean([r["epistemic_spread_m"] for r in per_event])),
            "rmse_m": float(np.mean([r["rmse_m"] for r in per_event])),
            "corr_vs_own_error": float(np.mean([r["corr_vs_own_error"] for r in per_event])),
            "corr_vs_baseline_error": float(np.mean([r["corr_vs_baseline_error"] for r in per_event])),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"baseline": BASELINE, "events": events, "settings": rows}, fh, indent=2)

    print(f"{'setting':<16} {'epi_sd':>8} {'rmse':>8} {'corr(own)':>10} {'corr(base)':>11}")
    print("-" * 58)
    for s in SETTINGS:
        if s not in rows:
            continue
        r = rows[s]
        print(f"{s:<16} {r['epistemic_spread_m']:>8.5f} {r['rmse_m']:>8.5f} "
              f"{r['corr_vs_own_error']:>10.3f} {r['corr_vs_baseline_error']:>11.3f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
