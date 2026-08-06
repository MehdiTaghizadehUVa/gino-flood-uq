"""Reconcile the reference-dispersion conventions, train vs test (probe A5).

Two numbers that look contradictory are in play:

  sigma_ref = 0.092 m   held-out audit: RMS over cells of the per-(t,cell)
                        standard deviation of the 100-member HEC-RAS ensemble
  E|H-H'|   = 0.031 m   Phase-C table: MEAN over wet cells of the dispersion
                        functional on the 10-member training ensembles

They are different statistics of a field whose maximum is ~6 m, so they cannot
be compared directly, and Gaussian conversion (sigma = E|H-H'| * sqrt(pi)/2)
is unsafe on a non-negative, strongly skewed depth field.  Getting this wrong
would mis-set the Phase-C penalty target by a large factor.

This computes BOTH statistics on BOTH populations over the same support, on the
same forecast window (train records are T=109 including spin-up; the evaluation
window is the last 94 steps), and reports:

  * the empirical sigma / E|H-H'| ratio on real depths vs the Gaussian 0.8862,
    which says how far from Gaussian the field is;
  * whether the training families the adapters fit have systematically
    different reference dispersion from the held-out families they are scored
    on -- which would bias any dispersion-pinning target transferred between
    them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

GAUSSIAN_SIGMA_OVER_MAD = float(np.sqrt(np.pi) / 2.0)   # 0.8862


def _pairsum(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    xs = np.sort(arr, axis=axis)
    n = arr.shape[axis]
    coef = (2 * np.arange(1, n + 1) - 1 - n).astype(np.float64)
    shape = [1] * arr.ndim
    shape[axis] = n
    return 2.0 * np.sum(xs * coef.reshape(shape), axis=axis)


def _stats(disp: np.ndarray, wet: np.ndarray, sd: np.ndarray | None = None) -> dict:
    out = {
        "mad_rms_all": float(np.sqrt(np.mean(disp ** 2))),
        "mad_mean_all": float(np.mean(disp)),
        "mad_rms_wet": float(np.sqrt(np.mean(disp[wet] ** 2))) if wet.any() else float("nan"),
        "mad_mean_wet": float(np.mean(disp[wet])) if wet.any() else float("nan"),
        "wet_fraction": float(wet.mean()),
    }
    if sd is not None:
        out["sigma_rms_all"] = float(np.sqrt(np.mean(sd ** 2)))
        out["sigma_rms_wet"] = float(np.sqrt(np.mean(sd[wet] ** 2))) if wet.any() else float("nan")
        out["empirical_sigma_over_mad_rms"] = out["sigma_rms_all"] / out["mad_rms_all"]
        out["empirical_sigma_over_mad_wet"] = out["sigma_rms_wet"] / out["mad_rms_wet"]
    return out


def test_population(art_dir: Path, limit: int = 0) -> dict:
    files = sorted(art_dir.glob("*.h5"))
    if limit:
        files = files[:limit]
    rows = []
    for i, f in enumerate(files, 1):
        with h5py.File(f, "r") as fh:
            ref = np.asarray(fh["ref_members_wd"], dtype=np.float64)   # [R,T,Nv]
        sd = ref.std(axis=0, ddof=1)
        disp = _pairsum(ref, 0) / (ref.shape[0] * (ref.shape[0] - 1))
        wet = ref.mean(axis=0) > 0.01
        rows.append(_stats(disp, wet, sd))
        if i % 10 == 0 or i == len(files):
            print(f"  [test {i}/{len(files)}] R={ref.shape[0]} T={ref.shape[1]}", flush=True)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"n_events": len(rows)}


def train_population(table_path: Path, n_eval_steps: int) -> dict:
    p = torch.load(table_path, map_location="cpu")
    disp = p["dispersion"].numpy().astype(np.float64)
    mean = p["reference_mean"].numpy().astype(np.float64)
    T = disp.shape[1]
    # train records include spin-up; the evaluation window is the trailing steps
    disp, mean = disp[:, T - n_eval_steps:, :], mean[:, T - n_eval_steps:, :]
    rows = [_stats(disp[i], mean[i] > 0.01) for i in range(disp.shape[0])]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {
        "n_families": int(disp.shape[0]), "n_time_used": int(disp.shape[1])}


def main() -> int:
    out_path = Path(sys.argv[1])
    test_dir = Path("/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/"
                    "n150_anchor5_no_output_projection_adapter_15ep_20260731_194436_58ae4f7/"
                    "heldout_eval/heldout_20260802_043554_58ae4f7/forecast_artifacts/heldout")
    table = Path("/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/phase_c/reference_dispersion_train.pt")

    print("=== held-out (test) families, R=100 ===", flush=True)
    te = test_population(test_dir)
    print("\n=== training families, R=10 ===", flush=True)
    tr = train_population(table, n_eval_steps=94)

    ratio = {
        "mad_rms_all_train_over_test": tr["mad_rms_all"] / te["mad_rms_all"],
        "mad_mean_wet_train_over_test": tr["mad_mean_wet"] / te["mad_mean_wet"],
    }
    payload = {"test": te, "train": tr, "train_over_test": ratio,
               "gaussian_sigma_over_mad": GAUSSIAN_SIGMA_OVER_MAD}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"{'statistic':<34} {'test (R=100)':>16} {'train (R=10)':>16}")
    print("-" * 72)
    for k in ("mad_rms_all", "mad_mean_all", "mad_rms_wet", "mad_mean_wet", "wet_fraction"):
        print(f"{k:<34} {te[k]:>16.5f} {tr[k]:>16.5f}")
    print("-" * 72)
    print(f"{'sigma_rms_all (audit convention)':<34} {te['sigma_rms_all']:>16.5f} {'n/a':>16}")
    print(f"{'empirical sigma/E|H-H| (all)':<34} {te['empirical_sigma_over_mad_rms']:>16.5f} "
          f"{'(Gaussian ' + format(GAUSSIAN_SIGMA_OVER_MAD, '.4f') + ')':>16}")
    print(f"{'empirical sigma/E|H-H| (wet)':<34} {te['empirical_sigma_over_mad_wet']:>16.5f}")
    print("-" * 72)
    print(f"train/test E|H-H'| RMS ratio : {ratio['mad_rms_all_train_over_test']:.4f}")
    print(f"train/test E|H-H'| wet-mean  : {ratio['mad_mean_wet_train_over_test']:.4f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
