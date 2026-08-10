"""Precompute the per-family HEC-RAS dispersion table (Phase C1).

Motivation
----------
Fair CRPS is proper, so under misspecification it does exactly what it should:
a forecast carrying location error `b` against a truth of spread `tau` has a
CRPS-optimal scale satisfying

    sigma / sqrt(sigma^2 + tau^2) = exp(b^2 / (2(sigma^2 + tau^2))) / sqrt(2)

which returns sigma = tau only when b = 0.  Stage-1 has a single dispersion
channel, so all of that error-covering variance was booked as *aleatory*, and
the epistemic channel was left with ~0.5% of the total.  No proper scoring rule
on the pooled predictive distribution can undo this: the reallocation study
moved the epistemic share by >10x and shifted crossed CRPS by only 1.4-3.0%.

Identification has to come from outside the score, and we already hold the
information -- the HEC-RAS reference ensembles.  This script precomputes the
reference *dispersion functional*

    D_ref[family, t, cell] = E|H - H'| = (1/(R(R-1))) sum_{r != r'} |H_r - H_r'|

so training can pin the within-particle channel to it.  D_ref is the same
functional that already appears inside fair CRPS as the self-distance term, and
at K=2 the model side is estimated by |X_1 - X_2|, which is unbiased for
E|X - X'| with no variance estimator required.

Also emits the per-family reference mean, which Phase F (location/scale
decoupled Stage-1) needs, because the I/O dominates and a second pass would be
pure waste.

Why not reuse the existing reader
---------------------------------
`read_hec_ras_hdf_run_series` eagerly reads wd, vx AND vy, which would triple
the I/O to ~39 GB for a depth-only table.  This reads only Cell Hydraulic Depth.

Usage
-----
    # shard (slurm array)
    python build_reference_dispersion_table.py shard --shard I --num-shards N --output DIR
    # merge
    python build_reference_dispersion_table.py merge --shards DIR --output table.pt
    # validate estimator against brute force
    python build_reference_dispersion_table.py selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralop.flood.data.hec_ras import HDF_PATHS, build_cell_point_index  # noqa: E402
from neuralop.flood.utils.runtime_core import parse_family_id_from_run_id  # noqa: E402

import h5py  # noqa: E402

DEFAULT_RESULTS = Path(
    "/scratch/jrj6wm/uncertainty_floodmodel_linux/results/coastal/"
    "Coastal_Flood_coastal_v1_5k_train_prod_t2_w64_20260318_233556/train"
)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / (
    "config/flood/coastal/gino_pluvial_flood_config_coastal_alr_fgn_pilot.yaml"
)


def resolve_hdf_paths(config_path: Path) -> dict:
    """Take the dataset's HDF layout from the project config, not the module default.

    The module-level HDF_PATHS names the 2D flow area generically ("Flow Area"),
    but this dataset's area is "Portsmouth" and the real layout is carried in
    flood.data.hdf_paths.  Reading it from the config keeps this script correct
    if the mesh is ever renamed, instead of hardcoding a second copy.
    """
    import yaml
    with Path(config_path).open() as fh:
        cfg = yaml.safe_load(fh)
    override = (cfg.get("flood", {}) or {}).get("data", {}).get("hdf_paths") or {}
    if not override:
        raise SystemExit(f"{config_path}: flood.data.hdf_paths is missing")
    paths = dict(HDF_PATHS)
    paths.update(override)
    return paths


def _pairsum(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sum over ORDERED pairs r != r' of |x_r - x_r'|, via 2*sum_i (2i-1-n) x_(i).

    O(R log R) instead of the O(R^2) double loop; identical value.
    """
    xs = np.sort(arr, axis=axis)
    n = arr.shape[axis]
    coef = (2 * np.arange(1, n + 1) - 1 - n).astype(np.float64)
    shape = [1] * arr.ndim
    shape[axis] = n
    return 2.0 * np.sum(xs * coef.reshape(shape), axis=axis)


def dispersion_from_members(members: np.ndarray) -> np.ndarray:
    """E|H - H'| from R members, shape [R, ...] -> [...]. Unbiased."""
    R = members.shape[0]
    if R < 2:
        raise ValueError("dispersion needs at least 2 reference members")
    return _pairsum(members, axis=0) / (R * (R - 1))


def reference_mean_variance_from_members(members: np.ndarray) -> np.ndarray:
    """Estimated sampling variance of the finite-ensemble mean, ``s^2/R``."""
    R = members.shape[0]
    if R < 2:
        raise ValueError("reference mean variance needs at least 2 members")
    return members.var(axis=0, ddof=1) / R


def _group_families(run_ids: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for rid in run_ids:
        families.setdefault(parse_family_id_from_run_id(rid), []).append(rid)
    return {k: sorted(v) for k, v in sorted(families.items())}


def _read_depth(path: Path, cell_index: np.ndarray, paths: dict) -> np.ndarray:
    """Read ONLY Cell Hydraulic Depth for the modelled cell points. [T, Nv]"""
    with h5py.File(path, "r") as fh:
        wd = np.asarray(fh[paths["wd"]][:, :], dtype=np.float32)
    return wd[:, cell_index]


def build_shard(results_dir: Path, shard: int, num_shards: int, out_dir: Path,
                paths: dict) -> Path:
    run_list = sorted(p.stem for p in results_dir.glob("*.hdf"))
    if not run_list:
        raise SystemExit(f"no .hdf files under {results_dir}")
    families = _group_families(run_list)
    keys = sorted(families)
    mine = keys[shard::num_shards]
    print(f"shard {shard}/{num_shards}: {len(mine)} of {len(keys)} families", flush=True)

    cell_index = build_cell_point_index(results_dir / f"{families[keys[0]][0]}.hdf", paths)
    Nv = int(len(cell_index))

    disp, mean, mean_variance, ids, T_ref = [], [], [], [], None
    for i, fam in enumerate(mine, 1):
        members = np.stack([_read_depth(results_dir / f"{r}.hdf", cell_index, paths)
                            for r in families[fam]])          # [R, T, Nv]
        if T_ref is None:
            T_ref = members.shape[1]
        elif members.shape[1] != T_ref:
            raise ValueError(f"{fam}: T={members.shape[1]} != {T_ref}; ragged records unsupported")
        m64 = members.astype(np.float64)
        disp.append(dispersion_from_members(m64).astype(np.float32))
        mean.append(m64.mean(axis=0).astype(np.float32))
        mean_variance.append(reference_mean_variance_from_members(m64).astype(np.float32))
        ids.append(fam)
        if i % 25 == 0 or i == len(mine):
            print(f"  [{i}/{len(mine)}] {fam} R={members.shape[0]} T={members.shape[1]}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{shard:04d}.pt"
    tmp = out.with_suffix(".pt.tmp")
    torch.save({
        "family_ids": ids,
        "dispersion": torch.from_numpy(np.stack(disp)),   # [F, T, Nv]
        "reference_mean": torch.from_numpy(np.stack(mean)),
        "reference_mean_variance": torch.from_numpy(np.stack(mean_variance)),
        "cell_count": Nv,
        "n_time": int(T_ref),
        "run_ids": [r for f in ids for r in families[f]],
    }, tmp)
    os.replace(tmp, out)
    print(f"wrote {out}", flush=True)
    return out


def merge(shard_dir: Path, out_path: Path) -> None:
    """Assemble shards into one table.

    Done in two passes and scattered into preallocated tensors: `cat` followed
    by a fancy-index reorder would hold three copies of a 1.3 GiB field at once
    and was killed by the login-node memory cap.  Peak here is one output pair
    plus a single shard.
    """
    shards = sorted(shard_dir.glob("shard_*.pt"))
    if not shards:
        raise SystemExit(f"no shards under {shard_dir}")

    # pass 1: global family ordering and geometry, holding no bulk tensors
    per_shard_ids, runs, Nv, T = [], [], None, None
    for s in shards:
        p = torch.load(s, map_location="cpu")
        if Nv is None:
            Nv, T = int(p["cell_count"]), int(p["n_time"])
        elif (int(p["cell_count"]), int(p["n_time"])) != (Nv, T):
            raise ValueError(f"{s.name}: shape ({p['cell_count']},{p['n_time']}) != ({Nv},{T})")
        per_shard_ids.append(list(p["family_ids"]))
        runs += list(p["run_ids"])
        del p

    ids = sorted(f for group in per_shard_ids for f in group)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate family ids across shards")
    index = {f: i for i, f in enumerate(ids)}
    F = len(ids)
    print(f"merging {len(shards)} shards -> {F} families, T={T}, Nv={Nv}", flush=True)

    dispersion = torch.empty((F, T, Nv), dtype=torch.float32)
    reference_mean = torch.empty((F, T, Nv), dtype=torch.float32)
    reference_mean_variance = torch.empty((F, T, Nv), dtype=torch.float32)

    # pass 2: scatter each shard's rows into their global positions
    for s, group in zip(shards, per_shard_ids):
        p = torch.load(s, map_location="cpu")
        rows = torch.tensor([index[f] for f in group], dtype=torch.long)
        dispersion[rows] = p["dispersion"].to(torch.float32)
        reference_mean[rows] = p["reference_mean"].to(torch.float32)
        reference_mean_variance[rows] = p["reference_mean_variance"].to(torch.float32)
        del p

    artifact = {
        "schema": "flood_reference_dispersion_v1",
        "family_ids": ids,
        "family_index": index,
        "dispersion": dispersion,
        "reference_mean": reference_mean,
        "reference_mean_variance": reference_mean_variance,
        "cell_count": int(Nv),
        "n_time": int(T),
        "n_families": F,
        "run_ids": sorted(runs),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(artifact, tmp)
    os.replace(tmp, out_path)

    summary = {k: artifact[k] for k in ("schema", "cell_count", "n_time", "n_families")}
    summary["dispersion_mean_m"] = float(artifact["dispersion"].mean())
    summary["dispersion_wet_mean_m"] = float(
        artifact["dispersion"][artifact["reference_mean"] > 0.01].mean())
    summary["reference_mean_se_rms_m"] = float(
        artifact["reference_mean_variance"].mean().sqrt())
    summary["n_runs"] = len(artifact["run_ids"])
    with out_path.with_suffix(".summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size / 2**30:.2f} GiB)")


def selftest() -> None:
    """The order-statistic estimator must equal brute force, and be unbiased."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 7, 13)) * rng.uniform(0.5, 2.0)
    R = x.shape[0]
    brute = np.zeros(x.shape[1:])
    for r in range(R):
        for s in range(R):
            if r != s:
                brute += np.abs(x[r] - x[s])
    brute /= R * (R - 1)
    fast = dispersion_from_members(x)
    assert np.allclose(brute, fast, rtol=1e-12, atol=1e-14), "order-statistic dispersion != brute force"
    print(f"[selftest 1] order-statistic == brute force (max abs diff "
          f"{np.abs(brute - fast).max():.2e})", flush=True)

    # unbiasedness for E|H - H'| of N(0,1): analytic value 2/sqrt(pi)
    target = 2.0 / np.sqrt(np.pi)
    est = [float(dispersion_from_members(rng.normal(size=(10, 1)))[0]) for _ in range(20000)]
    bias, se = float(np.mean(est)) - target, float(np.std(est, ddof=1) / np.sqrt(20000))
    assert abs(bias) < 4 * se, f"dispersion estimator biased: {bias:+.5f} vs 4*se {4*se:.5f}"
    print(f"[selftest 2] R=10 estimator unbiased for E|H-H'|: target {target:.5f} "
          f"est {np.mean(est):.5f} bias {bias:+.5f} (4*se {4*se:.5f})", flush=True)
    known = np.array([[0.0], [2.0], [4.0]], dtype=np.float64)
    np.testing.assert_allclose(
        reference_mean_variance_from_members(known),
        np.array([4.0 / 3.0]),
    )
    print("[selftest 3] reference mean sampling variance == s^2/R", flush=True)
    print("[selftest] passed", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    s.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    s.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--shards", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()

    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "shard":
        selftest()
        paths = resolve_hdf_paths(a.config)
        print(f"hdf wd path: {paths['wd']}", flush=True)
        build_shard(a.results_dir, a.shard, a.num_shards, a.output, paths)
    else:
        merge(a.shards, a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
