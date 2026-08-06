"""Design-aware rescoring of ALR-FGNO against its baselines (Phase A1/A2/A3).

Why this exists
---------------
The pilot comparison scored every method with the *flattened* fair CRPS, which
treats all 60 forecast members as exchangeable draws.  That assumption is false
for both nested designs, and it is false in *different ways*:

  ALR-FGNO   M=4 particles x K=15 aleatory draws from a SHARED bank (common
             random numbers).  Members sharing an aleatory draw are near
             duplicates (measured particle correlation 0.9999), so the
             flattened self-distance term collapses and the score is inflated.
  Deep ens.  J=3 checkpoints x 20 INDEPENDENT draws each (verified: the stored
             member_sample_id runs 0..59 with no repeats across checkpoints).
             Only the missing epistemic component biases its self-distance, a
             much smaller effect.

Scoring both with the same flattened estimator therefore penalised ALR and not
the deep ensemble.  This script rescores every method under three estimators
and reports all of them, because the "right" target is a modelling choice:

  flattened   status quo.  Self-distance over all N(N-1) ordered pairs.
              Biased for any nested design; reported for continuity only.

  pooled      The score of the finite ensemble actually shipped, i.e. of the
              M-component mixture (1/M) sum_m P_m.  Two independent draws pick
              the same component with probability 1/M:
                  E|X-X'| = (1/M) * within + ((M-1)/M) * between

  fair_inf    The fair (infinite-particle) target: particles are exchangeable
              draws from a posterior, so two independent draws of the
              predictive carry *distinct* parameters almost surely and the
              same-particle pairs drop out entirely:
                  E|X-X'| = between

`within` and `between` are always estimated from pairs that are genuinely
independent in both factors.  For a shared aleatory bank that means excluding
pairs which share an aleatory index as well as pairs which share a particle;
for independent nesting only the particle axis needs excluding.  Getting this
distinction right is the whole point of the script.

Also computes (A2/A3) the per-epistemic-member dispersion, which tests whether
each deep-ensemble checkpoint is *itself* over-dispersed -- if so its reported
between-checkpoint spread double counts variance that is really aleatory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

MEMBERS = 60          # match the published 60-member comparison
BOOTSTRAP_REPS = 10000


# --- order-statistic estimators -------------------------------------------
def _pairsum(arr: np.ndarray, axis: int) -> np.ndarray:
    """Sum over ORDERED pairs i!=j of |x_i - x_j|, via 2*sum_i (2i-1-n) x_(i)."""
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
    c = (xs[:, None, :] < y[None, :, :]).sum(axis=0)
    pc = np.take_along_axis(prefix, c, axis=0)
    return ((2.0 * c - n) * y - 2.0 * pc + total[None, :]).sum(axis=0) / (n * y.shape[0])


def self_distance_variants(nested: np.ndarray, shared_bank: bool) -> dict:
    """Estimates of E|X - X'| under each target, from a nested [M, K, nc] design.

    ``within`` and ``between`` are always built from pairs independent in BOTH
    factors.  Under a shared aleatory bank the (m != m', k == k') pairs share
    their latent draw and would bias ``between`` toward the pure parameter
    separation |f_m - f_m'|, so they are excluded; under independent nesting no
    such pairs exist and every cross-particle pair is usable.
    """
    M, K, _ = nested.shape
    x = nested.reshape(M * K, nested.shape[-1])
    n = M * K

    s_all = _pairsum(x, 0)
    s_within = _pairsum(nested, 1).sum(axis=0)          # same particle, k != k'
    if shared_bank:
        s_same_k = _pairsum(nested, 0).sum(axis=0)      # same aleatory draw, m != m'
        s_between = s_all - s_within - s_same_k
        n_between = M * (M - 1) * K * (K - 1)
    else:
        s_between = s_all - s_within
        n_between = M * (M - 1) * K * K

    out = {"flattened": s_all / (n * (n - 1))}
    if M > 1:
        within = s_within / (M * K * (K - 1))
        between = s_between / n_between
        out["pooled"] = within / M + between * (M - 1) / M
        out["fair_inf"] = between
    else:
        out["pooled"] = out["flattened"]
        out["fair_inf"] = np.full_like(np.asarray(out["flattened"]), np.nan)
    return out


def crps_variants(nested: np.ndarray, ref: np.ndarray, shared_bank: bool) -> dict:
    """Three fair-CRPS estimators for one timestep. nested [M,K,nc]; ref [R,nc]."""
    x = nested.reshape(nested.shape[0] * nested.shape[1], nested.shape[-1])
    cross_to_ref = _cross_term(x, ref)
    self_d = self_distance_variants(nested, shared_bank)
    return {k: cross_to_ref - v / 2.0 for k, v in self_d.items()}


# --- validation ------------------------------------------------------------
def _phi(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mean_abs_normal(m: float, s: float) -> float:
    """E|N(m, s^2)| in closed form."""
    import math
    return s * math.sqrt(2.0 / math.pi) * math.exp(-m * m / (2 * s * s)) + m * (2 * _phi(m / s) - 1)


def _selftest() -> None:
    rng = np.random.default_rng(0)

    # (1) Collapse identity: duplicated particles must reproduce the K-member
    #     fair CRPS exactly under both nested targets.
    K, nc = 15, 60
    base = rng.normal(size=(K, nc))
    nested = np.repeat(base[None], 4, axis=0)
    direct = _pairsum(base, 0) / (K * (K - 1))
    sd = self_distance_variants(nested, shared_bank=True)
    assert np.allclose(sd["pooled"], direct, rtol=1e-10, atol=1e-12), "pooled failed collapse identity"
    assert np.allclose(sd["fair_inf"], direct, rtol=1e-10, atol=1e-12), "fair_inf failed collapse identity"
    assert np.all(sd["flattened"] < direct), "flattened should be deflated by duplicate members"
    print(f"[selftest 1] collapse identity OK; flattened self-distance deflated "
          f"{100*float(np.mean(direct - sd['flattened'])/np.mean(direct)):.2f}% "
          f"(-> CRPS inflated)", flush=True)

    # (2) Unbiasedness of `pooled` for the finite-mixture E|X-X'|, against the
    #     analytic value, for BOTH designs.  mu_m distinct, unit-variance comps.
    mus = np.array([-1.0, 0.0, 1.0])
    M = len(mus)
    s2 = np.sqrt(2.0)
    truth = float(np.mean([[_mean_abs_normal(a - b, s2) for a in mus] for b in mus]))
    truth_between = float(np.mean([_mean_abs_normal(mus[i] - mus[j], s2)
                                   for i in range(M) for j in range(M) if i != j]))

    for shared in (True, False):
        reps, Kt = 4000, 20
        est_p, est_f = [], []
        for _ in range(reps):
            if shared:
                z = rng.normal(size=(1, Kt, 1))          # one bank, reused by every particle
                x = mus.reshape(M, 1, 1) + z
            else:
                x = mus.reshape(M, 1, 1) + rng.normal(size=(M, Kt, 1))
            sd = self_distance_variants(x, shared_bank=shared)
            est_p.append(float(sd["pooled"][0]))
            est_f.append(float(sd["fair_inf"][0]))
        for name, est, target in (("pooled", est_p, truth), ("fair_inf", est_f, truth_between)):
            bias = float(np.mean(est)) - target
            se = float(np.std(est, ddof=1) / np.sqrt(reps))
            assert abs(bias) < 4 * se, (
                f"{name} biased under shared_bank={shared}: {bias:+.5f} vs 4*se={4*se:.5f}")
            print(f"[selftest 2] shared_bank={shared!s:<5} {name:<9} "
                  f"target {target:.5f} est {np.mean(est):.5f} bias {bias:+.5f} (4*se {4*se:.5f}) OK",
                  flush=True)

    print("[selftest] all estimator checks passed\n", flush=True)


# --- artifact loading ------------------------------------------------------
def _decode(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])
    return a.astype(str)


def load_event(path: Path, members: int = MEMBERS):
    """Return nested [M,K,T,nc], ref [R,T,nc], and design metadata."""
    with h5py.File(path, "r") as fh:
        pred = np.asarray(fh["pred_members_wd"], dtype=np.float32)[:members]
        ref = np.asarray(fh["ref_members_wd"], dtype=np.float32)
        wettable = np.asarray(fh["wettable_mask"], dtype=bool)
        dry = np.asarray(fh["structural_dry_mask"], dtype=bool)
        keys = set(fh.keys())
        event = str(fh.attrs.get("hydrograph_id", path.stem))

        if "member_epistemic_id" in keys:                 # ALR: explicit nesting
            epi = _decode(fh["member_epistemic_id"])[:members]
            ale = _decode(fh["member_aleatory_id"])[:members]
            shared_bank = True
            design = "alr_crossed_common_random_numbers"
        elif "member_model_id" in keys and len(np.unique(_decode(fh["member_model_id"])[:members])) > 1:
            epi = _decode(fh["member_model_id"])[:members]  # deep ensemble
            ale = None
            shared_bank = False
            design = "deep_ensemble_independent_nested"
        else:                                              # single checkpoint
            epi = np.array(["0"] * len(pred))
            ale = None
            shared_bank = False
            design = "single_checkpoint_iid"

    active = wettable & (~dry)
    if not active.any():
        active = wettable
    pred = pred[:, :, active]
    ref = ref[:, :, active]

    m_vals = list(dict.fromkeys(epi))
    M = len(m_vals)
    K = len(pred) // M
    if M * K != len(pred):
        raise ValueError(f"{path.name}: {len(pred)} members do not tile into M={M}")

    nested = np.empty((M, K) + pred.shape[1:], dtype=np.float64)
    for mi, mv in enumerate(m_vals):
        block = pred[epi == mv]
        if ale is not None:
            k_vals = list(dict.fromkeys(ale))
            order = np.argsort([k_vals.index(v) for v in ale[epi == mv]])
            block = block[order]
        nested[mi] = block
    return nested, ref.astype(np.float64), event, M, K, shared_bank, design


def audit_method(art_dir: Path, label: str, limit: int = 0) -> list:
    files = sorted(art_dir.glob("*.h5"))
    if limit:
        files = files[:limit]
    rows = []
    for i, f in enumerate(files, 1):
        nested, ref, event, M, K, shared, design = load_event(f)
        T = nested.shape[2]

        acc = {k: [] for k in ("flattened", "pooled", "fair_inf")}
        for t in range(T):
            v = crps_variants(nested[:, :, t, :], ref[:, t, :], shared)
            for k in acc:
                acc[k].append(np.mean(v[k]) if np.ndim(v[k]) else v[k])

        flat = nested.reshape(M * K, T, -1)
        sd_pred = flat.std(axis=0, ddof=1)
        sd_ref = ref.std(axis=0, ddof=1)
        part_mean = nested.mean(axis=1)

        # A2/A3: is each epistemic member ITSELF over-dispersed?
        per_member_ratio = [
            float(np.sqrt(np.mean(nested[m].std(axis=0, ddof=1) ** 2))
                  / np.sqrt(np.mean(sd_ref ** 2)))
            for m in range(M)
        ]

        rows.append({
            "event": event, "M": M, "K": K, "design": design,
            "crps": {k: float(np.mean(vs)) for k, vs in acc.items()},
            "rmse_m": float(np.sqrt(np.mean((flat.mean(axis=0) - ref.mean(axis=0)) ** 2))),
            "sigma_pred_m": float(np.sqrt(np.mean(sd_pred ** 2))),
            "sigma_ref_m": float(np.sqrt(np.mean(sd_ref ** 2))),
            "dispersion_ratio": float(np.sqrt(np.mean(sd_pred ** 2)) / np.sqrt(np.mean(sd_ref ** 2))),
            "per_epistemic_member_dispersion_ratio": per_member_ratio,
            "epistemic_spread_m": float(np.sqrt(np.mean(part_mean.var(axis=0, ddof=1)))) if M > 1 else 0.0,
            "aleatory_spread_m": float(np.sqrt(np.mean(nested.var(axis=1, ddof=1).mean(axis=0)))),
        })
        print(f"  [{label} {i}/{len(files)}] {event}: flat {rows[-1]['crps']['flattened']:.6f} "
              f"pooled {rows[-1]['crps']['pooled']:.6f} fair_inf {rows[-1]['crps']['fair_inf']:.6f} "
              f"| disp {rows[-1]['dispersion_ratio']:.3f}", flush=True)
    return rows


def paired_bootstrap(a: list, b: list, key: str, reps: int = BOOTSTRAP_REPS) -> dict:
    """Paired CI on mean(a) - mean(b) over the common events."""
    ia = {r["event"]: r for r in a}
    ib = {r["event"]: r for r in b}
    common = sorted(set(ia) & set(ib))
    da = np.array([ia[e]["crps"][key] for e in common])
    db = np.array([ib[e]["crps"][key] for e in common])
    diff = da - db
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(common), size=(reps, len(common)))
    boot = diff[idx].mean(axis=1)
    return {"n_events": len(common), "mean_diff": float(diff.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "relative_pct": float(100.0 * diff.mean() / db.mean())}


def main() -> int:
    _selftest()
    if sys.argv[1] == "--selftest-only":
        return 0
    out_path = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    roots = {
        "alr_n150": Path("/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/n150_anchor5_no_output_projection_adapter_15ep_20260731_194436_58ae4f7/heldout_eval/heldout_20260802_043554_58ae4f7/forecast_artifacts/heldout"),
        "deep_ensemble_3x20": Path("/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn/eval_outputs/fgn_3x20_m100_all50_raw_20260530_032503/outputs/forecast_artifacts/test_raw_all50"),
        "single_checkpoint": Path("/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_uq_model_comparison/outputs/best_checkpoint_100members_m100_20260626_181452/fgno/forecast_artifacts/test_raw"),
    }

    per_method = {}
    for label, root in roots.items():
        if not root.is_dir():
            print(f"!! missing {label}: {root}", flush=True)
            continue
        print(f"\n=== {label} ({root}) ===", flush=True)
        per_method[label] = audit_method(root, label, limit)

    summary = {}
    for label, rows in per_method.items():
        summary[label] = {
            "n_events": len(rows), "design": rows[0]["design"], "M": rows[0]["M"], "K": rows[0]["K"],
            "crps": {k: float(np.mean([r["crps"][k] for r in rows]))
                     for k in ("flattened", "pooled", "fair_inf")},
            "rmse_m": float(np.mean([r["rmse_m"] for r in rows])),
            "dispersion_ratio": float(np.mean([r["dispersion_ratio"] for r in rows])),
            "per_epistemic_member_dispersion_ratio": float(np.mean(
                [np.mean(r["per_epistemic_member_dispersion_ratio"]) for r in rows])),
            "epistemic_spread_m": float(np.mean([r["epistemic_spread_m"] for r in rows])),
            "aleatory_spread_m": float(np.mean([r["aleatory_spread_m"] for r in rows])),
        }

    contrasts = {}
    if "alr_n150" in per_method and "deep_ensemble_3x20" in per_method:
        for k in ("flattened", "pooled", "fair_inf"):
            contrasts[f"alr_minus_de__{k}"] = paired_bootstrap(
                per_method["alr_n150"], per_method["deep_ensemble_3x20"], k)
    if "alr_n150" in per_method and "single_checkpoint" in per_method:
        for k in ("flattened", "pooled", "fair_inf"):
            contrasts[f"alr_minus_single__{k}"] = paired_bootstrap(
                per_method["alr_n150"], per_method["single_checkpoint"], k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"summary": summary, "contrasts": contrasts, "per_event": per_method}, fh, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps({"summary": summary, "contrasts": contrasts}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
