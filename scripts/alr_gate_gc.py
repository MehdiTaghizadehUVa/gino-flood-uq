"""Gate GC: does dispersion pinning fix the attribution without costing skill?

The criteria were pre-registered in the Phase-6 plan before any pinned run
existed, and are evaluated here mechanically so the verdict cannot drift to fit
whatever the runs produced.

  1. aleatory channel pinned      sigma_ale / sigma_ref <= 1.10
  2. epistemic channel populated  sigma_epi >= 0.030 m
  3. epistemic channel calibrated epistemic interval covers the reference mean
                                  >= 0.50 at nominal 95%
  4. skill preserved              crossed fair CRPS within +3% of baseline
  5. mean preserved               RMSE within 0.001 m of baseline
  6. THE REAL TEST                contraction negative -- the epistemic spread
                                  must SHRINK from n50 to n150 training families

Criterion 6 is the one that distinguishes a genuinely epistemic channel from a
correctly-sized but meaningless one.  The unpinned pilot showed significant
ANTI-contraction (CI [+5.86e-05, +7.59e-05]): with no residual posterior to
express, the adapters fit bootstrap noise, which does not shrink with more data.
Pinning is supposed to hand them a residual.  If contraction still fails, the
bootstrap mechanism itself is dead in this setting and Phase E (out-of-bag
epistemic CRPS) becomes mandatory rather than optional.

Verdict is three-way.  "indeterminate" is a real outcome: criteria 1-5 passing
while contraction is merely unavailable is NOT a pass, and must not be reported
as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_ALEATORY_OVER_REFERENCE = 1.10
MIN_EPISTEMIC_SPREAD_M = 0.030
MIN_EPISTEMIC_MEAN_COVERAGE_95 = 0.50
MAX_CRPS_REGRESSION_FRACTION = 0.03
MAX_RMSE_REGRESSION_M = 0.001


def _load(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    return payload.get("summary", payload)


def contraction_ci_upper(contraction: dict) -> float:
    """Upper bound of the paired n150-minus-n50 epistemic-spread CI.

    Two schemas exist on disk for the same quantity -- difference_ci95 in the
    earlier analysis and paired_difference_ci95 in the later one.  Accept
    both and raise on anything else: silently returning "unmeasured" for a
    contraction file that WAS supplied would turn a real fail into an
    indeterminate, which is the one direction this gate must never round.
    """
    epi = contraction.get("summary", {}).get("epistemic_spread_m")
    if not isinstance(epi, dict):
        raise KeyError(
            "contraction artifact has no summary.epistemic_spread_m block; "
            f"top-level keys were {sorted(contraction)}"
        )
    for key in ("paired_difference_ci95", "difference_ci95"):
        if key in epi:
            return float(epi[key][1])
    raise KeyError(
        "contraction artifact has no recognised epistemic difference CI "
        f"(looked for paired_difference_ci95/difference_ci95, found {sorted(epi)})"
    )


def evaluate(baseline: dict, candidate: dict, contraction: dict | None) -> dict:
    b_dec, c_dec = baseline["decomposition"], candidate["decomposition"]
    b_crps = baseline["crps"]["crossed_fair_crps_m"]
    c_crps = candidate["crps"]["crossed_fair_crps_m"]
    epi_cov = candidate["epistemic_interval_covers_reference_mean"]["95"]

    crps_delta = (c_crps - b_crps) / b_crps if b_crps else float("nan")
    rmse_delta = c_dec["rmse_m"] - b_dec["rmse_m"]

    checks = [
        {
            "name": "aleatory_pinned",
            "detail": "sigma_ale / sigma_ref",
            "value": c_dec["aleatory_over_reference"],
            "threshold": f"<= {MAX_ALEATORY_OVER_REFERENCE}",
            "baseline": b_dec["aleatory_over_reference"],
            "passed": c_dec["aleatory_over_reference"] <= MAX_ALEATORY_OVER_REFERENCE,
        },
        {
            "name": "epistemic_populated",
            "detail": "sigma_epi (m)",
            "value": c_dec["epistemic_spread_m"],
            "threshold": f">= {MIN_EPISTEMIC_SPREAD_M}",
            "baseline": b_dec["epistemic_spread_m"],
            "passed": c_dec["epistemic_spread_m"] >= MIN_EPISTEMIC_SPREAD_M,
        },
        {
            "name": "epistemic_calibrated",
            "detail": "covers reference mean @95",
            "value": epi_cov,
            "threshold": f">= {MIN_EPISTEMIC_MEAN_COVERAGE_95}",
            "baseline": baseline["epistemic_interval_covers_reference_mean"]["95"],
            "passed": epi_cov >= MIN_EPISTEMIC_MEAN_COVERAGE_95,
        },
        {
            "name": "skill_preserved",
            "detail": "crossed CRPS vs baseline",
            "value": crps_delta,
            "threshold": f"<= +{MAX_CRPS_REGRESSION_FRACTION:.0%}",
            "baseline": b_crps,
            "passed": crps_delta <= MAX_CRPS_REGRESSION_FRACTION,
        },
        {
            "name": "mean_preserved",
            "detail": "RMSE delta (m)",
            "value": rmse_delta,
            "threshold": f"<= {MAX_RMSE_REGRESSION_M}",
            "baseline": b_dec["rmse_m"],
            "passed": rmse_delta <= MAX_RMSE_REGRESSION_M,
        },
    ]

    if contraction is None:
        contraction_check = {
            "name": "contraction_negative", "detail": "n50 -> n150 epistemic spread",
            "value": None, "threshold": "< 0 (CI upper bound)", "baseline": None,
            "passed": None,
        }
    else:
        contraction_check = {
            "name": "contraction_negative", "detail": "n50 -> n150 epistemic spread",
            "value": contraction_ci_upper(contraction), "threshold": "< 0 (CI upper bound)",
            "baseline": None, "passed": None,
        }
        contraction_check["passed"] = contraction_check["value"] < 0.0
    checks.append(contraction_check)

    core = [c for c in checks if c["name"] != "contraction_negative"]
    if not all(c["passed"] for c in core):
        verdict = "fail"
        reason = "one or more pre-registered criteria were not met"
    elif contraction_check["passed"] is None:
        verdict = "indeterminate"
        reason = ("criteria 1-5 pass but contraction was not measured; a pinned run "
                  "at BOTH n50 and n150 is required before this can be called a pass")
    elif not contraction_check["passed"]:
        verdict = "fail_contraction"
        reason = ("attribution is fixed but the epistemic channel still does not "
                  "contract with more data: the bootstrap mechanism is not supplying "
                  "a posterior, so Phase E (out-of-bag epistemic CRPS) is mandatory")
    else:
        verdict = "pass"
        reason = "attribution fixed, skill preserved, and the channel contracts with data"

    return {"verdict": verdict, "reason": reason, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="C-0 dispersion audit json")
    ap.add_argument("--candidate", type=Path, required=True, help="C-1 dispersion audit json")
    ap.add_argument("--contraction", type=Path, default=None, help="contraction json (optional)")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()

    result = evaluate(
        _load(a.baseline),
        _load(a.candidate),
        json.loads(a.contraction.read_text()) if a.contraction else None,
    )

    print(f"{'criterion':<24} {'detail':<30} {'value':>12} {'threshold':>14}  {'baseline':>12}  result")
    print("-" * 104)
    for c in result["checks"]:
        v = "n/a" if c["value"] is None else f"{c['value']:.5f}"
        b = "" if c["baseline"] is None else f"{c['baseline']:.5f}"
        mark = {True: "PASS", False: "FAIL", None: "UNMEASURED"}[c["passed"]]
        print(f"{c['name']:<24} {c['detail']:<30} {v:>12} {c['threshold']:>14}  {b:>12}  {mark}")
    print("-" * 104)
    print(f"VERDICT: {result['verdict'].upper()}\n  {result['reason']}")

    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
