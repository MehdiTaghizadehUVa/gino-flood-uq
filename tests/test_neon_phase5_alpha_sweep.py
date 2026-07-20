from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "neon_phase5_alpha_sweep_finalize.py"
    )
    spec = importlib.util.spec_from_file_location("neon_phase5_alpha_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase_s() -> dict:
    return {
        "amplitude_gate": {"decision": "alpha_sweep", "alpha_sweep_eligible": True},
        "selected_evidence_target": {
            "ladder_rung": "B5",
            "prior_seed": 101,
            "dirichlet_particle_seed": 202,
        },
        "deep_ensemble": {
            "aggregate": {"deep_epistemic_variance_mean_m2": 0.04}
        },
    }


def _report(target: float, crps: float, *, passing: bool = True) -> dict:
    return {
        "ladder_rung": "B5",
        "prior_seed": 101,
        "dirichlet_particle_seed": 202,
        "amplitude_tuned_on_evaluation_events": False,
        "prior_scale_mode": "de_spread_target",
        "prior_scale_target_std_m": target,
        "noninferiority": {"pass": passing},
        "skill": {"both": {"fair_crps_m": crps, "rmse_m": 0.05}},
        "impact": {"both": {"area_crps_km2": 0.3}},
        "posterior_scale_m": {"estimate": 0.01},
    }


def test_alpha_sweep_requires_exact_targets_and_selects_only_noninferior_run():
    script = _load_script()
    reports = [
        _report(0.1, 0.020),
        _report(0.2, 0.018, passing=False),
        _report(0.4, 0.019),
    ]

    result = script.finalize_alpha_sweep(_phase_s(), reports)

    assert result["decision"] == "alpha_selected"
    assert result["selected"]["target_std_m"] == pytest.approx(0.4)
    assert result["evaluation_events_used_for_selection"] is False


def test_alpha_sweep_rejects_unapproved_or_posthoc_target():
    script = _load_script()
    phase_s = _phase_s()
    phase_s["amplitude_gate"]["alpha_sweep_eligible"] = False
    with pytest.raises(ValueError, match="does not authorize"):
        script.finalize_alpha_sweep(
            phase_s, [_report(0.1, 0.02), _report(0.2, 0.02), _report(0.4, 0.02)]
        )

    with pytest.raises(ValueError, match="0.5/1/2x"):
        script.finalize_alpha_sweep(
            _phase_s(),
            [_report(0.1, 0.02), _report(0.2, 0.02), _report(0.3, 0.02)],
        )
