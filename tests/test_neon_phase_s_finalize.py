from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_phase_s_finalize.py"
    spec = importlib.util.spec_from_file_location("neon_phase_s_finalize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return path


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    head = "abc123"
    checkpoint_sha = "f" * 64
    protocol = _write(
        tmp_path / "protocol" / "PROTOCOL.json",
        {"schema_version": "neon_phase5_protocol_v1"},
    )
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    gate = _write(
        tmp_path / "gate" / "DECISION.json",
        {
            "schema_version": "neon_phase5_gp1_v1",
            "analysis_git_head": head,
            "protocol_sha256": protocol_sha,
        },
    )
    gate_sha = hashlib.sha256(gate.read_bytes()).hexdigest()
    target = _write(
        tmp_path / "PHASE_S_TARGET.json",
        {
            "schema_version": "neon_phase_s_evidence_target_v1",
            "ladder_rung": "B5",
            "stage2_checkpoint_sha256": checkpoint_sha,
            "analysis_git_head": head,
            "protocol_sha256": protocol_sha,
            "governing_gate_sha256": gate_sha,
        },
    )
    scaleout = tmp_path / "scaleout"
    _write(
        scaleout / "contraction_analysis.json",
        {
            "schema_version": "neon_contraction_analysis_v1",
            "ladder_rung": "B5",
            "n_replicates": 5,
            "n_values": [25, 50, 100, 250, 400],
            "gamma_mean": 0.45,
            "gamma_bootstrap_95_ci": [0.30, 0.60],
            "analysis_git_head": head,
            "protocol_sha256": protocol_sha,
            "governing_gate_sha256": gate_sha,
        },
    )
    ood = tmp_path / "ood"
    _write(
        ood / "ranking.json",
        {
            "schema_version": "neon_ood_ranking_v1",
            "n_ood_events": 13,
            "spearman_epistemic_std_rmse": 0.4,
            "top3_error_event_recall": 2 / 3,
            "analysis_git_head": head,
            "protocol_sha256": protocol_sha,
            "stage2_checkpoint_sha256": checkpoint_sha,
        },
    )
    de = tmp_path / "de"
    _write(
        de / "deep_ensemble_comparison.json",
        {
            "schema_version": "neon_deep_ensemble_comparison_v2",
            "j_models": 3,
            "plan": {
                "physical_space": True,
                "common_aleatory_latent_bank": True,
                "stage2_checkpoint_sha256": checkpoint_sha,
                "top_q": 0.1,
            },
            "aggregate": {"spatial_correlation": 0.2},
            "per_family": [
                {
                    "family_id": f"F{index:03d}",
                    "spatial_corr": 0.45,
                    "topq_overlap": 0.35,
                    "neon_epistemic_variance_mean_m2": 1.0e-4,
                    "deep_epistemic_variance_mean_m2": 1.0e-3,
                }
                for index in range(50)
            ],
        },
    )
    _write(
        de / "SUBMITTED.json",
        {
            "schema_version": "neon_phase5_de_submission_v1",
            "git_head": head,
            "protocol_sha256": protocol_sha,
            "stage2_checkpoint_sha256": checkpoint_sha,
        },
    )
    submission = _write(
        tmp_path / "PHASE_S_SUBMITTED.json",
        {
            "schema_version": "neon_phase5_scaleout_submission_v1",
            "git_head": head,
            "protocol_root": str(protocol.parent),
            "protocol_sha256": protocol_sha,
            "governing_gate": str(gate),
            "governing_gate_sha256": gate_sha,
            "ladder_rung": "B5",
            "selected_evidence_target": json.loads(target.read_text()),
            "scaleout_root": str(scaleout),
            "ood_root": str(ood),
            "deep_ensemble_root": str(de),
            "required_terminal_jobs": ["1", "2", "3"],
        },
    )
    return submission, head


def test_phase_s_finalizer_audits_all_required_evidence(tmp_path):
    script = _load_script()
    submission, head = _fixture(tmp_path)

    result = script.finalize_phase_s(submission, expected_head=head)

    assert result["schema_version"] == "neon_phase_s_complete_v1"
    assert result["evidence_complete"] is True
    assert result["contraction"]["gamma_mean"] == pytest.approx(0.45)
    assert result["ood"]["n_events"] == 13
    assert result["deep_ensemble"]["j_models"] == 3
    assert result["amplitude_gate"]["decision"] == "alpha_sweep"
    assert result["mandatory_next"] == "run_predeclared_selected_rung_alpha_sweep"


def test_amplitude_gate_refuses_alpha_sweep_when_map_structure_is_unsupported():
    script = _load_script()
    report = {
        "plan": {"top_q": 0.1},
        "per_family": [
            {
                "spatial_corr": -0.1,
                "topq_overlap": 0.08,
                "neon_epistemic_variance_mean_m2": 1.0e-5,
                "deep_epistemic_variance_mean_m2": 1.0e-3,
            }
            for _ in range(20)
        ],
    }

    result = script.assess_phase_s_amplitude_residual(report, replicates=200)

    assert result["decision"] == "structure_residual"
    assert result["alpha_sweep_eligible"] is False


def test_amplitude_gate_skips_sweep_when_scale_is_equivalent():
    script = _load_script()
    report = {
        "plan": {"top_q": 0.1},
        "per_family": [
            {
                "spatial_corr": 0.4,
                "topq_overlap": 0.3,
                "neon_epistemic_variance_mean_m2": 1.0e-3,
                "deep_epistemic_variance_mean_m2": 1.0e-3,
            }
            for _ in range(20)
        ],
    }

    result = script.assess_phase_s_amplitude_residual(report, replicates=200)

    assert result["decision"] == "no_alpha_sweep"
    assert result["alpha_sweep_eligible"] is False


def test_phase_s_finalizer_rejects_wrong_de_checkpoint(tmp_path):
    script = _load_script()
    submission, head = _fixture(tmp_path)
    payload = json.loads(submission.read_text())
    de_path = Path(payload["deep_ensemble_root"]) / "deep_ensemble_comparison.json"
    de = json.loads(de_path.read_text())
    de["plan"]["stage2_checkpoint_sha256"] = "0" * 64
    _write(de_path, de)

    with pytest.raises(ValueError, match="selected Stage-2 checkpoint"):
        script.finalize_phase_s(submission, expected_head=head)


def test_phase_s_finalizer_requires_all_thirteen_ood_events(tmp_path):
    script = _load_script()
    submission, head = _fixture(tmp_path)
    payload = json.loads(submission.read_text())
    ranking = Path(payload["ood_root"]) / "ranking.json"
    report = json.loads(ranking.read_text())
    report["n_ood_events"] = 12
    _write(ranking, report)

    with pytest.raises(ValueError, match="13 OOD events"):
        script.finalize_phase_s(submission, expected_head=head)


def test_phase_s_finalizer_rejects_cross_checkpoint_ood_evidence(tmp_path):
    script = _load_script()
    submission, head = _fixture(tmp_path)
    payload = json.loads(submission.read_text())
    ranking = Path(payload["ood_root"]) / "ranking.json"
    report = json.loads(ranking.read_text())
    report["stage2_checkpoint_sha256"] = "0" * 64
    _write(ranking, report)

    with pytest.raises(ValueError, match="OOD.*checkpoint"):
        script.finalize_phase_s(submission, expected_head=head)


def test_phase_s_finalizer_rejects_cross_protocol_contraction(tmp_path):
    script = _load_script()
    submission, head = _fixture(tmp_path)
    payload = json.loads(submission.read_text())
    contraction = Path(payload["scaleout_root"]) / "contraction_analysis.json"
    report = json.loads(contraction.read_text())
    report["protocol_sha256"] = "0" * 64
    _write(contraction, report)

    with pytest.raises(ValueError, match="contraction.*protocol"):
        script.finalize_phase_s(submission, expected_head=head)


def test_phase_s_finalizer_rejects_protocol_mutation(tmp_path):
    script = _load_script()
    submission, head = _fixture(tmp_path)
    payload = json.loads(submission.read_text())
    protocol = Path(payload["protocol_root"]) / "PROTOCOL.json"
    _write(protocol, {"schema_version": "mutated"})

    with pytest.raises(ValueError, match="protocol differs"):
        script.finalize_phase_s(submission, expected_head=head)
