from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_scaleout_preflight.py"
    spec = importlib.util.spec_from_file_location("neon_scaleout_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def _write_result(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _write_checksum(path)
    return path


def _write_complete_stage2_result(
    root: Path,
    *,
    rung: str,
    prior_seed: int,
    support_seed: int | None,
    target_std_m: float,
) -> Path:
    for name in ("neon_stage2_best.pt", "history.json", "preflight.json"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name, encoding="utf-8")
    products = {
        "checkpoint": root / "neon_stage2_best.pt",
        "history": root / "history.json",
        "preflight": root / "preflight.json",
    }
    completion = _write_result(
        root / "TRAINING_COMPLETE.json",
        {
            "schema_version": "neon_stage2_training_complete_v1",
            "git_head": "abc123",
            "ladder_rung": rung,
            "n_train": 450,
            "input_paths": {
                key: str(path.resolve()) for key, path in products.items()
            },
            "input_sha256": {
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in products.items()
            },
        },
    )
    return _write_result(
        root / "phase5_eval" / "RESULT.json",
        {
            "schema_version": "neon_phase5_p1a_eval_plan_v1",
            "ladder_rung": rung,
            "prior_seed": prior_seed,
            "dirichlet_particle_seed": support_seed,
            "prior_scale_mode": "de_spread_target",
            "prior_scale_target_std_m": target_std_m,
        },
    )


def test_phase_s_target_rejects_checkpoint_mutated_after_training_completion(tmp_path):
    script = _load_script()
    reports = [
        _write_complete_stage2_result(
            tmp_path / f"b5_seed{seed}",
            rung="B5",
            prior_seed=123,
            support_seed=seed,
            target_std_m=0.046,
        )
        for seed in (11, 12, 13)
    ]
    gate = _write_result(
        tmp_path / "gp1.json",
        {
            "schema_version": "neon_phase5_gp1_v1",
            "gate": "GP1",
            "decision": "contraction_confirmation",
            "verdict_status": "acceptance_replicated",
            "p1a_seed_count": 3,
            "inputs": {"p1a": [str(path) for path in reports]},
        },
    )
    checkpoint = reports[0].parent.parent / "neon_stage2_best.pt"
    checkpoint.write_text("mutated", encoding="utf-8")

    with pytest.raises(ValueError, match="training-completion.*checkpoint SHA-256"):
        script.resolve_phase_s_evidence_target(
            gate,
            dirichlet_particle_seed=11,
        )


def _write_alpha_decision(
    path: Path,
    *,
    reports: list[Path],
    target_std_m: float,
    rung: str = "P1B_B",
    prior_seed: int = 456,
    support_seed: int | None = None,
) -> Path:
    return _write_result(
        path,
        {
            "schema_version": "neon_phase_s_alpha_sweep_v1",
            "decision": "alpha_selected",
            "ladder_rung": rung,
            "prior_seed": prior_seed,
            "dirichlet_particle_seed": support_seed,
            "evaluation_events_used_for_selection": False,
            "selected": {"target_std_m": target_std_m},
            "source_reports": [str(report) for report in reports],
        },
    )


def _write_replicated_gp1(path: Path, *, p1a_results: list[Path] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "neon_phase5_gp1_v1",
                "gate": "GP1",
                "decision": "contraction_confirmation",
                "verdict_status": "acceptance_replicated",
                "p1a_seed_count": 3,
                "seed_results": [
                    {"dirichlet_particle_seed": seed, "decision": "contraction_confirmation"}
                    for seed in (11, 12, 13)
                ],
                "inputs": {
                    "p1a": [] if p1a_results is None else [str(item) for item in p1a_results]
                },
            }
        )
    )
    return path


def test_scaleout_plan_has_five_nested_sizes_for_each_of_five_replicates(tmp_path):
    script = _load_script()
    g1 = _write_replicated_gp1(tmp_path / "gp1.json")
    plan = script.prepare_scaleout_plan(
        g1_report=g1,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        ladder_rung="B5",
    )
    assert len(plan["tasks"]) == 25
    for replicate in range(5):
        rows = [row for row in plan["tasks"] if row["subset_replicate"] == replicate]
        assert [row["n_train"] for row in rows] == [25, 50, 100, 250, 400]
        assert all(row["config"]["bootstrap_distribution"] == "probit_exponential" for row in rows)
    assert all(row["config"]["member_bootstrap_enabled"] is False for row in rows)
    assert plan["dirichlet_particle_seed"] == 11
    assert all(row["config"]["dirichlet_particle_seed"] == 11 for row in plan["tasks"])


def test_scaleout_plan_pins_protocol_and_governing_gate(tmp_path):
    script = _load_script()
    gate = _write_replicated_gp1(tmp_path / "gp1.json")
    gate_sha = hashlib.sha256(gate.read_bytes()).hexdigest()
    protocol_sha = "p" * 64
    plan = script.prepare_scaleout_plan(
        g1_report=gate,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        ladder_rung="B5",
        protocol_sha256=protocol_sha,
        governing_gate_sha256=gate_sha,
    )

    assert plan["protocol_sha256"] == protocol_sha
    assert plan["governing_gate_sha256"] == gate_sha
    task = script.validate_scaleout_task(
        tmp_path / "runs" / "scaleout_plan.json",
        task_id=0,
        expected_head="abc123",
        protocol_sha256=protocol_sha,
        governing_gate_sha256=gate_sha,
    )
    assert task["task_id"] == 0

    with pytest.raises(ValueError, match="protocol"):
        script.validate_scaleout_task(
            tmp_path / "runs" / "scaleout_plan.json",
            task_id=0,
            expected_head="abc123",
            protocol_sha256="q" * 64,
            governing_gate_sha256=gate_sha,
        )


def test_scaleout_plan_rejects_legacy_g1_even_when_it_passed(tmp_path):
    script = _load_script()
    g1 = tmp_path / "g1.json"
    g1.write_text(json.dumps({"schema_version": "neon_g1_gate_v1", "gate_passed": True}))
    with pytest.raises(ValueError, match="replicated Phase-5"):
        script.prepare_scaleout_plan(
            g1_report=g1,
            run_root=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
        )


def test_scaleout_gp1_requires_the_persistent_particle_b5_rung(tmp_path):
    script = _load_script()
    gate = _write_replicated_gp1(tmp_path / "gp1.json")
    with pytest.raises(ValueError, match="B5"):
        script.prepare_scaleout_plan(
            g1_report=gate,
            run_root=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
            ladder_rung="B3",
        )


def test_scaleout_gp1_rejects_unaccepted_dirichlet_support_seed(tmp_path):
    script = _load_script()
    gate = _write_replicated_gp1(tmp_path / "gp1.json")
    with pytest.raises(ValueError, match="accepted Dirichlet support seeds"):
        script.prepare_scaleout_plan(
            g1_report=gate,
            run_root=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
            ladder_rung="B5",
            dirichlet_particle_seed=99,
        )


def test_phase_s_target_resolves_predeclared_gp1_checkpoint(tmp_path):
    script = _load_script()
    reports = []
    for seed in (11, 12, 13):
        stage2 = tmp_path / f"b5_seed{seed}"
        reports.append(
            _write_complete_stage2_result(
                stage2,
                rung="B5",
                prior_seed=20260703,
                support_seed=seed,
                target_std_m=0.046,
            )
        )
    gate = _write_replicated_gp1(tmp_path / "gp1.json", p1a_results=reports)

    target = script.resolve_phase_s_evidence_target(
        gate, dirichlet_particle_seed=12
    )

    assert target["ladder_rung"] == "B5"
    assert target["dirichlet_particle_seed"] == 12
    assert target["prior_seed"] == 20260703
    assert target["stage2_dir"] == str(tmp_path / "b5_seed12")
    assert target["id_metrics"] == str(reports[1])
    assert target["selection_policy"] == "predeclared_seed_not_outcome_ranked"


def test_phase_s_target_production_resolution_requires_matching_provenance(tmp_path):
    script = _load_script()
    protocol = "p" * 64
    reports = []
    for seed in (11, 12, 13):
        stage2 = tmp_path / f"b5_seed{seed}"
        result = _write_complete_stage2_result(
            stage2,
            rung="B5",
            prior_seed=20260703,
            support_seed=seed,
            target_std_m=0.046,
        )
        _write_result(
            result.parent / "PROVENANCE.json",
            {"git_head": "abc123", "protocol_sha256": protocol},
        )
        reports.append(result)
    gate = _write_replicated_gp1(tmp_path / "gp1.json", p1a_results=reports)
    gate_payload = json.loads(gate.read_text())
    gate_payload.update(
        {"analysis_git_head": "abc123", "protocol_sha256": protocol}
    )
    gate.write_text(json.dumps(gate_payload), encoding="utf-8")
    _write_checksum(gate)

    target = script.resolve_phase_s_evidence_target(
        gate,
        dirichlet_particle_seed=12,
        expected_head="abc123",
        protocol_sha256=protocol,
    )
    assert target["analysis_git_head"] == "abc123"
    assert target["protocol_sha256"] == protocol
    assert target["governing_gate_sha256"] == hashlib.sha256(
        gate.read_bytes()
    ).hexdigest()

    provenance = reports[1].parent / "PROVENANCE.json"
    payload = json.loads(provenance.read_text())
    payload["git_head"] = "other"
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    _write_checksum(provenance)
    with pytest.raises(ValueError, match="report Git HEAD"):
        script.resolve_phase_s_evidence_target(
            gate,
            dirichlet_particle_seed=12,
            expected_head="abc123",
            protocol_sha256=protocol,
        )


def test_phase_s_target_resolves_accepted_pilot_prior_seed(tmp_path):
    script = _load_script()
    reports = []
    for seed in (101, 102, 103):
        stage2 = tmp_path / f"p1b_seed{seed}"
        reports.append(
            _write_complete_stage2_result(
                stage2,
                rung="P1B_B",
                prior_seed=seed,
                support_seed=None,
                target_std_m=0.046,
            )
        )
    gate = tmp_path / "pilot.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "neon_phase5_pilot_gate_v1",
                "gate": "PILOT_ACCEPTANCE",
                "decision": "pilot_accepted",
                "verdict_status": "acceptance_replicated",
                "scaleout_eligible": True,
                "ladder_rung": "P1B_B",
                "seed_count": 3,
                "seed_results": [
                    {"prior_seed": seed, "ladder_rung": "P1B_B", "pass": True}
                    for seed in (101, 102, 103)
                ],
                "inputs": {"pilot_results": [str(item) for item in reports]},
            }
        ),
        encoding="utf-8",
    )

    target = script.resolve_phase_s_evidence_target(gate, prior_seed=102)

    assert target["ladder_rung"] == "P1B_B"
    assert target["prior_seed"] == 102
    assert target["stage2_dir"] == str(tmp_path / "p1b_seed102")
    assert target["id_metrics"] == str(reports[1])


def test_phase_s_target_resolves_selected_alpha_run(tmp_path):
    script = _load_script()
    targets = (0.1, 0.2, 0.4)
    reports = [
        _write_complete_stage2_result(
            tmp_path / f"alpha_{target}",
            rung="P1B_B",
            prior_seed=456,
            support_seed=None,
            target_std_m=target,
        )
        for target in targets
    ]
    gate = _write_alpha_decision(
        tmp_path / "ALPHA_SWEEP_DECISION.json",
        reports=reports,
        target_std_m=0.2,
    )

    resolved = script.resolve_phase_s_evidence_target(gate, prior_seed=456)

    assert resolved["ladder_rung"] == "P1B_B"
    assert resolved["prior_seed"] == 456
    assert resolved["stage2_dir"] == str(tmp_path / "alpha_0.2")
    assert resolved["id_metrics"] == str(reports[1])
    assert resolved["prior_scale_target_std_m"] == pytest.approx(0.2)
    assert (
        resolved["selection_policy"]
        == "fixed_id_validation_alpha_selection_no_evaluation_events"
    )


def test_phase_s_alpha_target_rejects_explicit_seed_drift(tmp_path):
    script = _load_script()
    report = _write_complete_stage2_result(
        tmp_path / "alpha",
        rung="B5",
        prior_seed=456,
        support_seed=12,
        target_std_m=0.2,
    )
    gate = _write_alpha_decision(
        tmp_path / "ALPHA_SWEEP_DECISION.json",
        reports=[report],
        target_std_m=0.2,
        rung="B5",
        prior_seed=456,
        support_seed=12,
    )

    with pytest.raises(ValueError, match="prior seed"):
        script.resolve_phase_s_evidence_target(gate, prior_seed=999)
    with pytest.raises(ValueError, match="support seed"):
        script.resolve_phase_s_evidence_target(
            gate, dirichlet_particle_seed=999
        )


def test_scaleout_plan_reproduces_selected_alpha_target(tmp_path):
    script = _load_script()
    report = _write_complete_stage2_result(
        tmp_path / "alpha",
        rung="P1B_B",
        prior_seed=456,
        support_seed=None,
        target_std_m=0.2,
    )
    gate = _write_alpha_decision(
        tmp_path / "ALPHA_SWEEP_DECISION.json",
        reports=[report],
        target_std_m=0.2,
    )

    plan = script.prepare_scaleout_plan(
        g1_report=gate,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        ladder_rung="P1B_B",
        prior_seed=456,
    )

    assert plan["prior_scale_target_std_m"] == pytest.approx(0.2)
    assert all(
        task["config"]["prior_scale"]
        == {"mode": "de_spread_target", "target_std_m": 0.2}
        for task in plan["tasks"]
    )
    task = script.validate_scaleout_task(
        tmp_path / "runs" / "scaleout_plan.json",
        task_id=0,
        expected_head="abc123",
        prior_seed=456,
    )
    assert task["config"]["prior_scale"]["target_std_m"] == pytest.approx(0.2)


def test_scaleout_plan_rejects_selected_alpha_contract_drift(tmp_path):
    script = _load_script()
    report = _write_complete_stage2_result(
        tmp_path / "alpha",
        rung="P1B_B",
        prior_seed=456,
        support_seed=None,
        target_std_m=0.2,
    )
    gate = _write_alpha_decision(
        tmp_path / "ALPHA_SWEEP_DECISION.json",
        reports=[report],
        target_std_m=0.2,
    )

    with pytest.raises(ValueError, match="rung"):
        script.prepare_scaleout_plan(
            g1_report=gate,
            run_root=tmp_path / "rung-drift",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
            ladder_rung="P2",
            prior_seed=456,
        )
    with pytest.raises(ValueError, match="prior seed"):
        script.prepare_scaleout_plan(
            g1_report=gate,
            run_root=tmp_path / "seed-drift",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
            ladder_rung="P1B_B",
            prior_seed=999,
        )


def test_scaleout_plan_accepts_replicated_phase5_pilot_gate(tmp_path):
    script = _load_script()
    gate = tmp_path / "pilot.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "neon_phase5_pilot_gate_v1",
                "gate": "PILOT_ACCEPTANCE",
                "decision": "pilot_accepted",
                "verdict_status": "acceptance_replicated",
                "scaleout_eligible": True,
                "ladder_rung": "P1B_B",
                "seed_count": 3,
                "seed_results": [
                    {"prior_seed": 123},
                    {"prior_seed": 456},
                    {"prior_seed": 789},
                ],
            }
        )
    )
    plan = script.prepare_scaleout_plan(
        g1_report=gate,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        ladder_rung="P1B_B",
        prior_seed=456,
    )
    assert plan["ladder_rung"] == "P1B_B"
    assert plan["prior_seed"] == 456
    assert all(
        json.loads(Path(row["preflight_path"]).read_text())["prior_seed"] == 456
        for row in plan["tasks"]
    )
    assert all(row["config"]["bootstrap_index_dim"] == 128 for row in plan["tasks"])


def test_scaleout_rejects_unaccepted_pilot_prior_seed(tmp_path):
    script = _load_script()
    gate = tmp_path / "pilot.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "neon_phase5_pilot_gate_v1",
                "gate": "PILOT_ACCEPTANCE",
                "decision": "pilot_accepted",
                "verdict_status": "acceptance_replicated",
                "scaleout_eligible": True,
                "ladder_rung": "P1B_B",
                "seed_count": 3,
                "seed_results": [
                    {"prior_seed": 1},
                    {"prior_seed": 2},
                    {"prior_seed": 3},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="accepted pilot prior seeds"):
        script.prepare_scaleout_plan(
            g1_report=gate,
            run_root=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
            ladder_rung="P1B_B",
            prior_seed=99,
        )


def test_scaleout_task_rejects_changed_prior_seed(tmp_path):
    script = _load_script()
    g1 = _write_replicated_gp1(tmp_path / "gp1.json")
    plan = script.prepare_scaleout_plan(
        g1_report=g1,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        prior_seed=77,
        ladder_rung="B5",
    )
    task = plan["tasks"][0]
    preflight = Path(task["preflight_path"])
    payload = json.loads(preflight.read_text())
    payload["prior_seed"] = 78
    preflight.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prior seed"):
        script.validate_scaleout_task(
            tmp_path / "runs" / "scaleout_plan.json",
            task_id=0,
            expected_head="abc123",
        )


def test_scaleout_task_rejects_plan_mutation_after_preflight(tmp_path):
    script = _load_script()
    gate = _write_replicated_gp1(tmp_path / "gp1.json")
    script.prepare_scaleout_plan(
        g1_report=gate,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        ladder_rung="B5",
    )
    plan_path = tmp_path / "runs" / "scaleout_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["tasks"][0]["output_dir"] = str(tmp_path / "redirected")
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match="checksum"):
        script.validate_scaleout_task(
            plan_path,
            task_id=0,
            expected_head="abc123",
        )


def test_scaleout_task_rejects_runtime_prior_seed_mismatch(tmp_path):
    script = _load_script()
    g1 = _write_replicated_gp1(tmp_path / "gp1.json")
    script.prepare_scaleout_plan(
        g1_report=g1,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
        prior_seed=77,
        ladder_rung="B5",
    )
    with pytest.raises(ValueError, match="runtime prior seed"):
        script.validate_scaleout_task(
            tmp_path / "runs" / "scaleout_plan.json",
            task_id=0,
            expected_head="abc123",
            prior_seed=78,
        )


def test_scaleout_plan_preserves_container_visible_symlink_paths(tmp_path):
    script = _load_script()
    actual = tmp_path / "actual"
    actual.mkdir()
    logical = tmp_path / "logical"
    logical.symlink_to(actual, target_is_directory=True)
    g1 = _write_replicated_gp1(logical / "gp1.json")
    plan = script.prepare_scaleout_plan(
        g1_report=g1,
        run_root=logical / "runs",
        cache_dir=logical / "cache",
        expected_head="abc123",
        ladder_rung="B5",
    )
    assert "/logical/" in plan["g1_report"]
    assert "/logical/" in plan["cache_dir"]
    assert all("/logical/" in row["output_dir"] for row in plan["tasks"])
