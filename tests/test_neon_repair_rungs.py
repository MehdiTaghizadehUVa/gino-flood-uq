from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_stage2_tr_train.py"
    spec = importlib.util.spec_from_file_location("neon_stage2_tr_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase5_p1a_evaluator_imports_through_its_executable_seam(monkeypatch):
    """Submission preflight must be able to import every P1a dependency."""

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "neon_phase5_p1a_eval.py"
    spec = importlib.util.spec_from_file_location("neon_phase5_p1a_eval_import_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)


def test_resolved_ladder_config_separates_b1_mechanisms():
    script = _load_script()
    b1a = script._resolved_ladder_config(
        "B1A", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    b1b = script._resolved_ladder_config(
        "B1B", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    assert b1a.member_bootstrap_enabled is False
    assert b1a.bootstrap_distribution == "tempered_exponential"
    assert b1b.member_bootstrap_enabled is False
    assert b1b.bootstrap_distribution == "probit_exponential"


def test_preflight_manifest_records_fully_validated_config(tmp_path):
    script = _load_script()
    config = script._resolved_ladder_config(
        "B4", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30,
        de_spread_multiplier=0.5,
    )
    output = tmp_path / "preflight.json"
    script._write_preflight_manifest(
        output,
        config=config,
        rung="B4",
        n_train=450,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        subset_replicate=0,
    )
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "neon_repair_preflight_v1"
    assert payload["ladder_rung"] == "B4"
    assert payload["config"]["prior_scale"]["mode"] == "de_spread_target"
    assert payload["config"]["prior_scale"]["target_std_m"] == 0.023


def test_phase_s_alpha_target_can_be_applied_to_the_selected_rung():
    script = _load_script()

    config = script._resolved_ladder_config(
        "B5",
        prior_scale="auto_0p10_base_rmse",
        d_e=16,
        n_epochs=30,
        prior_target_std_m=0.031,
        dirichlet_particle_seed=771,
    )

    assert config.uses_de_spread_prior_scale is True
    assert config.de_spread_target_std_m == pytest.approx(0.031)
    assert config.dirichlet_particle_seed == 771


def test_training_state_paths_resume_only_from_an_existing_epoch_state(tmp_path):
    script = _load_script()

    latest, resume = script._training_state_paths(tmp_path)
    assert latest == tmp_path / "neon_stage2_latest_state.pt"
    assert resume is None

    latest.write_bytes(b"completed epoch state")
    latest_again, resume = script._training_state_paths(tmp_path)
    assert latest_again == latest
    assert resume == latest


def test_training_completion_manifest_freezes_checkpoint_history_and_preflight(tmp_path):
    script = _load_script()
    checkpoint = tmp_path / "neon_stage2_best.pt"
    history = tmp_path / "history.json"
    preflight = tmp_path / "preflight.json"
    checkpoint.write_bytes(b"checkpoint")
    history.write_text("{}\n", encoding="utf-8")
    preflight.write_text("{}\n", encoding="utf-8")

    manifest = script._write_training_completion_manifest(
        output_dir=tmp_path,
        checkpoint_path=checkpoint,
        history_path=history,
        preflight_path=preflight,
        git_head="abc123",
        rung="P1B_A",
        n_train=450,
    )

    verify_checksummed_artifact(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "neon_stage2_training_complete_v1"
    assert payload["git_head"] == "abc123"
    assert payload["ladder_rung"] == "P1B_A"
    assert payload["input_sha256"]["checkpoint"] == script._sha256(str(checkpoint))


def test_phase5_factorial_rungs_isolate_index_capacity_and_amortization():
    script = _load_script()
    arm_a = script._resolved_ladder_config(
        "P1B_A", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    arm_b = script._resolved_ladder_config(
        "P1B_B", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    arm_c = script._resolved_ladder_config(
        "P1B_C", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )

    assert arm_a.d_e == 128
    assert arm_a.bootstrap_index_dim is None
    assert arm_a.epistemic_quadratic_terms == 0
    assert arm_b.d_e == 16
    assert arm_b.bootstrap_index_dim == 128
    assert arm_b.train_parameter_match_basis_dim is None
    assert arm_c.d_e == 16
    assert arm_c.bootstrap_index_dim is None
    assert arm_c.train_parameter_match_basis_dim == 128


def test_phase5_nonrepresentable_prior_arm_preserves_b3_statistical_design():
    script = _load_script()
    p2 = script._resolved_ladder_config(
        "P2", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    assert p2.bootstrap_distribution == "probit_exponential"
    assert p2.member_bootstrap_enabled is False
    assert p2.prior_rff_dim == 32
    assert p2.epistemic_basis == "identity"
    assert p2.concat_index is False


def test_contraction_confirmation_routes_through_complete_phase_s_adapter():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_neon_phase5_conditional_pilot.sh"
    ).read_text()
    branch = script.split("contraction_confirmation)", 1)[1].split(";;", 1)[0]

    assert "submit_neon_phase5_scaleout.sh" in branch
    assert '"final_audit_job"' in branch
    assert "submit_neon_ablation_grid.sh" not in branch


def test_ood_array_pins_checkpoint_to_signed_id_evidence():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sbatch_neon_phase5_ood_array.sh"
    ).read_text()

    assert '"${NEON_ID_METRICS}"' in script
    assert "verify_phase5_evidence_artifact" in script
    assert "ID evidence and OOD checkpoint SHA-256 values differ" in script


def test_phase5_top_level_dry_run_stops_before_first_sbatch():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "submit_neon_phase5.sh"
    ).read_text()

    dry_run = script.index('"${NEON_SUBMIT_DRY_RUN:-0}" == 1')
    first_submission = script.index("D2=$(submit_task")
    assert dry_run < first_submission


def test_conditional_pilot_manifest_writer_imports_checksummed_writer():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_neon_phase5_conditional_pilot.sh"
    ).read_text()
    manifest_block = script.split(
        '"${TRAIN_JOBS[*]}" "${EVAL_JOBS[*]}" "${PILOT_DIRS[*]}"', 1
    )[1]

    assert (
        "from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact"
        in manifest_block
    )


def test_all_phase5_submission_manifests_are_checksummed_and_verified():
    root = Path(__file__).resolve().parents[1]
    initial = (root / "scripts" / "submit_neon_phase5.sh").read_text()
    p1a = (root / "scripts" / "submit_neon_phase5_p1a_replication.sh").read_text()
    pilot = (root / "scripts" / "submit_neon_phase5_pilot_replication.sh").read_text()

    assert "write_checksummed_artifact(path, payload)" in initial
    assert 'test -s "${ROOT}/SUBMITTED.json.sha256"' in p1a
    assert "verify_checksummed_artifact(submission)" in p1a
    assert "write_checksummed_artifact(" in p1a
    assert "write_checksummed_artifact(" in pilot


def test_phase_s_adapter_submits_all_predeclared_evidence_paths():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_neon_phase5_scaleout.sh"
    ).read_text()

    assert "resolve-evidence" in script
    assert "submit_neon_phase5_ood_evidence.sh" in script
    assert "submit_neon_phase5_de_evidence.sh" in script
    assert "sbatch_neon_phase_s_finalize.sh" in script
    assert '"final_audit_job"' in script
    assert '"required_terminal_jobs"' in script
    assert "NEON_SUBMIT_DRY_RUN=1" in script
    assert '"selected_evidence_target"' in script


def test_scaleout_submission_uses_array_safe_optional_arguments_and_protocol_root():
    repo = Path(__file__).resolve().parents[1]
    ablation = (repo / "scripts" / "submit_neon_ablation_grid.sh").read_text()
    scaleout = (repo / "scripts" / "submit_neon_phase5_scaleout.sh").read_text()
    selected = (
        repo / "scripts" / "submit_neon_phase5_selected_alpha_scaleout.sh"
    ).read_text()

    assert "PREPARE_SEED_ARGS=()" in ablation
    assert '"${PREPARE_SEED_ARGS[@]}"' in ablation
    assert "${NEON_DIRICHLET_PARTICLE_SEED:+" not in ablation
    assert 'PROTOCOL_ROOT=${NEON_PHASE5_PROTOCOL_ROOT:-${ROOT}}' in scaleout
    assert '"${PROTOCOL_ROOT}" "${STAGE2_DIR}"' in scaleout
    assert 'NEON_PHASE5_PROTOCOL_ROOT="${SOURCE_ROOT}"' in selected


def test_phase_s_alpha_sweep_is_fail_closed_and_reuses_selected_rung():
    finalize = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "neon_phase_s_finalize.py"
    ).read_text()
    submit = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_neon_phase5_alpha_sweep.sh"
    ).read_text()

    assert "structure_supported and amplitude_mismatch" in finalize
    assert 'gate["decision"] == "alpha_sweep"' in submit
    assert 'submit_neon_repair_rung.sh "${RUNG}"' in submit
    assert "NEON_PRIOR_TARGET_STD_M" in submit
    assert "MULTIPLIERS=(0.5 1.0 2.0)" in submit
    assert "neon_phase5_alpha_sweep_finalize.py" in (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sbatch_neon_phase5_alpha_finalize.sh"
    ).read_text()


def test_conditional_phase_s_does_not_force_an_unaccepted_dirichlet_seed():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_neon_phase5_conditional_pilot.sh"
    ).read_text()
    branch = script.split("contraction_confirmation)", 1)[1].split(";;", 1)[0]

    assert "NEON_DIRICHLET_PARTICLE_SEED=123" not in branch


def test_phase5_p1a_waits_for_a_successful_gd0_decision():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_neon_phase5.sh"
    ).read_text()

    assert 'NEON_SBATCH_DEPENDENCY="afterok:${GD0}"' in script
    assert 'NEON_SBATCH_DEPENDENCY="afterany:${GD0}"' not in script


def test_phase5_plan_only_does_not_request_a_gpu_inside_apptainer():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sbatch_neon_phase5_task.sh"
    ).read_text()

    assert 'APPTAINER_GPU_ARGS=()' in script
    assert 'APPTAINER_GPU_ARGS=(--nv)' in script
    assert 'apptainer exec "${APPTAINER_GPU_ARGS[@]}"' in script
    assert "apptainer exec --nv ${APPTAINER_BIND_ARGS}" not in script


def test_gd0_uses_direct_data_reoptimization_not_the_local_curvature_proxy():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "neon_phase5_decision.py"
    ).read_text()

    assert 'direct_scale_m=direct_data["direct_scale_m"]' in script
    assert 'direct_data["weight_induced_displacement_rms_m"]["estimate"]' in script
    assert 'result["functional_displacement_source"]' in script
    assert '"direct_data_weighted_reoptimization"' in script


def test_gp1_uses_support_matched_dirichlet_reoptimization_displacement():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "neon_phase5_decision.py"
    ).read_text()

    assert 'direct["weight_induced_displacement_rms_m"]["estimate"]' in script
    assert '"support_matched_dirichlet_direct_reoptimization"' in script


def test_neon_gap_runner_defaults_to_the_complete_neon_suite_only():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_neon_gap_tests.sh"
    ).read_text()

    assert 'if [[ "$#" -eq 0 ]]; then' in script
    assert "set -- tests/test_neon_*.py" in script
