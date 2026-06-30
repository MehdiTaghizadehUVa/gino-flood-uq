from pathlib import Path


def test_operator_eval_wrapper_uses_maintained_app():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "scripts" / "flood_wv_eval_operator.py").read_text()
    assert "neuralop.flood.eval.common" in text
    assert "operator_legacy" not in text


def test_eval_slurm_uses_runtime_checkpoint_default():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "scripts" / "slurm" / "eval" / "flood_wv_eval_operator.sh").read_text()
    assert "--checkpoint.eval_name" not in text

def test_masked_eval_slurm_uses_canonical_training_package():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "scripts" / "slurm" / "eval" / "flood_wv_eval_operator_masked_primary.sh").read_text()
    assert '--structural_dry.canonical_data_root "${TRAIN_ROOT}"' in text
    assert '--structural_dry.canonical_train_txt "${TRAIN_TXT_NAME}"' in text


def test_eval_slurm_uses_test_clean_boundary_file():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / 'scripts' / 'slurm' / 'eval' / 'flood_wv_eval_operator.sh').read_text()
    assert '--data.clean_boundary_file "${TEST_CLEAN_BOUNDARY_FILE}"' in text
    assert '--rollout_data.clean_boundary_file "${TEST_CLEAN_BOUNDARY_FILE}"' in text


def test_masked_eval_slurm_uses_test_clean_boundary_file():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / 'scripts' / 'slurm' / 'eval' / 'flood_wv_eval_operator_masked_primary.sh').read_text()
    assert '--data.clean_boundary_file "${TEST_CLEAN_BOUNDARY_FILE}"' in text
    assert '--rollout_data.clean_boundary_file "${TEST_CLEAN_BOUNDARY_FILE}"' in text


def test_diffusion_eval_wrapper_uses_maintained_app():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "neuralop" / "flood" / "cli" / "eval_diffusion.py").read_text()
    assert "neuralop.flood.eval.diffusion import main" in text
    assert "diffusion_legacy" not in text
