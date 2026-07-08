"""TDD tests for the NEON Stage-2 config namespace.

The config layer is intentionally torch-free so it can be validated in
isolation. Tests load the module by file path (mirroring the existing
tests/test_neon_stage2.py harness) so they run without importing the full
torch-heavy neuralop package.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_module(name: str, rel_path: str):
    _ensure_pkg("neuralop")
    _ensure_pkg("neuralop.flood")
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon_config = _load_module("neuralop.flood.neon_config", "neuralop/flood/neon_config.py")

NEONStage2Config = neon_config.NEONStage2Config
NEONConfigError = neon_config.NEONConfigError
load_neon_config = neon_config.load_neon_config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_config_matches_plan_defaults():
    cfg = NEONStage2Config()
    assert cfg.enabled is False
    assert cfg.feature_source == "decoder_pre_projection"
    assert cfg.dependency == "za_dependent"
    assert cfg.d_e == 16
    assert cfg.m_train == 4
    assert cfg.k_train == 8
    assert cfg.m_eval == 16
    assert cfg.k_eval == 50
    assert cfg.prior_scale == "auto_0p10_base_rmse"
    assert cfg.alpha is None
    assert cfg.branch_type == "projected"
    assert cfg.train_hidden_channels == 32
    assert cfg.prior_hidden_channels == 5
    assert cfg.branch_layers == 2
    assert cfg.branch_activation == "gelu"
    assert cfg.concat_index is True
    assert cfg.family_batch_size == 1
    assert cfg.effective_batch_size == 8
    assert cfg.shuffle_families is True
    assert cfg.epistemic_resample == "effective_batch"
    assert cfg.latent_bank_count == 4
    assert cfg.reference_member_subsample == 32
    assert cfg.objective == "per_epistemic_fcrps"
    assert cfg.reference_term_for_logging is True
    assert cfg.spatial_weights == "wettable_area"
    assert cfg.lead_time_weights == "uniform"
    assert cfg.bootstrap_enabled is True
    assert cfg.bootstrap_distribution == "tempered_exponential"
    assert cfg.bootstrap_temperature == pytest.approx(0.5)
    assert cfg.bootstrap_normalize == "per_epistemic_batch"
    assert cfg.bootstrap_min_weight == pytest.approx(0.05)
    assert cfg.bootstrap_max_weight == pytest.approx(5.0)
    assert cfg.bootstrap_seed == 0
    assert cfg.cancellation_diagnostics_enabled is True
    assert cfg.cancellation_warn_cosine_below == pytest.approx(-0.90)
    assert cfg.cancellation_warn_cancellation_above == pytest.approx(0.80)
    assert cfg.lambda_rpf == pytest.approx(0.0)
    assert cfg.lambda_smooth == pytest.approx(0.0)
    assert cfg.lambda_time == pytest.approx(0.0)
    assert cfg.lambda_pos == pytest.approx(0.0)
    assert cfg.lambda_mag == pytest.approx(0.0)
    assert cfg.learning_rate == pytest.approx(1.0e-4)
    assert cfg.weight_decay == pytest.approx(1.0e-4)
    assert cfg.n_epochs == 30


# ---------------------------------------------------------------------------
# Loading from the plan's YAML block (capitalized M/K keys)
# ---------------------------------------------------------------------------


def _plan_block():
    return {
        "enabled": True,
        "stage1_checkpoint_dir": "/path/to/pretrained_fgno",
        "stage2_checkpoint_dir": "/path/to/neon_stage2",
        "feature_source": "decoder_pre_projection",
        "dependency": "za_dependent",
        "d_e": 16,
        "M_train": 4,
        "K_train": 8,
        "M_eval": 16,
        "K_eval": 50,
        "prior_scale": "auto_0p10_base_rmse",
        "alpha": None,
        "branch_type": "projected",
        "train_hidden_channels": 32,
        "prior_hidden_channels": 5,
        "branch_layers": 2,
        "branch_activation": "gelu",
        "concat_index": True,
        "family_batch_size": 1,
        "effective_batch_size": 8,
        "shuffle_families": True,
        "epistemic_resample": "effective_batch",
        "latent_bank_count": 4,
        "reference_member_subsample": 32,
        "objective": "per_epistemic_fcrps",
        "reference_term_for_logging": True,
        "spatial_weights": "wettable_area",
        "lead_time_weights": "uniform",
        "bootstrap": {
            "enabled": True,
            "distribution": "tempered_exponential",
            "temperature": 0.5,
            "normalize": "per_epistemic_batch",
            "min_weight": 0.05,
            "max_weight": 5.0,
            "seed": 0,
        },
        "cancellation_diagnostics": {
            "enabled": True,
            "warn_cosine_below": -0.90,
            "warn_cancellation_above": 0.80,
        },
        "lambda_rpf": 0.0,
        "lambda_smooth": 0.0,
        "lambda_time": 0.0,
        "lambda_pos": 0.0,
        "lambda_mag": 0.0,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "n_epochs": 30,
    }


def test_load_maps_capitalized_M_K_keys():
    cfg = load_neon_config(_plan_block())
    assert cfg.m_train == 4
    assert cfg.k_train == 8
    assert cfg.m_eval == 16
    assert cfg.k_eval == 50
    assert cfg.enabled is True
    assert cfg.stage1_checkpoint_dir == "/path/to/pretrained_fgno"


def test_load_accepts_lowercase_m_k_aliases_too():
    block = _plan_block()
    del block["M_train"], block["K_train"], block["M_eval"], block["K_eval"]
    block.update({"m_train": 2, "k_train": 3, "m_eval": 5, "k_eval": 7})
    cfg = load_neon_config(block)
    assert (cfg.m_train, cfg.k_train, cfg.m_eval, cfg.k_eval) == (2, 3, 5, 7)


def test_load_handles_alpha_null():
    block = _plan_block()
    block["alpha"] = None
    cfg = load_neon_config(block)
    assert cfg.alpha is None


def test_load_handles_explicit_alpha_float():
    block = _plan_block()
    block["alpha"] = 0.12
    cfg = load_neon_config(block)
    assert cfg.alpha == pytest.approx(0.12)


def test_load_maps_legacy_hidden_channels_to_train_branch_only():
    block = _plan_block()
    block.pop("train_hidden_channels")
    block["hidden_channels"] = 48
    cfg = load_neon_config(block)
    assert cfg.train_hidden_channels == 48
    assert cfg.prior_hidden_channels == 5


def test_load_parses_nested_bootstrap_and_cancellation_blocks():
    block = _plan_block()
    block["bootstrap"]["temperature"] = 0.25
    block["bootstrap"]["seed"] = 99
    block["cancellation_diagnostics"]["warn_cancellation_above"] = 0.7
    cfg = load_neon_config(block)
    assert cfg.bootstrap_temperature == pytest.approx(0.25)
    assert cfg.bootstrap_seed == 99
    assert cfg.cancellation_warn_cancellation_above == pytest.approx(0.7)


def test_load_parses_minibatch_and_sampling_controls():
    block = _plan_block()
    block.update(
        {
            "family_batch_size": 2,
            "effective_batch_size": 12,
            "shuffle_families": False,
            "epistemic_resample": "epoch",
            "latent_bank_count": 6,
            "reference_member_subsample": 20,
        }
    )
    cfg = load_neon_config(block)
    assert cfg.family_batch_size == 2
    assert cfg.effective_batch_size == 12
    assert cfg.shuffle_families is False
    assert cfg.epistemic_resample == "epoch"
    assert cfg.latent_bank_count == 6
    assert cfg.reference_member_subsample == 20


def test_load_ignores_unknown_keys_but_warns_is_not_required():
    block = _plan_block()
    block["some_future_field"] = 123
    # Unknown keys must not crash the loader (forward-compat).
    cfg = load_neon_config(block)
    assert cfg.enabled is True


def test_load_from_wrapped_neon_namespace():
    cfg = load_neon_config({"neon": _plan_block()})
    assert cfg.enabled is True
    assert cfg.d_e == 16


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_rejects_bad_feature_source():
    with pytest.raises(NEONConfigError, match="feature_source"):
        NEONStage2Config(feature_source="nonsense").validate()


def test_validate_accepts_ablation_feature_sources():
    for src in ("decoder_pre_projection", "latent_grid_fno", "combined_mesh"):
        NEONStage2Config(feature_source=src).validate()


def test_validate_rejects_bad_dependency():
    with pytest.raises(NEONConfigError, match="dependency"):
        NEONStage2Config(dependency="both").validate()


def test_validate_accepts_za_independent_ablation():
    NEONStage2Config(dependency="za_independent").validate()


def test_validate_rejects_bad_objective():
    with pytest.raises(NEONConfigError, match="objective"):
        NEONStage2Config(objective="mystery").validate()


def test_validate_accepts_pooled_negative_control_and_l2_baseline():
    NEONStage2Config(objective="pooled_fcrps").validate()
    NEONStage2Config(objective="l2_mean").validate()


def test_validate_rejects_k_train_below_2():
    # Fair CRPS requires >= 2 aleatory samples.
    with pytest.raises(NEONConfigError, match="k_train"):
        NEONStage2Config(k_train=1).validate()


def test_validate_rejects_k_eval_below_2():
    with pytest.raises(NEONConfigError, match="k_eval"):
        NEONStage2Config(k_eval=1).validate()


def test_validate_rejects_nonpositive_d_e_and_m():
    with pytest.raises(NEONConfigError, match="d_e"):
        NEONStage2Config(d_e=0).validate()
    with pytest.raises(NEONConfigError, match="m_train"):
        NEONStage2Config(m_train=0).validate()
    with pytest.raises(NEONConfigError, match="m_eval"):
        NEONStage2Config(m_eval=0).validate()


def test_validate_rejects_negative_lambdas():
    with pytest.raises(NEONConfigError, match="lambda_rpf"):
        NEONStage2Config(lambda_rpf=-1.0).validate()


def test_validate_rejects_nonpositive_learning_rate_and_epochs():
    with pytest.raises(NEONConfigError, match="learning_rate"):
        NEONStage2Config(learning_rate=0.0).validate()
    with pytest.raises(NEONConfigError, match="n_epochs"):
        NEONStage2Config(n_epochs=0).validate()


# ---------------------------------------------------------------------------
# Prior scale parsing
# ---------------------------------------------------------------------------


def test_prior_scale_fraction_parses_auto_string():
    assert NEONStage2Config(prior_scale="auto_0p10_base_rmse").prior_scale_fraction == pytest.approx(0.10)
    assert NEONStage2Config(prior_scale="auto_0p05_base_rmse").prior_scale_fraction == pytest.approx(0.05)
    assert NEONStage2Config(prior_scale="auto_0p20_base_rmse").prior_scale_fraction == pytest.approx(0.20)


def test_prior_scale_fraction_from_float():
    assert NEONStage2Config(prior_scale=0.15).prior_scale_fraction == pytest.approx(0.15)


def test_prior_scale_fraction_rejects_out_of_range():
    with pytest.raises(NEONConfigError, match="prior_scale"):
        NEONStage2Config(prior_scale="auto_0p50_base_rmse").validate()
    with pytest.raises(NEONConfigError, match="prior_scale"):
        NEONStage2Config(prior_scale=0.5).validate()


def test_uses_auto_prior_scale_flag():
    assert NEONStage2Config(prior_scale="auto_0p10_base_rmse", alpha=None).uses_auto_prior_scale is True
    assert NEONStage2Config(prior_scale=0.1, alpha=None).uses_auto_prior_scale is False
    # An explicit alpha overrides prior-scale auto-calibration.
    assert NEONStage2Config(prior_scale="auto_0p10_base_rmse", alpha=0.1).uses_auto_prior_scale is False


# ---------------------------------------------------------------------------
# Loss-weight adapter
# ---------------------------------------------------------------------------


def test_to_loss_weights_dict_maps_lambdas():
    cfg = NEONStage2Config(
        lambda_rpf=1.0,
        lambda_smooth=2.0,
        lambda_time=3.0,
        lambda_pos=4.0,
        lambda_mag=5.0,
    )
    lw = cfg.to_loss_weights_dict()
    assert lw == {"rpf": 1.0, "smooth": 2.0, "time": 3.0, "pos": 4.0, "mag": 5.0}


# ---------------------------------------------------------------------------
# Shipped default YAML parses and validates
# ---------------------------------------------------------------------------


def test_shipped_default_yaml_parses_and_validates():
    yaml = pytest.importorskip("yaml")
    yaml_path = REPO_ROOT / "config" / "flood" / "coastal" / "neon_stage2.yaml"
    assert yaml_path.exists(), f"shipped NEON config missing: {yaml_path}"
    with yaml_path.open() as handle:
        raw = yaml.safe_load(handle)
    cfg = load_neon_config(raw)  # unwraps the top-level `neon:` block
    assert cfg.enabled is True
    assert cfg.feature_source == "decoder_pre_projection"
    assert cfg.dependency == "za_dependent"
    assert cfg.m_train == 4 and cfg.k_train == 8
    assert cfg.m_eval == 16 and cfg.k_eval == 50
    assert cfg.alpha is None
    assert cfg.uses_auto_prior_scale is True
    assert cfg.prior_scale_fraction == pytest.approx(0.10)
