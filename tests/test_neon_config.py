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
    assert cfg.train_hidden_channels == 16
    assert cfg.prior_hidden_channels == 16
    assert cfg.branch_layers == 2
    assert cfg.branch_activation == "gelu"
    assert cfg.concat_index is False
    assert cfg.prior_rff_dim == 0
    assert cfg.prior_rff_lengthscale == pytest.approx(0.25)
    assert cfg.prior_rff_include_lead is True
    assert cfg.epistemic_basis == "hermite_random_projection"
    assert cfg.epistemic_quadratic_terms == 16
    assert cfg.deterministic_head is True
    assert cfg.deterministic_head_feature == "canonical_aleatory_mean"
    assert cfg.deterministic_head_canonical_k == 32
    assert cfg.family_batch_size == 1
    assert cfg.effective_batch_size == 8
    assert cfg.shuffle_families is True
    assert cfg.epistemic_resample == "effective_batch"
    assert cfg.latent_bank_count == 4
    assert cfg.feature_prefetch_workers == 2
    assert cfg.feature_prefetch_depth == 2
    assert cfg.reference_member_subsample == 32
    assert cfg.objective == "per_epistemic_fcrps"
    assert cfg.reference_term_for_logging is True
    assert cfg.spatial_weights == "wettable_area"
    assert cfg.lead_time_weights == "uniform"
    assert cfg.bootstrap_enabled is True
    assert cfg.bootstrap_distribution == "probit_exponential"
    assert cfg.bootstrap_temperature == pytest.approx(0.5)
    assert cfg.bootstrap_normalize == "per_epistemic_batch"
    assert cfg.bootstrap_min_weight == pytest.approx(0.05)
    assert cfg.bootstrap_max_weight == pytest.approx(5.0)
    assert cfg.bootstrap_seed == 0
    assert cfg.member_bootstrap_enabled is False
    assert cfg.member_bootstrap_temperature == pytest.approx(1.0)
    assert cfg.member_bootstrap_seed == 1
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
    assert cfg.validation_interval == 1
    assert cfg.selection_min_retention == pytest.approx(0.0)
    assert cfg.selection_rmse_margin_m == pytest.approx(0.001)
    assert cfg.selection_metric == "mixture_crps"
    assert cfg.selection_enforce_rmse is True
    assert cfg.calibration_families == 4
    assert cfg.calibration_m == 64


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
        "prior_rff_dim": 32,
        "prior_rff_lengthscale": 0.25,
        "prior_rff_include_lead": True,
        "epistemic_basis": "identity",
        "deterministic_head": False,
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
        "member_bootstrap": {
            "enabled": True,
            "temperature": 1.0,
            "seed": 1,
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
        "selection_min_retention": 0.3,
        "calibration_families": 4,
        "calibration_m": 64,
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


def test_load_parses_nested_member_bootstrap_block():
    block = _plan_block()
    block["member_bootstrap"]["temperature"] = 0.25
    block["member_bootstrap"]["seed"] = 17
    cfg = load_neon_config(block)
    assert cfg.member_bootstrap_enabled is True
    assert cfg.member_bootstrap_temperature == pytest.approx(0.25)
    assert cfg.member_bootstrap_seed == 17


def test_load_film_branch_defaults_rff_off_for_backward_compatibility():
    block = _plan_block()
    block["branch_type"] = "film"
    block.pop("prior_rff_dim")
    cfg = load_neon_config(block)
    assert cfg.branch_type == "film"
    assert cfg.prior_rff_dim == 0


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


def test_validate_rejects_invalid_prior_rff_and_selection_settings():
    with pytest.raises(NEONConfigError, match="prior_rff_dim"):
        NEONStage2Config(prior_rff_dim=31).validate()
    with pytest.raises(NEONConfigError, match="prior_rff_lengthscale"):
        NEONStage2Config(prior_rff_lengthscale=0.0).validate()
    with pytest.raises(NEONConfigError, match="selection_min_retention"):
        NEONStage2Config(selection_min_retention=1.5).validate()
    with pytest.raises(NEONConfigError, match="calibration_m"):
        NEONStage2Config(calibration_m=1).validate()


def test_validate_rejects_nonpositive_learning_rate_and_epochs():
    with pytest.raises(NEONConfigError, match="learning_rate"):
        NEONStage2Config(learning_rate=0.0).validate()
    with pytest.raises(NEONConfigError, match="n_epochs"):
        NEONStage2Config(n_epochs=0).validate()
    with pytest.raises(NEONConfigError, match="validation_interval"):
        NEONStage2Config(validation_interval=0).validate()


@pytest.mark.parametrize("field", ["feature_prefetch_workers", "feature_prefetch_depth"])
def test_validate_rejects_negative_feature_prefetch_settings(field):
    with pytest.raises(NEONConfigError, match=field):
        NEONStage2Config(**{field: -1}).validate()


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


def test_de_spread_target_prior_scale_is_unit_aware_and_not_fraction_capped():
    cfg = load_neon_config(
        {"neon": {"prior_scale": {"mode": "de_spread_target", "target_std_m": 0.5}}}
    )
    assert cfg.uses_de_spread_prior_scale is True
    assert cfg.uses_calibrated_prior_scale is True
    assert cfg.de_spread_target_std_m == pytest.approx(0.5)


def test_de_spread_target_requires_positive_physical_target():
    with pytest.raises(NEONConfigError, match="target_std_m"):
        load_neon_config({"neon": {"prior_scale": {"mode": "de_spread_target"}}})


def test_dirichlet_particle_mode_requires_matching_persistent_support():
    cfg = NEONStage2Config(
        epistemic_index_mode="dirichlet_particles",
        d_e=16,
        dirichlet_num_particles=16,
        epistemic_basis="identity",
        concat_index=False,
    ).validate()
    assert cfg.dirichlet_num_particles == 16
    with pytest.raises(NEONConfigError, match="d_e =="):
        NEONStage2Config(
            epistemic_index_mode="dirichlet_particles",
            d_e=8,
            dirichlet_num_particles=16,
            epistemic_basis="identity",
            concat_index=False,
        ).validate()


def test_selection_metric_rejects_unknown_mode():
    with pytest.raises(NEONConfigError, match="selection_metric"):
        NEONStage2Config(selection_metric="retention_gate").validate()


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


def test_to_member_bootstrap_config_dict_maps_nested_block():
    cfg = NEONStage2Config(
        member_bootstrap_enabled=False,
        member_bootstrap_temperature=0.3,
        member_bootstrap_seed=22,
    )
    assert cfg.to_member_bootstrap_config_dict() == {
        "enabled": False,
        "temperature": 0.3,
        "seed": 22,
    }


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
    assert cfg.branch_type == "projected"
    assert cfg.train_hidden_channels == 16
    assert cfg.prior_hidden_channels == 16
    assert cfg.bootstrap_enabled is True
    assert cfg.bootstrap_distribution == "probit_exponential"
    assert cfg.bootstrap_temperature == pytest.approx(0.5)
    assert cfg.bootstrap_normalize == "per_epistemic_batch"
    assert cfg.bootstrap_min_weight == pytest.approx(0.05)
    assert cfg.bootstrap_max_weight == pytest.approx(5.0)
    assert cfg.bootstrap_seed == 0
    assert cfg.member_bootstrap_enabled is False
    assert cfg.member_bootstrap_temperature == pytest.approx(1.0)
    assert cfg.member_bootstrap_seed == 1
    assert cfg.prior_rff_dim == 0
    assert cfg.prior_rff_lengthscale == pytest.approx(0.25)
    assert cfg.prior_rff_include_lead is True
    assert cfg.selection_min_retention == pytest.approx(0.0)
    assert cfg.selection_rmse_margin_m == pytest.approx(0.001)
    assert cfg.epistemic_basis == "hermite_random_projection"
    assert cfg.deterministic_head is True
    assert cfg.deterministic_head_feature == "canonical_aleatory_mean"
    assert cfg.calibration_families == 4
    assert cfg.calibration_m == 64
    assert cfg.cancellation_diagnostics_enabled is True
    assert cfg.cancellation_warn_cosine_below == pytest.approx(-0.90)
    assert cfg.cancellation_warn_cancellation_above == pytest.approx(0.80)
