"""TDD tests for NEON Gap 5: integrated nested evaluation metrics."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_module(name: str, rel_path: str):
    for pkg in ("neuralop", "neuralop.flood", "neuralop.flood.eval"):
        _ensure_pkg(pkg)
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")
eval_neon = _load_module("neuralop.flood.eval.neon", "neuralop/flood/eval/neon.py")

neon_predictive_metrics = eval_neon.neon_predictive_metrics
neon_epistemic_error_correlation = eval_neon.neon_epistemic_error_correlation
evaluate_neon_nested = eval_neon.evaluate_neon_nested


def test_ensemble_mean_rmse_known_value():
    # all forecasts = 1, all references = 0 -> ensemble-mean RMSE = 1
    pred = torch.ones(1, 2, 3, 4, 5, 1)   # [B,M,K,T,Nv,C]
    ref = torch.zeros(1, 4, 4, 5, 1)      # [B,R,T,Nv,C]
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,))
    assert m["ensemble_mean_rmse"] == pytest.approx(1.0)


def test_marginal_fair_crps_matches_core_fixture():
    # 2 epistemic x 1 aleatory flattened -> 2 members {0, 2}; ref {0, 2}.
    pred = torch.tensor([[[[[[0.0]]]], [[[[2.0]]]]]])  # [1,2,1,1,1,1]
    ref = torch.tensor([[[[[0.0]]], [[[2.0]]]]])       # [1,2,1,1,1]
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,))
    # flattened ensemble {0,2} vs ref {0,2}: E|X-Y| = 1.0 ; fair self term 1.0
    assert m["marginal_fair_crps"] == pytest.approx(0.0, abs=1e-6)


def test_brier_exceedance_known_value():
    # threshold 0.5: forecast members all exceed (prob 1), reference none (prob 0)
    pred = torch.ones(1, 2, 2, 1, 3, 1)   # all = 1 > 0.5
    ref = torch.zeros(1, 3, 1, 3, 1)      # all = 0 < 0.5
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,))
    # (1 - 0)^2 = 1 everywhere
    assert m["brier_wd_exceed_0.5m"] == pytest.approx(1.0)


def test_csi_perfect_and_zero_overlap():
    # ensemble mean forecast wet exactly where reference is wet -> CSI = 1.
    # Use a realistic 2-member ensemble (M=1, K=2).
    pred = torch.zeros(1, 1, 2, 1, 4, 1)
    ref = torch.zeros(1, 2, 1, 4, 1)
    pred[..., :2, :] = 1.0   # first two cells wet
    ref[..., :2, :] = 1.0
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,))
    assert m["csi_0.5m"] == pytest.approx(1.0)

    # disjoint wetting -> CSI = 0
    pred2 = torch.zeros(1, 1, 2, 1, 4, 1)
    ref2 = torch.zeros(1, 2, 1, 4, 1)
    pred2[..., :2, :] = 1.0
    ref2[..., 2:, :] = 1.0
    m2 = neon_predictive_metrics(pred2, ref2, thresholds=(0.5,))
    assert m2["csi_0.5m"] == pytest.approx(0.0)


def test_marginal_fair_crps_is_nan_for_single_member_ensemble():
    # A degenerate single-member ensemble must not crash; CRPS reported NaN.
    pred = torch.ones(1, 1, 1, 1, 3, 1)
    ref = torch.zeros(1, 2, 1, 3, 1)
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,))
    assert m["marginal_fair_crps"] != m["marginal_fair_crps"]  # NaN
    assert "csi_0.5m" in m  # other metrics still computed


def test_predictive_metrics_respect_weights_on_brier():
    pred = torch.zeros(1, 2, 1, 1, 2, 1)
    ref = torch.zeros(1, 2, 1, 2, 1)
    pred[..., 1, :] = 1.0   # node 1 forecast exceeds; ref never
    # weight node 0 only -> node 1 disagreement is masked out -> brier 0
    weights = torch.tensor([[[[1.0], [0.0]]]])  # [B,T,Nv,C]
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,), weights=weights)
    assert m["brier_wd_exceed_0.5m"] == pytest.approx(0.0)


def test_epistemic_error_correlation_is_high_when_variance_tracks_error():
    # Build predictions whose epistemic spread grows with the (fixed) error.
    # Cell c has reference 0; epistemic members centered at error_c with spread
    # proportional to error_c -> |mean error| and epistemic variance co-vary.
    torch.manual_seed(0)
    B, M, K, T, Nv, C = 1, 8, 4, 1, 6, 1
    errors = torch.linspace(0.1, 2.0, Nv).view(1, 1, 1, 1, Nv, 1)
    # per-epistemic-particle mean spread scaled by error
    epi_offsets = torch.randn(1, M, 1, 1, 1, 1)
    pred = errors + errors * epi_offsets + 0.01 * torch.randn(B, M, K, T, Nv, C)
    ref = torch.zeros(B, 1, T, Nv, C)
    out = neon_epistemic_error_correlation(pred, ref)
    assert out["epistemic_abs_error_spatial_corr"] > 0.7


def test_evaluate_neon_nested_uses_weights_for_variance_summary():
    pred = torch.zeros(1, 2, 2, 1, 2, 1)
    pred[:, 1, :, :, 1, :] = 10.0  # epistemic variance only at masked-out node 1
    ref = torch.zeros(1, 2, 1, 2, 1)
    weights = torch.tensor([[[1.0], [0.0]]])

    bundle = evaluate_neon_nested(pred, ref, thresholds=(0.5,), weights=weights)

    assert bundle["variance_epistemic_mean"] == pytest.approx(0.0)
    assert bundle["variance_epistemic_anova_corrected_mean"] == pytest.approx(0.0)


def test_evaluate_neon_nested_bundles_predictive_and_epistemic():
    pred = torch.randn(1, 4, 3, 2, 5, 1).abs()
    ref = torch.randn(1, 6, 2, 5, 1).abs()
    bundle = evaluate_neon_nested(pred, ref, thresholds=(0.1, 0.3, 0.5))
    # predictive
    for key in ("ensemble_mean_rmse", "marginal_fair_crps", "brier_wd_exceed_0.3m", "csi_0.3m"):
        assert key in bundle
    # epistemic (domain-averaged variance summary + correlation)
    for key in (
        "variance_aleatory_mean",
        "variance_epistemic_mean",
        "variance_epistemic_anova_corrected_mean",
        "epistemic_abs_error_spatial_corr",
    ):
        assert key in bundle
    assert all(isinstance(v, float) for v in bundle.values())


def test_predictive_metrics_accept_TNvC_weights_regression():
    # Regression: [T, Nv, C] weights (the shape NEONFamilySample.weights uses)
    # must broadcast through the weighted RMSE/Brier path. Previously
    # _weighted_mean left-padded on the wrong axis and this raised.
    pred = torch.zeros(1, 2, 1, 1, 2, 1)
    ref = torch.zeros(1, 2, 1, 2, 1)
    pred[..., 1, :] = 1.0   # node 1 forecast exceeds; ref never
    weights = torch.tensor([[[1.0], [0.0]]])  # [T, Nv, C] = [1, 2, 1]
    assert weights.shape == (1, 2, 1)
    m = neon_predictive_metrics(pred, ref, thresholds=(0.5,), weights=weights)
    # node 1 disagreement masked out -> brier 0, rmse 0 (only wet node 0 agrees)
    assert m["brier_wd_exceed_0.5m"] == pytest.approx(0.0)
    assert m["ensemble_mean_rmse"] == pytest.approx(0.0)


def test_evaluate_neon_nested_accepts_TNvC_weights_regression():
    pred = torch.randn(1, 4, 3, 2, 5, 1).abs()
    ref = torch.randn(1, 6, 2, 5, 1).abs()
    weights = torch.ones(2, 5, 1)  # [T, Nv, C]
    bundle = evaluate_neon_nested(pred, ref, thresholds=(0.1, 0.3, 0.5), weights=weights)
    assert torch.isfinite(torch.tensor(list(bundle.values()))).all()
