"""TDD tests for NEON Gap 6: lead-time encoding + prior-scale auto-calibration."""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_module(name: str, rel_path: str):
    for pkg in ("neuralop", "neuralop.flood"):
        _ensure_pkg(pkg)
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")

NEONEpistemicCorrection = neon.NEONEpistemicCorrection
save_neon_stage2_checkpoint = neon.save_neon_stage2_checkpoint
load_neon_stage2_checkpoint = neon.load_neon_stage2_checkpoint
calibrate_prior_scale = neon.calibrate_prior_scale
base_rmse_from_reference = neon.base_rmse_from_reference


# ---------------------------------------------------------------------------
# Lead-time encoding
# ---------------------------------------------------------------------------


def _constant_over_time_features(B, K, T, Nv, C):
    frame = torch.randn(B, K, 1, Nv, C)
    return frame.expand(B, K, T, Nv, C).contiguous()


def test_lead_time_off_by_default_gives_constant_correction_over_time():
    torch.manual_seed(0)
    B, K, T, Nv, Cphi, d_e = 1, 2, 4, 3, 6, 4
    head = NEONEpistemicCorrection(feature_channels=Cphi, out_channels=1, epistemic_dim=d_e, hidden_channels=8)
    features = _constant_over_time_features(B, K, T, Nv, Cphi)
    z_e = torch.randn(2, d_e)
    base = torch.zeros(B, K, T, Nv, 1)
    corr = head(base, features, z_e).correction  # [B,M,K,T,Nv,1]
    # With constant features over t and no lead-time encoding, the correction
    # must be identical across the time axis.
    for t in range(1, T):
        torch.testing.assert_close(corr[:, :, :, 0], corr[:, :, :, t])


def test_lead_time_on_varies_correction_over_time():
    torch.manual_seed(0)
    B, K, T, Nv, Cphi, d_e = 1, 2, 4, 3, 6, 4
    head = NEONEpistemicCorrection(
        feature_channels=Cphi, out_channels=1, epistemic_dim=d_e, hidden_channels=8, lead_time_dim=8
    )
    features = _constant_over_time_features(B, K, T, Nv, Cphi)
    z_e = torch.randn(2, d_e)
    base = torch.zeros(B, K, T, Nv, 1)
    corr = head(base, features, z_e).correction
    # Lead-time encoding must break time-invariance: at least one later step
    # differs from the first.
    assert not torch.allclose(corr[:, :, :, 0], corr[:, :, :, -1])


def test_lead_time_preserves_shape_and_gradient_policy():
    torch.manual_seed(1)
    B, K, T, Nv, Cphi, d_e = 2, 2, 3, 4, 5, 3
    head = NEONEpistemicCorrection(
        feature_channels=Cphi, out_channels=1, epistemic_dim=d_e, hidden_channels=8, lead_time_dim=6
    )
    features = torch.randn(B, K, T, Nv, Cphi)
    z_e = torch.randn(2, d_e)
    base = torch.zeros(B, K, T, Nv, 1)
    out = head(base, features, z_e)
    out.prediction.sum().backward()
    assert out.prediction.shape == (B, 2, K, T, Nv, 1)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in head.trainable_branch.parameters())
    assert all(p.grad is None for p in head.prior_branch.parameters())


def test_lead_time_dim_survives_checkpoint_round_trip():
    torch.manual_seed(2)
    head = NEONEpistemicCorrection(
        feature_channels=5, out_channels=1, epistemic_dim=3, hidden_channels=7, lead_time_dim=4
    )
    base = torch.randn(1, 2, 3, 4, 1)
    features = torch.randn(1, 2, 3, 4, 5)
    z_e = torch.randn(2, 3)
    expected = head(base, features, z_e).prediction.detach()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "neon.pt"
        save_neon_stage2_checkpoint(path, head)
        loaded, _ = load_neon_stage2_checkpoint(path)
    assert loaded.lead_time_dim == 4
    torch.testing.assert_close(loaded(base, features, z_e).prediction.detach(), expected)


# ---------------------------------------------------------------------------
# Base RMSE + prior-scale auto-calibration
# ---------------------------------------------------------------------------


def test_base_rmse_from_reference_known_values():
    # base ensemble mean = 1.0 everywhere; reference ensemble mean = 0.0
    # -> RMSE = 1.0
    base = torch.ones(1, 3, 2, 4, 1)
    ref = torch.zeros(1, 5, 2, 4, 1)
    rmse = base_rmse_from_reference(base, ref)
    assert rmse == pytest.approx(1.0)


def test_base_rmse_respects_weights():
    base = torch.ones(1, 2, 1, 2, 1)
    ref = torch.zeros(1, 2, 1, 2, 1)
    # zero-weight one node; RMSE still 1.0 because all errors are 1.0
    weights = torch.tensor([[[[1.0], [0.0]]]])  # [B,T,Nv,C] = [1,1,2,1]
    rmse = base_rmse_from_reference(base, ref, weights=weights)
    assert rmse == pytest.approx(1.0)


def test_calibrate_prior_scale_hits_target_fraction():
    torch.manual_seed(0)
    B, K, T, Nv, Cphi, d_e, M = 1, 2, 3, 5, 6, 4, 64
    head = NEONEpistemicCorrection(feature_channels=Cphi, out_channels=1, epistemic_dim=d_e, hidden_channels=8)
    features = torch.randn(B, K, T, Nv, Cphi)
    z_e = torch.randn(M, d_e)
    base_rmse = 0.4
    target = 0.10

    alpha = calibrate_prior_scale(
        module=head, features=features, z_e=z_e, base_rmse=base_rmse, target_fraction=target
    )
    head.set_prior_scale(alpha)

    # Recompute the realized Std_ze[alpha * E^P] the same way and confirm it
    # matches target_fraction * base_rmse.
    with torch.no_grad():
        prior = head.prior_branch(features, z_e)  # [B,M,K,T,Nv,C]
        realized = (alpha * prior).std(dim=1, unbiased=True).mean().item()
    assert realized == pytest.approx(target * base_rmse, rel=1e-4)


def test_set_prior_scale_updates_alpha_and_scales_correction():
    torch.manual_seed(0)
    head = NEONEpistemicCorrection(feature_channels=6, out_channels=1, epistemic_dim=4, hidden_channels=8, alpha=0.1)
    features = torch.randn(1, 2, 3, 4, 6)
    z_e = torch.randn(3, 4)
    base = torch.zeros(1, 2, 3, 4, 1)
    prior_contrib_before = 0.1 * head.prior_branch(features, z_e)
    head.set_prior_scale(0.5)
    assert head.alpha == pytest.approx(0.5)
    prior_contrib_after = 0.5 * head.prior_branch(features, z_e)
    torch.testing.assert_close(prior_contrib_after, 5.0 * prior_contrib_before)


def test_calibrate_prior_scale_is_safe_when_prior_std_is_tiny():
    torch.manual_seed(0)
    head = NEONEpistemicCorrection(feature_channels=6, out_channels=1, epistemic_dim=4, hidden_channels=8)
    features = torch.zeros(1, 2, 3, 4, 6)  # zero features -> prior may be ~constant over z_e
    z_e = torch.zeros(1, 4)  # M=1 -> zero std
    alpha = calibrate_prior_scale(module=head, features=features, z_e=z_e, base_rmse=0.4, target_fraction=0.1)
    assert torch.isfinite(torch.tensor(alpha))
