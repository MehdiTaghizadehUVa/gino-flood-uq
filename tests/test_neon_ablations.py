"""TDD tests for NEON Gap 7: ablations + deep-ensemble comparison + variance maps."""

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

NEONEpistemicCorrection = neon.NEONEpistemicCorrection
pooled_fair_crps = neon.pooled_fair_crps
per_epistemic_fair_crps = neon.per_epistemic_fair_crps
deep_ensemble_epistemic_variance = eval_neon.deep_ensemble_epistemic_variance
compare_epistemic_maps = eval_neon.compare_epistemic_maps
write_variance_maps = eval_neon.write_variance_maps


# ---------------------------------------------------------------------------
# za_independent ablation
# ---------------------------------------------------------------------------


def test_za_independent_correction_is_shared_across_aleatory_members():
    torch.manual_seed(0)
    B, K, T, Nv, Cphi, d_e = 1, 3, 2, 4, 6, 4
    head = NEONEpistemicCorrection(
        feature_channels=Cphi, out_channels=1, epistemic_dim=d_e, hidden_channels=8,
        za_dependent=False,
    )
    features = torch.randn(B, K, T, Nv, Cphi)   # differ across K
    z_e = torch.randn(2, d_e)
    base = torch.zeros(B, K, T, Nv, 1)
    corr = head(base, features, z_e).correction  # [B,M,K,T,Nv,1]
    # za_independent: correction identical across the aleatory (K) axis.
    for k in range(1, K):
        torch.testing.assert_close(corr[:, :, 0], corr[:, :, k])


def test_za_dependent_correction_varies_across_aleatory_members():
    torch.manual_seed(0)
    B, K, T, Nv, Cphi, d_e = 1, 3, 2, 4, 6, 4
    head = NEONEpistemicCorrection(
        feature_channels=Cphi, out_channels=1, epistemic_dim=d_e, hidden_channels=8,
        za_dependent=True,
    )
    features = torch.randn(B, K, T, Nv, Cphi)
    z_e = torch.randn(2, d_e)
    base = torch.zeros(B, K, T, Nv, 1)
    corr = head(base, features, z_e).correction
    assert not torch.allclose(corr[:, :, 0], corr[:, :, 1])


def test_za_dependent_flag_survives_checkpoint():
    head = NEONEpistemicCorrection(
        feature_channels=5, out_channels=1, epistemic_dim=3, hidden_channels=7, za_dependent=False
    )
    import tempfile
    base = torch.randn(1, 2, 3, 4, 1)
    features = torch.randn(1, 2, 3, 4, 5)
    z_e = torch.randn(2, 3)
    expected = head(base, features, z_e).prediction.detach()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "neon.pt"
        neon.save_neon_stage2_checkpoint(path, head)
        loaded, _ = neon.load_neon_stage2_checkpoint(path)
    assert loaded.za_dependent is False
    torch.testing.assert_close(loaded(base, features, z_e).prediction.detach(), expected)


# ---------------------------------------------------------------------------
# Pooled (za, ze) negative control
# ---------------------------------------------------------------------------


def test_pooled_fair_crps_differs_from_per_epistemic_under_epistemic_spread():
    # Two epistemic particles far apart. Per-ze scoring correctly penalizes the
    # miscalibrated far particle (high mean score). Pooling both into one wide
    # ensemble inflates the fair-CRPS self-distance term, MASKING the
    # miscalibration and yielding a *lower* score -- exactly why pooled
    # (za, ze) CRPS is a negative control, not the main objective.
    pred = torch.tensor(
        [[
            [[[[0.0]]], [[[0.2]]]],     # particle 0: {0.0, 0.2}
            [[[[10.0]]], [[[10.2]]]],   # particle 1: {10.0, 10.2}
        ]]
    )  # [1,2,2,1,1,1]
    ref = torch.tensor([[[[[0.0]]], [[[0.2]]]]])  # ref ~ particle 0
    per = float(per_epistemic_fair_crps(pred, ref, reduction="mean").item())
    pooled = float(pooled_fair_crps(pred, ref, reduction="mean").item())
    assert pooled < per - 1.0  # pooling masks the miscalibration -> lower score


def test_pooled_fair_crps_matches_flattened_marginal():
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 2, 5, 1)
    ref = torch.randn(2, 6, 2, 5, 1)
    from neuralop.flood.neon import flatten_nested_predictions
    flat = flatten_nested_predictions(pred).unsqueeze(1)  # [B,1,M*K,T,Nv,C]
    expected = per_epistemic_fair_crps(flat, ref, reduction="mean")
    got = pooled_fair_crps(pred, ref, reduction="mean")
    torch.testing.assert_close(got, expected)


# ---------------------------------------------------------------------------
# Deep-ensemble comparison
# ---------------------------------------------------------------------------


def test_deep_ensemble_epistemic_variance():
    # model_means [B, J, T, Nv, C]; variance over J models.
    model_means = torch.tensor([[[[[[0.0]]]], [[[[2.0]]]]]]).reshape(1, 2, 1, 1, 1)
    v = deep_ensemble_epistemic_variance(model_means)
    assert v.shape == (1, 1, 1, 1)
    assert torch.allclose(v, torch.tensor([[[[2.0]]]]))  # var([0,2], unbiased) = 2


def test_compare_epistemic_maps_perfectly_aligned():
    a = torch.tensor([[[[0.1, 0.5, 0.9, 0.3]]]])  # [B,T,Nv? ...] flatten-friendly
    b = a.clone() * 3.0
    out = compare_epistemic_maps(a, b, top_q=0.5)
    assert out["spatial_corr"] == pytest.approx(1.0, abs=1e-5)
    assert out["topq_overlap"] == pytest.approx(1.0)
    assert out["variance_ratio"] == pytest.approx((a.mean() / b.mean()).item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Variance-map plotting
# ---------------------------------------------------------------------------


def test_write_variance_maps_emits_pngs(tmp_path):
    torch.manual_seed(0)
    B, M, K, T, Nv, C = 1, 4, 3, 1, 6, 1
    pred = torch.randn(B, M, K, T, Nv, C)
    geometry_xy = torch.rand(Nv, 2) * 1000.0
    paths = write_variance_maps(
        pred, geometry_xy=geometry_xy, output_dir=tmp_path, label="neon", time_index=0
    )
    # aleatory, epistemic, epistemic_anova_corrected, total -> 4 PNGs
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 0 and p.suffix == ".png" for p in paths)
