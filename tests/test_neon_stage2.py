import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_module(name: str, rel_path: str):
    _ensure_pkg("neuralop")
    _ensure_pkg("neuralop.flood")
    _ensure_pkg("neuralop.flood.train")
    _ensure_pkg("neuralop.flood.eval")
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")
train_neon = _load_module("neuralop.flood.train.neon", "neuralop/flood/train/neon.py")
scientific_calibration = _load_module(
    "neuralop.flood.eval.scientific_calibration",
    "neuralop/flood/eval/scientific_calibration.py",
)

NEONEpistemicCorrection = neon.NEONEpistemicCorrection
NEONStage2LossWeights = neon.NEONStage2LossWeights
anova_corrected_epistemic_variance = neon.anova_corrected_epistemic_variance
anova_corrected_epistemic_variance_independent = neon.anova_corrected_epistemic_variance_independent
cancellation_diagnostics = neon.cancellation_diagnostics
epistemic_variance_diagnostics = neon.epistemic_variance_diagnostics
compute_stage2_loss = neon.compute_stage2_loss
correction_magnitude_penalty = neon.correction_magnitude_penalty
epistemic_bootstrap_weights = neon.epistemic_bootstrap_weights
freeze_stage1_model = neon.freeze_stage1_model
graph_smoothness_penalty = neon.graph_smoothness_penalty
nested_variance_components = neon.nested_variance_components
per_epistemic_fair_crps = neon.per_epistemic_fair_crps
stage2_fit_score = neon.stage2_fit_score
positivity_penalty = neon.positivity_penalty
load_neon_stage2_checkpoint = neon.load_neon_stage2_checkpoint
save_neon_stage2_checkpoint = neon.save_neon_stage2_checkpoint
temporal_smoothness_penalty = neon.temporal_smoothness_penalty
collect_frozen_fgno_features = train_neon.collect_frozen_fgno_features
save_forecast_artifact = scientific_calibration.save_forecast_artifact
load_forecast_artifact = scientific_calibration.load_forecast_artifact
save_nested_forecast_artifact = _load_module(
    "neuralop.flood.eval.neon", "neuralop/flood/eval/neon.py"
).save_nested_forecast_artifact


def test_epinet_correction_preserves_nested_shape_and_gradient_policy():
    torch.manual_seed(0)
    base = torch.zeros(2, 3, 4, 5, 1)
    features = torch.randn(2, 3, 4, 5, 6)
    z_e = torch.randn(2, 4)
    head = NEONEpistemicCorrection(
        feature_channels=6,
        out_channels=1,
        epistemic_dim=4,
        hidden_channels=8,
        alpha=0.2,
    )

    out = head(base, features, z_e)
    loss = out.prediction.sum() + 0.1 * out.correction.pow(2).mean()
    loss.backward()

    assert out.prediction.shape == (2, 2, 3, 4, 5, 1)
    assert out.correction.shape == (2, 2, 3, 4, 5, 1)
    assert out.prior_correction.shape == (2, 2, 3, 4, 5, 1)
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in head.trainable_branch.parameters()
    )
    assert all(p.grad is None for p in head.prior_branch.parameters())


def test_projected_epinet_is_default_and_uses_smaller_prior_when_configured():
    head = NEONEpistemicCorrection(
        feature_channels=6,
        out_channels=1,
        epistemic_dim=4,
        train_hidden_channels=32,
        prior_hidden_channels=5,
    )

    assert head.branch_type == "projected"
    assert head.train_hidden_channels == 32
    assert head.prior_hidden_channels == 5
    assert all(not p.requires_grad for p in head.prior_branch.parameters())


def test_film_branch_remains_available_as_ablation():
    head = NEONEpistemicCorrection(
        feature_channels=6,
        out_channels=1,
        epistemic_dim=4,
        hidden_channels=8,
        branch_type="film",
    )
    out = head(torch.zeros(1, 2, 3, 4, 1), torch.randn(1, 2, 3, 4, 6), torch.randn(2, 4))

    assert head.branch_type == "film"
    assert out.prediction.shape == (1, 2, 2, 3, 4, 1)


def test_projected_branch_dot_product_matches_constant_coefficients():
    head = NEONEpistemicCorrection(
        feature_channels=2,
        out_channels=1,
        epistemic_dim=3,
        train_hidden_channels=4,
        prior_hidden_channels=2,
        alpha=0.0,
        branch_type="projected",
        concat_index=False,
    )
    for param in head.trainable_branch.mlp.parameters():
        param.data.zero_()
    # Final bias is reshaped to [out_channels=1, d_e=3], then dotted with z.
    head.trainable_branch.mlp[-1].bias.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    base = torch.zeros(1, 1, 1, 2, 1)
    features = torch.randn(1, 1, 1, 2, 2)
    z_e = torch.tensor([[1.0, 10.0, 100.0], [2.0, 0.0, -1.0]])

    out = head(base, features, z_e)

    expected = torch.tensor([321.0, -1.0]).view(1, 2, 1, 1, 1, 1).expand(1, 2, 1, 1, 2, 1)
    torch.testing.assert_close(out.prediction, expected)


def test_freeze_stage1_model_switches_to_eval_and_removes_gradients():
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(p=0.5), nn.Linear(4, 1))
    model.train()

    returned = freeze_stage1_model(model)

    assert returned is model
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_checkpoint_round_trip_preserves_prior_branch_and_architecture():
    torch.manual_seed(4)
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=3,
        hidden_channels=7,
        n_hidden_layers=3,
        alpha=0.17,
    )
    base = torch.randn(1, 2, 3, 4, 1)
    features = torch.randn(1, 2, 3, 4, 5)
    z_e = torch.randn(2, 3)
    expected = head(base, features, z_e).prediction.detach()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "neon.pt"
        save_neon_stage2_checkpoint(
            path,
            head,
            metadata={"stage1_checkpoint": "dummy_fgno.pt"},
        )
        loaded, metadata = load_neon_stage2_checkpoint(path)

    actual = loaded(base, features, z_e).prediction.detach()
    assert metadata["stage1_checkpoint"] == "dummy_fgno.pt"
    assert loaded.n_hidden_layers == 3
    assert loaded.hidden_channels == 7
    assert loaded.branch_type == "projected"
    assert all(not param.requires_grad for param in loaded.prior_branch.parameters())
    torch.testing.assert_close(actual, expected)


def test_per_epistemic_fair_crps_scores_each_epistemic_particle_separately():
    pred = torch.tensor(
        [
            [
                [[[[0.0]]], [[[2.0]]]],
                [[[[10.0]]], [[[12.0]]]],
            ]
        ]
    )
    ref = torch.tensor([[[[[0.0]]], [[[2.0]]]]])

    per_particle = per_epistemic_fair_crps(pred, ref, reduction="none")
    mean_score = per_epistemic_fair_crps(pred, ref, reduction="mean")

    assert per_particle.shape == (1, 2)
    assert torch.allclose(per_particle[0], torch.tensor([0.0, 9.0]))
    assert torch.isclose(mean_score, torch.tensor(4.5))


def test_stage2_fit_score_honors_configured_objective_modes():
    pred = torch.tensor(
        [[[[[[0.0]]], [[[2.0]]]], [[[[4.0]]], [[[6.0]]]]]]
    )  # [B=1, M=2, K=2, T=1, Nv=1, C=1]
    ref = torch.tensor([[[[[0.0]]], [[[2.0]]]]])  # [B=1, R=2, T=1, Nv=1, C=1]

    per_epi = stage2_fit_score(pred, ref, objective="per_epistemic_fcrps")
    pooled = stage2_fit_score(pred, ref, objective="pooled_fcrps")
    l2_mean = stage2_fit_score(pred, ref, objective="l2_mean")

    assert torch.isclose(per_epi, per_epistemic_fair_crps(pred, ref, reduction="mean"))
    assert pooled != per_epi
    # Flattened forecast mean is 3, reference mean is 1 -> MSE = 4.
    assert torch.isclose(l2_mean, torch.tensor(4.0))


def test_default_stage2_loss_is_native_fit_only():
    torch.manual_seed(3)
    base = torch.zeros(1, 2, 2, 3, 1)
    features = torch.randn(1, 2, 2, 3, 4)
    z_e = torch.randn(2, 3)
    head = NEONEpistemicCorrection(
        feature_channels=4,
        out_channels=1,
        epistemic_dim=3,
        train_hidden_channels=8,
        prior_hidden_channels=3,
        alpha=0.2,
    )
    out = head(base, features, z_e)
    ref = torch.zeros(1, 3, 2, 3, 1)

    default_losses = compute_stage2_loss(
        prediction=out.prediction,
        reference=ref,
        correction=out.correction,
        module=head,
    )
    opt_in_losses = compute_stage2_loss(
        prediction=out.prediction,
        reference=ref,
        correction=out.correction,
        module=head,
        loss_weights=NEONStage2LossWeights(rpf=0.1, mag=0.1),
    )

    torch.testing.assert_close(default_losses.total, default_losses.fit)
    torch.testing.assert_close(default_losses.rpf, torch.tensor(0.0))
    torch.testing.assert_close(default_losses.graph, torch.tensor(0.0))
    torch.testing.assert_close(default_losses.time, torch.tensor(0.0))
    torch.testing.assert_close(default_losses.pos, torch.tensor(0.0))
    torch.testing.assert_close(default_losses.mag, torch.tensor(0.0))
    assert opt_in_losses.rpf > 0.0
    assert opt_in_losses.mag > 0.0
    assert opt_in_losses.total > opt_in_losses.fit


def test_per_epistemic_fair_crps_honors_normalized_weights():
    pred = torch.tensor(
        [
            [
                [
                    [[[0.0], [100.0]]],
                    [[[2.0], [100.0]]],
                ]
            ]
        ]
    )
    ref = torch.tensor(
        [
            [
                [[[0.0], [0.0]]],
                [[[2.0], [0.0]]],
            ]
        ]
    )
    weights = torch.tensor([[[[1.0], [0.0]]]])

    score = per_epistemic_fair_crps(pred, ref, weights=weights, reduction="mean")

    assert torch.isclose(score, torch.tensor(0.0))


def test_per_epistemic_fair_crps_honors_sample_weights():
    pred = torch.tensor(
        [
            [
                [[[[0.0]]], [[[2.0]]]],
                [[[[10.0]]], [[[12.0]]]],
            ]
        ]
    )
    ref = torch.tensor([[[[[0.0]]], [[[2.0]]]]])
    weights = torch.tensor([[1.0, 0.0]])

    score = per_epistemic_fair_crps(
        pred, ref, sample_weights=weights, reduction="mean"
    )

    # Unweighted per-particle scores are [0, 9]; sample weights [1, 0]
    # leave mean([0, 0]) = 0.
    assert torch.isclose(score, torch.tensor(0.0))


def test_epistemic_bootstrap_weights_are_reproducible_and_particle_specific():
    z_e = torch.tensor([[1.0, 0.5, -0.5], [-0.3, 1.2, 0.7]])
    family_ids = ["fam-a", "fam-b", "fam-c"]

    w1 = epistemic_bootstrap_weights(family_ids, z_e, seed=123)
    w2 = epistemic_bootstrap_weights(family_ids, z_e, seed=123)
    disabled = epistemic_bootstrap_weights(family_ids, z_e, distribution="none")

    torch.testing.assert_close(w1, w2)
    assert w1.shape == (3, 2)
    assert torch.all(w1 > 0)
    torch.testing.assert_close(w1.mean(dim=0), torch.ones(2), rtol=1e-6, atol=1e-6)
    assert not torch.allclose(w1[:, 0], w1[:, 1])
    torch.testing.assert_close(disabled, torch.ones_like(disabled))


def test_cancellation_diagnostics_detect_anti_correlated_prior():
    train = torch.ones(1, 2, 1, 2, 3, 1)
    prior = -torch.ones_like(train)

    diag = cancellation_diagnostics(
        trainable_correction=train,
        prior_correction=prior,
        alpha=1.0,
    )

    assert diag["train_prior_cosine"] < -0.99
    assert diag["cancellation_fraction"] > 0.99
    assert diag["total_correction_rms"] == 0.0


def test_chunked_fair_crps_matches_unchunked_score():
    torch.manual_seed(1)
    pred = torch.randn(2, 3, 4, 2, 5, 1)
    ref = torch.randn(2, 6, 2, 5, 1)
    weights = torch.rand(2, 2, 5, 1)

    chunked = per_epistemic_fair_crps(pred, ref, weights=weights, reduction="none", chunk_size=3)
    unchunked = per_epistemic_fair_crps(pred, ref, weights=weights, reduction="none", chunk_size=None)

    torch.testing.assert_close(chunked, unchunked, rtol=1e-6, atol=1e-6)


def test_collect_frozen_fgno_features_adds_single_time_dimension_for_one_step_outputs():
    class DummyStage1(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            x = kwargs["x"]
            b, n, _ = x.shape
            return {
                "prediction": torch.zeros(b, n, 1, device=x.device),
                "features": {"decoder_pre_projection": torch.zeros(b, n, 6, device=x.device)},
            }

    stage1 = DummyStage1()
    batch = collect_frozen_fgno_features(
        stage1_model=stage1,
        model_kwargs={"x": torch.zeros(2, 5, 3)},
        aleatory_latents=torch.zeros(4, 8),
    )
    head = NEONEpistemicCorrection(feature_channels=6, out_channels=1, epistemic_dim=3)
    out = head(batch.base_prediction, batch.features, torch.zeros(2, 3))

    assert batch.base_prediction.shape == (2, 4, 1, 5, 1)
    assert batch.features.shape == (2, 4, 1, 5, 6)
    assert out.prediction.shape == (2, 2, 4, 1, 5, 1)
    assert not stage1.training
    assert all(not p.requires_grad for p in stage1.parameters())


def test_nested_variance_components_and_anova_correction():
    pred = torch.tensor(
        [[[[[[0.0]]], [[[2.0]]]], [[[[10.0]]], [[[12.0]]]]]]
    )

    components = nested_variance_components(pred)
    corrected = anova_corrected_epistemic_variance(pred)

    assert torch.allclose(components.aleatory, torch.tensor([[[[2.0]]]]))
    assert torch.allclose(components.epistemic, torch.tensor([[[[50.0]]]]))
    assert torch.allclose(components.total, torch.tensor([[[[52.0]]]]))
    assert torch.allclose(corrected, torch.tensor([[[[50.0]]]]))


def test_crossed_anova_recovers_epistemic_variance_with_shared_aleatory_effect():
    epistemic = torch.tensor([0.0, 1.0, 4.0, 7.0])
    shared_aleatory = torch.tensor([-3.0, -1.0, 2.0, 5.0, 8.0])
    pred = (epistemic[:, None] + shared_aleatory[None, :]).view(1, 4, 5, 1, 1, 1)

    expected = epistemic.var(unbiased=True)
    corrected = anova_corrected_epistemic_variance(pred)

    torch.testing.assert_close(corrected.squeeze(), expected)


def test_independent_nested_estimator_retains_legacy_within_member_correction():
    pred = torch.tensor(
        [[[[[[0.0]]], [[[2.0]]]], [[[[10.0]]], [[[12.0]]]]]]
    )

    corrected = anova_corrected_epistemic_variance_independent(pred)

    assert torch.allclose(corrected, torch.tensor([[[[49.0]]]]))


def test_tempered_exponential_bootstrap_is_even_in_epistemic_index():
    z_e = torch.tensor([[0.2, -0.4, 0.7], [-1.0, 0.5, 0.3]])
    positive = epistemic_bootstrap_weights(["a", "b", "c"], z_e, seed=17)
    negative = epistemic_bootstrap_weights(["a", "b", "c"], -z_e, seed=17)

    torch.testing.assert_close(positive, negative)


def test_nested_variance_handles_single_epistemic_particle():
    pred = torch.randn(1, 1, 3, 2, 4, 1)

    components = nested_variance_components(pred)
    corrected = anova_corrected_epistemic_variance(pred)

    assert torch.all(components.epistemic == 0)
    assert torch.all(corrected == 0)


def test_regularizers_penalize_only_the_intended_behavior():
    constant = torch.ones(1, 2, 3, 4, 5, 1)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    corrected = torch.tensor([[[[[[-1.0], [0.0], [2.0]]]]]])
    weights = torch.tensor([[[[[[1.0], [0.0], [1.0]]]]]])

    assert torch.isclose(graph_smoothness_penalty(constant, edges), torch.tensor(0.0))
    assert torch.isclose(temporal_smoothness_penalty(constant), torch.tensor(0.0))
    assert torch.isclose(positivity_penalty(corrected, zero_threshold=0.0), torch.tensor(1.0 / 3.0))
    assert torch.isclose(positivity_penalty(corrected, zero_threshold=0.0, weights=weights), torch.tensor(0.5))
    assert torch.isclose(correction_magnitude_penalty(torch.ones_like(constant)), torch.tensor(1.0))


def test_forecast_artifact_round_trips_nested_member_metadata():
    pred = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3).numpy()
    ref = torch.zeros(2, 2, 3).numpy()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "family.calibration_artifact.h5"
        save_forecast_artifact(
            path,
            hydrograph_id="family",
            pred_members_wd=pred,
            ref_members_wd=ref,
            member_model_id=["model0", "model0", "model1", "model1"],
            member_sample_id=[0, 1, 0, 1],
            member_epistemic_id=[0, 0, 1, 1],
            member_aleatory_id=[0, 1, 0, 1],
        )
        loaded = load_forecast_artifact(path, load_members=False)

    assert loaded["member_model_id"] == ["model0", "model0", "model1", "model1"]
    assert loaded["member_sample_id"] == ["0", "1", "0", "1"]
    assert loaded["member_epistemic_id"] == [0, 0, 1, 1]
    assert loaded["member_aleatory_id"] == [0, 1, 0, 1]


def test_nested_forecast_artifact_adapter_flattens_members_and_preserves_ids():
    pred = torch.arange(1 * 2 * 2 * 3 * 4 * 1, dtype=torch.float32).reshape(1, 2, 2, 3, 4, 1)
    ref = torch.zeros(5, 3, 4, 1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "nested.calibration_artifact.h5"
        save_nested_forecast_artifact(path, hydrograph_id="nested", prediction=pred, ref_members_wd=ref)
        loaded = load_forecast_artifact(path, load_members=True)

    assert loaded["pred_members_wd"].shape == (4, 3, 4)
    assert loaded["ref_members_wd"].shape == (5, 3, 4)
    assert loaded["member_epistemic_id"] == [0, 0, 1, 1]
    assert loaded["member_aleatory_id"] == [0, 1, 0, 1]


def test_epistemic_chunking_matches_full_fit():
    # M-chunking correctness: per-epistemic fair CRPS averaged over M must equal
    # the size-weighted sum of per-chunk means. This underwrites the
    # epistemic_chunk_size gradient-accumulation path used to fit big rollouts.
    torch.manual_seed(0)
    base = torch.zeros(2, 3, 4, 5, 1)          # [B, K, T, Nv, C]
    features = torch.randn(2, 3, 4, 5, 6)      # [B, K, T, Nv, C_phi]
    z_e = torch.randn(4, 4)                     # [M, d_e]
    ref = torch.randn(2, 3, 4, 5, 1).abs()     # [B, R, T, Nv, C]
    head = NEONEpistemicCorrection(
        feature_channels=6, out_channels=1, epistemic_dim=4, hidden_channels=8, alpha=0.2
    )
    head.eval()
    with torch.no_grad():
        full = float(
            stage2_fit_score(
                head(base, features, z_e).prediction, ref, objective="per_epistemic_fcrps"
            ).item()
        )
        M = int(z_e.shape[0])
        acc = 0.0
        for m in range(M):
            pred_m = head(base, features, z_e[m : m + 1]).prediction
            acc += float(
                stage2_fit_score(pred_m, ref, objective="per_epistemic_fcrps").item()
            ) * (1.0 / M)
    assert abs(acc - full) <= 1e-5 * abs(full) + 1e-6


def test_fair_crps_members_matches_pairwise_reference_implementation():
    # The sort-based O(N log N) fair CRPS must be numerically identical to the
    # pairwise per_epistemic_fair_crps (flattened, M'=1), weighted and not.
    torch.manual_seed(0)
    B, N, T, Nv, C, R = 2, 13, 3, 7, 1, 5
    pred = torch.randn(B, N, T, Nv, C, dtype=torch.float64).abs()
    ref = torch.randn(B, R, T, Nv, C, dtype=torch.float64).abs()
    fast = neon.fair_crps_members(pred, ref, reduction="mean")
    slow = neon.per_epistemic_fair_crps(pred.unsqueeze(1), ref, reduction="mean")
    assert abs(float(fast) - float(slow)) < 1e-10

    weights = torch.rand(T, Nv, C, dtype=torch.float64)
    fast_w = neon.fair_crps_members(pred, ref, weights=weights, reduction="mean")
    slow_w = neon.per_epistemic_fair_crps(pred.unsqueeze(1), ref, weights=weights, reduction="mean")
    assert abs(float(fast_w) - float(slow_w)) < 1e-10

    # ties between members and references must not break the identity
    pred_t = pred.round()
    ref_t = ref.round()
    fast_t = neon.fair_crps_members(pred_t, ref_t, reduction="mean")
    slow_t = neon.per_epistemic_fair_crps(pred_t.unsqueeze(1), ref_t, reduction="mean")
    assert abs(float(fast_t) - float(slow_t)) < 1e-10

    # chunking must not change the result
    fast_c = neon.fair_crps_members(pred, ref, reduction="mean", chunk_size=5)
    assert abs(float(fast_c) - float(slow)) < 1e-10


def test_epistemic_variance_survives_size_one_chunking():
    """Regression: epistemic variance assembled from size-1 z_e chunks must be
    nonzero and identical to the full-M computation. This is the exact failure
    mode behind all-zero epistemic-variance logs under epistemic_chunk_size=1."""
    torch.manual_seed(0)
    base = torch.zeros(2, 3, 4, 5, 1)          # [B, K, T, Nv, C]
    features = torch.randn(2, 3, 4, 5, 6)      # [B, K, T, Nv, C_phi]
    z_e = torch.randn(4, 4)                     # [M, d_e], M=4
    alpha = 0.2
    head = NEONEpistemicCorrection(
        feature_channels=6, out_channels=1, epistemic_dim=4, hidden_channels=8, alpha=alpha
    )
    head.eval()
    with torch.no_grad():
        full = head(base, features, z_e)
        mbar_prior = (alpha * full.prior_correction).mean(dim=2)
        mbar_total = full.trainable_correction.mean(dim=2) + mbar_prior
        ref = epistemic_variance_diagnostics(mbar_total=mbar_total, mbar_prior_scaled=mbar_prior)

        tr_chunks, pr_chunks = [], []
        for m in range(int(z_e.shape[0])):
            o = head(base, features, z_e[m : m + 1])
            tr_chunks.append(o.trainable_correction.mean(dim=2))
            pr_chunks.append((alpha * o.prior_correction).mean(dim=2))
        pr = torch.cat(pr_chunks, dim=1)
        tot = torch.cat(tr_chunks, dim=1) + pr
        chunked = epistemic_variance_diagnostics(mbar_total=tot, mbar_prior_scaled=pr)

    assert ref["total_epistemic_variance"] > 0.0
    assert ref["prior_epistemic_variance"] > 0.0
    assert chunked["total_epistemic_variance"] > 0.0
    assert abs(chunked["total_epistemic_variance"] - ref["total_epistemic_variance"]) <= 1e-6
    assert abs(chunked["prior_epistemic_variance"] - ref["prior_epistemic_variance"]) <= 1e-6
    assert abs(chunked["prior_retention_ratio"] - ref["prior_retention_ratio"]) <= 1e-6
