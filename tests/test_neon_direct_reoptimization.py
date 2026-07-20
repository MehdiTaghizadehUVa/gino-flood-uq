from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neuralop.flood.eval.neon_direct import (
    BatchedDirectLastLayer,
    DirectFamilyBatch,
    audit_direct_objective,
    direct_particle_mean_scale_interval,
    direct_prediction_scale_interval,
    direct_weighted_scores,
    fit_batched_direct_particles,
    matched_particle_displacement_interval,
    prior_linear_representability_error,
    subset_direct_module,
)
from neuralop.flood.neon import epistemic_bootstrap_weights, sample_epistemic_indices


def _load_direct_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_phase5_direct.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("neon_phase5_direct_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _family():
    torch.manual_seed(5)
    return DirectFamilyBatch(
        family_id="F0",
        base_prediction=torch.randn(3, 2, 4, 1, dtype=torch.float64),
        features=torch.randn(3, 2, 4, 2, dtype=torch.float64),
        reference=torch.randn(4, 2, 4, 1, dtype=torch.float64),
        score_weights=torch.ones(2, 4, 1, dtype=torch.float64),
    )


def test_batched_last_layer_keeps_particle_axis_and_independent_parameters():
    head = BatchedDirectLastLayer(3, feature_channels=2, out_channels=1)
    family = _family()
    output = head(family.features.float())
    assert output.shape == (3, 3, 2, 4, 1)
    assert head.weight.shape == (3, 2, 1)


def test_uniform_particle_weights_reproduce_shared_anchor_gradient():
    family = _family()
    anchor = BatchedDirectLastLayer(1, feature_channels=2, out_channels=1).double()
    particles = BatchedDirectLastLayer(4, feature_channels=2, out_channels=1).double()
    particles.copy_particle_from(anchor, source_particle=0)

    anchor_scores = direct_weighted_scores(
        anchor,
        family,
        particle_weights=torch.ones(1, dtype=torch.float64),
    )
    particle_scores = direct_weighted_scores(
        particles,
        family,
        particle_weights=torch.ones(4, dtype=torch.float64),
    )
    anchor_grad = torch.autograd.grad(anchor_scores.mean(), tuple(anchor.parameters()))
    particle_grad = torch.autograd.grad(particle_scores.mean(), tuple(particles.parameters()))

    torch.testing.assert_close(particle_scores, anchor_scores.expand(4))
    # Averaging four identical particle objectives divides each particle's
    # gradient by four; summing across particles recovers the anchor gradient.
    for observed, expected in zip(particle_grad, anchor_grad):
        torch.testing.assert_close(observed.sum(dim=0), expected.squeeze(0))


def test_subset_direct_module_and_objective_audit_match_selected_particles():
    torch.manual_seed(3)
    family = _family()
    module = BatchedDirectLastLayer(3, feature_channels=2, out_channels=1).double()
    with torch.no_grad():
        module.weight.normal_()
        module.bias.normal_()
    subset = subset_direct_module(module, [2, 0])
    torch.testing.assert_close(
        subset(family.features), module(family.features)[[2, 0]]
    )

    objective, gradient = audit_direct_objective(
        subset,
        family_ids=[family.family_id],
        load_family=lambda _index: family,
        family_particle_weights=torch.ones(1, 2),
        regularization=0.0,
    )
    expected = direct_weighted_scores(
        subset,
        family,
        particle_weights=torch.ones(2, dtype=torch.float64),
    ).mean()
    assert objective == pytest.approx(float(expected.detach()))
    assert gradient > 0.0


def test_direct_objective_regularization_is_invariant_to_particle_duplication():
    """The batched objective is the mean of independent particle risks."""

    family = _family()
    single = BatchedDirectLastLayer(1, feature_channels=2, out_channels=1).double()
    duplicated = BatchedDirectLastLayer(2, feature_channels=2, out_channels=1).double()
    with torch.no_grad():
        single.weight.fill_(0.25)
        single.bias.fill_(-0.1)
        duplicated.copy_particle_from(single, source_particle=0)

    single_objective, _ = audit_direct_objective(
        single,
        family_ids=[family.family_id],
        load_family=lambda _index: family,
        family_particle_weights=torch.ones(1, 1, dtype=torch.float64),
        regularization=0.3,
    )
    duplicated_objective, _ = audit_direct_objective(
        duplicated,
        family_ids=[family.family_id],
        load_family=lambda _index: family,
        family_particle_weights=torch.ones(1, 2, dtype=torch.float64),
        regularization=0.3,
    )

    assert duplicated_objective == pytest.approx(single_objective)


def test_prior_linear_representability_is_zero_for_a_linear_prior_slice():
    family = _family()
    coefficient = torch.tensor([[2.0], [-0.5]], dtype=torch.float64)
    bias = torch.tensor([0.2], dtype=torch.float64)
    prior = torch.einsum("ktnf,fo->ktno", family.features, coefficient) + bias

    result = prior_linear_representability_error(
        family.features,
        prior,
        weights=family.score_weights,
    )

    assert result["relative_squared_error"] < 1.0e-20


def test_prior_linear_representability_scores_every_particle():
    family = _family()
    coefficient = torch.tensor([[2.0], [-0.5]], dtype=torch.float64)
    linear = torch.einsum("ktnf,fo->ktno", family.features, coefficient)
    nonlinear = linear + 0.3 * family.features[..., :1].square()
    prior = torch.stack([linear, nonlinear], dim=0)

    combined = prior_linear_representability_error(
        family.features,
        prior,
        weights=family.score_weights,
        chunk_rows=3,
    )
    separate = [
        prior_linear_representability_error(
            family.features,
            prior[index],
            weights=family.score_weights,
            chunk_rows=3,
        )["relative_squared_error"]
        for index in range(2)
    ]

    assert combined["particle_count"] == 2
    assert combined["relative_squared_error_per_particle"] == pytest.approx(separate)
    assert combined["relative_squared_error"] == pytest.approx(sum(separate) / 2.0)
    assert separate[0] < 1.0e-20
    assert separate[1] > 0.0


def test_direct_scale_interval_recomputes_variance_when_draws_are_resampled():
    # Two families, four direct predictors, two aleatory members.
    predictions = [
        torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        .view(4, 1, 1, 1, 1)
        .expand(-1, 2, 1, 3, 1),
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        .view(4, 1, 1, 1, 1)
        .expand(-1, 2, 1, 3, 1),
    ]
    weights = [torch.ones(1, 3, 1, dtype=torch.float64) for _ in predictions]

    first = direct_prediction_scale_interval(
        predictions, score_weights=weights, replicates=100, seed=2
    )
    second = direct_prediction_scale_interval(
        predictions, score_weights=weights, replicates=100, seed=2
    )

    assert first == second
    assert first["estimate"] > 0.0
    assert first["ci95_lower"] <= first["ci95_upper"]


def test_particle_mean_gram_bootstrap_matches_direct_scale_estimate():
    torch.manual_seed(17)
    predictions = [torch.randn(5, 3, 2, 7, 1) for _ in range(3)]
    means = [value.mean(dim=1) for value in predictions]
    direct = direct_prediction_scale_interval(predictions, replicates=20, seed=9)
    gram = direct_particle_mean_scale_interval(means, replicates=20, seed=9)
    # Means are deliberately stored as float16/float32 in production before
    # the Gram reduction. Agreement is therefore bounded by source precision,
    # not float64 arithmetic in the reduction itself.
    assert gram["estimate"] == pytest.approx(direct["estimate"], rel=1.0e-6)


def test_matched_particle_displacement_uses_family_and_draw_axes():
    # Two families and three paired direct draws.  The anchor is exactly zero,
    # so the expected squared displacement is mean([0, 1, 4]) = 5/3.
    final = [
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        .view(3, 1, 1, 1)
        .expand(-1, 2, 3, 1)
        for _ in range(2)
    ]
    anchor = [torch.zeros_like(value) for value in final]
    weights = [torch.ones(2, 3, 1, dtype=torch.float64) for _ in final]

    result = matched_particle_displacement_interval(
        final,
        anchor,
        score_weights=weights,
        replicates=50,
        seed=11,
    )

    assert result["estimate"] == pytest.approx((5.0 / 3.0) ** 0.5)
    assert result["n_families"] == 2.0
    assert result["n_draws"] == 3.0
    assert result["ci95_lower"] <= result["estimate"] <= result["ci95_upper"]
    assert result["per_draw_rms"] == pytest.approx([0.0, 1.0, 2.0])


def test_direct_fit_selects_best_epoch_by_exact_full_objective():
    families = [_family(), _family()]
    families[1].family_id = "F1"
    module = BatchedDirectLastLayer(2, feature_channels=2, out_channels=1).double()
    prefetched = []

    result = fit_batched_direct_particles(
        module,
        family_ids=[family.family_id for family in families],
        load_family=lambda index: families[index],
        family_particle_weights=torch.ones(2, 2, dtype=torch.float64),
        epochs=2,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        shuffle_seed=4,
        prefetch_family=prefetched.append,
    )

    exact = [row["exact_fit_objective"] for row in result.history]
    assert result.best_loss == pytest.approx(min(exact))
    assert result.best_epoch == exact.index(min(exact))
    assert all("online_fit_loss" in row for row in result.history)
    assert set(prefetched) == {0, 1}


def test_direct_probit_law_uses_bootstrap_space_before_model_projection():
    script = _load_direct_script()
    projection = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    metadata = {
        "bootstrap_index_dim": 4,
        "bootstrap_model_projection": projection.tolist(),
        "bootstrap": {
            "distribution": "probit_exponential",
            "seed": 7,
            "temperature": 0.5,
            "normalize": "per_epistemic_batch",
            "min_weight": 0.05,
            "max_weight": 5.0,
        },
    }
    context = SimpleNamespace(
        train_families=[SimpleNamespace(family_id=f"F{i}") for i in range(5)],
        stage2=SimpleNamespace(epistemic_dim=2),
        stage2_metadata=metadata,
        device=torch.device("cpu"),
    )

    z_model, weights, design = script._support_and_weights(
        context, law="probit", draws=3, seed=11
    )

    bootstrap = sample_epistemic_indices(
        3,
        4,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(11),
    )
    expected_weights = epistemic_bootstrap_weights(
        [f"F{i}" for i in range(5)],
        bootstrap,
        seed=7,
        distribution="probit_exponential",
        temperature=0.5,
        normalize="per_epistemic_batch",
        min_weight=0.05,
        max_weight=5.0,
    ).double()
    torch.testing.assert_close(z_model, bootstrap @ projection)
    torch.testing.assert_close(weights, expected_weights)
    assert design == "crossed"
