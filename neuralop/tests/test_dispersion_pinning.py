"""Tests for dispersion pinning and the reference-dispersion table."""

from __future__ import annotations

import pytest
import torch

from neuralop.flood.data.reference_dispersion import (
    ReferenceDispersionTable,
    SIGMA_OVER_MEAN_ABS_DIFFERENCE,
    TRAIN_OVER_TEST_DISPERSION,
    validate_reference_dispersion_artifact,
)
from neuralop.flood.train.alr_fgn import AnchoredLowRankFGNTrainer
from neuralop.flood.train.dispersion_pinning import (
    dispersion_pinning_penalty,
    mean_abs_pairwise,
)


# --------------------------------------------------------------------------
# E|X - X'| estimator
# --------------------------------------------------------------------------
def test_mean_abs_pairwise_reduces_to_absolute_difference_at_k2():
    x = torch.tensor([[[1.0]], [[4.0]]]).permute(1, 0, 2)  # [1, 2, 1]
    assert torch.allclose(mean_abs_pairwise(x, dim=1), torch.tensor([[3.0]]))


def test_mean_abs_pairwise_matches_brute_force():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 2, 7)  # [M, K, B, N]
    k = x.shape[1]
    brute = torch.zeros(3, 2, 7)
    for i in range(k):
        for j in range(k):
            if i != j:
                brute += (x[:, i] - x[:, j]).abs()
    brute /= k * (k - 1)
    assert torch.allclose(mean_abs_pairwise(x, dim=1), brute, atol=1e-6)


def test_mean_abs_pairwise_is_unbiased_for_gaussian():
    """E|X - X'| for unit normals is 2/sqrt(pi); the estimator must hit it."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 40000, 1)
    est = mean_abs_pairwise(x, dim=1).mean()
    target = 2.0 / torch.pi**0.5
    assert abs(float(est) - target) < 0.02


def test_mean_abs_pairwise_rejects_single_draw():
    with pytest.raises(ValueError, match="at least 2 draws"):
        mean_abs_pairwise(torch.zeros(1, 1, 1, 1), dim=1)


# --------------------------------------------------------------------------
# penalty behaviour
# --------------------------------------------------------------------------
def _fixture(model_dispersion: torch.Tensor, ref: torch.Tensor, depth: torch.Tensor):
    """Build predictions whose K=2 spread is exactly ``model_dispersion``."""
    b, n = ref.shape
    half = model_dispersion / 2.0
    preds = torch.stack([-half, half], dim=0).unsqueeze(0).unsqueeze(-1)  # [1,2,B,N,1]
    return preds, depth.unsqueeze(-1)


def test_penalty_is_zero_when_dispersion_matches_reference():
    ref = torch.full((2, 6), 0.10)
    depth = torch.full((2, 6), 0.5)
    preds, target = _fixture(ref.clone(), ref, depth)
    out = dispersion_pinning_penalty(preds, target, ref)
    assert float(out.penalty) == pytest.approx(0.0, abs=1e-12)


def test_penalty_positive_and_gradient_shrinks_over_dispersion():
    ref = torch.full((2, 6), 0.10)
    depth = torch.full((2, 6), 0.5)
    spread = torch.full((2, 6), 0.20, requires_grad=True)
    half = spread / 2.0
    preds = torch.stack([-half, half], dim=0).unsqueeze(0).unsqueeze(-1)
    out = dispersion_pinning_penalty(preds, depth.unsqueeze(-1), ref)
    assert float(out.penalty) > 0
    out.penalty.backward()
    # over-dispersed => gradient must push the spread DOWN
    assert torch.all(spread.grad > 0), "gradient should reduce an over-dispersed channel"


def test_penalty_gradient_grows_an_under_dispersed_channel():
    ref = torch.full((2, 6), 0.20)
    depth = torch.full((2, 6), 0.5)
    spread = torch.full((2, 6), 0.05, requires_grad=True)
    half = spread / 2.0
    preds = torch.stack([-half, half], dim=0).unsqueeze(0).unsqueeze(-1)
    out = dispersion_pinning_penalty(preds, depth.unsqueeze(-1), ref)
    out.penalty.backward()
    assert torch.all(spread.grad < 0), "gradient should widen an under-dispersed channel"


def test_uniform_offset_gives_squared_offset():
    ref = torch.full((1, 8), 0.10)
    depth = torch.full((1, 8), 0.5)          # all in the 'deep' stratum
    preds, target = _fixture(ref + 0.03, ref, depth)
    out = dispersion_pinning_penalty(preds, target, ref)
    assert float(out.penalty) == pytest.approx(0.03**2, rel=1e-5)


def test_opposite_errors_within_one_stratum_cancel_by_design():
    """Documents the deliberate aggregate-then-square trade-off.

    Squaring per cell would penalise the irreducible K=2 sampling noise, so the
    penalty squares the stratum mean instead.  The price is that equal and
    opposite deviations inside a single stratum cancel; stratification is what
    stops that from hiding real structure.
    """
    ref = torch.full((1, 8), 0.10)
    depth = torch.full((1, 8), 0.5)
    model = ref.clone()
    model[0, :4] += 0.03
    model[0, 4:] -= 0.03
    preds, target = _fixture(model, ref, depth)
    out = dispersion_pinning_penalty(preds, target, ref)
    assert float(out.penalty) == pytest.approx(0.0, abs=1e-10)


def test_strata_are_independent():
    """An offset confined to the wet front must not leak into other strata."""
    ref = torch.full((1, 9), 0.10)
    depth = torch.tensor([[0.0, 0.005, 0.008, 0.02, 0.05, 0.09, 0.5, 1.0, 2.0]])
    model = ref.clone()
    model[0, 3:6] += 0.04                     # only the 'front' cells
    preds, target = _fixture(model, ref, depth)
    out = dispersion_pinning_penalty(preds, target, ref)
    assert float(out.per_stratum_mean_deviation_m["front"]) == pytest.approx(0.04, rel=1e-5)
    assert float(out.per_stratum_mean_deviation_m["dry"]) == pytest.approx(0.0, abs=1e-10)
    assert float(out.per_stratum_mean_deviation_m["deep"]) == pytest.approx(0.0, abs=1e-10)
    assert float(out.penalty) == pytest.approx(0.04**2, rel=1e-5)
    assert out.per_stratum_count == {"dry": 3, "front": 3, "deep": 3}


def test_structural_dry_cells_are_excluded():
    ref = torch.full((1, 6), 0.10)
    depth = torch.full((1, 6), 0.5)
    model = ref.clone()
    model[0, :3] += 1.0                       # huge error, but on masked cells
    preds, target = _fixture(model, ref, depth)
    mask = torch.tensor([True, True, True, False, False, False])
    out = dispersion_pinning_penalty(preds, target, ref, structural_dry_mask=mask)
    assert float(out.penalty) == pytest.approx(0.0, abs=1e-10)
    assert out.per_stratum_count["deep"] == 3


def test_empty_stratum_contributes_nothing():
    ref = torch.full((1, 4), 0.10)
    depth = torch.full((1, 4), 0.5)           # nothing dry, nothing at the front
    preds, target = _fixture(ref.clone(), ref, depth)
    out = dispersion_pinning_penalty(preds, target, ref)
    assert out.per_stratum_count["dry"] == 0
    assert float(out.penalty) == pytest.approx(0.0, abs=1e-12)


def test_penalty_rejects_shape_mismatch():
    preds = torch.zeros(1, 2, 2, 5, 1)
    with pytest.raises(ValueError, match=r"reference_dispersion must be"):
        dispersion_pinning_penalty(preds, torch.zeros(2, 5, 1), torch.zeros(2, 4))


# --------------------------------------------------------------------------
# reference dispersion table
# --------------------------------------------------------------------------
def _table(scale: float = 1.0) -> ReferenceDispersionTable:
    disp = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    return ReferenceDispersionTable(family_ids=["famA", "famB"], dispersion=disp, scale=scale)


def test_lookup_indexes_family_and_time():
    t = _table()
    got = t.lookup(["famB", "famA"], torch.tensor([1, 4]))
    assert torch.allclose(got[0], torch.tensor([18.0, 19.0, 20.0]))   # famB, t=1
    assert torch.allclose(got[1], torch.tensor([12.0, 13.0, 14.0]))   # famA, t=4


def test_lookup_step_offsets_time_for_ar_rollout():
    t = _table()
    base = t.lookup(["famA"], torch.tensor([1]))
    stepped = t.lookup(["famA"], torch.tensor([1]), step=2)
    assert torch.allclose(stepped, base + 6.0)     # two rows of 3 cells


def test_lookup_applies_scale():
    scaled = _table(scale=0.5).lookup(["famA"], torch.tensor([0]))
    assert torch.allclose(scaled, _table(1.0).lookup(["famA"], torch.tensor([0])) * 0.5)


def test_default_scale_corrects_train_to_test_dispersion():
    t = ReferenceDispersionTable(
        family_ids=["f"], dispersion=torch.ones(1, 2, 2)
    )
    assert float(t.lookup(["f"], torch.tensor([0]))[0, 0]) == pytest.approx(
        1.0 / TRAIN_OVER_TEST_DISPERSION, rel=1e-6
    )


def test_lookup_rejects_unknown_family_and_out_of_range_time():
    t = _table()
    with pytest.raises(KeyError, match="Unknown family id"):
        t.lookup(["nope"], torch.tensor([0]))
    with pytest.raises(IndexError, match="out of range"):
        t.lookup(["famA"], torch.tensor([4]), step=1)


def test_empirical_sigma_ratio_is_not_the_gaussian_value():
    """A5 measured 0.9705 on real depths; the Gaussian 0.8862 is 9% wrong."""
    assert SIGMA_OVER_MEAN_ABS_DIFFERENCE == pytest.approx(0.9705)
    assert abs(SIGMA_OVER_MEAN_ABS_DIFFERENCE - 0.8862) > 0.05


# --------------------------------------------------------------------------
# trainer integration: units
#
# Training runs in NORMALIZED space; D_ref is in metres.  Comparing the two
# directly would be wrong by the normalizer's scale factor, silently, and the
# penalty would drive the aleatory channel to the wrong target.  These tests
# pin the conversion using a normalizer with a known 2x scale.
# --------------------------------------------------------------------------
class _AffineNormalizer:
    """Physical = 2 * normalized + 10, so dispersion doubles under inversion."""

    def transform(self, values):
        return (values - 10.0) / 2.0

    def inverse_transform(self, values):
        return values * 2.0 + 10.0


class _TrainerStub:
    """Exercises the real trainer methods without constructing a model."""

    _to_physical = AnchoredLowRankFGNTrainer._to_physical
    _dispersion_penalty = AnchoredLowRankFGNTrainer._dispersion_penalty

    def __init__(self, table, weight):
        self.target_normalizer = _AffineNormalizer()
        self.reference_dispersion = table
        self.dispersion_penalty_weight = weight
        self.dispersion_wet_thresholds = (0.01, 0.10)
        self.water_depth_index = 0


def _normalized_batch(normalized_spread: float, n_cells: int = 4):
    half = normalized_spread / 2.0
    centre = torch.full((1, n_cells), 3.0)
    preds = torch.stack([centre - half, centre + half], dim=0)      # [K,B,N]
    preds = preds.unsqueeze(0).unsqueeze(-1)                        # [M=1,K,B,N,C=1]
    target = centre.unsqueeze(-1)
    sample = {"family_id": ["famA"], "time_index": torch.tensor([0])}
    return preds, target, sample


def _flat_table(value_m: float, n_cells: int = 4) -> ReferenceDispersionTable:
    return ReferenceDispersionTable(
        family_ids=["famA"],
        dispersion=torch.full((1, 3, n_cells), value_m),
        scale=1.0,
    )


def test_penalty_compares_physical_depths_not_normalized_ones():
    """A normalized spread of 0.05 is 0.10 m physically; matching that is zero."""
    preds, target, sample = _normalized_batch(0.05)
    stub = _TrainerStub(_flat_table(0.10), weight=1.0)
    out = stub._dispersion_penalty(preds, target, sample)
    assert float(out.penalty) == pytest.approx(0.0, abs=1e-10)


def test_penalty_is_nonzero_if_reference_is_read_in_normalized_units():
    """Guards the units bug: 0.05 would be the target only without inversion.

    Tolerance is 1e-4 rather than 1e-5 because this fixture's normalizer adds a
    deliberately large +10 offset, so recovering a 0.05 m spread from float32
    values near 16 costs about a part in 1e5.  Real normalized depths are O(1),
    where the conditioning is far better.
    """
    preds, target, sample = _normalized_batch(0.05)
    stub = _TrainerStub(_flat_table(0.05), weight=1.0)
    out = stub._dispersion_penalty(preds, target, sample)
    assert float(out.penalty) == pytest.approx(0.05**2, rel=1e-4)


def test_dispersion_penalty_disabled_returns_none():
    """Weight 0 must leave the loss graph untouched, not add an explicit zero."""
    preds, target, sample = _normalized_batch(0.05)
    assert _TrainerStub(_flat_table(0.10), weight=0.0)._dispersion_penalty(
        preds, target, sample
    ) is None
    assert _TrainerStub(None, weight=1.0)._dispersion_penalty(preds, target, sample) is None


def test_dispersion_penalty_requires_time_index():
    preds, target, _ = _normalized_batch(0.05)
    stub = _TrainerStub(_flat_table(0.10), weight=1.0)
    with pytest.raises(KeyError, match="time_index"):
        stub._dispersion_penalty(preds, target, {"family_id": ["famA"]})


def test_ar_step_advances_the_reference_time_index():
    """target_sequence[:, s] lives at raw time index time_index + s."""
    preds, target, sample = _normalized_batch(0.05)
    table = ReferenceDispersionTable(
        family_ids=["famA"],
        dispersion=torch.stack([
            torch.full((4,), 0.10), torch.full((4,), 0.20), torch.full((4,), 0.30),
        ]).unsqueeze(0),
        scale=1.0,
    )
    stub = _TrainerStub(table, weight=1.0)
    # physical spread is 0.10 m, so step 0 (ref 0.10) matches exactly
    assert float(stub._dispersion_penalty(preds, target, sample, step=0).penalty) == pytest.approx(
        0.0, abs=1e-10)
    # step 1 looks up ref 0.20 => deviation -0.10
    assert float(stub._dispersion_penalty(preds, target, sample, step=1).penalty) == pytest.approx(
        0.10**2, rel=1e-4)


def test_validate_rejects_malformed_artifacts():
    good = {
        "schema": "flood_reference_dispersion_v1",
        "family_ids": ["a", "b"],
        "dispersion": torch.ones(2, 3, 4),
        "cell_count": 4,
        "n_time": 3,
    }
    assert validate_reference_dispersion_artifact(good)["n_families"] == 2

    with pytest.raises(ValueError, match="duplicate family"):
        validate_reference_dispersion_artifact({**good, "family_ids": ["a", "a"]})
    with pytest.raises(ValueError, match="non-negative"):
        validate_reference_dispersion_artifact({**good, "dispersion": -torch.ones(2, 3, 4)})
    with pytest.raises(ValueError, match="cell axis"):
        validate_reference_dispersion_artifact({**good, "cell_count": 9})
    with pytest.raises(ValueError, match="missing 1 required families"):
        validate_reference_dispersion_artifact(good, expected_family_ids=["a", "b", "c"])
