from __future__ import annotations

import numpy as np
import torch

from neuralop.flood.eval.impact_metrics import (
    compute_nested_flood_impact_crps_metrics,
    ensemble_crps_scalar,
)


def _fixture():
    # [M, K, T, Nv]; each cell is 0.5 km^2.
    pred = np.asarray(
        [
            [
                [[0.2, 0.0], [0.2, 0.2]],
                [[0.0, 0.0], [0.2, 0.0]],
            ],
            [
                [[0.2, 0.2], [0.2, 0.2]],
                [[0.0, 0.2], [0.0, 0.2]],
            ],
        ],
        dtype=np.float64,
    )
    ref = np.asarray(
        [
            [[0.2, 0.0], [0.2, 0.2]],
            [[0.0, 0.0], [0.2, 0.0]],
        ],
        dtype=np.float64,
    )
    geometry = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    static = np.asarray([[0.0, 500_000.0], [0.0, 500_000.0]])
    return pred, ref, geometry, static


def test_nested_impact_crps_uses_crossed_sampling_design_and_km2_units():
    pred, ref, geometry, static = _fixture()

    result = compute_nested_flood_impact_crps_metrics(
        pred,
        ref,
        geometry,
        static_raw=static,
        sampling_design="crossed_common_random_numbers",
    )

    # Compare the first lead against the public crossed score on the same
    # derived inundated-area quantity. This catches accidental M*K flattening.
    pred_area = ((pred[:, :, 0] >= 0.1) * 0.5).sum(axis=-1)
    ref_area = ((ref[:, 0] >= 0.1) * 0.5).sum(axis=-1)
    expected = torch.tensor(pred_area).reshape(1, 2, 2, 1, 1, 1)
    expected_ref = torch.tensor(ref_area).reshape(1, 2, 1, 1, 1)
    from neuralop.flood.neon import crossed_fair_crps_members

    score = crossed_fair_crps_members(expected, expected_ref)
    np.testing.assert_allclose(result["crps_total_inundated_area_km2"][0], score.item())
    assert result["inundation_threshold_m"] == 0.1
    assert result["area_unit_scale_m2_per_output_unit"] == 1_000_000.0
    assert result["arrival_censor_step"] == 3.0


def test_crossed_nested_impact_is_invariant_to_collapsed_epistemic_duplication():
    _, ref, geometry, static = _fixture()
    base = np.asarray(
        [
            [[0.2, 0.0], [0.2, 0.2]],
            [[0.0, 0.0], [0.2, 0.0]],
        ],
        dtype=np.float64,
    )  # [K,T,Nv]
    nested = np.repeat(base[None, ...], 5, axis=0)

    result = compute_nested_flood_impact_crps_metrics(
        nested,
        ref,
        geometry,
        static_raw=static,
        sampling_design="crossed_common_random_numbers",
    )

    base_area = ((base >= 0.1) * 0.5).sum(axis=-1)
    ref_area = ((ref >= 0.1) * 0.5).sum(axis=-1)
    expected = np.asarray(
        [ensemble_crps_scalar(base_area[:, t], ref_area[:, t]) for t in range(2)]
    )
    np.testing.assert_allclose(result["crps_total_inundated_area_km2"], expected)


def test_nested_impact_supports_fixed_and_independent_designs():
    pred, ref, geometry, static = _fixture()

    fixed = compute_nested_flood_impact_crps_metrics(
        pred,
        ref,
        geometry,
        static_raw=static,
        sampling_design="fixed_epistemic_support_common_random_numbers",
    )
    independent = compute_nested_flood_impact_crps_metrics(
        pred,
        ref,
        geometry,
        static_raw=static,
        sampling_design="independent_nested",
    )

    assert np.isfinite(fixed["crps_peak_inundated_area_km2"]).all()
    assert np.isfinite(independent["crps_arrival_time_step"])


def test_nested_impact_excludes_structural_dry_cells():
    pred, ref, geometry, static = _fixture()
    pred[..., 1] = 9.0
    ref[..., 1] = 0.0

    masked = compute_nested_flood_impact_crps_metrics(
        pred,
        ref,
        geometry,
        static_raw=static,
        wettable_mask=np.asarray([True, False]),
        sampling_design="crossed_common_random_numbers",
    )
    explicit = compute_nested_flood_impact_crps_metrics(
        pred[..., :1],
        ref[..., :1],
        geometry[:1],
        static_raw=static[:1],
        sampling_design="crossed_common_random_numbers",
    )

    np.testing.assert_allclose(
        masked["crps_total_inundated_area_km2"],
        explicit["crps_total_inundated_area_km2"],
    )
    np.testing.assert_allclose(
        masked["crps_peak_inundated_area_km2"],
        explicit["crps_peak_inundated_area_km2"],
    )
    assert masked["crps_arrival_time_step"] == explicit["crps_arrival_time_step"]


def test_nested_impact_rejects_unknown_sampling_design():
    pred, ref, geometry, static = _fixture()
    try:
        compute_nested_flood_impact_crps_metrics(
            pred,
            ref,
            geometry,
            static_raw=static,
            sampling_design="flattened_iid",
        )
    except ValueError as exc:
        assert "sampling design" in str(exc)
    else:
        raise AssertionError("unknown design was accepted")
