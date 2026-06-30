from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "skill_cost_pareto.py"
spec = importlib.util.spec_from_file_location("skill_cost_pareto", SCRIPT_PATH)
skill_cost = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = skill_cost
spec.loader.exec_module(skill_cost)


def test_balanced_member_indices_are_checkpoint_balanced_and_deterministic():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    idx1 = skill_cost.balanced_member_indices(60, 10, n_models=3, rng=rng1)
    idx2 = skill_cost.balanced_member_indices(60, 10, n_models=3, rng=rng2)
    assert idx1.tolist() == idx2.tolist()
    assert len(idx1) == 10
    assert len(set(idx1.tolist())) == 10
    counts = [int(np.sum((idx1 >= i * 20) & (idx1 < (i + 1) * 20))) for i in range(3)]
    assert max(counts) - min(counts) <= 1


def test_fair_crps_masked_matches_manual_single_location():
    pred = np.array([[0.0], [2.0], [4.0]])
    ref = np.array([[1.0], [3.0]])
    mask = np.array([True])
    cross = np.mean(np.abs(pred[:, None, :] - ref[None, :, :]), axis=(0, 1))[0]
    ordered = 0.0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[0]):
            if i != j:
                ordered += abs(float(pred[i, 0] - pred[j, 0]))
    fair_within = ordered / (pred.shape[0] * (pred.shape[0] - 1))
    expected = cross - 0.5 * fair_within
    assert skill_cost._fair_crps_masked(pred, ref, mask) == expected


def test_brier_masked_uses_reference_exceedance_probability():
    pred = np.array([[0.0, 0.4], [0.5, 0.2]])
    ref = np.array([[0.2, 0.6], [0.4, 0.1], [0.6, 0.7]])
    mask = np.array([True, False])
    # For first cell at q=0.3: pred prob=0.5, ref prob=2/3.
    expected = (0.5 - (2.0 / 3.0)) ** 2
    assert np.isclose(skill_cost._brier_masked(pred, ref, mask, 0.3), expected)


def test_make_subsamples_records_requested_sizes():
    subs = skill_cost.make_subsamples(60, [1, 5], repetitions=3, seed=7, n_models=3)
    assert sorted(subs) == ["1", "5"]
    assert len(subs["1"]) == 3
    assert all(len(x) == 5 for x in subs["5"])


def test_sufficient_statistics_match_direct_masked_metrics():
    pred = np.array(
        [
            [[0.0, 0.2, 0.7], [0.1, 0.4, 0.8]],
            [[0.2, 0.1, 0.5], [0.0, 0.6, 0.9]],
            [[0.3, 0.3, 0.4], [0.2, 0.5, 1.0]],
            [[0.4, 0.0, 0.9], [0.3, 0.7, 0.7]],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [[0.0, 0.1, 0.6], [0.2, 0.5, 0.8]],
            [[0.1, 0.2, 0.7], [0.0, 0.4, 0.9]],
            [[0.2, 0.0, 0.8], [0.1, 0.6, 1.1]],
        ],
        dtype=np.float32,
    )
    mask = np.array([True, False, True])
    idx = [0, 2, 3]
    pred_flat = pred[:, :, mask].reshape(pred.shape[0], -1)
    ref_flat = ref[:, :, mask].reshape(ref.shape[0], -1)
    crps_cross = skill_cost._cross_member_mean_against_reference(pred_flat, ref_flat)
    crps_pair = skill_cost._forecast_pairwise_abs_mean(pred_flat)
    brier_pred_ref, brier_pred_pair, brier_ref_sq = skill_cost._brier_sufficient_stats(pred_flat, ref_flat, 0.3)
    fast_crps, fast_brier = skill_cost._skill_from_sufficient_stats(
        idx,
        crps_cross=crps_cross,
        crps_pair=crps_pair,
        brier_pred_ref=brier_pred_ref,
        brier_pred_pair=brier_pred_pair,
        brier_ref_sq_mean=brier_ref_sq,
    )

    direct_crps_values = []
    direct_brier_values = []
    for t in range(pred.shape[1]):
        direct_crps_values.append(skill_cost._fair_crps_masked(pred[idx, t, :], ref[:, t, :], mask))
        direct_brier_values.append(skill_cost._brier_masked(pred[idx, t, :], ref[:, t, :], mask, 0.3))
    assert np.isclose(fast_crps, np.mean(direct_crps_values))
    assert np.isclose(fast_brier, np.mean(direct_brier_values))
