import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "neon_legacy_estimator_remap",
    REPO_ROOT / "scripts" / "neon_legacy_estimator_remap.py",
)
remap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remap)


def test_member_index_matrix_recovers_complete_crossed_order_from_shuffled_ids():
    epistemic = np.array([1, 0, 1, 0])
    aleatory = np.array([1, 0, 0, 1])

    matrix = remap._member_index_matrix(epistemic, aleatory)

    np.testing.assert_array_equal(matrix, np.array([[1, 3], [2, 0]]))


def test_legacy_remap_distinguishes_independent_and_crossed_corrections():
    prediction = np.array([[0.0, 2.0], [10.0, 12.0]], dtype=np.float32)
    prediction = prediction[:, :, None, None]

    raw, independent, crossed = remap._corrected_fields(prediction)

    np.testing.assert_allclose(raw.squeeze(), 50.0)
    np.testing.assert_allclose(independent.squeeze(), 49.0)
    np.testing.assert_allclose(crossed.squeeze(), 50.0)
