import numpy as np

from neuralop.flood.eval.metrics import _compute_csi


def test_compute_csi_matches_expected_value():
    pred = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float32)
    gt = np.array([0.0, 0.0, 0.3, 0.5], dtype=np.float32)
    val = _compute_csi(0.05, pred, gt)
    # tp=2, fp=1, fn=0 -> 2/3
    assert np.isclose(val, 2.0 / 3.0)


def test_maintained_eval_common_imports():
    from neuralop.flood.eval.common import main  # noqa: F401

