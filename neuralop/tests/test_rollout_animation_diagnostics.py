import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from neuralop.flood.eval import render
from neuralop.flood.eval.render import (
    _diagnostic_boundary_panels,
    _save_hydrograph_uq_figures_and_animation,
)
from neuralop.flood.eval.rollout import _masked_relative_l2


def _tiny_rollout_fields(n_steps=3, n_cells=4):
    base = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ],
        dtype=np.float64,
    )[:n_steps, :n_cells]
    pred = base + 0.05
    return {
        "pred_mean_by_channel": {"wd": pred},
        "pred_std_by_channel": {"wd": np.full_like(pred, 0.02)},
        "gt_mean_by_channel": {"wd": base},
        "gt_std_by_channel": {"wd": np.full_like(pred, 0.01)},
    }


def test_diagnostic_boundary_panels_are_dataset_adaptive():
    assert _diagnostic_boundary_panels(["stage", "precipitation"], 2) == [
        (0, "line"),
        (1, "bar"),
    ]
    assert _diagnostic_boundary_panels(["inflow"], 1) == [(0, "line")]
    assert _diagnostic_boundary_panels(["stage"], 1) == [(0, "line")]


def test_masked_relative_l2_uses_domain_norm_and_mask():
    pred = np.array([2.0, 100.0, 3.0])
    gt = np.array([1.0, 100.0, 1.0])
    wettable = np.array([True, False, True])
    expected = np.sqrt((1.0 ** 2) + (2.0 ** 2)) / np.sqrt((1.0 ** 2) + (1.0 ** 2))
    assert np.isclose(_masked_relative_l2(pred, gt, wettable), expected)


def test_hydrograph_animation_writes_coastal_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ANIMATION_FPS", 1)
    monkeypatch.setattr(render, "ANIMATION_INTERVAL_MS", 20)
    monkeypatch.setattr(render, "PUBLICATION_TIMESTEPS", [0])
    geometry = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    fields = _tiny_rollout_fields()
    boundary = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.2],
            [1.3, 0.5],
            [1.2, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    _save_hydrograph_uq_figures_and_animation(
        geometry=geometry,
        target_variables=["wd"],
        out_dir=str(tmp_path),
        hydrograph_id="coastal_demo",
        dt_seconds=1200.0,
        n_ref_sims=2,
        n_ens=2,
        boundary_series_raw=boundary,
        boundary_channel_names=["stage", "precipitation"],
        relative_l2_by_channel={"wd": np.array([0.1, 0.2, 0.15])},
        rollout_start_index=2,
        **fields,
    )

    gif = tmp_path / "uq_figures_per_hydrograph" / "uq_rollout_coastal_demo.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0


def test_hydrograph_animation_writes_dynamic_inflow_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ANIMATION_FPS", 1)
    monkeypatch.setattr(render, "ANIMATION_INTERVAL_MS", 20)
    monkeypatch.setattr(render, "PUBLICATION_TIMESTEPS", [0])
    geometry = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    fields = _tiny_rollout_fields()
    boundary = np.array([[20.0], [22.0], [25.0], [23.0], [21.0]], dtype=np.float64)

    _save_hydrograph_uq_figures_and_animation(
        geometry=geometry,
        target_variables=["wd"],
        out_dir=str(tmp_path),
        hydrograph_id="dynamic_demo",
        dt_seconds=1200.0,
        n_ref_sims=2,
        n_ens=2,
        boundary_series_raw=boundary,
        boundary_channel_names=["inflow"],
        relative_l2_by_channel={"wd": np.array([0.1, 0.2, 0.15])},
        rollout_start_index=2,
        **fields,
    )

    gif = tmp_path / "uq_figures_per_hydrograph" / "uq_rollout_dynamic_demo.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0
