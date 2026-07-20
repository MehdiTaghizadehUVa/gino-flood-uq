"""TDD tests for NEON autoregressive feature rollout (Gap 2).

The pure ``autoregressive_feature_rollout`` manages the aleatory-member loop,
the dynamic-history sliding window, member feedback, and time stacking. It is
injected with a ``step_fn`` so it can be exercised without a real FGNO. The
thin ``collect_frozen_fgno_rollout_features`` adapter builds the step_fn from a
frozen GINO and is checked for plumbing/shape/freeze/feedback with a dummy
Stage-1 model.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    package = sys.modules.setdefault(name, types.ModuleType(name))
    package.__path__ = [str(REPO_ROOT.joinpath(*name.split(".")))]


def _load_module(name: str, rel_path: str):
    for pkg in ("neuralop", "neuralop.flood", "neuralop.flood.train"):
        _ensure_pkg(pkg)
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")
train_neon = _load_module("neuralop.flood.train.neon", "neuralop/flood/train/neon.py")

autoregressive_feature_rollout = train_neon.autoregressive_feature_rollout
collect_frozen_fgno_rollout_features = train_neon.collect_frozen_fgno_rollout_features


# ---------------------------------------------------------------------------
# Pure autoregressive rollout
# ---------------------------------------------------------------------------


def test_ar_rollout_shapes_and_time_stacking():
    K, n_hist, Nv, C, Cphi, T = 3, 3, 5, 1, 6, 4

    def step_fn(history, t, latent):
        # history: [n_hist, Nv, C]; return pred [Nv, C], feat [Nv, Cphi]
        pred = history[-1] + 1.0
        feat = torch.full((Nv, Cphi), float(t))
        return pred, feat

    init = torch.zeros(K, n_hist, Nv, C)
    latents = torch.zeros(K, 8)
    out = autoregressive_feature_rollout(
        step_fn=step_fn,
        initial_histories=init,
        aleatory_latents=latents,
        rollout_length=T,
    )
    assert out.base_prediction.shape == (K, T, Nv, C)
    assert out.features.shape == (K, T, Nv, Cphi)


def test_ar_rollout_member_feedback_increments_state():
    # pred = last_history_frame + 1 each step -> with feedback, the predicted
    # value at step t must be (t+1), proving the window slid and fed back.
    K, n_hist, Nv, C, T = 1, 3, 2, 1, 5

    def step_fn(history, t, latent):
        pred = history[-1] + 1.0
        feat = history[-1]
        return pred, feat

    init = torch.zeros(K, n_hist, Nv, C)
    out = autoregressive_feature_rollout(
        step_fn=step_fn,
        initial_histories=init,
        aleatory_latents=torch.zeros(K, 4),
        rollout_length=T,
    )
    expected = torch.arange(1, T + 1, dtype=torch.float32).view(1, T, 1, 1).expand(K, T, Nv, C)
    torch.testing.assert_close(out.base_prediction, expected)


def test_ar_rollout_members_are_independent():
    # Each member starts from a different history offset; feedback must keep
    # them independent (no cross-member contamination).
    K, n_hist, Nv, C, T = 2, 3, 1, 1, 3

    def step_fn(history, t, latent):
        pred = history[-1] + latent.sum()
        return pred, history[-1]

    init = torch.zeros(K, n_hist, Nv, C)
    latents = torch.tensor([[1.0], [10.0]])  # member 0 adds 1, member 1 adds 10
    out = autoregressive_feature_rollout(
        step_fn=step_fn,
        initial_histories=init,
        aleatory_latents=latents,
        rollout_length=T,
    )
    # member 0: 1,2,3 ; member 1: 10,20,30
    assert torch.allclose(out.base_prediction[0].flatten(), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out.base_prediction[1].flatten(), torch.tensor([10.0, 20.0, 30.0]))


def test_ar_rollout_passes_timestep_and_latent_to_step_fn():
    seen = []

    def step_fn(history, t, latent):
        seen.append((int(t), float(latent[0])))
        return history[-1], history[-1]

    autoregressive_feature_rollout(
        step_fn=step_fn,
        initial_histories=torch.zeros(2, 3, 1, 1),
        aleatory_latents=torch.tensor([[7.0], [9.0]]),
        rollout_length=3,
    )
    assert seen == [
        (0, 7.0), (1, 7.0), (2, 7.0),   # member 0, latent 7 across all t
        (0, 9.0), (1, 9.0), (2, 9.0),   # member 1, latent 9
    ]


def test_ar_rollout_rejects_bad_rollout_length():
    with pytest.raises(ValueError, match="rollout_length"):
        autoregressive_feature_rollout(
            step_fn=lambda h, t, z: (h[-1], h[-1]),
            initial_histories=torch.zeros(1, 3, 1, 1),
            aleatory_latents=torch.zeros(1, 2),
            rollout_length=0,
        )


def test_ar_rollout_rejects_mismatched_member_counts():
    with pytest.raises(ValueError, match="aleatory_latents"):
        autoregressive_feature_rollout(
            step_fn=lambda h, t, z: (h[-1], h[-1]),
            initial_histories=torch.zeros(3, 3, 1, 1),
            aleatory_latents=torch.zeros(2, 2),
            rollout_length=2,
        )


def test_ar_rollout_detaches_by_default():
    leaf = torch.zeros(1, 3, 2, 1, requires_grad=True)

    def step_fn(history, t, latent):
        return history[-1] * 2.0, history[-1]

    out = autoregressive_feature_rollout(
        step_fn=step_fn,
        initial_histories=leaf,
        aleatory_latents=torch.zeros(1, 2),
        rollout_length=2,
    )
    assert not out.base_prediction.requires_grad
    assert not out.features.requires_grad


# ---------------------------------------------------------------------------
# Frozen-GINO adapter
# ---------------------------------------------------------------------------


class _DummyStage1(nn.Module):
    """Returns prediction = mean(last history frame) broadcast, + a feature.

    Consumes the flattened GINO-style ``x`` and echoes a deterministic function
    of it so AR feedback is observable across timesteps.
    """

    def __init__(self, n_history=3, out_channels=1, feature_channels=6):
        super().__init__()
        self.n_history = n_history
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, x, ada_in=None, return_features=False, feature_source="decoder_pre_projection", **kwargs):
        # x: [1, Nv, C_in] where the last out_channels of the dynamic block is
        # the most-recent frame. We just add 1 to a fixed slice for determinism.
        b, nv, _ = x.shape
        pred = torch.ones(b, nv, self.out_channels) + x[..., -self.out_channels:]
        feat = torch.zeros(b, nv, self.feature_channels)
        if not return_features:
            return pred
        return {
            "prediction": pred,
            "features": {"decoder_pre_projection": feat},
            "feature_source": feature_source,
        }


def test_collect_rollout_features_produces_nested_time_shapes():
    K, n_hist, Nv, C, Cphi, T = 2, 3, 4, 1, 6, 3
    model = _DummyStage1(n_history=n_hist, out_channels=C, feature_channels=Cphi)
    static = torch.zeros(1, Nv, 7)
    geometry = torch.zeros(1, Nv, 2)
    query_points = torch.zeros(1, 8, 8, 2)
    boundary_sequence = torch.zeros(T + n_hist, Nv, 2)
    initial_histories = torch.zeros(K, n_hist, Nv, C)
    latents = torch.zeros(K, 8)

    batch = collect_frozen_fgno_rollout_features(
        stage1_model=model,
        static=static,
        geometry=geometry,
        query_points=query_points,
        boundary_sequence=boundary_sequence,
        initial_histories=initial_histories,
        aleatory_latents=latents,
        rollout_length=T,
        n_history=n_hist,
    )
    assert batch.base_prediction.shape == (1, K, T, Nv, C)
    assert batch.features.shape == (1, K, T, Nv, Cphi)
    # Frozen.
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_collect_rollout_features_feeds_prediction_back_across_time():
    # DummyStage1 pred = 1 + last dynamic frame. With feedback, the dynamic
    # history grows by 1 each step, so predictions should strictly increase.
    K, n_hist, Nv, C, T = 1, 3, 2, 1, 4
    model = _DummyStage1(n_history=n_hist, out_channels=C, feature_channels=5)
    batch = collect_frozen_fgno_rollout_features(
        stage1_model=model,
        static=torch.zeros(1, Nv, 3),
        geometry=torch.zeros(1, Nv, 2),
        query_points=torch.zeros(1, 4, 4, 2),
        boundary_sequence=torch.zeros(T + n_hist, Nv, 2),
        initial_histories=torch.zeros(K, n_hist, Nv, C),
        aleatory_latents=torch.zeros(K, 4),
        rollout_length=T,
        n_history=n_hist,
    )
    preds = batch.base_prediction[0, 0, :, 0, 0]  # [T]
    # step0: 1 + 0 = 1; step1: 1 + 1 = 2; ... strictly increasing by 1
    torch.testing.assert_close(preds, torch.tensor([1.0, 2.0, 3.0, 4.0]))
