import pytest
import torch
import torch.nn as nn

from neuralop.diffusion import ConditioningConfig, ConditionalDDOForecaster, PointRFFGaussianProcessSampler


class _DummyDenoiser(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, 1)

    def forward(self, input_geom, latent_queries, output_queries, x, ada_in=None):
        return self.lin(x)


def _sample(batch: int = 2, points: int = 12):
    context = torch.randn(batch, points, 4)
    target = torch.randn(batch, points, 1)
    geom = torch.rand(1, points, 2)
    q = torch.rand(1, 8, 8, 2)
    return {
        "context": context,
        "target": target,
        "input_geom": geom,
        "latent_queries": q,
        "output_queries": geom,
    }


def test_invalid_time_injection_rejected():
    with pytest.raises(ValueError, match="time_injection"):
        ConditionalDDOForecaster(
            denoiser=_DummyDenoiser(in_features=5),
            gp_sampler=PointRFFGaussianProcessSampler(gp_type="independent"),
            conditioning=ConditioningConfig(time_injection="bad_mode"),
        )


def test_add_noisy_target_required():
    with pytest.raises(ValueError, match="add_noisy_target"):
        ConditionalDDOForecaster(
            denoiser=_DummyDenoiser(in_features=4),
            gp_sampler=PointRFFGaussianProcessSampler(gp_type="independent"),
            conditioning=ConditioningConfig(add_noisy_target=False, time_injection="channel"),
        )


def test_adain_invalid_embedding_dims_rejected():
    with pytest.raises(ValueError, match="time_embedding_dim"):
        ConditionalDDOForecaster(
            denoiser=_DummyDenoiser(in_features=5),
            gp_sampler=PointRFFGaussianProcessSampler(gp_type="independent"),
            conditioning=ConditioningConfig(time_injection="adain", time_embedding_dim=0),
        )
    with pytest.raises(ValueError, match="time_embedding_hidden_dim"):
        ConditionalDDOForecaster(
            denoiser=_DummyDenoiser(in_features=5),
            gp_sampler=PointRFFGaussianProcessSampler(gp_type="independent"),
            conditioning=ConditioningConfig(time_injection="adain", time_embedding_hidden_dim=0),
        )


def test_backward_compatible_channel_defaults_smoke():
    model = ConditionalDDOForecaster(
        denoiser=_DummyDenoiser(in_features=7),
        gp_sampler=PointRFFGaussianProcessSampler(gp_type="independent"),
        conditioning=ConditioningConfig(),
        sampler_num_steps=4,
    )
    loss, _ = model.training_loss(_sample())
    assert torch.isfinite(loss)
