import torch
import torch.nn as nn

from neuralop.diffusion import ConditioningConfig, ConditionalDDOForecaster, PointRFFGaussianProcessSampler


class DummyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(7, 1)

    def forward(self, input_geom, latent_queries, output_queries, x, ada_in=None):
        return self.lin(x)


def test_conditional_ddo_forecaster_training_and_sampling_smoke():
    denoiser = DummyDenoiser()
    gp = PointRFFGaussianProcessSampler(gp_type="independent", sigma=1.0, rff_features=32)
    model = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp,
        conditioning=ConditioningConfig(add_noisy_target=True, add_time_features=True, time_feature_type="sincos"),
        sampler_num_steps=8,
    )

    b, n = 3, 20
    context = torch.randn(b, n, 4)
    target = torch.randn(b, n, 1)
    geom = torch.rand(1, n, 2)
    q = torch.rand(1, 8, 8, 2)

    sample = {
        "context": context,
        "target": target,
        "input_geom": geom,
        "latent_queries": q,
        "output_queries": geom,
    }
    loss, stats = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert "mse_eps" in stats

    pred = model.sample_next(
        context=context,
        input_geom=geom,
        latent_queries=q,
        output_queries=geom,
        stochastic=True,
    )
    assert pred.shape == (b, n, 1)
    assert torch.isfinite(pred).all()
