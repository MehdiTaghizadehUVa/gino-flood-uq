import torch
import torch.nn as nn

from neuralop.diffusion import ConditioningConfig, ConditionalDDOForecaster, PointRFFGaussianProcessSampler


class DummyDenoiser(nn.Module):
    def __init__(self, in_features: int, adain_dim: int = 0):
        super().__init__()
        self.lin_x = nn.Linear(in_features, 1, bias=False)
        self.lin_adain = nn.Linear(adain_dim, 1, bias=False) if adain_dim > 0 else None
        self.last_ada_in = None

    def forward(self, input_geom, latent_queries, output_queries, x, ada_in=None):
        self.last_ada_in = ada_in
        out = self.lin_x(x)
        if self.lin_adain is not None:
            if ada_in is None:
                raise ValueError("Expected ada_in for adain-mode denoiser.")
            out = out + self.lin_adain(ada_in).unsqueeze(1)
        return out


def _make_sample(b: int = 3, n: int = 20):
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
    return sample


def test_conditional_ddo_forecaster_channel_mode_smoke():
    # context=4, noisy_target=1, time(sincos)=2 -> in_features=7
    denoiser = DummyDenoiser(in_features=7, adain_dim=0)
    gp = PointRFFGaussianProcessSampler(gp_type="independent", sigma=1.0, rff_features=32)
    model = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp,
        conditioning=ConditioningConfig(
            add_noisy_target=True,
            add_time_features=True,
            time_feature_type="sincos",
            time_injection="channel",
        ),
        sampler_num_steps=8,
    )

    sample = _make_sample()
    loss, stats = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert "mse_eps" in stats
    assert denoiser.last_ada_in is None

    pred = model.sample_next(
        context=sample["context"],
        input_geom=sample["input_geom"],
        latent_queries=sample["latent_queries"],
        output_queries=sample["output_queries"],
        stochastic=True,
    )
    assert pred.shape == (sample["context"].shape[0], sample["context"].shape[1], 1)
    assert denoiser.last_ada_in is None
    assert torch.isfinite(pred).all()


def test_conditional_ddo_forecaster_adain_mode_smoke():
    # context=4, noisy_target=1; time goes through ada_in, no extra channels.
    denoiser = DummyDenoiser(in_features=5, adain_dim=32)
    gp = PointRFFGaussianProcessSampler(gp_type="independent", sigma=1.0, rff_features=32)
    model = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp,
        conditioning=ConditioningConfig(
            add_noisy_target=True,
            add_time_features=True,  # ignored for adain
            time_feature_type="sincos",
            time_injection="adain",
            time_embedding_dim=32,
            time_embedding_hidden_dim=64,
            time_embedding_scale=10000.0,
        ),
        sampler_num_steps=8,
    )

    sample = _make_sample()
    loss, stats = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert "mse_eps" in stats
    assert denoiser.last_ada_in is not None
    assert denoiser.last_ada_in.shape == (sample["context"].shape[0], 32)

    pred = model.sample_next(
        context=sample["context"],
        input_geom=sample["input_geom"],
        latent_queries=sample["latent_queries"],
        output_queries=sample["output_queries"],
        stochastic=True,
    )
    assert pred.shape == (sample["context"].shape[0], sample["context"].shape[1], 1)
    assert denoiser.last_ada_in is not None
    assert denoiser.last_ada_in.shape == (sample["context"].shape[0], 32)
    assert torch.isfinite(pred).all()


def test_conditional_ddo_forecaster_adain_timestep_changes_predictions():
    denoiser = DummyDenoiser(in_features=5, adain_dim=32)
    gp = PointRFFGaussianProcessSampler(gp_type="independent", sigma=1.0, rff_features=32)
    model = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp,
        conditioning=ConditioningConfig(
            add_noisy_target=True,
            time_injection="adain",
            time_embedding_dim=32,
            time_embedding_hidden_dim=64,
            time_embedding_scale=10000.0,
        ),
        sampler_num_steps=4,
    )

    sample = _make_sample(b=2, n=10)
    t1 = torch.full((2,), 0.2, dtype=sample["context"].dtype)
    t2 = torch.full((2,), 0.8, dtype=sample["context"].dtype)
    ada1 = model._build_time_adain(t1)
    ada2 = model._build_time_adain(t2)
    assert ada1.shape == ada2.shape == (2, 32)
    assert not torch.allclose(ada1, ada2)

    z_t = torch.zeros(2, 10, 1, dtype=sample["context"].dtype)
    x_in = model._build_denoiser_input(context=sample["context"], z_t=z_t, t=t1)
    out1 = model._predict_eps(
        x_in=x_in,
        input_geom=sample["input_geom"],
        latent_queries=sample["latent_queries"],
        output_queries=sample["output_queries"],
        ada_in=ada1,
    )
    out2 = model._predict_eps(
        x_in=x_in,
        input_geom=sample["input_geom"],
        latent_queries=sample["latent_queries"],
        output_queries=sample["output_queries"],
        ada_in=ada2,
    )
    assert not torch.allclose(out1, out2)
