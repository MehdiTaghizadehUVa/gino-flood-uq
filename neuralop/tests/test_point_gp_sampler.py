import torch

from neuralop.diffusion.point_gp import PointRFFGaussianProcessSampler


def _coords(n=64):
    x = torch.linspace(0, 1, int(n**0.5))
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def test_independent_sampler_variance_scale():
    sampler = PointRFFGaussianProcessSampler(gp_type="independent", sigma=2.0, rff_features=64)
    coords = _coords(64)
    samples = sampler.sample(coords=coords, batch_size=256, n_channels=1)
    # Variance should be close to sigma^2 for white noise.
    var = samples.var(unbiased=False).item()
    assert 2.5 < var < 5.5


def test_rff_sampler_returns_correlated_fields():
    sampler = PointRFFGaussianProcessSampler(
        gp_type="rff_rbf",
        sigma=1.0,
        length_scale=0.2,
        rff_features=256,
        seed=7,
    )
    coords = _coords(64)
    samples = sampler.sample(coords=coords, batch_size=128, n_channels=1)[..., 0]

    # Compare covariance of two nearby points; should be non-trivial.
    a = samples[:, 0]
    b = samples[:, 1]
    cov = ((a - a.mean()) * (b - b.mean())).mean().item()
    assert abs(cov) > 1e-3
