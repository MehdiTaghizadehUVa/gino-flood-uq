import torch

from neuralop.diffusion.ddo_schedule import get_vp_cosine_params, low_discrepancy_rand


def test_low_discrepancy_rand_bounds_and_shape():
    t = low_discrepancy_rand(batch_size=32)
    assert t.shape == (32,)
    assert torch.all(t >= 0.0)
    assert torch.all(t < 1.0)


def test_vp_cosine_alpha_sigma_monotonic_and_finite():
    t = torch.linspace(1e-4, 1.0 - 1e-4, 256)
    _, _, _, alpha, sigma = get_vp_cosine_params(t, lmbd0=10.0, lmbd1=-10.0)
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(sigma).all()
    # alpha decreases with t; sigma increases with t.
    assert torch.all(alpha[:-1] >= alpha[1:] - 1e-7)
    assert torch.all(sigma[:-1] <= sigma[1:] + 1e-7)
    assert torch.all(alpha > 0)
    assert torch.all(sigma > 0)
