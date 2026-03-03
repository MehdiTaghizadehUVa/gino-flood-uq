import torch

from neuralop.diffusion.ddo_reverse import ddo_denoise_step_vp
from neuralop.diffusion.ddo_schedule import get_vp_cosine_params


def _alpha_sigma(t: torch.Tensor):
    _, _, _, alpha, sigma = get_vp_cosine_params(t, lmbd0=10.0, lmbd1=-10.0)
    return alpha, sigma


def test_ddo_denoise_step_shape_and_finite():
    b, n, c = 4, 128, 1
    z_t = torch.randn(b, n, c)
    eps_hat = torch.randn(b, n, c)
    t = torch.full((b,), 0.75)

    z_prev, mu, sigma2 = ddo_denoise_step_vp(
        z_t=z_t,
        eps_hat=eps_hat,
        t=t,
        num_steps=40,
        alpha_sigma_fn=_alpha_sigma,
        stochastic=True,
    )
    assert z_prev.shape == (b, n, c)
    assert mu.shape == (b, n, c)
    assert sigma2.shape == (b, 1, 1)
    assert torch.isfinite(z_prev).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma2).all()


def test_ddo_denoise_step_deterministic_path():
    b, n, c = 2, 16, 1
    z_t = torch.randn(b, n, c)
    eps_hat = torch.randn(b, n, c)
    t = torch.full((b,), 0.5)

    z_prev, mu, _ = ddo_denoise_step_vp(
        z_t=z_t,
        eps_hat=eps_hat,
        t=t,
        num_steps=20,
        alpha_sigma_fn=_alpha_sigma,
        stochastic=False,
    )
    assert torch.allclose(z_prev, mu)


def test_ddo_denoise_step_with_explicit_t_prev_is_finite():
    b, n, c = 3, 32, 1
    z_t = torch.randn(b, n, c)
    eps_hat = torch.randn(b, n, c)
    t = torch.full((b,), 0.6)
    t_prev = torch.full((b,), 0.55)

    z_prev, mu, sigma2 = ddo_denoise_step_vp(
        z_t=z_t,
        eps_hat=eps_hat,
        t=t,
        t_prev=t_prev,
        num_steps=40,
        alpha_sigma_fn=_alpha_sigma,
        stochastic=True,
    )
    assert z_prev.shape == (b, n, c)
    assert mu.shape == (b, n, c)
    assert sigma2.shape == (b, 1, 1)
    assert torch.isfinite(z_prev).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma2).all()
