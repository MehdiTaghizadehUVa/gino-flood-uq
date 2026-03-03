import torch

from neuralop.losses.probabilistic_losses import GaussianNLLLoss, split_gaussian_packed


def _manual_diag_gaussian_nll(mu, logvar, y):
    return 0.5 * (
        logvar
        + (y - mu).pow(2) * torch.exp(-logvar)
        + torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
    )


def test_split_gaussian_packed():
    pred = torch.randn(2, 4, 6)
    mu, logvar = split_gaussian_packed(pred, n_channels=3)
    assert mu.shape == (2, 4, 3)
    assert logvar.shape == (2, 4, 3)


def test_gaussian_nll_matches_manual_formula():
    torch.manual_seed(0)
    B, P, C = 2, 5, 3
    mu = torch.randn(B, P, C)
    logvar = torch.randn(B, P, C)
    y = torch.randn(B, P, C)
    pred = torch.cat([mu, logvar], dim=-1)

    loss_fn = GaussianNLLLoss(
        reduction="mean",
        min_logvar=-20.0,
        max_logvar=20.0,
        logvar_reg_weight=0.0,
    )
    got = loss_fn(pred, y)
    expected = _manual_diag_gaussian_nll(mu, logvar, y).mean()
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_gaussian_nll_clamps_logvar():
    torch.manual_seed(1)
    B, P, C = 1, 6, 2
    mu = torch.randn(B, P, C)
    logvar = torch.full((B, P, C), 100.0)
    y = torch.randn(B, P, C)
    pred = torch.cat([mu, logvar], dim=-1)

    min_lv, max_lv = -2.0, 1.0
    loss_fn = GaussianNLLLoss(
        reduction="mean",
        min_logvar=min_lv,
        max_logvar=max_lv,
        logvar_reg_weight=0.0,
    )
    got = loss_fn(pred, y)
    expected = _manual_diag_gaussian_nll(mu, torch.full_like(logvar, max_lv), y).mean()
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_gaussian_nll_weighted_mean():
    torch.manual_seed(2)
    B, P, C = 2, 4, 2
    mu = torch.randn(B, P, C)
    logvar = torch.randn(B, P, C)
    y = torch.randn(B, P, C)
    pred = torch.cat([mu, logvar], dim=-1)

    channel_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
    spatial_weights = torch.rand(B, P, C, dtype=torch.float32) + 0.1

    loss_fn = GaussianNLLLoss(
        channel_weights=channel_weights,
        reduction="mean",
        min_logvar=-10.0,
        max_logvar=10.0,
        logvar_reg_weight=0.0,
    )
    got = loss_fn(pred, y, spatial_weights=spatial_weights)

    logvar_c = torch.clamp(logvar, min=-10.0, max=10.0)
    nll = _manual_diag_gaussian_nll(mu, logvar_c, y)
    nll = nll * channel_weights.view(1, 1, C)
    expected = (nll * spatial_weights).sum() / spatial_weights.sum()
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
