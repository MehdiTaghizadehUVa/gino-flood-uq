"""DDO-style reverse denoising step for VP diffusion (point-native)."""

from __future__ import annotations

from typing import Callable, Tuple

import torch


def _expand_for_field(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = v
    while out.ndim < ref.ndim:
        out = out.unsqueeze(-1)
    return out


def ddo_denoise_step_vp(
    z_t: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    num_steps: int,
    alpha_sigma_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    stochastic: bool = True,
    eps: torch.Tensor | None = None,
    delta: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One reverse denoising step equivalent to DDO's linear VP case (without DCT operators).

    Parameters
    ----------
    z_t : [B, N, C]
    eps_hat : [B, N, C]
    t : [B]
    num_steps : int
    alpha_sigma_fn : callable returning alpha, sigma from t
    stochastic : bool
    eps : optional custom noise [B, N, C]

    Returns
    -------
    z_prev, mu_denoise, sigma2_denoise
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1")
    t = t.reshape(-1)
    t_prev = torch.clamp(t - 1.0 / float(num_steps), min=0.0, max=1.0)

    alpha_s, sigma_s = alpha_sigma_fn(t_prev)
    alpha_t, sigma_t = alpha_sigma_fn(t)

    alpha_s = _expand_for_field(alpha_s, z_t)
    sigma_s = _expand_for_field(sigma_s, z_t)
    alpha_t = _expand_for_field(alpha_t, z_t)
    sigma_t = _expand_for_field(sigma_t, z_t)

    sigma2_s = sigma_s ** 2
    sigma2_t = sigma_t ** 2

    alpha_ts = alpha_t / torch.clamp(alpha_s, min=delta)
    alpha_st = alpha_s / torch.clamp(alpha_t, min=delta)
    sigma2_ts = sigma2_t - alpha_ts ** 2 * sigma2_s
    sigma2_ts = torch.clamp(sigma2_ts, min=delta)

    sigma2_denoise = sigma2_ts * sigma2_s / torch.clamp(sigma2_t, min=delta)

    coeff_term1 = alpha_ts * sigma2_s / torch.clamp(sigma2_t, min=delta)
    coeff_term2 = alpha_st * sigma2_ts / torch.clamp(sigma2_t, min=delta)

    mu_denoise = coeff_term1 * z_t + coeff_term2 * (z_t - sigma_t * eps_hat)

    if not stochastic:
        return mu_denoise, mu_denoise, sigma2_denoise

    if eps is None:
        eps = torch.randn_like(z_t)
    z_prev = mu_denoise + torch.sqrt(torch.clamp(sigma2_denoise, min=delta)) * eps
    return z_prev, mu_denoise, sigma2_denoise
