"""DDO-style VP cosine schedule helpers for function-space diffusion."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def low_discrepancy_rand(batch_size: int, device=None, dtype=None) -> torch.Tensor:
    """Low-discrepancy samples in [0, 1)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    if dtype is None:
        dtype = torch.float32
    u0 = torch.rand(1, device=device, dtype=dtype)
    grid = torch.linspace(0, batch_size - 1, batch_size, device=device, dtype=dtype)
    return (u0 + grid / float(batch_size)) % 1.0


def sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x)


def lmbd_cosine(t: torch.Tensor) -> torch.Tensor:
    return -2.0 * torch.log(torch.tan(0.5 * math.pi * t))


def inv_lmbd_cosine(lmbd: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.atan(torch.exp(-0.5 * lmbd)) / math.pi


def pdf_cosine(lmbd: torch.Tensor) -> torch.Tensor:
    return sech(0.5 * lmbd) / (2.0 * math.pi)


def dlmbd_dt_cosine(t: torch.Tensor) -> torch.Tensor:
    return -math.pi / (torch.tan(0.5 * math.pi * t) * torch.cos(0.5 * math.pi * t) ** 2)


def _get_t0_t1(lmbd0: float, lmbd1: float) -> Tuple[float, float]:
    t0 = float(inv_lmbd_cosine(torch.tensor(lmbd0)).item())
    t1 = float(inv_lmbd_cosine(torch.tensor(lmbd1)).item())
    return t0, t1


def _truncated_lmbd(t: torch.Tensor, lmbd0: float, lmbd1: float) -> torch.Tensor:
    t0, t1 = _get_t0_t1(lmbd0, lmbd1)
    return lmbd_cosine(t0 + (t1 - t0) * t)


def _truncated_inv_lmbd(lmbd: torch.Tensor, lmbd0: float, lmbd1: float) -> torch.Tensor:
    t0, t1 = _get_t0_t1(lmbd0, lmbd1)
    return (inv_lmbd_cosine(lmbd) - t0) / (t1 - t0)


def _truncated_pdf(lmbd: torch.Tensor, lmbd0: float, lmbd1: float) -> torch.Tensor:
    t0, t1 = _get_t0_t1(lmbd0, lmbd1)
    mask = (lmbd >= lmbd1).to(lmbd.dtype) * (lmbd <= lmbd0).to(lmbd.dtype)
    return mask * pdf_cosine(lmbd) / max(t1 - t0, 1e-12)


def _truncated_dlmbd_dt(t: torch.Tensor, lmbd0: float, lmbd1: float) -> torch.Tensor:
    t0, t1 = _get_t0_t1(lmbd0, lmbd1)
    return dlmbd_dt_cosine(t0 + (t1 - t0) * t) * (t1 - t0)


def normalize_lmbd(lmbd: torch.Tensor, lmbd0: float, lmbd1: float) -> torch.Tensor:
    if not (lmbd0 > lmbd1):
        raise ValueError("Expected lmbd0 > lmbd1")
    return (lmbd - lmbd1) / (lmbd0 - lmbd1)


def vp_alpha_sigma_from_lmbd(lmbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.sqrt(torch.sigmoid(lmbd))
    sigma = torch.sqrt(torch.sigmoid(-lmbd))
    return alpha, sigma


def vp_f_g_from_lmbd(lmbd: torch.Tensor, lmbd0: float, lmbd1: float) -> Tuple[torch.Tensor, torch.Tensor]:
    t = _truncated_inv_lmbd(lmbd, lmbd0=lmbd0, lmbd1=lmbd1)
    dlmbd_dt = _truncated_dlmbd_dt(t, lmbd0=lmbd0, lmbd1=lmbd1)
    sig_neg = torch.sigmoid(-lmbd)
    f = 0.5 * sig_neg * dlmbd_dt
    g2 = -sig_neg * dlmbd_dt
    g = torch.sqrt(torch.clamp(g2, min=0.0))
    return f, g


def get_vp_cosine_params(
    t: torch.Tensor,
    lmbd0: float = 10.0,
    lmbd1: float = -10.0,
    eps_t: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (lmbd, lmbd_normalized, pdf, alpha, sigma) for truncated VP-cosine schedule.
    """
    t = t.reshape(-1)
    t = torch.clamp(t, min=eps_t, max=1.0 - eps_t)
    lmbd = _truncated_lmbd(t, lmbd0=lmbd0, lmbd1=lmbd1)
    lmbd_norm = normalize_lmbd(lmbd, lmbd0=lmbd0, lmbd1=lmbd1)
    pdf = _truncated_pdf(lmbd, lmbd0=lmbd0, lmbd1=lmbd1)
    alpha, sigma = vp_alpha_sigma_from_lmbd(lmbd)
    return lmbd, lmbd_norm, pdf, alpha, sigma


def get_weight(lmbd: torch.Tensor, weight_method: str = "shifted_sigmoid_2") -> torch.Tensor:
    if weight_method == "shifted_sigmoid_1":
        k = 1.0
    elif weight_method == "shifted_sigmoid_2":
        k = 2.0
    elif weight_method == "shifted_sigmoid_3":
        k = 3.0
    else:
        raise ValueError(f"Unsupported weight method: {weight_method}")
    return torch.sigmoid(-lmbd + k)


def sample_timesteps(
    batch_size: int,
    sampler: str,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample timesteps in (0, 1] with configurable sampler."""
    if sampler == "uniform":
        t = torch.rand(batch_size, device=device, dtype=dtype)
    elif sampler == "low_discrepancy":
        t = low_discrepancy_rand(batch_size=batch_size, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported timestep sampler: {sampler}")
    return torch.clamp(t, min=1e-6, max=1.0 - 1e-6)
