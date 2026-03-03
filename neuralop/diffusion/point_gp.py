"""Point-native Gaussian process samplers for diffusion noise."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PointRFFGaussianProcessSampler(nn.Module):
    """
    Sample noise on unstructured points with either:
    - independent white noise
    - RFF approximation of an RBF GP kernel
    """

    def __init__(
        self,
        dim: int = 2,
        gp_type: str = "rff_rbf",
        sigma: float = 1.0,
        length_scale: float = 0.05,
        rff_features: int = 256,
        seed: int = 0,
        jitter: float = 1e-6,
    ):
        super().__init__()
        if gp_type not in {"independent", "rff_rbf"}:
            raise ValueError(f"Unsupported gp_type: {gp_type}")
        if rff_features <= 0:
            raise ValueError("rff_features must be >= 1")
        if length_scale <= 0:
            raise ValueError("length_scale must be > 0")

        self.dim = int(dim)
        self.gp_type = gp_type
        self.sigma = float(sigma)
        self.length_scale = float(length_scale)
        self.rff_features = int(rff_features)
        self.jitter = float(jitter)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        w = torch.randn(self.rff_features, self.dim, generator=g) / self.length_scale
        b = 2.0 * math.pi * torch.rand(self.rff_features, generator=g)
        self.register_buffer("rff_w", w)
        self.register_buffer("rff_b", b)

    def _to_coords(self, coords: torch.Tensor, device, dtype) -> torch.Tensor:
        if coords.ndim == 3:
            # [B, N, D] -> use first sample (geometry is shared in this pipeline)
            coords = coords[0]
        if coords.ndim != 2:
            raise ValueError(f"coords must be [N, D] or [B, N, D], got {tuple(coords.shape)}")
        if coords.shape[1] != self.dim:
            raise ValueError(f"Expected coords dim={self.dim}, got {coords.shape[1]}")
        return coords.to(device=device, dtype=dtype)

    def _sample_independent(
        self,
        batch_size: int,
        n_points: int,
        n_channels: int,
        device,
        dtype,
    ) -> torch.Tensor:
        return self.sigma * torch.randn(batch_size, n_points, n_channels, device=device, dtype=dtype)

    def _sample_rff(
        self,
        coords: torch.Tensor,
        batch_size: int,
        n_channels: int,
        device,
        dtype,
    ) -> torch.Tensor:
        # phi: [N, F]
        w = self.rff_w.to(device=device, dtype=dtype)
        b = self.rff_b.to(device=device, dtype=dtype)
        proj = coords @ w.t() + b.unsqueeze(0)
        phi = math.sqrt(2.0 / float(self.rff_features)) * torch.cos(proj)

        # coeff: [B, F, C]
        coeff = torch.randn(batch_size, self.rff_features, n_channels, device=device, dtype=dtype)
        coeff = self.sigma * coeff
        out = torch.einsum("nf,bfc->bnc", phi, coeff)

        if self.jitter > 0.0:
            out = out + self.jitter * torch.randn_like(out)
        return out

    def sample(
        self,
        coords: torch.Tensor,
        batch_size: int,
        n_channels: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if n_channels <= 0:
            raise ValueError("n_channels must be >= 1")

        if device is None:
            device = coords.device
        if dtype is None:
            dtype = coords.dtype if coords.is_floating_point() else torch.float32

        coords_2d = self._to_coords(coords, device=device, dtype=dtype)
        n_points = int(coords_2d.shape[0])

        if self.gp_type == "independent":
            return self._sample_independent(batch_size, n_points, n_channels, device, dtype)
        return self._sample_rff(coords_2d, batch_size, n_channels, device, dtype)
