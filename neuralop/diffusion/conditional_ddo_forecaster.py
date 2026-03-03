"""Conditional DDO-style diffusion forecaster for point-based flood prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ddo_reverse import ddo_denoise_step_vp
from .ddo_schedule import get_vp_cosine_params, get_weight, sample_timesteps
from .point_gp import PointRFFGaussianProcessSampler


@dataclass
class ConditioningConfig:
    add_noisy_target: bool = True
    add_time_features: bool = True
    time_feature_type: str = "sincos"  # sincos|raw


class ConditionalDDOForecaster(nn.Module):
    """
    DDO-style conditional diffusion wrapper around an existing denoiser backbone.

    The denoiser predicts epsilon on wd (depth-only v1) conditioned on context
    features assembled from static + boundary history + dynamic history.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        gp_sampler: PointRFFGaussianProcessSampler,
        parameterization: str = "epsilon",
        timestep_sampler: str = "low_discrepancy",
        lmbd0: float = 10.0,
        lmbd1: float = -10.0,
        weight_method: Optional[str] = "shifted_sigmoid_2",
        conditioning: ConditioningConfig = ConditioningConfig(),
        sampler_method: str = "denoise",
        sampler_num_steps: int = 40,
        sampler_s_min: float = 1e-4,
        sampler_return_mean_last: bool = True,
    ):
        super().__init__()
        if parameterization != "epsilon":
            raise ValueError("Only epsilon parameterization is supported in v1.")
        if sampler_method != "denoise":
            raise ValueError("Only denoise sampler is supported in v1.")

        self.denoiser = denoiser
        self.gp_sampler = gp_sampler
        self.parameterization = parameterization
        self.timestep_sampler = timestep_sampler
        self.lmbd0 = float(lmbd0)
        self.lmbd1 = float(lmbd1)
        self.weight_method = weight_method
        self.conditioning = conditioning
        self.sampler_method = sampler_method
        self.sampler_num_steps = int(sampler_num_steps)
        self.sampler_s_min = float(sampler_s_min)
        self.sampler_return_mean_last = bool(sampler_return_mean_last)

    def diffusion_hparams(self) -> Dict[str, Any]:
        return {
            "parameterization": self.parameterization,
            "timestep_sampler": self.timestep_sampler,
            "schedule": {
                "ns_method": "vp_cosine",
                "lmbd0": self.lmbd0,
                "lmbd1": self.lmbd1,
                "weight_method": self.weight_method,
            },
            "gp": {
                "type": self.gp_sampler.gp_type,
                "sigma": self.gp_sampler.sigma,
                "length_scale": self.gp_sampler.length_scale,
                "rff_features": self.gp_sampler.rff_features,
            },
            "sampler": {
                "method": self.sampler_method,
                "num_steps": self.sampler_num_steps,
                "s_min": self.sampler_s_min,
                "return_mean_last": self.sampler_return_mean_last,
            },
            "conditioning": {
                "add_noisy_target": self.conditioning.add_noisy_target,
                "add_time_features": self.conditioning.add_time_features,
                "time_feature_type": self.conditioning.time_feature_type,
            },
        }

    @staticmethod
    def _expand_scalar(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = v
        while out.ndim < ref.ndim:
            out = out.unsqueeze(-1)
        return out

    def _alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return get_vp_cosine_params(t=t, lmbd0=self.lmbd0, lmbd1=self.lmbd1)

    def _time_features(self, t: torch.Tensor, n_points: int, dtype: torch.dtype) -> torch.Tensor:
        if not self.conditioning.add_time_features:
            return torch.empty(t.shape[0], n_points, 0, device=t.device, dtype=dtype)

        t_col = t.reshape(-1, 1).to(dtype=dtype)
        if self.conditioning.time_feature_type == "raw":
            feat = t_col
        else:
            feat = torch.cat(
                [torch.sin(2.0 * np.pi * t_col), torch.cos(2.0 * np.pi * t_col)],
                dim=1,
            )
        return feat.unsqueeze(1).repeat(1, n_points, 1)

    def _build_denoiser_input(self, context: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        parts = [context]
        if self.conditioning.add_noisy_target:
            parts.append(z_t)
        tf = self._time_features(t=t, n_points=context.shape[1], dtype=context.dtype)
        if tf.shape[-1] > 0:
            parts.append(tf)
        return torch.cat(parts, dim=-1)

    def _predict_eps(
        self,
        x_in: torch.Tensor,
        input_geom: torch.Tensor,
        latent_queries: torch.Tensor,
        output_queries: torch.Tensor,
        ada_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_geom": input_geom,
            "latent_queries": latent_queries,
            "output_queries": output_queries,
            "x": x_in,
        }
        if ada_in is not None:
            kwargs["ada_in"] = ada_in
        out = self.denoiser(**kwargs)
        if out.ndim != 3:
            raise ValueError(f"Denoiser output must be [B, N, C], got {tuple(out.shape)}")
        # Depth-only v1: first channel is epsilon(wd).
        return out[..., :1]

    def training_loss(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted DSM epsilon loss."""
        context = sample["context"]
        target = sample["target"]
        input_geom = sample["input_geom"]
        latent_queries = sample["latent_queries"]
        output_queries = sample["output_queries"]

        if target.ndim != 3:
            raise ValueError(f"target must be [B, N, C], got {tuple(target.shape)}")
        bsz = target.shape[0]
        n_points = target.shape[1]

        t = sample_timesteps(
            batch_size=bsz,
            sampler=self.timestep_sampler,
            device=target.device,
            dtype=target.dtype,
        )
        lmbd, _, pdf, alpha, sigma = self._alpha_sigma(t)
        alpha_e = self._expand_scalar(alpha, target)
        sigma_e = self._expand_scalar(sigma, target)

        eps = self.gp_sampler.sample(
            coords=input_geom,
            batch_size=bsz,
            n_channels=target.shape[-1],
            device=target.device,
            dtype=target.dtype,
        )
        z_t = alpha_e * target + sigma_e * eps
        x_in = self._build_denoiser_input(context=context, z_t=z_t, t=t)
        eps_hat = self._predict_eps(
            x_in=x_in,
            input_geom=input_geom,
            latent_queries=latent_queries,
            output_queries=output_queries,
        )

        # 0.5 * ||eps_hat - eps||^2, averaged over locations/channels then over batch.
        mse = 0.5 * (eps_hat - eps) ** 2
        per_sample = mse.reshape(bsz, -1).mean(dim=1)
        if self.weight_method:
            w = get_weight(lmbd, weight_method=self.weight_method)
            per_sample = per_sample * (w / torch.clamp(pdf, min=1e-12))
        loss = per_sample.mean()

        with torch.no_grad():
            stats = {
                "loss": float(loss.item()),
                "mse_eps": float(mse.mean().item()),
                "t_mean": float(t.mean().item()),
            }
        return loss, stats

    def sample_next(
        self,
        context: torch.Tensor,
        input_geom: torch.Tensor,
        latent_queries: torch.Tensor,
        output_queries: torch.Tensor,
        num_steps: Optional[int] = None,
        stochastic: bool = True,
        return_trace: bool = False,
        ada_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """Sample one-step next-state field from conditional reverse diffusion."""
        bsz = context.shape[0]
        n_steps = int(self.sampler_num_steps if num_steps is None else num_steps)
        if n_steps <= 0:
            raise ValueError("num_steps must be >= 1")

        z_t = self.gp_sampler.sample(
            coords=input_geom,
            batch_size=bsz,
            n_channels=1,
            device=context.device,
            dtype=context.dtype,
        )

        trace = [] if return_trace else None
        mu_last = None
        for step in range(n_steps, 0, -1):
            t = torch.full((bsz,), float(step) / float(n_steps), device=context.device, dtype=context.dtype)
            x_in = self._build_denoiser_input(context=context, z_t=z_t, t=t)
            eps_hat = self._predict_eps(
                x_in=x_in,
                input_geom=input_geom,
                latent_queries=latent_queries,
                output_queries=output_queries,
                ada_in=ada_in,
            )
            if stochastic:
                eps = self.gp_sampler.sample(
                    coords=input_geom,
                    batch_size=bsz,
                    n_channels=1,
                    device=context.device,
                    dtype=context.dtype,
                )
            else:
                eps = torch.zeros_like(z_t)

            z_t, mu_t, _ = ddo_denoise_step_vp(
                z_t=z_t,
                eps_hat=eps_hat,
                t=t,
                num_steps=n_steps,
                alpha_sigma_fn=lambda tt: self._alpha_sigma(tt)[3:5],
                stochastic=stochastic,
                eps=eps,
            )
            mu_last = mu_t
            if trace is not None:
                trace.append(z_t.detach().cpu())

        out = mu_last if self.sampler_return_mean_last and mu_last is not None else z_t
        if trace is not None:
            return out, trace
        return out

    def forward(
        self,
        input_geom: torch.Tensor,
        latent_queries: torch.Tensor,
        output_queries: torch.Tensor,
        x: torch.Tensor,
        ada_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward for evaluator compatibility: `x` is treated as context features.
        Returns one sampled next-state depth field [B, N, 1].
        """
        return self.sample_next(
            context=x,
            input_geom=input_geom,
            latent_queries=latent_queries,
            output_queries=output_queries,
            stochastic=True,
            ada_in=ada_in,
        )
