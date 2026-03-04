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
    time_injection: str = "channel"  # channel|adain
    time_embedding_dim: int = 32
    time_embedding_hidden_dim: int = 128
    time_embedding_scale: float = 10000.0


class ConditionalDDOForecaster(nn.Module):
    """
    DDO-style conditional diffusion wrapper around an existing denoiser backbone.

    The denoiser predicts epsilon on wd (depth-only v1) conditioned on context
    features assembled from static + boundary history + dynamic history.

    Context layout contract: [static | flattened boundary history | flattened
    dynamic history], all on native mesh points with shape [B, N, C_context].
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
        conditioning: Optional[ConditioningConfig] = None,
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
        self.conditioning = conditioning or ConditioningConfig()
        self.sampler_method = sampler_method
        self.sampler_num_steps = int(sampler_num_steps)
        self.sampler_s_min = float(sampler_s_min)
        self.sampler_return_mean_last = bool(sampler_return_mean_last)
        self._validate_conditioning_config()
        if self.conditioning.time_injection == "adain":
            self.time_mlp: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.conditioning.time_embedding_dim, self.conditioning.time_embedding_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.conditioning.time_embedding_hidden_dim, self.conditioning.time_embedding_dim),
            )
        else:
            self.time_mlp = None
        if not (0.0 <= self.sampler_s_min < 1.0):
            raise ValueError(f"sampler_s_min must be in [0, 1), got {self.sampler_s_min}")

    def _validate_conditioning_config(self) -> None:
        cfg = self.conditioning
        if not bool(cfg.add_noisy_target):
            raise ValueError("Diffusion baseline requires conditioning.add_noisy_target=true.")
        if cfg.time_injection not in {"channel", "adain"}:
            raise ValueError(
                f"conditioning.time_injection must be one of {{'channel', 'adain'}}, got {cfg.time_injection!r}"
            )
        if cfg.time_feature_type not in {"sincos", "raw"}:
            raise ValueError(
                f"conditioning.time_feature_type must be one of {{'sincos', 'raw'}}, got {cfg.time_feature_type!r}"
            )
        if int(cfg.time_embedding_dim) <= 0:
            raise ValueError(
                f"conditioning.time_embedding_dim must be > 0, got {cfg.time_embedding_dim}"
            )
        if int(cfg.time_embedding_hidden_dim) <= 0:
            raise ValueError(
                "conditioning.time_embedding_hidden_dim must be > 0, "
                f"got {cfg.time_embedding_hidden_dim}"
            )
        if float(cfg.time_embedding_scale) <= 0:
            raise ValueError(
                f"conditioning.time_embedding_scale must be > 0, got {cfg.time_embedding_scale}"
            )

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
                "time_injection": self.conditioning.time_injection,
                "time_embedding_dim": self.conditioning.time_embedding_dim,
                "time_embedding_hidden_dim": self.conditioning.time_embedding_hidden_dim,
                "time_embedding_scale": self.conditioning.time_embedding_scale,
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

    @staticmethod
    def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, scale: float) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got shape={tuple(t.shape)}")
        dim = int(dim)
        scale = float(scale)
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")

        if dim == 1:
            return t.reshape(-1, 1)

        half = dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(scale, device=t.device, dtype=t.dtype))
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = (2.0 * np.pi * t.reshape(-1, 1)) * freq.reshape(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device, dtype=t.dtype)], dim=1)
        return emb

    def _build_time_adain(self, t: torch.Tensor) -> torch.Tensor:
        if self.conditioning.time_injection != "adain":
            raise RuntimeError("_build_time_adain called when conditioning.time_injection != 'adain'.")
        if self.time_mlp is None:
            raise RuntimeError("time_mlp is not initialized for adain conditioning.")
        emb = self._sinusoidal_timestep_embedding(
            t=t,
            dim=self.conditioning.time_embedding_dim,
            scale=self.conditioning.time_embedding_scale,
        )
        return self.time_mlp(emb)

    def _time_features(self, t: torch.Tensor, n_points: int, dtype: torch.dtype) -> torch.Tensor:
        if self.conditioning.time_injection != "channel":
            return torch.empty(t.shape[0], n_points, 0, device=t.device, dtype=dtype)
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

    def _reverse_time_grid(self, n_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Reverse schedule grid from t=1.0 down to sampler_s_min.

        Returns n_steps+1 values so each denoise step can consume (t_cur, t_prev).
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        return torch.linspace(
            1.0,
            self.sampler_s_min,
            steps=n_steps + 1,
            device=device,
            dtype=dtype,
        )

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
        """
        Compute weighted DSM epsilon loss.

        Expected sample keys:
        - context: [B, N, C_context], layout documented in class docstring
        - target: [B, N, 1] (depth-only v1)
        - input_geom/latent_queries/output_queries: geometry/query tensors
        """
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
        ada_in = self._build_time_adain(t) if self.conditioning.time_injection == "adain" else None
        eps_hat = self._predict_eps(
            x_in=x_in,
            input_geom=input_geom,
            latent_queries=latent_queries,
            output_queries=output_queries,
            ada_in=ada_in,
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
        initial_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Sample one-step next-state field from conditional reverse diffusion.

        Notes
        -----
        - If ``initial_latent`` is provided, it is used as z_T and must be [B, N, 1].
        - If ``initial_latent`` is not provided:
          - stochastic=True: z_T is sampled from the configured GP prior.
          - stochastic=False: z_T is initialized to zeros for deterministic output.
        """
        bsz = context.shape[0]
        n_steps = int(self.sampler_num_steps if num_steps is None else num_steps)
        if n_steps <= 0:
            raise ValueError("num_steps must be >= 1")

        n_points = int(context.shape[1])
        if initial_latent is not None:
            z_t = initial_latent.to(device=context.device, dtype=context.dtype)
            if z_t.ndim != 3 or z_t.shape != (bsz, n_points, 1):
                raise ValueError(
                    "initial_latent must have shape [B, N, 1] matching context "
                    f"({bsz}, {n_points}, 1), got {tuple(z_t.shape)}"
                )
        elif stochastic:
            z_t = self.gp_sampler.sample(
                coords=input_geom,
                batch_size=bsz,
                n_channels=1,
                device=context.device,
                dtype=context.dtype,
            )
        else:
            # Deterministic path for validation/ablation reproducibility.
            z_t = torch.zeros(bsz, n_points, 1, device=context.device, dtype=context.dtype)

        trace = [] if return_trace else None
        mu_last = None
        t_grid = self._reverse_time_grid(n_steps=n_steps, device=context.device, dtype=context.dtype)
        for step_idx in range(n_steps):
            t_cur = t_grid[step_idx].expand(bsz)
            t_prev = t_grid[step_idx + 1].expand(bsz)
            x_in = self._build_denoiser_input(context=context, z_t=z_t, t=t_cur)
            step_ada_in = self._build_time_adain(t_cur) if self.conditioning.time_injection == "adain" else None
            if ada_in is not None:
                step_ada_in = ada_in
            eps_hat = self._predict_eps(
                x_in=x_in,
                input_geom=input_geom,
                latent_queries=latent_queries,
                output_queries=output_queries,
                ada_in=step_ada_in,
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
                t=t_cur,
                num_steps=n_steps,
                alpha_sigma_fn=lambda tt: self._alpha_sigma(tt)[3:5],
                t_prev=t_prev,
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
