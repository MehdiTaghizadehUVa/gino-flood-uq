"""Flood feature assembly and data processing helpers."""

from __future__ import annotations

import torch

from neuralop.data.transforms.data_processors import DataProcessor
from neuralop.flood.data.structural_dry import apply_structural_dry_zero_mask
from neuralop.losses.probabilistic_losses import split_gaussian_packed

class FloodGINODataProcessor(DataProcessor):
    """
    Preprocesses samples for GINO and optionally inverse-transforms outputs for eval.
    Training: pred and y are both in normalized space (same as dataset target).
    Eval (when inverse_test=True): pred and y are inverse-transformed so metrics are in physical space.
    """
    def __init__(
        self,
        device="cuda",
        target_norm=None,
        inverse_test=True,
        output_distribution: str = "deterministic",
    ):
        super().__init__()
        self.device = device
        self.model = None
        self.target_norm = target_norm
        self.inverse_test = inverse_test
        self.output_distribution = str(output_distribution).strip().lower()
        if self.output_distribution not in {"deterministic", "gaussian"}:
            raise ValueError(
                "output_distribution must be one of {'deterministic', 'gaussian'}, "
                f"got {output_distribution!r}."
            )

    def preprocess(self, sample: dict) -> dict:
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)
        # Preserve optional keys (e.g. ada_in for FGN) — we only add/overwrite below

        # dynamic => (B, num_cells, n_history * n_target_channels)
        dyn_ = sample["dynamic"]
        if dyn_.dim() == 3:
            dyn_ = dyn_.unsqueeze(0)
        dyn_ = dyn_.permute(0, 2, 1, 3)
        B, N, H, D = dyn_.shape
        dyn_ = dyn_.reshape(B, N, H * D)

        # boundary => (B, num_cells, n_history * bc_dim)
        bc_ = sample["boundary"]
        if bc_.dim() == 3:
            bc_ = bc_.unsqueeze(0)
        bc_ = bc_.permute(0, 2, 1, 3)
        B2, N2, H2, C2 = bc_.shape
        bc_ = bc_.reshape(B2, N2, H2 * C2)

        # static => (B, num_cells, static_dim)
        st_ = sample["static"]
        if st_.dim() == 2:
            st_ = st_.unsqueeze(0)

        x_ = torch.cat([st_, bc_, dyn_], dim=2)

        geom_ = sample["geometry"]
        if geom_.dim() == 2:
            geom_ = geom_.unsqueeze(0)
        # GINO expects geometry with leading dim 1 (shared across batch). Use first sample when B > 1.
        if geom_.shape[0] > 1:
            geom_ = geom_[0:1]

        y_ = sample.get("target", None)
        if y_ is not None and y_.dim() == 2:
            y_ = y_.unsqueeze(0)

        q_ = sample["query_points"]
        if q_.dim() == 3:
            q_ = q_.unsqueeze(0)
        # GINO expects latent_queries / output_queries with leading dim 1 (shared). Use first when B > 1.
        if q_.shape[0] > 1:
            q_ = q_[0:1]

        sample["input_geom"] = geom_
        sample["latent_queries"] = q_
        sample["output_queries"] = geom_.clone()
        sample["x"] = x_
        sample["y"] = y_
        return sample

    @staticmethod
    def _match_stat_ndim(stat: torch.Tensor, val_ndim: int) -> torch.Tensor:
        out = stat
        while out.ndim > val_ndim and out.shape[0] == 1:
            out = out.squeeze(0)
        while out.ndim < val_ndim:
            out = out.unsqueeze(0)
        return out

    def postprocess(self, out: torch.Tensor, sample: dict):
        if (not self.training) and self.inverse_test and (self.target_norm is not None):
            structural_dry_mask = sample.get("structural_dry_mask")
            if self.output_distribution == "gaussian":
                y_ref = sample.get("y")
                n_channels = y_ref.shape[-1] if y_ref is not None else (out.shape[-1] // 2)
                mu, logvar = split_gaussian_packed(out, n_channels=n_channels)
                mu = self.target_norm.inverse_transform(mu)
                mu = apply_structural_dry_zero_mask(
                    mu,
                    structural_dry_mask=structural_dry_mask,
                )

                std_stat = self._match_stat_ndim(self.target_norm.std, logvar.ndim)
                if std_stat.device != logvar.device:
                    std_stat = std_stat.to(logvar.device)
                eps = float(getattr(self.target_norm, "eps", 1e-7))
                logvar = logvar + 2.0 * torch.log(std_stat + eps)
                out = torch.cat([mu, logvar], dim=-1)
            else:
                out = self.target_norm.inverse_transform(out)
                out = apply_structural_dry_zero_mask(
                    out,
                    structural_dry_mask=structural_dry_mask,
                )
            if sample["y"] is not None:
                sample["y"] = self.target_norm.inverse_transform(sample["y"])
        return out, sample

    def to(self, device: str):
        self.device = device
        if self.target_norm is not None:
            self.target_norm.to(device)
        return self

    def wrap(self, model: torch.nn.Module):
        self.model = model

    def forward(self, sample: dict):
        sample = self.preprocess(sample)
        if self.model is None:
            raise RuntimeError("No model attached. Call wrap(model).")
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample


def get_flood_crps_weights(
    static: torch.Tensor,
    y: torch.Tensor,
    wet_threshold: float = 0.01,
    wet_smooth_scale: float = 0.02,
    dry_weight_alpha: float = 0.1,
    static_normalizer=None,
) -> torch.Tensor:
    """
    Compute per-(batch, cell, channel) weights for flood CRPS: area weighting plus
    soft wet/dry masking so the loss approximates a physical integral and avoids
    ill-defined velocities in dry cells.

    - Area: from static column index 1 (Cells Surface Area). If static_normalizer
      is provided, area is denormalized then used as weight_i = area_i / total_area
      (so weights sum to 1 per batch item). If no normalizer, raw static[:,:,1] is
      used and still normalized by total_area when possible.
    - Soft wetness m = sigmoid((depth - wet_threshold) / wet_smooth_scale) from
      ground-truth depth y[..., 0].
    - Depth weight: (area_ratio) * (alpha + (1 - alpha) * m).
    - Velocity weights (u, v): (area_ratio) * m.

    Parameters
    ----------
    static : torch.Tensor
        (B, n_cells, n_static) or (n_cells, n_static). Area must be at index 1
        (HDF "Cells Surface Area", same order as dataset: elevation, area, ...).
    y : torch.Tensor
        (B, n_cells, 3) targets [depth (h), vx (u), vy (v)].
    wet_threshold : float
        Depth threshold (m) below which cell is considered dry.
    wet_smooth_scale : float
        Smoothing scale (m) for sigmoid transition at wet/dry front.
    dry_weight_alpha : float
        Relative weight for dry-cell depth errors (e.g. 0.1 => dry counts 10x less than wet).
    static_normalizer : optional
        UnitGaussianNormalizer fit on static (dim [0,1]). If provided, area column
        (index 1) is inverse-transformed to physical units before area/total_area.

    Returns
    -------
    spatial_weights : torch.Tensor
        (B, n_cells, 3) with [w_depth, w_vel, w_vel], same device/dtype as y.
    """
    if static.dim() == 2:
        static = static.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    B, n_cells, _ = y.shape
    if static.shape[0] != B:
        static = static.expand(B, -1, -1)
    if static.shape[1] < n_cells:
        raise ValueError("get_flood_crps_weights: static n_cells < y n_cells")
    static = static[:, :n_cells, :]
    if static.shape[2] < 2:
        # No area column: fallback to uniform weight (1/n_cells per cell)
        area_ratio = torch.ones(B, n_cells, device=y.device, dtype=y.dtype) / n_cells
    else:
        area_raw = static[:, :, 1].clone().to(device=y.device, dtype=y.dtype)
        if static_normalizer is not None and hasattr(static_normalizer, "mean") and hasattr(static_normalizer, "std"):
            # Denormalize area column (index 1). Normalizer mean/std shape (1, 1, n_static)
            m = static_normalizer.mean.to(y.device)
            s = static_normalizer.std.to(y.device)
            if m.dim() >= 3 and m.shape[2] > 1:
                area_mean = m[0, 0, 1]
                area_std = s[0, 0, 1] + getattr(static_normalizer, "eps", 1e-7)
                area_phys = area_raw * area_std + area_mean
            else:
                area_phys = area_raw
            area_phys = torch.clamp(area_phys, min=0.0)
        else:
            area_phys = torch.clamp(area_raw, min=0.0)
        total_area = area_phys.sum(dim=1, keepdim=True).clamp(min=1e-12)
        area_ratio = area_phys / total_area
    depth = y[:, :, 0]
    m = torch.sigmoid((depth - wet_threshold) / max(wet_smooth_scale, 1e-8))
    w_depth = area_ratio * (dry_weight_alpha + (1.0 - dry_weight_alpha) * m)
    w_vel = area_ratio * m
    spatial_weights = torch.stack([w_depth, w_vel, w_vel], dim=-1)
    return spatial_weights


def _get_area_ratio_from_static(
    static: torch.Tensor,
    n_cells: int,
    device: torch.device,
    dtype: torch.dtype,
    static_normalizer=None,
) -> torch.Tensor:
    """
    Return area_ratio (B, n_cells) with sum(area_ratio, dim=1) = 1.
    Used by pooled functionals (e.g. hazard proxy). Static column index 1 = area.
    """
    if static.dim() == 2:
        static = static.unsqueeze(0)
    B = static.shape[0]
    if static.shape[1] < n_cells:
        raise ValueError("_get_area_ratio_from_static: static n_cells < n_cells")
    static = static[:, :n_cells, :]
    if static.shape[2] < 2:
        return torch.ones(B, n_cells, device=device, dtype=dtype) / n_cells
    area_raw = static[:, :, 1].clone().to(device=device, dtype=dtype)
    if static_normalizer is not None and hasattr(static_normalizer, "mean") and hasattr(static_normalizer, "std"):
        m = static_normalizer.mean.to(device)
        s = static_normalizer.std.to(device)
        if m.dim() >= 3 and m.shape[2] > 1:
            area_mean = m[0, 0, 1]
            area_std = s[0, 0, 1] + getattr(static_normalizer, "eps", 1e-7)
            area_phys = area_raw * area_std + area_mean
        else:
            area_phys = area_raw
        area_phys = torch.clamp(area_phys, min=0.0)
    else:
        area_phys = torch.clamp(area_raw, min=0.0)
    total_area = area_phys.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return area_phys / total_area


def compute_hazard_proxy_pooled(
    static: torch.Tensor,
    field: torch.Tensor,
    wet_threshold: float = 0.01,
    wet_smooth_scale: float = 0.02,
    static_normalizer=None,
) -> torch.Tensor:
    """
    Velocity–depth hazard proxy pooled over the mesh (scalar per batch, or per ensemble and batch).

    H = sum_i (area_ratio_i) * m_i * h_i * (u_i^2 + v_i^2),
    with m_i = sigmoid((h_i - tau) / eps) (soft wet mask). Couples (h, u, v) and penalizes
    nonphysical fast water in near-dry cells. Differentiable.

    Parameters
    ----------
    static : torch.Tensor
        (B, n_cells, n_static) or (n_cells, n_static). Area at column index 1.
    field : torch.Tensor
        (B, n_cells, 3) or (N, B, n_cells, 3) with channels [depth (h), vx (u), vy (v)].
    wet_threshold : float
        Depth threshold (m) for soft wet mask.
    wet_smooth_scale : float
        Sigmoid smoothing scale (m).
    static_normalizer : optional
        UnitGaussianNormalizer for static; if provided, area is denormalized before area_ratio.

    Returns
    -------
    pooled : torch.Tensor
        (B,) or (N, B) scalar hazard proxy per batch item (and per ensemble member if field is 4D).
    """
    has_ensemble = field.dim() == 4
    if has_ensemble:
        N, B, n_cells, _ = field.shape
        area_ratio = _get_area_ratio_from_static(
            static, n_cells, field.device, field.dtype, static_normalizer
        )
        h = field[:, :, :, 0]
        u = field[:, :, :, 1]
        v = field[:, :, :, 2]
        m = torch.sigmoid((h - wet_threshold) / max(wet_smooth_scale, 1e-8))
        kinetic = h * (u * u + v * v)
        pooled = (area_ratio.unsqueeze(0) * m * kinetic).sum(dim=2)
        return pooled
    else:
        if field.dim() == 2:
            field = field.unsqueeze(0)
        B, n_cells, _ = field.shape
        area_ratio = _get_area_ratio_from_static(
            static, n_cells, field.device, field.dtype, static_normalizer
        )
        h = field[:, :, 0]
        u = field[:, :, 1]
        v = field[:, :, 2]
        m = torch.sigmoid((h - wet_threshold) / max(wet_smooth_scale, 1e-8))
        kinetic = h * (u * u + v * v)
        pooled = (area_ratio * m * kinetic).sum(dim=1)
        return pooled


def _build_x_from_dynamic_boundary(static: torch.Tensor, boundary: torch.Tensor, dynamic: torch.Tensor):
    """
    Build GINO input x from (static, boundary, dynamic) in the same format as FloodGINODataProcessor.
    static (B, n_cells, s), boundary (B, n_history, n_cells, bc), dynamic (B, n_history, n_cells, 3).
    Returns x (B, n_cells, s + n_history*bc + n_history*3).
    """
    if dynamic.dim() == 3:
        dynamic = dynamic.unsqueeze(0)
    if boundary.dim() == 3:
        boundary = boundary.unsqueeze(0)
    if static.dim() == 2:
        static = static.unsqueeze(0)
    dyn_ = dynamic.permute(0, 2, 1, 3).reshape(dynamic.shape[0], dynamic.shape[2], -1)
    bc_ = boundary.permute(0, 2, 1, 3).reshape(boundary.shape[0], boundary.shape[2], -1)
    x_ = torch.cat([static, bc_, dyn_], dim=2)
    return x_


def _gaussian_mean_from_packed(out: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Extract Gaussian predictive mean from packed [mu, logvar] output."""
    mu, _ = split_gaussian_packed(out, n_channels=n_channels)
    return mu


def _sample_from_packed_gaussian(
    out: torch.Tensor,
    n_channels: int,
    min_logvar: float = -9.0,
    max_logvar: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reparameterized sample from packed [mu, logvar] output.

    Returns (sample, mu, logvar_clamped).
    """
    mu, logvar = split_gaussian_packed(out, n_channels=n_channels)
    logvar = torch.clamp(logvar, min=float(min_logvar), max=float(max_logvar))
    std = torch.exp(0.5 * logvar)
    sample = mu + std * torch.randn_like(mu)
    return sample, mu, logvar


###############################################################################
# 6b) FGN Trainer (two forwards + CRPS per batch)
###############################################################################
