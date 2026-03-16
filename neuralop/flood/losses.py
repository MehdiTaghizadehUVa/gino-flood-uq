"""Flood-specific masked loss and diagnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuralop.flood.data.structural_dry import broadcast_wettable_mask, dry_mask_to_wettable_mask
from neuralop.losses.data_losses import LpLoss
from neuralop.losses.probabilistic_losses import CRPSLoss, GaussianNLLLoss, split_gaussian_packed


def _normalize_policy(policy: str) -> str:
    return str(policy).strip().lower()


def _extract_structural_dry_mask(
    structural_dry_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor | None:
    if structural_dry_mask is not None:
        return structural_dry_mask
    return kwargs.get("structural_dry_mask")


def _wettable_weights(
    ref: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> torch.Tensor | None:
    dry_mask = _extract_structural_dry_mask(structural_dry_mask=structural_dry_mask, **kwargs)
    if dry_mask is None:
        return None
    wettable = dry_mask_to_wettable_mask(dry_mask).to(device=ref.device)
    return broadcast_wettable_mask(wettable, ref, dtype=dtype or ref.dtype)


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = _wettable_weights(target, structural_dry_mask=structural_dry_mask)
    if weights is None:
        return torch.sqrt(torch.mean((pred - target) ** 2))
    sq_err = (pred - target).pow(2) * weights
    denom = weights.sum().clamp_min(1e-12)
    return torch.sqrt(sq_err.sum() / denom)


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = _wettable_weights(target, structural_dry_mask=structural_dry_mask)
    if weights is None:
        return torch.mean(torch.abs(pred - target))
    abs_err = torch.abs(pred - target) * weights
    denom = weights.sum().clamp_min(1e-12)
    return abs_err.sum() / denom


def dry_falsewet_rate(
    pred: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    dry_mask = torch.as_tensor(structural_dry_mask, dtype=torch.bool, device=pred.device)
    if dry_mask.ndim == 1:
        dry_mask = dry_mask.unsqueeze(0).expand(pred.shape[0], -1)
    if dry_mask.ndim == 2 and pred.ndim == 3:
        dry_mask = dry_mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])
    elif dry_mask.ndim == 2 and pred.ndim == 2:
        dry_mask = dry_mask.expand(pred.shape[0], pred.shape[1])
    elif dry_mask.ndim != pred.ndim:
        raise ValueError(
            f"dry_falsewet_rate cannot broadcast dry mask {tuple(dry_mask.shape)} to {tuple(pred.shape)}."
        )
    if dry_mask.sum() <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    falsewet = (pred > float(threshold)) & dry_mask
    return falsewet.to(dtype=pred.dtype).sum() / dry_mask.to(dtype=pred.dtype).sum().clamp_min(1.0)


def dry_pred_std_mean(
    pred_std: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor,
) -> torch.Tensor:
    dry_mask = torch.as_tensor(structural_dry_mask, dtype=torch.bool, device=pred_std.device)
    if dry_mask.ndim == 1:
        dry_mask = dry_mask.unsqueeze(0).expand(pred_std.shape[0], -1)
    if dry_mask.ndim == 2 and pred_std.ndim == 3:
        dry_mask = dry_mask.unsqueeze(-1).expand(pred_std.shape[0], pred_std.shape[1], pred_std.shape[2])
    elif dry_mask.ndim != pred_std.ndim:
        raise ValueError(
            f"dry_pred_std_mean cannot broadcast dry mask {tuple(dry_mask.shape)} to {tuple(pred_std.shape)}."
        )
    if dry_mask.sum() <= 0:
        return torch.tensor(0.0, device=pred_std.device, dtype=pred_std.dtype)
    return pred_std[dry_mask].mean()


@dataclass
class FloodDryBackgroundRMSE:
    reduction: str = "mean"
    expects_samples: bool = False
    expects_packed: bool = False

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        dry_mask = _extract_structural_dry_mask(**kwargs)
        if dry_mask is None:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        pred = y_pred[..., : y.shape[-1]]
        dry_weights = broadcast_wettable_mask(
            torch.as_tensor(dry_mask, dtype=torch.bool, device=y_pred.device),
            y,
            dtype=y.dtype,
        )
        denom = dry_weights.sum().clamp_min(1.0)
        return torch.sqrt((((pred - y) ** 2) * dry_weights).sum() / denom)


@dataclass
class FloodDryBackgroundMAE:
    reduction: str = "mean"
    expects_samples: bool = False
    expects_packed: bool = False

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        dry_mask = _extract_structural_dry_mask(**kwargs)
        if dry_mask is None:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        pred = y_pred[..., : y.shape[-1]]
        dry_weights = broadcast_wettable_mask(
            torch.as_tensor(dry_mask, dtype=torch.bool, device=y_pred.device),
            y,
            dtype=y.dtype,
        )
        denom = dry_weights.sum().clamp_min(1.0)
        return (torch.abs(pred - y) * dry_weights).sum() / denom


@dataclass
class FloodDryBackgroundFalseWetRate:
    threshold: float
    reduction: str = "mean"
    expects_samples: bool = False
    expects_packed: bool = False

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del y
        dry_mask = _extract_structural_dry_mask(**kwargs)
        if dry_mask is None:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        pred = y_pred[..., :1]
        return dry_falsewet_rate(
            pred,
            structural_dry_mask=dry_mask,
            threshold=self.threshold,
        )


@dataclass
class FloodGaussianDryPredStdMean:
    n_channels: int
    min_logvar: float
    max_logvar: float
    reduction: str = "mean"
    expects_samples: bool = False
    expects_packed: bool = True

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del y
        dry_mask = _extract_structural_dry_mask(**kwargs)
        if dry_mask is None:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        pred_std = gaussian_pred_std(
            y_pred,
            n_channels=self.n_channels,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )
        return dry_pred_std_mean(pred_std, structural_dry_mask=dry_mask)


@dataclass
class FloodEnsembleDryPredStdMean:
    reduction: str = "mean"
    expects_samples: bool = True
    expects_packed: bool = False

    def __call__(self, pred_samples: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del y
        dry_mask = _extract_structural_dry_mask(**kwargs)
        if dry_mask is None:
            return torch.tensor(0.0, device=pred_samples.device, dtype=pred_samples.dtype)
        pred_std = pred_samples.std(dim=0)
        return dry_pred_std_mean(pred_std, structural_dry_mask=dry_mask)


@dataclass
class FloodMaskedRelLpLoss:
    policy: str = "legacy_full_domain"
    base_loss: LpLoss | None = None
    eps: float = 1e-12
    reduction: str = "sum"
    expects_samples: bool = False
    expects_packed: bool = False

    def __post_init__(self):
        self.policy = _normalize_policy(self.policy)
        self.base_loss = self.base_loss or LpLoss(d=2, p=2, reduction=self.reduction)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.policy != "masked_primary":
            return self.base_loss(y_pred, y)
        weights = _wettable_weights(y, **kwargs)
        if weights is None:
            return self.base_loss(y_pred, y)
        diff = ((y_pred - y) * weights).reshape(y.shape[0], -1)
        y_masked = (y * weights).reshape(y.shape[0], -1)
        diff_norm = torch.norm(diff, p=2, dim=1)
        y_norm = torch.norm(y_masked, p=2, dim=1).clamp_min(self.eps)
        rel = diff_norm / y_norm
        if self.reduction == "mean":
            return rel.mean()
        return rel.sum()


@dataclass
class FloodMaskedAbsLpLoss:
    policy: str = "legacy_full_domain"
    reduction: str = "sum"
    expects_samples: bool = False
    expects_packed: bool = False

    def __post_init__(self):
        self.policy = _normalize_policy(self.policy)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.policy != "masked_primary":
            base = LpLoss(d=2, p=2, reduction=self.reduction)
            return base.abs(y_pred, y)
        weights = _wettable_weights(y, **kwargs)
        if weights is None:
            base = LpLoss(d=2, p=2, reduction=self.reduction)
            return base.abs(y_pred, y)
        diff_sq = ((y_pred - y) ** 2) * weights
        per_sample = torch.sqrt(
            diff_sq.reshape(y.shape[0], -1).sum(dim=1)
            / weights.reshape(y.shape[0], -1).sum(dim=1).clamp_min(1e-12)
        )
        if self.reduction == "mean":
            return per_sample.mean()
        return per_sample.sum()


@dataclass
class FloodMaskedCRPSLoss:
    policy: str = "legacy_full_domain"
    base_loss: CRPSLoss | None = None
    expects_samples: bool = True
    expects_packed: bool = False

    def __post_init__(self):
        self.policy = _normalize_policy(self.policy)
        self.base_loss = self.base_loss or CRPSLoss(n_samples=2, reduction="mean")
        self.reduction = getattr(self.base_loss, "reduction", "mean")

    def __call__(self, pred_samples: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.policy != "masked_primary":
            return self.base_loss(pred_samples, y, **kwargs)
        weights = _wettable_weights(y, **kwargs)
        if weights is None:
            return self.base_loss(pred_samples, y, **kwargs)
        return self.base_loss(pred_samples, y, spatial_weights=weights, **kwargs)


@dataclass
class FloodMaskedGaussianNLLLoss:
    policy: str = "legacy_full_domain"
    base_loss: GaussianNLLLoss | None = None
    expects_samples: bool = False
    expects_packed: bool = True

    def __post_init__(self):
        self.policy = _normalize_policy(self.policy)
        self.base_loss = self.base_loss or GaussianNLLLoss(reduction="mean")
        self.reduction = getattr(self.base_loss, "reduction", "mean")

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.policy != "masked_primary":
            return self.base_loss(y_pred, y, **kwargs)
        weights = _wettable_weights(y, **kwargs)
        if weights is None:
            return self.base_loss(y_pred, y, **kwargs)
        return self.base_loss(y_pred, y, spatial_weights=weights, **kwargs)


def gaussian_pred_std(
    packed_pred: torch.Tensor,
    *,
    n_channels: int,
    min_logvar: float,
    max_logvar: float,
) -> torch.Tensor:
    _, logvar = split_gaussian_packed(packed_pred, n_channels=n_channels)
    logvar = torch.clamp(logvar, min=float(min_logvar), max=float(max_logvar))
    return torch.exp(0.5 * logvar)
