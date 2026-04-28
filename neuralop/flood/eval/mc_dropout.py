"""MC-dropout evaluation helpers for flood operator baselines."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

from neuralop.flood.data.structural_dry import (
    broadcast_wettable_mask,
    dry_mask_to_wettable_mask,
)
from neuralop.flood.losses import (
    FloodEnsembleDryPredStdMean,
    FloodMaskedCRPSLoss,
)
from neuralop.losses.probabilistic_losses import CRPSLoss
from neuralop.training.determinism import deterministic_seed_context, stable_seed_from_parts


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)
_ALLOWED_METHODS = {"none", "", "mc_dropout", "mcdropout", "mc-dropout"}


@dataclass(frozen=True)
class MCDropoutConfig:
    enabled: bool
    samples: int
    seed: int
    dropout_probability: float
    activate_modules: str


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except (AttributeError, KeyError, TypeError):
        pass
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        return default


def _section(config: Any, name: str) -> Any:
    return _cfg_get(config, name, {})


def _normalize_method(value: Any) -> str:
    method = str(value or "none").strip().lower()
    if method not in _ALLOWED_METHODS:
        raise ValueError(
            f"Unsupported uq.method={value!r}. Expected mc_dropout or omitted/none."
        )
    if method in {"mc_dropout", "mcdropout", "mc-dropout"}:
        return "mc_dropout"
    return "none"


def resolve_mc_dropout_config(config: Any) -> MCDropoutConfig:
    """Resolve the optional uq.mc_dropout config block."""
    uq_cfg = _section(config, "uq")
    method = _normalize_method(_cfg_get(uq_cfg, "method", "none"))
    mc_cfg = _cfg_get(uq_cfg, "mc_dropout", {})
    gino_cfg = _section(config, "gino")
    dist_cfg = _section(config, "distributed")

    default_seed = int(_cfg_get(dist_cfg, "seed", 123))
    samples = int(_cfg_get(uq_cfg, "mc_samples", _cfg_get(mc_cfg, "mc_samples", 32)))
    p = float(
        _cfg_get(
            mc_cfg,
            "dropout_probability",
            _cfg_get(gino_cfg, "fno_channel_mlp_dropout", 0.0),
        )
    )
    activate_modules = str(_cfg_get(mc_cfg, "activate_modules", "dropout_only")).strip().lower()
    seed = int(_cfg_get(mc_cfg, "seed", default_seed))
    return MCDropoutConfig(
        enabled=(method == "mc_dropout"),
        samples=samples,
        seed=seed,
        dropout_probability=p,
        activate_modules=activate_modules,
    )


def validate_mc_dropout_config(config: Any, *, require_training_loss_l2: bool = False) -> MCDropoutConfig:
    """Validate MC-dropout settings and reject mixed UQ mechanisms."""
    mc = resolve_mc_dropout_config(config)
    if not mc.enabled:
        return mc

    if mc.samples < 2:
        raise ValueError(f"uq.mc_samples must be >= 2 for MC dropout diagnostics, got {mc.samples}.")
    if not (0.0 < mc.dropout_probability < 1.0):
        raise ValueError(
            "uq.mc_dropout.dropout_probability must be in (0, 1) for MC dropout, "
            f"got {mc.dropout_probability}."
        )
    if mc.activate_modules != "dropout_only":
        raise ValueError(
            "Only uq.mc_dropout.activate_modules=dropout_only is supported. "
            f"Got {mc.activate_modules!r}."
        )

    gino_cfg = _section(config, "gino")
    opt_cfg = _section(config, "opt")
    use_fgn = bool(_cfg_get(gino_cfg, "use_fgn_noise", False))
    out_dist = str(_cfg_get(gino_cfg, "output_distribution", "deterministic")).strip().lower()
    train_loss = str(_cfg_get(opt_cfg, "training_loss", "l2")).strip().lower()
    gino_dropout = float(_cfg_get(gino_cfg, "fno_channel_mlp_dropout", 0.0))

    if use_fgn:
        raise ValueError("uq.method=mc_dropout is incompatible with gino.use_fgn_noise=true.")
    if out_dist == "gaussian":
        raise ValueError("uq.method=mc_dropout is incompatible with gino.output_distribution=gaussian.")
    if abs(gino_dropout - mc.dropout_probability) > 1e-12:
        raise ValueError(
            "MC dropout requires gino.fno_channel_mlp_dropout to match "
            "uq.mc_dropout.dropout_probability. "
            f"Got gino={gino_dropout}, uq={mc.dropout_probability}."
        )
    if require_training_loss_l2 and train_loss != "l2":
        raise ValueError(
            "MC-dropout baseline training must use opt.training_loss=l2. "
            f"Got {train_loss!r}."
        )
    return mc


def enable_mc_dropout_only(model: nn.Module) -> int:
    """Put the model in eval mode, then reactivate only dropout modules."""
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
            count += 1
    return count


def mc_dropout_seed_context(enabled: bool, seed: Optional[int], *parts: Any):
    """Return a deterministic RNG context for one MC forward, or a no-op context."""
    if not enabled:
        return nullcontext()
    base_seed = 123 if seed is None else int(seed)
    return deterministic_seed_context(stable_seed_from_parts("mc_dropout", base_seed, *parts))


def _clone_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}


def _masked_tensor_mean(
    values: torch.Tensor,
    ref: torch.Tensor,
    *,
    structural_dry_mask: Optional[torch.Tensor],
    policy: str,
) -> torch.Tensor:
    if policy != "masked_primary" or structural_dry_mask is None:
        return values.mean()
    wettable = dry_mask_to_wettable_mask(structural_dry_mask).to(device=values.device)
    weights = broadcast_wettable_mask(wettable, ref, dtype=values.dtype)
    if weights.shape != values.shape:
        weights = torch.broadcast_to(weights, values.shape)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def _interval_coverage_and_width(
    pred_samples: torch.Tensor,
    target: torch.Tensor,
    *,
    channel_idx: int,
    alpha: float,
    structural_dry_mask: Optional[torch.Tensor],
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lo = 0.5 * (1.0 - float(alpha))
    q_hi = 1.0 - q_lo
    samples_ch = pred_samples[..., channel_idx]
    target_ch = target[..., channel_idx]
    lo = torch.quantile(samples_ch, q_lo, dim=0)
    hi = torch.quantile(samples_ch, q_hi, dim=0)
    covered = ((target_ch >= lo) & (target_ch <= hi)).to(dtype=pred_samples.dtype)
    width = hi - lo
    ref = target[..., channel_idx : channel_idx + 1]
    return (
        _masked_tensor_mean(
            covered.unsqueeze(-1),
            ref,
            structural_dry_mask=structural_dry_mask,
            policy=policy,
        ),
        _masked_tensor_mean(
            width.unsqueeze(-1),
            ref,
            structural_dry_mask=structural_dry_mask,
            policy=policy,
        ),
    )


def _accumulate_metric(
    totals: Dict[str, float],
    name: str,
    value: Any,
    *,
    batch_size: int,
    mean_reduction: bool,
) -> None:
    scalar = float(value.item() if hasattr(value, "item") else value)
    totals[name] = totals.get(name, 0.0) + (scalar * batch_size if mean_reduction else scalar)


def evaluate_mc_dropout_one_step(
    *,
    model: nn.Module,
    data_processor: Any,
    data_loader: Any,
    eval_losses: Dict[str, Any],
    config: Any,
    device: torch.device,
    logger: logging.Logger,
    log_prefix: str = "test",
    model_idx: int = 0,
) -> Dict[str, float]:
    """Evaluate one-step MC-dropout metrics on the MC mean and sample stack."""
    mc = validate_mc_dropout_config(config)
    if not mc.enabled:
        raise ValueError("evaluate_mc_dropout_one_step called with uq.method not set to mc_dropout.")

    model = model.to(device)
    dropout_count = enable_mc_dropout_only(model)
    if dropout_count <= 0:
        raise ValueError(
            "MC dropout requested but no torch.nn.Dropout modules are present in the loaded model. "
            "Check gino.fno_channel_mlp_dropout and checkpoint metadata."
        )
    if data_processor is not None:
        data_processor = data_processor.to(device)
        data_processor.eval()

    structural_policy = str(_cfg_get(_section(config, "structural_dry"), "policy", "legacy_full_domain")).strip().lower()
    target_variables = list(_cfg_get(_section(config, "data"), "target_variables", ["wd", "vx", "vy"]))
    wd_idx = target_variables.index("wd") if "wd" in target_variables else None
    crps_metric = FloodMaskedCRPSLoss(
        policy=structural_policy,
        base_loss=CRPSLoss(n_samples=mc.samples, reduction="mean"),
    )
    dry_std_metric = (
        FloodEnsembleDryPredStdMean(channel_idx=wd_idx)
        if wd_idx is not None
        else None
    )
    totals: Dict[str, float] = {f"{log_prefix}_{name}": 0.0 for name in eval_losses.keys()}
    n_samples_total = 0
    mean_pred_std_sum = 0.0

    logger.info(
        "MC-dropout one-step eval: samples=%d seed=%d dropout_modules=%d policy=%s",
        mc.samples,
        mc.seed,
        dropout_count,
        structural_policy,
    )

    with torch.no_grad():
        for batch_idx, raw_sample in enumerate(data_loader):
            pred_members = []
            processed_ref = None
            for mc_idx in range(mc.samples):
                sample = _clone_sample(raw_sample)
                if data_processor is not None:
                    sample = data_processor.preprocess(sample)
                else:
                    sample = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in sample.items()
                    }
                with mc_dropout_seed_context(True, mc.seed, "one_step", model_idx, batch_idx, mc_idx):
                    out = model(**sample)
                if data_processor is not None:
                    out, sample = data_processor.postprocess(out, sample)
                pred_members.append(out)
                if processed_ref is None:
                    processed_ref = sample
            if processed_ref is None:
                continue
            pred_stack = torch.stack(pred_members, dim=0)
            pred_mean = pred_stack.mean(dim=0)
            pred_std = pred_stack.std(dim=0, unbiased=False)
            y = processed_ref["y"]
            batch_size = int(y.shape[0])
            n_samples_total += batch_size
            mean_pred_std_sum += float(pred_std.mean().item()) * batch_size
            loss_kwargs = {
                "y": y,
            }
            if "structural_dry_mask" in processed_ref:
                loss_kwargs["structural_dry_mask"] = processed_ref["structural_dry_mask"]
            if "spatial_weights" in processed_ref:
                loss_kwargs["spatial_weights"] = processed_ref["spatial_weights"]

            for loss_name, loss_fn in eval_losses.items():
                val = loss_fn(pred_mean, **loss_kwargs)
                mean_reduction = getattr(loss_fn, "reduction", None) == "mean"
                _accumulate_metric(
                    totals,
                    f"{log_prefix}_{loss_name}",
                    val,
                    batch_size=batch_size,
                    mean_reduction=mean_reduction,
                )

            crps_val = crps_metric(pred_stack, **loss_kwargs)
            _accumulate_metric(
                totals,
                f"{log_prefix}_mc_crps",
                crps_val,
                batch_size=batch_size,
                mean_reduction=True,
            )
            spread_val = _masked_tensor_mean(
                pred_std,
                y,
                structural_dry_mask=processed_ref.get("structural_dry_mask"),
                policy=structural_policy,
            )
            _accumulate_metric(
                totals,
                f"{log_prefix}_mc_spread_mean",
                spread_val,
                batch_size=batch_size,
                mean_reduction=True,
            )
            if dry_std_metric is not None:
                dry_std_val = dry_std_metric(pred_stack, **loss_kwargs)
                _accumulate_metric(
                    totals,
                    f"{log_prefix}_mc_pred_std_mean_dry_background_wd",
                    dry_std_val,
                    batch_size=batch_size,
                    mean_reduction=True,
                )
                for alpha in (0.50, 0.80, 0.90, 0.95):
                    coverage, width = _interval_coverage_and_width(
                        pred_stack,
                        y,
                        channel_idx=wd_idx,
                        alpha=alpha,
                        structural_dry_mask=processed_ref.get("structural_dry_mask"),
                        policy=structural_policy,
                    )
                    pct = int(round(alpha * 100))
                    _accumulate_metric(
                        totals,
                        f"{log_prefix}_mc_coverage_wd_{pct}",
                        coverage,
                        batch_size=batch_size,
                        mean_reduction=True,
                    )
                    _accumulate_metric(
                        totals,
                        f"{log_prefix}_mc_width_wd_{pct}",
                        width,
                        batch_size=batch_size,
                        mean_reduction=True,
                    )

    if n_samples_total <= 0:
        return {k: 0.0 for k in totals}
    metrics = {k: v / n_samples_total for k, v in totals.items()}
    logger.info(
        "MC-dropout one-step eval finished: samples=%d mean_pred_std=%.6e metrics=%s",
        n_samples_total,
        mean_pred_std_sum / n_samples_total,
        metrics,
    )
    return metrics
