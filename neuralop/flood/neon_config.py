"""Config namespace for NEON-aligned Stage-2 epistemic FGNO experiments.

This module is intentionally torch-free: it parses and validates the ``neon:``
configuration block from the plan and exposes a typed, immutable
:class:`NEONStage2Config`. Training/evaluation entrypoints consume this config
rather than reaching into raw dictionaries.

The plan's YAML block uses capitalized ensemble keys (``M_train``, ``K_train``,
``M_eval``, ``K_eval``); this loader accepts both those and lowercase aliases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Union

_VALID_FEATURE_SOURCES = frozenset({"decoder_pre_projection", "latent_grid_fno", "combined_mesh"})
_VALID_DEPENDENCIES = frozenset({"za_dependent", "za_independent"})
_VALID_OBJECTIVES = frozenset({"per_epistemic_fcrps", "pooled_fcrps", "l2_mean"})
_VALID_SPATIAL_WEIGHTS = frozenset({"wettable_area", "wet_front", "uniform"})
_VALID_LEAD_WEIGHTS = frozenset({"uniform", "lead_emphasis"})
_VALID_BRANCH_TYPES = frozenset({"projected", "film"})
_VALID_BRANCH_ACTIVATIONS = frozenset({"gelu", "silu", "relu"})
_VALID_EPISTEMIC_RESAMPLE = frozenset({"epoch", "effective_batch"})
_VALID_BOOTSTRAP_DISTRIBUTIONS = frozenset(
    {"tempered_exponential", "exponential", "bernoulli", "none"}
)
_VALID_BOOTSTRAP_NORMALIZE = frozenset({"per_epistemic_batch", "none"})

_PRIOR_SCALE_MIN = 0.05
_PRIOR_SCALE_MAX = 0.20
_AUTO_PRIOR_RE = re.compile(r"^auto_0p(\d{2})_base_rmse$")

#: Mapping from accepted YAML keys to dataclass field names.
_KEY_ALIASES = {
    "M_train": "m_train",
    "K_train": "k_train",
    "M_eval": "m_eval",
    "K_eval": "k_eval",
    "hidden_channels": "train_hidden_channels",
}


class NEONConfigError(ValueError):
    """Raised when a NEON Stage-2 config is malformed or out of range."""


@dataclass(frozen=True)
class NEONStage2Config:
    """Typed, validated NEON Stage-2 configuration.

    Defaults yield the NEON-aligned reference objective: per-epistemic fit
    scoring with bootstrap-indexed views and randomized priors. Flood-specific
    penalties remain available only as opt-in ablation knobs.
    """

    enabled: bool = False
    stage1_checkpoint_dir: Optional[str] = None
    stage2_checkpoint_dir: Optional[str] = None
    feature_source: str = "decoder_pre_projection"
    dependency: str = "za_dependent"
    d_e: int = 16
    m_train: int = 4
    k_train: int = 8
    m_eval: int = 16
    k_eval: int = 50
    prior_scale: Union[str, float] = "auto_0p10_base_rmse"
    alpha: Optional[float] = None
    lead_time_dim: int = 0
    branch_type: str = "projected"
    train_hidden_channels: int = 32
    prior_hidden_channels: int = 5
    branch_layers: int = 2
    branch_activation: str = "gelu"
    concat_index: bool = True
    family_batch_size: int = 1
    effective_batch_size: int = 8
    shuffle_families: bool = True
    epistemic_resample: str = "effective_batch"
    latent_bank_count: int = 4
    reference_member_subsample: Optional[int] = 32
    progress_log_interval_effective_batches: int = 10
    objective: str = "per_epistemic_fcrps"
    reference_term_for_logging: bool = True
    spatial_weights: str = "wettable_area"
    lead_time_weights: str = "uniform"
    bootstrap_enabled: bool = True
    bootstrap_distribution: str = "tempered_exponential"
    bootstrap_temperature: float = 0.5
    bootstrap_normalize: str = "per_epistemic_batch"
    bootstrap_min_weight: float = 0.05
    bootstrap_max_weight: float = 5.0
    bootstrap_seed: int = 0
    cancellation_diagnostics_enabled: bool = True
    cancellation_warn_cosine_below: float = -0.90
    cancellation_warn_cancellation_above: float = 0.80
    lambda_rpf: float = 0.0
    lambda_smooth: float = 0.0
    lambda_time: float = 0.0
    lambda_pos: float = 0.0
    lambda_mag: float = 0.0
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    n_epochs: int = 30

    # ------------------------------------------------------------------
    # Derived accessors
    # ------------------------------------------------------------------

    @property
    def uses_auto_prior_scale(self) -> bool:
        """True iff the prior scale should be auto-calibrated on a batch.

        An explicitly provided ``alpha`` overrides auto-calibration (the
        operator pinned a concrete prior scale), so this returns False in that
        case regardless of ``prior_scale``.
        """
        if self.alpha is not None:
            return False
        return isinstance(self.prior_scale, str) and self.prior_scale.startswith("auto_")

    @property
    def prior_scale_fraction(self) -> float:
        """Return the target ``Std_ze[alpha E^P] / RMSE_base`` fraction.

        Parses ``auto_0pNN_base_rmse`` -> ``0.NN`` or returns a float
        ``prior_scale`` directly. Does not itself range-check (``validate``
        does); callers that need the number pre-validation get the raw value.
        """
        return _parse_prior_scale_fraction(self.prior_scale)

    def to_loss_weights_dict(self) -> dict[str, float]:
        """Return a dict consumable as ``NEONStage2LossWeights(**...)``.

        Kept as a plain dict so this module stays torch-free; the training
        code adapts it to the torch-side dataclass.
        """
        return {
            "rpf": float(self.lambda_rpf),
            "smooth": float(self.lambda_smooth),
            "time": float(self.lambda_time),
            "pos": float(self.lambda_pos),
            "mag": float(self.lambda_mag),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> "NEONStage2Config":
        if self.feature_source not in _VALID_FEATURE_SOURCES:
            raise NEONConfigError(
                f"feature_source must be one of {sorted(_VALID_FEATURE_SOURCES)}, "
                f"got {self.feature_source!r}."
            )
        if self.dependency not in _VALID_DEPENDENCIES:
            raise NEONConfigError(
                f"dependency must be one of {sorted(_VALID_DEPENDENCIES)}, got {self.dependency!r}."
            )
        if self.objective not in _VALID_OBJECTIVES:
            raise NEONConfigError(
                f"objective must be one of {sorted(_VALID_OBJECTIVES)}, got {self.objective!r}."
            )
        if self.spatial_weights not in _VALID_SPATIAL_WEIGHTS:
            raise NEONConfigError(
                f"spatial_weights must be one of {sorted(_VALID_SPATIAL_WEIGHTS)}, "
                f"got {self.spatial_weights!r}."
            )
        if self.lead_time_weights not in _VALID_LEAD_WEIGHTS:
            raise NEONConfigError(
                f"lead_time_weights must be one of {sorted(_VALID_LEAD_WEIGHTS)}, "
                f"got {self.lead_time_weights!r}."
            )
        if self.branch_type not in _VALID_BRANCH_TYPES:
            raise NEONConfigError(
                f"branch_type must be one of {sorted(_VALID_BRANCH_TYPES)}, "
                f"got {self.branch_type!r}."
            )
        if self.branch_activation not in _VALID_BRANCH_ACTIVATIONS:
            raise NEONConfigError(
                f"branch_activation must be one of {sorted(_VALID_BRANCH_ACTIVATIONS)}, "
                f"got {self.branch_activation!r}."
            )
        if self.epistemic_resample not in _VALID_EPISTEMIC_RESAMPLE:
            raise NEONConfigError(
                f"epistemic_resample must be one of {sorted(_VALID_EPISTEMIC_RESAMPLE)}, "
                f"got {self.epistemic_resample!r}."
            )
        if self.bootstrap_distribution not in _VALID_BOOTSTRAP_DISTRIBUTIONS:
            raise NEONConfigError(
                "bootstrap_distribution must be one of "
                f"{sorted(_VALID_BOOTSTRAP_DISTRIBUTIONS)}, got {self.bootstrap_distribution!r}."
            )
        if self.bootstrap_normalize not in _VALID_BOOTSTRAP_NORMALIZE:
            raise NEONConfigError(
                f"bootstrap_normalize must be one of {sorted(_VALID_BOOTSTRAP_NORMALIZE)}, "
                f"got {self.bootstrap_normalize!r}."
            )
        # Positive-integer dimensions.
        if int(self.d_e) < 1:
            raise NEONConfigError(f"d_e must be >= 1, got {self.d_e}.")
        for name in (
            "train_hidden_channels",
            "prior_hidden_channels",
            "branch_layers",
            "family_batch_size",
            "effective_batch_size",
            "latent_bank_count",
            "progress_log_interval_effective_batches",
        ):
            if int(getattr(self, name)) < 1:
                raise NEONConfigError(f"{name} must be >= 1, got {getattr(self, name)}.")
        if int(self.effective_batch_size) < int(self.family_batch_size):
            raise NEONConfigError(
                "effective_batch_size must be >= family_batch_size, got "
                f"{self.effective_batch_size} < {self.family_batch_size}."
            )
        if self.reference_member_subsample is not None and int(self.reference_member_subsample) < 1:
            raise NEONConfigError(
                "reference_member_subsample must be >= 1 when set, got "
                f"{self.reference_member_subsample}."
            )
        for name in ("m_train", "m_eval"):
            if int(getattr(self, name)) < 1:
                raise NEONConfigError(f"{name} must be >= 1, got {getattr(self, name)}.")
        # Fair CRPS needs >= 2 aleatory samples on both train and eval.
        for name in ("k_train", "k_eval"):
            if int(getattr(self, name)) < 2:
                raise NEONConfigError(
                    f"{name} must be >= 2 (fair CRPS needs >= 2 aleatory samples), "
                    f"got {getattr(self, name)}."
                )
        if int(self.n_epochs) < 1:
            raise NEONConfigError(f"n_epochs must be >= 1, got {self.n_epochs}.")
        if int(self.lead_time_dim) < 0:
            raise NEONConfigError(f"lead_time_dim must be >= 0, got {self.lead_time_dim}.")
        if not (0.0 <= float(self.bootstrap_temperature) <= 1.0):
            raise NEONConfigError(
                f"bootstrap_temperature must be in [0, 1], got {self.bootstrap_temperature}."
            )
        if float(self.bootstrap_min_weight) <= 0.0:
            raise NEONConfigError(
                f"bootstrap_min_weight must be > 0, got {self.bootstrap_min_weight}."
            )
        if float(self.bootstrap_max_weight) < float(self.bootstrap_min_weight):
            raise NEONConfigError(
                "bootstrap_max_weight must be >= bootstrap_min_weight, got "
                f"{self.bootstrap_max_weight} < {self.bootstrap_min_weight}."
            )
        # Non-negative regularization weights.
        for name in ("lambda_rpf", "lambda_smooth", "lambda_time", "lambda_pos", "lambda_mag", "weight_decay"):
            if float(getattr(self, name)) < 0.0:
                raise NEONConfigError(f"{name} must be >= 0, got {getattr(self, name)}.")
        if float(self.learning_rate) <= 0.0:
            raise NEONConfigError(f"learning_rate must be > 0, got {self.learning_rate}.")
        # Prior-scale range (only meaningful when alpha is not explicitly set).
        if self.alpha is None:
            frac = _parse_prior_scale_fraction(self.prior_scale)
            if not (_PRIOR_SCALE_MIN - 1e-9 <= frac <= _PRIOR_SCALE_MAX + 1e-9):
                raise NEONConfigError(
                    f"prior_scale fraction must be in [{_PRIOR_SCALE_MIN}, {_PRIOR_SCALE_MAX}], "
                    f"got {frac} (from {self.prior_scale!r})."
                )
        elif float(self.alpha) < 0.0:
            raise NEONConfigError(f"alpha must be >= 0 when set, got {self.alpha}.")
        return self

    def to_bootstrap_config_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.bootstrap_enabled),
            "distribution": str(self.bootstrap_distribution),
            "temperature": float(self.bootstrap_temperature),
            "normalize": str(self.bootstrap_normalize),
            "min_weight": float(self.bootstrap_min_weight),
            "max_weight": float(self.bootstrap_max_weight),
            "seed": int(self.bootstrap_seed),
        }

    def to_cancellation_diagnostics_config_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.cancellation_diagnostics_enabled),
            "warn_cosine_below": float(self.cancellation_warn_cosine_below),
            "warn_cancellation_above": float(self.cancellation_warn_cancellation_above),
        }


def _parse_prior_scale_fraction(prior_scale: Union[str, float]) -> float:
    if isinstance(prior_scale, (int, float)) and not isinstance(prior_scale, bool):
        return float(prior_scale)
    if isinstance(prior_scale, str):
        match = _AUTO_PRIOR_RE.match(prior_scale.strip())
        if match:
            return int(match.group(1)) / 100.0
        # Bare "auto" defaults to the plan's 0.10 base-RMSE target.
        if prior_scale.strip() == "auto":
            return 0.10
    raise NEONConfigError(
        "prior_scale must be a float or an 'auto_0pNN_base_rmse' string, "
        f"got {prior_scale!r}."
    )


def load_neon_config(mapping: Mapping[str, Any]) -> NEONStage2Config:
    """Build and validate a :class:`NEONStage2Config` from a mapping.

    Accepts either the ``neon:`` block directly or a wrapper mapping that
    contains a top-level ``"neon"`` key. Unknown keys are ignored for
    forward-compatibility. Capitalized ``M_train``/``K_train``/``M_eval``/
    ``K_eval`` keys are mapped to their lowercase fields.
    """
    if mapping is None:
        return NEONStage2Config().validate()
    block = mapping.get("neon", mapping) if hasattr(mapping, "get") else mapping
    if not hasattr(block, "items"):
        raise NEONConfigError(f"neon config must be a mapping, got {type(block).__name__}.")

    field_names = {f.name for f in fields(NEONStage2Config)}
    kwargs: dict[str, Any] = {}
    for raw_key, value in block.items():
        if raw_key == "bootstrap" and hasattr(value, "items"):
            for nested_key, nested_value in value.items():
                mapped = f"bootstrap_{nested_key}"
                if mapped in field_names:
                    kwargs[mapped] = nested_value
            continue
        if raw_key == "cancellation_diagnostics" and hasattr(value, "items"):
            for nested_key, nested_value in value.items():
                mapped = f"cancellation_{nested_key}"
                if mapped == "cancellation_warn_cosine_below":
                    kwargs[mapped] = nested_value
                elif mapped == "cancellation_warn_cancellation_above":
                    kwargs[mapped] = nested_value
                elif mapped == "cancellation_enabled":
                    kwargs["cancellation_diagnostics_enabled"] = nested_value
            continue
        key = _KEY_ALIASES.get(raw_key, raw_key)
        if key in field_names:
            kwargs[key] = value
        # Unknown keys are silently ignored (forward-compat).
    return NEONStage2Config(**kwargs).validate()
