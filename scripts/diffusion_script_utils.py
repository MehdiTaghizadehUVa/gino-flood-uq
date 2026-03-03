"""Shared utilities for diffusion train/eval scripts."""

from __future__ import annotations

import inspect
import json
import logging
import numbers
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import torch


def to_builtin(obj: Any) -> Any:
    """Recursively convert config-like objects to Python builtin containers."""
    # Preserve scalar-like values before __dict__-based conversion.
    if obj is None or isinstance(obj, (str, bytes, bool)):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return float(obj)
    if isinstance(obj, numbers.Complex):
        return complex(obj)

    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: to_builtin(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        out: Dict[str, Any] = {}
        for k, v in vars(obj).items():
            if callable(v):
                continue
            out[k] = to_builtin(v)
        return out
    return obj


def safe_get(obj: Any, key: str, default: Any) -> Any:
    """Get attribute/dict/index key from mixed config objects safely."""
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except AttributeError:
        pass
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if isinstance(obj, MutableMapping):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError):
        return default


def checkpoint_sidecar_paths(checkpoint_path: Path) -> Tuple[Path, Path]:
    """Return sidecar paths for safe weights and JSON metadata."""
    base = checkpoint_path.with_suffix("")
    return (
        base.with_name(f"{base.name}_weights.pt"),
        base.with_name(f"{base.name}_metadata.json"),
    )


def _supports_weights_only() -> bool:
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


def load_torch_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_legacy_load: bool,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load a torch checkpoint with a safe-first strategy.

    If this torch build supports `weights_only`, it is attempted first. Legacy
    pickle fallback is allowed only when `allow_unsafe_legacy_load=True`.
    """
    cp = Path(checkpoint_path)
    supports_weights_only = _supports_weights_only()
    if supports_weights_only:
        try:
            loaded = torch.load(cp, map_location=map_location, weights_only=True)
            if not isinstance(loaded, dict):
                raise TypeError(f"Expected dict checkpoint, got {type(loaded)}")
            return loaded
        except Exception as exc:
            if not allow_unsafe_legacy_load:
                raise RuntimeError(
                    f"Safe checkpoint load failed for {cp} and unsafe legacy loading "
                    "is disabled. Provide sidecar safe checkpoints or enable "
                    "`checkpoint.allow_unsafe_legacy_load=true` only for trusted files."
                ) from exc
            if logger is not None:
                logger.warning(
                    "Safe checkpoint load failed for %s (%s). Falling back to "
                    "pickle-based load; use only trusted checkpoints.",
                    cp,
                    exc,
                )
            loaded = torch.load(cp, map_location=map_location, weights_only=False)
            if not isinstance(loaded, dict):
                raise TypeError(f"Expected dict checkpoint, got {type(loaded)}")
            return loaded

    if not allow_unsafe_legacy_load:
        raise RuntimeError(
            "This torch build does not support `weights_only`, and "
            "checkpoint.allow_unsafe_legacy_load=false."
        )
    if logger is not None:
        logger.warning(
            "Torch build lacks `weights_only`; loading %s with pickle semantics. "
            "Use only trusted checkpoints.",
            cp,
        )
    loaded = torch.load(cp, map_location=map_location)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(loaded)}")
    return loaded


def save_checkpoint_sidecars(
    checkpoint_path: Path,
    *,
    denoiser_state_dict: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Save safe sidecar files: tensor-only weights and JSON metadata."""

    # BaseModel.state_dict() can include a non-tensor "_metadata" payload.
    # Strip non-tensor entries so the sidecar stays weights_only-loadable.
    tensor_only_state: Dict[str, torch.Tensor] = {
        k: v for k, v in denoiser_state_dict.items() if isinstance(v, torch.Tensor)
    }
    if not tensor_only_state:
        raise ValueError("denoiser_state_dict did not contain any tensor entries.")

    weights_path, metadata_path = checkpoint_sidecar_paths(checkpoint_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"denoiser_state_dict": tensor_only_state}, weights_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return weights_path, metadata_path


def load_checkpoint_bundle(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_legacy_load: bool,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint metadata + weights.

    Preferred path:
    - `<stem>_weights.pt` loaded via safe strategy
    - `<stem>_metadata.json`

    Legacy fallback path:
    - original checkpoint `.pt` via `load_torch_checkpoint(...)`
    """
    cp = Path(checkpoint_path)
    weights_path, metadata_path = checkpoint_sidecar_paths(cp)

    has_weights_sidecar = weights_path.exists()
    has_metadata_sidecar = metadata_path.exists()
    if has_weights_sidecar and has_metadata_sidecar:
        try:
            weights = load_torch_checkpoint(
                weights_path,
                map_location=map_location,
                allow_unsafe_legacy_load=False,
                logger=logger,
            )
            if "denoiser_state_dict" not in weights:
                raise KeyError(f"Missing denoiser_state_dict in sidecar file: {weights_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                raise TypeError(f"Expected metadata dict in {metadata_path}, got {type(metadata)}")
            merged = dict(metadata)
            merged["denoiser_state_dict"] = weights["denoiser_state_dict"]
            return merged
        except Exception as exc:
            if not allow_unsafe_legacy_load:
                raise
            if logger is not None:
                logger.warning(
                    "Sidecar safe-load failed for %s (%s). Falling back to legacy "
                    "checkpoint load with unsafe semantics enabled.",
                    weights_path,
                    exc,
                )

    if has_weights_sidecar != has_metadata_sidecar and logger is not None:
        logger.warning(
            "Incomplete sidecar checkpoint for %s (weights=%s, metadata=%s). "
            "Falling back to legacy checkpoint load.",
            cp,
            has_weights_sidecar,
            has_metadata_sidecar,
        )

    return load_torch_checkpoint(
        cp,
        map_location=map_location,
        allow_unsafe_legacy_load=allow_unsafe_legacy_load,
        logger=logger,
    )
