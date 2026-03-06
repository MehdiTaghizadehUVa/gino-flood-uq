"""Shared utilities for diffusion train/eval scripts."""

from __future__ import annotations

import inspect
import json
import logging
import numbers
import threading
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
    # Mapping-like configs (including OmegaConf DictConfig) should prefer key
    # access because getattr() on missing keys may raise KeyError.
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if isinstance(obj, MutableMapping):
        return obj.get(key, default)
    try:
        return getattr(obj, key)
    except (AttributeError, KeyError, TypeError):
        pass
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


def _tensor_only_state_dict(state_dict: Dict[str, Any], name: str) -> Dict[str, torch.Tensor]:
    """Drop non-tensor entries (e.g. PyTorch internal metadata) from state_dict."""
    tensor_only_state: Dict[str, torch.Tensor] = {
        k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)
    }
    if not tensor_only_state:
        raise ValueError(f"{name} did not contain any tensor entries.")
    return tensor_only_state


def save_checkpoint_sidecars(
    checkpoint_path: Path,
    *,
    denoiser_state_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    extra_state_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Path, Path]:
    """Save safe sidecar files: tensor-only weights payload and JSON metadata."""

    weights_payload: Dict[str, Dict[str, torch.Tensor]] = {
        "denoiser_state_dict": _tensor_only_state_dict(
            denoiser_state_dict, name="denoiser_state_dict"
        )
    }
    for key, state_dict in (extra_state_dicts or {}).items():
        weights_payload[str(key)] = _tensor_only_state_dict(state_dict, name=str(key))

    weights_path, metadata_path = checkpoint_sidecar_paths(checkpoint_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights_payload, weights_path)
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
            for key, value in weights.items():
                merged[key] = value
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


def shutdown_dataloader_workers(loader: Any, logger: Optional[logging.Logger] = None, name: str = "loader") -> None:
    """
    Best-effort shutdown for DataLoader worker processes.

    Uses PyTorch private iterator hooks to prevent teardown hangs in long-running
    jobs with multiprocessing workers.
    """
    if loader is None:
        return
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            setattr(loader, "_iterator", None)
    except Exception as exc:  # pragma: no cover - defensive cleanup
        if logger is not None:
            logger.warning("DataLoader cleanup failed for %s: %s", name, exc)


def safe_wandb_finish(
    run: Any,
    *,
    logger: Optional[logging.Logger] = None,
    timeout_seconds: float = 120.0,
) -> None:
    """
    Finish a W&B run with timeout protection.

    W&B finalization can occasionally hang on cluster teardown/network. This
    wrapper avoids stale Slurm jobs by bounding shutdown time.
    """
    if run is None:
        return
    timeout_seconds = max(1.0, float(timeout_seconds))
    finish_error: Dict[str, Exception] = {}

    def _finish() -> None:
        try:
            finish_fn = getattr(run, "finish", None)
            if callable(finish_fn):
                finish_fn()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            finish_error["exc"] = exc

    t = threading.Thread(target=_finish, name="wandb-finish", daemon=True)
    t.start()
    t.join(timeout_seconds)

    if t.is_alive():
        if logger is not None:
            logger.warning(
                "W&B finish exceeded %.1fs timeout; continuing shutdown to avoid stale job.",
                timeout_seconds,
            )
        return

    if finish_error and logger is not None:
        logger.warning("W&B finish raised an exception during shutdown: %s", finish_error["exc"])
