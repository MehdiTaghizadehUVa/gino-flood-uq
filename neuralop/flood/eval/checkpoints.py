"""Checkpoint discovery and model loading for flood evaluation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from neuralop import get_model
from neuralop.flood.eval.runtime import clone_model_config_for_get_model, preferred_eval_checkpoint_name
from neuralop.models.base_model import BaseModel
from neuralop.training.training_state import load_training_state
from neuralop.flood.eval.runtime import CHECKPOINT_BEST, CHECKPOINT_FILES, CHECKPOINT_LAST

def _resolve_checkpoint_in_dir(save_dir: Path, preferred_alias: str = CHECKPOINT_LAST) -> Tuple[Path, str]:
    """Return (dir, alias) for a single checkpoint directory."""
    aliases = [preferred_alias]
    aliases.extend(alias for alias in (CHECKPOINT_LAST, CHECKPOINT_BEST) if alias not in aliases)
    for alias in aliases:
        if alias == CHECKPOINT_BEST and (save_dir / "best_model_state_dict.pt").exists():
            return save_dir, CHECKPOINT_BEST
        if alias == CHECKPOINT_LAST and (save_dir / "model_state_dict.pt").exists():
            return save_dir, CHECKPOINT_LAST
    found = [p.name for p in save_dir.iterdir()] if save_dir.exists() else []
    raise FileNotFoundError(
        f"No checkpoint in {save_dir}. Expected one of {CHECKPOINT_FILES}. Found: {found}"
    )


def _discover_checkpoint_runs(checkpoint_path: Path, preferred_alias: str = CHECKPOINT_LAST) -> List[Tuple[Path, str, str]]:
    """
    Discover checkpoint runs.

    Returns
    -------
    list of (checkpoint_dir, alias, label)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    runs: List[Tuple[Path, str, str]] = []
    try:
        run_dir, alias = _resolve_checkpoint_in_dir(checkpoint_path, preferred_alias=preferred_alias)
        runs.append((run_dir, alias, checkpoint_path.name or "model0"))
        return runs
    except FileNotFoundError:
        pass

    child_dirs = sorted([p for p in checkpoint_path.iterdir() if p.is_dir()])
    for child in child_dirs:
        try:
            run_dir, alias = _resolve_checkpoint_in_dir(child, preferred_alias=preferred_alias)
            runs.append((run_dir, alias, child.name))
        except FileNotFoundError:
            continue

    if not runs:
        found = [p.name for p in checkpoint_path.iterdir()]
        raise FileNotFoundError(
            "No valid checkpoints found in "
            f"{checkpoint_path}. Expected checkpoint files in this directory or immediate subdirectories. "
            f"Found entries: {found}"
        )
    return runs


def _checkpoint_metadata_candidates(run_dir: Path, alias: str) -> List[Path]:
    """Candidate metadata files storing the exact model init kwargs used at training."""
    candidates = [
        run_dir / f"{alias}_metadata.pkl",
        run_dir / "model_metadata.pkl",
        run_dir / "best_model_metadata.pkl",
    ]
    seen: set = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _instantiate_model_from_metadata(metadata: Dict[str, Any]) -> Any:
    """
    Build model directly from checkpoint metadata.

    This reproduces the exact init kwargs used when the checkpoint was trained.
    """
    arch_name = str(metadata.get("_name", "")).strip().lower()
    if not arch_name:
        raise KeyError("Missing '_name' in checkpoint metadata.")
    model_cls = BaseModel._models.get(arch_name)
    if model_cls is None:
        raise KeyError(
            f"Unknown model name '{arch_name}' in checkpoint metadata. "
            f"Known models: {list(BaseModel._models.keys())}"
        )
    init_kwargs = dict(metadata)
    init_args = init_kwargs.pop("args", ())
    init_kwargs.pop("_version", None)
    init_kwargs.pop("_name", None)
    if not isinstance(init_args, (list, tuple)):
        init_args = (init_args,)
    return model_cls(*init_args, **init_kwargs)


def _build_model_for_run(
    config: Any,
    run_dir: Path,
    alias: str,
    label: str,
    logger: logging.Logger,
) -> Any:
    """
    Build model for a checkpoint run.

    Priority:
      1) checkpoint metadata (exact training init kwargs)
      2) deep-copied config fallback (avoids in-place config mutation in get_model)
    """
    for meta_path in _checkpoint_metadata_candidates(run_dir, alias):
        if not meta_path.exists():
            continue
        try:
            metadata = torch.load(meta_path, map_location="cpu")
            if isinstance(metadata, dict):
                model = _instantiate_model_from_metadata(metadata)
                logger.info(
                    "Model '%s': initialized from checkpoint metadata %s",
                    label,
                    meta_path,
                )
                return model
            logger.warning(
                "Model '%s': metadata file %s is not a dict (type=%s); falling back to config.",
                label,
                meta_path,
                type(metadata).__name__,
            )
        except Exception as exc:
            logger.warning(
                "Model '%s': failed loading metadata %s (%s); falling back to config.",
                label,
                meta_path,
                exc,
            )

    model_cfg = clone_model_config_for_get_model(config)
    logger.info(
        "Model '%s': metadata not used; initializing from evaluation config snapshot.",
        label,
    )
    return get_model(model_cfg)


def _preferred_checkpoint_alias(config: Any) -> str:
    """Resolve the default checkpoint alias for evaluation."""
    return preferred_eval_checkpoint_name(config)


def _load_models_from_runs(
    config: Any,
    device: torch.device,
    checkpoint_runs: List[Tuple[Path, str, str]],
    logger: logging.Logger,
) -> List[Any]:
    """Load all models from discovered checkpoints."""
    models: List[Any] = []
    for run_dir, alias, label in checkpoint_runs:
        model = _build_model_for_run(config, run_dir, alias, label, logger)
        load_training_state(save_dir=run_dir, save_name=alias, model=model)
        model = model.to(device).eval()
        models.append(model)
        logger.info("Loaded model '%s' from %s (%s)", label, run_dir, alias)
    return models


def _get_cli_arg_value(flag: str) -> Optional[str]:
    """Return CLI value for '--flag value' or '--flag=value', without consuming argv."""
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
