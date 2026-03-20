"""Shared runtime constants and CLI helpers for flood evaluation."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from neuralop import get_model
from neuralop.models.base_model import BaseModel
from neuralop.training.training_state import load_training_state

CHECKPOINT_BEST = "best_model"
CHECKPOINT_LAST = "model"
CHECKPOINT_FILES = ("best_model_state_dict.pt", "model_state_dict.pt")
DEFAULT_STATIC_FILES = ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]
PUBLICATION_TIMESTEPS = [12, 24, 36, 48, 60, 72]
TRAIN_FRAC = 0.9
MIN_EPS = 1e-9
CBAR_FRAC = 0.046
CBAR_PAD = 0.02
ANIMATION_FPS = 5
ANIMATION_INTERVAL_MS = 200
LEGACY_3CH = ("wd", "vx", "vy")
CSI_THRESHOLDS = (0.05, 0.3)
NORMALIZER_KEYS = ("geometry", "static", "boundary", "dynamic", "target")
DEVICE_REF_KEYS = ("dynamic", "target", "geometry")
CHANNEL_INDEX = {"wd": 0, "vx": 1, "vy": 2}
ROLLOUT_METRICS_NPZ = "rollout_metrics_data.npz"
ROLLOUT_SUMMARY_PNG = "rollout_metrics_summary.png"
ROLLOUT_METRICS_HYDRO_NPZ = "rollout_metrics_per_hydrograph.npz"
ROLLOUT_SUMMARY_HYDRO_PNG = "rollout_metrics_per_hydrograph.png"
ROLLOUT_SUMMARY_HYDRO_FULL_PNG = "rollout_metrics_per_hydrograph_full.png"
UQ_OVERALL_JSON = "uq_overall_metrics.json"
UQ_RELIABILITY_PNG = "uq_reliability_wd_exceedance.png"
UQ_PIT_RANK_PNG = "uq_pit_rank_histograms.png"
UQ_SPREAD_SKILL_PNG = "uq_spread_skill_scatter.png"
UQ_INTERVAL_COVERAGE_PNG = "uq_interval_coverage.png"
UQ_BOXPLOT_PNG = "uq_metric_boxplots.png"
UQ_VAR_DECOMP_PNG = "uq_variance_decomposition_wd.png"
DEFAULT_EVAL_LOG = "eval_post_training.log"
HYDROGRAPH_SIM_PATTERN = re.compile(r"^(.+)_sim(\d+)$")
UQ_EXCEEDANCE_THRESHOLD = 0.05
ROLLOUT_INIT_MEAN_HISTORY = "mean_history"
ROLLOUT_INIT_MEMBER_HISTORY = "member_history"
ROLLOUT_INIT_MODES = (
    ROLLOUT_INIT_MEAN_HISTORY,
    ROLLOUT_INIT_MEMBER_HISTORY,
)
_REPO_ROOT = Path(__file__).resolve().parents[3]

def _opt(config: Any, section: Optional[str], key: str, default: Any) -> Any:
    """Get config.section.key with default. Use section=None for top-level config keys."""
    def _safe_get(obj: Any, name: str, dflt: Any) -> Any:
        try:
            return getattr(obj, name)
        except (AttributeError, KeyError, TypeError):
            pass
        if isinstance(obj, dict):
            return obj.get(name, dflt)
        try:
            return obj[name]
        except Exception:
            return dflt

    if section is None:
        return _safe_get(config, key, default)
    obj = _safe_get(config, section, None)
    if obj is None:
        return default
    return _safe_get(obj, key, default)


def _opt_float(config: Any, section: Optional[str], key: str, default: float) -> float:
    """Safe float getter with fallback for None/invalid config values."""
    val = _opt(config, section, key, default)
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def parse_hydrograph_run_id(run_id: str) -> Tuple[str, Optional[int]]:
    """Parse run_id convention: {hydrograph_id}_sim{sim_id}."""
    rid = str(run_id).strip()
    m = HYDROGRAPH_SIM_PATTERN.match(rid)
    if m:
        return m.group(1), int(m.group(2))
    return rid, None


def group_run_ids_by_hydrograph(run_ids: List[str]) -> Dict[str, List[str]]:
    """Group run IDs by hydrograph ID inferred from *_simN naming."""
    groups: Dict[str, List[str]] = {}
    for rid in run_ids:
        hydro_id, _ = parse_hydrograph_run_id(rid)
        groups.setdefault(hydro_id, []).append(rid)
    for hydro_id in groups:
        groups[hydro_id] = sorted(groups[hydro_id])
    return groups


def _config_to_builtin(obj: Any) -> Any:
    """Convert configmypy/namespace-style config objects into plain builtins."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): _config_to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_config_to_builtin(v) for v in obj]
    if hasattr(obj, "items"):
        try:
            return {str(k): _config_to_builtin(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            str(k): _config_to_builtin(v)
            for k, v in vars(obj).items()
            if not str(k).startswith("_")
        }
    try:
        return {str(k): _config_to_builtin(obj[k]) for k in obj}
    except Exception:
        return obj


def clone_model_config_for_get_model(config: Any) -> Dict[str, Any]:
    """
    Return an isolated plain-dict config snapshot for get_model().

    Some config providers used in this repo are dict-like but do not implement
    ``copy.deepcopy`` safely because ``getattr(..., "__deepcopy__")`` raises
    ``KeyError`` instead of ``AttributeError``. Converting to builtins avoids
    that runtime-specific behavior while still giving get_model() a mutable,
    isolated config tree to consume.
    """
    builtins = _config_to_builtin(config)
    if not isinstance(builtins, dict):
        raise TypeError(
            "Expected dict-like config for model construction, got "
            f"{type(builtins).__name__}."
        )
    return builtins


def normalize_rollout_init_mode(mode: Any) -> str:
    """Normalize grouped-hydrograph rollout initialization mode."""
    normalized = str(mode or ROLLOUT_INIT_MEAN_HISTORY).strip().lower()
    if normalized not in ROLLOUT_INIT_MODES:
        raise ValueError(
            "rollout.init_mode must be one of "
            f"{ROLLOUT_INIT_MODES}, got {mode!r}."
        )
    return normalized


def _select_rollout_reference_member_indices(
    *,
    n_ref: int,
    n_members: int,
) -> List[int]:
    """Choose reference members deterministically for member-history initialization."""
    if n_ref < 1:
        raise ValueError(f"Expected at least one reference trajectory, got n_ref={n_ref}.")
    if n_members < 1:
        raise ValueError(f"Expected at least one rollout member, got n_members={n_members}.")
    if n_members == 1:
        return [0]
    if n_members <= n_ref:
        indices = np.round(np.linspace(0, n_ref - 1, num=n_members)).astype(int)
        return indices.clip(0, n_ref - 1).tolist()
    full_cycles, remainder = divmod(n_members, n_ref)
    return list(range(n_ref)) * full_cycles + list(range(remainder))


def build_rollout_initial_histories(
    dynamic_ref: torch.Tensor,
    *,
    skip_before_timestep: int,
    start_pred_t: int,
    n_members: int,
    rollout_init_mode: str,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Build initial autoregressive histories for grouped-hydrograph rollout evaluation."""
    histories = dynamic_ref[:, skip_before_timestep:start_pred_t]
    if histories.ndim != 4:
        raise ValueError(
            "Grouped rollout expects dynamic_ref with shape [n_ref, history, n_cells, n_channels], "
            f"got shape {tuple(dynamic_ref.shape)}."
        )
    if histories.shape[1] <= 0:
        raise ValueError(
            "Grouped rollout history window is empty. "
            f"skip_before_timestep={skip_before_timestep}, start_pred_t={start_pred_t}."
        )

    mode = normalize_rollout_init_mode(rollout_init_mode)
    if mode == ROLLOUT_INIT_MEAN_HISTORY:
        mean_history = histories.mean(dim=0)
        return [mean_history.clone() for _ in range(n_members)], [-1] * n_members

    member_indices = _select_rollout_reference_member_indices(
        n_ref=int(histories.shape[0]),
        n_members=int(n_members),
    )
    return [histories[idx].clone() for idx in member_indices], member_indices


class _PhaseTimer:
    """Context manager that logs phase name and duration."""

    def __init__(self, logger: logging.Logger, phase_name: str) -> None:
        self.logger = logger
        self.phase_name = phase_name
        self._t0 = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._t0 = time.perf_counter()
        self.logger.info(">>> %s", self.phase_name)
        return self

    def __exit__(self, exc_type: type, exc: BaseException, _tb: Any) -> bool:
        dt = time.perf_counter() - self._t0
        if exc is None:
            self.logger.info("<<< %s completed in %.2fs", self.phase_name, dt)
        else:
            self.logger.exception("<<< %s failed after %.2fs: %s", self.phase_name, dt, exc)
        return False

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve device string to torch.device; fallback to CPU if CUDA unavailable."""
    if isinstance(device, torch.device):
        return device
    if "cuda" in device and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resolve_checkpoint_in_dir(save_dir: Path) -> Tuple[Path, str]:
    """Return (dir, alias) for a single checkpoint directory."""
    if (save_dir / "best_model_state_dict.pt").exists():
        return save_dir, CHECKPOINT_BEST
    if (save_dir / "model_state_dict.pt").exists():
        return save_dir, CHECKPOINT_LAST
    found = [p.name for p in save_dir.iterdir()] if save_dir.exists() else []
    raise FileNotFoundError(
        f"No checkpoint in {save_dir}. Expected one of {CHECKPOINT_FILES}. Found: {found}"
    )


def _discover_checkpoint_runs(checkpoint_path: Path) -> List[Tuple[Path, str, str]]:
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
        run_dir, alias = _resolve_checkpoint_in_dir(checkpoint_path)
        runs.append((run_dir, alias, checkpoint_path.name or "model0"))
        return runs
    except FileNotFoundError:
        pass

    child_dirs = sorted([p for p in checkpoint_path.iterdir() if p.is_dir()])
    for child in child_dirs:
        try:
            run_dir, alias = _resolve_checkpoint_in_dir(child)
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
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _resolve_cli_config_path(cli_value: Optional[str]) -> Optional[Path]:
    """Resolve --config_path value using the same relative-path logic as training script."""
    if not cli_value:
        return None
    cfg = Path(cli_value)
    if not cfg.is_absolute():
        cfg = _REPO_ROOT / cfg
    return cfg.resolve()


def _move_normalizers_to_device(normalizers: Dict[str, Any]) -> torch.device:
    """Move all normalizers to one device (in-place). Return that device."""
    ref: torch.device = torch.device("cpu")
    for key in DEVICE_REF_KEYS:
        if key in normalizers and normalizers[key] is not None:
            if hasattr(normalizers[key], "mean"):
                ref = normalizers[key].mean.device
                break
    for key in NORMALIZER_KEYS:
        if key in normalizers and normalizers[key] is not None:
            normalizers[key].to(ref)
    return ref


def _parse_args() -> argparse.Namespace:
    """Parse CLI and strip known args so config pipeline only sees its own."""
    parser = argparse.ArgumentParser(
        description="Post-training evaluation for flood GINO WV."
    )
    parser.add_argument(
        "--eval_log_file",
        type=str,
        default=DEFAULT_EVAL_LOG,
        help="Log file path (relative to checkpoint dir or absolute).",
    )
    parser.add_argument(
        "--run_single_step",
        action="store_true",
        help="Run one-step test evaluation.",
    )
    parser.add_argument(
        "--skip_single_step",
        action="store_true",
        help="Skip one-step evaluation.",
    )
    parser.add_argument(
        "--run_rollout",
        action="store_true",
        help="Force rollout evaluation even if config says no.",
    )
    parser.add_argument(
        "--skip_rollout",
        action="store_true",
        help="Skip rollout evaluation.",
    )
    parser.add_argument(
        "--gaussian_state_update",
        type=str,
        choices=("sample", "mu"),
        default=None,
        help=(
            "Gaussian rollout state transition update mode. "
            "'sample' uses sampled next-state, 'mu' uses Gaussian mean. "
            "If omitted, uses rollout.gaussian_state_update from config (default: sample)."
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Raise if mutually exclusive flags are both set."""
    if args.run_single_step and args.skip_single_step:
        raise ValueError("Cannot set both --run_single_step and --skip_single_step")
    if args.run_rollout and args.skip_rollout:
        raise ValueError("Cannot set both --run_rollout and --skip_rollout")
