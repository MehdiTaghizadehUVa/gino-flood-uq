"""Shared runtime/config helpers for WV flood workflows."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig

from neuralop.training import setup

_REPO_ROOT = Path(__file__).resolve().parents[3]

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all relevant random seeds and PyTorch/CuDNN settings for full reproducibility.
    Call once at the start of main() after config is loaded.

    When deterministic=True, CuDNN benchmark is disabled so that convolutions etc.
    are deterministic; training may be slightly slower. For extra reproducibility
    (e.g. hash-based dict order), set env PYTHONHASHSEED=0 before starting Python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def make_dataloader_generator(seed: int):
    """Return a fresh torch.Generator with the given seed for reproducible DataLoader shuffle."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_split_generator(seed: int):
    """Return a fresh torch.Generator with the given seed for reproducible random_split."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def parse_target_variables(target_variables):
    """
    Parse configured target variable names.

    Allowed names (case-insensitive): wd, vx, vy.
    Returns a normalized ordered list containing a non-empty subset of these.
    """
    allowed = ("wd", "vx", "vy")
    if target_variables is None:
        return list(allowed)
    out = []
    for v in target_variables:
        key = str(v).strip().lower()
        if key not in allowed:
            raise ValueError(
                f"Unknown target variable '{v}'. Allowed: {allowed}."
            )
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("target_variables must contain at least one of: wd, vx, vy.")
    return out


def _safe_float(val, default: float) -> float:
    """Convert config value to float with fallback for None/invalid inputs."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(val, default: int) -> int:
    """Convert config value to int with fallback for None/invalid inputs."""
    if val is None:
        return int(default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


def _is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def _cfg_get(obj, key, default):
    """Safe config access for ConfigPipeline nodes that may raise KeyError on missing attributes."""
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        return default


FGN_LATENT_TEMPORAL_MODES = {"stepwise", "persistent"}
FGN_AR_STATE_UPDATE_MODES = {"mean_feedback", "member_feedback"}
BOUNDARY_SOURCE_MODES = {"member_hdf", "clean_family"}
STRUCTURAL_DRY_POLICIES = {"legacy_full_domain", "masked_primary"}
STRUCTURAL_DRY_MASK_DEFINITIONS = {"exact_zero"}
DEFAULT_CLEAN_BOUNDARY_FILES = {
    "train": "Hydrographs_Train_Clean.txt",
    "val": "Hydrographs_Val_Clean.txt",
    "test": "Hydrographs_Test_Clean.txt",
}
_CLEAN_BOUNDARY_CACHE = {}


def _normalize_choice(value, default, allowed, label):
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"Unknown {label}={value!r}. Expected one of {sorted(allowed)}."
        )
    return normalized


def normalize_fgn_latent_temporal_mode(value, default="stepwise"):
    return _normalize_choice(
        value,
        default=default,
        allowed=FGN_LATENT_TEMPORAL_MODES,
        label="gino.fgn_latent_temporal_mode",
    )


def normalize_fgn_ar_state_update(value, default="mean_feedback"):
    return _normalize_choice(
        value,
        default=default,
        allowed=FGN_AR_STATE_UPDATE_MODES,
        label="opt.fgn_ar_state_update",
    )


def normalize_boundary_source(value, default="member_hdf"):
    return _normalize_choice(
        value,
        default=default,
        allowed=BOUNDARY_SOURCE_MODES,
        label="boundary_source",
    )


def normalize_structural_dry_policy(value, default="legacy_full_domain"):
    return _normalize_choice(
        value,
        default=default,
        allowed=STRUCTURAL_DRY_POLICIES,
        label="structural_dry.policy",
    )


def normalize_structural_dry_mask_definition(value, default="exact_zero"):
    return _normalize_choice(
        value,
        default=default,
        allowed=STRUCTURAL_DRY_MASK_DEFINITIONS,
        label="structural_dry.mask_definition",
    )


def _resolve_normalizer_path_from_config(
    config,
    *,
    allow_data_root_fallback: bool = False,
):
    data_cfg = _cfg_get(config, "data", None)
    normalizer_path = _cfg_get(data_cfg, "normalizer_path", None)
    if normalizer_path is None:
        if not allow_data_root_fallback:
            return None
        root = _cfg_get(data_cfg, "normalizer_root", None) or _cfg_get(data_cfg, "root", None)
        if root is None:
            return None
        return (Path(str(root)).resolve() / "normalizers_depth_only.pt").resolve()

    p = Path(str(normalizer_path))
    if p.is_absolute():
        return p.resolve()
    normalizer_root = _cfg_get(data_cfg, "normalizer_root", None)
    if normalizer_root is not None:
        return (Path(str(normalizer_root)).resolve() / p).resolve()
    if allow_data_root_fallback:
        data_root = _cfg_get(data_cfg, "root", None)
        if data_root is not None:
            return (Path(str(data_root)).resolve() / p).resolve()
    raise ValueError(
        "Relative data.normalizer_path requires data.normalizer_root. "
        "Refusing to resolve structural-dry artifacts against data.root in this mode."
    )


def resolve_structural_dry_artifact_path(
    config,
    *,
    normalizer_path=None,
    allow_data_root_fallback: bool = False,
):
    structural_cfg = _cfg_get(config, "structural_dry", None)
    mask_definition = normalize_structural_dry_mask_definition(
        _cfg_get(structural_cfg, "mask_definition", "exact_zero")
    )
    explicit_mask_path = _cfg_get(structural_cfg, "mask_path", None)
    resolved_normalizer_path = None
    if normalizer_path is not None:
        resolved_normalizer_path = Path(str(normalizer_path)).resolve()
    else:
        resolved_normalizer_path = _resolve_normalizer_path_from_config(
            config,
            allow_data_root_fallback=allow_data_root_fallback,
        )

    if explicit_mask_path is not None:
        mask_path = Path(str(explicit_mask_path))
        if not mask_path.is_absolute():
            if resolved_normalizer_path is not None:
                base_dir = (
                    resolved_normalizer_path.parent
                    if resolved_normalizer_path.suffix
                    else resolved_normalizer_path
                )
            elif allow_data_root_fallback:
                data_root = _cfg_get(_cfg_get(config, "data", None), "root", None)
                if data_root is None:
                    raise ValueError(
                        "Relative structural_dry.mask_path requires either an explicit normalizer path "
                        "or data.root fallback."
                    )
                base_dir = Path(str(data_root)).resolve()
            else:
                raise ValueError(
                    "Relative structural_dry.mask_path requires an explicit normalizer path or "
                    "data.normalizer_root."
                )
            mask_path = (base_dir / mask_path).resolve()
    else:
        if resolved_normalizer_path is not None:
            base_dir = (
                resolved_normalizer_path.parent
                if resolved_normalizer_path.suffix
                else resolved_normalizer_path
            )
        elif allow_data_root_fallback:
            data_root = _cfg_get(_cfg_get(config, "data", None), "root", None)
            if data_root is None:
                raise ValueError(
                    "Unable to resolve default structural-dry artifact path without data.root."
                )
            base_dir = Path(str(data_root)).resolve()
        else:
            raise ValueError(
                "Unable to resolve structural-dry artifact path without a normalizer path."
            )
        mask_path = (base_dir / f"structural_dry_mask_{mask_definition}.pt").resolve()

    summary_path = mask_path.with_name(f"{mask_path.stem}_summary.json")
    return mask_path, summary_path


def get_structural_dry_policy_kwargs(
    config,
    *,
    normalizer_path=None,
    allow_data_root_fallback: bool = False,
):
    structural_cfg = _cfg_get(config, "structural_dry", None)
    policy = normalize_structural_dry_policy(
        _cfg_get(structural_cfg, "policy", "legacy_full_domain")
    )
    mask_definition = normalize_structural_dry_mask_definition(
        _cfg_get(structural_cfg, "mask_definition", "exact_zero")
    )
    artifact_path = summary_path = None
    if policy == "masked_primary":
        artifact_path, summary_path = resolve_structural_dry_artifact_path(
            config,
            normalizer_path=normalizer_path,
            allow_data_root_fallback=allow_data_root_fallback,
        )
    return {
        "policy": policy,
        "mask_definition": mask_definition,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
        "report_full_domain_secondary": bool(
            _cfg_get(structural_cfg, "report_full_domain_secondary", True)
        ),
        "report_dry_background_secondary": bool(
            _cfg_get(structural_cfg, "report_dry_background_secondary", True)
        ),
    }


def wait_for_structural_dry_artifact(
    artifact_path,
    *,
    timeout_seconds: float = 7200.0,
    poll_interval_seconds: float = 5.0,
):
    from neuralop.flood.data.structural_dry import load_structural_dry_artifact

    artifact_path = Path(str(artifact_path)).resolve()
    deadline = time.monotonic() + float(timeout_seconds)
    last_error = None
    while time.monotonic() < deadline:
        if artifact_path.exists():
            try:
                return load_structural_dry_artifact(artifact_path)
            except Exception as exc:  # pragma: no cover - race-safe polling
                last_error = exc
        time.sleep(float(poll_interval_seconds))
    if last_error is not None:
        raise RuntimeError(
            f"Timed out waiting for a readable structural-dry artifact at {artifact_path}."
        ) from last_error
    raise TimeoutError(
        f"Timed out waiting for structural-dry artifact to appear at {artifact_path}."
    )


def parse_family_id_from_run_id(run_id: str) -> str:
    run_id = str(run_id).strip()
    if "_sim" not in run_id:
        raise ValueError(
            f"run_id={run_id!r} does not match expected family-member convention '<family_id>_simNN'."
        )
    family_id, _, suffix = run_id.rpartition("_sim")
    if not family_id or not suffix:
        raise ValueError(
            f"run_id={run_id!r} does not match expected family-member convention '<family_id>_simNN'."
        )
    return family_id


def family_id_lookup_candidates(run_id: str) -> list[str]:
    """Return exact-first family-id candidates for clean-boundary lookup.

    The dynamic dataset package can encode run IDs with a dataset prefix
    (for example ``M40_TR000001_sim00``) while the clean hydrograph table uses
    the bare family/event id (for example ``TR000001``). We keep exact match as
    the primary contract and only add suffix aliases as fallbacks.
    """
    family_id = parse_family_id_from_run_id(run_id)
    candidates = [family_id]
    parts = [part for part in family_id.split("_") if part]
    if len(parts) > 1:
        for start in range(1, len(parts)):
            alias = "_".join(parts[start:])
            if alias and alias not in candidates:
                candidates.append(alias)
    return candidates


def resolve_family_id_for_boundary(run_id: str, boundary_by_family: dict[str, np.ndarray]) -> str:
    """Resolve the clean-boundary family id for a run id.

    Exact match is required when available. If the exact family id is not found,
    try suffix aliases so datasets with prefixed run IDs can still consume clean
    boundary tables keyed by bare event IDs.
    """
    candidates = family_id_lookup_candidates(run_id)
    for candidate in candidates:
        if candidate in boundary_by_family:
            return candidate
    raise KeyError(
        f"Family {candidates[0]!r} not found in clean boundary file. Tried aliases: {candidates}."
    )


def _resolve_clean_boundary_root(clean_boundary_root, section_root):
    if clean_boundary_root is None:
        raise ValueError(
            "boundary_source='clean_family' requires clean_boundary_root to be set."
        )
    root_path = Path(str(clean_boundary_root))
    if root_path.is_absolute():
        return root_path.resolve()
    if section_root is None:
        raise ValueError(
            "Relative clean_boundary_root requires the corresponding data root to be set."
        )
    return (Path(str(section_root)) / root_path).resolve()


def get_dataset_boundary_kwargs(config_section, *, split=None):
    section_root = _cfg_get(config_section, "root", None)
    boundary_source = normalize_boundary_source(
        _cfg_get(config_section, "boundary_source", "member_hdf")
    )
    kwargs = {"boundary_source": boundary_source}
    if boundary_source != "clean_family":
        return kwargs

    clean_boundary_root = _resolve_clean_boundary_root(
        _cfg_get(config_section, "clean_boundary_root", None),
        section_root,
    )
    clean_boundary_file = _cfg_get(config_section, "clean_boundary_file", None)
    if clean_boundary_file is None and split is not None:
        clean_boundary_file = DEFAULT_CLEAN_BOUNDARY_FILES.get(str(split).strip().lower())
    if clean_boundary_file is None:
        raise ValueError(
            "boundary_source='clean_family' requires clean_boundary_file to be set."
        )
    kwargs["clean_boundary_root"] = str(clean_boundary_root)
    kwargs["clean_boundary_file"] = str(clean_boundary_file)
    return kwargs


def _load_clean_boundary_table(clean_boundary_root, clean_boundary_file):
    table_path = (Path(str(clean_boundary_root)) / str(clean_boundary_file)).resolve()
    cache_key = str(table_path)
    cached = _CLEAN_BOUNDARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not table_path.exists():
        raise FileNotFoundError(f"Clean boundary file not found: {table_path}")

    with open(table_path, "r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip()
    if not header:
        raise ValueError(f"Clean boundary file {table_path} is empty.")
    family_ids = [col.strip() for col in header.split("\t") if col.strip()]
    if not family_ids:
        raise ValueError(f"Clean boundary file {table_path} has no family-id header columns.")

    raw = np.loadtxt(str(table_path), delimiter="\t", skiprows=1, dtype=np.float32)
    if raw.ndim == 1:
        if len(family_ids) == 1:
            raw = raw.reshape(-1, 1)
        else:
            raw = raw.reshape(1, -1)
    if raw.shape[1] != len(family_ids):
        raise ValueError(
            f"Clean boundary file {table_path} has {len(family_ids)} headers but data shape {tuple(raw.shape)}."
        )

    boundary_by_family = {
        str(family_id): np.asarray(raw[:, idx], dtype=np.float32).copy()
        for idx, family_id in enumerate(family_ids)
    }
    out = {
        "path": table_path,
        "n_time": int(raw.shape[0]),
        "boundary_by_family": boundary_by_family,
    }
    _CLEAN_BOUNDARY_CACHE[cache_key] = out
    return out

def _to_builtin(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {
            str(k): _to_builtin(v)
            for k, v in vars(obj).items()
            if not str(k).startswith("_")
        }
    try:
        return {str(k): _to_builtin(obj[k]) for k in obj}
    except Exception:
        return str(obj)


def save_effective_config_snapshot(config, save_dir, logger=None):
    save_path = Path(save_dir) / "effective_config.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(config), handle, indent=2, sort_keys=True)
    if logger is not None:
        logger.info("Saved effective config snapshot to %s", save_path)


def dataloader_worker_init(worker_id: int, base_seed: int) -> None:
    """Top-level worker init for Windows multiprocessing pickling compatibility."""
    worker_seed = int(base_seed) + int(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------------------------------
# Logging setup: file + console, optional rotation, config-driven level/path
# ---------------------------------------------------------------------------
LOG_FORMAT_DETAILED = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_FORMAT_CONSOLE = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_file_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    log_file_backup_count: int = 3,
    logger_name: str = "flood_train",
) -> logging.Logger:
    """
    Configure and return a logger with optional file (rotating) and console handlers.
    Does not add duplicate handlers if the logger already has them.
    """
    logger = logging.getLogger(logger_name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers when re-calling (e.g. in tests)
    if logger.handlers:
        return logger

    formatter_file = logging.Formatter(LOG_FORMAT_DETAILED, datefmt=LOG_DATEFMT)
    formatter_console = logging.Formatter(LOG_FORMAT_CONSOLE, datefmt=LOG_DATEFMT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter_console)
    logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=log_file_max_bytes,
            backupCount=log_file_backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter_file)
        logger.addHandler(fh)
        logger.info("Logging to file: %s", log_path.resolve())

    return logger


###############################################################################
# 1) CONFIG & SETUP
###############################################################################
def load_config_and_setup():
    """
    Reads gino_pluvial_flood_config_WV.yaml (or --config_path <path>) and sets up device.
    Use --config_path to avoid clash with ArgparseConfig's --config_name/--config_file.
    """
    import sys
    config_name = "flood"
    config_path = _REPO_ROOT / "config" / "gino_pluvial_flood_config_WV_depth_only.yaml"
    argv = list(sys.argv[1:])
    for i, a in enumerate(argv):
        if a == "--config_path" and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            if not config_path.is_absolute():
                config_path = _REPO_ROOT / config_path
            # Remove --config_path and its value so ArgparseConfig does not see them
            idx = sys.argv.index("--config_path")
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
            break
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    pipe = ConfigPipeline([
        YamlConfig(str(config_path), config_name=config_name, config_folder=str(_REPO_ROOT / "config")),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
    ])
    config = pipe.read_conf()

    # Setup device (and distributed environment if needed)
    device, is_logger = setup(config)
    return config, device, is_logger


###############################################################################
# 1a) Write train.txt from all HDF run IDs in data_root
###############################################################################
def write_train_txt_from_data_root(
    data_root, train_txt: str = "train.txt", hdf_suffix: str = ".hdf"
):
    """
    Write train.txt with one run ID per line (filename stem of each *hdf_suffix in data_root).
    Returns the list of run_ids. Call when train.txt is missing or to refresh with all existing simulations.
    """
    data_root = Path(data_root)
    run_ids = sorted(p.stem for p in data_root.glob(f"*{hdf_suffix}") if p.is_file())
    out_path = data_root / train_txt
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(run_ids) + ("\n" if run_ids else ""))
    return run_ids
