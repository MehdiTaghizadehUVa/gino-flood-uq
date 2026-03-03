#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate DDO-style diffusion forecaster for WV flood depth-only rollout UQ."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate_post_training_flood_WV import (  # noqa: E402
    _build_rollout_normalized_dataset,
    _opt,
    _rollout_prediction_generic,
    _rollout_prediction_per_hydrograph,
)
from neuralop import get_model  # noqa: E402
from neuralop.data.transforms.normalizers import load_normalizers  # noqa: E402
from neuralop.diffusion import (  # noqa: E402
    ConditioningConfig,
    ConditionalDDOForecaster,
    PointRFFGaussianProcessSampler,
)
from train_gino_flood_train_rollout_animation_WV import (  # noqa: E402
    parse_target_variables,
    set_seed,
    setup_logging,
)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if hasattr(obj, "items"):
        try:
            return {k: _to_builtin(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        out = {}
        for k, v in vars(obj).items():
            if callable(v):
                continue
            out[k] = _to_builtin(v)
        return out
    return obj


def _safe_get(obj: Any, key: str, default: Any) -> Any:
    try:
        return getattr(obj, key)
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        return default


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diffusion forecaster rollout UQ")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(_REPO_ROOT / "config" / "gino_pluvial_flood_config_WV_depth_only_diffusion.yaml"),
        help="Path to diffusion config YAML.",
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        default=None,
        help="Comma-separated checkpoint file/dir paths. If omitted, discover from checkpoint_root/config.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=None,
        help="Root dir containing checkpoint files or run subdirectories.",
    )
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args


def _load_config(config_path: Path) -> Any:
    config_name = "flood"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    pipe = ConfigPipeline(
        [
            YamlConfig(str(config_path), config_name=config_name, config_folder=str(_REPO_ROOT / "config")),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        ]
    )
    return pipe.read_conf()


def _resolve_device(config: Any) -> torch.device:
    configured = str(_safe_get(_safe_get(config, "distributed", {}), "device", "cuda:0"))
    if configured.startswith("cuda") and torch.cuda.is_available():
        return torch.device(configured)
    return torch.device("cpu")


def _resolve_normalizer_path(config: Any, fallback: Optional[Path] = None) -> Optional[Path]:
    norm_path = _safe_get(_safe_get(config, "data", {}), "normalizer_path", None)
    if norm_path is None and fallback is not None:
        return fallback
    if norm_path is None:
        return None
    p = Path(str(norm_path))
    if not p.is_absolute():
        p = Path(str(_safe_get(_safe_get(config, "data", {}), "root", "."))) / p
    return p.resolve()


def _expand_checkpoint_candidates(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []

    direct = []
    for name in ("checkpoint_best.pt", "checkpoint.pt"):
        p = path / name
        if p.exists():
            direct.append(p)

    if direct:
        return direct

    found: List[Path] = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        for name in ("checkpoint_best.pt", "checkpoint.pt"):
            p = child / name
            if p.exists():
                found.append(p)
                break
    return found


def _discover_checkpoints(args: argparse.Namespace, config: Any) -> List[Path]:
    candidates: List[Path] = []
    if args.checkpoint_paths:
        for raw in str(args.checkpoint_paths).split(","):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = (_REPO_ROOT / p).resolve()
            candidates.extend(_expand_checkpoint_candidates(p))
    else:
        root_raw = args.checkpoint_root
        if root_raw is None:
            root_raw = _safe_get(_safe_get(config, "checkpoint", {}), "resume_from_dir", None)
        if root_raw is None:
            root_raw = _safe_get(_safe_get(config, "checkpoint", {}), "save_dir", None)
        if root_raw is None:
            raise ValueError("No checkpoint source provided. Set --checkpoint_paths or --checkpoint_root.")
        root = Path(str(root_raw))
        if not root.is_absolute():
            root = (_SCRIPT_DIR / root).resolve()
        candidates.extend(_expand_checkpoint_candidates(root))

    unique: List[Path] = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)
    if not unique:
        raise FileNotFoundError("No diffusion checkpoints discovered.")
    return unique


def _load_diffusion_model(ckpt_path: Path, config: Any, device: torch.device) -> ConditionalDDOForecaster:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    gino_cfg = ckpt.get("gino_config", _to_builtin(_safe_get(config, "gino", {})))
    model_cfg = {"arch": "gino", "gino": copy.deepcopy(gino_cfg)}
    denoiser = get_model(model_cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser_state_dict"], strict=True)

    diff_hparams = ckpt.get("diffusion_hparams", {})
    gp_h = diff_hparams.get("gp", {})
    cond_h = diff_hparams.get("conditioning", {})
    sampler_h = diff_hparams.get("sampler", {})
    schedule_h = diff_hparams.get("schedule", {})

    # Allow runtime overrides from config.diffusion for evaluation.
    diff_cfg = _safe_get(config, "diffusion", {})
    gp_cfg = _safe_get(diff_cfg, "gp", {})
    cond_cfg = _safe_get(diff_cfg, "conditioning", {})
    sampler_cfg = _safe_get(diff_cfg, "sampler", {})
    schedule_cfg = _safe_get(diff_cfg, "schedule", {})

    gp_sampler = PointRFFGaussianProcessSampler(
        dim=2,
        gp_type=str(_safe_get(gp_cfg, "type", gp_h.get("type", "rff_rbf"))),
        sigma=float(_safe_get(gp_cfg, "sigma", gp_h.get("sigma", 1.0))),
        length_scale=float(_safe_get(gp_cfg, "length_scale", gp_h.get("length_scale", 0.05))),
        rff_features=int(_safe_get(gp_cfg, "rff_features", gp_h.get("rff_features", 256))),
        seed=int(ckpt.get("seed", _safe_get(_safe_get(config, "distributed", {}), "seed", 123))),
    ).to(device)

    forecaster = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp_sampler,
        parameterization=str(_safe_get(diff_cfg, "parameterization", diff_hparams.get("parameterization", "epsilon"))),
        timestep_sampler=str(_safe_get(diff_cfg, "timestep_sampler", diff_hparams.get("timestep_sampler", "low_discrepancy"))),
        lmbd0=float(_safe_get(schedule_cfg, "lmbd0", schedule_h.get("lmbd0", 10.0))),
        lmbd1=float(_safe_get(schedule_cfg, "lmbd1", schedule_h.get("lmbd1", -10.0))),
        weight_method=_safe_get(schedule_cfg, "weight_method", schedule_h.get("weight_method", "shifted_sigmoid_2")),
        conditioning=ConditioningConfig(
            add_noisy_target=bool(_safe_get(cond_cfg, "add_noisy_target", cond_h.get("add_noisy_target", True))),
            add_time_features=bool(_safe_get(cond_cfg, "add_time_features", cond_h.get("add_time_features", True))),
            time_feature_type=str(_safe_get(cond_cfg, "time_feature_type", cond_h.get("time_feature_type", "sincos"))),
        ),
        sampler_method=str(_safe_get(sampler_cfg, "method", sampler_h.get("method", "denoise"))),
        sampler_num_steps=int(_safe_get(sampler_cfg, "num_steps", sampler_h.get("num_steps", 40))),
        sampler_s_min=float(_safe_get(sampler_cfg, "s_min", sampler_h.get("s_min", 1e-4))),
        sampler_return_mean_last=bool(_safe_get(sampler_cfg, "return_mean_last", sampler_h.get("return_mean_last", True))),
    ).to(device)

    forecaster.eval()
    return forecaster


def main() -> int:
    args = _parse_cli()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()
    config = _load_config(config_path)

    device = _resolve_device(config)
    log_file = _safe_get(config, "log_file", "eval_diffusion.log")
    if not Path(str(log_file)).is_absolute():
        log_file = str((_SCRIPT_DIR / str(log_file)).resolve())
    logger = setup_logging(
        log_level=str(_safe_get(config, "log_level", "INFO")),
        log_file=log_file,
        logger_name="flood_diffusion_eval",
    )

    seed = int(_safe_get(_safe_get(config, "distributed", {}), "seed", 123))
    deterministic = bool(_safe_get(config, "deterministic", True))
    set_seed(seed, deterministic=deterministic)
    logger.info("Using device=%s seed=%d", device, seed)

    target_variables = parse_target_variables(_safe_get(_safe_get(config, "data", {}), "target_variables", ["wd"]))
    if target_variables != ["wd"]:
        raise ValueError("Diffusion v1 evaluation supports target_variables=['wd'] only.")

    checkpoint_paths = _discover_checkpoints(args, config)
    logger.info("Discovered %d diffusion checkpoints", len(checkpoint_paths))
    for p in checkpoint_paths:
        logger.info("  checkpoint: %s", p)

    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")
    fallback_normalizer = first_ckpt.get("normalizer_path")
    fallback_normalizer_path = Path(str(fallback_normalizer)).resolve() if fallback_normalizer else None
    normalizer_path = _resolve_normalizer_path(config, fallback=fallback_normalizer_path)
    if normalizer_path is None or not normalizer_path.exists():
        raise FileNotFoundError(
            f"Normalizer file not found. config.data.normalizer_path={normalizer_path}"
        )
    normalizers = load_normalizers(normalizer_path, device=None)
    logger.info("Loaded normalizers from %s", normalizer_path)

    models: List[ConditionalDDOForecaster] = []
    for ckpt_path in checkpoint_paths:
        model = _load_diffusion_model(ckpt_path, config=config, device=device)
        models.append(model)
    logger.info("Loaded %d diffusion model(s)", len(models))

    out_dir = Path(str(_safe_get(_safe_get(config, "rollout", {}), "out_dir", "./rollout_uq_WV_depth_only_diffusion")))
    if not out_dir.is_absolute():
        out_dir = (_SCRIPT_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_norm_ds, hydrograph_samples = _build_rollout_normalized_dataset(
        config=config,
        normalizers=normalizers,
        target_variables=target_variables,
        logger=logger,
    )

    n_models = len(models)
    ens_per_model = _opt(config, "rollout", "n_ensemble_samples_per_model", None)
    if ens_per_model is not None:
        n_ensemble = int(ens_per_model) * max(1, n_models)
    else:
        n_ensemble = int(_opt(config, "rollout", "n_ensemble_samples", 25))
    n_ensemble = max(1, n_ensemble)

    logger.info(
        "Running diffusion rollout evaluation: n_models=%d n_ensemble=%d hydrograph_grouped=%s",
        n_models,
        n_ensemble,
        bool(hydrograph_samples),
    )

    if hydrograph_samples:
        _rollout_prediction_per_hydrograph(
            models=models,
            hydrograph_samples=hydrograph_samples,
            rollout_length=int(_safe_get(_safe_get(config, "data", {}), "rollout_length", 78)),
            history_steps=int(_safe_get(_safe_get(config, "data", {}), "n_history", 3)),
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            device=device,
            skip_before_timestep=int(_opt(config, "data", "skip_before_timestep", 0)),
            dt=float(_safe_get(_safe_get(config, "data", {}), "dt", 1200.0)),
            out_dir=str(out_dir),
            target_variables=target_variables,
            logger=logger,
            fgn_noise_dim=None,
            n_ensemble_samples=n_ensemble,
            gaussian_mode=False,
            gaussian_min_logvar=-9.0,
            gaussian_max_logvar=4.0,
        )
    else:
        _rollout_prediction_generic(
            models=models,
            rollout_dataset=rollout_norm_ds,
            rollout_length=int(_safe_get(_safe_get(config, "data", {}), "rollout_length", 78)),
            history_steps=int(_safe_get(_safe_get(config, "data", {}), "n_history", 3)),
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            device=device,
            skip_before_timestep=int(_opt(config, "data", "skip_before_timestep", 0)),
            dt=float(_safe_get(_safe_get(config, "data", {}), "dt", 1200.0)),
            out_dir=str(out_dir),
            target_variables=target_variables,
            logger=logger,
            fgn_noise_dim=None,
            n_ensemble_samples=n_ensemble,
            gaussian_mode=False,
            gaussian_min_logvar=-9.0,
            gaussian_max_logvar=4.0,
        )

    logger.info("Diffusion rollout evaluation completed. Outputs: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
