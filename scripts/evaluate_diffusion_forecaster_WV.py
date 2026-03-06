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
from scripts.diffusion_script_utils import (  # noqa: E402
    load_checkpoint_bundle,
    safe_get,
    to_builtin,
)
from train_gino_flood_train_rollout_animation_WV import (  # noqa: E402
    parse_target_variables,
    set_seed,
    setup_logging,
)


def _load_state_dict_compat(module: torch.nn.Module, state_dict: Dict[str, Any], *, name: str) -> None:
    """Load state dict with fallback for legacy DDP `module.` prefixes."""
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            raise RuntimeError(
                f"Could not load {name}: mixed/non-DDP keys detected (first key={key!r})."
            )
    module.load_state_dict(stripped, strict=True)


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
    configured = str(safe_get(safe_get(config, "distributed", {}), "device", "cuda:0"))
    if configured.startswith("cuda") and torch.cuda.is_available():
        return torch.device(configured)
    return torch.device("cpu")


def _resolve_normalizer_path(config: Any, fallback: Optional[Path] = None) -> Optional[Path]:
    norm_path = safe_get(safe_get(config, "data", {}), "normalizer_path", None)
    if norm_path is None and fallback is not None:
        return fallback
    if norm_path is None:
        return None
    p = Path(str(norm_path))
    if not p.is_absolute():
        p = Path(str(safe_get(safe_get(config, "data", {}), "root", "."))) / p
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
            root_raw = safe_get(safe_get(config, "checkpoint", {}), "resume_from_dir", None)
        if root_raw is None:
            root_raw = safe_get(safe_get(config, "checkpoint", {}), "save_dir", None)
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


def _load_diffusion_model(
    ckpt_path: Path,
    config: Any,
    device: torch.device,
    *,
    allow_unsafe_legacy_load: bool,
    logger,
) -> ConditionalDDOForecaster:
    ckpt = load_checkpoint_bundle(
        ckpt_path,
        map_location="cpu",
        allow_unsafe_legacy_load=allow_unsafe_legacy_load,
        logger=logger,
    )

    gino_cfg = copy.deepcopy(ckpt.get("gino_config", to_builtin(safe_get(config, "gino", {}))))
    model_cfg = {"arch": "gino", "gino": copy.deepcopy(gino_cfg)}
    denoiser = get_model(model_cfg).to(device)
    _load_state_dict_compat(denoiser, ckpt["denoiser_state_dict"], name="denoiser_state_dict")

    diff_hparams = ckpt.get("diffusion_hparams", {})
    gp_h = diff_hparams.get("gp", {})
    cond_h = diff_hparams.get("conditioning", {})
    sampler_h = diff_hparams.get("sampler", {})
    schedule_h = diff_hparams.get("schedule", {})

    # Allow runtime overrides from config.diffusion for evaluation.
    diff_cfg = safe_get(config, "diffusion", {})
    gp_cfg = safe_get(diff_cfg, "gp", {})
    cond_cfg_runtime = safe_get(diff_cfg, "conditioning", {})
    sampler_cfg = safe_get(diff_cfg, "sampler", {})
    schedule_cfg = safe_get(diff_cfg, "schedule", {})
    allow_cond_override = bool(safe_get(cond_cfg_runtime, "allow_eval_conditioning_override", False))

    loaded_fno_norm = str(safe_get(gino_cfg, "fno_norm", "")).lower()
    loaded_time_injection = "adain" if loaded_fno_norm == "ada_in" else "channel"
    runtime_time_injection = safe_get(cond_cfg_runtime, "time_injection", None)
    checkpoint_time_injection = cond_h.get("time_injection", loaded_time_injection)
    requested_time_injection = str(
        runtime_time_injection if runtime_time_injection is not None else checkpoint_time_injection
    ).lower()
    if requested_time_injection not in {"channel", "adain"}:
        raise ValueError(
            "diffusion.conditioning.time_injection must be one of {'channel', 'adain'}, "
            f"got {requested_time_injection!r}"
        )
    if requested_time_injection != loaded_time_injection and not allow_cond_override:
        raise ValueError(
            "Conditioning mode mismatch between runtime config and checkpoint denoiser. "
            f"Requested time_injection={requested_time_injection!r} but loaded gino.fno_norm={loaded_fno_norm!r}. "
            "Set diffusion.conditioning.allow_eval_conditioning_override=true only if you intentionally "
            "want to override this behavior."
        )
    if requested_time_injection != loaded_time_injection and allow_cond_override:
        logger.warning(
            "Overriding checkpoint conditioning mode: requested=%s loaded=%s",
            requested_time_injection,
            loaded_time_injection,
        )

    default_time_embedding_dim = int(
        cond_h.get("time_embedding_dim", safe_get(gino_cfg, "fno_ada_in_dim", 32))
    )
    if requested_time_injection == "channel":
        default_time_embedding_dim = int(cond_h.get("time_embedding_dim", 32))

    cond_cfg = ConditioningConfig(
        add_noisy_target=bool(safe_get(cond_cfg_runtime, "add_noisy_target", cond_h.get("add_noisy_target", True))),
        add_time_features=bool(safe_get(cond_cfg_runtime, "add_time_features", cond_h.get("add_time_features", True))),
        time_feature_type=str(safe_get(cond_cfg_runtime, "time_feature_type", cond_h.get("time_feature_type", "sincos"))),
        time_injection=requested_time_injection,
        time_embedding_dim=int(safe_get(cond_cfg_runtime, "time_embedding_dim", default_time_embedding_dim)),
        time_embedding_hidden_dim=int(
            safe_get(cond_cfg_runtime, "time_embedding_hidden_dim", cond_h.get("time_embedding_hidden_dim", 128))
        ),
        time_embedding_scale=float(
            safe_get(cond_cfg_runtime, "time_embedding_scale", cond_h.get("time_embedding_scale", 10000.0))
        ),
    )

    gp_sampler = PointRFFGaussianProcessSampler(
        dim=2,
        gp_type=str(safe_get(gp_cfg, "type", gp_h.get("type", "rff_rbf"))),
        sigma=float(safe_get(gp_cfg, "sigma", gp_h.get("sigma", 1.0))),
        length_scale=float(safe_get(gp_cfg, "length_scale", gp_h.get("length_scale", 0.05))),
        rff_features=int(safe_get(gp_cfg, "rff_features", gp_h.get("rff_features", 256))),
        seed=int(ckpt.get("seed", safe_get(safe_get(config, "distributed", {}), "seed", 123))),
    ).to(device)

    forecaster = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp_sampler,
        parameterization=str(safe_get(diff_cfg, "parameterization", diff_hparams.get("parameterization", "epsilon"))),
        timestep_sampler=str(safe_get(diff_cfg, "timestep_sampler", diff_hparams.get("timestep_sampler", "low_discrepancy"))),
        lmbd0=float(safe_get(schedule_cfg, "lmbd0", schedule_h.get("lmbd0", 10.0))),
        lmbd1=float(safe_get(schedule_cfg, "lmbd1", schedule_h.get("lmbd1", -10.0))),
        weight_method=safe_get(schedule_cfg, "weight_method", schedule_h.get("weight_method", "shifted_sigmoid_2")),
        conditioning=cond_cfg,
        sampler_method=str(safe_get(sampler_cfg, "method", sampler_h.get("method", "denoise"))),
        sampler_num_steps=int(safe_get(sampler_cfg, "num_steps", sampler_h.get("num_steps", 40))),
        sampler_s_min=float(safe_get(sampler_cfg, "s_min", sampler_h.get("s_min", 1e-4))),
        sampler_return_mean_last=bool(safe_get(sampler_cfg, "return_mean_last", sampler_h.get("return_mean_last", True))),
    ).to(device)
    time_mlp_state = ckpt.get("time_mlp_state_dict", None)
    if forecaster.time_mlp is not None:
        if time_mlp_state is not None:
            _load_state_dict_compat(
                forecaster.time_mlp,
                time_mlp_state,
                name="time_mlp_state_dict",
            )
        else:
            logger.warning(
                "Checkpoint %s has no time_mlp_state_dict; using current initialization for AdaIN time MLP.",
                ckpt_path,
            )
    elif time_mlp_state is not None:
        logger.warning(
            "Checkpoint %s includes time_mlp_state_dict but runtime conditioning has no time_mlp; ignoring.",
            ckpt_path,
        )
    logger.info(
        (
            "Loaded diffusion model %s with time_injection=%s time_embedding_dim=%d "
            "add_noisy_target=%s fno_norm=%s in_channels=%s"
        ),
        ckpt_path,
        cond_cfg.time_injection,
        cond_cfg.time_embedding_dim,
        cond_cfg.add_noisy_target,
        str(safe_get(gino_cfg, "fno_norm", "unknown")),
        str(safe_get(gino_cfg, "data_channels", "unknown")),
    )

    forecaster.eval()
    return forecaster


def main() -> int:
    args = _parse_cli()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()
    config = _load_config(config_path)

    device = _resolve_device(config)
    log_file = safe_get(config, "log_file", "eval_diffusion.log")
    if not Path(str(log_file)).is_absolute():
        log_file = str((_SCRIPT_DIR / str(log_file)).resolve())
    logger = setup_logging(
        log_level=str(safe_get(config, "log_level", "INFO")),
        log_file=log_file,
        logger_name="flood_diffusion_eval",
    )

    seed = int(safe_get(safe_get(config, "distributed", {}), "seed", 123))
    deterministic = bool(safe_get(config, "deterministic", True))
    set_seed(seed, deterministic=deterministic)
    logger.info("Using device=%s seed=%d", device, seed)
    allow_unsafe_legacy_load = bool(
        safe_get(safe_get(config, "checkpoint", {}), "allow_unsafe_legacy_load", True)
    )
    if allow_unsafe_legacy_load:
        logger.warning(
            "checkpoint.allow_unsafe_legacy_load=true. Legacy pickle checkpoints "
            "must be treated as trusted inputs."
        )

    target_variables = parse_target_variables(safe_get(safe_get(config, "data", {}), "target_variables", ["wd"]))
    if target_variables != ["wd"]:
        raise ValueError("Diffusion v1 evaluation supports target_variables=['wd'] only.")

    checkpoint_paths = _discover_checkpoints(args, config)
    logger.info("Discovered %d diffusion checkpoints", len(checkpoint_paths))
    for p in checkpoint_paths:
        logger.info("  checkpoint: %s", p)

    first_ckpt = load_checkpoint_bundle(
        checkpoint_paths[0],
        map_location="cpu",
        allow_unsafe_legacy_load=allow_unsafe_legacy_load,
        logger=logger,
    )
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
        model = _load_diffusion_model(
            ckpt_path,
            config=config,
            device=device,
            allow_unsafe_legacy_load=allow_unsafe_legacy_load,
            logger=logger,
        )
        models.append(model)
    logger.info("Loaded %d diffusion model(s)", len(models))

    out_dir = Path(str(safe_get(safe_get(config, "rollout", {}), "out_dir", "./rollout_uq_WV_depth_only_diffusion")))
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
            rollout_length=int(safe_get(safe_get(config, "data", {}), "rollout_length", 78)),
            history_steps=int(safe_get(safe_get(config, "data", {}), "n_history", 3)),
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            device=device,
            skip_before_timestep=int(_opt(config, "data", "skip_before_timestep", 0)),
            dt=float(safe_get(safe_get(config, "data", {}), "dt", 1200.0)),
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
            rollout_length=int(safe_get(safe_get(config, "data", {}), "rollout_length", 78)),
            history_steps=int(safe_get(safe_get(config, "data", {}), "n_history", 3)),
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            device=device,
            skip_before_timestep=int(_opt(config, "data", "skip_before_timestep", 0)),
            dt=float(safe_get(safe_get(config, "data", {}), "dt", 1200.0)),
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
