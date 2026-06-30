"""Canonical flood operator evaluation application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from neuralop.flood.data.wv import NormalizedDatasetOnTheFly
from neuralop.flood.eval.checkpoints import _discover_checkpoint_runs, _load_models_from_runs, _preferred_checkpoint_alias
from neuralop.flood.eval.datasets import (
    _build_one_step_datasets,
    _build_rollout_normalized_dataset,
    _build_test_loader,
    _load_structural_dry_artifact_for_eval,
    _load_or_fit_normalizers,
    _set_dataset_structural_dry_mask,
)
from neuralop.flood.eval.metrics import _build_eval_losses, _is_gaussian_mode
from neuralop.flood.eval.mc_dropout import (
    evaluate_mc_dropout_one_step,
    validate_mc_dropout_config,
)
from neuralop.flood.eval.rollout import (
    _make_trainer,
    _rollout_prediction_generic,
    _rollout_prediction_per_hydrograph,
)
from neuralop.flood.eval.runtime import (
    _PhaseTimer,
    _get_cli_arg_value,
    _opt,
    _opt_float,
    _parse_args,
    _resolve_cli_config_path,
    _resolve_device,
    _validate_args,
    DEFAULT_EVAL_LOG,
    normalize_rollout_init_mode,
)
from neuralop.flood.processing.wv import FloodGINODataProcessor
from neuralop.flood.utils.runtime import (
    load_config_and_setup,
    normalize_fgn_ar_state_update,
    normalize_fgn_latent_temporal_mode,
    set_seed,
    setup_logging,
    write_train_txt_from_data_root,
)


def _resolve_rollout_length_for_evaluation(config, rollout_dataset, logger) -> int:
    """Resolve -1 to the full available rollout horizon; positive values stay fixed."""
    configured = int(config.data.rollout_length)
    if configured != -1:
        logger.info(
            "Using configured data.rollout_length=%d.",
            configured,
        )
        return configured
    available = getattr(rollout_dataset, "available_rollout_length", None)
    if available is None:
        raise ValueError(
            "data.rollout_length=-1 requested full-length rollout, but dataset did not expose "
            "an available horizon."
        )
    available = int(available)
    if available < 1:
        raise ValueError(
            "data.rollout_length=-1 requested full-length rollout, but available forecast horizon is < 1 step."
        )
    logger.info(
        "Resolved data.rollout_length=-1 to full available rollout horizon=%d steps.",
        available,
    )
    return available


def _normalize_config_rollout_out_dir(config) -> Path:
    """Normalize rollout.out_dir once so every output writer sees the same path."""
    raw_out_dir = _opt(config, "rollout", "out_dir", "rollout_outputs")
    out_dir = Path(str(raw_out_dir)).expanduser()
    if not out_dir.is_absolute():
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(config, "rollout"):
        setattr(config.rollout, "out_dir", str(out_dir))
    return out_dir


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _optional_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    """Run post-training one-step and/or rollout evaluation."""
    args = _parse_args()
    _validate_args(args)
    cli_config_path = _resolve_cli_config_path(_get_cli_arg_value("--config_path"))
    config, device, is_logger = load_config_and_setup()
    device = _resolve_device(device)
    seed = _opt(config, "distributed", "seed", 123)
    deterministic = _opt(config, None, "deterministic", True)
    set_seed(seed, deterministic=deterministic)

    checkpoint_path = Path(_opt(config, "checkpoint", "save_dir", "."))
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    rollout_out_dir = _normalize_config_rollout_out_dir(config)
    forecast_artifact_dir = _optional_path(_opt(config, "rollout", "forecast_artifact_dir", None))
    write_visualizations = _as_bool(_opt(config, "rollout", "write_visualizations", True), True)
    preferred_checkpoint_alias = _preferred_checkpoint_alias(config)
    try:
        checkpoint_runs = _discover_checkpoint_runs(
            checkpoint_path, preferred_alias=preferred_checkpoint_alias
        )
    except TypeError as exc:
        if "preferred_alias" not in str(exc):
            raise
        checkpoint_runs = _discover_checkpoint_runs(checkpoint_path)
    primary_dir, primary_alias, _ = checkpoint_runs[0]
    eval_log = Path(args.eval_log_file).expanduser()
    if args.eval_log_file == DEFAULT_EVAL_LOG:
        eval_log = rollout_out_dir / DEFAULT_EVAL_LOG
    elif not eval_log.is_absolute():
        eval_log = primary_dir / eval_log
    logger = setup_logging(
        log_level=_opt(config, None, "log_level", "INFO"),
        log_file=str(eval_log),
        logger_name="flood_eval",
    )
    if is_logger:
        logger.info("Post-training evaluation started")
        logger.info("Device=%s | Seed=%s | Deterministic=%s", device, seed, deterministic)
        if cli_config_path is not None:
            logger.info("Config source (--config_path): %s", cli_config_path)
        else:
            logger.warning(
                "No --config_path provided. Evaluator is using the training script default config path."
            )
        logger.info(
            "Checkpoint path=%s | discovered models=%d",
            checkpoint_path,
            len(checkpoint_runs),
        )
        logger.info("Rollout output directory=%s", config.rollout.out_dir)
        logger.info("data.root=%s", _opt(config, "data", "root", "N/A"))

    out_dist = str(_opt(config, "gino", "output_distribution", "deterministic")).strip().lower()
    train_loss_name = str(_opt(config, "opt", "training_loss", "l2")).strip().lower()
    use_fgn_cfg = bool(_opt(config, "gino", "use_fgn_noise", False))
    mc_dropout_cfg = validate_mc_dropout_config(config)
    if mc_dropout_cfg.enabled and is_logger:
        logger.info(
            "MC-dropout evaluation enabled: samples=%s dropout_probability=%.6f seed=%s",
            mc_dropout_cfg.samples,
            mc_dropout_cfg.dropout_probability,
            mc_dropout_cfg.seed,
        )
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
        _opt(config, "gino", "fgn_latent_temporal_mode", "stepwise")
    )
    fgn_ar_state_update = normalize_fgn_ar_state_update(
        _opt(config, "opt", "fgn_ar_state_update", "mean_feedback")
    )
    if hasattr(config, "gino"):
        setattr(config.gino, "fgn_latent_temporal_mode", fgn_latent_temporal_mode)
    if hasattr(config, "opt"):
        setattr(config.opt, "fgn_ar_state_update", fgn_ar_state_update)
    if train_loss_name == "gaussian_nll" and out_dist != "gaussian":
        raise ValueError(
            "training_loss='gaussian_nll' requires gino.output_distribution='gaussian'."
        )
    if out_dist == "gaussian" and use_fgn_cfg:
        raise ValueError(
            "gino.output_distribution='gaussian' requires gino.use_fgn_noise=false."
        )

    if _opt(config, "data", "write_train_txt", False):
        with _PhaseTimer(logger, "Refreshing train.txt"):
            run_ids = write_train_txt_from_data_root(
                config.data.root,
                train_txt=_opt(config, "data", "train_txt", "train.txt"),
                hdf_suffix=".hdf",
            )
        logger.info("train.txt refreshed: %d run IDs", len(run_ids))

    train_raw, test_raw, target_variables = _build_one_step_datasets(
        config, seed, logger
    )
    normalizers, normalizer_path = _load_or_fit_normalizers(config, train_raw, primary_dir, logger)
    structural_dry_policy, structural_dry_artifact = _load_structural_dry_artifact_for_eval(
        config,
        normalizer_path=normalizer_path,
        expected_cell_count=getattr(getattr(train_raw, "dataset", train_raw), "reference_cell_count", None),
        expected_run_ids=getattr(getattr(train_raw, "dataset", train_raw), "run_ids", None),
        logger=logger,
    )
    if structural_dry_artifact is not None:
        _set_dataset_structural_dry_mask(train_raw, structural_dry_artifact["dry_mask"])
        _set_dataset_structural_dry_mask(test_raw, structural_dry_artifact["dry_mask"])
    with _PhaseTimer(logger, "Wrapping normalized datasets"):
        train_norm = NormalizedDatasetOnTheFly(
            train_raw, normalizers, query_res=config.data.query_res
        )
        test_norm = NormalizedDatasetOnTheFly(
            test_raw, normalizers, query_res=config.data.query_res
        )
    test_loader = _build_test_loader(test_norm, config.data.batch_size)
    logger.info(
        "Test loader: batch_size=%s batches=%d",
        config.data.batch_size, len(test_loader),
    )

    with _PhaseTimer(logger, "Loading model checkpoint(s)"):
        models = _load_models_from_runs(config, device, checkpoint_runs, logger)
    inverse_test = _opt(config, None, "inverse_test", True)
    if normalizers.get("target") is not None:
        normalizers["target"] = normalizers["target"].to(device)
    use_fgn = _opt(config, "gino", "use_fgn_noise", False)
    gaussian_mode = _is_gaussian_mode(config)
    if gaussian_mode:
        for model_idx, model in enumerate(models):
            model_fno_norm = getattr(model, "fno_norm", None)
            model_fno_norm = "none" if model_fno_norm is None else str(model_fno_norm).strip().lower()
            if model_fno_norm != "instance_norm":
                raise ValueError(
                    "Gaussian evaluation expects checkpoints trained with "
                    "gino.fno_norm='instance_norm'. "
                    f"Model[{model_idx}] has fno_norm={model_fno_norm!r}. "
                    "Train/use Gaussian checkpoints with gino.fno_norm set to instance_norm."
                )
    try:
        eval_losses = _build_eval_losses(
            config, use_fgn, target_variables=target_variables
        )
    except TypeError as exc:
        if "target_variables" not in str(exc):
            raise
        eval_losses = _build_eval_losses(config, use_fgn)
    logger.info(
        "Eval losses=%s inverse_test=%s n_models=%d gaussian_mode=%s",
        list(eval_losses.keys()),
        inverse_test,
        len(models),
        gaussian_mode,
    )
    if use_fgn:
        logger.info(
            "FGN settings: latent_temporal_mode=%s, ar_state_update=%s, crps_n_samples=%s, fgn_noise_dim=%s",
            fgn_latent_temporal_mode,
            fgn_ar_state_update,
            max(2, int(_opt(config, "opt", "crps_n_samples", 2))),
            int(_opt(config, "gino", "fgn_noise_dim", 32)),
        )

    run_single = args.run_single_step and not args.skip_single_step
    if run_single:
        model_metrics: List[Dict[str, float]] = []
        with _PhaseTimer(logger, "One-step evaluation (all models)"):
            for model_idx, model in enumerate(models):
                data_processor = FloodGINODataProcessor(
                    device=device,
                    target_norm=normalizers.get("target"),
                    inverse_test=inverse_test,
                    output_distribution=str(
                        _opt(config, "gino", "output_distribution", "deterministic")
                    ).strip().lower(),
                )
                data_processor.wrap(model)
                trainer = _make_trainer(config, model, data_processor, device, logger)
                if mc_dropout_cfg.enabled:
                    metrics = evaluate_mc_dropout_one_step(
                        model=model,
                        data_processor=data_processor,
                        data_loader=test_loader,
                        eval_losses=eval_losses,
                        config=config,
                        device=device,
                        logger=logger,
                        log_prefix=f"test_m{model_idx}",
                        model_idx=model_idx,
                    )
                else:
                    metrics = trainer.evaluate(
                        eval_losses, test_loader, log_prefix=f"test_m{model_idx}"
                    )
                clean = {
                    k: float(v.item() if hasattr(v, "item") else v)
                    for k, v in metrics.items()
                    if not k.endswith("_outputs")
                }
                model_metrics.append(clean)
                logger.info("One-step metrics model[%d]: %s", model_idx, clean)
        logger.info("One-step TEST metrics summary across %d models:", len(model_metrics))
        common_keys = sorted(set().union(*[m.keys() for m in model_metrics]))
        for key in common_keys:
            vals = [m[key] for m in model_metrics if key in m]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            logger.info("  %s: mean=%.6e std=%.6e", key, mean_v, std_v)
            print(f"  {key}: mean={mean_v:.6e} std={std_v:.6e}")

    run_rollout_cfg = bool(_opt(config, "rollout", "run_after_training", False))
    run_rollout = (run_rollout_cfg or args.run_rollout) and not args.skip_rollout
    if not run_rollout:
        logger.info(
            "Skipping rollout (run_after_training=false and --run_rollout not set)."
        )
        return 0

    rollout_init_mode = normalize_rollout_init_mode(
        _opt(config, "rollout", "init_mode", "mean_history")
    )
    logger.info("Rollout initialization mode=%s", rollout_init_mode)
    rollout_n_ensemble = int(_opt(config, "rollout", "n_ensemble_samples", 1))
    gaussian_state_update_cfg = str(
        _opt(config, "rollout", "gaussian_state_update", "sample")
    ).strip().lower()
    gaussian_state_update = (
        str(args.gaussian_state_update).strip().lower()
        if args.gaussian_state_update is not None
        else gaussian_state_update_cfg
    )
    if gaussian_mode and gaussian_state_update not in {"sample", "mu"}:
        raise ValueError(
            "Invalid Gaussian rollout state update mode. Expected one of "
            "{'sample', 'mu'}, got "
            f"{gaussian_state_update!r}."
        )
    if gaussian_mode:
        logger.info(
            "Gaussian rollout state update mode resolved to '%s' (%s).",
            gaussian_state_update,
            "CLI override" if args.gaussian_state_update is not None else "config",
        )
    ens_per_model = _opt(config, "rollout", "n_ensemble_samples_per_model", None)
    if mc_dropout_cfg.enabled:
        rollout_n_ensemble = int(mc_dropout_cfg.samples)
        ens_per_model = None
        logger.info(
            "Using MC-dropout rollout ensemble members=%d (uq.mc_samples); temporal_mode=stepwise.",
            rollout_n_ensemble,
        )
    if ens_per_model is not None:
        ens_per_model_int = int(ens_per_model)
        if ens_per_model_int < 1:
            raise ValueError(
                "rollout.n_ensemble_samples_per_model must be >= 1 "
                f"(got {ens_per_model_int})."
            )
        rollout_n_ensemble = ens_per_model_int * max(1, len(models))
        logger.info(
            "Using rollout.n_ensemble_samples_per_model=%d with n_models=%d => total ensemble members=%d",
            ens_per_model_int,
            len(models),
            rollout_n_ensemble,
        )
    else:
        logger.info(
            "Using rollout.n_ensemble_samples=%d total members across %d model(s).",
            rollout_n_ensemble,
            len(models),
        )

    rollout_norm_ds, hydrograph_samples = _build_rollout_normalized_dataset(
        config,
        normalizers,
        target_variables,
        logger,
        structural_dry_artifact=structural_dry_artifact,
    )
    effective_rollout_length = _resolve_rollout_length_for_evaluation(
        config,
        rollout_norm_ds,
        logger,
    )
    logger.info("Rollout normalized dataset: %d runs", len(rollout_norm_ds))
    if hydrograph_samples:
        logger.info(
            "Hydrograph-grouped mode enabled: %d hydrographs with reference ensembles.",
            len(hydrograph_samples),
        )
    if forecast_artifact_dir is not None:
        logger.info("Forecast-member HDF5 artifact writing enabled: %s", forecast_artifact_dir)
    if not write_visualizations:
        logger.info("Rollout visualization rendering disabled by rollout.write_visualizations=false.")
    with _PhaseTimer(logger, "Rollout evaluation + plotting"):
        if hydrograph_samples:
            _rollout_prediction_per_hydrograph(
                models=models,
                hydrograph_samples=hydrograph_samples,
                rollout_length=effective_rollout_length,
                history_steps=config.data.n_history,
                dynamic_norm=normalizers["dynamic"],
                target_norm=normalizers["target"],
                device=device,
                skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                dt=config.data.dt,
                out_dir=config.rollout.out_dir,
                target_variables=target_variables,
                logger=logger,
                fgn_noise_dim=_opt(config, "gino", "fgn_noise_dim", 32) if use_fgn else None,
                n_ensemble_samples=rollout_n_ensemble,
                fgn_latent_temporal_mode=fgn_latent_temporal_mode,
                fgn_ar_state_update=fgn_ar_state_update,
                gaussian_mode=gaussian_mode,
                gaussian_min_logvar=_opt_float(config, "opt", "gaussian_min_logvar", -9.0),
                gaussian_max_logvar=_opt_float(config, "opt", "gaussian_max_logvar", 4.0),
                gaussian_state_update=gaussian_state_update,
                rollout_init_mode=rollout_init_mode,
                mc_dropout_enabled=mc_dropout_cfg.enabled,
                mc_dropout_seed=mc_dropout_cfg.seed,
                visualization_config=_opt(config, None, "visualization", None),
                impact_metrics_config=_opt(config, "rollout", "impact_metrics", None),
                forecast_artifact_dir=str(forecast_artifact_dir) if forecast_artifact_dir is not None else None,
                write_visualizations=write_visualizations,
                member_boundary_mode=_opt(config, "rollout", "member_boundary_mode", "shared"),
            )
        else:
            _rollout_prediction_generic(
                models=models,
                rollout_dataset=rollout_norm_ds,
                rollout_length=effective_rollout_length,
                history_steps=config.data.n_history,
                dynamic_norm=normalizers["dynamic"],
                target_norm=normalizers["target"],
                device=device,
                skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                dt=config.data.dt,
                out_dir=config.rollout.out_dir,
                target_variables=target_variables,
                logger=logger,
                fgn_noise_dim=_opt(config, "gino", "fgn_noise_dim", 32) if use_fgn else None,
                n_ensemble_samples=rollout_n_ensemble,
                fgn_latent_temporal_mode=fgn_latent_temporal_mode,
                fgn_ar_state_update=fgn_ar_state_update,
                gaussian_mode=gaussian_mode,
                gaussian_min_logvar=_opt_float(config, "opt", "gaussian_min_logvar", -9.0),
                gaussian_max_logvar=_opt_float(config, "opt", "gaussian_max_logvar", 4.0),
                gaussian_state_update=gaussian_state_update,
                mc_dropout_enabled=mc_dropout_cfg.enabled,
                mc_dropout_seed=mc_dropout_cfg.seed,
                visualization_config=_opt(config, None, "visualization", None),
            )
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
