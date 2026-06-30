"""Canonical calibrated flood operator evaluation application."""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List

from neuralop.flood.data.wv import NormalizedDatasetOnTheFly
from neuralop.flood.eval.scientific_calibration import (
    CALIBRATION_COMPARISON_JSON,
    ARTIFACT_FORMAT,
    CalibrationBins,
    build_calibration_comparison,
    compute_artifact_uq_metrics,
    fit_crps_member_by_member_from_artifacts,
    fit_exceedance_isotonic_from_artifacts,
    save_fit_diagnostics,
    list_forecast_artifacts,
    load_forecast_artifact,
    save_crps_mbm_coefficients,
    save_exceedance_isotonic,
    save_metrics_json,
    validate_reference_split_no_leakage,
    validate_scientific_calibration_config,
)
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
from neuralop.flood.eval.rollout import (
    _make_trainer,
    _rollout_prediction_generic,
    _rollout_prediction_per_hydrograph,
)
from neuralop.flood.eval.runtime import (
    UQ_OVERALL_JSON,
    _PhaseTimer,
    _get_cli_arg_value,
    _opt,
    _opt_float,
    _parse_args,
    _resolve_cli_config_path,
    _resolve_device,
    _validate_args,
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


def _resolve_split_guard_root(config, section: str) -> Path:
    root_raw = str(_opt(config, section, "root", "")).strip()
    root_path = Path(root_raw) if root_raw else Path()
    if section == "rollout_calibration" and (not root_raw or not root_path.exists()):
        fallback_root = str(_opt(config, "rollout_data", "root", "")).strip()
        if not fallback_root:
            raise FileNotFoundError(
                "rollout_calibration.root is unset and rollout_data.root is unavailable."
            )
        return Path(fallback_root).resolve()
    return root_path.resolve()


def _nested_get(obj, *keys, default=None):
    cur = obj
    for key in keys:
        if cur is None:
            return default
        try:
            cur = getattr(cur, key)
            continue
        except (AttributeError, KeyError, TypeError):
            pass
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
            continue
        try:
            cur = cur[key]
        except Exception:
            return default
    return cur


def _set_cfg_value(obj, key: str, value) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _expand_run_path(value, *, default: str | None = None) -> Path:
    raw = default if value in (None, "") else str(value)
    if raw is None or str(raw).strip() == "":
        raise ValueError("Expected a non-empty path.")
    return Path(os.path.expandvars(str(raw))).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, float]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    checkpoint_runs = _discover_checkpoint_runs(checkpoint_path, preferred_alias=_preferred_checkpoint_alias(config))
    primary_dir, primary_alias, _ = checkpoint_runs[0]
    eval_log = Path(args.eval_log_file)
    if not eval_log.is_absolute():
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
        logger.info("data.root=%s", _opt(config, "data", "root", "N/A"))

    out_dist = str(_opt(config, "gino", "output_distribution", "deterministic")).strip().lower()
    train_loss_name = str(_opt(config, "opt", "training_loss", "l2")).strip().lower()
    use_fgn_cfg = bool(_opt(config, "gino", "use_fgn_noise", False))
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
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
        _opt(config, "gino", "fgn_latent_temporal_mode", "stepwise")
    )
    fgn_ar_state_update = normalize_fgn_ar_state_update(
        _opt(config, "opt", "fgn_ar_state_update", "mean_feedback")
    )
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
    eval_losses = _build_eval_losses(config, use_fgn)
    logger.info(
        "Eval losses=%s inverse_test=%s n_models=%d gaussian_mode=%s",
        list(eval_losses.keys()),
        inverse_test,
        len(models),
        gaussian_mode,
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

    rollout_cal_enabled = bool(_opt(config, "rollout_calibration", "enabled", True))
    if rollout_cal_enabled:
        validate_scientific_calibration_config(config)
        validate_reference_split_no_leakage(config)
        if "wd" not in target_variables:
            raise ValueError(
                "rollout_calibration.enabled=true requires WD channel in target_variables."
            )

        calibration_cfg = _nested_get(config, "rollout_calibration")
        reference_cfg = _nested_get(calibration_cfg, "reference")
        artifact_cfg = _nested_get(calibration_cfg, "forecast_artifacts")
        optimizer_cfg = _nested_get(calibration_cfg, "optimizer")
        exceedance_cfg = _nested_get(calibration_cfg, "exceedance")

        calibration_root = _expand_run_path(_nested_get(reference_cfg, "calibration_root"))
        calibration_txt = str(_nested_get(reference_cfg, "calibration_txt"))
        heldout_test_root = _expand_run_path(_nested_get(reference_cfg, "test_root"))
        heldout_test_txt = str(_nested_get(reference_cfg, "test_txt"))
        min_ref_members = int(_nested_get(reference_cfg, "min_reference_members_per_family", default=100))
        regenerate_calibrated_visuals = bool(_nested_get(calibration_cfg, "regenerate_calibrated_visuals", default=False))
        artifact_root = _expand_run_path(
            _nested_get(artifact_cfg, "root", default=None),
            default=str(Path(config.rollout.out_dir) / "forecast_artifacts"),
        )
        artifact_format = str(_nested_get(artifact_cfg, "format", default=ARTIFACT_FORMAT))
        if artifact_format != ARTIFACT_FORMAT:
            raise ValueError(
                f"Unsupported forecast artifact format {artifact_format!r}; expected {ARTIFACT_FORMAT!r}."
            )
        calib_artifact_dir = artifact_root / "calibration"
        test_artifact_dir = artifact_root / "test_raw"
        calibration_diagnostics_dir = str(Path(config.rollout.out_dir) / "calibration_fit_diagnostics")

        logger.info(
            "Scientific calibration enabled: method=crps_member_by_member calibration_root=%s calibration_txt=%s test_root=%s test_txt=%s artifact_root=%s",
            calibration_root,
            calibration_txt,
            heldout_test_root,
            heldout_test_txt,
            artifact_root,
        )

        # The rollout dataset builder still expects flat section fields. Resolve the
        # new leakage-safe nested reference contract into those runtime fields.
        _set_cfg_value(config.rollout_data, "root", str(heldout_test_root))
        _set_cfg_value(config.rollout_data, "test_txt", heldout_test_txt)
        _set_cfg_value(calibration_cfg, "root", str(calibration_root))
        _set_cfg_value(calibration_cfg, "test_txt", calibration_txt)
        if _nested_get(calibration_cfg, "static_text_files", default=None) is None:
            _set_cfg_value(
                calibration_cfg,
                "static_text_files",
                _opt(config, "rollout_data", "static_text_files", _opt(config, "data", "static_text_files", [])),
            )

        rollout_norm_ds_test, hydrograph_samples_test = _build_rollout_normalized_dataset(
            config,
            normalizers,
            target_variables,
            logger,
            structural_dry_artifact=structural_dry_artifact,
            split_txt=heldout_test_txt,
            split_name="test",
            config_section="rollout_data",
        )
        logger.info("Held-out rollout test dataset: %d runs", len(rollout_norm_ds_test))
        if not hydrograph_samples_test:
            raise ValueError(
                "Scientific calibration requires grouped hydrograph test data with multiple reference simulations per family."
            )

        rollout_norm_ds_calib, hydrograph_samples_calib = _build_rollout_normalized_dataset(
            config,
            normalizers,
            target_variables,
            logger,
            structural_dry_artifact=structural_dry_artifact,
            split_txt=calibration_txt,
            split_name="val",
            config_section="rollout_calibration",
        )
        logger.info("Calibration rollout dataset: %d runs", len(rollout_norm_ds_calib))
        if not hydrograph_samples_calib:
            raise ValueError(
                "Scientific calibration split has no grouped hydrograph samples; expected >=2 reference simulations per family."
            )

        with _PhaseTimer(logger, "Generating calibration forecast-member artifacts"):
            _rollout_prediction_per_hydrograph(
                models=models,
                hydrograph_samples=hydrograph_samples_calib,
                rollout_length=config.data.rollout_length,
                history_steps=config.data.n_history,
                dynamic_norm=normalizers["dynamic"],
                target_norm=normalizers["target"],
                device=device,
                skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                dt=config.data.dt,
                out_dir=calibration_diagnostics_dir,
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
                visualization_config=_opt(config, None, "visualization", None),
                impact_metrics_config=_opt(config, "rollout", "impact_metrics", None),
                forecast_artifact_dir=str(calib_artifact_dir),
                calibration_metadata={
                    "artifact_role": "calibration_fit",
                    "calibration_root": str(calibration_root),
                    "calibration_txt": calibration_txt,
                },
            )

        calib_artifacts = list_forecast_artifacts(calib_artifact_dir)
        for artifact_path in calib_artifacts:
            artifact_meta = load_forecast_artifact(artifact_path, load_members=False)
            if int(artifact_meta["n_reference_members"]) < min_ref_members:
                raise ValueError(
                    f"Calibration artifact {artifact_path} has only {artifact_meta['n_reference_members']} reference members; "
                    f"required >= {min_ref_members}."
                )

        bins = CalibrationBins.from_config(config)
        bounds = _nested_get(optimizer_cfg, "bounds", default=None) or {
            "a_m": [-2.0, 2.0],
            "beta": [0.2, 2.5],
            "gamma": [0.05, 3.0],
        }
        with _PhaseTimer(logger, "Fitting CRPS member-by-member WD calibration"):
            calibration_model = fit_crps_member_by_member_from_artifacts(
                calib_artifacts,
                bins=bins,
                max_fit_points_per_bin=int(_nested_get(optimizer_cfg, "max_fit_points_per_bin", default=1000000)),
                min_fit_points_per_bin=int(_nested_get(optimizer_cfg, "min_fit_points_per_bin", default=1000)),
                seed=int(_nested_get(optimizer_cfg, "seed", default=123)),
                bounds=bounds,
                objective=str(_nested_get(optimizer_cfg, "objective", default="empirical_crps")),
                tail_threshold_m=float(_nested_get(optimizer_cfg, "tail_threshold_m", default=0.30)),
                tail_weight=float(_nested_get(optimizer_cfg, "tail_weight", default=4.0)),
                mean_rmse_weight=float(_nested_get(optimizer_cfg, "mean_rmse_weight", default=0.0)),
                spread_ratio_weight=float(_nested_get(optimizer_cfg, "spread_ratio_weight", default=0.0)),
                target_spread_ratio=float(_nested_get(optimizer_cfg, "target_spread_ratio", default=1.0)),
                multistart=bool(_nested_get(optimizer_cfg, "multistart", default=True)),
            )
        calibration_dir = Path(config.rollout.out_dir) / "calibration"
        coeff_path = save_crps_mbm_coefficients(calibration_model, calibration_dir)
        diag_path = save_fit_diagnostics(calibration_model, calibration_dir)
        logger.info("Saved CRPS MBM coefficients to %s", coeff_path)
        logger.info("Saved CRPS MBM fit diagnostics to %s", diag_path)
        for warning in calibration_model.get("diagnostics", {}).get("warnings", []):
            logger.warning("Calibration fit diagnostic warning: %s", warning)

        thresholds_m = _nested_get(exceedance_cfg, "thresholds_m", default=[0.01, 0.05, 0.10, 0.30, 0.50])
        if str(_nested_get(exceedance_cfg, "method", default="isotonic")).strip().lower() == "isotonic":
            with _PhaseTimer(logger, "Fitting isotonic exceedance calibration"):
                exceedance_model = fit_exceedance_isotonic_from_artifacts(
                    calib_artifacts,
                    bins=bins,
                    wet_frequency_by_cell=calibration_model["wet_frequency_by_cell"],
                    thresholds_m=thresholds_m,
                    min_fit_points_per_bin=int(_nested_get(exceedance_cfg, "min_fit_points_per_bin", default=128)),
                    calibration_model=calibration_model,
                )
            iso_path = save_exceedance_isotonic(exceedance_model, calibration_dir)
            logger.info("Saved isotonic exceedance curves to %s", iso_path)
        else:
            logger.warning("Exceedance calibration disabled or unsupported; skipping isotonic probability calibration.")

        raw_out_dir = os.path.join(config.rollout.out_dir, "raw")
        calibrated_out_dir = os.path.join(config.rollout.out_dir, "calibrated")
        with _PhaseTimer(logger, "Rollout evaluation + plotting (raw held-out test)"):
            _rollout_prediction_per_hydrograph(
                models=models,
                hydrograph_samples=hydrograph_samples_test,
                rollout_length=config.data.rollout_length,
                history_steps=config.data.n_history,
                dynamic_norm=normalizers["dynamic"],
                target_norm=normalizers["target"],
                device=device,
                skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                dt=config.data.dt,
                out_dir=raw_out_dir,
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
                visualization_config=_opt(config, None, "visualization", None),
                impact_metrics_config=_opt(config, "rollout", "impact_metrics", None),
                forecast_artifact_dir=str(test_artifact_dir),
                calibration_metadata={
                    "artifact_role": "heldout_test_raw",
                    "test_root": str(heldout_test_root),
                    "test_txt": heldout_test_txt,
                },
            )

        test_artifacts = list_forecast_artifacts(test_artifact_dir)
        for artifact_path in test_artifacts:
            artifact_meta = load_forecast_artifact(artifact_path, load_members=False)
            if int(artifact_meta["n_reference_members"]) < min_ref_members:
                raise ValueError(
                    f"Held-out test artifact {artifact_path} has only {artifact_meta['n_reference_members']} reference members; "
                    f"required >= {min_ref_members}."
                )

        if regenerate_calibrated_visuals:
            with _PhaseTimer(logger, "Rollout evaluation + plotting (CRPS-MBM calibrated held-out test)"):
                _rollout_prediction_per_hydrograph(
                    models=models,
                    hydrograph_samples=hydrograph_samples_test,
                    rollout_length=config.data.rollout_length,
                    history_steps=config.data.n_history,
                    dynamic_norm=normalizers["dynamic"],
                    target_norm=normalizers["target"],
                    device=device,
                    skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                    dt=config.data.dt,
                    out_dir=calibrated_out_dir,
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
                    visualization_config=_opt(config, None, "visualization", None),
                    impact_metrics_config=_opt(config, "rollout", "impact_metrics", None),
                    calibration_model=calibration_model,
                    calibration_metadata={
                        "artifact_role": "heldout_test_calibrated",
                        "coefficient_path": str(coeff_path),
                    },
                )
        else:
            logger.info("Skipping calibrated rollout regeneration; calibrated metrics will be computed from held-out test artifacts.")

        apply_isotonic = bool(_nested_get(exceedance_cfg, "apply_isotonic", default=True))
        raw_artifact_metrics = compute_artifact_uq_metrics(test_artifacts, thresholds_m=thresholds_m)
        calibrated_artifact_metrics = compute_artifact_uq_metrics(
            test_artifacts,
            calibration_model=calibration_model,
            isotonic_model=exceedance_model if "exceedance_model" in locals() else None,
            apply_isotonic=apply_isotonic,
            thresholds_m=thresholds_m,
        )
        save_metrics_json(raw_artifact_metrics, calibration_dir / "artifact_raw_uq_overall_metrics.json")
        save_metrics_json(calibrated_artifact_metrics, calibration_dir / "artifact_calibrated_uq_overall_metrics.json")

        raw_metrics_path = Path(raw_out_dir) / UQ_OVERALL_JSON
        calibrated_metrics_path = Path(calibrated_out_dir) / UQ_OVERALL_JSON
        if regenerate_calibrated_visuals and raw_metrics_path.exists() and calibrated_metrics_path.exists():
            comparison = build_calibration_comparison(
                _load_json(raw_metrics_path),
                _load_json(calibrated_metrics_path),
            )
        else:
            comparison = build_calibration_comparison(raw_artifact_metrics, calibrated_artifact_metrics)
        comparison.update(
            {
                "method": "crps_member_by_member",
                "raw_metrics_json": str(raw_metrics_path),
                "calibrated_metrics_json": str(calibrated_metrics_path),
                "artifact_raw_metrics_json": str(calibration_dir / "artifact_raw_uq_overall_metrics.json"),
                "artifact_calibrated_metrics_json": str(calibration_dir / "artifact_calibrated_uq_overall_metrics.json"),
                "coefficient_json": str(coeff_path),
                "calibration_artifact_dir": str(calib_artifact_dir),
                "heldout_test_artifact_dir": str(test_artifact_dir),
            }
        )
        comparison_path = Path(config.rollout.out_dir) / CALIBRATION_COMPARISON_JSON
        save_metrics_json(comparison, comparison_path)
        logger.info("Saved scientific calibration comparison JSON to %s", comparison_path)
    else:
        rollout_norm_ds, hydrograph_samples = _build_rollout_normalized_dataset(
            config,
            normalizers,
            target_variables,
            logger,
            structural_dry_artifact=structural_dry_artifact,
            config_section="rollout_data",
        )
        logger.info("Rollout normalized dataset: %d runs", len(rollout_norm_ds))
        if hydrograph_samples:
            logger.info(
                "Hydrograph-grouped mode enabled: %d hydrographs with reference ensembles.",
                len(hydrograph_samples),
            )
        with _PhaseTimer(logger, "Rollout evaluation + plotting"):
            if hydrograph_samples:
                _rollout_prediction_per_hydrograph(
                    models=models,
                    hydrograph_samples=hydrograph_samples,
                    rollout_length=config.data.rollout_length,
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
                    calibration_coeffs_wd=None,
                    visualization_config=_opt(config, None, "visualization", None),
                    impact_metrics_config=_opt(config, "rollout", "impact_metrics", None),
                )
            else:
                _rollout_prediction_generic(
                    models=models,
                    rollout_dataset=rollout_norm_ds,
                    rollout_length=config.data.rollout_length,
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
                    visualization_config=_opt(config, None, "visualization", None),
                )
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
