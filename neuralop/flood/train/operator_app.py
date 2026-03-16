"""Canonical WV flood operator training application."""

from __future__ import annotations

import logging
import os
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from neuralop import get_model
from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers
from neuralop.flood.data.structural_dry import (
    build_structural_dry_artifact,
    load_structural_dry_artifact,
    save_structural_dry_artifact,
    validate_structural_dry_artifact,
)
from neuralop.flood.data.wv import (
    FloodDatasetHDF,
    FloodRolloutTestDatasetHDF,
    NormalizedDatasetOnTheFly,
    NormalizedRolloutTestDataset,
    fit_normalizers_streaming,
)
from neuralop.flood.losses import (
    FloodDryBackgroundFalseWetRate,
    FloodDryBackgroundMAE,
    FloodDryBackgroundRMSE,
    FloodEnsembleDryPredStdMean,
    FloodGaussianDryPredStdMean,
    FloodMaskedAbsLpLoss,
    FloodMaskedCRPSLoss,
    FloodMaskedGaussianNLLLoss,
    FloodMaskedRelLpLoss,
)
from neuralop.flood.processing.wv import (
    FloodGINODataProcessor,
    compute_hazard_proxy_pooled,
    get_flood_crps_weights,
)
from neuralop.flood.train.debug import overfit_sanity_check, verify_training_gradient_flow
from neuralop.flood.train.fgn import FGNTrainer
from neuralop.flood.train.gaussian import GaussianNLLTrainer
from neuralop.flood.train.rollout import rollout_prediction
from neuralop.flood.utils.runtime_core import (
    _cfg_get,
    get_dataset_boundary_kwargs,
    get_structural_dry_policy_kwargs,
    _is_power_of_two,
    _safe_float,
    _safe_int,
    _to_builtin,
    dataloader_worker_init,
    load_config_and_setup,
    make_dataloader_generator,
    make_split_generator,
    normalize_fgn_ar_state_update,
    normalize_fgn_latent_temporal_mode,
    parse_target_variables,
    save_effective_config_snapshot,
    set_seed,
    setup_logging,
    wait_for_structural_dry_artifact,
    write_train_txt_from_data_root,
)
from neuralop.losses.data_losses import LpLoss
from neuralop.losses.probabilistic_losses import CRPSLoss, GaussianNLLLoss, fair_crps_univariate
from neuralop.training import AdamW
from neuralop.training.trainer import Trainer
from neuralop.utils import get_wandb_api_key


def _resolve_training_normalizer_path(config):
    normalizer_path = _cfg_get(config.data, "normalizer_path", None)
    if normalizer_path is None:
        return None
    normalizer_path = Path(normalizer_path)
    if not normalizer_path.is_absolute():
        normalizer_root = _cfg_get(config.data, "normalizer_root", None)
        if normalizer_root is not None:
            normalizer_path = Path(str(normalizer_root)) / normalizer_path
        else:
            normalizer_path = Path(config.data.root) / normalizer_path
    return normalizer_path.resolve()


def _prepare_structural_dry_artifact_for_training(
    config,
    *,
    dataset,
    normalizer_path,
    logger,
    use_distributed,
    global_rank,
):
    policy_kwargs = get_structural_dry_policy_kwargs(
        config,
        normalizer_path=normalizer_path,
        allow_data_root_fallback=True,
    )
    if policy_kwargs["policy"] != "masked_primary":
        return policy_kwargs, None

    artifact_path = policy_kwargs["artifact_path"]
    summary_path = policy_kwargs["summary_path"]
    artifact = None
    if artifact_path.exists():
        artifact = load_structural_dry_artifact(artifact_path)
    elif use_distributed and global_rank != 0:
        artifact = wait_for_structural_dry_artifact(artifact_path)
    else:
        artifact = build_structural_dry_artifact(
            data_root=config.data.root,
            run_ids=dataset.run_ids,
            train_txt=_cfg_get(config.data, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
            hdf_paths=dataset.hdf_paths,
            cell_point_index=dataset.cell_point_index,
            mask_definition=policy_kwargs["mask_definition"],
        )
        save_structural_dry_artifact(
            artifact,
            artifact_path=artifact_path,
            summary_path=summary_path,
        )
    if use_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        if artifact is None:
            artifact = load_structural_dry_artifact(artifact_path)

    artifact = validate_structural_dry_artifact(
        artifact,
        expected_cell_count=dataset.reference_cell_count,
        expected_run_ids=dataset.run_ids,
    )
    dataset.set_structural_dry_mask(artifact["dry_mask"])
    logger.info(
        "Structural-dry policy=%s mask_definition=%s n_dry=%s n_wettable=%s artifact=%s",
        policy_kwargs["policy"],
        policy_kwargs["mask_definition"],
        artifact["n_dry"],
        artifact["n_wettable"],
        artifact_path,
    )
    return policy_kwargs, artifact

def main():
    config, device, is_logger = load_config_and_setup()

    # Logging: file (rotating) + console, config-driven level and path
    log_level = _cfg_get(config, "log_level", "INFO")
    log_file = _cfg_get(config, "log_file", None)
    if log_file is not None:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            save_dir = _cfg_get(config.checkpoint, "save_dir", ".")
            if save_dir is None:
                save_dir = "."
            log_path = Path(save_dir) / log_path
    else:
        save_dir = _cfg_get(config.checkpoint, "save_dir", ".")
        if save_dir is None:
            save_dir = "."
        log_path = Path(save_dir) / "training.log"
    logger = setup_logging(
        log_level=log_level,
        log_file=str(log_path),
        logger_name="flood_train",
    )
    logger.info("Config loaded; device=%s", device)

    # Reproducibility: set all RNG seeds and deterministic CuDNN (override setup() for full reproducibility)
    seed = _cfg_get(config.distributed, "seed", 123)
    deterministic = _cfg_get(config, "deterministic", True)
    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    use_distributed = bool(_cfg_get(config.distributed, "use_distributed", False)) and dist.is_available() and dist.is_initialized()
    effective_seed = int(seed) + int(global_rank)
    set_seed(effective_seed, deterministic=deterministic)
    logger.info(
        "Random seed set to %s (base=%s, rank=%s, world_size=%s, deterministic=%s)",
        effective_seed,
        seed,
        global_rank,
        world_size,
        deterministic,
    )

    # Possibly adjust FNO modes
    if hasattr(config.data, "resolution") and (config.data.resolution < config.gino.fno_n_modes[0]):
        config.gino.fno_n_modes = [config.data.resolution] * 2

    # Initialize wandb if needed
    wandb_init_args = {}
    if config.wandb.log and is_logger:
        wandb.login(key=get_wandb_api_key())
        wandb_name = config.wandb.name if config.wandb.name else f"flood-run_{_cfg_get(config.data, 'resolution', 64)}"
        wandb_init_args = dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        wandb.init(**wandb_init_args)

    # ---------------------- Setup training dataset (HDF only) -----------------------------
    skip_before_timestep = _cfg_get(config.data, "skip_before_timestep", 0)
    noise_type = _cfg_get(config.data, "noise_type", "none")
    noise_std = _cfg_get(config.data, "noise_std", None)
    static_text_files = _cfg_get(config.data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"])
    n_history = config.data.n_history
    target_variables = parse_target_variables(_cfg_get(config.data, "target_variables", ["wd", "vx", "vy"]))
    n_target_channels = len(target_variables)
    # Optionally (over)write train.txt with all existing *\.hdf run IDs in data.root
    if _cfg_get(config.data, "write_train_txt", False):
        run_ids = write_train_txt_from_data_root(
            config.data.root,
            train_txt=_cfg_get(config.data, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )
        logger.info("Wrote train.txt with %s run IDs from %s", len(run_ids), config.data.root)
    # Static: 2 from HDF (elevation, area) + text files (CS, CU, FA) — aligned with HEC_RAS_Automation
    n_static = 2 + len(static_text_files)
    data_channels = n_static + n_history * 1 + n_history * n_target_channels
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target_channels)
    ar_rollout_steps = max(1, _safe_int(_cfg_get(config.opt, "ar_rollout_steps", 1), 1))
    data_boundary_kwargs = get_dataset_boundary_kwargs(config.data)
    logger.info(
        "Training dataset boundary_source=%s%s",
        data_boundary_kwargs["boundary_source"],
        f", clean_boundary_file={data_boundary_kwargs['clean_boundary_file']}"
        if data_boundary_kwargs["boundary_source"] == "clean_family"
        else "",
    )
    full_dataset = FloodDatasetHDF(
        data_root=config.data.root,
        n_history=config.data.n_history,
        query_res=_cfg_get(config.data, "query_res", [64, 64]),
        run_ids=None,
        train_txt=_cfg_get(config.data, "train_txt", "train.txt"),
        static_text_files=static_text_files,
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=skip_before_timestep,
        noise_type=noise_type,
        noise_std=noise_std,
        ar_rollout_steps=ar_rollout_steps,
        target_variables=target_variables,
        **data_boundary_kwargs,
    )
    normalizer_path = _resolve_training_normalizer_path(config)
    structural_dry_policy, structural_dry_artifact = _prepare_structural_dry_artifact_for_training(
        config,
        dataset=full_dataset,
        normalizer_path=normalizer_path,
        logger=logger,
        use_distributed=use_distributed,
        global_rank=global_rank,
    )
    n_samples_max = _cfg_get(config.data, "n_samples_max", None)
    if n_samples_max is not None:
        total_avail = len(full_dataset)
        n_samples_max = int(n_samples_max)  # CLI may pass str
        n_use = min(n_samples_max, total_avail)
        full_dataset = Subset(full_dataset, range(n_use))
        logger.info("Limited to %s samples (n_samples_max=%s)", n_use, n_samples_max)

    total_len = len(full_dataset)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, test_data_raw_temp = random_split(
        full_dataset, [train_sz, test_sz], generator=make_split_generator(seed)
    )

    logger.info("Dataset: total=%s, train=%s, test (one-step)=%s", total_len, train_sz, test_sz)

    # No leakage: normalizers are fit only on train_data_raw. Test data is transformed with
    # train-fit stats in NormalizedDatasetOnTheFly; evaluation uses model.eval() and torch.no_grad().
    # Normalizers: load from disk if path exists and is set; otherwise fit and optionally save
    if normalizer_path is not None and normalizer_path.exists():
        normalizers = load_normalizers(normalizer_path, device=None)
        logger.info("Loaded normalizers from %s", normalizer_path)
    else:
        norm_chunk_size = _cfg_get(config.data, "normalizer_chunk_size", 10000)
        normalizers = fit_normalizers_streaming(
            train_data_raw,
            chunk_size=norm_chunk_size,
            expect_target=True,
            structural_dry_policy=structural_dry_policy["policy"],
        )
        if normalizer_path is not None:
            save_normalizers(normalizers, normalizer_path)
            logger.info("Saved normalizers to %s", normalizer_path)

    train_normalized_dataset = NormalizedDatasetOnTheFly(
        train_data_raw, normalizers, query_res=config.data.query_res
    )
    num_workers = _cfg_get(config.data, "num_workers", 0)

    pin_memory = bool(_cfg_get(config.data, "pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(_cfg_get(config.data, "persistent_workers", True))
    prefetch_default = 1 if os.name == "nt" else 2
    prefetch_factor = int(_cfg_get(config.data, "prefetch_factor", prefetch_default))

    # Windows + multiprocessing + long-running HDF workloads can be less stable with
    # persistent workers; prefer safer defaults unless explicitly overridden in code.
    if os.name == "nt" and num_workers > 0 and persistent_workers:
        logger.warning(
            "Windows detected with num_workers=%s and persistent_workers=True; "
            "overriding persistent_workers=False for DataLoader stability.",
            num_workers,
        )
        persistent_workers = False

    worker_seed_base = int(seed) + int(global_rank) * 100_000
    worker_init_fn = partial(dataloader_worker_init, base_seed=worker_seed_base) if num_workers > 0 else None
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_sampler = None
    if use_distributed:
        train_sampler = DistributedSampler(
            train_normalized_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )

    train_loader = DataLoader(
        train_normalized_dataset,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        generator=make_dataloader_generator(effective_seed) if train_sampler is None else None,
        **loader_kwargs,
    )

    # One-step test: same on-the-fly normalization (no stacking test set)
    test_normalized_dataset = NormalizedDatasetOnTheFly(
        test_data_raw_temp, normalizers, query_res=config.data.query_res
    )
    test_loader = DataLoader(
        test_normalized_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    logger.info(
        "Data: device=%s, train_samples=%s, test_samples=%s, batch_size=%s, noise=%s std=%s, distributed=%s, rank=%s/%s",
        device,
        train_sz,
        len(test_normalized_dataset),
        config.data.batch_size,
        noise_type,
        noise_std,
        use_distributed,
        global_rank,
        world_size,
    )

    # Model
    model = get_model(config)

    # Optimizer/scheduler
    optimizer = AdamW(model.parameters(),
                      lr=config.opt.learning_rate,
                      weight_decay=config.opt.weight_decay)

    if config.opt.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=_cfg_get(config.opt, "gamma", 0.5),
            patience=_cfg_get(config.opt, "scheduler_patience", 5),
            mode=_cfg_get(config.opt, "scheduler_mode", "min"),
            threshold=_cfg_get(config.opt, "scheduler_threshold", 1e-4),
            threshold_mode=_cfg_get(config.opt, "scheduler_threshold_mode", "rel"),
            cooldown=_cfg_get(config.opt, "scheduler_cooldown", 0),
            min_lr=_cfg_get(config.opt, "scheduler_min_lr", 0.0),
        )
    elif config.opt.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=_cfg_get(config.opt, "scheduler_T_max", 200),
            eta_min=_cfg_get(config.opt, "scheduler_eta_min", 0.0),
        )
    elif config.opt.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=_cfg_get(config.opt, "step_size", 50),
            gamma=_cfg_get(config.opt, "gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler {config.opt.scheduler}")

    # Loss (LpLoss(d=2,p=2): relative L2 by default; use 'l2_abs' if relative plateaus)
    # reduction='sum' so train_err = sum(loss)/n_samples = mean per sample (same scale as test_l2)
    l2loss = LpLoss(d=2, p=2)
    structural_policy_name = structural_dry_policy["policy"]
    primary_l2_loss = FloodMaskedRelLpLoss(
        policy=structural_policy_name,
        base_loss=l2loss,
        reduction="sum",
    )
    primary_l2_abs_loss = FloodMaskedAbsLpLoss(
        policy=structural_policy_name,
        reduction="sum",
    )
    use_fgn = bool(_cfg_get(config.gino, "use_fgn_noise", False))
    output_distribution = str(_cfg_get(config.gino, "output_distribution", "deterministic")).strip().lower()
    if output_distribution not in {"deterministic", "gaussian"}:
        raise ValueError(
            f"Unknown gino.output_distribution={output_distribution!r}. "
            "Use 'deterministic' or 'gaussian'."
        )
    setattr(config.gino, "output_distribution", output_distribution)
    fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
        _cfg_get(config.gino, "fgn_latent_temporal_mode", "stepwise")
    )
    setattr(config.gino, "fgn_latent_temporal_mode", fgn_latent_temporal_mode)
    training_loss_name = str(_cfg_get(config.opt, "training_loss", "l2")).strip().lower()
    setattr(config.opt, "training_loss", training_loss_name)
    fgn_ar_state_update = normalize_fgn_ar_state_update(
        _cfg_get(config.opt, "fgn_ar_state_update", "mean_feedback")
    )
    setattr(config.opt, "fgn_ar_state_update", fgn_ar_state_update)

    if training_loss_name == "gaussian_nll" and output_distribution != "gaussian":
        raise ValueError(
            "training_loss='gaussian_nll' requires gino.output_distribution='gaussian'."
        )
    fno_norm_mode = _cfg_get(config.gino, "fno_norm", None)
    fno_norm_mode = None if fno_norm_mode is None else str(fno_norm_mode).strip().lower()
    if output_distribution == "gaussian" and use_fgn:
        raise ValueError(
            "gino.output_distribution='gaussian' requires gino.use_fgn_noise=false."
        )
    if output_distribution == "gaussian" and fno_norm_mode != "instance_norm":
        if fno_norm_mode == "ada_in":
            warnings.warn(
                "gino.output_distribution='gaussian' is incompatible with gino.fno_norm='ada_in'. "
                "Overriding gino.fno_norm -> 'instance_norm'.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "gino.output_distribution='gaussian' uses gino.fno_norm='instance_norm' in this "
                f"pipeline (got {fno_norm_mode!r}). Overriding gino.fno_norm -> 'instance_norm'.",
                UserWarning,
                stacklevel=2,
            )
        setattr(config.gino, "fno_norm", "instance_norm")
    if training_loss_name == "crps" and not use_fgn:
        raise ValueError(
            "training_loss='crps' requires gino.use_fgn_noise=true in this pipeline."
        )

    if training_loss_name == "l2":
        train_loss_fn = primary_l2_loss
        if use_fgn:
            warnings.warn(
                "gino.use_fgn_noise is True but training_loss is 'l2'. "
                "FGN noise will not be trained; use training_loss: 'crps' for probabilistic FGN.",
                UserWarning,
                stacklevel=2,
            )
    elif training_loss_name == "crps" and use_fgn:
        crps_n_samples = max(2, _safe_int(_cfg_get(config.opt, "crps_n_samples", 2), 2))
        crps_channel_weights = _cfg_get(config.opt, "crps_channel_weights", None)
        train_loss_fn = FloodMaskedCRPSLoss(
            policy=structural_policy_name,
            base_loss=CRPSLoss(
                n_samples=crps_n_samples,
                channel_weights=crps_channel_weights,
                reduction="mean",
            ),
        )
    elif training_loss_name == "gaussian_nll":
        train_loss_fn = FloodMaskedGaussianNLLLoss(
            policy=structural_policy_name,
            base_loss=GaussianNLLLoss(
                channel_weights=_cfg_get(config.opt, "crps_channel_weights", None),
                reduction="mean",
                min_logvar=_safe_float(_cfg_get(config.opt, "gaussian_min_logvar", -9.0), -9.0),
                max_logvar=_safe_float(_cfg_get(config.opt, "gaussian_max_logvar", 4.0), 4.0),
                logvar_reg_weight=_safe_float(
                    _cfg_get(config.opt, "gaussian_logvar_reg_weight", 1e-6), 1e-6
                ),
            ),
        )
    elif training_loss_name == "l2_abs":
        # Absolute L2 (||pred-y||) instead of relative (||pred-y||/||y||); can help if relative plateaus
        train_loss_fn = primary_l2_abs_loss
        if use_fgn:
            warnings.warn(
                "gino.use_fgn_noise is True but training_loss is 'l2_abs'. FGN will not be trained.",
                UserWarning,
                stacklevel=2,
            )
    else:
        raise ValueError(
            f"Unknown training loss: {config.opt.training_loss}. "
            "Use 'l2', 'l2_abs', 'gaussian_nll', or for FGN: 'crps' with gino.use_fgn_noise: true."
        )

    if config.opt.testing_loss == "l2":
        test_loss_fn = primary_l2_loss
    else:
        test_loss_fn = primary_l2_loss

    # Eval metrics:
    # - FGN/CRPS: L2 on ensemble mean + CRPS.
    # - Gaussian NLL: L2 on predictive mean + Gaussian NLL on packed output.
    if output_distribution == "gaussian" and training_loss_name == "gaussian_nll":
        eval_losses = {
            "l2": test_loss_fn,
            "gaussian_nll": FloodMaskedGaussianNLLLoss(
                policy=structural_policy_name,
                base_loss=GaussianNLLLoss(
                    channel_weights=_cfg_get(config.opt, "crps_channel_weights", None),
                    reduction="mean",
                    min_logvar=_safe_float(_cfg_get(config.opt, "gaussian_min_logvar", -9.0), -9.0),
                    max_logvar=_safe_float(_cfg_get(config.opt, "gaussian_max_logvar", 4.0), 4.0),
                    logvar_reg_weight=0.0,
                ),
            ),
        }
    elif use_fgn and training_loss_name == "crps":
        crps_n_samples = max(2, _safe_int(_cfg_get(config.opt, "crps_n_samples", 2), 2))
        crps_channel_weights = _cfg_get(config.opt, "crps_channel_weights", None)
        eval_losses = {
            "l2": test_loss_fn,
            "crps": FloodMaskedCRPSLoss(
                policy=structural_policy_name,
                base_loss=CRPSLoss(
                    n_samples=crps_n_samples,
                    channel_weights=crps_channel_weights,
                    reduction="mean",
                ),
            ),
        }
    else:
        eval_losses = {config.opt.testing_loss: test_loss_fn}

    if structural_policy_name == "masked_primary":
        eval_losses["l2_full_domain"] = l2loss
        eval_losses["rmse_dry_background_wd"] = FloodDryBackgroundRMSE()
        eval_losses["mae_dry_background_wd"] = FloodDryBackgroundMAE()
        eval_losses["falsewet_rate_001_dry_background_wd"] = FloodDryBackgroundFalseWetRate(
            threshold=0.01
        )
        eval_losses["falsewet_rate_005_dry_background_wd"] = FloodDryBackgroundFalseWetRate(
            threshold=0.05
        )
        if output_distribution == "gaussian" and training_loss_name == "gaussian_nll":
            eval_losses["gaussian_nll_full_domain"] = GaussianNLLLoss(
                channel_weights=_cfg_get(config.opt, "crps_channel_weights", None),
                reduction="mean",
                min_logvar=_safe_float(_cfg_get(config.opt, "gaussian_min_logvar", -9.0), -9.0),
                max_logvar=_safe_float(_cfg_get(config.opt, "gaussian_max_logvar", 4.0), 4.0),
                logvar_reg_weight=0.0,
            )
            eval_losses["pred_std_mean_dry_background_wd"] = FloodGaussianDryPredStdMean(
                n_channels=n_target_channels,
                min_logvar=_safe_float(_cfg_get(config.opt, "gaussian_min_logvar", -9.0), -9.0),
                max_logvar=_safe_float(_cfg_get(config.opt, "gaussian_max_logvar", 4.0), 4.0),
            )
        elif use_fgn and training_loss_name == "crps":
            eval_losses["crps_full_domain"] = FloodMaskedCRPSLoss(
                policy="legacy_full_domain",
                base_loss=CRPSLoss(
                    n_samples=crps_n_samples,
                    channel_weights=crps_channel_weights,
                    reduction="mean",
                ),
            )
            eval_losses["pred_std_mean_dry_background_wd"] = FloodEnsembleDryPredStdMean()

    # DataProcessor: training loss is always in normalized space (pred and y from dataset are normalized).
    # Eval: when inverse_test=True, pred and y are inverse-transformed so test_l2/test_crps are in physical space.
    inverse_test = _cfg_get(config, "inverse_test", True)
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target", None),
        inverse_test=inverse_test,
        output_distribution=output_distribution,
    )
    data_processor.wrap(model)

    # Trainer (FGN: two forwards + CRPS per batch)
    fgn_noise_dim = _cfg_get(config.gino, "fgn_noise_dim", 32)  # used for rollout ensemble when use_fgn
    use_progress_bar = _cfg_get(config, "use_progress_bar", True)
    scheduler_monitor = _cfg_get(config.opt, "scheduler_monitor", "train_err")
    eval_interval = _cfg_get(config.wandb, "eval_interval", 1)
    mixed_precision = bool(_cfg_get(config.opt, "amp_autocast", False))
    query_res = _cfg_get(config.data, "query_res", None)
    if query_res is None:
        query_res = [_cfg_get(config.data, "resolution", None), _cfg_get(config.data, "resolution", None)]
    try:
        spatial_dims = [int(query_res[0]), int(query_res[1])]
    except Exception:
        spatial_dims = []
    if mixed_precision and spatial_dims and any(not _is_power_of_two(d) for d in spatial_dims):
        logger.warning(
            "Disabling AMP autocast: cuFFT half-precision requires power-of-two spatial sizes, "
            "but got query_res=%s.",
            spatial_dims,
        )
        mixed_precision = False
    grad_accum_steps = max(1, _safe_int(_cfg_get(config.opt, "grad_accum_steps", 1), 1))
    if use_fgn and training_loss_name == "crps":
        crps_l2_weight = _cfg_get(config.opt, "crps_l2_weight", 0.5)
        ar_finetune_start_epoch = max(0, _safe_int(_cfg_get(config.opt, "ar_finetune_start_epoch", 0), 0))
        ar_curriculum_epochs_per_step = max(0, _safe_int(_cfg_get(config.opt, "ar_curriculum_epochs_per_step", 0), 0))
        use_flood_crps_spatial_weights = _cfg_get(config.opt, "flood_crps_spatial_weights", False)
        flood_crps_wet_threshold = _cfg_get(config.opt, "wet_threshold", 0.01)
        flood_crps_wet_smooth_scale = _cfg_get(config.opt, "wet_smooth_scale", 0.02)
        flood_crps_dry_weight_alpha = _cfg_get(config.opt, "dry_weight_alpha", 0.1)
        crps_n_samples = max(2, _safe_int(_cfg_get(config.opt, "crps_n_samples", 2), 2))
        ar_gradient_mode = str(_cfg_get(config.opt, "ar_gradient_mode", "adaptive")).strip().lower()
        ar_truncation_steps = max(1, _safe_int(_cfg_get(config.opt, "ar_truncation_steps", 1), 1))
        crps_sample_chunk_size = max(1, _safe_int(_cfg_get(config.opt, "crps_sample_chunk_size", 1), 1))
        use_activation_checkpointing = bool(_cfg_get(config.opt, "use_activation_checkpointing", False))
        trainer = FGNTrainer(
            model=model,
            n_epochs=config.opt.n_epochs,
            data_processor=data_processor,
            device=device,
            wandb_log=config.wandb.log,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
            use_distributed=use_distributed,
            mixed_precision=mixed_precision,
            grad_accum_steps=grad_accum_steps,
            fgn_noise_dim=fgn_noise_dim,
            crps_n_samples=crps_n_samples,
            rel_l2_loss_fn=primary_l2_loss,
            crps_l2_weight=crps_l2_weight,
            ar_finetune_start_epoch=ar_finetune_start_epoch,
            ar_rollout_steps=ar_rollout_steps,
            ar_curriculum_epochs_per_step=ar_curriculum_epochs_per_step,
            use_flood_crps_spatial_weights=use_flood_crps_spatial_weights,
            flood_crps_wet_threshold=flood_crps_wet_threshold,
            flood_crps_wet_smooth_scale=flood_crps_wet_smooth_scale,
            flood_crps_dry_weight_alpha=flood_crps_dry_weight_alpha,
            static_normalizer=normalizers.get("static") if use_flood_crps_spatial_weights else None,
            use_hazard_proxy_crps=_cfg_get(config.opt, "hazard_proxy_crps", False),
            hazard_proxy_crps_weight=_cfg_get(config.opt, "hazard_proxy_crps_weight", 0.15),
            ar_pooled_crps_gamma=_cfg_get(config.opt, "ar_pooled_crps_gamma", 1.0),
            ar_gradient_mode=ar_gradient_mode,
            ar_truncation_steps=ar_truncation_steps,
            crps_sample_chunk_size=crps_sample_chunk_size,
            use_activation_checkpointing=use_activation_checkpointing,
            fgn_latent_temporal_mode=fgn_latent_temporal_mode,
            fgn_ar_state_update=fgn_ar_state_update,
        )
    elif output_distribution == "gaussian" and training_loss_name == "gaussian_nll":
        trainer = GaussianNLLTrainer(
            model=model,
            n_epochs=config.opt.n_epochs,
            data_processor=data_processor,
            device=device,
            wandb_log=config.wandb.log,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
            use_distributed=use_distributed,
            mixed_precision=mixed_precision,
            grad_accum_steps=grad_accum_steps,
            rel_l2_loss_fn=primary_l2_loss,
            ar_finetune_start_epoch=max(0, _safe_int(_cfg_get(config.opt, "ar_finetune_start_epoch", 0), 0)),
            ar_rollout_steps=ar_rollout_steps,
            ar_curriculum_epochs_per_step=max(0, _safe_int(_cfg_get(config.opt, "ar_curriculum_epochs_per_step", 0), 0)),
            gaussian_min_logvar=_safe_float(_cfg_get(config.opt, "gaussian_min_logvar", -9.0), -9.0),
            gaussian_max_logvar=_safe_float(_cfg_get(config.opt, "gaussian_max_logvar", 4.0), 4.0),
        )
    else:
        trainer = Trainer(
            model=model,
            n_epochs=config.opt.n_epochs,
            data_processor=data_processor,
            device=device,
            wandb_log=config.wandb.log,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
            use_distributed=use_distributed,
            mixed_precision=mixed_precision,
            grad_accum_steps=grad_accum_steps,
        )
    logger.info(
        "Trainer settings: distributed=%s, mixed_precision=%s, grad_accum_steps=%s%s",
        use_distributed,
        mixed_precision,
        grad_accum_steps,
        (
            f", ar_gradient_mode={_cfg_get(config.opt, 'ar_gradient_mode', 'adaptive')}, "
            f"ar_truncation_steps={_cfg_get(config.opt, 'ar_truncation_steps', 1)}, "
            f"use_activation_checkpointing={_cfg_get(config.opt, 'use_activation_checkpointing', False)}"
            if use_fgn and training_loss_name == "crps"
            else ""
        ),
    )
    if use_fgn and training_loss_name == "crps":
        logger.info(
            "FGN settings: latent_temporal_mode=%s, ar_state_update=%s, crps_n_samples=%s, fgn_noise_dim=%s",
            fgn_latent_temporal_mode,
            fgn_ar_state_update,
            crps_n_samples,
            fgn_noise_dim,
        )

    # Optional: verify gradient flow and overfit one batch (set verify_training: true in config)
    try:
        do_verify = config.verify_training
    except (KeyError, AttributeError):
        do_verify = False
    if use_distributed and global_rank != 0:
        do_verify = False
    if do_verify:
        logger.info("--- Training verification ---")
        trainer.optimizer = optimizer  # trainer.train() sets these; set early for verification
        trainer.regularizer = None
        trainer.n_samples = 0
        trainer.epoch = 0
        trainer.model = trainer.model.to(trainer.device)
        if trainer.data_processor is not None and trainer.data_processor.device != trainer.device:
            trainer.data_processor = trainer.data_processor.to(trainer.device)
        verify_training_gradient_flow(trainer, train_loader, train_loss_fn)
        overfit_sanity_check(trainer, train_loader, train_loss_fn, optimizer, n_steps=15)
        logger.info("--- End verification ---")
    # Train: always save a rolling "last" checkpoint and a metric-based "best" checkpoint.
    save_every_raw = _cfg_get(config.checkpoint, "save_every", 1)
    try:
        save_every = int(save_every_raw)
    except (TypeError, ValueError):
        logger.warning("Invalid checkpoint.save_every=%r; using 1.", save_every_raw)
        save_every = 1
    if save_every < 1:
        logger.warning("checkpoint.save_every=%s is invalid; using 1.", save_every)
        save_every = 1
    available_eval_metrics = [f"test_{k}" for k in eval_losses.keys()]
    configured_best_metric = _cfg_get(config.checkpoint, "save_best_metric", None)
    if configured_best_metric is not None:
        save_best_metric = str(configured_best_metric).strip() or None
    else:
        if output_distribution == "gaussian" and training_loss_name == "gaussian_nll":
            preferred_metric = "test_gaussian_nll"
        elif use_fgn and training_loss_name == "crps":
            preferred_metric = "test_crps"
        else:
            preferred_metric = "test_l2"
        if preferred_metric in available_eval_metrics:
            save_best_metric = preferred_metric
        elif available_eval_metrics:
            save_best_metric = available_eval_metrics[0]
        else:
            save_best_metric = None

    if save_best_metric is not None and save_best_metric not in available_eval_metrics:
        raise ValueError(
            f"checkpoint.save_best_metric={save_best_metric!r} is not in available "
            f"eval metrics: {available_eval_metrics}"
        )

    if logger is not None:
        logger.info(
            "Checkpointing policy: save_every=%s (last=model), save_best=%s",
            save_every,
            save_best_metric,
        )

    checkpoint_save_dir = _cfg_get(config.checkpoint, "save_dir", ".")
    if checkpoint_save_dir is None:
        checkpoint_save_dir = "."
    checkpoint_resume_dir = _cfg_get(config.checkpoint, "resume_from_dir", None)
    if is_logger:
        save_effective_config_snapshot(config, checkpoint_save_dir, logger=logger)

    trainer.train(
        train_loader=train_loader,
        test_loaders={'test': test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=train_loss_fn,
        eval_losses=eval_losses,
        regularizer=None,
        save_every=save_every,
        save_best=save_best_metric,
        save_dir=checkpoint_save_dir,
        resume_from_dir=checkpoint_resume_dir
    )

    # ----------------- Optional: rollout evaluation on new data -----------------------
    run_rollout = _cfg_get(config.rollout, "run_after_training", False)
    if not run_rollout:
        if is_logger:
            logger.info("Skipping rollout (run_after_training: false).")
        if config.wandb.log:
            wandb.finish()
        return
    if n_target_channels != 3:
        if is_logger:
            logger.warning(
                "Skipping rollout plotting/eval because target_variables=%s (C_out=%s) "
                "is not supported by rollout visualization code (expects [wd, vx, vy]).",
                target_variables, n_target_channels,
            )
        if config.wandb.log:
            wandb.finish()
        return

    rollout_length = config.data.rollout_length
    history_steps = config.data.n_history
    rollout_skip_before_timestep = _cfg_get(config.data, "skip_before_timestep", 0)

    rollout_data_root = config.rollout_data.root
    rollout_boundary_kwargs = get_dataset_boundary_kwargs(config.rollout_data, split="test")
    logger.info(
        "Rollout dataset boundary_source=%s%s",
        rollout_boundary_kwargs["boundary_source"],
        f", clean_boundary_file={rollout_boundary_kwargs['clean_boundary_file']}"
        if rollout_boundary_kwargs["boundary_source"] == "clean_family"
        else "",
    )
    rollout_test_dataset = FloodRolloutTestDatasetHDF(
        rollout_data_root=rollout_data_root,
        n_history=history_steps,
        rollout_length=rollout_length,
        run_ids=None,
        test_txt=_cfg_get(config.rollout_data, "test_txt", "test.txt"),
        static_text_files=_cfg_get(config.rollout_data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]),
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=rollout_skip_before_timestep,
        **rollout_boundary_kwargs,
    )
    if structural_dry_artifact is not None:
        rollout_test_dataset.set_structural_dry_mask(structural_dry_artifact["dry_mask"])

    # Normalizing rollout data
    rollout_geom_list, rollout_static_list, rollout_boundary_list, rollout_dyn_list, _ = collect_all_fields(
        rollout_test_dataset, expect_target=False
    )

    def transform_with_existing_normalizers(geom_list, static_list, boundary_list, dyn_list, normalizers):
        ref_device = None
        for key in ("dynamic", "target", "geometry"):
            if key in normalizers and normalizers[key] is not None and hasattr(normalizers[key], "mean"):
                ref_device = normalizers[key].mean.device
                break
        if ref_device is None:
            ref_device = torch.device("cpu")

        for key in ["geometry", "static", "boundary", "dynamic", "target"]:
            if key in normalizers and normalizers[key] is not None:
                normalizers[key].to(ref_device)

        geometry_big = torch.stack(geom_list, dim=0) if geom_list else None
        static_big = torch.stack(static_list, dim=0) if static_list else None
        boundary_big = torch.stack(boundary_list, dim=0) if boundary_list else None
        dynamic_big = torch.stack(dyn_list, dim=0) if dyn_list else None

        if ref_device is not None:
            if geometry_big is not None:
                geometry_big = geometry_big.to(ref_device)
            if static_big is not None:
                static_big = static_big.to(ref_device)
            if boundary_big is not None:
                boundary_big = boundary_big.to(ref_device)
            if dynamic_big is not None:
                dynamic_big = dynamic_big.to(ref_device)

        if geometry_big is not None and "geometry" in normalizers:
            geometry_big = normalizers["geometry"].transform(geometry_big)
        if static_big is not None and "static" in normalizers:
            static_big = normalizers["static"].transform(static_big)
        if boundary_big is not None and "boundary" in normalizers:
            boundary_big = normalizers["boundary"].transform(boundary_big)
        if dynamic_big is not None and "dynamic" in normalizers:
            dynamic_big = normalizers["dynamic"].transform(dynamic_big)

        return {
            "geometry": geometry_big,
            "static": static_big,
            "boundary": boundary_big,
            "dynamic": dynamic_big,
        }

    transformed_rollout = transform_with_existing_normalizers(
        rollout_geom_list,
        rollout_static_list,
        rollout_boundary_list,
        rollout_dyn_list,
        normalizers
    )

    normalized_rollout_samples = []
    for i in range(len(rollout_test_dataset)):
        normalized_rollout_samples.append({
            "run_id": rollout_test_dataset.valid_run_ids[i],
            "geometry": transformed_rollout["geometry"][i],
            "static": transformed_rollout["static"][i],
            "boundary": transformed_rollout["boundary"][i],
            "dynamic": transformed_rollout["dynamic"][i],
            **(
                {"structural_dry_mask": structural_dry_artifact["dry_mask"]}
                if structural_dry_artifact is not None
                else {}
            ),
        })

    rollout_normalized_dataset = NormalizedRolloutTestDataset(
        normalized_samples=normalized_rollout_samples,
        query_res=config.data.query_res
    )

    rollout_prediction(
        trainer=trainer,
        rollout_dataset=rollout_normalized_dataset,
        rollout_length=rollout_length,
        history_steps=history_steps,
        dynamic_norm=normalizers["dynamic"],
        target_norm=normalizers["target"],
        device=device,
        skip_before_timestep=rollout_skip_before_timestep,
        dt=config.data.dt,
        out_dir=config.rollout.out_dir,
        fgn_noise_dim=fgn_noise_dim if use_fgn else None,
        n_ensemble_samples=_cfg_get(config.rollout, "n_ensemble_samples", 1),
        fgn_latent_temporal_mode=fgn_latent_temporal_mode,
        gaussian_min_logvar=_safe_float(_cfg_get(config.opt, "gaussian_min_logvar", -9.0), -9.0),
        gaussian_max_logvar=_safe_float(_cfg_get(config.opt, "gaussian_max_logvar", 4.0), 4.0),
    )

    if config.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
