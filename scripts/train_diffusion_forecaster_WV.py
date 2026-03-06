#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train DDO-style conditional diffusion forecaster on WV flood depth-only data."""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import wandb
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neuralop import get_model  # noqa: E402
from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers  # noqa: E402
from neuralop.diffusion import (  # noqa: E402
    ConditioningConfig,
    ConditionalDDOForecaster,
    PointRFFGaussianProcessSampler,
)
from scripts.diffusion_script_utils import (  # noqa: E402
    save_checkpoint_sidecars,
    safe_wandb_finish,
    safe_get,
    shutdown_dataloader_workers,
    to_builtin,
)
from neuralop.utils import get_wandb_api_key  # noqa: E402
from train_gino_flood_train_rollout_animation_WV import (  # noqa: E402
    FloodDatasetHDF,
    NormalizedDatasetOnTheFly,
    dataloader_worker_init,
    fit_normalizers_streaming,
    make_dataloader_generator,
    make_split_generator,
    parse_target_variables,
    set_seed,
    setup_logging,
)

TRAIN_FRAC = 0.9
DEFAULT_PRINT_EVERY = 100
DEFAULT_MAX_VAL_BATCHES = 64
DEFAULT_BOUNDARY_CHANNELS = 1
TIME_FEATURE_DIM_SINCOS = 2
TIME_FEATURE_DIM_RAW = 1


def _load_config(config_default: Path) -> Any:
    import sys as _sys

    config_name = "flood"
    config_path = config_default
    argv = list(_sys.argv[1:])
    for i, a in enumerate(argv):
        if a == "--config_path" and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            if not config_path.is_absolute():
                config_path = (_REPO_ROOT / config_path).resolve()
            idx = _sys.argv.index("--config_path")
            _sys.argv.pop(idx + 1)
            _sys.argv.pop(idx)
            break

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipe = ConfigPipeline(
        [
            YamlConfig(str(config_path), config_name=config_name, config_folder=str(_REPO_ROOT / "config")),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        ]
    )
    config = pipe.read_conf()
    return config


def _resolve_device(config: Any) -> torch.device:
    configured = str(safe_get(safe_get(config, "distributed", {}), "device", "cuda:0"))
    if configured.startswith("cuda") and torch.cuda.is_available():
        return torch.device(configured)
    return torch.device("cpu")


def _resolve_normalizer_path(config: Any) -> Optional[Path]:
    normalizer_path = safe_get(safe_get(config, "data", {}), "normalizer_path", None)
    if normalizer_path is None:
        return None
    p = Path(str(normalizer_path))
    if not p.is_absolute():
        p = Path(str(safe_get(safe_get(config, "data", {}), "root", "."))) / p
    return p.resolve()


def _prepare_datasets(
    config: Any,
    target_variables: list[str],
    logger,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any], Path, int, int]:
    data_cfg = safe_get(config, "data", {})
    seed = int(safe_get(safe_get(config, "distributed", {}), "seed", 123))
    static_text_files = list(safe_get(data_cfg, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]))

    dataset_kwargs = dict(
        data_root=str(safe_get(data_cfg, "root", "")),
        n_history=int(safe_get(data_cfg, "n_history", 3)),
        train_txt=str(safe_get(data_cfg, "train_txt", "train.txt")),
        static_text_files=static_text_files,
        hdf_suffix=".hdf",
        noise_type=str(safe_get(data_cfg, "noise_type", "none")),
        noise_std=safe_get(data_cfg, "noise_std", None),
        skip_before_timestep=int(safe_get(data_cfg, "skip_before_timestep", 0)),
        ar_rollout_steps=max(1, int(safe_get(safe_get(config, "opt", {}), "ar_rollout_steps", 1))),
        target_variables=target_variables,
    )
    # Backward compatibility: only pass write_train_txt if this FloodDatasetHDF
    # implementation supports it.
    if "write_train_txt" in inspect.signature(FloodDatasetHDF.__init__).parameters:
        dataset_kwargs["write_train_txt"] = bool(safe_get(data_cfg, "write_train_txt", False))
    full_dataset = FloodDatasetHDF(**dataset_kwargs)

    n_samples_max = safe_get(data_cfg, "n_samples_max", None)
    if n_samples_max is not None and int(n_samples_max) > 0:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))

    total_len = len(full_dataset)
    train_sz = max(1, int(TRAIN_FRAC * total_len))
    test_sz = total_len - train_sz
    train_raw, test_raw = random_split(
        full_dataset,
        [train_sz, test_sz],
        generator=make_split_generator(seed),
    )
    logger.info("Split dataset: total=%d train=%d test=%d", total_len, train_sz, test_sz)

    normalizer_path = _resolve_normalizer_path(config)
    if normalizer_path is not None and normalizer_path.exists():
        normalizers = load_normalizers(normalizer_path, device=None)
        logger.info("Loaded normalizers from %s", normalizer_path)
    else:
        normalizers = fit_normalizers_streaming(
            train_raw,
            chunk_size=int(safe_get(data_cfg, "normalizer_chunk_size", 10000)),
            expect_target=True,
        )
        if normalizer_path is not None:
            save_normalizers(normalizers, normalizer_path)
            logger.info("Saved normalizers to %s", normalizer_path)

    query_res = list(safe_get(data_cfg, "query_res", [48, 48]))
    train_norm = NormalizedDatasetOnTheFly(train_raw, normalizers, query_res=query_res)
    test_norm = NormalizedDatasetOnTheFly(test_raw, normalizers, query_res=query_res)

    batch_size = int(safe_get(data_cfg, "batch_size", 8))
    num_workers = int(safe_get(data_cfg, "num_workers", 0))
    pin_memory = bool(safe_get(data_cfg, "pin_memory", True))
    persistent_workers = bool(safe_get(data_cfg, "persistent_workers", False)) and num_workers > 0
    prefetch_factor = safe_get(data_cfg, "prefetch_factor", 2)

    loader_seed = seed
    train_loader_kwargs = dict(
        dataset=train_norm,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=make_dataloader_generator(loader_seed),
    )
    if num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=int(prefetch_factor),
            worker_init_fn=lambda wid: dataloader_worker_init(wid, loader_seed),
        )
    train_loader = DataLoader(**train_loader_kwargs)
    test_loader = DataLoader(
        test_norm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    n_static = 2 + len(static_text_files)
    n_boundary_channels = DEFAULT_BOUNDARY_CHANNELS
    if len(train_norm) > 0:
        boundary_sample = train_norm[0].get("boundary", None)
        if isinstance(boundary_sample, torch.Tensor) and boundary_sample.ndim >= 1:
            n_boundary_channels = int(boundary_sample.shape[-1])
        else:
            logger.warning(
                "Could not infer boundary channels from dataset sample; "
                "falling back to n_boundary_channels=%d.",
                DEFAULT_BOUNDARY_CHANNELS,
            )
    normalizer_path = normalizer_path or Path("<not_saved>")
    return (
        train_loader,
        test_loader,
        normalizers,
        normalizer_path,
        n_static,
        n_boundary_channels,
    )


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Normalize batch shape and build diffusion inputs.

    Context channel layout:
    - static features
    - flattened boundary history (all boundary channels)
    - flattened dynamic history (target channels)
    """
    sample: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v.to(device)

    dyn = sample["dynamic"]
    if dyn.ndim == 3:
        dyn = dyn.unsqueeze(0)
    dyn = dyn.permute(0, 2, 1, 3)
    bsz, n_cells, n_hist, n_dyn_ch = dyn.shape
    dyn_flat = dyn.reshape(bsz, n_cells, n_hist * n_dyn_ch)

    bc = sample["boundary"]
    if bc.ndim == 3:
        bc = bc.unsqueeze(0)
    bc = bc.permute(0, 2, 1, 3)
    bsz2, n_cells2, n_hist2, n_bc_ch = bc.shape
    bc_flat = bc.reshape(bsz2, n_cells2, n_hist2 * n_bc_ch)

    static = sample["static"]
    if static.ndim == 2:
        static = static.unsqueeze(0)

    context = torch.cat([static, bc_flat, dyn_flat], dim=2)

    geom = sample["geometry"]
    if geom.ndim == 2:
        geom = geom.unsqueeze(0)
    geom_shared = geom[0:1] if geom.shape[0] > 1 else geom

    q = sample["query_points"]
    if q.ndim == 3:
        q = q.unsqueeze(0)
    q_shared = q[0:1] if q.shape[0] > 1 else q

    y = sample["target"]
    if y.ndim == 2:
        y = y.unsqueeze(0)

    return {
        "context": context,
        "target": y,
        "input_geom": geom_shared,
        "latent_queries": q_shared,
        "output_queries": geom_shared.clone(),
    }


def _configure_denoiser(
    config: Any,
    n_static: int,
    n_boundary_channels: int,
    n_target_channels: int,
) -> Tuple[Dict[str, Any], int, int, str, int]:
    diff_cfg = safe_get(config, "diffusion", {})
    cond_cfg = safe_get(diff_cfg, "conditioning", {})
    n_history = int(safe_get(safe_get(config, "data", {}), "n_history", 3))
    time_injection = str(safe_get(cond_cfg, "time_injection", "channel")).lower()
    if time_injection not in {"channel", "adain"}:
        raise ValueError(
            f"diffusion.conditioning.time_injection must be one of {{'channel', 'adain'}}, got {time_injection!r}"
        )
    time_embedding_dim = int(safe_get(cond_cfg, "time_embedding_dim", 32))

    base_channels = n_static + n_history * n_boundary_channels + n_history * n_target_channels
    extra = 0
    if bool(safe_get(cond_cfg, "add_noisy_target", True)):
        extra += n_target_channels
    if time_injection == "channel" and bool(safe_get(cond_cfg, "add_time_features", True)):
        t_type = str(safe_get(cond_cfg, "time_feature_type", "sincos")).lower()
        extra += TIME_FEATURE_DIM_SINCOS if t_type == "sincos" else TIME_FEATURE_DIM_RAW

    total_in_channels = base_channels + extra

    gino_cfg = copy.deepcopy(to_builtin(safe_get(config, "gino", {})))
    gino_cfg["data_channels"] = int(total_in_channels)
    gino_cfg["out_channels"] = int(n_target_channels)
    gino_cfg["output_distribution"] = "deterministic"
    gino_cfg["use_fgn_noise"] = False
    # Respect config-defined checkpoint/model behavior; no forced AR override here.
    if time_injection == "adain":
        gino_cfg["fno_norm"] = "ada_in"
        gino_cfg["fno_ada_in_dim"] = int(time_embedding_dim)
    elif str(safe_get(gino_cfg, "fno_norm", "")).lower() == "ada_in":
        gino_cfg["fno_norm"] = "instance_norm"
    return gino_cfg, base_channels, total_in_channels, time_injection, time_embedding_dim


def _init_scheduler(config: Any, optimizer: torch.optim.Optimizer):
    opt = safe_get(config, "opt", {})
    name = str(safe_get(opt, "scheduler", "")).lower()
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(safe_get(opt, "scheduler_mode", "min")),
            patience=int(safe_get(opt, "scheduler_patience", 5)),
            threshold=float(safe_get(opt, "scheduler_threshold", 1e-4)),
            threshold_mode=str(safe_get(opt, "scheduler_threshold_mode", "rel")),
            cooldown=int(safe_get(opt, "scheduler_cooldown", 0)),
            min_lr=float(safe_get(opt, "scheduler_min_lr", 0.0)),
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(safe_get(opt, "scheduler_T_max", 200)),
            eta_min=float(safe_get(opt, "scheduler_eta_min", 0.0)),
        )
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(safe_get(opt, "step_size", 50)),
            gamma=float(safe_get(opt, "gamma", 0.5)),
        )
    return None


def _build_wandb_names(config: Any, seed: int) -> Tuple[str, str]:
    diff_cfg = safe_get(config, "diffusion", {})
    gp_cfg = safe_get(diff_cfg, "gp", {})
    sampler_cfg = safe_get(diff_cfg, "sampler", {})
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    gp_type = str(safe_get(gp_cfg, "type", "rff_rbf"))
    ls = float(safe_get(gp_cfg, "length_scale", 0.05))
    n_steps = int(safe_get(sampler_cfg, "num_steps", 40))
    group = f"ddofs_wv_m40_depth_job{job_id}"
    name = f"ddofs_wv_m40_depth_seed{seed}_steps{n_steps}_gp{gp_type}_ls{ls:.4f}"
    return group, name


def _maybe_init_wandb(config: Any, seed: int, logger) -> Optional[Any]:
    wb_cfg = safe_get(config, "wandb", {})
    if not bool(safe_get(wb_cfg, "log", False)):
        return None

    key = get_wandb_api_key()
    if key:
        wandb.login(key=key, relogin=False)

    group_default, name_default = _build_wandb_names(config, seed)
    run = wandb.init(
        project=str(safe_get(wb_cfg, "project", "Flood_GINO_NoPhysics")),
        entity=safe_get(wb_cfg, "entity", None),
        group=str(safe_get(wb_cfg, "group", group_default) or group_default),
        name=str(safe_get(wb_cfg, "name", name_default) or name_default),
        config=to_builtin(config),
        dir=str(_SCRIPT_DIR / "wandb"),
        reinit=True,
    )
    logger.info("Initialized W&B run: %s", run.name)
    return run


def _evaluate_validation(
    forecaster: ConditionalDDOForecaster,
    loader: DataLoader,
    device: torch.device,
    target_norm: Optional[Any],
    max_batches: int = DEFAULT_MAX_VAL_BATCHES,
) -> Dict[str, float]:
    forecaster.eval()
    losses = []
    rmse_norm = []
    rmse_phys = []

    if target_norm is not None:
        target_norm.to(device)

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break
            sample = _prepare_batch(batch, device)
            loss, _ = forecaster.training_loss(sample)
            losses.append(float(loss.item()))

            pred = forecaster.sample_next(
                context=sample["context"],
                input_geom=sample["input_geom"],
                latent_queries=sample["latent_queries"],
                output_queries=sample["output_queries"],
                stochastic=False,
                initial_latent=torch.zeros_like(sample["target"]),
            )
            tgt = sample["target"]
            rmse_norm.append(float(torch.sqrt(torch.mean((pred - tgt) ** 2)).item()))

            if target_norm is not None:
                pred_phys = target_norm.inverse_transform(pred)
                tgt_phys = target_norm.inverse_transform(tgt)
                rmse_phys.append(float(torch.sqrt(torch.mean((pred_phys - tgt_phys) ** 2)).item()))

    out = {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_rmse_norm": float(np.mean(rmse_norm)) if rmse_norm else float("nan"),
        "val_rmse_phys": float(np.mean(rmse_phys)) if rmse_phys else float("nan"),
    }
    forecaster.train()
    return out


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    seed: int,
    best_val_loss: float,
    normalizer_path: Path,
    target_variables: list[str],
    gino_cfg: Dict[str, Any],
    forecaster: ConditionalDDOForecaster,
    scheduler: Optional[Any] = None,
) -> None:
    metadata = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "seed": int(seed),
        "best_val_loss": float(best_val_loss),
        "normalizer_path": str(normalizer_path),
        "target_variables": list(target_variables),
        "gino_config": gino_cfg,
        "diffusion_hparams": forecaster.diffusion_hparams(),
    }
    payload = {
        **metadata,
        "denoiser_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    save_checkpoint_sidecars(
        path,
        denoiser_state_dict=model.state_dict(),
        metadata=metadata,
    )


def main() -> int:
    config = _load_config(_REPO_ROOT / "config" / "gino_pluvial_flood_config_WV_depth_only_diffusion.yaml")
    device = _resolve_device(config)

    log_file = safe_get(config, "log_file", "train_diffusion.log")
    if not Path(log_file).is_absolute():
        log_file = str((_SCRIPT_DIR / log_file).resolve())
    logger = setup_logging(
        log_level=str(safe_get(config, "log_level", "INFO")),
        log_file=log_file,
        logger_name="flood_diffusion_train",
    )

    seed = int(safe_get(safe_get(config, "distributed", {}), "seed", 123))
    deterministic = bool(safe_get(config, "deterministic", True))
    set_seed(seed, deterministic=deterministic)
    logger.info("Using device=%s seed=%d deterministic=%s", device, seed, deterministic)

    target_variables = parse_target_variables(safe_get(safe_get(config, "data", {}), "target_variables", ["wd"]))
    if target_variables != ["wd"]:
        raise ValueError("Diffusion v1 supports depth-only target_variables=['wd'].")

    train_loader, test_loader, normalizers, normalizer_path, n_static, n_boundary_channels = _prepare_datasets(
        config,
        target_variables=target_variables,
        logger=logger,
    )
    logger.info("Prepared loaders: train_batches=%d test_batches=%d", len(train_loader), len(test_loader))

    gino_cfg, _, total_in_channels, time_injection, time_embedding_dim = _configure_denoiser(
        config,
        n_static=n_static,
        n_boundary_channels=n_boundary_channels,
        n_target_channels=1,
    )
    model_cfg = {"arch": "gino", "gino": copy.deepcopy(gino_cfg)}
    # get_model() consumes/mutates config keys (e.g., pops data_channels), so
    # pass an isolated model config copy to keep our saved metadata stable.
    model = get_model(model_cfg).to(device)
    logger.info(
        "Initialized denoiser (GINO) with in_channels=%d (n_boundary_channels=%d)",
        total_in_channels,
        n_boundary_channels,
    )

    diff_cfg = safe_get(config, "diffusion", {})
    gp_cfg = safe_get(diff_cfg, "gp", {})
    cond_cfg_raw = safe_get(diff_cfg, "conditioning", {})
    cond_cfg = ConditioningConfig(
        add_noisy_target=bool(safe_get(cond_cfg_raw, "add_noisy_target", True)),
        add_time_features=bool(safe_get(cond_cfg_raw, "add_time_features", True)),
        time_feature_type=str(safe_get(cond_cfg_raw, "time_feature_type", "sincos")),
        time_injection=str(safe_get(cond_cfg_raw, "time_injection", "channel")).lower(),
        time_embedding_dim=int(safe_get(cond_cfg_raw, "time_embedding_dim", 32)),
        time_embedding_hidden_dim=int(safe_get(cond_cfg_raw, "time_embedding_hidden_dim", 128)),
        time_embedding_scale=float(safe_get(cond_cfg_raw, "time_embedding_scale", 10000.0)),
    )
    logger.info(
        (
            "Diffusion conditioning: time_injection=%s time_embedding_dim=%d "
            "add_noisy_target=%s add_time_features=%s total_in_channels=%d fno_norm=%s"
        ),
        cond_cfg.time_injection,
        cond_cfg.time_embedding_dim,
        cond_cfg.add_noisy_target,
        cond_cfg.add_time_features,
        total_in_channels,
        str(safe_get(gino_cfg, "fno_norm", "unknown")),
    )
    if cond_cfg.time_injection != time_injection:
        raise ValueError(
            "Mismatch between denoiser wiring and conditioning config: "
            f"time_injection={time_injection!r} vs {cond_cfg.time_injection!r}"
        )
    if cond_cfg.time_injection == "adain" and cond_cfg.time_embedding_dim != time_embedding_dim:
        raise ValueError(
            "Mismatch between denoiser fno_ada_in_dim and conditioning.time_embedding_dim: "
            f"{time_embedding_dim} vs {cond_cfg.time_embedding_dim}"
        )
    gp_sampler = PointRFFGaussianProcessSampler(
        dim=2,
        gp_type=str(safe_get(gp_cfg, "type", "rff_rbf")),
        sigma=float(safe_get(gp_cfg, "sigma", 1.0)),
        length_scale=float(safe_get(gp_cfg, "length_scale", 0.05)),
        rff_features=int(safe_get(gp_cfg, "rff_features", 256)),
        seed=seed,
    ).to(device)

    sched_cfg = safe_get(diff_cfg, "schedule", {})
    sampler_cfg = safe_get(diff_cfg, "sampler", {})
    forecaster = ConditionalDDOForecaster(
        denoiser=model,
        gp_sampler=gp_sampler,
        parameterization=str(safe_get(diff_cfg, "parameterization", "epsilon")),
        timestep_sampler=str(safe_get(diff_cfg, "timestep_sampler", "low_discrepancy")),
        lmbd0=float(safe_get(sched_cfg, "lmbd0", 10.0)),
        lmbd1=float(safe_get(sched_cfg, "lmbd1", -10.0)),
        weight_method=safe_get(sched_cfg, "weight_method", "shifted_sigmoid_2"),
        conditioning=cond_cfg,
        sampler_method=str(safe_get(sampler_cfg, "method", "denoise")),
        sampler_num_steps=int(safe_get(sampler_cfg, "num_steps", 40)),
        sampler_s_min=float(safe_get(sampler_cfg, "s_min", 1e-4)),
        sampler_return_mean_last=bool(safe_get(sampler_cfg, "return_mean_last", True)),
    ).to(device)

    opt_cfg = safe_get(config, "opt", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(safe_get(opt_cfg, "learning_rate", 1e-4)),
        weight_decay=float(safe_get(opt_cfg, "weight_decay", 1e-4)),
    )
    scheduler = _init_scheduler(config, optimizer)

    run = _maybe_init_wandb(config, seed=seed, logger=logger)
    wandb_finish_timeout_seconds = float(
        safe_get(safe_get(config, "wandb", {}), "finish_timeout_seconds", 120.0)
    )

    ckpt_dir = Path(str(safe_get(safe_get(config, "checkpoint", {}), "save_dir", "./checkpoints_WV_depth_only_diffusion")))
    if not ckpt_dir.is_absolute():
        ckpt_dir = (_SCRIPT_DIR / ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = int(safe_get(opt_cfg, "n_epochs", 1))
    print_every = int(safe_get(config, "print_every", DEFAULT_PRINT_EVERY))
    max_val_batches = int(safe_get(config, "max_val_batches", DEFAULT_MAX_VAL_BATCHES))
    grad_clip = safe_get(opt_cfg, "clip_grad_norm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    best_val_loss = float("inf")
    global_step = 0
    first_batch_checked = False
    logger.info("Starting diffusion training for %d epochs", n_epochs)

    try:
        for epoch in range(1, n_epochs + 1):
            forecaster.train()
            epoch_loss = 0.0
            t0 = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)
            for batch in pbar:
                sample = _prepare_batch(batch, device)
                if not first_batch_checked:
                    batch_size = sample["context"].shape[0]
                    t_probe = torch.full(
                        (batch_size,),
                        0.5,
                        device=device,
                        dtype=sample["context"].dtype,
                    )
                    if cond_cfg.time_injection == "adain":
                        ada_probe = forecaster._build_time_adain(t_probe)
                        expected_shape = (batch_size, cond_cfg.time_embedding_dim)
                        if tuple(ada_probe.shape) != expected_shape:
                            raise ValueError(
                                f"AdaIN timestep embedding shape mismatch: got {tuple(ada_probe.shape)} "
                                f"expected {expected_shape}"
                            )
                    else:
                        if cond_cfg.time_injection != "channel":
                            raise ValueError(
                                f"Unexpected conditioning mode in first-batch check: {cond_cfg.time_injection!r}"
                            )
                    first_batch_checked = True
                optimizer.zero_grad(set_to_none=True)

                loss, stats = forecaster.training_loss(sample)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                global_step += 1
                epoch_loss += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4e}")

                if run is not None:
                    wandb.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/mse_eps": stats["mse_eps"],
                            "train/t_mean": stats["t_mean"],
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )

                if global_step % max(1, print_every) == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.6e lr=%.6e",
                        epoch,
                        global_step,
                        float(loss.item()),
                        float(optimizer.param_groups[0]["lr"]),
                    )

            train_loss_epoch = epoch_loss / max(1, len(train_loader))
            val_stats = _evaluate_validation(
                forecaster=forecaster,
                loader=test_loader,
                device=device,
                target_norm=normalizers.get("target", None),
                max_batches=max_val_batches,
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_stats["val_loss"])
                else:
                    scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d | train_loss=%.6e | val_loss=%.6e | val_rmse_norm=%.6e | val_rmse_phys=%.6e | %.2fs",
                epoch,
                n_epochs,
                train_loss_epoch,
                val_stats["val_loss"],
                val_stats["val_rmse_norm"],
                val_stats["val_rmse_phys"],
                elapsed,
            )

            if run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss_epoch": train_loss_epoch,
                        "val/loss": val_stats["val_loss"],
                        "val/rmse_norm": val_stats["val_rmse_norm"],
                        "val/rmse_phys": val_stats["val_rmse_phys"],
                    },
                    step=global_step,
                )

            latest_path = ckpt_dir / "checkpoint.pt"
            _save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                best_val_loss=best_val_loss,
                normalizer_path=normalizer_path,
                target_variables=target_variables,
                gino_cfg=gino_cfg,
                forecaster=forecaster,
                scheduler=scheduler,
            )

            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = float(val_stats["val_loss"])
                best_path = ckpt_dir / "checkpoint_best.pt"
                _save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    seed=seed,
                    best_val_loss=best_val_loss,
                    normalizer_path=normalizer_path,
                    target_variables=target_variables,
                    gino_cfg=gino_cfg,
                    forecaster=forecaster,
                    scheduler=scheduler,
                )
                logger.info("Saved new best checkpoint: %s", best_path)

        logger.info("Training complete. Best val_loss=%.6e | checkpoints=%s", best_val_loss, ckpt_dir)

        metadata = {
            "checkpoint_dir": str(ckpt_dir),
            "best_val_loss": best_val_loss,
            "normalizer_path": str(normalizer_path),
            "target_variables": target_variables,
            "global_step": global_step,
        }
        with open(ckpt_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return 0
    finally:
        shutdown_dataloader_workers(train_loader, logger=logger, name="train_loader")
        shutdown_dataloader_workers(test_loader, logger=logger, name="test_loader")
        if run is not None:
            safe_wandb_finish(run, logger=logger, timeout_seconds=wandb_finish_timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
