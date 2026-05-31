"""Canonical flood diffusion training application."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from neuralop import get_model
from neuralop.diffusion import ConditioningConfig, ConditionalDDOForecaster, PointRFFGaussianProcessSampler
from neuralop.flood.train.diffusion_data import _prepare_batch, _prepare_datasets
from neuralop.flood.train.diffusion_loop import _evaluate_validation, _resolve_resume_checkpoint, _save_checkpoint
from neuralop.flood.train.diffusion_runtime import (
    DEFAULT_MAX_VAL_BATCHES,
    DEFAULT_PRINT_EVERY,
    _REPO_ROOT,
    _cfg_dist,
    _configure_denoiser,
    _dist_barrier,
    _init_distributed,
    _init_scheduler,
    _load_config,
    _load_state_dict_compat,
    _maybe_init_wandb,
    _optimizer_to_device,
    _rank0_info,
    _reduce_sum,
    _resolve_device,
    _unwrap_module,
    _SCRIPT_DIR,
)
from neuralop.flood.utils.diffusion_script_utils import (
    load_checkpoint_bundle,
    safe_get,
    safe_wandb_finish,
    shutdown_dataloader_workers,
)
from neuralop.flood.utils.runtime import parse_target_variables, set_seed, setup_logging
from neuralop.training.determinism import (
    deterministic_seed_context,
    restore_rng_state,
    seed_dataloader_for_epoch,
    stable_seed_from_parts,
)


def _diffusion_ar_enabled(opt_cfg) -> bool:
    return bool(safe_get(opt_cfg, "diffusion_ar_rollout_training", False))


def _effective_diffusion_ar_steps(epoch: int, opt_cfg, max_available_steps: int) -> int:
    ar_rollout_steps = max(1, int(safe_get(opt_cfg, "ar_rollout_steps", 1)))
    ar_start_epoch = max(0, int(safe_get(opt_cfg, "ar_finetune_start_epoch", 0)))
    if (not _diffusion_ar_enabled(opt_cfg)) or ar_rollout_steps <= 1 or epoch < ar_start_epoch:
        return 1
    ar_curriculum_epochs = max(0, int(safe_get(opt_cfg, "ar_curriculum_epochs_per_step", 0)))
    ar_start_steps = max(1, int(safe_get(opt_cfg, "ar_curriculum_start_steps", 1)))
    if ar_curriculum_epochs > 0:
        ar_epoch_index = max(0, epoch - ar_start_epoch)
        step_index = ar_epoch_index // ar_curriculum_epochs
        requested_steps = min(ar_start_steps + step_index, ar_rollout_steps)
    else:
        requested_steps = ar_rollout_steps
    return max(1, min(int(requested_steps), int(max_available_steps)))


def _build_diffusion_context(static: torch.Tensor, boundary_hist: torch.Tensor, dynamic_hist: torch.Tensor) -> torch.Tensor:
    if static.ndim == 2:
        static = static.unsqueeze(0)
    if boundary_hist.ndim != 4 or dynamic_hist.ndim != 4:
        raise ValueError(
            "Diffusion AR context expects boundary/dynamic histories shaped [B, H, N, C]; "
            f"got boundary={tuple(boundary_hist.shape)} dynamic={tuple(dynamic_hist.shape)}"
        )
    bsz, n_hist, n_cells, n_dyn = dynamic_hist.shape
    bsz2, n_hist2, n_cells2, n_bc = boundary_hist.shape
    if (bsz2, n_hist2, n_cells2) != (bsz, n_hist, n_cells):
        raise ValueError(
            "Diffusion AR boundary history must match dynamic history over batch/history/cells: "
            f"boundary={(bsz2, n_hist2, n_cells2)} dynamic={(bsz, n_hist, n_cells)}"
        )
    dyn_flat = dynamic_hist.permute(0, 2, 1, 3).reshape(bsz, n_cells, n_hist * n_dyn)
    bc_flat = boundary_hist.permute(0, 2, 1, 3).reshape(bsz, n_cells, n_hist * n_bc)
    return torch.cat([static, bc_flat, dyn_flat], dim=2)


def _require_diffusion_ar_tensors(sample: dict) -> None:
    missing = [
        key
        for key in ("target_sequence", "boundary_sequence", "dynamic", "boundary", "static")
        if key not in sample or sample[key] is None
    ]
    if missing:
        raise ValueError(
            "Diffusion AR rollout training requires target_sequence, boundary_sequence, "
            f"dynamic, boundary, and static tensors. Missing: {missing}."
        )


def _slice_boundary_sequence_step(boundary_sequence: torch.Tensor, step: int) -> torch.Tensor:
    bc_step = boundary_sequence[:, step]
    if bc_step.ndim != 3:
        raise ValueError(f"boundary_sequence step must be [B, N, C], got {tuple(bc_step.shape)}")
    return bc_step.unsqueeze(1)


def _diffusion_train_loss_for_epoch(
    *,
    forecaster: ConditionalDDOForecaster,
    sample: dict,
    opt_cfg,
    epoch: int,
    batch_idx: int,
    seed: int,
    rank: int,
    deterministic: bool,
    logger,
    dist_ctx,
):
    if not _diffusion_ar_enabled(opt_cfg):
        batch_seed = None
        if deterministic:
            batch_seed = stable_seed_from_parts("diffusion_train", int(seed), int(epoch), int(batch_idx), int(rank))
        with deterministic_seed_context(batch_seed):
            return forecaster.training_loss(sample)

    _require_diffusion_ar_tensors(sample)
    target_sequence = sample["target_sequence"]
    boundary_sequence = sample["boundary_sequence"]
    max_available_steps = int(target_sequence.shape[1])
    n_ar_steps = _effective_diffusion_ar_steps(epoch, opt_cfg, max_available_steps)
    if n_ar_steps <= 1:
        batch_seed = None
        if deterministic:
            batch_seed = stable_seed_from_parts("diffusion_train", int(seed), int(epoch), int(batch_idx), int(rank))
        with deterministic_seed_context(batch_seed):
            loss, stats = forecaster.training_loss(sample)
        stats = dict(stats)
        stats["ar_rollout_steps"] = 1.0
        return loss, stats

    state_num_steps = max(1, int(safe_get(opt_cfg, "diffusion_ar_state_num_steps", 1)))
    if batch_idx == 0:
        _rank0_info(
            logger,
            dist_ctx,
            "Diffusion AR rollout training: epoch=%d rollout_steps=%d (max=%d) [curriculum], state_update_sampler_steps=%d, state_update=detached_deterministic.",
            epoch,
            n_ar_steps,
            max(1, int(safe_get(opt_cfg, "ar_rollout_steps", 1))),
            state_num_steps,
        )

    dynamic_sliding = sample["dynamic"].clone()
    boundary_sliding = sample["boundary"].clone()
    static = sample["static"]
    total_loss = None
    mse_eps_total = 0.0
    t_mean_total = 0.0
    for step in range(n_ar_steps):
        target_step = target_sequence[:, step]
        if target_step.ndim == 2:
            target_step = target_step.unsqueeze(0)
        step_sample = {
            "context": _build_diffusion_context(static, boundary_sliding, dynamic_sliding),
            "target": target_step,
            "input_geom": sample["input_geom"],
            "latent_queries": sample["latent_queries"],
            "output_queries": sample["output_queries"],
        }
        for optional_key in ("point_weights", "structural_dry_mask"):
            if optional_key in sample:
                step_sample[optional_key] = sample[optional_key]
        step_seed = None
        if deterministic:
            step_seed = stable_seed_from_parts(
                "diffusion_train_ar",
                int(seed),
                int(epoch),
                int(batch_idx),
                int(step),
                int(rank),
            )
        with deterministic_seed_context(step_seed):
            loss_step, stats_step = forecaster.training_loss(step_sample)
        total_loss = loss_step if total_loss is None else total_loss + loss_step
        mse_eps_total += float(stats_step.get("mse_eps", 0.0))
        t_mean_total += float(stats_step.get("t_mean", 0.0))

        if step + 1 < n_ar_steps:
            with torch.no_grad():
                pred_step = forecaster.sample_next(
                    context=step_sample["context"],
                    input_geom=step_sample["input_geom"],
                    latent_queries=step_sample["latent_queries"],
                    output_queries=step_sample["output_queries"],
                    num_steps=state_num_steps,
                    stochastic=False,
                    initial_latent=torch.zeros_like(target_step),
                )
            dynamic_sliding = torch.cat([dynamic_sliding[:, 1:], pred_step.detach().unsqueeze(1)], dim=1)
            bc_step = _slice_boundary_sequence_step(boundary_sequence, step)
            boundary_sliding = torch.cat([boundary_sliding[:, 1:], bc_step], dim=1)[:, -dynamic_sliding.shape[1]:]

    loss = total_loss / float(n_ar_steps)
    return loss, {
        "loss": float(loss.detach().item()),
        "mse_eps": mse_eps_total / float(n_ar_steps),
        "t_mean": t_mean_total / float(n_ar_steps),
        "ar_rollout_steps": float(n_ar_steps),
        "ar_state_num_steps": float(state_num_steps),
    }

def main() -> int:
    config = _load_config(_REPO_ROOT / "config" / "gino_pluvial_flood_config_WV_depth_only_diffusion.yaml")
    dist_ctx = _init_distributed(config)
    device = _resolve_device(config, dist_ctx=dist_ctx)

    log_file = safe_get(config, "log_file", "train_diffusion.log")
    if not Path(log_file).is_absolute():
        log_file = str((_SCRIPT_DIR / log_file).resolve())
    logger = setup_logging(
        log_level=str(safe_get(config, "log_level", "INFO")),
        log_file=log_file,
        logger_name="flood_diffusion_train",
    )
    if dist_ctx.use_distributed and not dist_ctx.is_rank0:
        logger.setLevel(logging.ERROR)

    base_seed = int(safe_get(safe_get(config, "distributed", {}), "seed", 123))
    seed = int(base_seed + (dist_ctx.rank if dist_ctx.use_distributed else 0))
    deterministic = bool(safe_get(config, "deterministic", True))
    set_seed(seed, deterministic=deterministic)
    _rank0_info(
        logger,
        dist_ctx,
        "Using device=%s base_seed=%d rank_seed=%d deterministic=%s distributed=%s rank=%d/%d",
        device,
        base_seed,
        seed,
        deterministic,
        dist_ctx.use_distributed,
        dist_ctx.rank,
        dist_ctx.world_size,
    )

    target_variables = parse_target_variables(safe_get(safe_get(config, "data", {}), "target_variables", ["wd"]))
    if target_variables != ["wd"]:
        raise ValueError("Diffusion v1 supports depth-only target_variables=['wd'].")

    (
        train_loader,
        test_loader,
        train_sampler,
        _test_sampler,
        normalizers,
        normalizer_path,
        n_static,
        n_boundary_channels,
    ) = _prepare_datasets(
        config,
        target_variables=target_variables,
        logger=logger,
        dist_ctx=dist_ctx,
        is_resuming=_resolve_resume_checkpoint(config) is not None,
    )
    _rank0_info(
        logger,
        dist_ctx,
        "Prepared loaders: train_batches=%d test_batches=%d",
        len(train_loader),
        len(test_loader),
    )

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
    _rank0_info(
        logger,
        dist_ctx,
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
    _rank0_info(
        logger,
        dist_ctx,
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

    dist_cfg = _cfg_dist(config)
    find_unused_parameters = bool(safe_get(dist_cfg, "find_unused_parameters", False))
    if dist_ctx.use_distributed:
        if device.type == "cuda":
            ddp_kwargs = dict(
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            ddp_kwargs = dict(find_unused_parameters=find_unused_parameters)
        model = DDP(model, **ddp_kwargs)
        forecaster.denoiser = model
        if forecaster.time_mlp is not None:
            forecaster.time_mlp = DDP(forecaster.time_mlp, **ddp_kwargs)
        _rank0_info(
            logger,
            dist_ctx,
            "Enabled DDP: world_size=%d local_rank=%d find_unused_parameters=%s",
            dist_ctx.world_size,
            dist_ctx.local_rank,
            find_unused_parameters,
        )

    opt_cfg = safe_get(config, "opt", {})
    optim_params = list(model.parameters())
    if forecaster.time_mlp is not None:
        optim_params.extend(list(forecaster.time_mlp.parameters()))
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(safe_get(opt_cfg, "learning_rate", 1e-4)),
        weight_decay=float(safe_get(opt_cfg, "weight_decay", 1e-4)),
    )
    scheduler = _init_scheduler(config, optimizer)

    run = _maybe_init_wandb(config, seed=seed, logger=logger, is_rank0=dist_ctx.is_rank0)
    wandb_finish_timeout_seconds = float(
        safe_get(safe_get(config, "wandb", {}), "finish_timeout_seconds", 120.0)
    )

    ckpt_dir = Path(str(safe_get(safe_get(config, "checkpoint", {}), "save_dir", "./checkpoints_WV_depth_only_diffusion")))
    if not ckpt_dir.is_absolute():
        ckpt_dir = (_SCRIPT_DIR / ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    allow_unsafe_legacy_load = bool(
        safe_get(safe_get(config, "checkpoint", {}), "allow_unsafe_legacy_load", True)
    )
    if allow_unsafe_legacy_load and dist_ctx.is_rank0:
        logger.warning(
            "checkpoint.allow_unsafe_legacy_load=true. Legacy pickle checkpoints must be trusted."
        )

    n_epochs = int(safe_get(opt_cfg, "n_epochs", 1))
    print_every = int(safe_get(config, "print_every", DEFAULT_PRINT_EVERY))
    max_val_batches = int(safe_get(config, "max_val_batches", DEFAULT_MAX_VAL_BATCHES))
    grad_clip = safe_get(opt_cfg, "clip_grad_norm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    resume_checkpoint = _resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        resume_bundle = load_checkpoint_bundle(
            resume_checkpoint,
            map_location="cpu",
            allow_unsafe_legacy_load=allow_unsafe_legacy_load,
            merge_legacy_training_state=True,
            logger=logger,
        )
        _load_state_dict_compat(
            _unwrap_module(model),
            resume_bundle["denoiser_state_dict"],
            name="denoiser_state_dict",
        )
        resume_time_mlp = resume_bundle.get("time_mlp_state_dict", None)
        if forecaster.time_mlp is not None:
            if resume_time_mlp is not None:
                _load_state_dict_compat(
                    _unwrap_module(forecaster.time_mlp),
                    resume_time_mlp,
                    name="time_mlp_state_dict",
                )
            else:
                _rank0_info(
                    logger,
                    dist_ctx,
                    "Resume checkpoint %s has no time_mlp_state_dict; continuing with current initialization.",
                    resume_checkpoint,
                )
        optimizer_state = resume_bundle.get("optimizer_state_dict")
        if optimizer_state is None:
            raise RuntimeError(
                "Resume checkpoint is missing optimizer_state_dict. Diffusion resume must "
                "restore optimizer state to preserve training semantics."
            )
        optimizer.load_state_dict(optimizer_state)
        _optimizer_to_device(optimizer, device)
        if scheduler is not None:
            scheduler_state = resume_bundle.get("scheduler_state_dict")
            if scheduler_state is None:
                raise RuntimeError(
                    "Resume checkpoint is missing scheduler_state_dict. Diffusion resume must "
                    "restore scheduler state to preserve learning-rate schedule semantics."
                )
            scheduler.load_state_dict(scheduler_state)
        best_val_loss = float(resume_bundle.get("best_val_loss", best_val_loss))
        global_step = int(resume_bundle.get("global_step", 0))
        start_epoch = int(resume_bundle.get("epoch", 0)) + 1
        restore_rng_state(resume_bundle.get("rng_state"))
        _rank0_info(
            logger,
            dist_ctx,
            "Resumed from %s: start_epoch=%d global_step=%d best_val_loss=%.6e",
            resume_checkpoint,
            start_epoch,
            global_step,
            best_val_loss,
        )

    _dist_barrier(dist_ctx)
    first_batch_checked = False
    _rank0_info(logger, dist_ctx, "Starting diffusion training for %d epochs", n_epochs)

    try:
        for epoch in range(start_epoch, n_epochs + 1):
            forecaster.train()
            epoch_loss = 0.0
            epoch_batches = 0
            t0 = time.time()

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            else:
                seed_dataloader_for_epoch(train_loader, base_seed=seed, epoch=epoch)

            train_iter = train_loader
            pbar = None
            if dist_ctx.is_rank0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)
                train_iter = pbar

            for batch_idx, batch in enumerate(train_iter):
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
                batch_seed = None
                if deterministic:
                    batch_seed = stable_seed_from_parts(
                        "diffusion_train",
                        int(seed),
                        int(epoch),
                        int(batch_idx),
                        int(dist_ctx.rank),
                    )
                loss, stats = _diffusion_train_loss_for_epoch(
                    forecaster=forecaster,
                    sample=sample,
                    opt_cfg=opt_cfg,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    seed=seed,
                    rank=dist_ctx.rank,
                    deterministic=deterministic,
                    logger=logger,
                    dist_ctx=dist_ctx,
                )
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
                optimizer.step()

                global_step += 1
                epoch_loss += float(loss.item())
                epoch_batches += 1
                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss.item():.4e}")

                if run is not None:
                    wandb.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/mse_eps": stats["mse_eps"],
                            "train/t_mean": stats["t_mean"],
                            "train/ar_rollout_steps": stats.get("ar_rollout_steps", 1.0),
                            "train/ar_state_num_steps": stats.get("ar_state_num_steps", 0.0),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )

                if global_step % max(1, print_every) == 0:
                    _rank0_info(
                        logger,
                        dist_ctx,
                        "epoch=%d step=%d loss=%.6e lr=%.6e",
                        epoch,
                        global_step,
                        float(loss.item()),
                        float(optimizer.param_groups[0]["lr"]),
                    )

            train_loss_epoch = _reduce_sum(epoch_loss, device=device, dist_ctx=dist_ctx)
            train_batches_global = _reduce_sum(epoch_batches, device=device, dist_ctx=dist_ctx)
            train_loss_epoch = train_loss_epoch / max(1.0, train_batches_global)
            val_stats = _evaluate_validation(
                forecaster=forecaster,
                loader=test_loader,
                device=device,
                target_norm=normalizers.get("target", None),
                dist_ctx=dist_ctx,
                max_batches=max_val_batches,
                deterministic_eval=deterministic,
                eval_seed=seed,
                epoch=epoch,
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_stats["val_loss"])
                else:
                    scheduler.step()

            elapsed = time.time() - t0
            _rank0_info(
                logger,
                dist_ctx,
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
                payload = {
                    "epoch": epoch,
                    "train/loss_epoch": train_loss_epoch,
                }
                for key, value in val_stats.items():
                    payload[f"val/{key}"] = value
                wandb.log(payload, step=global_step)

            improved = val_stats["val_loss"] < best_val_loss
            checkpoint_best_val_loss = float(
                val_stats["val_loss"] if improved else best_val_loss
            )
            latest_path = ckpt_dir / "checkpoint.pt"
            _save_checkpoint(
                path=latest_path,
                model=model,
                time_mlp=forecaster.time_mlp,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                best_val_loss=checkpoint_best_val_loss,
                normalizer_path=normalizer_path,
                target_variables=target_variables,
                gino_cfg=gino_cfg,
                forecaster=forecaster,
                scheduler=scheduler,
            )

            if improved:
                best_val_loss = float(val_stats["val_loss"])
                best_path = ckpt_dir / "checkpoint_best.pt"
                _save_checkpoint(
                    path=best_path,
                    model=model,
                    time_mlp=forecaster.time_mlp,
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
                if dist_ctx.is_rank0:
                    logger.info("Saved new best checkpoint: %s", best_path)
            _dist_barrier(dist_ctx)

        _rank0_info(
            logger,
            dist_ctx,
            "Training complete. Best val_loss=%.6e | checkpoints=%s",
            best_val_loss,
            ckpt_dir,
        )

        metadata = {
            "checkpoint_dir": str(ckpt_dir),
            "best_val_loss": best_val_loss,
            "normalizer_path": str(normalizer_path),
            "target_variables": target_variables,
            "global_step": global_step,
        }
        if dist_ctx.is_rank0:
            with open(ckpt_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        return 0
    finally:
        shutdown_dataloader_workers(train_loader, logger=logger, name="train_loader")
        shutdown_dataloader_workers(test_loader, logger=logger, name="test_loader")
        if run is not None:
            safe_wandb_finish(run, logger=logger, timeout_seconds=wandb_finish_timeout_seconds)
        if dist_ctx.use_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
