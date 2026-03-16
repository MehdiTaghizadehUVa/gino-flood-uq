"""FGN rollout latent helpers and trainer implementation."""

from __future__ import annotations

import math

import torch

from neuralop.losses.data_losses import LpLoss
from neuralop.losses.probabilistic_losses import CRPSLoss, fair_crps_univariate
from neuralop.training.trainer import Trainer

from neuralop.flood.processing.wv_impl import (
    _build_x_from_dynamic_boundary,
    compute_hazard_proxy_pooled,
    get_flood_crps_weights,
)
from neuralop.flood.utils.runtime_core import (
    _cfg_get,
    normalize_fgn_ar_state_update,
    normalize_fgn_latent_temporal_mode,
)

def sample_fgn_rollout_latent_bank(
    num_members,
    batch_size,
    latent_dim,
    device,
    dtype,
    temporal_mode="stepwise",
):
    temporal_mode = normalize_fgn_latent_temporal_mode(temporal_mode)
    if temporal_mode != "persistent":
        return None
    return torch.randn(
        int(num_members),
        int(batch_size),
        int(latent_dim),
        device=device,
        dtype=dtype,
    )


def get_fgn_rollout_latent(
    latent_bank,
    member_idx,
    batch_size,
    latent_dim,
    device,
    dtype,
):
    if latent_bank is not None:
        return latent_bank[int(member_idx)]
    return torch.randn(
        int(batch_size),
        int(latent_dim),
        device=device,
        dtype=dtype,
    )


def update_fgn_dynamic_members(
    dynamic_members,
    pred_samples,
    pred_mean,
    n_history,
    state_update_mode="mean_feedback",
):
    state_update_mode = normalize_fgn_ar_state_update(state_update_mode)
    if pred_samples.ndim != 4:
        raise ValueError(
            f"pred_samples must have shape [N, B, n_cells, C], got {tuple(pred_samples.shape)}."
        )
    if pred_mean.shape != pred_samples.shape[1:]:
        raise ValueError(
            f"pred_mean shape {tuple(pred_mean.shape)} must match pred_samples[0] shape "
            f"{tuple(pred_samples.shape[1:])}."
        )
    if len(dynamic_members) != pred_samples.shape[0]:
        raise ValueError(
            f"dynamic_members length {len(dynamic_members)} must match ensemble size "
            f"{pred_samples.shape[0]}."
        )
    updated_members = []
    for member_idx, dyn_hist in enumerate(dynamic_members):
        next_state = pred_samples[member_idx] if state_update_mode == "member_feedback" else pred_mean
        updated_hist = torch.cat([dyn_hist[:, 1:], next_state.unsqueeze(1)], dim=1)
        updated_members.append(updated_hist[:, -n_history:])
    return updated_members

class FGNTrainer(Trainer):
    """
    Trainer for FGN (Functional Generative Networks): two forward passes per batch
    with different noise z, then CRPS loss on (out1, out2, y).

    Same data sample (x, y) is used for both passes; only the noise z differs.
    Returns (loss, {'rel_l2': rel_l2}). rel_l2 should be sum-over-batch so train_err = sum(rel_l2)/n_samples
    (mean per sample, same scale as test_l2). Use LpLoss with reduction='sum' for rel_l2_loss_fn.

    Supports autoregressive (AR) fine-tuning: when epoch >= ar_finetune_start_epoch and
    ar_rollout_steps > 1, runs an AR rollout, computes loss at each step, averages over
    steps, and backpropagates through the rollout (FGN-style). Optional curriculum: set
    ar_curriculum_epochs_per_step > 0 to ramp rollout length (1 step for E epochs, then
    2 steps for E epochs, ... up to ar_rollout_steps).
    """

    def __init__(
        self,
        fgn_noise_dim=32,
        crps_n_samples=2,
        rel_l2_loss_fn=None,
        crps_l2_weight=0.0,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=1,
        ar_curriculum_epochs_per_step=0,
        use_flood_crps_spatial_weights=False,
        flood_crps_wet_threshold=0.01,
        flood_crps_wet_smooth_scale=0.02,
        flood_crps_dry_weight_alpha=0.1,
        static_normalizer=None,
        use_hazard_proxy_crps=False,
        hazard_proxy_crps_weight=0.15,
        ar_pooled_crps_gamma=1.0,
        ar_gradient_mode="full",
        ar_truncation_steps=1,
        crps_sample_chunk_size=1,
        use_activation_checkpointing=False,
        fgn_latent_temporal_mode="stepwise",
        fgn_ar_state_update="mean_feedback",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fgn_noise_dim = fgn_noise_dim
        self.crps_n_samples = max(2, int(crps_n_samples))
        self.rel_l2_loss_fn = rel_l2_loss_fn
        self.crps_l2_weight = float(crps_l2_weight)
        self.ar_finetune_start_epoch = max(0, int(ar_finetune_start_epoch))
        self.ar_rollout_steps = max(1, int(ar_rollout_steps))
        self.ar_curriculum_epochs_per_step = max(0, int(ar_curriculum_epochs_per_step))
        self.use_flood_crps_spatial_weights = bool(use_flood_crps_spatial_weights)
        self.flood_crps_wet_threshold = float(flood_crps_wet_threshold)
        self.flood_crps_wet_smooth_scale = float(flood_crps_wet_smooth_scale)
        self.flood_crps_dry_weight_alpha = float(flood_crps_dry_weight_alpha)
        self.static_normalizer = static_normalizer
        self.use_hazard_proxy_crps = bool(use_hazard_proxy_crps)
        self.hazard_proxy_crps_weight = float(hazard_proxy_crps_weight)
        self.ar_pooled_crps_gamma = float(ar_pooled_crps_gamma)
        self.ar_gradient_mode = str(ar_gradient_mode).strip().lower()
        if self.ar_gradient_mode not in {"full", "adaptive", "truncated"}:
            raise ValueError(
                f"Unknown opt.ar_gradient_mode={ar_gradient_mode!r}. Use 'full', 'adaptive', or 'truncated'."
            )
        self.ar_truncation_steps = max(1, int(ar_truncation_steps))
        self.crps_sample_chunk_size = max(1, int(crps_sample_chunk_size))
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.fgn_latent_temporal_mode = normalize_fgn_latent_temporal_mode(
            fgn_latent_temporal_mode
        )
        self.fgn_ar_state_update = normalize_fgn_ar_state_update(
            fgn_ar_state_update
        )
        self._force_truncated_next_batch = False
        self._oom_fallback_count = 0
        self._oom_microbatch_count = 0
        self._epoch_oom_microbatch_fallbacks = 0
        self._last_ar_runtime_context = {}
        self._current_batch_weight = 1.0

    def _forward_fgn(self, kwargs_base, z):
        if not self.use_activation_checkpointing:
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    return self.model(**kwargs_base, ada_in=z)
            return self.model(**kwargs_base, ada_in=z)

        def _forward_ckpt(x, input_geom, latent_queries, output_queries, ada_in):
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    return self.model(
                        x=x,
                        input_geom=input_geom,
                        latent_queries=latent_queries,
                        output_queries=output_queries,
                        ada_in=ada_in,
                    )
            return self.model(
                x=x,
                input_geom=input_geom,
                latent_queries=latent_queries,
                output_queries=output_queries,
                ada_in=ada_in,
            )

        return torch.utils.checkpoint.checkpoint(
            _forward_ckpt,
            kwargs_base["x"],
            kwargs_base["input_geom"],
            kwargs_base["latent_queries"],
            kwargs_base["output_queries"],
            z,
            use_reentrant=False,
        )

    def retry_batch_after_oom(self, idx, sample, training_loss):
        is_ar_active = self.ar_rollout_steps > 1 and self.epoch >= self.ar_finetune_start_epoch
        if self.ar_gradient_mode != "adaptive" or not is_ar_active:
            raise RuntimeError("OOM retry requested but opt.ar_gradient_mode is not 'adaptive'.")
        ar_ctx = dict(getattr(self, "_last_ar_runtime_context", {}) or {})
        failed_sample_increment = self._estimate_sample_increment(sample)
        baseline_n_samples = max(0, int(self.n_samples) - int(failed_sample_increment))
        self.n_samples = baseline_n_samples
        self._force_truncated_next_batch = True
        retry_mode = "truncated"
        try:
            loss, metrics = self.train_one_batch(idx, sample, training_loss)
        except Exception as exc:
            if self._is_cuda_oom(exc):
                self._clear_cuda_oom_state()
                self.n_samples = baseline_n_samples
                retry_mode = "truncated-microbatch"
                loss, metrics = self._train_batch_via_truncated_microbatches(
                    idx, sample, training_loss
                )
            else:
                raise RuntimeError(
                    (
                        "Adaptive AR retry failed "
                        f"(epoch={getattr(self, 'epoch', -1)}, batch={idx}, "
                        f"rollout_steps={ar_ctx.get('rollout_steps', 'unknown')}, "
                        f"original_mode={ar_ctx.get('gradient_mode', 'full')}, "
                        f"retry_mode=truncated, detach_every={self.ar_truncation_steps})."
                    )
                ) from exc
        finally:
            self._force_truncated_next_batch = False
        self._oom_fallback_count += 1
        if isinstance(metrics, dict):
            metrics["oom_fallback"] = 1.0
            metrics["oom_fallback_total"] = float(self._oom_fallback_count)
        if self.logger is not None:
            self.logger.warning(
                (
                    "Adaptive AR OOM fallback triggered at epoch=%s batch=%s "
                    "(rollout_steps=%s, original_mode=%s, retry_mode=%s, detach_every=%s, total=%s)."
                ),
                getattr(self, "epoch", -1),
                idx,
                ar_ctx.get("rollout_steps", "unknown"),
                ar_ctx.get("gradient_mode", "full"),
                retry_mode,
                self.ar_truncation_steps,
                self._oom_fallback_count,
            )
        return loss, metrics

    def _backward_ar_window(self, window_loss, *, n_ar_steps: int):
        if window_loss is None:
            return
        accum_divisor = max(1, int(getattr(self, "_current_accum_divisor", 1)))
        batch_weight = float(getattr(self, "_current_batch_weight", 1.0))
        scaled_loss = (window_loss * batch_weight) / (
            float(n_ar_steps) * float(accum_divisor)
        )
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _estimate_sample_increment(self, sample):
        batch_size = None
        if isinstance(sample, dict):
            for key in ("y", "dynamic", "x"):
                value = sample.get(key, None)
                if torch.is_tensor(value) and value.dim() > 0:
                    batch_size = int(value.shape[0])
                    break
        if batch_size is None:
            return 0
        use_ar = (
            self.ar_rollout_steps > 1
            and self.epoch >= self.ar_finetune_start_epoch
            and "target_sequence" in sample
            and sample["target_sequence"] is not None
        )
        if not use_ar:
            return batch_size
        target_sequence = sample["target_sequence"]
        if target_sequence.dim() == 5:
            target_sequence = target_sequence.squeeze(1)
        max_available_steps = int(target_sequence.shape[1])
        if self.ar_curriculum_epochs_per_step > 0:
            ar_epoch_index = self.epoch - self.ar_finetune_start_epoch
            curriculum_step_index = ar_epoch_index // self.ar_curriculum_epochs_per_step
            effective_ar_steps = min(curriculum_step_index + 1, self.ar_rollout_steps)
        else:
            effective_ar_steps = self.ar_rollout_steps
        n_ar_steps = min(int(effective_ar_steps), max_available_steps)
        return batch_size * n_ar_steps

    @staticmethod
    def _slice_batch_like(sample, start: int, end: int):
        if not isinstance(sample, dict):
            raise TypeError("Expected sample to be a dict for microbatch slicing.")
        ref_batch = None
        for key in ("y", "dynamic", "x"):
            value = sample.get(key, None)
            if torch.is_tensor(value) and value.dim() > 0:
                ref_batch = int(value.shape[0])
                break
        if ref_batch is None:
            raise ValueError("Unable to infer batch size for microbatch slicing.")
        subset = {}
        for key, value in sample.items():
            if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == ref_batch:
                subset[key] = value[start:end]
            else:
                subset[key] = value
        return subset

    def _train_batch_via_truncated_microbatches(self, idx, sample, training_loss):
        full_batch_size = int(sample["y"].shape[0])
        if full_batch_size <= 1:
            raise RuntimeError(
                "Adaptive AR microbatch fallback cannot shrink a batch of size 1 any further."
            )
        retry_ctx = dict(getattr(self, "_last_ar_runtime_context", {}) or {})
        chunk_size = max(1, full_batch_size // 2)
        baseline_n_samples = int(getattr(self, "n_samples", 0))
        prev_batch_weight = float(getattr(self, "_current_batch_weight", 1.0))
        self._force_truncated_next_batch = True
        try:
            while True:
                self.n_samples = baseline_n_samples
                self._clear_cuda_oom_state()
                total_loss_scalar = 0.0
                total_rel_l2 = 0.0
                chunk_counter = 0
                try:
                    for start in range(0, full_batch_size, chunk_size):
                        end = min(full_batch_size, start + chunk_size)
                        batch_weight = float(end - start) / float(full_batch_size)
                        self._current_batch_weight = batch_weight
                        chunk_sample = self._slice_batch_like(sample, start, end)
                        loss, metrics = self.train_one_batch(idx, chunk_sample, training_loss)
                        if not isinstance(metrics, dict) or not metrics.get("_backward_done", False):
                            raise RuntimeError(
                                "Adaptive AR microbatch fallback expected truncated backward "
                                "to complete inside the trainer."
                            )
                        total_loss_scalar += float(loss.detach().item()) * batch_weight
                        rel_l2 = metrics.get("rel_l2", None)
                        if rel_l2 is not None:
                            total_rel_l2 += float(
                                rel_l2.detach().item() if torch.is_tensor(rel_l2) else rel_l2
                            )
                        chunk_counter += 1
                    self._oom_microbatch_count += 1
                    self._epoch_oom_microbatch_fallbacks += 1
                    if self.logger is not None:
                        self.logger.warning(
                            (
                                "Adaptive AR microbatch fallback succeeded at epoch=%s batch=%s "
                                "(rollout_steps=%s, chunk_size=%s, chunks=%s, total=%s)."
                            ),
                            getattr(self, "epoch", -1),
                            idx,
                            retry_ctx.get("rollout_steps", "unknown"),
                            chunk_size,
                            chunk_counter,
                            self._oom_microbatch_count,
                        )
                    return (
                        torch.tensor(
                            total_loss_scalar,
                            device=self.device,
                            dtype=sample["y"].dtype,
                        ),
                        {
                            "rel_l2": torch.tensor(
                                total_rel_l2, device=self.device, dtype=sample["y"].dtype
                            ),
                            "_backward_done": True,
                            "oom_fallback_microbatch": 1.0,
                            "oom_fallback_microbatch_total": float(self._oom_microbatch_count),
                            "oom_fallback_microbatch_chunks": float(chunk_counter),
                            "oom_fallback_microbatch_chunk_size": float(chunk_size),
                        },
                    )
                except RuntimeError as exc:
                    if self._is_cuda_oom(exc) and chunk_size > 1:
                        next_chunk_size = max(1, math.ceil(chunk_size / 2))
                        if next_chunk_size == chunk_size:
                            next_chunk_size = max(1, chunk_size - 1)
                        if self.logger is not None:
                            self.logger.warning(
                                (
                                    "Adaptive AR microbatch fallback reduced chunk size after OOM "
                                    "(epoch=%s batch=%s rollout_steps=%s chunk_size=%s -> %s)."
                                ),
                                getattr(self, "epoch", -1),
                                idx,
                                retry_ctx.get("rollout_steps", "unknown"),
                                chunk_size,
                                next_chunk_size,
                            )
                        chunk_size = next_chunk_size
                        continue
                    self.n_samples = baseline_n_samples
                    raise RuntimeError(
                        (
                            "Adaptive AR microbatch retry failed "
                            f"(epoch={getattr(self, 'epoch', -1)}, batch={idx}, "
                            f"rollout_steps={retry_ctx.get('rollout_steps', 'unknown')}, "
                            f"retry_mode=truncated-microbatch, chunk_size={chunk_size}, "
                            f"detach_every={self.ar_truncation_steps})."
                        )
                    ) from exc
        finally:
            self._current_batch_weight = prev_batch_weight
            self._force_truncated_next_batch = False

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        """Single-step FGN: n_crps forward passes with different z, loss on (pred_samples, y)."""
        n_crps = self.crps_n_samples
        outs = []
        batch_size = sample["x"].shape[0]
        kwargs_base = {
            "x": sample["x"],
            "input_geom": sample["input_geom"],
            "latent_queries": sample["latent_queries"],
            "output_queries": sample["output_queries"],
        }
        for _start in range(0, n_crps, self.crps_sample_chunk_size):
            chunk_n = min(self.crps_sample_chunk_size, n_crps - _start)
            for _ in range(chunk_n):
                z = torch.randn(
                    batch_size, self.fgn_noise_dim, device=self.device, dtype=sample["x"].dtype
                )
                out = self._forward_fgn(kwargs_base, z)
                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)
                outs.append(out)
        pred_samples = torch.stack(outs, dim=0)
        pred_mean = pred_samples.mean(dim=0)
        y_target = sample["y"]
        structural_dry_mask = sample.get("structural_dry_mask")
        if self.use_flood_crps_spatial_weights and "static" in sample and y_target.shape[-1] >= 3:
            spatial_weights = get_flood_crps_weights(
                sample["static"],
                y_target,
                wet_threshold=self.flood_crps_wet_threshold,
                wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                dry_weight_alpha=self.flood_crps_dry_weight_alpha,
                static_normalizer=self.static_normalizer,
            )
            loss = training_loss(
                pred_samples,
                y_target,
                spatial_weights=spatial_weights,
                structural_dry_mask=structural_dry_mask,
            )
        else:
            loss = training_loss(
                pred_samples,
                y_target,
                structural_dry_mask=structural_dry_mask,
            )
        if self.crps_l2_weight > 0 and self.rel_l2_loss_fn is not None:
            loss = loss + self.crps_l2_weight * self.rel_l2_loss_fn(
                pred_mean,
                sample["y"],
                structural_dry_mask=structural_dry_mask,
            )
        if self.use_hazard_proxy_crps and "static" in sample and y_target.shape[-1] >= 3:
            pred_pooled = compute_hazard_proxy_pooled(
                sample["static"],
                pred_samples,
                wet_threshold=self.flood_crps_wet_threshold,
                wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                static_normalizer=self.static_normalizer,
            )
            y_pooled = compute_hazard_proxy_pooled(
                sample["static"],
                y_target,
                wet_threshold=self.flood_crps_wet_threshold,
                wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                static_normalizer=self.static_normalizer,
            )
            crps_pooled = fair_crps_univariate(pred_pooled, y_pooled).mean()
            loss = loss + self.hazard_proxy_crps_weight * crps_pooled
        metrics = {}
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                metrics["rel_l2"] = self.rel_l2_loss_fn(
                    pred_mean,
                    sample["y"],
                    structural_dry_mask=structural_dry_mask,
                )
        return loss, metrics

    def _train_one_batch_ar(self, idx, sample, training_loss):
        target_sequence = sample["target_sequence"]
        boundary_sequence = sample["boundary_sequence"]
        # Normalize to (B, T, ...) only if collate produced 5D (B, 1, T, n_cells, C)
        if target_sequence.dim() == 5:
            target_sequence = target_sequence.squeeze(1)
        if boundary_sequence.dim() == 5:
            boundary_sequence = boundary_sequence.squeeze(1)
        max_available_steps = target_sequence.shape[1]
        if self.ar_curriculum_epochs_per_step > 0:
            ar_epoch_index = self.epoch - self.ar_finetune_start_epoch
            curriculum_step_index = ar_epoch_index // self.ar_curriculum_epochs_per_step
            effective_ar_steps = min(curriculum_step_index + 1, self.ar_rollout_steps)
        else:
            effective_ar_steps = self.ar_rollout_steps
        n_ar_steps = min(effective_ar_steps, max_available_steps)
        self.n_samples += sample["y"].shape[0] * n_ar_steps

        gradient_mode = self.ar_gradient_mode
        if gradient_mode == "adaptive":
            gradient_mode = "truncated" if self._force_truncated_next_batch else "full"
        detach_every = self.ar_truncation_steps if gradient_mode == "truncated" else 0
        self._last_ar_runtime_context = {
            "batch_idx": int(idx),
            "rollout_steps": int(n_ar_steps),
            "gradient_mode": str(gradient_mode),
            "detach_every": int(detach_every),
        }

        n_history = sample["dynamic"].shape[1]
        dynamic_members = [sample["dynamic"].clone() for _ in range(self.crps_n_samples)]
        boundary_sliding = sample["boundary"].clone()
        static = sample["static"]
        geom = sample["input_geom"]
        q = sample["latent_queries"]
        out_q = sample["output_queries"]

        total_loss = None
        total_loss_scalar = 0.0
        window_loss = None
        last_rel_l2 = None
        n_crps = self.crps_n_samples
        latent_bank = sample_fgn_rollout_latent_bank(
            num_members=n_crps,
            batch_size=sample["dynamic"].shape[0],
            latent_dim=self.fgn_noise_dim,
            device=self.device,
            dtype=sample["dynamic"].dtype,
            temporal_mode=self.fgn_latent_temporal_mode,
        )
        for s in range(n_ar_steps):
            y_s = target_sequence[:, s]
            if y_s.dim() == 2:
                y_s = y_s.unsqueeze(0)
            structural_dry_mask = sample.get("structural_dry_mask")
            outs_s = []
            for _start in range(0, n_crps, self.crps_sample_chunk_size):
                chunk_n = min(self.crps_sample_chunk_size, n_crps - _start)
                for member_idx in range(_start, _start + chunk_n):
                    x = _build_x_from_dynamic_boundary(
                        static, boundary_sliding, dynamic_members[member_idx]
                    )
                    kwargs_base = {
                        "input_geom": geom,
                        "latent_queries": q,
                        "output_queries": out_q,
                        "x": x,
                    }
                    z = get_fgn_rollout_latent(
                        latent_bank=latent_bank,
                        member_idx=member_idx,
                        batch_size=x.shape[0],
                        latent_dim=self.fgn_noise_dim,
                        device=self.device,
                        dtype=x.dtype,
                    )
                    out = self._forward_fgn(kwargs_base, z)
                    if self.data_processor is not None:
                        out, _ = self.data_processor.postprocess(out, {**sample, "y": y_s})
                    outs_s.append(out)
            pred_samples = torch.stack(outs_s, dim=0)
            pred_mean = pred_samples.mean(dim=0)
            if self.use_flood_crps_spatial_weights and "static" in sample and y_s.shape[-1] >= 3:
                spatial_weights_s = get_flood_crps_weights(
                    static,
                    y_s,
                    wet_threshold=self.flood_crps_wet_threshold,
                    wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                    dry_weight_alpha=self.flood_crps_dry_weight_alpha,
                    static_normalizer=self.static_normalizer,
                )
                loss_s = training_loss(
                    pred_samples,
                    y_s,
                    spatial_weights=spatial_weights_s,
                    structural_dry_mask=structural_dry_mask,
                )
            else:
                loss_s = training_loss(
                    pred_samples,
                    y_s,
                    structural_dry_mask=structural_dry_mask,
                )
            if self.crps_l2_weight > 0 and self.rel_l2_loss_fn is not None:
                loss_s = loss_s + self.crps_l2_weight * self.rel_l2_loss_fn(
                    pred_mean,
                    y_s,
                    structural_dry_mask=structural_dry_mask,
                )
            if self.use_hazard_proxy_crps and y_s.shape[-1] >= 3:
                pred_pooled_s = compute_hazard_proxy_pooled(
                    static,
                    pred_samples,
                    wet_threshold=self.flood_crps_wet_threshold,
                    wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                    static_normalizer=self.static_normalizer,
                )
                y_pooled_s = compute_hazard_proxy_pooled(
                    static,
                    y_s,
                    wet_threshold=self.flood_crps_wet_threshold,
                    wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                    static_normalizer=self.static_normalizer,
                )
                gamma_s = self.ar_pooled_crps_gamma ** s
                loss_s = loss_s + gamma_s * self.hazard_proxy_crps_weight * fair_crps_univariate(pred_pooled_s, y_pooled_s).mean()
            if gradient_mode != "truncated":
                total_loss = loss_s if total_loss is None else (total_loss + loss_s)
            total_loss_scalar += float(loss_s.detach().item())
            if gradient_mode == "truncated":
                window_loss = loss_s if window_loss is None else (window_loss + loss_s)
            if self.rel_l2_loss_fn is not None:
                with torch.no_grad():
                    last_rel_l2 = self.rel_l2_loss_fn(
                        pred_mean,
                        y_s,
                        structural_dry_mask=structural_dry_mask,
                    )
            dynamic_members = update_fgn_dynamic_members(
                dynamic_members=dynamic_members,
                pred_samples=pred_samples,
                pred_mean=pred_mean,
                n_history=n_history,
                state_update_mode=self.fgn_ar_state_update,
            )
            reached_truncation_boundary = detach_every > 0 and ((s + 1) % detach_every == 0)
            final_step = (s + 1) == n_ar_steps
            if gradient_mode == "truncated" and (reached_truncation_boundary or final_step):
                self._backward_ar_window(window_loss, n_ar_steps=n_ar_steps)
                window_loss = None
                dynamic_members = [hist.detach() for hist in dynamic_members]
            if boundary_sequence.dim() == 5:
                time_dim = next(i for i, sz in enumerate(boundary_sequence.shape) if sz == max_available_steps)
                sl = [slice(None)] * boundary_sequence.dim()
                sl[time_dim] = slice(s, s + 1)
                bc_step = boundary_sequence[tuple(sl)].squeeze(time_dim)
            else:
                bc_step = boundary_sequence[:, s : s + 1]
            if bc_step.dim() == 3:
                bc_step = bc_step.unsqueeze(1)
            boundary_sliding = torch.cat([boundary_sliding[:, 1:], bc_step], dim=1)[:, -n_history:]

        if gradient_mode == "truncated":
            loss = torch.tensor(
                total_loss_scalar / float(n_ar_steps),
                device=self.device,
                dtype=sample["dynamic"].dtype,
            )
        else:
            loss = total_loss / n_ar_steps
        metrics = {
            "rel_l2": last_rel_l2 if last_rel_l2 is not None else torch.tensor(0.0, device=self.device),
        }
        if gradient_mode == "truncated":
            metrics["_backward_done"] = True
        if idx == 0 and self.logger is not None:
            self.logger.info(
                "AR fine-tuning: epoch=%s, rollout_steps=%s (max=%s)%s, mode=%s%s.",
                self.epoch,
                n_ar_steps,
                self.ar_rollout_steps,
                " [curriculum]" if self.ar_curriculum_epochs_per_step > 0 else "",
                gradient_mode,
                f", detach_every={detach_every}" if detach_every > 0 else "",
            )
            self.logger.info(
                "FGN AR semantics: latent_temporal_mode=%s, state_update=%s, crps_n_samples=%s, fgn_noise_dim=%s",
                self.fgn_latent_temporal_mode,
                self.fgn_ar_state_update,
                self.crps_n_samples,
                self.fgn_noise_dim,
            )
        return loss, metrics

    def train_one_batch(self, idx, sample, training_loss):
        if not getattr(self, "_skip_internal_zero_grad", False):
            self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        use_ar = (
            self.ar_rollout_steps > 1
            and self.epoch >= self.ar_finetune_start_epoch
            and "target_sequence" in sample
            and sample["target_sequence"] is not None
        )
        if use_ar:
            loss, metrics = self._train_one_batch_ar(idx, sample, training_loss)
        else:
            self.n_samples += sample["y"].shape[0]
            loss, metrics = self._train_one_batch_single_step(idx, sample, training_loss)

        if self.epoch == 0 and idx == 0 and self.verbose:
            B = sample["y"].shape[0]
            print(f"FGN {'AR' if use_ar else 'single-step'}: loss = {loss.item():.8f} (B={B})")

        if self.regularizer:
            regularizer_loss = self.regularizer.loss
            if isinstance(metrics, dict) and metrics.get("_backward_done", False):
                if torch.is_tensor(regularizer_loss) and regularizer_loss.requires_grad:
                    accum_divisor = max(1, int(getattr(self, "_current_accum_divisor", 1)))
                    batch_weight = float(getattr(self, "_current_batch_weight", 1.0))
                    scaled_reg = (regularizer_loss * batch_weight) / float(accum_divisor)
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled_reg).backward()
                    else:
                        scaled_reg.backward()
                loss = loss + regularizer_loss.detach() * float(
                    getattr(self, "_current_batch_weight", 1.0)
                )
            else:
                loss = loss + regularizer_loss
        return loss, metrics

    def eval_one_batch(self, sample, eval_losses, return_output=False):
        """FGN eval: crps_n_samples forward passes with different z; L2 on ensemble mean, CRPS on pred_samples."""
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].size(0)
        n_crps = self.crps_n_samples
        outs = []
        y_eval = sample["y"]
        batch_size = sample["x"].shape[0]
        for _ in range(n_crps):
            z = torch.randn(
                batch_size, self.fgn_noise_dim, device=self.device, dtype=sample["x"].dtype
            )
            samp = {**sample, "ada_in": z}
            out = self.model(**samp)
            if self.data_processor is not None:
                # Important: avoid repeatedly inverse-transforming the same y across ensemble
                # members when inverse_test=True. Use an isolated sample dict per pass.
                sample_for_post = {"y": sample["y"]}
                out, sample_post = self.data_processor.postprocess(out, sample_for_post)
                y_eval = sample_post["y"]
            outs.append(out)
        pred_samples = torch.stack(outs, dim=0)
        pred_mean = pred_samples.mean(dim=0)

        eval_step_losses = {}
        structural_dry_mask = sample.get("structural_dry_mask")
        for loss_name, loss_fn in eval_losses.items():
            if getattr(loss_fn, "expects_samples", False) or isinstance(loss_fn, CRPSLoss) or loss_name == "crps":
                val = loss_fn(
                    pred_samples,
                    y_eval,
                    structural_dry_mask=structural_dry_mask,
                )
            else:
                val = loss_fn(
                    pred_mean,
                    y_eval,
                    structural_dry_mask=structural_dry_mask,
                )
            eval_step_losses[loss_name] = val

        out = pred_mean if return_output else None
        return eval_step_losses, out
