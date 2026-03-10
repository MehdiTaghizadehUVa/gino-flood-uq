"""Gaussian-head trainer implementation for WV flood models."""

from __future__ import annotations

import torch

from neuralop.training.trainer import Trainer

from neuralop.flood.processing.wv_impl import (
    _build_x_from_dynamic_boundary,
    _gaussian_mean_from_packed,
    _sample_from_packed_gaussian,
)

class GaussianNLLTrainer(Trainer):
    """
    Trainer for heteroscedastic Gaussian outputs packed as [mu, logvar].

    Supports single-step and autoregressive (AR) training.
    AR updates are sampled via reparameterization to preserve trajectory diversity.
    """

    def __init__(
        self,
        rel_l2_loss_fn=None,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=1,
        ar_curriculum_epochs_per_step=0,
        gaussian_min_logvar=-9.0,
        gaussian_max_logvar=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rel_l2_loss_fn = rel_l2_loss_fn
        self.ar_finetune_start_epoch = max(0, int(ar_finetune_start_epoch))
        self.ar_rollout_steps = max(1, int(ar_rollout_steps))
        self.ar_curriculum_epochs_per_step = max(0, int(ar_curriculum_epochs_per_step))
        self.gaussian_min_logvar = float(gaussian_min_logvar)
        self.gaussian_max_logvar = float(gaussian_max_logvar)

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = training_loss(out, sample["y"])
        metrics = {}
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                pred_mean = _gaussian_mean_from_packed(out, sample["y"].shape[-1])
                metrics["rel_l2"] = self.rel_l2_loss_fn(pred_mean, sample["y"])
        if self.epoch == 0 and idx == 0 and self.verbose:
            B = sample["y"].shape[0]
            print(f"Gaussian NLL single-step: loss = {loss.item():.8f} (B={B})")
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
            target_sequence = sample["target_sequence"]
            boundary_sequence = sample["boundary_sequence"]
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

            n_history = sample["dynamic"].shape[1]
            dynamic_sliding = sample["dynamic"].clone()
            boundary_sliding = sample["boundary"].clone()
            static = sample["static"]
            geom = sample["input_geom"]
            q = sample["latent_queries"]
            out_q = sample["output_queries"]

            total_loss = 0.0
            last_rel_l2 = None
            for s in range(n_ar_steps):
                x = _build_x_from_dynamic_boundary(static, boundary_sliding, dynamic_sliding)
                y_s = target_sequence[:, s]
                if y_s.dim() == 2:
                    y_s = y_s.unsqueeze(0)
                kwargs_base = {
                    "input_geom": geom,
                    "latent_queries": q,
                    "output_queries": out_q,
                    "x": x,
                }
                if self.mixed_precision:
                    with torch.autocast(device_type=self.autocast_device_type):
                        out = self.model(**kwargs_base)
                else:
                    out = self.model(**kwargs_base)
                if self.data_processor is not None:
                    out, _ = self.data_processor.postprocess(out, {**sample, "y": y_s})

                loss_s = training_loss(out, y_s)
                total_loss = total_loss + loss_s

                sampled_next, pred_mean, _ = _sample_from_packed_gaussian(
                    out,
                    n_channels=y_s.shape[-1],
                    min_logvar=self.gaussian_min_logvar,
                    max_logvar=self.gaussian_max_logvar,
                )
                if self.rel_l2_loss_fn is not None:
                    with torch.no_grad():
                        last_rel_l2 = self.rel_l2_loss_fn(pred_mean, y_s)

                dynamic_sliding = torch.cat(
                    [dynamic_sliding[:, 1:], sampled_next.unsqueeze(1)], dim=1
                )
                dynamic_sliding = dynamic_sliding[:, -n_history:]
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

            loss = total_loss / n_ar_steps
            metrics = {"rel_l2": last_rel_l2 if last_rel_l2 is not None else torch.tensor(0.0, device=self.device)}
            if idx == 0 and self.logger is not None:
                self.logger.info(
                    "Gaussian AR fine-tuning: epoch=%s, rollout_steps=%s (max=%s)%s.",
                    self.epoch,
                    n_ar_steps,
                    self.ar_rollout_steps,
                    " [curriculum]" if self.ar_curriculum_epochs_per_step > 0 else "",
                )
        else:
            self.n_samples += sample["y"].shape[0]
            loss, metrics = self._train_one_batch_single_step(idx, sample, training_loss)

        if self.regularizer:
            loss = loss + self.regularizer.loss
        return loss, metrics

    def eval_one_batch(self, sample, eval_losses, return_output=False):
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].size(0)
        out = self.model(**sample)
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        pred_mean = _gaussian_mean_from_packed(out, sample["y"].shape[-1])

        eval_step_losses = {}
        for loss_name, loss_fn in eval_losses.items():
            if loss_name == "l2":
                val = loss_fn(pred_mean, sample["y"])
            else:
                val = loss_fn(out, sample["y"])
            eval_step_losses[loss_name] = val

        return eval_step_losses, (pred_mean if return_output else None)


###############################################################################
# 7) NormalizedRolloutTestDataset
###############################################################################
