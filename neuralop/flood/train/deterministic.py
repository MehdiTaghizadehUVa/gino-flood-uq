"""Deterministic L2 trainer with autoregressive fine-tuning support."""

from __future__ import annotations

import torch

from neuralop.training.trainer import Trainer
from neuralop.flood.processing.wv_impl import _build_x_from_dynamic_boundary


class DeterministicARTrainer(Trainer):
    """Trainer for deterministic GINO outputs with optional AR fine-tuning.

    The base ``Trainer`` is intentionally single-step. This subclass reuses the
    existing flood AR config keys and turns them into real deterministic rollout
    training once ``epoch >= ar_finetune_start_epoch``.
    """

    def __init__(
        self,
        rel_l2_loss_fn=None,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=1,
        ar_curriculum_epochs_per_step=0,
        ar_curriculum_start_steps=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rel_l2_loss_fn = rel_l2_loss_fn
        self.ar_finetune_start_epoch = max(0, int(ar_finetune_start_epoch))
        self.ar_rollout_steps = max(1, int(ar_rollout_steps))
        self.ar_curriculum_epochs_per_step = max(0, int(ar_curriculum_epochs_per_step))
        self.ar_curriculum_start_steps = max(1, int(ar_curriculum_start_steps))

    def _effective_ar_steps(self, max_available_steps: int) -> int:
        if self.ar_curriculum_epochs_per_step > 0:
            ar_epoch_index = max(0, self.epoch - self.ar_finetune_start_epoch)
            curriculum_step_index = ar_epoch_index // self.ar_curriculum_epochs_per_step
            effective_steps = min(
                self.ar_curriculum_start_steps + curriculum_step_index,
                self.ar_rollout_steps,
            )
        else:
            effective_steps = self.ar_rollout_steps
        return max(1, min(int(effective_steps), int(max_available_steps)))

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        structural_dry_mask = sample.get("structural_dry_mask")
        loss = training_loss(out, sample["y"], structural_dry_mask=structural_dry_mask)
        metrics = {
            "_log_loss_total": loss.detach(),
            "_log_loss_weight": float(sample["y"].shape[0]),
        }
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                rel_l2 = self.rel_l2_loss_fn(
                    out,
                    sample["y"],
                    structural_dry_mask=structural_dry_mask,
                )
            metrics.update(
                rel_l2=rel_l2,
                _log_rel_l2_total=rel_l2.detach(),
                _log_rel_l2_weight=float(sample["y"].shape[0]),
            )
        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"Deterministic single-step: loss = {loss.item():.8f} (B={sample['y'].shape[0]})")
        return loss, metrics

    @staticmethod
    def _slice_boundary_step(boundary_sequence, step: int, max_available_steps: int):
        if boundary_sequence.dim() == 5:
            time_dim = next(
                i for i, size in enumerate(boundary_sequence.shape) if size == max_available_steps
            )
            sl = [slice(None)] * boundary_sequence.dim()
            sl[time_dim] = slice(step, step + 1)
            bc_step = boundary_sequence[tuple(sl)].squeeze(time_dim)
        else:
            bc_step = boundary_sequence[:, step : step + 1]
        if bc_step.dim() == 3:
            bc_step = bc_step.unsqueeze(1)
        return bc_step

    def _require_ar_tensors(self, sample):
        missing = [
            key
            for key in ("target_sequence", "boundary_sequence", "dynamic", "boundary", "static")
            if key not in sample or sample[key] is None
        ]
        if missing:
            raise ValueError(
                "Deterministic AR fine-tuning requires rollout tensors "
                "target_sequence, boundary_sequence, dynamic, boundary, and static. "
                f"Missing: {missing}. Disable AR or build the dataset with ar_rollout_steps>1."
            )

    def train_one_batch(self, idx, sample, training_loss):
        if not getattr(self, "_skip_internal_zero_grad", False):
            self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}
        if not hasattr(self, "n_samples"):
            self.n_samples = 0

        ar_requested = self.ar_rollout_steps > 1 and self.epoch >= self.ar_finetune_start_epoch
        if ar_requested:
            self._require_ar_tensors(sample)
        else:
            self.n_samples += sample["y"].shape[0]
            loss, metrics = self._train_one_batch_single_step(idx, sample, training_loss)
            if self.regularizer:
                loss = loss + self.regularizer.loss
            return loss, metrics

        target_sequence = sample["target_sequence"]
        boundary_sequence = sample["boundary_sequence"]
        if target_sequence.dim() == 5:
            target_sequence = target_sequence.squeeze(1)
        if boundary_sequence.dim() == 5 and boundary_sequence.shape[1] == 1:
            boundary_sequence = boundary_sequence.squeeze(1)

        max_available_steps = target_sequence.shape[1]
        n_ar_steps = self._effective_ar_steps(max_available_steps)
        batch_size = sample["y"].shape[0]
        self.n_samples += batch_size * n_ar_steps

        n_history = sample["dynamic"].shape[1]
        dynamic_sliding = sample["dynamic"].clone()
        boundary_sliding = sample["boundary"].clone()
        static = sample["static"]
        geom = sample["input_geom"]
        q = sample["latent_queries"]
        out_q = sample["output_queries"]
        structural_dry_mask = sample.get("structural_dry_mask")

        total_loss = None
        rel_l2_total = None
        for step in range(n_ar_steps):
            x = _build_x_from_dynamic_boundary(static, boundary_sliding, dynamic_sliding)
            y_step = target_sequence[:, step]
            if y_step.dim() == 2:
                y_step = y_step.unsqueeze(0)
            kwargs = {
                "input_geom": geom,
                "latent_queries": q,
                "output_queries": out_q,
                "x": x,
            }
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    out = self.model(**kwargs)
            else:
                out = self.model(**kwargs)
            if self.data_processor is not None:
                out, _ = self.data_processor.postprocess(out, {**sample, "y": y_step})

            loss_step = training_loss(
                out,
                y_step,
                structural_dry_mask=structural_dry_mask,
            )
            total_loss = loss_step if total_loss is None else total_loss + loss_step

            if self.rel_l2_loss_fn is not None:
                with torch.no_grad():
                    rel_l2_step = self.rel_l2_loss_fn(
                        out,
                        y_step,
                        structural_dry_mask=structural_dry_mask,
                    )
                rel_l2_total = rel_l2_step if rel_l2_total is None else rel_l2_total + rel_l2_step

            dynamic_sliding = torch.cat([dynamic_sliding[:, 1:], out.unsqueeze(1)], dim=1)
            dynamic_sliding = dynamic_sliding[:, -n_history:]
            bc_step = self._slice_boundary_step(boundary_sequence, step, max_available_steps)
            boundary_sliding = torch.cat([boundary_sliding[:, 1:], bc_step], dim=1)[:, -n_history:]

        loss = total_loss / n_ar_steps
        metrics = {
            "_log_loss_total": total_loss.detach(),
            "_log_loss_weight": float(batch_size * n_ar_steps),
        }
        if rel_l2_total is not None:
            metrics.update(
                rel_l2=rel_l2_total / n_ar_steps,
                _log_rel_l2_total=rel_l2_total.detach(),
                _log_rel_l2_weight=float(batch_size * n_ar_steps),
            )
        if idx == 0 and self.logger is not None:
            self.logger.info(
                "Deterministic AR fine-tuning: epoch=%s, rollout_steps=%s (max=%s)%s.",
                self.epoch,
                n_ar_steps,
                self.ar_rollout_steps,
                " [curriculum]" if self.ar_curriculum_epochs_per_step > 0 else "",
            )

        if self.regularizer:
            loss = loss + self.regularizer.loss
        return loss, metrics
