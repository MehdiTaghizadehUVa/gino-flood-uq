import logging

import pytest
import torch
from torch import nn

from neuralop.flood.train.fgn import FGNTrainer


class TinyFGNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x=None, ada_in=None, **kwargs):
        if x is None or ada_in is None:
            raise ValueError("TinyFGNModel expects x and ada_in.")
        latent_bias = ada_in.mean(dim=1, keepdim=True).unsqueeze(1)
        return self.scale * x[..., :1] + latent_bias


def _make_sample(batch_size=1, n_cells=4, n_history=2, rollout_steps=2):
    geometry = torch.stack(
        [
            torch.linspace(0.0, 1.0, n_cells),
            torch.linspace(0.0, 1.0, n_cells),
        ],
        dim=1,
    ).unsqueeze(0)
    query = geometry.clone()
    static = torch.randn(batch_size, n_cells, 1)
    boundary = torch.randn(batch_size, n_history, n_cells, 1)
    dynamic = torch.randn(batch_size, n_history, n_cells, 1)
    target_sequence = torch.randn(batch_size, rollout_steps, n_cells, 1)
    boundary_sequence = torch.randn(batch_size, rollout_steps, n_cells, 1)
    return {
        "x": torch.randn(batch_size, n_cells, 1 + n_history + n_history),
        "y": target_sequence[:, 0],
        "static": static,
        "boundary": boundary,
        "dynamic": dynamic,
        "target_sequence": target_sequence,
        "boundary_sequence": boundary_sequence,
        "input_geom": geometry,
        "latent_queries": query,
        "output_queries": query,
    }


def _training_loss(pred_samples, y):
    return (pred_samples.mean(dim=0) - y).pow(2).mean()


def _make_trainer(*, gradient_mode: str) -> FGNTrainer:
    model = TinyFGNModel()
    trainer = FGNTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        mixed_precision=False,
        verbose=False,
        wandb_log=False,
        data_processor=None,
        fgn_noise_dim=4,
        crps_n_samples=2,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=2,
        ar_gradient_mode=gradient_mode,
        ar_truncation_steps=1,
        fgn_latent_temporal_mode="persistent",
        fgn_ar_state_update="member_feedback",
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=1.0)
    trainer.regularizer = None
    trainer.epoch = 0
    trainer.n_samples = 0
    trainer.logger = logging.getLogger(f"test_fgn_adaptive_{gradient_mode}")
    return trainer


def test_fgn_truncated_train_one_epoch_windowed_backward_updates_model():
    trainer = _make_trainer(gradient_mode="truncated")
    sample = _make_sample()
    before = trainer.model.scale.detach().clone()
    train_err, avg_loss, _, _ = trainer.train_one_epoch(0, [sample], _training_loss)
    after = trainer.model.scale.detach().clone()

    assert torch.isfinite(torch.tensor(train_err))
    assert torch.isfinite(torch.tensor(avg_loss))
    assert not torch.equal(before, after)


def test_fgn_full_ar_train_one_batch_defers_backward_to_trainer():
    trainer = _make_trainer(gradient_mode="full")
    trainer._skip_internal_zero_grad = True
    trainer.optimizer.zero_grad(set_to_none=True)
    sample = _make_sample()
    loss, metrics = trainer.train_one_batch(0, sample, _training_loss)

    assert "_backward_done" not in metrics
    assert loss.requires_grad
    assert trainer.model.scale.grad is None


def test_fgn_adaptive_retry_logs_and_counts_fallback(caplog):
    trainer = _make_trainer(gradient_mode="adaptive")
    trainer._last_ar_runtime_context = {
        "rollout_steps": 2,
        "gradient_mode": "full",
    }
    trainer.logger.propagate = True

    def _retry_train_one_batch(idx, sample, training_loss):
        assert trainer._force_truncated_next_batch is True
        return torch.tensor(0.5), {"rel_l2": torch.tensor(0.25), "_backward_done": True}

    trainer.train_one_batch = _retry_train_one_batch  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        loss, metrics = trainer.retry_batch_after_oom(3, _make_sample(), _training_loss)

    assert float(loss.item()) == pytest.approx(0.5)
    assert metrics["oom_fallback"] == pytest.approx(1.0)
    assert metrics["oom_fallback_total"] == pytest.approx(1.0)
    assert trainer._oom_fallback_count == 1
    assert trainer._force_truncated_next_batch is False
    assert "retry_mode=truncated" in caplog.text


def test_fgn_truncated_microbatch_retry_resets_sample_count_and_logs(caplog):
    trainer = _make_trainer(gradient_mode="adaptive")
    trainer._last_ar_runtime_context = {
        "rollout_steps": 2,
        "gradient_mode": "truncated",
    }
    trainer.logger.propagate = True
    sample = _make_sample(batch_size=4)
    calls = {"count": 0}

    def _microbatch_train_one_batch(idx, micro_sample, training_loss):
        trainer.n_samples += micro_sample["y"].shape[0] * 2
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return torch.tensor(0.5), {"rel_l2": torch.tensor(0.25), "_backward_done": True}

    trainer.train_one_batch = _microbatch_train_one_batch  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        loss, metrics = trainer._train_batch_via_truncated_microbatches(3, sample, _training_loss)

    assert float(loss.item()) == pytest.approx(0.5)
    assert trainer.n_samples == 8
    assert trainer._oom_microbatch_count == 1
    assert trainer._epoch_oom_microbatch_fallbacks == 1
    assert metrics["oom_fallback_microbatch"] == pytest.approx(1.0)
    assert metrics["oom_fallback_microbatch_chunks"] == pytest.approx(4.0)
    assert metrics["oom_fallback_microbatch_chunk_size"] == pytest.approx(1.0)
    assert "chunk_size=2 -> 1" in caplog.text
    assert "microbatch fallback succeeded" in caplog.text


def test_fgn_adaptive_retry_rolls_back_failed_truncated_sample_count_before_microbatch():
    trainer = _make_trainer(gradient_mode="adaptive")
    trainer._last_ar_runtime_context = {
        "rollout_steps": 2,
        "gradient_mode": "full",
    }
    sample = _make_sample(batch_size=2)
    trainer.n_samples = trainer._estimate_sample_increment(sample)
    seen = {}

    def _retry_train_one_batch(idx, sample_arg, training_loss):
        trainer.n_samples += trainer._estimate_sample_increment(sample_arg)
        raise RuntimeError("CUDA out of memory")

    def _microbatch_retry(idx, sample_arg, training_loss):
        seen["baseline_n_samples"] = trainer.n_samples
        trainer.n_samples += trainer._estimate_sample_increment(sample_arg)
        return torch.tensor(0.5), {"rel_l2": torch.tensor(0.25), "_backward_done": True}

    trainer.train_one_batch = _retry_train_one_batch  # type: ignore[method-assign]
    trainer._train_batch_via_truncated_microbatches = _microbatch_retry  # type: ignore[method-assign]

    loss, metrics = trainer.retry_batch_after_oom(1, sample, _training_loss)

    assert float(loss.item()) == pytest.approx(0.5)
    assert seen["baseline_n_samples"] == 0
    assert trainer.n_samples == trainer._estimate_sample_increment(sample)
    assert metrics["oom_fallback"] == pytest.approx(1.0)
