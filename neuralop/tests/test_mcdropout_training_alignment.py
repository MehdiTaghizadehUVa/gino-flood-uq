import pytest
import torch
from torch import nn

from neuralop.training.trainer import Trainer
from neuralop.flood.train.deterministic import DeterministicARTrainer


class _Loader:
    dataset = [0]

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 1


class _ScriptedTrainer(Trainer):
    def __init__(self, eval_values, **kwargs):
        super().__init__(**kwargs)
        self._eval_values = iter(eval_values)
        self.epochs_run = []
        self.checkpoints = []

    def train_one_epoch(self, epoch, train_loader, training_loss):
        self.epochs_run.append(epoch)
        return 0.5, 0.5, None, 0.0

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        return {"test_l2": float(next(self._eval_values))}

    def checkpoint(self, save_dir, save_name):
        self.checkpoints.append(save_name)


def _sum_l2(out, y, structural_dry_mask=None):
    return ((out - y) ** 2).sum()


class _XLastChannelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, **kwargs):
        return self.scale * x[..., -1:]


def test_reduce_on_plateau_can_monitor_validation_metric():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0, threshold=0.0
    )
    trainer = _ScriptedTrainer(
        [1.0, 1.1],
        model=model,
        n_epochs=2,
        device="cpu",
        scheduler_monitor="test_l2",
    )

    trainer.train(
        _Loader(),
        {"test": _Loader()},
        optimizer,
        scheduler,
        training_loss=_sum_l2,
        eval_losses={"l2": _sum_l2},
        save_best="test_l2",
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert trainer.checkpoints == ["best_model"]


def test_invalid_validation_scheduler_monitor_fails_fast():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    trainer = _ScriptedTrainer(
        [1.0],
        model=model,
        n_epochs=1,
        device="cpu",
        scheduler_monitor="missing_l2",
    )

    with pytest.raises(ValueError, match="scheduler_monitor"):
        trainer.train(
            _Loader(),
            {"test": _Loader()},
            optimizer,
            scheduler,
            training_loss=_sum_l2,
            eval_losses={"l2": _sum_l2},
        )


def test_early_stopping_tracks_scheduler_monitor_and_saves_last_epoch():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = _ScriptedTrainer(
        [1.0, 1.1, 1.2, 1.3],
        model=model,
        n_epochs=4,
        device="cpu",
        scheduler_monitor="test_l2",
        early_stopping_enabled=True,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
    )

    trainer.train(
        _Loader(),
        {"test": _Loader()},
        optimizer,
        scheduler=None,
        training_loss=_sum_l2,
        eval_losses={"l2": _sum_l2},
        save_every=1,
        save_best="test_l2",
    )

    assert trainer.epochs_run == [0, 1, 2]
    assert trainer.checkpoints[-1] == "model"


def _ar_sample(batch=2, n_history=2, cells=3, steps=2):
    static = torch.zeros(batch, cells, 1)
    boundary = torch.zeros(batch, n_history, cells, 1)
    dynamic = torch.ones(batch, n_history, cells, 1)
    target_sequence = torch.full((batch, steps, cells, 1), 2.0)
    boundary_sequence = torch.zeros(batch, steps, cells, 1)
    return {
        "static": static,
        "boundary": boundary,
        "dynamic": dynamic,
        "target_sequence": target_sequence,
        "boundary_sequence": boundary_sequence,
        "input_geom": torch.zeros(batch, cells, 2),
        "latent_queries": torch.zeros(batch, cells, 2),
        "output_queries": torch.zeros(batch, cells, 2),
        "y": target_sequence[:, 0],
    }


def test_deterministic_ar_requires_sequence_tensors_after_threshold():
    trainer = DeterministicARTrainer(
        model=_XLastChannelModel(),
        n_epochs=1,
        device="cpu",
        rel_l2_loss_fn=_sum_l2,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=2,
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.regularizer = None
    trainer.epoch = 0

    sample = _ar_sample()
    sample.pop("target_sequence")

    with pytest.raises(ValueError, match="target_sequence"):
        trainer.train_one_batch(0, sample, _sum_l2)


def test_deterministic_ar_reports_rollout_step_weighted_metrics():
    trainer = DeterministicARTrainer(
        model=_XLastChannelModel(),
        n_epochs=1,
        device="cpu",
        rel_l2_loss_fn=_sum_l2,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=2,
    )
    trainer.regularizer = None
    trainer.epoch = 0
    trainer._skip_internal_zero_grad = True

    loss, metrics = trainer.train_one_batch(0, _ar_sample(), _sum_l2)

    assert trainer.n_samples == 4
    assert metrics["_log_loss_weight"] == 4.0
    assert metrics["_log_rel_l2_weight"] == 4.0
    assert metrics["_log_loss_total"].item() == pytest.approx(12.0)
    assert metrics["_log_rel_l2_total"].item() == pytest.approx(12.0)
    assert loss.item() == pytest.approx(6.0)
