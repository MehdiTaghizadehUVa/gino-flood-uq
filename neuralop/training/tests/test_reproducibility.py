import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralop import Trainer
from neuralop.tests.test_utils import DummyDataset, DummyModel
from neuralop.training.training_state import load_training_state, save_training_state


class RandomEvalTrainer(Trainer):
    def eval_one_batch(self, sample, eval_losses, return_output=False):
        self.n_samples += sample["y"].size(0)
        return {"metric": torch.rand((), device=sample["y"].device)}, None


def _stochastic_loss(out, y, **kwargs):
    return ((out - y) ** 2).mean() * (0.5 + torch.rand((), device=out.device))


def _make_training_stack(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = DummyDataset(32)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = DummyModel(50)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return loader, model, optimizer, scheduler


def test_training_state_restores_rng_progression(tmp_path):
    save_dir = tmp_path / "rng_state"
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    model = DummyModel(50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_training_state(save_dir=save_dir, save_name="model", model=model, optimizer=optimizer, epoch=0)

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(4)

    random.random()
    np.random.rand()
    torch.rand(4)

    restored_model = DummyModel(50)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    load_training_state(
        save_dir=save_dir,
        save_name="model",
        model=restored_model,
        optimizer=restored_optimizer,
        restore_rng_state_on_load=True,
    )

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_trainer_resume_restores_stochastic_training_progression(tmp_path):
    reference_loader, reference_model, reference_optimizer, reference_scheduler = _make_training_stack(7)
    reference_trainer = Trainer(model=reference_model, n_epochs=4)
    reference_trainer.train(
        train_loader=reference_loader,
        test_loaders={},
        optimizer=reference_optimizer,
        scheduler=reference_scheduler,
        training_loss=_stochastic_loss,
        save_every=None,
    )

    resume_dir = tmp_path / "resume_ckpt"
    partial_loader, partial_model, partial_optimizer, partial_scheduler = _make_training_stack(7)
    partial_trainer = Trainer(model=partial_model, n_epochs=2)
    partial_trainer.train(
        train_loader=partial_loader,
        test_loaders={},
        optimizer=partial_optimizer,
        scheduler=partial_scheduler,
        training_loss=_stochastic_loss,
        save_dir=resume_dir,
        save_every=1,
    )

    resume_loader, resumed_model, resumed_optimizer, resumed_scheduler = _make_training_stack(7)
    resumed_trainer = Trainer(model=resumed_model, n_epochs=4)
    resumed_trainer.train(
        train_loader=resume_loader,
        test_loaders={},
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        training_loss=_stochastic_loss,
        resume_from_dir=resume_dir,
    )

    for ref_param, resumed_param in zip(
        reference_trainer.model.parameters(),
        resumed_trainer.model.parameters(),
    ):
        assert torch.allclose(ref_param, resumed_param, atol=1e-7, rtol=0.0)

    assert reference_trainer.optimizer.param_groups[0]["lr"] == resumed_trainer.optimizer.param_groups[0]["lr"]


def test_trainer_deterministic_eval_seed_makes_stochastic_eval_repeatable():
    torch.manual_seed(3)
    loader = DataLoader(DummyDataset(8), batch_size=4, shuffle=False)
    trainer = RandomEvalTrainer(
        model=DummyModel(50),
        n_epochs=1,
        deterministic_eval=True,
        eval_seed=1234,
    )

    metric1 = trainer.evaluate({"metric": object()}, loader, log_prefix="test", epoch=2)
    metric2 = trainer.evaluate({"metric": object()}, loader, log_prefix="test", epoch=2)
    metric3 = trainer.evaluate({"metric": object()}, loader, log_prefix="test", epoch=3)

    assert metric1["test_metric"] == metric2["test_metric"]
    assert metric1["test_metric"] != metric3["test_metric"]
