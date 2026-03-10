import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from neuralop.flood.eval.common import (
        _build_eval_losses,
        _make_trainer,
        _rollout_prediction_generic,
    )
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"Evaluator script import unavailable: {exc}", allow_module_level=True)


class DummyGaussianModel(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.logvar = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x=None, **kwargs):
        if x is None:
            raise ValueError("DummyGaussianModel expects x.")
        mu = x[..., : self.out_channels] + self.bias.view(1, 1, -1)
        logvar = self.logvar.view(1, 1, -1).expand_as(mu)
        return torch.cat([mu, logvar], dim=-1)


class DictDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class IdentityNormalizer:
    def __init__(self, n_channels: int):
        self.std = torch.ones(1, 1, n_channels, dtype=torch.float32)
        self.eps = 1e-7

    def to(self, device):
        self.std = self.std.to(device)
        return self

    def inverse_transform(self, x):
        return x


def _gaussian_config():
    return SimpleNamespace(
        gino=SimpleNamespace(output_distribution="gaussian", use_fgn_noise=False),
        opt=SimpleNamespace(
            training_loss="gaussian_nll",
            testing_loss="l2",
            n_epochs=1,
            scheduler_monitor="train_err",
            ar_finetune_start_epoch=0,
            ar_rollout_steps=1,
            ar_curriculum_epochs_per_step=0,
            gaussian_min_logvar=-9.0,
            gaussian_max_logvar=4.0,
            crps_channel_weights=[1.0],
        ),
        wandb=SimpleNamespace(eval_interval=1),
        use_progress_bar=False,
    )


def test_eval_gaussian_one_step_smoke():
    torch.manual_seed(0)
    cfg = _gaussian_config()
    model = DummyGaussianModel(out_channels=1)
    trainer = _make_trainer(
        config=cfg,
        model=model,
        data_processor=None,
        device=torch.device("cpu"),
        logger=None,
    )
    eval_losses = _build_eval_losses(cfg, use_fgn=False)
    ds = DictDataset(
        [
            {
                "x": torch.randn(8, 3),
                "y": torch.randn(8, 1),
            }
        ]
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    metrics = trainer.evaluate(eval_losses, loader, log_prefix="test")
    assert "test_l2" in metrics
    assert "test_gaussian_nll" in metrics
    assert torch.isfinite(torch.tensor(metrics["test_l2"]))
    assert torch.isfinite(torch.tensor(metrics["test_gaussian_nll"]))


def test_eval_gaussian_rollout_smoke(tmp_path: Path):
    torch.manual_seed(1)
    n_cells = 12
    n_steps = 5
    model = DummyGaussianModel(out_channels=1).eval()
    norm = IdentityNormalizer(n_channels=1)
    geometry = torch.stack(
        [
            torch.linspace(0.0, 1.0, n_cells),
            torch.linspace(0.0, 1.0, n_cells),
        ],
        dim=1,
    )
    sample = {
        "run_id": "smoke_run",
        "geometry": geometry,
        "static": torch.randn(n_cells, 2),
        "boundary": torch.randn(n_steps, n_cells, 1),
        "dynamic": torch.randn(n_steps, n_cells, 1),
        "query_points": torch.randn(6, 6, 2),
    }
    out_dir = str(tmp_path / "rollout")
    _rollout_prediction_generic(
        models=[model],
        rollout_dataset=[sample],
        rollout_length=2,
        history_steps=2,
        dynamic_norm=norm,
        target_norm=norm,
        device=torch.device("cpu"),
        skip_before_timestep=0,
        dt=60.0,
        out_dir=out_dir,
        target_variables=["wd"],
        logger=logging.getLogger("test_eval_gaussian"),
        fgn_noise_dim=None,
        n_ensemble_samples=2,
        gaussian_mode=True,
        gaussian_min_logvar=-9.0,
        gaussian_max_logvar=4.0,
    )
    assert (tmp_path / "rollout" / "rollout_metrics_data.npz").exists()
