import logging
from pathlib import Path

import pytest
import torch
from torch import nn

try:
    from neuralop.flood.train.operator import (
        get_fgn_rollout_latent,
        sample_fgn_rollout_latent_bank,
        update_fgn_dynamic_members,
    )
    from neuralop.flood.eval.common import (
        _rollout_prediction_generic,
        _rollout_prediction_per_hydrograph,
    )
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"FGN rollout scripts unavailable: {exc}", allow_module_level=True)


class IdentityNormalizer:
    def to(self, device):
        return self

    def inverse_transform(self, x):
        return x


class DummyFGNModel(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.ada_calls = []

    def forward(self, x=None, ada_in=None, **kwargs):
        if x is None:
            raise ValueError("DummyFGNModel expects x.")
        if ada_in is None:
            raise ValueError("DummyFGNModel expects ada_in for FGN tests.")
        self.ada_calls.append(ada_in.detach().cpu().clone())
        latent_bias = ada_in.mean(dim=1, keepdim=True).unsqueeze(1)
        base = x[..., : self.out_channels]
        return base + latent_bias


def _make_geometry(n_cells: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.linspace(0.0, 1.0, n_cells),
            torch.linspace(0.0, 1.0, n_cells),
        ],
        dim=1,
    )


def _make_query_points() -> torch.Tensor:
    xx, yy = torch.meshgrid(
        torch.linspace(0.0, 1.0, 4),
        torch.linspace(0.0, 1.0, 4),
        indexing="xy",
    )
    return torch.stack([xx, yy], dim=-1)


def test_persistent_latent_bank_reuses_member_latent():
    torch.manual_seed(0)
    bank = sample_fgn_rollout_latent_bank(
        num_members=2,
        batch_size=1,
        latent_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        temporal_mode="persistent",
    )
    z0_t0 = get_fgn_rollout_latent(
        bank, member_idx=0, batch_size=1, latent_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    z0_t1 = get_fgn_rollout_latent(
        bank, member_idx=0, batch_size=1, latent_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    z1 = get_fgn_rollout_latent(
        bank, member_idx=1, batch_size=1, latent_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    assert torch.equal(z0_t0, z0_t1)
    assert not torch.equal(z0_t0, z1)


def test_stepwise_latent_sampling_changes_across_calls():
    torch.manual_seed(0)
    z0_t0 = get_fgn_rollout_latent(
        None, member_idx=0, batch_size=1, latent_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    z0_t1 = get_fgn_rollout_latent(
        None, member_idx=0, batch_size=1, latent_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    assert not torch.equal(z0_t0, z0_t1)


def test_update_fgn_dynamic_members_member_feedback_diverges():
    n_history = 2
    dynamic_members = [
        torch.zeros(1, n_history, 3, 1),
        torch.zeros(1, n_history, 3, 1),
    ]
    pred_samples = torch.tensor(
        [
            [[[1.0], [1.0], [1.0]]],
            [[[2.0], [2.0], [2.0]]],
        ]
    )
    pred_mean = pred_samples.mean(dim=0)
    updated = update_fgn_dynamic_members(
        dynamic_members=dynamic_members,
        pred_samples=pred_samples,
        pred_mean=pred_mean,
        n_history=n_history,
        state_update_mode="member_feedback",
    )
    assert not torch.equal(updated[0], updated[1])
    assert torch.allclose(updated[0][:, -1], pred_samples[0])
    assert torch.allclose(updated[1][:, -1], pred_samples[1])


def test_update_fgn_dynamic_members_mean_feedback_matches_legacy():
    n_history = 2
    dynamic_members = [
        torch.zeros(1, n_history, 3, 1),
        torch.zeros(1, n_history, 3, 1),
    ]
    pred_samples = torch.tensor(
        [
            [[[1.0], [1.0], [1.0]]],
            [[[2.0], [2.0], [2.0]]],
        ]
    )
    pred_mean = pred_samples.mean(dim=0)
    updated = update_fgn_dynamic_members(
        dynamic_members=dynamic_members,
        pred_samples=pred_samples,
        pred_mean=pred_mean,
        n_history=n_history,
        state_update_mode="mean_feedback",
    )
    assert torch.equal(updated[0], updated[1])
    assert torch.allclose(updated[0][:, -1], pred_mean)


def test_eval_generic_fgn_persistent_rollout_smoke(tmp_path: Path):
    torch.manual_seed(0)
    n_cells = 12
    n_steps = 5
    model = DummyFGNModel().eval()
    norm = IdentityNormalizer()
    geometry = _make_geometry(n_cells)
    sample = {
        "run_id": "smoke_run",
        "geometry": geometry,
        "static": torch.randn(n_cells, 2),
        "boundary": torch.randn(n_steps, n_cells, 1),
        "dynamic": torch.randn(n_steps, n_cells, 1),
        "query_points": _make_query_points(),
    }
    out_dir = str(tmp_path / "rollout_generic")
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
        logger=logging.getLogger("test_eval_fgn_generic"),
        fgn_noise_dim=8,
        n_ensemble_samples=2,
        fgn_latent_temporal_mode="persistent",
        gaussian_mode=False,
    )
    assert (tmp_path / "rollout_generic" / "rollout_metrics_data.npz").exists()
    assert len(model.ada_calls) == 4
    assert torch.equal(model.ada_calls[0], model.ada_calls[2])
    assert torch.equal(model.ada_calls[1], model.ada_calls[3])


def test_eval_hydrograph_fgn_single_member_rollout_smoke(tmp_path: Path):
    torch.manual_seed(0)
    n_cells = 12
    n_steps = 5
    model = DummyFGNModel().eval()
    norm = IdentityNormalizer()
    geometry = _make_geometry(n_cells)
    hydro_sample = {
        "hydrograph_id": "H0",
        "geometry": geometry,
        "static": torch.randn(n_cells, 2),
        "boundary": torch.randn(n_steps, n_cells, 1),
        "dynamic_ref": torch.randn(2, n_steps, n_cells, 1),
        "query_points": _make_query_points(),
        "n_ref_sims": 2,
    }
    out_dir = str(tmp_path / "rollout_hydro_single")
    _rollout_prediction_per_hydrograph(
        models=[model],
        hydrograph_samples=[hydro_sample],
        rollout_length=2,
        history_steps=2,
        dynamic_norm=norm,
        target_norm=norm,
        device=torch.device("cpu"),
        skip_before_timestep=0,
        dt=60.0,
        out_dir=out_dir,
        target_variables=["wd"],
        logger=logging.getLogger("test_eval_fgn_hydro_single"),
        fgn_noise_dim=8,
        n_ensemble_samples=1,
        fgn_latent_temporal_mode="persistent",
        gaussian_mode=False,
    )
    assert (tmp_path / "rollout_hydro_single" / "rollout_metrics_per_hydrograph.npz").exists()
    assert len(model.ada_calls) == 2


def test_eval_hydrograph_fgn_persistent_rollout_smoke(tmp_path: Path):
    torch.manual_seed(0)
    n_cells = 12
    n_steps = 5
    model = DummyFGNModel().eval()
    norm = IdentityNormalizer()
    geometry = _make_geometry(n_cells)
    hydro_sample = {
        "hydrograph_id": "H0",
        "geometry": geometry,
        "static": torch.randn(n_cells, 2),
        "boundary": torch.randn(n_steps, n_cells, 1),
        "dynamic_ref": torch.randn(2, n_steps, n_cells, 1),
        "query_points": _make_query_points(),
        "n_ref_sims": 2,
    }
    out_dir = str(tmp_path / "rollout_hydro")
    _rollout_prediction_per_hydrograph(
        models=[model],
        hydrograph_samples=[hydro_sample],
        rollout_length=2,
        history_steps=2,
        dynamic_norm=norm,
        target_norm=norm,
        device=torch.device("cpu"),
        skip_before_timestep=0,
        dt=60.0,
        out_dir=out_dir,
        target_variables=["wd"],
        logger=logging.getLogger("test_eval_fgn_hydro"),
        fgn_noise_dim=8,
        n_ensemble_samples=2,
        fgn_latent_temporal_mode="persistent",
        gaussian_mode=False,
    )
    assert (tmp_path / "rollout_hydro" / "rollout_metrics_per_hydrograph.npz").exists()
    assert len(model.ada_calls) == 4
    assert torch.equal(model.ada_calls[0], model.ada_calls[2])
    assert torch.equal(model.ada_calls[1], model.ada_calls[3])
