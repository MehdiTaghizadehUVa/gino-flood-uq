from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.initial_conditions import InitialConditionResult
from neuralop.flood.serving.inference import DomainAssets, ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.run_spec import RunSpec


class IdentityNormalizer:
    def __init__(self, channels: int):
        self.mean = torch.zeros((1, 1, channels), dtype=torch.float32)
        self.std = torch.ones((1, 1, channels), dtype=torch.float32)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class TinyFGN(torch.nn.Module):
    output_distribution = "deterministic"

    def __init__(self, model_offset: float):
        super().__init__()
        self.model_offset = float(model_offset)

    def forward(self, *, input_geom, latent_queries, output_queries, x, ada_in):
        n_cells = x.shape[1]
        latest_stage = x[..., 11:12]
        latest_precip = x[..., 12:13]
        latest_dyn = x[..., -1:]
        latent = ada_in[:, :1].reshape(-1, 1, 1).expand(-1, n_cells, 1)
        # Positive, forcing-dependent WD with a small stochastic FGN perturbation.
        return torch.relu(0.05 * latest_stage + 0.001 * latest_precip + 0.02 * latest_dyn + 0.01 * latent + self.model_offset)


class ConstantInitialConditionProvider:
    def __init__(self, value: float):
        self.value = float(value)

    def resolve(self, forcing_input, *, bundle, n_cells: int):
        return InitialConditionResult(
            wd_history_m=np.full((bundle.n_history, n_cells, 1), self.value, dtype=np.float32),
            selection={
                "mode": "test_constant",
                "selected_reference_ids": ["fixture"],
                "weights": [1.0],
                "distances": [0.0],
            },
        )


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _bundle(tmp_path: Path):
    checkpoint_dirs = []
    for idx in range(3):
        d = tmp_path / f"ckpt_{idx}"
        _touch(d / "best_model_state_dict.pt")
        checkpoint_dirs.append(str(d))
    normalizer_path = tmp_path / "normalizers.pt"
    _touch(normalizer_path)
    coeff = tmp_path / "crps_mbm.json"
    coeff.write_text(json.dumps({"method": "identity"}), encoding="utf-8")
    iso = tmp_path / "isotonic.json"
    iso.write_text(json.dumps({"method": "identity"}), encoding="utf-8")
    static_files = []
    for idx in range(5):
        f = tmp_path / f"static_{idx}.txt"
        _touch(f)
        static_files.append(str(f))
    manifest = tmp_path / "bundle.json"
    manifest.write_text(
        json.dumps(
            {
                "bundle_id": "coastal-fgn-test",
                "domain_name": "coastal",
                "git_commit": "test",
                "checkpoint_dirs": checkpoint_dirs,
                "checkpoint_alias": "best_model",
                "normalizer_path": str(normalizer_path),
                "static_files": static_files,
                "calibration_coefficients_path": str(coeff),
                "isotonic_curves_path": str(iso),
                "boundary_channels": ["stage", "precipitation"],
                "dt_seconds": 900,
                "n_history": 3,
                "skip_before_timestep": 12,
                "max_forecast_steps": 6,
                "fgn_noise_dim": 32,
                "members_per_checkpoint": 20,
                "query_res": [4, 4],
            }
        ),
        encoding="utf-8",
    )
    return load_model_bundle(manifest)


def _forcing_csv(n_rows: int = 24) -> str:
    lines = ["time_seconds,stage,precipitation"]
    for i in range(n_rows):
        lines.append(f"{i * 900},{1.0 + 0.1 * i},{2.0 * i}")
    return "\n".join(lines)


def test_production_fgn_service_runs_reproducible_60_member_rollout(tmp_path):
    bundle = _bundle(tmp_path)
    geometry = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, 3), torch.linspace(0, 1, 2), indexing="ij"),
        dim=-1,
    ).reshape(-1, 2)
    static = torch.ones((geometry.shape[0], 7), dtype=torch.float32)
    dry_mask = torch.zeros((geometry.shape[0],), dtype=torch.bool)
    dry_mask[0] = True
    assets = DomainAssets(geometry=geometry, static=static, structural_dry_mask=dry_mask, query_res=(4, 4))
    normalizers = {
        "geometry": IdentityNormalizer(2),
        "static": IdentityNormalizer(7),
        "boundary": IdentityNormalizer(2),
        "dynamic": IdentityNormalizer(1),
        "target": IdentityNormalizer(1),
    }
    models = [TinyFGN(0.01), TinyFGN(0.02), TinyFGN(0.03)]
    service = ProductionFGNInferenceService(
        bundle,
        device="cpu",
        preloaded_models=models,
        preloaded_normalizers=normalizers,
        preloaded_domain_assets=assets,
    )
    forcing = parse_forcing_csv(_forcing_csv(), bundle=bundle, requested_forecast_steps=4)
    spec = RunSpec.new(
        user_id="user@example.com",
        bundle_id=bundle.bundle_id,
        input_hash=forcing.input_hash,
        forecast_steps=forcing.forecast_steps,
        seed=99,
    )

    first = service.run(spec, forcing)
    second = service.run(spec, forcing)

    assert first.members_wd.shape == (60, 4, geometry.shape[0])
    assert np.allclose(first.members_wd, second.members_wd)
    assert first.metadata["adapter"] == "production_fgn"
    assert first.metadata["member_model_id"].count("model_0") == 20
    assert first.metadata["member_model_id"].count("model_1") == 20
    assert first.metadata["member_model_id"].count("model_2") == 20
    assert np.all(first.members_wd[:, :, 0] == 0.0)
    assert np.any(np.std(first.members_wd[:, :, 1:], axis=0) > 0.0)
    assert first.wettable_mask.tolist() == [False, True, True, True, True, True]


def test_production_fgn_service_rejects_wrong_model_count(tmp_path):
    bundle = _bundle(tmp_path)
    assets = DomainAssets(
        geometry=torch.zeros((2, 2), dtype=torch.float32),
        static=torch.zeros((2, 7), dtype=torch.float32),
        query_res=(2, 2),
    )
    normalizers = {
        "geometry": IdentityNormalizer(2),
        "static": IdentityNormalizer(7),
        "boundary": IdentityNormalizer(2),
        "dynamic": IdentityNormalizer(1),
        "target": IdentityNormalizer(1),
    }
    service = ProductionFGNInferenceService(
        bundle,
        device="cpu",
        preloaded_models=[TinyFGN(0.0)],
        preloaded_normalizers=normalizers,
        preloaded_domain_assets=assets,
    )
    forcing = parse_forcing_csv(_forcing_csv(), bundle=bundle, requested_forecast_steps=2)
    spec = RunSpec.new(user_id="u", bundle_id=bundle.bundle_id, input_hash=forcing.input_hash, forecast_steps=2)
    with pytest.raises(ValueError, match="expected 3 models"):
        service.run(spec, forcing)


def test_production_fgn_service_uses_injected_initial_history(tmp_path):
    bundle = _bundle(tmp_path)
    geometry = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, 3), torch.linspace(0, 1, 2), indexing="ij"),
        dim=-1,
    ).reshape(-1, 2)
    static = torch.ones((geometry.shape[0], 7), dtype=torch.float32)
    assets = DomainAssets(geometry=geometry, static=static, query_res=(4, 4))
    normalizers = {
        "geometry": IdentityNormalizer(2),
        "static": IdentityNormalizer(7),
        "boundary": IdentityNormalizer(2),
        "dynamic": IdentityNormalizer(1),
        "target": IdentityNormalizer(1),
    }
    models = [TinyFGN(0.0), TinyFGN(0.0), TinyFGN(0.0)]
    forcing = parse_forcing_csv(_forcing_csv(), bundle=bundle, requested_forecast_steps=2)
    spec = RunSpec.new(user_id="u", bundle_id=bundle.bundle_id, input_hash=forcing.input_hash, forecast_steps=2)
    dry = ProductionFGNInferenceService(
        bundle,
        device="cpu",
        preloaded_models=models,
        preloaded_normalizers=normalizers,
        preloaded_domain_assets=assets,
    ).run(spec, forcing)
    conditioned = ProductionFGNInferenceService(
        bundle,
        device="cpu",
        preloaded_models=models,
        preloaded_normalizers=normalizers,
        preloaded_domain_assets=assets,
        initial_condition_provider=ConstantInitialConditionProvider(0.25),
    ).run(spec, forcing)

    assert conditioned.metadata["initial_condition"]["mode"] == "test_constant"
    assert not np.allclose(conditioned.members_wd[:, 0, :], dry.members_wd[:, 0, :])
