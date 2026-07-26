import logging

import torch

from neuralop.flood.eval.alr_fgn import (
    ALRMemberLayout,
    alr_nested_variance_components,
    alr_crossed_variance_components,
    forward_alr_rollout_step,
)
from neuralop.flood.eval.checkpoints import _attach_alr_bootstrap_from_state
from neuralop.flood.eval.rollout import _rollout_prediction_per_hydrograph
from neuralop.flood.eval.scientific_calibration import load_forecast_artifact


class _RecordingALR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.particle_ids = None
        self.latents = None

    def forward(self, *, x, ada_in, particle_ids, **kwargs):
        del kwargs
        self.calls += 1
        self.particle_ids = particle_ids.detach().clone()
        self.latents = ada_in.detach().clone()
        return x[..., :1] + particle_ids[:, None, None] + ada_in[:, None, :1]


def test_alr_member_layout_is_particle_major_and_reconstructs_nested_shape():
    layout = ALRMemberLayout(num_particles=3, aleatory_samples=2)

    assert layout.member_epistemic_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert layout.member_aleatory_id.tolist() == [0, 1, 0, 1, 0, 1]
    assert layout.n_members == 6

    flat = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    nested = layout.unflatten_members(flat)
    assert nested.shape == (3, 2, 4)
    torch.testing.assert_close(nested[2, 1], flat[5])


def test_vectorized_alr_rollout_uses_one_call_and_common_latent_bank():
    model = _RecordingALR()
    # [M=2, K=3, H=2, N=2, C=1]
    histories = torch.zeros(2, 3, 2, 2, 1)
    for m in range(2):
        for k in range(3):
            histories[m, k, -1, :, 0] = 10 * m + k
    static = torch.zeros(1, 2, 1)
    boundary = torch.zeros(2, 2, 1)
    latent_bank = torch.tensor([[[0.1]], [[0.2]], [[0.3]]])

    prediction = forward_alr_rollout_step(
        model,
        histories=histories,
        static=static,
        boundary=boundary,
        input_geom=torch.zeros(1, 2, 2),
        latent_queries=torch.zeros(1, 2, 2, 2),
        output_queries=torch.zeros(1, 2, 2),
        latent_bank=latent_bank,
    )

    assert model.calls == 1
    assert prediction.shape == (2, 3, 1, 2, 1)
    assert model.particle_ids.tolist() == [0, 0, 0, 1, 1, 1]
    latents = model.latents.reshape(2, 3, 1)
    torch.testing.assert_close(latents[0], latents[1])
    assert not torch.equal(prediction[0, 0], prediction[1, 0])


def test_alr_nested_variance_separates_shared_aleatory_and_particle_effects():
    particle = torch.tensor([0.0, 10.0]).view(2, 1, 1, 1)
    aleatory = torch.tensor([0.0, 2.0, 4.0]).view(1, 3, 1, 1)
    prediction = particle + aleatory

    components = alr_nested_variance_components(prediction)

    # Sample variance of [0, 2, 4] and [2, 12] respectively.
    torch.testing.assert_close(components["variance_aleatory"], torch.tensor([[4.0]]))
    torch.testing.assert_close(components["variance_epistemic"], torch.tensor([[50.0]]))
    torch.testing.assert_close(components["variance_total"], torch.tensor([[54.0]]))


def test_alr_crossed_variance_removes_finite_k_interaction_contamination():
    particle = torch.tensor([0.0, 10.0]).view(2, 1, 1, 1)
    aleatory = torch.tensor([0.0, 2.0]).view(1, 2, 1, 1)
    interaction = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).view(2, 2, 1, 1)
    prediction = particle + aleatory + interaction

    components = alr_crossed_variance_components(prediction)

    torch.testing.assert_close(
        components["variance_epistemic_uncorrected"], torch.tensor([[50.0]])
    )
    torch.testing.assert_close(
        components["variance_epistemic"], torch.tensor([[48.0]])
    )


def test_alr_checkpoint_loader_reconstructs_persistent_bootstrap_before_load():
    class _CheckpointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchored_low_rank_enabled = True
            self.anchored_low_rank_num_particles = 2

    model = _CheckpointModel()
    state = {
        "alr_family_bootstrap.weights": torch.tensor(
            [[0.5, 1.5], [1.25, 0.75]], dtype=torch.float32
        ),
        "alr_family_bootstrap._extra_state": {
            "family_ids": ["F001", "F002"],
            "num_particles": 2,
            "seed": 41,
        },
    }

    attached = _attach_alr_bootstrap_from_state(model, state)

    assert attached is True
    assert model.alr_family_bootstrap.family_ids == ["F001", "F002"]
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        model.alr_family_bootstrap.weights,
        state["alr_family_bootstrap.weights"],
    )


class _IdentityNormalizer:
    def to(self, device):
        del device
        return self

    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


def test_hydrograph_alr_rollout_writes_reconstructable_nested_member_metadata(tmp_path):
    torch.manual_seed(0)
    model = _RecordingALR().eval()
    model.anchored_low_rank_enabled = True
    model.anchored_low_rank_num_particles = 2
    n_cells = 5
    n_steps = 5
    geometry = torch.stack(
        [torch.linspace(0, 1, n_cells), torch.linspace(1, 0, n_cells)], dim=-1
    )
    artifact_dir = tmp_path / "artifacts"
    sample = {
        "hydrograph_id": "ALR_H0",
        "geometry": geometry,
        "geometry_raw": geometry,
        "static": torch.zeros(n_cells, 1),
        "boundary": torch.zeros(n_steps, n_cells, 1),
        "dynamic_ref": torch.full((3, n_steps, n_cells, 1), 0.2),
        "query_points": torch.zeros(2, 2, 2),
        "n_ref_sims": 3,
        "structural_dry_mask": torch.tensor([False, False, False, False, True]),
    }

    _rollout_prediction_per_hydrograph(
        models=[model],
        hydrograph_samples=[sample],
        rollout_length=2,
        history_steps=2,
        dynamic_norm=_IdentityNormalizer(),
        target_norm=_IdentityNormalizer(),
        device=torch.device("cpu"),
        skip_before_timestep=0,
        dt=900.0,
        out_dir=str(tmp_path / "rollout"),
        target_variables=["wd"],
        logger=logging.getLogger("test_alr_hydrograph_rollout"),
        fgn_noise_dim=1,
        n_ensemble_samples=4,
        fgn_latent_temporal_mode="persistent",
        fgn_ar_state_update="member_feedback",
        forecast_artifact_dir=str(artifact_dir),
        write_visualizations=False,
        alr_num_particles=2,
        alr_aleatory_samples=2,
    )

    assert model.calls == 2
    artifact = load_forecast_artifact(artifact_dir / "ALR_H0.calibration_artifact.h5")
    assert artifact["pred_members_wd"].shape == (4, 2, n_cells)
    assert float(artifact["pred_members_wd"].min()) >= 0.0
    assert artifact["member_epistemic_id"] == [0, 0, 1, 1]
    assert artifact["member_aleatory_id"] == [0, 1, 0, 1]
    assert artifact["metadata"]["alr_num_particles"] == 2
