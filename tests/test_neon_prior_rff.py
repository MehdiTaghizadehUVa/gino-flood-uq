import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    package = sys.modules.setdefault(name, types.ModuleType(name))
    package.__path__ = [str(REPO_ROOT.joinpath(*name.split(".")))]


def _load_module(name: str, rel_path: str):
    _ensure_pkg("neuralop")
    _ensure_pkg("neuralop.flood")
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")

NEONEpistemicCorrection = neon.NEONEpistemicCorrection
load_neon_stage2_checkpoint = neon.load_neon_stage2_checkpoint
prior_psi_floor_diagnostic = neon.prior_psi_floor_diagnostic
save_neon_stage2_checkpoint = neon.save_neon_stage2_checkpoint


def _inputs():
    torch.manual_seed(0)
    base = torch.zeros(1, 2, 3, 4, 1)
    features = torch.randn(1, 2, 3, 4, 5)
    z_e = torch.randn(3, 4)
    coords = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    return base, features, z_e, coords


def test_rff_prior_changes_with_node_coords_but_trainable_branch_does_not():
    base, features, z_e, coords = _inputs()
    torch.manual_seed(1)
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        train_hidden_channels=8,
        prior_hidden_channels=8,
        prior_rff_dim=8,
        alpha=0.2,
    )

    out_a = head(base, features, z_e, node_coords=coords)
    out_b = head(base, features, z_e, node_coords=coords.flip(dims=[1]))

    torch.testing.assert_close(out_a.trainable_correction, out_b.trainable_correction)
    assert not torch.allclose(out_a.prior_correction, out_b.prior_correction)


def test_rff_prior_requires_node_coords_and_legacy_mode_does_not():
    base, features, z_e, coords = _inputs()
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        prior_rff_dim=8,
    )
    with pytest.raises(ValueError, match="node_coords"):
        head(base, features, z_e)

    legacy = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        prior_rff_dim=0,
    )
    out = legacy(base, features, z_e)
    assert out.prediction.shape == (1, 3, 2, 3, 4, 1)


def test_film_branch_rejects_rff_prior_features():
    with pytest.raises(ValueError, match="prior_rff_dim"):
        NEONEpistemicCorrection(
            feature_channels=5,
            out_channels=1,
            epistemic_dim=4,
            branch_type="film",
            prior_rff_dim=8,
        )


def test_rff_checkpoint_round_trip_preserves_outputs_and_buffers():
    base, features, z_e, coords = _inputs()
    torch.manual_seed(2)
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        train_hidden_channels=8,
        prior_hidden_channels=8,
        prior_rff_dim=8,
        prior_rff_lengthscale=0.4,
        prior_rff_include_lead=True,
    )
    expected = head(base, features, z_e, node_coords=coords).prediction.detach()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "neon_rff.pt"
        save_neon_stage2_checkpoint(path, head)
        loaded, _ = load_neon_stage2_checkpoint(path)

    actual = loaded(base, features, z_e, node_coords=coords).prediction.detach()
    assert loaded.prior_rff_dim == 8
    assert loaded.prior_rff_lengthscale == pytest.approx(0.4)
    assert loaded.prior_rff_freqs.shape == (3, 4)
    torch.testing.assert_close(actual, expected)


def test_old_style_arch_dict_without_rff_keys_loads_as_legacy(tmp_path):
    torch.manual_seed(3)
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        prior_rff_dim=0,
    )
    path = tmp_path / "old.pt"
    save_neon_stage2_checkpoint(path, head)
    payload = torch.load(path)
    payload["architecture"].pop("prior_rff_dim")
    payload["architecture"].pop("prior_rff_lengthscale")
    payload["architecture"].pop("prior_rff_include_lead")
    torch.save(payload, path)

    loaded, _ = load_neon_stage2_checkpoint(path)
    assert loaded.prior_rff_dim == 0


def test_pre_rff_checkpoint_without_buffer_loads_as_legacy(tmp_path):
    torch.manual_seed(4)
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        prior_rff_dim=0,
    )
    path = tmp_path / "pre_rff.pt"
    save_neon_stage2_checkpoint(path, head)
    payload = torch.load(path)
    payload["architecture"].pop("prior_rff_dim")
    payload["architecture"].pop("prior_rff_lengthscale")
    payload["architecture"].pop("prior_rff_include_lead")
    payload["state_dict"].pop("prior_rff_freqs")
    torch.save(payload, path)

    loaded, _ = load_neon_stage2_checkpoint(path)
    assert loaded.prior_rff_dim == 0
    assert loaded.prior_rff_freqs.numel() == 0


def test_prior_floor_diagnostic_positive_on_constant_features():
    base, features, z_e, coords = _inputs()
    features = torch.zeros_like(features)
    head = NEONEpistemicCorrection(
        feature_channels=5,
        out_channels=1,
        epistemic_dim=4,
        prior_hidden_channels=8,
        prior_rff_dim=8,
        alpha=0.2,
    )
    diag = prior_psi_floor_diagnostic(
        module=head,
        features=features,
        z_e=z_e,
        node_coords=coords,
    )
    assert diag["prior_floor_var"] > 0.0
