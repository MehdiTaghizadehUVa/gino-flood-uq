from pathlib import Path

from neuralop.flood.eval.diffusion import _expand_checkpoint_candidates as expand_maintained
from neuralop.flood.eval.diffusion_legacy import _expand_checkpoint_candidates as expand_legacy


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    return path


def test_diffusion_checkpoint_discovery_prefers_latest_checkpoint_when_requested(tmp_path: Path):
    latest = _touch(tmp_path / "checkpoint.pt")
    _touch(tmp_path / "checkpoint_best.pt")

    assert expand_legacy(tmp_path, preferred_alias="model") == [latest]
    assert expand_maintained(tmp_path, preferred_alias="model") == [latest]


def test_diffusion_checkpoint_discovery_prefers_best_checkpoint_when_requested(tmp_path: Path):
    _touch(tmp_path / "checkpoint.pt")
    best = _touch(tmp_path / "checkpoint_best.pt")

    assert expand_legacy(tmp_path, preferred_alias="best_model") == [best]
    assert expand_maintained(tmp_path, preferred_alias="best_model") == [best]


def test_diffusion_checkpoint_discovery_falls_back_when_preferred_missing(tmp_path: Path):
    best = _touch(tmp_path / "checkpoint_best.pt")

    assert expand_legacy(tmp_path, preferred_alias="model") == [best]
    assert expand_maintained(tmp_path, preferred_alias="model") == [best]


def test_diffusion_checkpoint_discovery_applies_preference_to_child_runs(tmp_path: Path):
    latest_a = _touch(tmp_path / "run_a" / "checkpoint.pt")
    latest_b = _touch(tmp_path / "run_b" / "checkpoint.pt")
    _touch(tmp_path / "run_a" / "checkpoint_best.pt")
    _touch(tmp_path / "run_b" / "checkpoint_best.pt")

    assert expand_legacy(tmp_path, preferred_alias="model") == [latest_a, latest_b]
    assert expand_maintained(tmp_path, preferred_alias="model") == [latest_a, latest_b]


class _Args:
    def __init__(self, checkpoint_root: str):
        self.checkpoint_paths = None
        self.checkpoint_root = checkpoint_root


class _CheckpointConfig:
    def __init__(self, eval_name: str):
        self.eval_name = eval_name


class _Config:
    def __init__(self, eval_name: str):
        self.checkpoint = _CheckpointConfig(eval_name)


def test_diffusion_discover_checkpoints_honors_eval_name(tmp_path: Path):
    latest = _touch(tmp_path / "checkpoint.pt")
    best = _touch(tmp_path / "checkpoint_best.pt")
    args = _Args(str(tmp_path))

    from neuralop.flood.eval.diffusion import _discover_checkpoints as discover_maintained
    from neuralop.flood.eval.diffusion_legacy import _discover_checkpoints as discover_legacy

    assert discover_legacy(args, _Config("model")) == [latest]
    assert discover_maintained(args, _Config("model")) == [latest]
    assert discover_legacy(args, _Config("best_model")) == [best]
    assert discover_maintained(args, _Config("best_model")) == [best]
