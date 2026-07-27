from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from neuralop.flood.train.operator_app import _resolve_alr_config
from neuralop.flood.eval.operator_app import _resolve_eval_alr_layout


def _config(*, enabled=True, training=True):
    model_alr = SimpleNamespace(
        enabled=enabled,
        num_particles=4,
    )
    training_alr = (
        SimpleNamespace(
            k_train=2,
            k_eval=15,
            adapter_warmup_epochs=5,
            joint_finetune_epochs=25,
            k_validation=5,
        )
        if training
        else None
    )
    return SimpleNamespace(
        model=SimpleNamespace(anchored_low_rank=model_alr),
        training=SimpleNamespace(anchored_low_rank=training_alr),
        gino=SimpleNamespace(),
        opt=SimpleNamespace(),
        inverse_test=True,
    )


def test_alr_config_resolves_stage1_schedule_and_nested_sample_counts():
    config = _config()

    enabled, model_cfg, training_cfg = _resolve_alr_config(config)

    assert enabled is True
    assert model_cfg == vars(config.model.anchored_low_rank)
    assert isinstance(model_cfg, dict)
    assert training_cfg is config.training.anchored_low_rank
    assert config.gino.anchored_low_rank is model_cfg
    assert config.gino.fgn_latent_temporal_mode == "persistent"
    assert config.opt.fgn_ar_state_update == "member_feedback"
    assert config.opt.crps_n_samples == 2
    assert config.opt.n_epochs == 30
    assert config.opt.alr_eval_n_samples == 20


def test_alr_resolved_architecture_metadata_is_torch_serializable():
    config = _config()
    _, model_cfg, _ = _resolve_alr_config(config)

    buffer = BytesIO()
    torch.save({"anchored_low_rank": model_cfg}, buffer)

    assert buffer.tell() > 0


def test_alr_config_requires_training_namespace():
    config = _config(training=False)

    with pytest.raises(ValueError, match="training.anchored_low_rank"):
        _resolve_alr_config(config)


def test_disabled_alr_config_does_not_mutate_stage1_settings():
    config = _config(enabled=False)

    enabled, _, training_cfg = _resolve_alr_config(config)

    assert enabled is False
    assert training_cfg is None
    assert vars(config.gino) == {}
    assert vars(config.opt) == {}


@pytest.mark.parametrize(
    ("warmup", "joint", "message"),
    [
        (-1, 25, "adapter_warmup_epochs"),
        (5, -1, "joint_finetune_epochs"),
        (0, 0, "at least one training epoch"),
    ],
)
def test_alr_config_rejects_invalid_epoch_schedule(warmup, joint, message):
    config = _config()
    config.training.anchored_low_rank.adapter_warmup_epochs = warmup
    config.training.anchored_low_rank.joint_finetune_epochs = joint

    with pytest.raises(ValueError, match=message):
        _resolve_alr_config(config)


def test_alr_config_requires_physical_validation_metrics():
    config = _config()
    config.inverse_test = False

    with pytest.raises(ValueError, match="inverse_test=true"):
        _resolve_alr_config(config)


def test_alr_config_rejects_negative_rmse_margin():
    config = _config()
    config.training.anchored_low_rank.rmse_noninferiority_margin = -0.1

    with pytest.raises(ValueError, match="rmse_noninferiority_margin"):
        _resolve_alr_config(config)


def test_eval_alr_layout_uses_checkpoint_particle_count_and_configured_k():
    config = _config()
    model = SimpleNamespace(
        anchored_low_rank_enabled=True,
        anchored_low_rank_num_particles=4,
    )

    layout = _resolve_eval_alr_layout(config, [model])

    assert layout.num_particles == 4
    assert layout.aleatory_samples == 15
    assert layout.n_members == 60


def test_eval_alr_layout_rejects_multiple_shared_backbones():
    config = _config()
    model = SimpleNamespace(
        anchored_low_rank_enabled=True,
        anchored_low_rank_num_particles=4,
    )

    with pytest.raises(ValueError, match="exactly one"):
        _resolve_eval_alr_layout(config, [model, model])


def test_alr_config_supports_adapter_only_schedule():
    config = _config()
    config.training.anchored_low_rank.adapter_warmup_epochs = 15
    config.training.anchored_low_rank.joint_finetune_epochs = 0

    enabled, _, _ = _resolve_alr_config(config)

    assert enabled is True
    assert config.opt.n_epochs == 15
