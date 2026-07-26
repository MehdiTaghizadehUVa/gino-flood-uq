from types import SimpleNamespace

import pytest

from neuralop.flood.train.operator_app import _resolve_alr_config


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
        )
        if training
        else None
    )
    return SimpleNamespace(
        model=SimpleNamespace(anchored_low_rank=model_alr),
        training=SimpleNamespace(anchored_low_rank=training_alr),
        gino=SimpleNamespace(),
        opt=SimpleNamespace(),
    )


def test_alr_config_resolves_stage1_schedule_and_nested_sample_counts():
    config = _config()

    enabled, model_cfg, training_cfg = _resolve_alr_config(config)

    assert enabled is True
    assert model_cfg is config.model.anchored_low_rank
    assert training_cfg is config.training.anchored_low_rank
    assert config.gino.anchored_low_rank is model_cfg
    assert config.gino.fgn_latent_temporal_mode == "persistent"
    assert config.opt.fgn_ar_state_update == "member_feedback"
    assert config.opt.crps_n_samples == 2
    assert config.opt.n_epochs == 30
    assert config.opt.alr_eval_n_samples == 60


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
    [(-1, 25, "adapter_warmup_epochs"), (5, 0, "joint_finetune_epochs")],
)
def test_alr_config_rejects_invalid_epoch_schedule(warmup, joint, message):
    config = _config()
    config.training.anchored_low_rank.adapter_warmup_epochs = warmup
    config.training.anchored_low_rank.joint_finetune_epochs = joint

    with pytest.raises(ValueError, match=message):
        _resolve_alr_config(config)
