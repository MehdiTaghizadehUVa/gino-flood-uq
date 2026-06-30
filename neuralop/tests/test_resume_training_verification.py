from types import SimpleNamespace

from neuralop.flood.train.operator_app import should_run_training_verification


def test_training_verification_runs_for_fresh_single_process_launch():
    config = SimpleNamespace(verify_training=True)

    assert should_run_training_verification(
        config,
        checkpoint_resume_dir=None,
        use_distributed=False,
        global_rank=0,
    )


def test_training_verification_is_skipped_for_resume_launch():
    config = SimpleNamespace(verify_training=True)

    assert not should_run_training_verification(
        config,
        checkpoint_resume_dir="/tmp/checkpoints/train",
        use_distributed=False,
        global_rank=0,
    )


def test_training_verification_is_skipped_on_nonzero_distributed_rank():
    config = SimpleNamespace(verify_training=True)

    assert not should_run_training_verification(
        config,
        checkpoint_resume_dir=None,
        use_distributed=True,
        global_rank=1,
    )


def test_training_verification_defaults_to_disabled_when_config_absent():
    config = SimpleNamespace()

    assert not should_run_training_verification(
        config,
        checkpoint_resume_dir=None,
        use_distributed=False,
        global_rank=0,
    )
