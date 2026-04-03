from pathlib import Path

import yaml
from configmypy import ConfigPipeline, YamlConfig

from neuralop.flood.utils.runtime_core import _prepare_config_path_for_pipeline


def test_prepare_config_path_wraps_flat_payload(tmp_path: Path):
    config_path = tmp_path / 'flat.yaml'
    config_path.write_text(yaml.safe_dump({'arch': 'gino', 'opt': {'n_epochs': 3}}, sort_keys=False))

    prepared_path, transient_path = _prepare_config_path_for_pipeline(config_path, config_name='flood')
    try:
        assert transient_path is not None
        payload = yaml.safe_load(prepared_path.read_text())
        assert payload == {'flood': {'arch': 'gino', 'opt': {'n_epochs': 3}}}

        config = ConfigPipeline([YamlConfig(str(prepared_path), config_name='flood', config_folder=str(tmp_path))]).read_conf()
        assert config.arch == 'gino'
        assert config.opt.n_epochs == 3
    finally:
        if transient_path is not None:
            transient_path.unlink(missing_ok=True)


def test_prepare_config_path_preserves_wrapped_payload(tmp_path: Path):
    config_path = tmp_path / 'wrapped.yaml'
    config_path.write_text(yaml.safe_dump({'flood': {'arch': 'gino'}}, sort_keys=False))

    prepared_path, transient_path = _prepare_config_path_for_pipeline(config_path, config_name='flood')

    assert prepared_path == config_path
    assert transient_path is None
