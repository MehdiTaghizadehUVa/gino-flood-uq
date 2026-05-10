import importlib


def test_serving_worker_module_imports_without_optional_web_deps():
    mod = importlib.import_module("neuralop.flood.serving.worker")
    assert hasattr(mod, "main")


def test_serving_cli_module_imports_without_optional_web_deps():
    mod = importlib.import_module("neuralop.flood.serving.cli")
    assert hasattr(mod, "validate_bundle_main")
