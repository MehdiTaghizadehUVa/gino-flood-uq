import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier


def test_stage1_bundle_compat_manifests_are_isolated_across_concurrent_loads(tmp_path):
    from neuralop.flood.cli.train_neon_stage2 import _patched_stage1_bundle_manifest

    source = tmp_path / "coastal_fgn_bundle.json"
    original = {"dt_seconds": 123, "checkpoint": "relative/model.pt"}
    source.write_text(json.dumps(original), encoding="utf-8")
    barrier = Barrier(2)

    def read_compat_manifest():
        with _patched_stage1_bundle_manifest(source) as patched:
            barrier.wait(timeout=5)
            assert patched.parent == source.parent
            return patched, json.loads(patched.read_text(encoding="utf-8"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: read_compat_manifest(), range(2)))

    paths = [path for path, _ in results]
    assert paths[0] != paths[1]
    assert all(payload["dt_seconds"] == 900 for _, payload in results)
    assert all(payload["checkpoint"] == "relative/model.pt" for _, payload in results)
    assert all(not path.exists() for path in paths)
    assert json.loads(source.read_text(encoding="utf-8")) == original
