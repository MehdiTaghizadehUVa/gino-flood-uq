import warnings

import pytest

from neuralop.flood.train import debug


def test_run_debug_with_deterministic_fallback_retries_once(monkeypatch):
    calls = {"count": 0, "restore": 0, "set": []}

    monkeypatch.setattr(debug, "_deterministic_guard_enabled", lambda: True)
    monkeypatch.setattr(debug, "_deterministic_warn_only_enabled", lambda: False)
    monkeypatch.setattr(
        debug,
        "_set_deterministic_algorithms",
        lambda enabled, warn_only=False: calls["set"].append((enabled, warn_only)),
    )

    def restore_state():
        calls["restore"] += 1

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError(
                "upsample_bicubic2d_backward_out_cuda does not have a deterministic implementation, "
                "but you set torch.use_deterministic_algorithms(True)"
            )
        return "ok"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = debug._run_debug_with_deterministic_fallback(
            flaky,
            context="Verify",
            restore_state=restore_state,
        )

    assert result == "ok"
    assert calls["count"] == 2
    assert calls["restore"] == 1
    assert calls["set"] == [(False, False), (True, False)]
    assert any("Retrying this debug-only check once" in str(w.message) for w in caught)


def test_run_debug_with_deterministic_fallback_preserves_other_runtime_errors(monkeypatch):
    monkeypatch.setattr(debug, "_deterministic_guard_enabled", lambda: True)

    def boom():
        raise RuntimeError("some other runtime error")

    with pytest.raises(RuntimeError, match="some other runtime error"):
        debug._run_debug_with_deterministic_fallback(boom, context="Verify")
