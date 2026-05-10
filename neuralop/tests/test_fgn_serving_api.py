"""HTTP-level integration tests for the FastAPI serving adapter.

These tests exercise the public API surface end-to-end through an in-process
TestClient with fake adapters. They verify auth, ownership, validation, and
admin-only access at the HTTP boundary — not by calling Python methods directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from neuralop.flood.serving.access import AccessPolicy, User
from neuralop.flood.serving.api import create_app
from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.inference import FakeFGNInferenceService
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.products import ForecastProductBuilder
from neuralop.flood.serving.queue import InMemoryJobQueue
from neuralop.flood.serving.repository import InMemoryRunRepository
from neuralop.flood.serving.storage import LocalArtifactStore

from neuralop.tests.test_fgn_serving_contracts import _bundle, _valid_csv


class _UserProvider:
    """Pluggable auth callable; each test sets who the next request comes from."""

    def __init__(self) -> None:
        self.current: User | None = None

    def __call__(self, _request=None):  # FastAPI passes the Request; TestClient invokes lazily
        if self.current is None:
            raise PermissionError("not authenticated")
        return self.current


@pytest.fixture
def env(tmp_path):
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    repository = InMemoryRunRepository()
    queue = InMemoryJobQueue()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    policy = AccessPolicy(
        allowed_emails=["alice@example.com", "bob@example.com"],
        admin_emails=["admin@example.com"],
    )
    inference = FakeFGNInferenceService(bundle, n_cells=8)
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=repository,
        queue=queue,
        artifact_store=artifacts,
        access_policy=policy,
        inference_service=inference,
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    provider = _UserProvider()
    app = create_app(orchestrator, current_user=provider)
    client = TestClient(app)
    return {
        "client": client,
        "provider": provider,
        "orchestrator": orchestrator,
        "repository": repository,
        "policy": policy,
    }


def _alice(disclaimer: bool = True) -> User:
    return User(user_id="alice@example.com", email="alice@example.com", disclaimer_acknowledged=disclaimer)


def _bob() -> User:
    return User(user_id="bob@example.com", email="bob@example.com", disclaimer_acknowledged=True)


def _admin() -> User:
    return User(
        user_id="admin@example.com",
        email="admin@example.com",
        is_admin=True,
        disclaimer_acknowledged=True,
    )


def _outsider() -> User:
    return User(user_id="mallory@example.com", email="mallory@example.com", disclaimer_acknowledged=True)


def _post_csv(client, csv_text: str, *, label: str = "smoke") -> "httpx.Response":
    return client.post(
        "/api/runs",
        files={"file": ("forcing.csv", csv_text, "text/csv")},
        data={"label": label},
    )


def test_model_bundle_returns_public_metadata(env):
    response = env["client"].get("/api/model-bundle")
    assert response.status_code == 200
    body = response.json()
    assert body["bundle_id"] == "coastal-fgn-60-v1"
    assert body["domain_name"] == "coastal"
    # Must not leak filesystem paths.
    assert "checkpoint_dirs" not in body
    assert "normalizer_path" not in body
    assert body["input_contract"]["required_columns"] == ["time_seconds", "stage", "precipitation"]


def test_forcing_template_is_directly_valid(env):
    env["provider"].current = _alice()
    template = env["client"].get("/api/forcing-template")
    assert template.status_code == 200
    validation = env["client"].post(
        "/api/forcing/validate",
        files={"file": ("template.csv", template.text, "text/csv")},
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True


def test_user_can_persist_disclaimer_acknowledgement(env):
    env["provider"].current = _alice(disclaimer=False)
    response = env["client"].post("/api/me/disclaimer")
    assert response.status_code == 200
    assert response.json()["disclaimer_acknowledged"] is True
    env["provider"].current = _alice(disclaimer=False)
    me = env["client"].get("/api/me")
    assert me.status_code == 200
    assert me.json()["disclaimer_acknowledged"] is True


def test_create_run_unauthenticated_returns_401(env):
    env["provider"].current = None
    response = _post_csv(env["client"], _valid_csv())
    assert response.status_code == 401


def test_create_run_disclaimer_required_returns_403(env):
    env["provider"].current = _alice(disclaimer=False)
    response = _post_csv(env["client"], _valid_csv())
    assert response.status_code == 403
    assert "disclaimer" in response.text.lower()


def test_create_run_disallowed_email_returns_403(env):
    env["provider"].current = _outsider()
    response = _post_csv(env["client"], _valid_csv())
    assert response.status_code == 403


def test_create_run_invalid_csv_returns_400_and_does_not_queue(env):
    env["provider"].current = _alice()
    bad_csv = "stage,precipitation\n1.0\n"  # missing rows + bad timestep
    response = _post_csv(env["client"], bad_csv)
    assert response.status_code == 400
    # Nothing must be enqueued for an invalid upload.
    assert list(env["repository"].list_for_user("alice@example.com")) == []


def test_create_run_succeeds_for_allowlisted_user_and_returns_queued(env):
    env["provider"].current = _alice()
    response = _post_csv(env["client"], _valid_csv())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "QUEUED"
    run_id = body["run_id"]
    record = env["repository"].get(run_id)
    assert record.spec.user_id == "alice@example.com"


def test_get_run_owner_can_read(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    response = env["client"].get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["run_id"] == run_id
    assert response.json()["progress"] == 0.25
    assert response.json()["result_availability"]["forcing_csv"] is True


def test_get_run_other_user_cannot_read_returns_403(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    env["provider"].current = _bob()
    response = env["client"].get(f"/api/runs/{run_id}")
    assert response.status_code == 403


def test_admin_can_read_other_user_run(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    env["provider"].current = _admin()
    response = env["client"].get(f"/api/runs/{run_id}")
    assert response.status_code == 200


def test_list_runs_only_returns_callers_own_runs(env):
    env["provider"].current = _alice()
    _post_csv(env["client"], _valid_csv(), label="alice-run")
    env["provider"].current = _bob()
    _post_csv(env["client"], _valid_csv(), label="bob-run")
    env["provider"].current = _alice()
    response = env["client"].get("/api/runs")
    assert response.status_code == 200
    bodies = response.json()
    assert len(bodies) == 1
    assert bodies[0]["spec"]["user_id"] == "alice@example.com"


def test_get_artifact_owner_can_download_and_other_user_cannot(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    response = env["client"].get(f"/api/runs/{run_id}/artifacts/forcing.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    env["provider"].current = _bob()
    response_bob = env["client"].get(f"/api/runs/{run_id}/artifacts/forcing.csv")
    assert response_bob.status_code == 403


def test_owner_can_cancel_queued_run(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    canceled = env["client"].post(f"/api/runs/{run_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "CANCELED"


def test_admin_pin_unpin_and_cancel_endpoints_require_admin(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    # Non-admin attempts.
    pin_attempt = env["client"].post(f"/api/admin/runs/{run_id}/pin")
    assert pin_attempt.status_code == 403
    list_attempt = env["client"].get("/api/admin/runs")
    assert list_attempt.status_code == 403
    # Admin succeeds.
    env["provider"].current = _admin()
    pinned = env["client"].post(f"/api/admin/runs/{run_id}/pin")
    assert pinned.status_code == 200
    assert pinned.json()["pinned"] is True
    unpinned = env["client"].post(f"/api/admin/runs/{run_id}/unpin")
    assert unpinned.status_code == 200
    assert unpinned.json()["pinned"] is False
    canceled = env["client"].post(f"/api/admin/runs/{run_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "CANCELED"
    # Admin can list all runs.
    listing = env["client"].get("/api/admin/runs")
    assert listing.status_code == 200
    assert any(item["run_id"] == run_id for item in listing.json())


def test_admin_can_add_remove_and_list_allowlist_entries(env):
    env["provider"].current = _admin()
    # List initial allowlist (alice + bob + admin via env fixture).
    initial = env["client"].get("/api/admin/users")
    assert initial.status_code == 200
    initial_emails = {row["email"] for row in initial.json()}
    assert {"alice@example.com", "bob@example.com", "admin@example.com"}.issubset(initial_emails)

    # Add a new researcher.
    add_response = env["client"].post(
        "/api/admin/users",
        json={"email": "Charlie@Example.com", "is_admin": False},
    )
    assert add_response.status_code == 200
    assert add_response.json() == {
        "email": "charlie@example.com",
        "is_admin": False,
        "disclaimer_acknowledged": False,
    }

    # Charlie can now submit runs (after disclaimer + allowlist).
    env["provider"].current = User(
        user_id="charlie@example.com",
        email="charlie@example.com",
        disclaimer_acknowledged=True,
    )
    response = _post_csv(env["client"], _valid_csv())
    assert response.status_code == 200

    # Admin removes Charlie. Subsequent submissions are blocked.
    env["provider"].current = _admin()
    delete_response = env["client"].delete("/api/admin/users/charlie@example.com")
    assert delete_response.status_code == 200
    env["provider"].current = User(
        user_id="charlie@example.com",
        email="charlie@example.com",
        disclaimer_acknowledged=True,
    )
    blocked = _post_csv(env["client"], _valid_csv())
    assert blocked.status_code == 403


def test_admin_users_endpoints_reject_non_admin_callers(env):
    env["provider"].current = _alice()
    list_attempt = env["client"].get("/api/admin/users")
    assert list_attempt.status_code == 403
    add_attempt = env["client"].post(
        "/api/admin/users",
        json={"email": "evil@example.com"},
    )
    assert add_attempt.status_code == 403


def test_admin_add_user_rejects_invalid_email(env):
    env["provider"].current = _admin()
    response = env["client"].post(
        "/api/admin/users",
        json={"email": "not-an-email"},
    )
    assert response.status_code == 400


def test_admin_remove_user_returns_404_when_unknown(env):
    env["provider"].current = _admin()
    response = env["client"].delete("/api/admin/users/ghost@example.com")
    assert response.status_code == 404


def test_admin_can_promote_existing_user_to_admin(env):
    env["provider"].current = _admin()
    response = env["client"].post(
        "/api/admin/users",
        json={"email": "bob@example.com", "is_admin": True},
    )
    assert response.status_code == 200
    assert response.json() == {
        "email": "bob@example.com",
        "is_admin": True,
        "disclaimer_acknowledged": False,
    }
    # Bob can now hit admin-only endpoints.
    env["provider"].current = User(
        user_id="bob@example.com",
        email="bob@example.com",
        disclaimer_acknowledged=True,
    )
    listing = env["client"].get("/api/admin/users")
    assert listing.status_code == 200


def test_full_lifecycle_submit_execute_then_download_summaries(env):
    env["provider"].current = _alice()
    submit = _post_csv(env["client"], _valid_csv())
    run_id = submit.json()["run_id"]
    env["orchestrator"].execute(run_id)
    response = env["client"].get(f"/api/runs/{run_id}/artifacts")
    assert response.status_code == 200
    artifact_ids = {item["artifact_id"] for item in response.json()}
    assert {"raw_summary.json", "calibrated_summary.json", "comparison_summary.json"}.issubset(artifact_ids)
    summary = env["client"].get(f"/api/runs/{run_id}/artifacts/calibrated_summary.json")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["label"] == "calibrated"
    # Isotonic flag is False here because the test bundle has empty isotonic.
    assert payload["isotonic_calibration_applied"] is False
