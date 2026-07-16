from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deployment" / "fgn-serving"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_production_compose_uses_published_images_not_builds() -> None:
    compose = _read(DEPLOY_DIR / "docker-compose.yml")

    assert "FGN_API_IMAGE" in compose
    assert "FGN_WORKER_IMAGE" in compose
    assert "FGN_CLEANUP_IMAGE" in compose
    assert "FGN_FRONTEND_IMAGE" in compose
    assert "FGN_PROXY_IMAGE" in compose
    assert "dockerfile: deployment/fgn-serving/python.Dockerfile" not in compose
    assert "context: ../../apps/fgn-serving-frontend" not in compose


def test_local_build_override_keeps_manual_debug_builds_available() -> None:
    override = _read(DEPLOY_DIR / "docker-compose.local-build.yml")

    assert "fgn-serving-python:local" in override
    assert "fgn-serving-frontend:local" in override
    assert "dockerfile: deployment/fgn-serving/python.Dockerfile" in override
    assert "context: ../../apps/fgn-serving-frontend" in override


def test_lab_deploy_workflow_targets_self_hosted_gpu_runner() -> None:
    workflow = _read(WORKFLOW_DIR / "fgn-serving-deploy-lab.yml")

    assert "runs-on: [self-hosted, fgn-lab, linux, gpu]" in workflow
    assert "group: fgn-lab-production" in workflow
    assert "deployment/fgn-serving/scripts/deploy_lab.sh" in workflow
    assert "deployment/fgn-serving/scripts/smoke_lab.sh" in workflow


def test_image_workflow_publishes_sha_tagged_ghcr_images() -> None:
    workflow = _read(WORKFLOW_DIR / "fgn-serving-images.yml")

    assert "ghcr.io/${GITHUB_REPOSITORY,,}" in workflow
    assert "${namespace}/fgn-serving-python:${sha}" in workflow
    assert "${namespace}/fgn-serving-frontend:${sha}" in workflow
    assert "${namespace}/fgn-serving-proxy:${sha}" in workflow
    assert "${namespace}/fgn-serving-python:main" in workflow
    assert "${namespace}/fgn-serving-frontend:main" in workflow
    assert "docker/build-push-action@v6" in workflow


def test_deployment_scripts_are_shell_syntax_valid() -> None:
    scripts = sorted((DEPLOY_DIR / "scripts").glob("*.sh"))
    assert scripts

    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_deploy_dry_run_does_not_write_deployment_records(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "ENV_FILE": str(tmp_path / "missing.env"),
            "FGN_SITE_HOSTNAME": "fgn-lab.example.edu",
            "FGN_DATA_ROOT": str(tmp_path / "data"),
            "FGN_POSTGRES_DATA_ROOT": str(tmp_path / "postgres"),
            "POSTGRES_PASSWORD": "test-postgres-password",
            "FGN_API_IMAGE": "ghcr.io/example/fgn-serving-python:test",
            "FGN_WORKER_IMAGE": "ghcr.io/example/fgn-serving-python:test",
            "FGN_CLEANUP_IMAGE": "ghcr.io/example/fgn-serving-python:test",
            "FGN_FRONTEND_IMAGE": "ghcr.io/example/fgn-serving-frontend:test",
            "FGN_PROXY_IMAGE": "ghcr.io/example/fgn-serving-proxy:test",
        }
    )

    subprocess.run(
        [str(DEPLOY_DIR / "scripts" / "deploy_lab.sh"), "--dry-run"],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )

    assert not (tmp_path / "data" / "deployments").exists()
