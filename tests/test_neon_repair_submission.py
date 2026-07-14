from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMIT = REPO_ROOT / "scripts" / "submit_neon_repair_rung.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _submission_fixture(tmp_path: Path) -> tuple[dict[str, str], Path]:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "scripts").mkdir()
    _write_executable(repo / "scripts" / "sbatch_neon_repair_rung.sh", "#!/bin/sh\nexit 0\n")
    container = tmp_path / "container.sif"
    container.write_bytes(b"container")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        """#!/bin/sh
case " $* " in
  *" rev-parse HEAD "*) echo deadbeef ;;
  *" status --porcelain "*) : ;;
  *) exit 2 ;;
esac
""",
    )
    _write_executable(fake_bin / "module", "#!/bin/sh\nexit 0\n")
    _write_executable(
        fake_bin / "apptainer",
        """#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    NEON_PREFLIGHT_PATH=*) path=${arg#NEON_PREFLIGHT_PATH=} ;;
  esac
done
if [ -n "${path:-}" ]; then
  mkdir -p "$(dirname "$path")"
  printf '{"schema_version":"neon_repair_preflight_v1","token":"%s"}\n' \
    "${MOCK_PREFLIGHT_TOKEN:-stable}" > "$path"
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "squeue",
        "#!/bin/sh\n[ \"${MOCK_SQUEUE_ACTIVE:-0}\" = 1 ] && echo active\nexit 0\n",
    )
    output = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        NEON_REPO=str(repo),
        NEON_CONTAINER=str(container),
        NEON_RUN_ROOT=str(tmp_path / "runs"),
        NEON_OUT_DIR=str(output),
        NEON_CACHE_DIR=str(tmp_path / "cache"),
        NEON_SUBMIT_DRY_RUN="1",
    )
    return env, output


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMIT), "B0"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_submission_guards_initial_and_exact_continuation(tmp_path):
    env, output = _submission_fixture(tmp_path)
    initial = _run(env)
    assert initial.returncode == 0, initial.stderr
    assert "mode=initial" in initial.stdout

    output.mkdir(exist_ok=True)
    (output / "job_id.txt").write_text("123\n", encoding="utf-8")
    env["NEON_CONTINUE"] = "1"
    missing_state = _run(env)
    assert missing_state.returncode == 2
    assert "completed-epoch state" in missing_state.stderr

    (output / "neon_stage2_latest_state.pt").write_bytes(b"state")
    (output / "git_head.txt").write_text("deadbeef\n", encoding="utf-8")
    continuation = _run(env)
    assert continuation.returncode == 0, continuation.stderr
    assert "mode=continuation" in continuation.stdout

    env["MOCK_SQUEUE_ACTIVE"] = "1"
    duplicate = _run(env)
    assert duplicate.returncode == 2
    assert "still active" in duplicate.stderr


def test_submission_rejects_cross_commit_continuation(tmp_path):
    env, output = _submission_fixture(tmp_path)
    output.mkdir()
    (output / "job_id.txt").write_text("123\n", encoding="utf-8")
    (output / "neon_stage2_latest_state.pt").write_bytes(b"state")
    (output / "git_head.txt").write_text("different-head\n", encoding="utf-8")
    env["NEON_CONTINUE"] = "1"

    result = _run(env)
    assert result.returncode == 2
    assert "cross-commit continuation" in result.stderr


def test_continuation_preflight_mismatch_preserves_authoritative_manifest(tmp_path):
    env, output = _submission_fixture(tmp_path)
    initial = _run(env)
    assert initial.returncode == 0, initial.stderr
    authoritative = output / "preflight.json"
    original = authoritative.read_bytes()

    (output / "job_id.txt").write_text("123\n", encoding="utf-8")
    (output / "neon_stage2_latest_state.pt").write_bytes(b"state")
    (output / "git_head.txt").write_text("deadbeef\n", encoding="utf-8")
    env.update(NEON_CONTINUE="1", MOCK_PREFLIGHT_TOKEN="changed")

    mismatch = _run(env)

    assert mismatch.returncode == 2
    assert "validated preflight config changed" in mismatch.stderr
    assert authoritative.read_bytes() == original
    assert not list(output.glob("preflight.json.candidate.*"))
