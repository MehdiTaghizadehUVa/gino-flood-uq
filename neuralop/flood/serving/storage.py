"""Artifact storage seam for flood serving."""

from __future__ import annotations

import json
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class ArtifactRef:
    run_id: str
    artifact_id: str
    path: Path
    content_type: str
    size_bytes: int


class ArtifactStore(Protocol):
    def put_bytes(self, run_id: str, artifact_id: str, data: bytes, *, content_type: str) -> ArtifactRef: ...
    def put_json(self, run_id: str, artifact_id: str, payload: object) -> ArtifactRef: ...
    def list(self, run_id: str) -> Iterable[ArtifactRef]: ...
    def read_bytes(self, run_id: str, artifact_id: str) -> bytes: ...
    def delete_run_artifacts(self, run_id: str) -> None: ...


class LocalArtifactStore:
    """Local filesystem artifact adapter."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        safe = Path(run_id).name
        if safe != run_id:
            raise ValueError("run_id may not contain path separators.")
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path(self, run_id: str, artifact_id: str) -> Path:
        safe = Path(artifact_id).name
        if safe != artifact_id:
            raise ValueError("artifact_id may not contain path separators.")
        return self._run_dir(run_id) / safe

    def put_bytes(self, run_id: str, artifact_id: str, data: bytes, *, content_type: str) -> ArtifactRef:
        path = self._artifact_path(run_id, artifact_id)
        path.write_bytes(data)
        return ArtifactRef(run_id, artifact_id, path, content_type, path.stat().st_size)

    def put_json(self, run_id: str, artifact_id: str, payload: object) -> ArtifactRef:
        if not artifact_id.endswith(".json"):
            artifact_id = f"{artifact_id}.json"
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        return self.put_bytes(run_id, artifact_id, data, content_type="application/json")

    def list(self, run_id: str) -> Iterable[ArtifactRef]:
        path = self._run_dir(run_id)
        refs = []
        for item in sorted(path.iterdir()):
            if item.is_file():
                ctype = _content_type_for_suffix(item)
                refs.append(ArtifactRef(run_id, item.name, item, ctype, item.stat().st_size))
        return refs

    def read_bytes(self, run_id: str, artifact_id: str) -> bytes:
        path = self._artifact_path(run_id, artifact_id)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        return path.read_bytes()

    def delete_run_artifacts(self, run_id: str) -> None:
        path = self.root / Path(run_id).name
        if path.exists():
            shutil.rmtree(path)


def _content_type_for_suffix(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix in {".h5", ".hdf5"}:
        return "application/x-hdf5"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"
