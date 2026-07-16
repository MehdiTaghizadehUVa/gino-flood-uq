"""Encode the Portsmouth hero sequence without adding FFmpeg to serving images."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


def _resolve_ffmpeg(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("FGN_FFMPEG_PATH") or shutil.which("ffmpeg")
    if not candidate:
        raise FileNotFoundError(
            "FFmpeg was not found. Set FGN_FFMPEG_PATH or pass --ffmpeg to the encoder."
        )
    return str(candidate)


def _path_for_process(path: Path, *, executable: str) -> str:
    """Translate WSL paths when invoking a Windows FFmpeg executable."""
    if os.name != "nt" and executable.lower().endswith(".exe"):
        completed = subprocess.run(
            ["wslpath", "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    return str(path)


def _run_encode(
    *,
    ffmpeg: str,
    input_pattern: Path,
    destination: Path,
    frame_rate: int,
    codec_args: Sequence[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    local_temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    local_temporary.unlink(missing_ok=True)
    windows_ffmpeg_from_wsl = os.name != "nt" and ffmpeg.lower().endswith(".exe")
    if windows_ffmpeg_from_wsl:
        windows_temp = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", "echo %TEMP%"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        wsl_temp = subprocess.run(
            ["wslpath", "-u", windows_temp],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with tempfile.NamedTemporaryFile(
            prefix="flooduq-hero-",
            suffix=destination.suffix,
            dir=wsl_temp,
            delete=False,
        ) as handle:
            process_temporary = Path(handle.name)
    else:
        process_temporary = local_temporary
    process_temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        str(frame_rate),
        "-start_number",
        "1",
        "-i",
        _path_for_process(input_pattern, executable=ffmpeg),
        "-vf",
        "scale=1280:-2:flags=lanczos,format=yuv420p",
        "-an",
        "-map_metadata",
        "-1",
        *codec_args,
        _path_for_process(process_temporary, executable=ffmpeg),
    ]
    try:
        subprocess.run(command, check=True)
        if not process_temporary.is_file() or process_temporary.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create a valid output: {destination}")
        if process_temporary != local_temporary:
            shutil.copyfile(process_temporary, local_temporary)
        local_temporary.replace(destination)
    finally:
        process_temporary.unlink(missing_ok=True)
        local_temporary.unlink(missing_ok=True)


def encode_case_study_hero(
    *,
    manifest_path: str | Path,
    ffmpeg_path: str | None = None,
    keep_frames: bool = False,
) -> tuple[Path, Path]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    hero = manifest.get("flagship", {}).get("hero", {})
    frame_count = int(hero.get("frameCount", 0))
    frame_rate = int(hero.get("frameRate", 0))
    if frame_count < 2 or frame_rate < 1:
        raise ValueError("Case-study manifest does not define a valid hero frame sequence.")

    frame_dir = manifest_file.parent / "hero" / "frames"
    frames = sorted(frame_dir.glob("irene_*.webp"))
    if len(frames) != frame_count:
        raise ValueError(
            f"Hero frame count mismatch: manifest expects {frame_count}, found {len(frames)} in {frame_dir}."
        )
    expected_names = [f"irene_{index:03d}.webp" for index in range(1, frame_count + 1)]
    if [path.name for path in frames] != expected_names:
        raise ValueError("Hero frame sequence must be contiguous and deterministically numbered.")

    ffmpeg = _resolve_ffmpeg(ffmpeg_path)
    pattern = frame_dir / "irene_%03d.webp"
    mp4 = manifest_file.parent / Path(str(hero["mp4Src"])).name
    webm = manifest_file.parent / Path(str(hero["webmSrc"])).name
    # Assets are stored in the hero directory; URL paths in the manifest are
    # intentionally independent of the operator's local filesystem layout.
    mp4 = frame_dir.parent / mp4.name
    webm = frame_dir.parent / webm.name

    _run_encode(
        ffmpeg=ffmpeg,
        input_pattern=pattern,
        destination=mp4,
        frame_rate=frame_rate,
        codec_args=("-c:v", "libx264", "-preset", "slow", "-crf", "24", "-movflags", "+faststart"),
    )
    _run_encode(
        ffmpeg=ffmpeg,
        input_pattern=pattern,
        destination=webm,
        frame_rate=frame_rate,
        codec_args=(
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "good",
            "-cpu-used",
            "2",
            "-crf",
            "34",
            "-b:v",
            "0",
            "-row-mt",
            "1",
        ),
    )
    if not keep_frames:
        for frame in frames:
            frame.unlink()
        frame_dir.rmdir()
    return mp4, webm


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode the deterministic Portsmouth hero animation.")
    parser.add_argument("--manifest", required=True, help="Path to the exported Portsmouth manifest.json.")
    parser.add_argument("--ffmpeg", help="Optional FFmpeg executable path.")
    parser.add_argument("--keep-frames", action="store_true", help="Retain map-only frame intermediates.")
    args = parser.parse_args(argv)
    mp4, webm = encode_case_study_hero(
        manifest_path=args.manifest,
        ffmpeg_path=args.ffmpeg,
        keep_frames=args.keep_frames,
    )
    print(json.dumps({"mp4": str(mp4), "webm": str(webm)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
