"""Encode Portsmouth marketing videos without adding FFmpeg to serving images."""

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
    frame_count: int | None,
    playback_frame_rate: int | None,
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
    filters = ["scale=1280:-2:flags=lanczos"]
    if playback_frame_rate is not None and playback_frame_rate > frame_rate:
        if frame_count is None or frame_count < 2:
            raise ValueError("Interpolated video encoding requires a valid source frame count.")
        # Source states remain exact. Blended display frames prevent browser
        # decode gaps without inventing additional model output timesteps.
        filters.extend(
            (
                f"tpad=stop_mode=clone:stop_duration={2.0 / frame_rate:.6f}",
                f"minterpolate=fps={playback_frame_rate}:mi_mode=blend",
                f"trim=duration={frame_count / frame_rate:.6f}",
                "setpts=PTS-STARTPTS",
            )
        )
    filters.append("format=yuv420p")
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
        ",".join(filters),
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
        frame_count=frame_count,
        playback_frame_rate=None,
        codec_args=("-c:v", "libx264", "-preset", "slow", "-crf", "24", "-movflags", "+faststart"),
    )
    _run_encode(
        ffmpeg=ffmpeg,
        input_pattern=pattern,
        destination=webm,
        frame_rate=frame_rate,
        frame_count=frame_count,
        playback_frame_rate=None,
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


def encode_case_study_products(
    *,
    manifest_path: str | Path,
    ffmpeg_path: str | None = None,
) -> dict[str, Path]:
    """Encode the complete product sequences declared in the case-study manifest."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    products = manifest.get("flagship", {}).get("products", [])
    if not products:
        raise ValueError("Case-study manifest does not define product animation sequences.")

    validated: list[tuple[str, Path, dict[str, object]]] = []
    for product in products:
        product_id = str(product.get("id", ""))
        animation = product.get("animation", {})
        if not isinstance(animation, dict):
            raise ValueError(f"{product_id or 'unknown'} product animation metadata is malformed.")
        frame_count = int(animation.get("frameCount", 0))
        source_frame_rate = int(animation.get("sourceFrameRate", 0))
        playback_frame_rate = int(animation.get("playbackFrameRate", 0))
        if not product_id or frame_count < 2 or source_frame_rate < 1 or playback_frame_rate < source_frame_rate:
            raise ValueError(f"{product_id or 'unknown'} product animation metadata is incomplete.")

        frame_dir = manifest_file.parent / "frames" / product_id
        frames = sorted(frame_dir.glob("irene_t*.webp"))
        if len(frames) != frame_count:
            raise ValueError(
                f"{product_id} frame count mismatch: manifest expects {frame_count}, "
                f"found {len(frames)} in {frame_dir}."
            )
        expected_names = [f"irene_t{index:03d}.webp" for index in range(1, frame_count + 1)]
        if [path.name for path in frames] != expected_names:
            raise ValueError(f"{product_id} frame sequence must be contiguous and deterministically numbered.")
        validated.append((product_id, frame_dir, animation))

    ffmpeg = _resolve_ffmpeg(ffmpeg_path)
    outputs: dict[str, Path] = {}
    for product_id, frame_dir, animation in validated:
        source_frame_rate = int(animation["sourceFrameRate"])
        playback_frame_rate = int(animation["playbackFrameRate"])
        animation_dir = manifest_file.parent / "animations"
        mp4 = animation_dir / Path(str(animation["mp4Src"])).name
        pattern = frame_dir / "irene_t%03d.webp"
        _run_encode(
            ffmpeg=ffmpeg,
            input_pattern=pattern,
            destination=mp4,
            frame_rate=source_frame_rate,
            frame_count=int(animation["frameCount"]),
            playback_frame_rate=playback_frame_rate,
            codec_args=("-c:v", "libx264", "-preset", "slow", "-crf", "25", "-movflags", "+faststart"),
        )
        outputs[product_id] = mp4
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode deterministic Portsmouth marketing animations.")
    parser.add_argument("--manifest", required=True, help="Path to the exported Portsmouth manifest.json.")
    parser.add_argument("--ffmpeg", help="Optional FFmpeg executable path.")
    parser.add_argument("--keep-frames", action="store_true", help="Retain map-only frame intermediates.")
    parser.add_argument(
        "--scope",
        choices=("all", "hero", "products"),
        default="all",
        help="Video family to encode (default: all).",
    )
    args = parser.parse_args(argv)
    payload: dict[str, object] = {}
    if args.scope in {"all", "hero"}:
        mp4, webm = encode_case_study_hero(
            manifest_path=args.manifest,
            ffmpeg_path=args.ffmpeg,
            keep_frames=args.keep_frames,
        )
        payload["hero"] = {"mp4": str(mp4), "webm": str(webm)}
    if args.scope in {"all", "products"}:
        payload["products"] = {
            product_id: {"mp4": str(path)}
            for product_id, path in encode_case_study_products(
                manifest_path=args.manifest,
                ffmpeg_path=args.ffmpeg,
            ).items()
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
