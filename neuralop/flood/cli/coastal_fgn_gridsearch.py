from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

LEARNING_RATES = (5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4)
WEIGHT_DECAYS = (1.0e-4, 5.0e-4)
GNO_RADII = (0.08, 0.10, 0.12)
FNO_HIDDEN_CHANNELS = (64, 96)
FGN_NOISE_DIMS = (16, 32)
N_EPOCHS_DEFAULT = 30
SEED_DEFAULT = 123
DEFAULT_STATUS = "completed"
_RUN_TAG_PREFIX = "coastal_fgn_gs30"


_GRID = tuple(
    itertools.product(
        LEARNING_RATES,
        WEIGHT_DECAYS,
        GNO_RADII,
        FNO_HIDDEN_CHANNELS,
        FGN_NOISE_DIMS,
    )
)


def grid_size() -> int:
    return len(_GRID)


def _format_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    exp = int(exponent)
    return f"{mantissa}e{exp}"


def _format_radius(value: float) -> str:
    return f"{value:.2f}"


def spec_for_index(index: int) -> dict[str, Any]:
    if index < 0 or index >= grid_size():
        raise ValueError(f"Grid index {index} is out of range for {grid_size()} combinations.")
    learning_rate, weight_decay, gno_radius, fno_hidden_channels, fgn_noise_dim = _GRID[index]
    return {
        "index": int(index),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gno_radius": float(gno_radius),
        "fno_hidden_channels": int(fno_hidden_channels),
        "fgn_noise_dim": int(fgn_noise_dim),
    }


def build_run_tag(spec: dict[str, Any]) -> str:
    return (
        f"{_RUN_TAG_PREFIX}_"
        f"lr{_format_scientific(spec['learning_rate'])}_"
        f"wd{_format_scientific(spec['weight_decay'])}_"
        f"r{_format_radius(spec['gno_radius'])}_"
        f"h{spec['fno_hidden_channels']}_"
        f"z{spec['fgn_noise_dim']}_"
        f"idx{spec['index']}"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "flood" not in data or not isinstance(data["flood"], dict):
        raise ValueError(f"Expected top-level 'flood' mapping in {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_and_apply_common_overrides(
    *,
    base_config_path: Path,
    data_root: str | None,
    clean_boundary_root: str | None,
    seed: int,
    deterministic: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_yaml(base_config_path)
    config = payload["flood"]

    if data_root is not None:
        config.setdefault("data", {})["root"] = data_root
        config.setdefault("rollout_data", {})["root"] = data_root
    _ensure_clean_family_root(config.setdefault("data", {}), clean_boundary_root)
    _ensure_clean_family_root(config.setdefault("rollout_data", {}), clean_boundary_root)

    config.setdefault("distributed", {})["seed"] = int(seed)
    config["deterministic"] = bool(deterministic)
    config["verify_training"] = False
    config.setdefault("rollout", {})["run_after_training"] = False
    return payload, config


def _ensure_clean_family_root(config_section: dict[str, Any], clean_boundary_root: str | None) -> None:
    if clean_boundary_root is None:
        return
    boundary = config_section.get("boundary", {})
    channels = boundary.get("channels", []) if isinstance(boundary, dict) else []
    for channel in channels:
        if isinstance(channel, dict) and channel.get("mode") == "clean_family":
            channel["clean_boundary_root"] = clean_boundary_root


def render_config(
    *,
    base_config_path: Path,
    output_config_path: Path,
    index: int,
    checkpoint_dir: Path,
    normalizer_root: Path,
    wandb_group: str,
    wandb_name: str,
    data_root: str | None = None,
    clean_boundary_root: str | None = None,
    n_epochs: int = N_EPOCHS_DEFAULT,
    seed: int = SEED_DEFAULT,
    deterministic: bool = False,
    wandb_log: bool = True,
) -> dict[str, Any]:
    payload, config = _load_and_apply_common_overrides(
        base_config_path=base_config_path,
        data_root=data_root,
        clean_boundary_root=clean_boundary_root,
        seed=seed,
        deterministic=deterministic,
    )
    spec = spec_for_index(index)

    config.setdefault("data", {})["normalizer_root"] = str(normalizer_root)
    config["data"]["normalizer_path"] = "normalizers_depth_only.pt"

    config.setdefault("checkpoint", {})["save_dir"] = str(checkpoint_dir)
    config["checkpoint"]["resume_from_dir"] = None
    config["checkpoint"]["save_every"] = int(n_epochs)
    config["checkpoint"]["save_best_metric"] = "test_crps"

    config.setdefault("opt", {})["n_epochs"] = int(n_epochs)
    config["opt"]["learning_rate"] = float(spec["learning_rate"])
    config["opt"]["weight_decay"] = float(spec["weight_decay"])
    config["opt"]["amp_autocast"] = False

    config.setdefault("gino", {})["gno_radius"] = float(spec["gno_radius"])
    config["gino"]["fno_hidden_channels"] = int(spec["fno_hidden_channels"])
    config["gino"]["fgn_noise_dim"] = int(spec["fgn_noise_dim"])

    config.setdefault("wandb", {})["log"] = bool(wandb_log)
    config["wandb"]["group"] = str(wandb_group)
    config["wandb"]["name"] = str(wandb_name)

    _write_yaml(output_config_path, payload)

    rendered = dict(spec)
    rendered.update(
        {
            "run_tag": build_run_tag(spec),
            "config_path": str(output_config_path),
            "checkpoint_dir": str(checkpoint_dir),
            "normalizer_root": str(normalizer_root),
            "wandb_group": str(wandb_group),
            "wandb_name": str(wandb_name),
            "n_epochs": int(n_epochs),
            "seed": int(seed),
            "deterministic": bool(deterministic),
        }
    )
    return rendered


def precompute_normalizers(
    *,
    base_config_path: Path,
    normalizer_root: Path,
    data_root: str | None = None,
    clean_boundary_root: str | None = None,
    seed: int = SEED_DEFAULT,
    chunk_size: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    from torch.utils.data import Subset, random_split

    from neuralop.data.transforms.normalizers import save_normalizers
    from neuralop.flood.data.wv import (
        FloodDatasetHDF,
        build_normalizer_metadata,
        fit_normalizers,
        resolve_normalizer_metadata_path,
        save_normalizer_metadata,
    )
    from neuralop.flood.utils.runtime_core import (
        _cfg_get,
        get_dataset_boundary_kwargs,
        get_dataset_hdf_paths,
        make_split_generator,
        parse_target_variables,
        write_train_txt_from_data_root,
    )

    _, config = _load_and_apply_common_overrides(
        base_config_path=base_config_path,
        data_root=data_root,
        clean_boundary_root=clean_boundary_root,
        seed=seed,
        deterministic=False,
    )
    data_cfg = config.setdefault("data", {})
    opt_cfg = config.setdefault("opt", {})

    normalizer_root = normalizer_root.resolve()
    normalizer_root.mkdir(parents=True, exist_ok=True)
    normalizer_path = (normalizer_root / "normalizers_depth_only.pt").resolve()
    metadata_path = resolve_normalizer_metadata_path(normalizer_path)
    if normalizer_path.exists() and not overwrite:
        return {
            "status": "exists",
            "normalizer_path": str(normalizer_path),
            "metadata_path": str(metadata_path),
            "seed": int(seed),
        }

    if bool(_cfg_get(data_cfg, "write_train_txt", False)):
        write_train_txt_from_data_root(
            data_cfg["root"],
            train_txt=_cfg_get(data_cfg, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )

    target_variables = parse_target_variables(_cfg_get(data_cfg, "target_variables", ["wd", "vx", "vy"]))
    full_dataset = FloodDatasetHDF(
        data_root=data_cfg["root"],
        n_history=int(_cfg_get(data_cfg, "n_history", 1)),
        query_res=_cfg_get(data_cfg, "query_res", [64, 64]),
        run_ids=None,
        train_txt=_cfg_get(data_cfg, "train_txt", "train.txt"),
        static_text_files=_cfg_get(data_cfg, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]),
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=int(_cfg_get(data_cfg, "skip_before_timestep", 0)),
        noise_type=_cfg_get(data_cfg, "noise_type", "none"),
        noise_std=_cfg_get(data_cfg, "noise_std", None),
        hdf_paths=get_dataset_hdf_paths(data_cfg),
        ar_rollout_steps=max(1, int(_cfg_get(opt_cfg, "ar_rollout_steps", 1))),
        target_variables=target_variables,
        **get_dataset_boundary_kwargs(data_cfg),
    )

    n_samples_max = _cfg_get(data_cfg, "n_samples_max", None)
    if n_samples_max is not None:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))

    total_len = len(full_dataset)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, _ = random_split(
        full_dataset, [train_sz, test_sz], generator=make_split_generator(seed)
    )

    structural_dry_cfg = config.setdefault("structural_dry", {})
    structural_dry_policy = str(_cfg_get(structural_dry_cfg, "policy", "legacy_full_domain"))
    fit_chunk_size = int(chunk_size if chunk_size is not None else _cfg_get(data_cfg, "normalizer_chunk_size", 10000))
    normalizers, resolved_method = fit_normalizers(
        train_data_raw,
        chunk_size=fit_chunk_size,
        expect_target=True,
        structural_dry_policy=structural_dry_policy,
        method=str(_cfg_get(data_cfg, "normalizer_fit_method", "auto")),
        return_method=True,
    )
    save_normalizers(normalizers, normalizer_path)
    save_normalizer_metadata(
        metadata_path,
        build_normalizer_metadata(
            train_data_raw,
            structural_dry_policy=structural_dry_policy,
            fit_method=resolved_method,
        ),
    )
    return {
        "status": "computed",
        "normalizer_path": str(normalizer_path),
        "metadata_path": str(metadata_path),
        "seed": int(seed),
        "total_samples": int(total_len),
        "train_samples": int(train_sz),
        "test_samples": int(test_sz),
        "chunk_size": int(fit_chunk_size),
        "target_variables": list(target_variables),
        "resolved_method": str(resolved_method),
    }


_EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+time=(?P<time>[0-9.]+)s\s+\|\s+avg_loss=(?P<avg_loss>[0-9.eE+-]+)\s+\|\s+train_err=(?P<train_err>[0-9.eE+-]+)"
)
_EVAL_RE = re.compile(r"Eval:\s+(?P<body>.+)")


def parse_training_log(log_path: Path) -> dict[str, Any]:
    epochs: dict[int, dict[str, float]] = {}
    current_epoch: int | None = None
    if not log_path.exists():
        return {
            "epochs": [],
            "best_epoch": None,
            "best_test_crps": None,
            "best_epoch_test_l2": None,
            "final_epoch": None,
            "final_metrics": {},
        }

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            epoch_match = _EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group("epoch"))
                record = epochs.setdefault(current_epoch, {})
                record["epoch"] = float(current_epoch)
                record["time"] = float(epoch_match.group("time"))
                record["avg_loss"] = float(epoch_match.group("avg_loss"))
                record["train_err"] = float(epoch_match.group("train_err"))
                continue

            eval_match = _EVAL_RE.search(line)
            if eval_match and current_epoch is not None:
                record = epochs.setdefault(current_epoch, {})
                for piece in eval_match.group("body").split(","):
                    piece = piece.strip()
                    if not piece or "=" not in piece:
                        continue
                    key, value = piece.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        record[key] = float(value)
                    except ValueError:
                        continue

    ordered_epochs = [epochs[idx] for idx in sorted(epochs.keys())]
    final_metrics = ordered_epochs[-1] if ordered_epochs else {}

    best_epoch = None
    best_test_crps = None
    best_epoch_test_l2 = None
    for record in ordered_epochs:
        value = record.get("test_crps")
        if value is None:
            continue
        if best_test_crps is None or value < best_test_crps:
            best_test_crps = value
            best_epoch = int(record["epoch"])
            best_epoch_test_l2 = record.get("test_l2")

    return {
        "epochs": ordered_epochs,
        "best_epoch": best_epoch,
        "best_test_crps": best_test_crps,
        "best_epoch_test_l2": best_epoch_test_l2,
        "final_epoch": int(final_metrics["epoch"]) if final_metrics else None,
        "final_metrics": final_metrics,
    }


def write_run_summary(
    *,
    summary_path: Path,
    index: int,
    status: str,
    job_id: str | None,
    array_task_id: str | None,
    git_sha: str,
    config_path: Path,
    checkpoint_dir: Path,
    log_path: Path,
    run_tag: str | None = None,
) -> dict[str, Any]:
    spec = spec_for_index(index)
    parsed = parse_training_log(log_path)
    summary = {
        "status": status,
        "job_id": job_id,
        "array_task_id": array_task_id,
        "git_sha": git_sha,
        "run_tag": run_tag or build_run_tag(spec),
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(log_path),
        "hyperparameters": {
            "learning_rate": spec["learning_rate"],
            "weight_decay": spec["weight_decay"],
            "gno_radius": spec["gno_radius"],
            "fno_hidden_channels": spec["fno_hidden_channels"],
            "fgn_noise_dim": spec["fgn_noise_dim"],
        },
        "best_epoch": parsed["best_epoch"],
        "best_test_crps": parsed["best_test_crps"],
        "best_epoch_test_l2": parsed["best_epoch_test_l2"],
        "final_epoch": parsed["final_epoch"],
        "final_metrics": parsed["final_metrics"],
        "epochs_seen": len(parsed["epochs"]),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def rank_summaries(summary_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(summary_root.rglob("run_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["summary_path"] = str(path)
        summaries.append(payload)

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        best_crps = item.get("best_test_crps")
        best_l2 = item.get("best_epoch_test_l2")
        return (
            best_crps is None,
            math.inf if best_crps is None else float(best_crps),
            best_l2 is None,
            math.inf if best_l2 is None else float(best_l2),
            item.get("run_tag", ""),
        )

    return sorted(summaries, key=sort_key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helpers for coastal FGN grid-search runs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Describe one grid-search combination")
    describe.add_argument("--index", type=int, required=True)
    describe.add_argument("--format", choices=("json", "shell"), default="json")

    render = subparsers.add_parser("render-config", help="Render a per-run config file")
    render.add_argument("--index", type=int, required=True)
    render.add_argument("--base-config", type=Path, required=True)
    render.add_argument("--output-config", type=Path, required=True)
    render.add_argument("--checkpoint-dir", type=Path, required=True)
    render.add_argument("--normalizer-root", type=Path, required=True)
    render.add_argument("--wandb-group", type=str, required=True)
    render.add_argument("--wandb-name", type=str, required=True)
    render.add_argument("--data-root", type=str, default=None)
    render.add_argument("--clean-boundary-root", type=str, default=None)
    render.add_argument("--n-epochs", type=int, default=N_EPOCHS_DEFAULT)
    render.add_argument("--seed", type=int, default=SEED_DEFAULT)
    render.add_argument("--deterministic", choices=("true", "false"), default="false")
    render.add_argument("--wandb-log", choices=("true", "false"), default="true")

    precompute = subparsers.add_parser("precompute-normalizers", help="Fit and save one shared normalizer artifact")
    precompute.add_argument("--base-config", type=Path, required=True)
    precompute.add_argument("--normalizer-root", type=Path, required=True)
    precompute.add_argument("--data-root", type=str, default=None)
    precompute.add_argument("--clean-boundary-root", type=str, default=None)
    precompute.add_argument("--seed", type=int, default=SEED_DEFAULT)
    precompute.add_argument("--chunk-size", type=int, default=None)
    precompute.add_argument("--overwrite", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Write run_summary.json from a training log")
    summarize.add_argument("--index", type=int, required=True)
    summarize.add_argument("--summary-path", type=Path, required=True)
    summarize.add_argument("--log-path", type=Path, required=True)
    summarize.add_argument("--config-path", type=Path, required=True)
    summarize.add_argument("--checkpoint-dir", type=Path, required=True)
    summarize.add_argument("--status", type=str, default=DEFAULT_STATUS)
    summarize.add_argument("--job-id", type=str, default=None)
    summarize.add_argument("--array-task-id", type=str, default=None)
    summarize.add_argument("--git-sha", type=str, required=True)
    summarize.add_argument("--run-tag", type=str, default=None)

    rank = subparsers.add_parser("rank", help="Rank run summaries under a directory")
    rank.add_argument("--summary-root", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def _emit_shell_description(spec: dict[str, Any]) -> str:
    payload = {
        "GS_INDEX": spec["index"],
        "GS_LR": spec["learning_rate"],
        "GS_WD": spec["weight_decay"],
        "GS_RADIUS": spec["gno_radius"],
        "GS_HIDDEN": spec["fno_hidden_channels"],
        "GS_NOISE_DIM": spec["fgn_noise_dim"],
        "GS_RUN_TAG": build_run_tag(spec),
    }
    lines = [f"export {key}={shlex.quote(str(value))}" for key, value in payload.items()]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    if args.command == "describe":
        spec = spec_for_index(args.index)
        spec["run_tag"] = build_run_tag(spec)
        if args.format == "json":
            print(json.dumps(spec, indent=2, sort_keys=True))
        else:
            print(_emit_shell_description(spec))
        return 0

    if args.command == "render-config":
        rendered = render_config(
            base_config_path=args.base_config,
            output_config_path=args.output_config,
            index=args.index,
            checkpoint_dir=args.checkpoint_dir,
            normalizer_root=args.normalizer_root,
            wandb_group=args.wandb_group,
            wandb_name=args.wandb_name,
            data_root=args.data_root,
            clean_boundary_root=args.clean_boundary_root,
            n_epochs=args.n_epochs,
            seed=args.seed,
            deterministic=(args.deterministic == "true"),
            wandb_log=(args.wandb_log == "true"),
        )
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return 0

    if args.command == "precompute-normalizers":
        result = precompute_normalizers(
            base_config_path=args.base_config,
            normalizer_root=args.normalizer_root,
            data_root=args.data_root,
            clean_boundary_root=args.clean_boundary_root,
            seed=args.seed,
            chunk_size=args.chunk_size,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "summarize":
        summary = write_run_summary(
            summary_path=args.summary_path,
            index=args.index,
            status=args.status,
            job_id=args.job_id,
            array_task_id=args.array_task_id,
            git_sha=args.git_sha,
            config_path=args.config_path,
            checkpoint_dir=args.checkpoint_dir,
            log_path=args.log_path,
            run_tag=args.run_tag,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "rank":
        ranked = rank_summaries(args.summary_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(ranked, handle, indent=2, sort_keys=True)
        print(json.dumps({"summary_root": str(args.summary_root), "n_ranked": len(ranked), "output": str(args.output)}, indent=2, sort_keys=True))
        return 0

    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
