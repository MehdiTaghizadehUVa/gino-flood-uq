"""Prepare train-split normalizer artifacts for flood training without leakage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from torch.utils.data import Subset, random_split

from neuralop.data.transforms.normalizers import save_normalizers
from neuralop.flood.data.wv import (
    FloodDatasetHDF,
    build_normalizer_metadata,
    fit_normalizers,
    load_normalizer_metadata,
    normalizer_metadata_matches,
    resolve_normalizer_fit_method,
    resolve_normalizer_metadata_path,
    save_normalizer_metadata,
)
from neuralop.flood.utils.runtime_core import (
    _cfg_get,
    get_dataset_boundary_kwargs,
    get_dataset_hdf_paths,
    make_split_generator,
    parse_target_variables,
    resolve_split_seed,
    write_train_txt_from_data_root,
)


def _read_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    if isinstance(payload.get("flood"), dict):
        return payload["flood"]
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_raw_dataset(config: Dict[str, Any]) -> FloodDatasetHDF:
    data_cfg = config.setdefault("data", {})
    opt_cfg = config.setdefault("opt", {})
    if bool(_cfg_get(data_cfg, "write_train_txt", False)):
        write_train_txt_from_data_root(
            data_cfg["root"],
            train_txt=_cfg_get(data_cfg, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )
    target_variables = parse_target_variables(
        _cfg_get(data_cfg, "target_variables", ["wd", "vx", "vy"])
    )
    return FloodDatasetHDF(
        data_root=data_cfg["root"],
        n_history=int(_cfg_get(data_cfg, "n_history", 1)),
        query_res=_cfg_get(data_cfg, "query_res", [64, 64]),
        run_ids=None,
        train_txt=_cfg_get(data_cfg, "train_txt", "train.txt"),
        static_text_files=_cfg_get(
            data_cfg,
            "static_text_files",
            ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"],
        ),
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


def prepare_normalizers(
    *,
    config_path: Path,
    artifact_root: Path,
    data_root: Optional[str] = None,
    split_seed: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    started_at = time.time()
    config = _read_config(config_path)
    data_cfg = config.setdefault("data", {})
    dist_cfg = config.setdefault("distributed", {})
    structural_dry_cfg = config.setdefault("structural_dry", {})

    if data_root is not None:
        data_cfg["root"] = str(data_root)
    if split_seed is not None:
        data_cfg["split_seed"] = int(split_seed)

    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    data_cfg["normalizer_root"] = str(artifact_root)

    normalizer_name = str(_cfg_get(data_cfg, "normalizer_path", "normalizers_depth_only.pt"))
    normalizer_path = (artifact_root / Path(normalizer_name).name).resolve()
    data_cfg["normalizer_path"] = normalizer_path.name

    raw_dataset = _build_raw_dataset(config)
    dataset_for_split = raw_dataset
    n_samples_max = _cfg_get(data_cfg, "n_samples_max", None)
    if n_samples_max is not None:
        n_use = min(int(n_samples_max), len(raw_dataset))
        dataset_for_split = Subset(raw_dataset, range(n_use))
    else:
        n_use = len(raw_dataset)

    total_len = len(dataset_for_split)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    resolved_split_seed = resolve_split_seed(data_cfg, int(_cfg_get(dist_cfg, "seed", 123)))
    train_data_raw, _ = random_split(
        dataset_for_split,
        [train_sz, test_sz],
        generator=make_split_generator(resolved_split_seed),
    )

    structural_dry_policy = str(_cfg_get(structural_dry_cfg, "policy", "legacy_full_domain"))
    requested_method = str(_cfg_get(data_cfg, "normalizer_fit_method", "auto"))
    fit_method = resolve_normalizer_fit_method(
        train_data_raw,
        method=requested_method,
        structural_dry_policy=structural_dry_policy,
    )

    metadata_path = resolve_normalizer_metadata_path(normalizer_path)
    expected_metadata = build_normalizer_metadata(
        train_data_raw,
        structural_dry_policy=structural_dry_policy,
        fit_method=fit_method,
    )

    normalizer_status = "computed"
    existing_metadata = load_normalizer_metadata(metadata_path) if normalizer_path.exists() else None
    # CLI prep tool: legitimate strict equality use of the deprecated
    # primitive. The lifecycle helper is the right answer inside a
    # training loop; this script is a standalone batch tool that only
    # cares whether the on-disk artifact matches the requested split.
    _existing_matches = False
    if normalizer_path.exists() and existing_metadata is not None:
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", DeprecationWarning)
            _existing_matches = normalizer_metadata_matches(
                expected_metadata, existing_metadata
            )
    if _existing_matches:
        normalizer_status = "reused"
    elif normalizer_path.exists() and not overwrite:
        raise RuntimeError(
            f"Existing normalizer metadata mismatch for {normalizer_path}. Use --overwrite to recompute."
        )

    if normalizer_status != "reused":
        normalizers, resolved_method = fit_normalizers(
            train_data_raw,
            chunk_size=int(_cfg_get(data_cfg, "normalizer_chunk_size", 10000)),
            expect_target=True,
            structural_dry_policy=structural_dry_policy,
            method=fit_method,
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
        fit_method = resolved_method

    result = {
        "status": "ok",
        "config_path": str(Path(config_path).resolve()),
        "artifact_root": str(artifact_root),
        "data_root": str(Path(data_cfg["root"]).resolve()),
        "normalizer_path": str(normalizer_path),
        "normalizer_metadata_path": str(metadata_path),
        "normalizer_status": normalizer_status,
        "normalizer_fit_method": str(fit_method),
        "structural_dry_policy": structural_dry_policy,
        "split_seed": int(resolved_split_seed),
        "n_samples_effective": int(n_use),
        "total_samples": int(total_len),
        "train_samples": int(train_sz),
        "test_samples": int(test_sz),
        "elapsed_seconds": float(time.time() - started_at),
    }
    summary_path = artifact_root / "prepare_flood_normalizers_summary.json"
    _write_json(summary_path, result)
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = prepare_normalizers(
        config_path=args.config_path,
        artifact_root=args.artifact_root,
        data_root=args.data_root,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
