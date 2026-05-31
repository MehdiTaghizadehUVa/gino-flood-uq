"""Prepare structural-dry and normalizer artifacts for maintained flood training."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import yaml
from torch.utils.data import Subset, random_split

from neuralop.data.transforms.normalizers import save_normalizers
from neuralop.flood.data.structural_dry import (
    build_structural_dry_artifact,
    load_structural_dry_artifact,
    save_structural_dry_artifact,
    validate_structural_dry_artifact,
)
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
    get_structural_dry_policy_kwargs,
    make_split_generator,
    parse_target_variables,
    write_train_txt_from_data_root,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("flood"), dict):
        raise ValueError(f"Expected top-level flood mapping in {path}")
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


def prepare_training_artifacts(
    *,
    config_path: Path,
    artifact_root: Path,
    data_root: Optional[str] = None,
    seed: int = 123,
    overwrite: bool = False,
) -> Dict[str, Any]:
    started_at = time.time()
    payload = _read_yaml(config_path)
    config = payload["flood"]
    data_cfg = config.setdefault("data", {})
    if data_root is not None:
        data_cfg["root"] = str(data_root)

    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    data_cfg["normalizer_root"] = str(artifact_root)

    normalizer_name = str(
        _cfg_get(data_cfg, "normalizer_path", "normalizers_depth_only_masked_primary.pt")
    )
    normalizer_path = (artifact_root / Path(normalizer_name).name).resolve()
    data_cfg["normalizer_path"] = normalizer_path.name

    policy_kwargs = get_structural_dry_policy_kwargs(
        config,
        normalizer_path=normalizer_path,
        allow_data_root_fallback=True,
    )
    if policy_kwargs["policy"] != "masked_primary":
        raise ValueError(
            "prepare_training_artifacts expects structural_dry.policy='masked_primary', "
            f"got {policy_kwargs['policy']!r}."
        )

    raw_dataset = _build_raw_dataset(config)
    artifact_path = policy_kwargs["artifact_path"]
    summary_path = policy_kwargs["summary_path"]
    if artifact_path is None or summary_path is None:
        raise RuntimeError("masked_primary policy did not resolve a structural-dry artifact path")

    artifact_status = "built"
    if artifact_path.exists() and not overwrite:
        artifact = load_structural_dry_artifact(artifact_path)
        artifact = validate_structural_dry_artifact(
            artifact,
            expected_cell_count=raw_dataset.reference_cell_count,
            expected_run_ids=raw_dataset.run_ids,
        )
        artifact_status = "reused"
    else:
        artifact = build_structural_dry_artifact(
            data_root=data_cfg["root"],
            run_ids=raw_dataset.run_ids,
            train_txt=_cfg_get(data_cfg, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
            hdf_paths=raw_dataset.hdf_paths,
            cell_point_index=raw_dataset.cell_point_index,
            mask_definition=policy_kwargs["mask_definition"],
        )
        save_structural_dry_artifact(
            artifact,
            artifact_path=artifact_path,
            summary_path=summary_path,
        )
    raw_dataset.set_structural_dry_mask(artifact["dry_mask"])

    dataset_for_split = raw_dataset
    n_samples_max = _cfg_get(data_cfg, "n_samples_max", None)
    if n_samples_max is not None:
        n_use = min(int(n_samples_max), len(raw_dataset))
        dataset_for_split = Subset(raw_dataset, range(n_use))

    total_len = len(dataset_for_split)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, _ = random_split(
        dataset_for_split,
        [train_sz, test_sz],
        generator=make_split_generator(seed),
    )

    requested_method = str(_cfg_get(data_cfg, "normalizer_fit_method", "auto"))
    fit_method = resolve_normalizer_fit_method(
        train_data_raw,
        method=requested_method,
        structural_dry_policy=policy_kwargs["policy"],
    )
    if fit_method != "streaming":
        raise RuntimeError(
            f"masked_primary prep must resolve to streaming, got {fit_method!r}."
        )

    metadata_path = resolve_normalizer_metadata_path(normalizer_path)
    expected_metadata = build_normalizer_metadata(
        train_data_raw,
        structural_dry_policy=policy_kwargs["policy"],
        fit_method=fit_method,
    )

    normalizer_status = "computed"
    if normalizer_path.exists():
        actual_metadata = load_normalizer_metadata(metadata_path)
        # CLI prep tool: legitimate strict equality use of the deprecated
        # primitive (see normalizer_metadata_matches docstring). Suppress
        # the deprecation noise here because this is a standalone batch
        # script, not a training loop that needs the lifecycle helper.
        import warnings as _warnings

        _matches_existing = False
        if actual_metadata is not None:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", DeprecationWarning)
                _matches_existing = normalizer_metadata_matches(
                    expected_metadata, actual_metadata
                )
        if actual_metadata is None:
            if not overwrite:
                raise RuntimeError(
                    f"Existing normalizer metadata missing for {normalizer_path}. "
                    "Use --overwrite to recompute."
                )
        elif _matches_existing:
            normalizer_status = "reused"
        elif not overwrite:
            raise RuntimeError(
                f"Existing normalizer metadata mismatch for {normalizer_path}. "
                "Use --overwrite to recompute."
            )

    if normalizer_status != "reused":
        normalizers, resolved_method = fit_normalizers(
            train_data_raw,
            chunk_size=int(_cfg_get(data_cfg, "normalizer_chunk_size", 10000)),
            expect_target=True,
            structural_dry_policy=policy_kwargs["policy"],
            method="streaming",
            return_method=True,
        )
        if resolved_method != "streaming":
            raise RuntimeError(
                f"masked_primary prep must resolve to streaming, got {resolved_method!r}."
            )
        save_normalizers(normalizers, normalizer_path)
        save_normalizer_metadata(
            metadata_path,
            build_normalizer_metadata(
                train_data_raw,
                structural_dry_policy=policy_kwargs["policy"],
                fit_method=resolved_method,
            ),
        )
        fit_method = resolved_method

    result = {
        "status": "ok",
        "artifact_status": artifact_status,
        "normalizer_status": normalizer_status,
        "config_path": str(Path(config_path).resolve()),
        "artifact_root": str(artifact_root),
        "data_root": str(Path(data_cfg["root"]).resolve()),
        "normalizer_path": str(normalizer_path),
        "normalizer_metadata_path": str(metadata_path),
        "structural_dry_artifact_path": str(artifact_path),
        "structural_dry_summary_path": str(summary_path),
        "structural_dry_policy": str(policy_kwargs["policy"]),
        "mask_definition": str(policy_kwargs["mask_definition"]),
        "normalizer_fit_method": str(fit_method),
        "seed": int(seed),
        "total_samples": int(total_len),
        "train_samples": int(train_sz),
        "test_samples": int(test_sz),
        "n_dry": int(artifact["n_dry"]),
        "n_wettable": int(artifact["n_wettable"]),
        "cell_count": int(artifact["cell_count"]),
        "run_count": int(len(artifact["run_ids"])),
        "elapsed_seconds": float(time.time() - started_at),
    }
    prep_summary_path = artifact_root / "prepare_flood_training_artifacts_summary.json"
    _write_json(prep_summary_path, result)
    result["prep_summary_path"] = str(prep_summary_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = prepare_training_artifacts(
        config_path=Path(args.config_path),
        artifact_root=Path(args.artifact_root),
        data_root=args.data_root,
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
