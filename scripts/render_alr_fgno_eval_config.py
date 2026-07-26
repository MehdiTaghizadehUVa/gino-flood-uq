#!/usr/bin/env python3
"""Render one event-level ALR-FGNO evaluation configuration."""

import argparse
import os
from pathlib import Path

import yaml


HELDOUT_ROOT = Path(
    "/scratch/jrj6wm/uncertainty_floodmodel/results/coastal/portsmouth/"
    "Coastal_Flood_coastal_v1_5k_test_m100_prod_20260505/test"
)
HELDOUT_BOUNDARY_ROOT = Path(
    "/scratch/jrj6wm/uncertainty_floodmodel/synthetic/coastal/portsmouth/"
    "dynamic_v1_5k_test_m100"
)
HISTORICAL_ROOT = Path(
    "/scratch/jrj6wm/uncertainty_floodmodel/results/coastal/portsmouth/"
    "Coastal_Flood_historical_extreme_events_15min_20260625/test"
)
HISTORICAL_BOUNDARY_ROOT = Path(
    "/scratch/jrj6wm/uncertainty_floodmodel/synthetic/coastal/portsmouth/"
    "historical_extreme_events_15min_20260625_single_member"
)


def _absolute_without_symlink_resolution(path):
    return Path(os.path.abspath(str(path.expanduser())))


def _load(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("flood"), dict):
        raise ValueError("Expected a top-level flood mapping in {}.".format(path))
    return payload


def _dump_compatible(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            yaml.safe_dump(payload, handle, sort_keys=False)
        except TypeError:
            yaml.safe_dump(payload, handle)


def _checkpoint_state_exists(checkpoint_dir):
    return any(
        (checkpoint_dir / name).is_file()
        for name in ("best_model_state_dict.pt", "model_state_dict.pt")
    )


def _event_runs(dataset, event_id):
    if dataset == "heldout":
        if not event_id.startswith("TE") or len(event_id) != 8:
            raise ValueError("Held-out event IDs must have form TE000001.")
        return [
            "Flood_coastal_{}_sim{:02d}".format(event_id, member)
            for member in range(100)
        ]
    if not event_id.startswith("Flood_coastal_HIST_"):
        raise ValueError(
            "Historical event IDs must be HDF stems without the _sim00 suffix."
        )
    return [event_id + "_sim00"]


def render_eval_config(base, output, run_dir, checkpoint_dir, dataset, event_id):
    if not _checkpoint_state_exists(checkpoint_dir):
        raise FileNotFoundError(
            "No ALR model state found in {}.".format(checkpoint_dir)
        )
    payload = _load(base)
    flood = payload["flood"]
    split_path = run_dir / "splits" / (event_id + ".txt")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(_event_runs(dataset, event_id)) + "\n")

    if dataset == "heldout":
        data_root = HELDOUT_ROOT
        boundary_root = HELDOUT_BOUNDARY_ROOT
        stage_file = "Stage_Hydrographs_Test_Clean.txt"
        precipitation_file = "Precipitation_Test_Clean.txt"
    elif dataset == "historical":
        data_root = HISTORICAL_ROOT
        boundary_root = HISTORICAL_BOUNDARY_ROOT
        stage_file = "Stage_Hydrographs_Historical_Clean.txt"
        precipitation_file = "Precipitation_Historical_Clean.txt"
    else:
        raise ValueError("dataset must be heldout or historical")

    for section_name in ("data", "rollout_data"):
        section = flood[section_name]
        section["root"] = str(data_root)
        boundary = section["boundary"]["channels"]
        boundary[0]["clean_boundary_root"] = str(boundary_root)
        boundary[0]["clean_boundary_file"] = stage_file
        boundary[1]["clean_boundary_root"] = str(boundary_root)
        boundary[1]["clean_boundary_file"] = precipitation_file

    flood["data"]["train_txt"] = str(split_path)
    flood["data"]["write_train_txt"] = False
    flood["data"]["force_load_normalizers"] = True
    flood["data"]["batch_size"] = 64
    flood["data"]["dt"] = 900
    flood["rollout_data"]["test_txt"] = str(split_path)

    # Metadata is the primary reconstruction path. Keeping the architecture
    # namespace in gino also makes config fallback correct for older checkpoints.
    flood["gino"]["anchored_low_rank"] = flood["model"]["anchored_low_rank"]
    flood["checkpoint"]["save_dir"] = str(checkpoint_dir)
    flood["checkpoint"]["resume_from_dir"] = None
    flood["checkpoint"]["eval_name"] = "best"
    event_out = run_dir / "events" / event_id
    flood["rollout"]["run_after_training"] = False
    flood["rollout"]["out_dir"] = str(event_out)
    flood["rollout"]["forecast_artifact_dir"] = str(
        run_dir / "forecast_artifacts" / dataset
    )
    flood["rollout"]["n_ensemble_samples"] = 60
    flood["rollout"]["alr_aleatory_samples"] = 15
    flood["rollout"]["write_visualizations"] = False
    flood["rollout"]["forward_timing_path"] = str(
        event_out / "forward_only_timing.json"
    )
    flood["visualization"]["map"]["enabled"] = False
    flood["visualization"]["output"]["write_gif"] = False
    flood["visualization"]["output"]["write_mp4"] = False
    flood["log_file"] = str(event_out / "evaluation.log")
    flood["use_progress_bar"] = False
    flood["wandb"]["log"] = False
    flood["alr_evaluation"] = {
        "dataset": dataset,
        "event_id": event_id,
        "checkpoint_dir": str(checkpoint_dir),
        "members": 60,
        "particles": 4,
        "aleatory_per_particle": 15,
    }
    _dump_compatible(payload, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("heldout", "historical"), required=True)
    parser.add_argument("--event-id", required=True)
    args = parser.parse_args()
    path = render_eval_config(
        _absolute_without_symlink_resolution(args.base),
        _absolute_without_symlink_resolution(args.output),
        _absolute_without_symlink_resolution(args.run_dir),
        _absolute_without_symlink_resolution(args.checkpoint_dir),
        args.dataset,
        args.event_id,
    )
    print(path)


if __name__ == "__main__":
    main()
