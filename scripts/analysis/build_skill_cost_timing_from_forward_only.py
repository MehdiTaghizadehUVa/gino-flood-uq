#!/usr/bin/env python3
"""Build skill-cost timing JSON from forward-only benchmark outputs.

Supports both the older flat timing directory and the newer ensemble-size grid
layout produced under n01/n02/n05/... directories. Missing requested sizes are
linearly extrapolated and marked estimated=true.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

DEFAULT_FORWARD_ROOT = Path(
    "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_uq_model_comparison/"
    "outputs/skill_cost_forward_timing_a100_10events_20260612_180547"
)
DEFAULT_SIZES = (1, 2, 5, 10, 20, 40, 60)
METHOD_NAME = {
    "fgno": "FGNO",
    "mc_dropout": "MC-dropout GINO",
    "diffusion": "Diffusion",
}
DIFFUSION_RE = re.compile(r"Forward-only diffusion (?P<event>\S+): (?P<seconds>[0-9.]+)s for (?P<steps>\d+) steps x (?P<members>\d+) members")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_operator_result(path: Path) -> Dict[str, object]:
    payload = _read_json(path)
    n_measured = int(payload["ensemble_members"])
    forward = payload.get("forward_rollout_seconds", {})
    per_member = payload.get("seconds_per_member_rollout", {})
    return {
        "n_members": n_measured,
        "seconds_mean": float(forward.get("mean", float(per_member["mean"]) * n_measured)),
        "seconds_std": float(forward.get("std", float(per_member.get("std", 0.0)) * n_measured)),
        "measured_event_count": int(payload.get("n_events", len(payload.get("events", [])) or 0)),
        "cuda_device_name": str(payload.get("cuda_device_name", "")),
        "source": str(path),
    }


def _load_diffusion_from_log(path: Path) -> Dict[str, object]:
    values = []
    members = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DIFFUSION_RE.search(line)
        if not match:
            continue
        values.append(float(match.group("seconds")))
        members = int(match.group("members"))
    if not values or members is None:
        raise RuntimeError(f"No diffusion forward-only timing lines found in {path}")
    mean = sum(values) / len(values)
    std = 0.0 if len(values) == 1 else (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return {
        "n_members": members,
        "seconds_mean": mean,
        "seconds_std": std,
        "measured_event_count": len(values),
        "cuda_device_name": "A100-class GPU from source Slurm log",
        "source": str(path),
    }


def _method_key_from_payload(path: Path, payload: Mapping[str, object]) -> Optional[str]:
    raw = str(payload.get("method", ""))
    if raw in METHOD_NAME:
        return METHOD_NAME[raw]
    name = path.name
    for key, display in METHOD_NAME.items():
        if key in name:
            return display
    return None


def _load_grid_results(root: Path) -> Dict[str, Dict[int, Dict[str, object]]]:
    measured: Dict[str, Dict[int, Dict[str, object]]] = {display: {} for display in METHOD_NAME.values()}
    for n_dir in sorted(root.glob("n[0-9][0-9]")):
        try:
            n_members = int(n_dir.name[1:])
        except ValueError:
            continue
        results_dir = n_dir / "results"
        if not results_dir.exists():
            continue
        for path in sorted(results_dir.glob("forward_only_*.json")):
            payload = _read_json(path)
            method = _method_key_from_payload(path, payload)
            if method is None:
                continue
            row = _load_operator_result(path)
            if int(row["n_members"]) != n_members:
                raise ValueError(f"{path}: directory size n{n_members:02d} differs from JSON ensemble_members={row['n_members']}")
            measured[method][n_members] = row
    return {method: rows for method, rows in measured.items() if rows}


def _linear_estimate(rows: Mapping[int, Mapping[str, object]], n: int) -> Dict[str, object]:
    xs = sorted(rows)
    if not xs:
        raise ValueError("Cannot estimate timing without measured rows")
    if len(xs) == 1:
        measured_n = xs[0]
        base = rows[measured_n]
        scale = float(n) / float(measured_n)
        return {
            "seconds_per_event_rollout_mean": float(base["seconds_mean"]) * scale,
            "seconds_per_event_rollout_std": float(base.get("seconds_std", 0.0)) * scale,
            "model": "linear_per_member_forward_only_cost_single_anchor",
        }
    import numpy as np

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray([float(rows[i]["seconds_mean"]) for i in xs], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = max(0.0, float(intercept + slope * float(n)))
    std_per_member = [float(rows[i].get("seconds_std", 0.0)) / max(float(i), 1.0) for i in xs]
    return {
        "seconds_per_event_rollout_mean": pred,
        "seconds_per_event_rollout_std": float(sum(std_per_member) / len(std_per_member) * float(n)),
        "model": "linear_fit_forward_only_cost",
        "linear_fit_intercept": float(intercept),
        "linear_fit_slope": float(slope),
    }


def _sizes_payload(measured_rows: Mapping[int, Mapping[str, object]], sizes: Iterable[int]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for raw_n in sizes:
        n = int(raw_n)
        if n in measured_rows:
            row = measured_rows[n]
            out[str(n)] = {
                "n_members": n,
                "seconds_per_event_rollout_mean": float(row["seconds_mean"]),
                "seconds_per_event_rollout_std": float(row.get("seconds_std", 0.0)),
                "estimated": False,
                "model": "measured_forward_only_cost",
                "source": str(row.get("source", "")),
            }
        else:
            est = _linear_estimate(measured_rows, n)
            out[str(n)] = {
                "n_members": n,
                "estimated": True,
                "source": "linear_fit_from_measured_forward_only_grid",
                **est,
            }
    return out


def _load_flat_results(root: Path) -> Dict[str, Dict[int, Dict[str, object]]]:
    measured: Dict[str, Dict[int, Dict[str, object]]] = {}
    results_dir = root / "results"
    if not results_dir.exists():
        return measured
    fgno_path = results_dir / "forward_only_fgno.json"
    mc_path = results_dir / "forward_only_mc_dropout.json"
    if fgno_path.exists():
        row = _load_operator_result(fgno_path)
        measured.setdefault("FGNO", {})[int(row["n_members"])] = row
    if mc_path.exists():
        row = _load_operator_result(mc_path)
        measured.setdefault("MC-dropout GINO", {})[int(row["n_members"])] = row
    diffusion_json = results_dir / "forward_only_diffusion.json"
    if diffusion_json.exists():
        row = _load_operator_result(diffusion_json)
        measured.setdefault("Diffusion", {})[int(row["n_members"])] = row
    elif (root / "logs" / "diffusion_eval.log").exists():
        row = _load_diffusion_from_log(root / "logs" / "diffusion_eval.log")
        measured.setdefault("Diffusion", {})[int(row["n_members"])] = row
    return measured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-root", default=str(DEFAULT_FORWARD_ROOT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.forward_root).expanduser().resolve()
    measured = _load_grid_results(root)
    if not measured:
        measured = _load_flat_results(root)
    missing_methods = [method for method in METHOD_NAME.values() if method not in measured]
    if missing_methods:
        raise FileNotFoundError(f"Missing timing results for methods {missing_methods} under {root}")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_forward_only_root": str(root),
        "timing_policy": (
            "Measured forward-only wall-clock seconds per event rollout ensemble when available. "
            "Missing requested sizes are linear extrapolations from the measured grid and marked estimated=true."
        ),
        "methods": {},
    }
    for method in METHOD_NAME.values():
        rows = measured[method]
        first = rows[sorted(rows)[0]]
        payload["methods"][method] = {
            "measured_event_count": int(first.get("measured_event_count", 0)),
            "measured_sizes": sorted(int(x) for x in rows),
            "cuda_device_name": str(first.get("cuda_device_name", "")),
            "sizes": _sizes_payload(rows, args.sizes),
        }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
