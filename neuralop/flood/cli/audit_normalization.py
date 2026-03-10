#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rigorous audit for normalization/denormalization and split leakage.

Checks performed:
1) Train/test split integrity (disjoint indices, no duplicates)
2) Normalizer source (load from file if present, else fit on train split only)
3) Round-trip consistency on BOTH train and test:
      inverse_transform(transform(x)) ~= x
   for keys: geometry, static, boundary, target, dynamic
4) Direct formula consistency:
      z == (x - mean) / (std + eps)
5) DataProcessor inverse behavior:
   - eval + inverse_test=True: outputs and y are inverse-transformed
   - train mode: no inverse transform

Run:
  python -m neuralop.flood.cli.audit_normalization
  python -m neuralop.flood.cli.audit_normalization --config_path config/flood/wv/gino_pluvial_flood_config_WV_depth_only.yaml
  python -m neuralop.flood.cli.audit_normalization --checks_per_split 512
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset, random_split

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neuralop.flood.data.wv import (  # noqa: E402
    FloodDatasetHDF,
    NormalizedDatasetOnTheFly,
    fit_normalizers_streaming,
)
from neuralop.flood.processing.wv import FloodGINODataProcessor  # noqa: E402
from neuralop.flood.utils.runtime import (  # noqa: E402
    get_dataset_boundary_kwargs,
    load_config_and_setup,
    make_split_generator,
    parse_target_variables,
    set_seed,
    write_train_txt_from_data_root,
)
from neuralop.data.transforms.normalizers import load_normalizers  # noqa: E402


def _sample_positions(n, k, seed):
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    rng = random.Random(seed)
    return rng.sample(list(range(n)), k)


def _norm_for_key(normalizers, key):
    if key in normalizers and normalizers[key] is not None:
        return normalizers[key]
    if key == "dynamic" and "target" in normalizers and normalizers["target"] is not None:
        return normalizers["target"]
    return None


def _allclose(a, b, atol=1e-5, rtol=1e-4):
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def _run_roundtrip_checks(split_name, raw_split, norm_split, normalizers, positions, atol, rtol):
    keys = ["geometry", "static", "boundary", "target", "dynamic"]
    max_abs = {k: 0.0 for k in keys}
    n_checked = 0

    for pos in positions:
        raw_sample = raw_split[pos]
        norm_sample = norm_split[pos]
        n_checked += 1
        for key in keys:
            if key not in raw_sample or raw_sample[key] is None:
                continue
            if key not in norm_sample or norm_sample[key] is None:
                continue
            norm = _norm_for_key(normalizers, key)
            if norm is None:
                continue

            raw_val = raw_sample[key]
            z = norm_sample[key]
            denorm = norm.inverse_transform(z.unsqueeze(0)).squeeze(0)
            err = (denorm - raw_val).abs().max().item()
            max_abs[key] = max(max_abs[key], err)
            if not _allclose(denorm, raw_val, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"[{split_name}] round-trip failed for key='{key}' at pos={pos}, max_abs={err}"
                )

            # Formula check for a couple of keys
            if key in ("target", "static"):
                manual = (raw_val.unsqueeze(0) - norm.mean.to(raw_val.device)) / (
                    norm.std.to(raw_val.device) + norm.eps
                )
                manual = manual.squeeze(0)
                if not _allclose(manual, z, atol=1e-6, rtol=1e-5):
                    err2 = (manual - z).abs().max().item()
                    raise AssertionError(
                        f"[{split_name}] normalize formula mismatch for key='{key}' at pos={pos}, max_abs={err2}"
                    )

    return n_checked, max_abs


def main():
    parser = argparse.ArgumentParser(description="Audit normalization/denormalization and leakage.")
    parser.add_argument("--checks_per_split", type=int, default=512, help="Random samples to check per split")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for round-trip")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for round-trip")
    parser.add_argument(
        "--force_refit",
        action="store_true",
        help="Ignore normalizer_path file and force fitting normalizers on train split only.",
    )
    args, _ = parser.parse_known_args()
    for flag in ["--checks_per_split", "--atol", "--rtol", "--force_refit"]:
        if flag in sys.argv:
            i = sys.argv.index(flag)
            del sys.argv[i]
            if i < len(sys.argv) and not sys.argv[i].startswith("-"):
                del sys.argv[i]

    config, device, _ = load_config_and_setup()
    seed = getattr(config.distributed, "seed", 123)
    set_seed(seed, deterministic=getattr(config, "deterministic", True))

    skip_before_timestep = getattr(config.data, "skip_before_timestep", 0)
    noise_type = getattr(config.data, "noise_type", "none")
    noise_std = getattr(config.data, "noise_std", None)
    static_text_files = getattr(config.data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"])
    target_variables = parse_target_variables(getattr(config.data, "target_variables", ["wd", "vx", "vy"]))
    if getattr(config.data, "write_train_txt", False):
        write_train_txt_from_data_root(
            config.data.root,
            train_txt=getattr(config.data, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )

    ar_rollout_steps = max(1, int(getattr(config.opt, "ar_rollout_steps", 1)))
    full_dataset = FloodDatasetHDF(
        data_root=config.data.root,
        n_history=config.data.n_history,
        query_res=getattr(config.data, "query_res", [64, 64]),
        run_ids=None,
        train_txt=getattr(config.data, "train_txt", "train.txt"),
        static_text_files=static_text_files,
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=skip_before_timestep,
        noise_type=noise_type,
        noise_std=noise_std,
        ar_rollout_steps=ar_rollout_steps,
        target_variables=target_variables,
        **get_dataset_boundary_kwargs(config.data),
    )
    n_samples_max = getattr(config.data, "n_samples_max", None)
    if n_samples_max is not None:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))

    total_len = len(full_dataset)
    if total_len == 0:
        raise RuntimeError("No samples available.")
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_raw, test_raw = random_split(full_dataset, [train_sz, test_sz], generator=make_split_generator(seed))

    # Split leakage checks
    train_idx = list(train_raw.indices)
    test_idx = list(test_raw.indices)
    if len(set(train_idx)) != len(train_idx):
        raise AssertionError("Duplicate indices inside train split.")
    if len(set(test_idx)) != len(test_idx):
        raise AssertionError("Duplicate indices inside test split.")
    overlap = set(train_idx).intersection(set(test_idx))
    if overlap:
        raise AssertionError(f"Leakage: train/test overlap has {len(overlap)} indices.")

    # Load or fit normalizers exactly like training code path
    normalizer_path = getattr(config.data, "normalizer_path", None)
    if normalizer_path is not None:
        normalizer_path = Path(normalizer_path)
        if not normalizer_path.is_absolute():
            normalizer_path = Path(config.data.root) / normalizer_path
    loaded_from_file = (not args.force_refit) and normalizer_path is not None and normalizer_path.exists()
    if loaded_from_file:
        normalizers = load_normalizers(normalizer_path, device=None)
    else:
        normalizers = fit_normalizers_streaming(
            train_raw,
            chunk_size=getattr(config.data, "normalizer_chunk_size", 10000),
            expect_target=True,
        )

    train_norm = NormalizedDatasetOnTheFly(train_raw, normalizers, query_res=config.data.query_res)
    test_norm = NormalizedDatasetOnTheFly(test_raw, normalizers, query_res=config.data.query_res)

    # Round-trip checks on both splits
    train_pos = _sample_positions(len(train_raw), args.checks_per_split, seed + 11)
    test_pos = _sample_positions(len(test_raw), args.checks_per_split, seed + 29)
    n_train_checked, train_max_abs = _run_roundtrip_checks(
        "train", train_raw, train_norm, normalizers, train_pos, atol=args.atol, rtol=args.rtol
    )
    n_test_checked, test_max_abs = _run_roundtrip_checks(
        "test", test_raw, test_norm, normalizers, test_pos, atol=args.atol, rtol=args.rtol
    )

    # DataProcessor inverse behavior audit
    if normalizers.get("target") is not None:
        target_norm = normalizers["target"]
        dp = FloodGINODataProcessor(device=device, target_norm=target_norm, inverse_test=True)
        ref = test_norm[test_pos[0] if test_pos else 0]["target"].unsqueeze(0)

        # eval mode: should inverse-transform
        dp.eval()
        out_eval, sample_eval = dp.postprocess(ref.clone(), {"y": ref.clone()})
        inv_ref = target_norm.inverse_transform(ref.clone())
        if not _allclose(out_eval.cpu(), inv_ref.cpu(), atol=1e-5, rtol=1e-4):
            raise AssertionError("DataProcessor eval inverse on out failed.")
        if not _allclose(sample_eval["y"].cpu(), inv_ref.cpu(), atol=1e-5, rtol=1e-4):
            raise AssertionError("DataProcessor eval inverse on y failed.")

        # train mode: should not inverse-transform
        dp.train()
        out_train, sample_train = dp.postprocess(ref.clone(), {"y": ref.clone()})
        if not _allclose(out_train.cpu(), ref.cpu(), atol=1e-7, rtol=1e-6):
            raise AssertionError("DataProcessor train-mode unexpectedly inverse-transformed out.")
        if not _allclose(sample_train["y"].cpu(), ref.cpu(), atol=1e-7, rtol=1e-6):
            raise AssertionError("DataProcessor train-mode unexpectedly inverse-transformed y.")

    print("=== NORMALIZATION / LEAKAGE AUDIT PASSED ===")
    print(f"split_total={total_len}, train={len(train_raw)}, test={len(test_raw)}, overlap=0")
    print(f"target_variables={target_variables}")
    print(f"normalizers={'loaded_from_file' if loaded_from_file else 'fit_on_train_split_only'}")
    print(f"roundtrip_checked: train={n_train_checked}, test={n_test_checked}")
    print(f"max_abs_roundtrip_train={train_max_abs}")
    print(f"max_abs_roundtrip_test={test_max_abs}")
    print("DataProcessor inverse behavior: PASS (eval=inverse, train=no-inverse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
