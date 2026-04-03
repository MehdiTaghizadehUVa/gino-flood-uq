"""
Evaluate single-step prediction on TRAIN or TEST set using a saved checkpoint.

Uses the exact same train/test split as training (90/10, same seed) so no leakage.
Normalizers are loaded from disk (fit on train only during training).

  python scripts/flood_wv_eval_single_step.py --split train
  python scripts/flood_wv_eval_single_step.py --split test --checkpoint.save_dir ./checkpoints_TargetOnly_FGN

FGN: random z. --n_ensemble 2 and --normalized_space to match training log.
"""
import argparse
import sys
from pathlib import Path

import torch

# Repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Eval single-step on training set")
    parser.add_argument("--checkpoint.save_dir", type=str, default=None, dest="save_dir",
                        help="Checkpoint dir (default: from config)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, dest="checkpoint_dir",
                        help="Legacy alias for --checkpoint.save_dir")
    parser.add_argument("--data-root", type=str, default=None, dest="data_root",
                        help="Legacy override for config.data.root")
    parser.add_argument("--normalizer-path", type=str, default=None, dest="normalizer_path_override",
                        help="Legacy override for config.data.normalizer_path")
    parser.add_argument("--normalizer-root", type=str, default=None, dest="normalizer_root_override",
                        help="Legacy override for config.data.normalizer_root")
    parser.add_argument("--n-samples-max", type=int, default=None, dest="n_samples_max_override",
                        help="Legacy override for config.data.n_samples_max")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size_override",
                        help="Legacy override for config.data.batch_size")
    parser.add_argument("--data.n_samples_max", type=int, default=None, dest="n_samples_max",
                        help="Max train samples to evaluate (default: all)")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Max batches to run (default 200)")
    parser.add_argument("--n_ensemble", type=int, default=1,
                        help="FGN: number of random z forwards to average (default 1; use 2 to match training/test)")
    parser.add_argument("--normalized_space", action="store_true",
                        help="Report error in normalized space (no inverse transform; default: physical space)")
    parser.add_argument("--split", type=str, default="train", choices=("train", "test"),
                        help="Evaluate on 'train' or 'test' (same 90/10 split as training, no leakage)")
    args, _ = parser.parse_known_args()
    # Remove script-only args from sys.argv so load_config_and_setup (ArgparseConfig) does not see them
    for flag in [
        "--checkpoint_dir",
        "--data-root",
        "--normalizer-path",
        "--normalizer-root",
        "--n-samples-max",
        "--batch-size",
        "--max_batches",
        "--n_ensemble",
        "--normalized_space",
        "--split",
    ]:
        if flag in sys.argv:
            i = sys.argv.index(flag)
            del sys.argv[i]
            if i < len(sys.argv) and not sys.argv[i].startswith("-"):
                del sys.argv[i]

    # Load config and setup (same as training)
    from neuralop.flood.utils.runtime import (
        get_dataset_boundary_kwargs,
        load_config_and_setup,
        make_split_generator,
        parse_target_variables,
        write_train_txt_from_data_root,
    )
    from neuralop.flood.data.wv import FloodDatasetHDF, NormalizedDatasetOnTheFly, fit_normalizers_streaming
    from neuralop.flood.processing.wv import FloodGINODataProcessor
    from torch.utils.data import DataLoader, random_split, Subset
    from neuralop import get_model
    from neuralop.data.transforms.normalizers import load_normalizers
    from neuralop.training.training_state import load_training_state
    from neuralop.losses.data_losses import LpLoss
    from neuralop.losses.probabilistic_losses import CRPSLoss, split_gaussian_packed

    # Config (load_config_and_setup reads default WV config; override via CLI)
    config, device, _ = load_config_and_setup()
    if isinstance(device, str) and "cuda" in device and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    if args.data_root is not None:
        config.data.root = args.data_root
    if args.normalizer_path_override is not None:
        config.data.normalizer_path = args.normalizer_path_override
    if args.normalizer_root_override is not None:
        config.data.normalizer_root = args.normalizer_root_override
    if args.n_samples_max_override is not None:
        config.data.n_samples_max = int(args.n_samples_max_override)
    if args.batch_size_override is not None:
        config.data.batch_size = int(args.batch_size_override)

    save_dir = args.save_dir or args.checkpoint_dir or getattr(config.checkpoint, "save_dir", ".")
    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        save_dir = _REPO_ROOT / save_dir
    if not save_dir.exists():
        print(f"ERROR: Checkpoint dir not found: {save_dir}")
        return 1

    # Which checkpoint
    if (save_dir / "best_model_state_dict.pt").exists():
        save_name = "best_model"
    elif (save_dir / "model_state_dict.pt").exists():
        save_name = "model"
    else:
        print(f"ERROR: No model_state_dict.pt or best_model_state_dict.pt in {save_dir}")
        return 1

    # Data (train only)
    skip_before_timestep = getattr(config.data, "skip_before_timestep", 0)
    static_text_files = getattr(
        config.data, "static_text_files",
        ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"],
    )
    if getattr(config.data, "write_train_txt", False):
        write_train_txt_from_data_root(
            config.data.root,
            train_txt=getattr(config.data, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )
    ar_rollout_steps = max(1, int(getattr(config.opt, "ar_rollout_steps", 1)))
    n_history = config.data.n_history
    target_variables = parse_target_variables(getattr(config.data, "target_variables", ["wd", "vx", "vy"]))
    n_target_channels = len(target_variables)
    n_static = 2 + len(static_text_files)
    data_channels = n_static + n_history * 1 + n_history * n_target_channels
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target_channels)

    full_dataset = FloodDatasetHDF(
        data_root=config.data.root,
        n_history=config.data.n_history,
        query_res=getattr(config.data, "query_res", [48, 48]),
        run_ids=None,
        train_txt=getattr(config.data, "train_txt", "train.txt"),
        static_text_files=static_text_files,
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=skip_before_timestep,
        noise_type=getattr(config.data, "noise_type", "none"),
        noise_std=getattr(config.data, "noise_std", None),
        ar_rollout_steps=ar_rollout_steps,
        target_variables=target_variables,
        **get_dataset_boundary_kwargs(config.data),
    )
    # Same as training: optional n_samples_max then 90/10 split with fixed seed (no leakage)
    n_samples_max = getattr(config.data, "n_samples_max", None)
    if n_samples_max is not None and int(n_samples_max) > 0:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))
    total_len = len(full_dataset)
    seed = getattr(config.distributed, "seed", 123)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, test_data_raw_temp = random_split(
        full_dataset, [train_sz, test_sz], generator=make_split_generator(seed)
    )
    if args.split == "test":
        eval_dataset = test_data_raw_temp
        eval_size = test_sz
    else:
        eval_dataset = train_data_raw
        eval_size = train_sz
    print(f"Split: {args.split} (same 90/10 as training, seed={seed}) | train_sz={train_sz}, test_sz={test_sz}, eval_samples={eval_size}")

    normalizer_path = getattr(config.data, "normalizer_path", None)
    if normalizer_path is not None:
        normalizer_path = Path(normalizer_path)
        if not normalizer_path.is_absolute():
            normalizer_root = getattr(config.data, "normalizer_root", None)
            if normalizer_root is None:
                normalizer_root = getattr(config.data, "train_root", None)
            if normalizer_root is None:
                print(
                    "ERROR: Relative normalizer_path requires data.normalizer_root "
                    "(or data.train_root). Refusing to resolve against data.root."
                )
                return 1
            normalizer_path = Path(normalizer_root) / normalizer_path
    if normalizer_path is None or not normalizer_path.exists():
        print("ERROR: normalizer_path not set or file missing; need pre-fit normalizers.")
        return 1
    normalizers = load_normalizers(normalizer_path, device=None)
    # Move target normalizer to device for postprocess (inverse transform)
    if normalizers.get("target") is not None:
        normalizers["target"] = normalizers["target"].to(device)

    eval_normalized = NormalizedDatasetOnTheFly(
        eval_dataset, normalizers, query_res=config.data.query_res
    )
    eval_loader = DataLoader(
        eval_normalized,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = get_model(config)
    load_training_state(save_dir=save_dir, save_name=save_name, model=model)
    model = model.to(device)
    model.eval()

    # When --normalized_space: no inverse transform so L2 is in normalized space
    inverse_test = not args.normalized_space and getattr(config, "inverse_test", True)
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target", None),
        inverse_test=inverse_test,
        output_distribution=str(
            getattr(config.gino, "output_distribution", "deterministic")
        ).strip().lower(),
    )
    data_processor.eval()

    use_fgn = getattr(config.gino, "use_fgn_noise", False)
    output_distribution = str(
        getattr(config.gino, "output_distribution", "deterministic")
    ).strip().lower()
    fgn_noise_dim = getattr(config.gino, "fgn_noise_dim", 32) if use_fgn else None
    n_ensemble = max(1, int(args.n_ensemble))
    l2loss = LpLoss(d=2, p=2)
    crps_loss_fn = None
    if n_ensemble >= 2:
        crps_channel_weights = getattr(config.opt, "crps_channel_weights", None)
        crps_loss_fn = CRPSLoss(n_samples=n_ensemble, channel_weights=crps_channel_weights, reduction="mean")

    total_rel_l2_sum = 0.0
    total_crps_weighted_sum = 0.0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            if idx >= args.max_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sample = data_processor.preprocess(batch)
            B = sample["y"].shape[0]

            if use_fgn:
                preds = []
                y_eval = sample["y"]
                for _ in range(n_ensemble):
                    z = torch.randn(B, fgn_noise_dim, device=device, dtype=sample["x"].dtype)
                    samp = {**sample, "ada_in": z}
                    out = model(**samp)
                    if data_processor is not None:
                        # Keep y handling identical to trainer eval path: avoid repeatedly
                        # inverse-transforming the same y tensor across ensemble members.
                        out, sample_post = data_processor.postprocess(
                            out,
                            {
                                "y": sample["y"],
                                "structural_dry_mask": sample.get("structural_dry_mask"),
                            },
                        )
                        y_eval = sample_post["y"]
                    preds.append(out)
                pred_samples = torch.stack(preds, dim=0)
                pred = pred_samples.mean(dim=0)
            else:
                pred = model(**sample)
                if data_processor is not None:
                    pred, sample_post = data_processor.postprocess(pred, {"y": sample["y"]})
                    y_eval = sample_post["y"]
                else:
                    y_eval = sample["y"]
                pred_samples = None
                if output_distribution == "gaussian":
                    pred, _ = split_gaussian_packed(pred, y_eval.shape[-1])

            # Match trainer evaluation aggregation:
            # - l2loss (reduction='sum') contributes directly then divide by n_samples.
            rel_l2_batch = l2loss(pred, y_eval)
            total_rel_l2_sum += rel_l2_batch.item() if hasattr(rel_l2_batch, "item") else float(rel_l2_batch)
            if crps_loss_fn is not None and pred_samples is not None:
                # CRPSLoss here uses reduction='mean': weight by batch size before final /n_samples,
                # same as Trainer.evaluate().
                crps_batch = crps_loss_fn(pred_samples, y_eval)
                crps_val = crps_batch.item() if hasattr(crps_batch, "item") else float(crps_batch)
                total_crps_weighted_sum += crps_val * B
            total_samples += B
            n_batches += 1

    if total_samples == 0:
        print("No samples evaluated.")
        return 1

    # Exact same formulas as training log aggregation:
    # train_err = sum(rel_l2_batch)/n_samples
    # avg_loss  = sum(batch_mean_crps * batch_size)/n_samples
    train_err = total_rel_l2_sum / total_samples
    space_label = "normalized" if args.normalized_space else "physical"
    print(f"Single-step on {args.split.upper()}: {total_samples} samples ({n_batches} batches), n_ensemble={n_ensemble}")
    print(f"  train_err = {train_err:.6f}  (rel L2, {space_label} space)")
    if crps_loss_fn is not None:
        avg_loss = total_crps_weighted_sum / total_samples
        print(f"  avg_loss  = {avg_loss:.6f}  (CRPS)")
    # Comparison to training log
    print("  --- Comparison to training log ---")
    print("  Training reports: train_err and avg_loss in NORMALIZED space (no inverse transform).")
    if args.normalized_space:
        print("  This run is in normalized space => directly comparable to training log.")
    else:
        print("  This run is in physical space => use --normalized_space to match training log.")
    if use_fgn and n_ensemble == 1 and crps_loss_fn is None:
        print("  Training uses CRPS with crps_n_samples (e.g. 2) => use --n_ensemble 2 to get avg_loss.")

    # Simple sanity: if model has learned train set at all, train_err should be in a reasonable range
    if train_err < 0.5:
        print("  -> PASS: very low train error (model fits training set well).")
    elif train_err < 2.0:
        print("  -> OK: moderate train error (model is learning training set).")
    else:
        print("  -> WARN: high train error (model may not be fitting training set well).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
