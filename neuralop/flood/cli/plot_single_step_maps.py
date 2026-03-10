"""
Generate 2x2 comparison maps: single-step prediction vs ground truth in normalized and physical space.

Uses one training sample, same split and checkpoint as training. Produces one figure per channel
(water depth, vx, vy): [Pred Normalized | Pred Physical; GT Normalized | GT Physical].

  python scripts/flood_wv_plot_single_step_maps.py
  python scripts/flood_wv_plot_single_step_maps.py --target_timestep 20 --out_dir ./maps
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _scatter_map(ax, x, y, data, title, cmap="viridis", vmin=None, vmax=None, cblabel=""):
    if vmin is None:
        vmin = np.nanmin(data) if np.any(np.isfinite(data)) else 0
    if vmax is None:
        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1
    if vmax <= vmin:
        vmax = vmin + 1e-6
    sc = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, s=6, marker="s", linewidths=0, rasterized=True)
    ax.set_title(title, pad=8, fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    if cblabel:
        cbar.set_label(cblabel, labelpad=10, fontsize=10)
    return sc


def main():
    parser = argparse.ArgumentParser(description="Plot single-step pred vs GT in normalized and physical space")
    parser.add_argument("--checkpoint.save_dir", type=str, default=None, dest="save_dir", help="Checkpoint dir")
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
    parser.add_argument("--out_dir", type=str, default="./single_step_maps", help="Output directory for figures")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of training sample (within first batch)")
    parser.add_argument("--target_timestep", type=int, default=None, help="Use a sample whose target is at this timestep (e.g. 20 for t=20)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for FGN z (reproducibility)")
    args, _ = parser.parse_known_args()
    for flag in [
        "--checkpoint_dir",
        "--data-root",
        "--normalizer-path",
        "--normalizer-root",
        "--n-samples-max",
        "--batch-size",
        "--out_dir",
        "--sample_index",
        "--target_timestep",
        "--seed",
    ]:
        if flag in sys.argv:
            i = sys.argv.index(flag)
            del sys.argv[i]
            if i < len(sys.argv) and not sys.argv[i].startswith("-"):
                del sys.argv[i]

    from neuralop.flood.utils.runtime import (
        get_dataset_boundary_kwargs,
        load_config_and_setup,
        make_split_generator,
        parse_target_variables,
        write_train_txt_from_data_root,
    )
    from neuralop.flood.data.wv import FloodDatasetHDF, NormalizedDatasetOnTheFly
    from neuralop.flood.processing.wv import FloodGINODataProcessor
    from torch.utils.data import DataLoader, random_split, Subset
    from neuralop import get_model
    from neuralop.data.transforms.normalizers import load_normalizers
    from neuralop.losses.probabilistic_losses import split_gaussian_packed
    from neuralop.training.training_state import load_training_state

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

    save_name = "best_model" if (save_dir / "best_model_state_dict.pt").exists() else "model"
    if not (save_dir / f"{save_name}_state_dict.pt").exists():
        print(f"ERROR: No checkpoint in {save_dir}")
        return 1

    # Data: same as eval script (train split)
    static_text_files = getattr(config.data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"])
    target_variables = parse_target_variables(getattr(config.data, "target_variables", ["wd", "vx", "vy"]))
    n_target_channels = len(target_variables)
    if getattr(config.data, "write_train_txt", False):
        write_train_txt_from_data_root(
            config.data.root,
            train_txt=getattr(config.data, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )
    skip_before_timestep = getattr(config.data, "skip_before_timestep", 0)
    ar_rollout_steps = max(1, int(getattr(config.opt, "ar_rollout_steps", 1)))
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
    n_samples_max = getattr(config.data, "n_samples_max", None)
    if n_samples_max is not None and int(n_samples_max) > 0:
        n_use = min(int(n_samples_max), len(full_dataset))
        full_dataset = Subset(full_dataset, range(n_use))
    total_len = len(full_dataset)
    seed_split = getattr(config.distributed, "seed", 123)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, _ = random_split(
        full_dataset, [train_sz, test_sz], generator=make_split_generator(seed_split)
    )
    normalizer_path = getattr(config.data, "normalizer_path", None)
    if normalizer_path is None:
        print("ERROR: normalizer_path not set.")
        return 1
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
    if not normalizer_path.exists():
        print("ERROR: normalizers file not found.")
        return 1
    normalizers = load_normalizers(normalizer_path, device=None)
    if normalizers.get("target") is not None:
        normalizers["target"] = normalizers["target"].to(device)
    train_normalized = NormalizedDatasetOnTheFly(
        train_data_raw, normalizers, query_res=config.data.query_res
    )
    # Select sample: by target timestep (e.g. t=20) or by sample_index in first batch
    if getattr(args, "target_timestep", None) is not None:
        target_timestep = int(args.target_timestep)
        full_ds = train_data_raw.dataset
        found = None
        for pos in range(len(train_data_raw)):
            global_idx = train_data_raw.indices[pos]
            run_id, t = full_ds.sample_index[global_idx]
            if t == target_timestep:
                found = pos
                break
        if found is None:
            print(f"WARNING: No train sample with target_timestep={target_timestep}; using first sample.")
            sample_idx = 0
        else:
            sample_idx = found
            print(f"Using train sample with target at timestep t={target_timestep} (run_id={full_ds.sample_index[train_data_raw.indices[sample_idx]][0]})")
    else:
        sample_idx = min(getattr(args, "sample_index", 0), len(train_data_raw) - 1)

    # Load single sample and make batch of size 1
    sample_single = train_normalized[sample_idx]
    batch = {}
    for k, v in sample_single.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v
    target_key = "y" if "y" in batch else "target"
    y_batch = batch[target_key]
    # Processor without inverse so we get normalized outputs
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target", None),
        inverse_test=False,
        output_distribution=str(
            getattr(config.gino, "output_distribution", "deterministic")
        ).strip().lower(),
    )
    data_processor.eval()
    sample = data_processor.preprocess(batch)
    gt_norm = sample["y"].detach().clone()
    # Match training: set data_channels so model matches checkpoint (yaml may have wrong value)
    n_history = config.data.n_history
    n_static = 2 + len(static_text_files)
    data_channels = n_static + n_history * 1 + n_history * n_target_channels
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target_channels)
    # Model forward (single z for FGN for reproducibility)
    model = get_model(config)
    load_training_state(save_dir=save_dir, save_name=save_name, model=model)
    model = model.to(device)
    model.eval()
    use_fgn = getattr(config.gino, "use_fgn_noise", False)
    output_distribution = str(
        getattr(config.gino, "output_distribution", "deterministic")
    ).strip().lower()
    fgn_noise_dim = getattr(config.gino, "fgn_noise_dim", 32) if use_fgn else None
    with torch.no_grad():
        if use_fgn and fgn_noise_dim is not None:
            g = torch.Generator(device=device).manual_seed(args.seed)
            z = torch.randn(
                sample["y"].shape[0],
                fgn_noise_dim,
                device=device,
                dtype=sample["x"].dtype,
                generator=g,
            )
            samp = {**sample, "ada_in": z}
            pred_norm = model(**samp)
        else:
            pred_norm = model(**sample)
    if output_distribution == "gaussian":
        pred_norm, _ = split_gaussian_packed(pred_norm, gt_norm.shape[-1])
    # Physical = inverse transform
    target_norm = normalizers.get("target", None)
    if target_norm is not None:
        pred_phys = target_norm.inverse_transform(pred_norm.detach())
        gt_phys = target_norm.inverse_transform(gt_norm)
    else:
        pred_phys = pred_norm.detach()
        gt_phys = gt_norm

    # Take the single sample (batch size 1)
    idx = 0
    pred_norm = pred_norm[idx].cpu().numpy()
    pred_phys = pred_phys[idx].cpu().numpy()
    gt_norm = gt_norm[idx].cpu().numpy()
    gt_phys = gt_phys[idx].cpu().numpy()
    # geometry: (B, n_cells, 2) or (n_cells, 2)
    geom = sample["geometry"]
    if geom.dim() == 3:
        geom = geom[idx]
    geom = geom.cpu().numpy()
    x, y = geom[:, 0], geom[:, 1]

    # Channels follow config.data.target_variables order.
    channel_meta = {
        "wd": ("water_depth", "Water depth", "viridis", "m", " (norm)"),
        "vx": ("vx", "VX velocity", "coolwarm", "m/s", " (norm)"),
        "vy": ("vy", "VY velocity", "coolwarm", "m/s", " (norm)"),
    }
    channel_info = [channel_meta[v] for v in target_variables]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ch, (name, title, cmap, unit, norm_suffix) in enumerate(channel_info):
        pn = pred_norm[:, ch]
        pp = pred_phys[:, ch]
        gn = gt_norm[:, ch]
        gp = gt_phys[:, ch]
        if ch == 0:
            vmin_phys = 0
            vmax_phys = max(np.nanmax(gp), np.nanmax(pp), 1e-6)
        else:
            vabs = max(np.nanmax(np.abs(gp)), np.nanmax(np.abs(pp)), 1e-9)
            vmin_phys = -vabs
            vmax_phys = vabs
        vmin_norm = min(np.nanmin(gn), np.nanmin(pn)) if np.any(np.isfinite(gn)) else -1
        vmax_norm = max(np.nanmax(gn), np.nanmax(pn)) if np.any(np.isfinite(gn)) else 1

        fig, axs = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
        _scatter_map(axs[0, 0], x, y, pn, f"Prediction – Normalized", cmap=cmap, vmin=vmin_norm, vmax=vmax_norm, cblabel=title + norm_suffix)
        _scatter_map(axs[0, 1], x, y, pp, f"Prediction – Physical", cmap=cmap, vmin=vmin_phys, vmax=vmax_phys, cblabel=f"{title} ({unit})")
        _scatter_map(axs[1, 0], x, y, gn, f"Ground truth – Normalized", cmap=cmap, vmin=vmin_norm, vmax=vmax_norm, cblabel=title + norm_suffix)
        _scatter_map(axs[1, 1], x, y, gp, f"Ground truth – Physical", cmap=cmap, vmin=vmin_phys, vmax=vmax_phys, cblabel=f"{title} ({unit})")
        tt = getattr(args, "target_timestep", None)
        sub = f", target t={tt}" if tt is not None else f", index={sample_idx}"
        fig.suptitle(f"Single-step: {title} (train sample{sub})", fontsize=14)
        fpath = out_dir / f"single_step_{name}_norm_phys.png"
        fig.savefig(fpath, bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close(fig)
        print(f"Saved: {fpath}")

    print(f"Done. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
