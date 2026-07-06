"""Deep-ensemble cross-check of the NEON Stage-2 epistemic map.

For held-out TR validation families, compares:
  - NEON epistemic map: ANOVA-corrected variance of the nested prediction
    (tr_n450 EpiNet over M=32 z_e, K=50 cached frozen rollouts), vs
  - Deep-ensemble epistemic map: Var over the 3 independently trained FGNs in
    the serving bundle of their K=20 aleatory-mean rollouts (matches the
    operational 3x20 configuration).

Reports per-family and aggregate: spatial correlation, top-10% high-variance
region overlap, variance ratio (compare_epistemic_maps), plus the spatial
correlation of each epistemic map with |ensemble-mean error| — the discriminator
for whether the negative NEON epistemic-error correlation is a property of the
problem or of the method.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("neon_de_compare")

BASE = Path("/scratch/jrj6wm/GINO_Model/neon_stage2_full_train")
TR_CONFIG = str(BASE / "config/coast_fgn_neon_tr450.yaml")
BUNDLE = ("/scratch/jrj6wm/GINO_Model/model_bundles/"
          "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json")
CKPT = str(BASE / "tr_n450/neon_stage2_best.pt")
OUT = BASE / "de_compare"
N_FAMILIES = 20
M_EVAL, K_NEON, K_DE = 32, 50, 20


def _rowcorr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).to(torch.float64)
    y = y.reshape(-1).to(torch.float64)
    xm, ym = x - x.mean(), y - y.mean()
    d = torch.sqrt(xm.pow(2).sum() * ym.pow(2).sum()).clamp_min(1e-12)
    return float((xm * ym).sum() / d)


def main() -> int:
    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.eval.datasets import _load_or_fit_normalizers
    from neuralop.flood.eval.neon import compare_epistemic_maps, deep_ensemble_epistemic_variance
    from neuralop.flood.neon import (
        anova_corrected_epistemic_variance,
        load_neon_stage2_checkpoint,
        sample_epistemic_indices,
    )
    from neuralop.flood.train.neon import neon_stage2_eval_forward
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import (
        make_cached_feature_collector,
        make_feature_collector_from_frozen_model,
    )
    from neuralop.flood.utils.runtime_core import load_config_and_setup, parse_target_variables

    OUT.mkdir(parents=True, exist_ok=True)
    saved = list(sys.argv)
    try:
        sys.argv = ["de_compare", "--config_path", TR_CONFIG]
        cfg, _d, _l = load_config_and_setup()
    finally:
        sys.argv = saved
    tvars = parse_target_variables(getattr(cfg.data, "target_variables", ["wd"]))
    normalizers, _ = _load_or_fit_normalizers(cfg, None, None, LOG)
    _train, val_fam = build_families_from_config(cfg, normalizers, tvars, LOG, val_fraction=0.1)
    _train = None
    families = sorted(val_fam, key=lambda f: f.family_id)[:N_FAMILIES]
    LOG.info("families: %d", len(families))

    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle       # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared   # type: ignore[attr-defined]
    models = prepared["models"]
    LOG.info("deep-ensemble members J=%d", len(models))
    assert len(models) >= 2, "need >=2 independently trained models for the DE map"

    module, meta = load_neon_stage2_checkpoint(CKPT, map_location="cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()

    neon_coll = make_cached_feature_collector(
        make_feature_collector_from_frozen_model(
            stage1, feature_source=meta.get("feature_source", "decoder_pre_projection"),
            n_history=int(bundle.n_history), latent_dim=int(bundle.fgn_noise_dim),
            generator=torch.Generator().manual_seed(0),
        ),
        cache_dir=BASE / "feature_cache_tr_k50_eval",
    )
    de_colls = [
        make_feature_collector_from_frozen_model(
            m, feature_source="decoder_pre_projection",
            n_history=int(bundle.n_history), latent_dim=int(bundle.fgn_noise_dim),
            generator=torch.Generator().manual_seed(100 + j),
        )
        for j, m in enumerate(models)
    ]

    z_gen = torch.Generator().manual_seed(1)
    rows = []
    for f_idx, fam in enumerate(families):
        batch = neon_coll(fam, num_aleatory=K_NEON)
        z_e = sample_epistemic_indices(M_EVAL, module.epistemic_dim,
                                       device=batch.features.device,
                                       dtype=batch.features.dtype, generator=z_gen)
        parts = []
        for m in range(M_EVAL):
            for ks in range(0, K_NEON, 16):
                parts.append(neon_stage2_eval_forward(
                    module=module,
                    base_prediction=batch.base_prediction[:, ks:ks + 16],
                    features=batch.features[:, ks:ks + 16],
                    z_e=z_e[m:m + 1],
                ).detach().to("cpu", torch.float32))
        ncols = (K_NEON + 15) // 16
        pred = torch.cat([torch.cat(parts[i * ncols:(i + 1) * ncols], dim=2)
                          for i in range(M_EVAL)], dim=1)
        neon_epi = anova_corrected_epistemic_variance(pred)          # [1,T,Nv,C]
        del batch, parts

        model_means, base_all = [], []
        for coll in de_colls:
            b = coll(fam, num_aleatory=K_DE)
            bp = b.base_prediction.detach().to("cpu", torch.float32)  # [1,K,T,Nv,C]
            model_means.append(bp.mean(dim=1))
            base_all.append(bp)
            del b
        de_epi = deep_ensemble_epistemic_variance(torch.stack(model_means, dim=1))
        ref_mean = fam.reference.unsqueeze(0).mean(dim=1)
        err_all = (torch.cat(base_all, dim=1).mean(dim=1) - ref_mean).abs()
        del base_all

        cm = compare_epistemic_maps(neon_epi, de_epi, top_q=0.10)
        row = {
            "family_id": fam.family_id,
            **cm,
            "neon_epi_err_corr": _rowcorr(neon_epi, err_all),
            "de_epi_err_corr": _rowcorr(de_epi, err_all),
            "neon_epi_mean": float(neon_epi.mean()),
            "de_epi_mean": float(de_epi.mean()),
        }
        rows.append(row)
        LOG.info("family %s done (%d/%d): corr=%.3f overlap=%.3f de_err_corr=%.3f neon_err_corr=%.3f",
                 fam.family_id, f_idx + 1, len(families),
                 cm["spatial_corr"], cm["topq_overlap"],
                 row["de_epi_err_corr"], row["neon_epi_err_corr"])

    agg = {k: float(torch.tensor([r[k] for r in rows]).mean())
           for k in rows[0] if k != "family_id"}
    payload = {"n_families": len(rows), "m_eval": M_EVAL, "k_neon": K_NEON,
               "k_de": K_DE, "j_models": len(models), "aggregate": agg, "per_family": rows}
    with (OUT / "deep_ensemble_comparison.json").open("w") as fh:
        json.dump(payload, fh, indent=2)
    LOG.info("aggregate: %s", json.dumps(agg, sort_keys=True))
    print("DE COMPARE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
