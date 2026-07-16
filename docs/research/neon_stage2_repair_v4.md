# NEON Stage-2 Repair v4

## Implemented statistical contracts

- Crossed common-random-number ANOVA is the default estimator; the independent nested estimator remains explicit.
- Validation metrics and RMSE parity diagnostics are evaluated after inverse transformation to physical water-depth units.
- Probit-exponential family bootstrap weights break the former even-index parity; reference-member bootstrap is off by default.
- The continuous EpiNet uses a population-centered low-rank Hermite basis shared by trainable and frozen-prior branches.
- A deterministic mean head uses one hash-validated canonical antithetic FGNO latent bank, independent of runtime ensemble size.
- Selection uses a design-correct posterior-predictive fair CRPS subject to a 0.001 m model-0 RMSE non-inferiority margin. Retention is diagnostic only.
- A unit-aware deep-ensemble spread target is available as an explicit ablation and is not the default.
- The B5 control persists a fixed Dirichlet particle support, complete family ordering, split fingerprint, and particle-by-family weight matrix.
- Frozen rollouts clamp structural-dry cells before autoregressive feedback. Cache schema `neon_feature_cache_v3` records the dry policy and canonical bank.

## G0 result

The exact 50-family legacy remap completed under Slurm job `16932324`. On identical saved prediction tensors:

| Quantity | Independent nested estimator | Crossed CRN estimator |
|---|---:|---:|
| Mean epistemic variance (m2) | 1.6122e-6 | 4.9523e-6 |
| All-wettable std-error correlation | -0.2376 | 0.2548 |
| All-wettable depth-adjusted association | -0.0750 | 0.2061 |
| Wet-front depth-adjusted association | -0.0449 | 0.0951 |

The estimator caused a substantial portion of the apparent collapse, but wet-front association remains weak. Prior-scale rungs must therefore be treated as controlled ablations rather than assuming a tenfold increase.

Outputs: `/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/legacy_estimator_repair_20260712/tr_n450_exact_tensors`.

## Predictive-score sampling contract

The Stage-2 forecast is a factorial sample over epistemic indices and aleatory
latents. The same ordered aleatory bank is reused for every epistemic index to
reduce Monte Carlo noise, so flattening the `M x K` tensor and applying the
ordinary iid finite-ensemble correction is invalid. In particular, a collapsed
epistemic axis would duplicate each of the `K` Stage-1 members and receive a
different score despite representing the same forecast.

For sampled continuous epistemic indices, the fair self-distance therefore
uses only pairs with both `m != m'` and `k != k'`. For B5's complete persistent
particle support, the particle axis is integrated exactly, so equal-particle
pairs are retained while equal columns of the shared aleatory bank are
excluded. Independently nested aleatory banks use their corresponding
same-epistemic exclusion. Validation, G1 selection, and nested evaluation store
the selected sampling design in their provenance.

This correction first applies at commit `6631946`. Earlier partial B2 output at
`b2_n450_zero_init_8dc87f7` is explicitly invalidated and is not admissible in
the attribution table.

## Attribution execution

Use `scripts/sbatch_neon_repair_rung.sh` with `NEON_LADDER_RUNG=B0`, then `B1a`, `B1b`, `B2`, and `B3`. Do not run B4 or B5 until the G1 table confirms mixture-CRPS improvement and model-0 RMSE parity. B4 additionally requires `NEON_DE_SPREAD_MULTIPLIER` in `{0.5, 1.0, 2.0}`. B5 always branches from the B3 statistical policy.

The N-sweep script uses five seeded permutations with nested family prefixes for `N={25,50,100,250,400}`. No training job may launch from a dirty tree.
