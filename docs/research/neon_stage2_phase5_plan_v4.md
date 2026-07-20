# NEON Stage-2 — Phase 5 Plan v4 (final): Diagnose-then-Pilot (post-G1 stop-loss)

## Context

**v4-repair verified complete at `37633fe`.** B0→B3 finding: RMSE parity solved; estimator/selection exonerated; trainable branch cancels the prior (cosine −0.95, retention 3.2%, posterior epistemic std ≈ 0.0009 m vs scaled-prior 0.0063 m). **Open question:** pathological collapse vs **bootstrap-consistent contraction under the frozen-feature Stage-2 estimator**. Decisive object: whether bootstrap indices imply different **optimal predictors** — via common-anchor score-gradient geometry and precisely-defined direct particle re-optimization; never weight diversity or scalar risks.

**Hazards:** checkout 23 commits behind `37633fe`; unsanctioned `b4_*`/`b5_*` dirs.

**Binding constraints (review rounds 1–3):** fit/eval separation (all perturbations/gradients/re-opts on the **450 fit families**; the 50 val families only evaluate resulting predictors); pinned scale functional `s_epi = [Σ wᵢV_epi,ᵢ/Σ wᵢ]^{1/2}` (never mean-of-√V), strata = all-wettable / wet>0.01 m / front 0.01–0.10 m / early-mid-late lead, intervals resample **families + draws** (never cells); fixed predeclared margins; two gates GD0/GP1 with an explicit **indeterminate** outcome; P1a mandatory in all branches; design-aware CRPS everywhere including impact metrics; terminology "bootstrap-consistent contraction under the frozen-feature Stage-2 estimator".

**Reuse:** crossed/fixed-support fair-CRPS estimators; `PersistentDirichletParticleControl`; `--allow-single-reference`; legacy remap script (+`--write-artifacts` exports); contraction/OOD-ranking modules; preflight/audit tooling; paired per-family diffs; impact-metric area/arrival functionals (member-level series to be rescored design-aware).
**Build:** direct-particle re-optimization driver (two modes + anchors); common-anchor gradient-geometry diagnostics; whitened stratified cancellation; design-aware impact CRPS; `crps_noninferiority` with sensitivity table; P1b factorial arms.

---

## Step 0 — Repo hygiene (blocks everything)
Verify checkout; fast-forward to `37633fe`; branch `codex/neon-stage2-phase5-diagnostics`; preflight refuses mismatched git heads. Quarantine `b4_*`/`b5_*` (markers referencing DECISION.txt). Sole-author commits; red-test-first; no dirty-tree launches.

## Phase D — Zero-training diagnostics (≈6–8 GPU-h)

Order: **D2 → D3 → D1 → D1′ → GD0**; D4/D5 parallel (descriptive, non-gating); D6 alongside.

### D1′ — Direct particle re-optimization (defines `s_direct`; the plan's centerpiece)

**Model class per draw b** (frozen: Stage-1, δ0(φ̄_canonical), feature map; index `u_b ~ N(0,I)` defines *both* the weights and the prior slice):

  β̂_b = argmin_β  Σ_{i∈fit} w_i(u_b) · S_fCRPS({G_i^k + δ0(φ̄_i) + q_β(φ_i^k) + p_b(φ_i^k)}_k, H_i) + R(β),  with p_b(φ) = α·E^P(φ, u_b)

- **Two modes:** (i) **data-bootstrap spread**: α=0 (isolates contraction from data); (ii) **RPF direct spread**: indexed prior retained (isolates non-amortized prior cancellation/retention).
- **Two q_β classes:** (a) **last-layer benchmark** — non-indexed linear correction on the same features (32 draws; gate-relevant); (b) **broader-head sensitivity check** — full independent copy of the small correction head (8 draws; qualitative flag, never in the gate interval). **Never** re-optimize the shared Hermite coefficient layer at a single fixed u_b (unidentifiable directions).
- **Representability caveat (stated in outputs):** linear q_β cannot fully represent the MLP prior slice ⇒ some retention in the last-layer RPF mode is guaranteed *by construction*; the full-head variant can fully cancel. The (a)/(b) pair under RPF mode brackets cancellation **by representability** — last-layer RPF retention is never reported as "healthy" without this caveat.
- **Anchoring:** α=0 mode uses one shared uniform-weight anchor θ̂₀ (optimized from B3 under w≡1 to verified residual; all draws start there; identical latent banks/members/masks/regularization/numerics). RPF mode uses **per-draw uniform anchors θ̂₀(u_b)** (no common uniform optimum exists across indexed priors); report per-draw weight-induced displacement (from θ̂₀(u_b)) separately from the total prior-inclusive spread — the latter is what an amortized network should match.
- `s_direct` = pinned scale functional of the fitted predictors **evaluated on the 50 val families**; intervals over families+draws; multi-start ×2 spot-checks (DC objective); w≡1 reproduces the anchor (test).

### Other D tasks

| # | Task | Design |
|---|---|---|
| D2 | **Common-anchor gradient geometry** (450 fit families, frozen features) | Decisive: `Δg(u) = Σᵢ(wᵢ(u)−1)·gᵢ(β̂₀)` with every family gradient at the **same uniform anchor** β̂₀; analyze effective rank of {Δg}, pairwise cosines, variation vs minibatch-noise floor, displacement `H_λ⁻¹Δg` and its functional size on val families. **Separately** report B3 indexed-model gradients (contain prior/prediction-state effects — never attributed to bootstrap differentiation). Weight diagnostics: ESS distribution (d-stable by construction — verify), off-diagonal family-logit correlation (predicted E\|cos\| ≈ √(2/πd): 0.20 @16 → 0.07 @128), participation ratio, Jacobian spectrum; exact `Φ⁻¹(1−e^{−w})` rank test on **raw, unclipped, untempered, unnormalized** weights only. Tests: uniform weights ⇒ Δg≡0 at the anchor; synthetic distinct-optima case detected. |
| D3 | **Whitened, stratified cancellation** (saved B3 tensors) | Compute the Hermite-basis Gram (`Cov[(qⱼᵀu)²−1,(q_ℓᵀu)²−1] = 2(qⱼᵀq_ℓ)²` + linear block), eigendecompose, report cancellation along **orthogonal covariance modes** (coordinate-free). Per stratum (lead bins × wetness regimes): trainable–prior cosine, regression slope of trainable on prior, prior variance, trainable variance, retained posterior variance, residual variance after best linear cancellation fit (stable when prior ≈ 0). Cancellation-vs-lead slope. |
| D1 | **Local last-layer sensitivity** (screen) | `Δθ⁽ᵇ⁾ ≈ −H_λ⁻¹Σ_{i∈fit}(wᵢ⁽ᵇ⁾−1)gᵢ(β̂₀)`, ~256 draws, evaluated on val families with the pinned functional. Labeled local sensitivity (DC objective ⇒ no exactness/bound claims); demoted to screening if it disagrees with D1′. |
| D4 | **Legacy N-sweep remap** (descriptive) | Exports → remap → γ̂ + per-N covariate table (base RMSE, prior scale, posterior scale, retention/cancellation, epoch/convergence, subset hash). Preliminary label. |
| D5 | **OOD probe** (13 events, R=1; descriptive) | Existing R=1 path + ranking module (CIs, LOEO, rank table, top-3). Non-ranking ⇒ limited OOD utility, **not** contraction evidence. Approved claim wording only. |
| D6 | **`crps_noninferiority`** | Fixed δ = 1×10⁻⁴ m (predeclared ≈0.5% of base CRPS); test `UCB95(paired Δ) ≤ δ`; **sensitivity table**: δ=0 (strict), δ=10⁻⁴ (primary), raw paired diff + CI. B3 re-worded; artifacts untouched. |

**Design-aware impact scoring (built here, used by pilots):** compute member-level functionals first — `A_{m,k,t} = Σᵢ Aᵢ·1[h_{m,k,t,i} > q]` (and arrival times / pooled depths per (m,k)) — then score with **crossed** fair-CRPS for shared-bank continuous indices, **fixed-support** for persistent particles, independent only for independent nesting. Never flatten M×K in impact space (would recreate the duplicate-member bias fixed for depth).

**Gate GD0 (provisional; predeclared log-scale equivalence):** with `log ρ = log s_observed − log s_direct` and equivalence region `|log ρ| ≤ log 2`:
- **Contraction-consistent:** CI(log ρ) ⊂ equivalence region ∧ Δg differentiation weak (below noise).
- **Under-delivery:** UCB(ρ) < 0.5 ∧ Δg implies distinct optima (displacements functionally meaningful).
- **Indeterminate:** neither. → In **all three** outcomes, P1a runs as the mandatory adjudication experiment.

## Phase P — Pilots (v3 cache, TR-450; screen 1 seed, accept ≥3)

**P1a — Persistent Dirichlet particles (mandatory).** Existing mode. Diagnostics: between-particle Δg/risk differentiation vs within-particle noise; pinned scale vs `s_direct`; design-aware (fixed-support) depth + impact CRPS.

**Gate GP1 (final mechanism verdict; hypothesis-dependent readings of P1a):**
- *Confirmation of contraction:* P1a scale agrees with `s_direct` (equivalence region) ∧ differentiation **weak** ∧ skill + impact non-inferior. → verdict "bootstrap-consistent contraction under the frozen-feature Stage-2 estimator"; paper pivots to γ̂ + OOD association; P1b/P2 dropped.
- *Repair of under-delivery:* P1a retains substantially more than the continuous EpiNet ∧ approaches `s_direct` ∧ differentiation **exceeds** noise ∧ skill non-inferior. → continuous-amortization failure established; P1b factorial runs.
- *Shared-subspace pathology* (tightened — all four required): `s_direct` materially above P1a spread ∧ weight-induced functional displacements meaningful ∧ **independent particles still cancel their own prior** ∧ skill does not require that cancellation. → P2 eligible.
- *Mixture / indeterminate:* factorial + targeted follow-up.

**P1b — Factorial (only if GP1 implicates amortization/dimensionality):** A: shared `u∈R^128`, full 128 linear terms (combined richer-index+capacity; growth acknowledged). B: `u∈R^128` weights, fixed stored 16-dim model projection (*deliberate* fiber-averaging — isolates the amortization bottleneck). C: `d_e=16`, parameter-count-matched wider head (capacity control). **Identical across arms:** prior amplitude, family order, latent banks, optimizer budget, selection criteria, validation draws. Separate `z_w ⊥ z_e` remains rejected (expectation-averaging collapse).

**P1 pass criteria (arm-appropriate per GP1 reading):** RMSE ≤ 0.001 m margin; `crps_noninferiority` (δ=10⁻⁴ m, sensitivity table); **design-aware inundated-area trajectory CRPS non-inferior** (arrival-time reported); stratified/whitened cancellation moves as D3 predicts; `s_epi` reported against `s_direct`; differentiation criterion per hypothesis (weak for confirmation; above-noise for repair); ≥3-seed consistency.

**P2 — Partially non-representable prior (only under GP1 shared-subspace verdict).** Reuse `prior_rff_dim`. Acceptance: depth/frequency/front-controlled association ∧ RMSE + crossed-CRPS + impact non-inferiority ∧ ≥3 fixed-prior seeds ∧ risk-coverage improvement ∧ LOEO-stable OOD ranking ∧ amplitude never tuned on evaluation events. Retention/cancellation excluded by construction; RPF-deviation noted.

## Phase S — Scale-out (after a pilot passes on ≥3 seeds)
Replicated N-sweep (≥3 seeded nested permutations; γ̂ with across-replicate uncertainty); full OOD; DE cross-check; α-sweep only if amplitude—not structure—is the residual gap. Quarantine stands until a new DECISION artifact supersedes.

## Execution order (master)
1. Step 0 hygiene/provenance (verify, don't assume). 2. Define + implement the direct-particle model class (D1′ scaffolding). 3. D2 common-anchor gradient geometry (fit families). 4. D3 whitened cancellation. 5. D1 sensitivity. 6. D1′ both modes + anchors. 7. Design-aware impact CRPS. 8. **GD0** (three-way). 9. **P1a** (mandatory). 10. **GP1**. 11. P1b factorial / P2 (conditional on GP1). 12. D4/D5 throughout as descriptive support. 13. Phase S.

## Verification
Full suite green per commit (container runner); new tests: anchor-reproduction (w≡1 ⇒ θ̂₀), uniform ⇒ Δg≡0, synthetic distinct-optima detection, raw-weight transform-rank, whitened-mode invariance under basis rotation, design-aware impact estimator vs brute force, margin fixtures. GD0/GP1 emit checksummed DECISION artifacts (ρ + CIs, Δg geometry, whitened strata tables, γ̂+covariates, OOD ranking); pilot reports in the `rung_attribution.csv` schema.

## Compute budget
D-phase ≈ 6–8 GPU-h (θ̂₀ anchors + 32 last-layer ×2 modes + 8 full-head + exports) · P1a ≈ 2–6 GPU-h screen, +4–12 acceptance · P1b factorial (conditional) ≈ 9–24 GPU-h · P2 (conditional) ≈ 12–20 GPU-h · Phase S ≈ 20–60 GPU-h. GD0/GP1 stop-loss the spend.

## Risks / self-detection
- DC-objective nonconvexity in re-opts → common/per-draw anchors + multi-start spot-checks + residual assertions.
- Last-layer RPF retention floor (representability) → caveat bound to the output; full-head pair brackets it.
- `s_direct` sampling error → CIs into ρ; equivalence-region gating; indeterminate outcome available.
- d=128 under-differentiation if fit-family gradients near-collinear → D2 effective rank detects pre-P1b.
- Broader-head check contradicting last-layer `s_direct` → flags unreliability (never silently gates).
- P2 decorative risk → multi-seed/coverage/firewall criteria + GP1 conditioning.
- Checkout drift / parallel sessions → Step 0 pin + preflight refusal. n=13 OOD noise; single-subset γ̂ → descriptive labels; replication in Phase S.
---

## Implementation status

The implementation is organized around fail-closed, checksummed evidence artifacts. Code availability does not imply that a scientific gate has passed; the scheduler launches conditional work only after the corresponding replicated decision artifact authorizes it.

| Plan component | Implementation entry point | Execution status |
|---|---|---|
| Protocol and quarantine | `scripts/neon_phase5_preflight.py`, `scripts/neon_phase5_quarantine.py` | Implemented; run-time preflight required |
| D2/D1 geometry | `scripts/neon_phase5_geometry.py` | Implemented; not interpreted until artifacts pass provenance checks |
| D3 cancellation and skill attribution | `scripts/neon_phase5_cancellation.py` | Implemented with one full correction pass per chunk |
| D1' direct re-optimization | `scripts/neon_phase5_direct.py` | Implemented for data/RPF and last/full modes |
| GD0 and GP1 decisions | `scripts/neon_phase5_decision.py` | Implemented with explicit indeterminate outcomes and replicated evidence checks |
| P1a diagnostics | `scripts/neon_phase5_p1a_eval.py` | Implemented with common-anchor gradient and risk signal-to-noise diagnostics |
| P1b/P2 pilots | `scripts/submit_neon_phase5_conditional_pilot.sh`, `scripts/neon_phase5_pilot_gate.py` | Conditional; cannot launch without a compatible replicated GP1 decision |
| Replicated N-sweep | `scripts/submit_neon_phase5_scaleout.sh`, `scripts/neon_contraction_analysis.py` | Conditional on replicated acceptance |
| Historical OOD ranking | `scripts/submit_neon_phase5_ood_evidence.sh`, `scripts/neon_ood_ranking_analysis.py` | Conditional on replicated acceptance; checkpoint-pinned |
| Deep-ensemble cross-check | `scripts/submit_neon_phase5_de_evidence.sh` | Conditional on replicated acceptance; checkpoint-pinned |
| Terminal audit | `scripts/neon_phase_s_finalize.py` | Implemented; verifies code, protocol, gate, checkpoint, and all terminal evidence lineages |

### Provenance and launch safety

- Every completed Stage-2 training run writes a checksummed
  `TRAINING_COMPLETE.json` that pins the checkpoint, history, preflight, Git
  commit, rung, and training-family count.
- Phase-5 diagnostics accept the legacy B3 source only when it matches the
  root checksummed preflight. Derived pilot sources must match their signed
  training-completion manifest.
- Phase-S target resolution re-verifies the selected completion manifest and
  records its path and SHA-256 in `PHASE_S_TARGET.json`.
- Every historical OOD array shard verifies that its Stage-2 checkpoint is the
  checkpoint recorded by the signed in-distribution evidence before inference.
- Submission, decision, provenance, and terminal evidence manifests are
  atomically written with SHA-256 sidecars. Submission scripts refuse dirty
  repositories, incompatible Git heads, reused output roots, and unsatisfied
  scientific gates.
