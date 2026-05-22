# 0005. Phase 2-Only Retraining-Candidate Policy

## Status

Accepted

## Context

Phase 1 reference-envelope screening is useful for explaining when an uploaded forcing or completed forecast descriptor falls outside the train/test reference distribution. However, univariate envelope checks can be noisy when a descriptor is fragile, boundary-sensitive, or not directly interpretable as a retraining need. One observed example is `calibrated_max_mean_wd_m` falling outside the reference envelope for a known test-set event, which created a `NEW` candidate even though the signal was not strong enough to justify HEC-RAS labeling by itself.

## Decision

Keep Phase 1 univariate percentile checks as visible diagnostics, but make them decision-ineligible. Phase 1 flags remain in the Monitoring Report with `details.decision_eligible = false` and are shown in the UI as "Reference-envelope diagnostics."

New retraining candidates are created only by Phase 2 decision signals:

1. Candidate-level regularized Mahalanobis outlier against empirical reference distance percentiles.
2. Candidate-level reference-derived post-run heuristic score, including high uncertainty-to-signal, high affected-area uncertainty, large calibration shift, and high checkpoint-disagreement metrics.
3. Deterministic control sampling for balanced candidate capture.
4. Persistent population drift paired with an individually suspicious, non-fragile reference-envelope descriptor from the same descriptor family.

Fragile descriptors are retained for report/debug visibility but cannot independently create candidates. This includes `calibrated_max_mean_wd_m`, raw/full-domain max-depth descriptors, raw cell-count descriptors, and static wettable-area constants.

Existing candidate records are preserved for audit history. Admins may reject older noisy candidates manually; the policy does not auto-delete prior records.

## Consequences

- Phase 1 still explains descriptor movement and supports debugging.
- Candidate packages are less likely to be created from fragile one-dimensional signals.
- The UI separates "Candidate decision" from "Reference-envelope diagnostics."
- Known train/test events with only a fragile Phase 1 envelope hit no longer become `NEW` candidates.
- Candidate creation remains research-oriented and warn/allow; monitoring is not proof of model error or operational flood guidance.

## Alternatives Considered

- **Delete Phase 1 screening**: Rejected because univariate diagnostics are interpretable and helpful for debugging.
- **Keep Phase 1 as candidate-eligible for all descriptors**: Rejected because it creates noisy candidates from fragile descriptors.
- **Manually curate every Phase 1 descriptor as candidate-eligible or not**: Partially adopted only for the narrow population-drift reinforcement exception; broad Phase 1 candidate decisions remain disabled.
