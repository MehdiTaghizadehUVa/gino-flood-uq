# 0003. Drift Screening Preserves Candidate Events Without Blocking Runs

## Status

Accepted

## Context

The Coastal FGN Serving site accepts research scenarios that may differ from the train/test reference events. Researchers need a guardrail that identifies events worth later HEC-RAS labeling and retraining, but Phase 1 must not become an operational decision system or an automatic retraining scheduler.

## Decision

Use a Monitoring Bundle to screen uploaded forcings and completed forecast products against reference-event descriptor distributions. The site warns users when a run is performance-risky or useful for retraining, but valid CSVs are still allowed to run. When a run is recommended as a Retraining Candidate, the system copies a minimal candidate package into a Candidate Event Store that is separate from normal run-artifact retention. Owners can view monitoring reports; admins manage candidate status.

## Consequences

- Researchers can keep using the GPU service while unusual events are captured for later review.
- Candidate evidence survives 30-day run cleanup without pinning every full run artifact.
- Monitoring scores remain transparent research guardrails, not proof of model error or operational flood guidance.
- HEC-RAS scheduling and automatic retraining remain separate Phase 2 work.
