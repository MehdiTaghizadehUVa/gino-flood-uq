# ADR 0006: Forcing-Conditioned Baseline Initial WD History

## Status

Accepted

## Context

The first serving implementation used an all-zero water-depth history for the three dynamic frames before rollout. That dry-start policy was reproducible and simple, but it made early lead times less physically realistic for uploaded events whose spin-up forcing resembles known wet reference conditions. Asking users to upload a 5904-cell initial WD field would add a fragile data contract and a high validation burden.

The deployed coastal model already has HEC-RAS histories for train/calibration reference events. Those histories can provide a deterministic baseline state when selected by the uploaded event's first spin-up/history forcing rows.

## Decision

Normal web serving uses a Forcing-Conditioned Baseline by default. The worker compares the first `skip_before_timestep + n_history` forcing rows against an Initial Condition Library built from train/calibration events only. It selects the nearest eligible references in robustly scaled feature space, blends their HEC-RAS WD histories, normalizes that physical WD history with the train-fit dynamic normalizer, and starts autoregressive rollout from that state.

Dry zero history remains available through `DryInitialConditionProvider` for tests, diagnostics, and explicit parity experiments. Held-out test events are excluded from the normal Initial Condition Library to avoid hidden test-ground-truth leakage.

Each run writes `initial_condition_selection.json` and records the selection in the run manifest and HDF5 attributes when ensemble export is enabled.

## Consequences

Early lead-time forecasts should be more physically plausible for events similar to train/calibration histories. Results will not exactly match old dry-start artifacts by design. Exact parity with held-out evaluation artifacts requires an explicit parity library or uploaded initial WD feature, neither of which is part of V1 serving.

The Model Bundle contract now validates the initial-condition library when the default mode requires one. Deployment must therefore copy the `.npz` library alongside checkpoints, normalizers, domain tensors, and calibration files.
