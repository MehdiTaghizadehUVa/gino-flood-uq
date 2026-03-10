# Flood Application Layout

Flood-specific workflow code lives under `neuralop.flood`.

- `neuralop.flood.cli`: canonical Python entrypoints
- `neuralop.flood.data`: flood dataset and HEC-RAS readers
- `neuralop.flood.processing`: data processors
- `neuralop.flood.train`: operator training interfaces
- `neuralop.flood.eval`: operator, calibrated, and diffusion evaluation modules
- `neuralop.flood.utils`: checkpoint/runtime helpers
- `neuralop.flood.visualization`: rollout/publication plotting helpers

Root `scripts/` now exposes professional `flood_wv_*` Python wrappers; archived legacy wrapper names live under `scripts/archive/legacy_python_wrappers/`.
Canonical shell launchers live under `scripts/slurm/`; root shell wrappers have been archived under `scripts/archive/root_shell_wrappers/`.
WV flood configs are available under `config/flood/wv/`, while legacy config paths remain available.
