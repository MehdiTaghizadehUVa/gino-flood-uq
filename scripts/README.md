# Scripts Layout

Root `scripts/` now contains the active `flood_wv_*` Python entry wrappers and operational folders.
Canonical Python entrypoints live under `neuralop.flood.cli` and `neuralop.flood.eval`.
Canonical Slurm launchers live under:

- `scripts/slurm/train/`
- `scripts/slurm/eval/`
- `scripts/slurm/lib/`
- `scripts/release/`
- `scripts/dev/`
- `scripts/runtime/`
- `scripts/archive/`

Runtime artifacts for maintained launchers should go under `scripts/runtime/`:

- `scripts/runtime/logs/`
- `scripts/runtime/eval_outputs/`
- `scripts/runtime/checkpoints/`

Use the package entrypoints and canonical shell scripts for new automation and code integration.
Redundant root-level shell wrappers have been moved under `scripts/archive/root_shell_wrappers/`.
Archived legacy Python wrapper names live under `scripts/archive/legacy_python_wrappers/`; active wrappers use the `flood_wv_*` naming scheme.
