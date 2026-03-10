# Scripts Layout

Root `scripts/` now contains compatibility wrappers, Slurm launcher wrappers, and operational helpers.
Canonical Python entrypoints live under `neuralop.flood.cli` and `neuralop.flood.eval`.
Canonical Slurm launchers live under:

- `scripts/slurm/train/`
- `scripts/slurm/eval/`
- `scripts/slurm/lib/`
- `scripts/release/`
- `scripts/dev/`

Use the package entrypoints for new automation and code integration. Existing root script paths are
preserved as thin wrappers so current Slurm jobs and ad hoc commands continue to work.
