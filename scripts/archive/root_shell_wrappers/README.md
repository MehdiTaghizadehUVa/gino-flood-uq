This folder contains archived root-level shell wrappers that previously forwarded into:

- `scripts/slurm/`
- `scripts/dev/`
- `scripts/release/`

They were removed from the main `scripts/` directory to keep the repo surface clean.

Canonical replacements:

- train/eval launchers: `scripts/slurm/train/` and `scripts/slurm/eval/`
- release helper: `scripts/release/release_to_github.sh`
- GitHub SSH helper: `scripts/dev/set_github_remote_ssh.sh`
