# Releasing This Project on GitHub

This repo currently has `origin` pointing at the upstream NeuralOperator repo. To **release your variant** (GINO flood, WV configs, etc.), use your own GitHub repository.

## Prerequisites

- A GitHub account.
- A **new repository** on GitHub under [MehdiTaghizadehUVa](https://github.com/MehdiTaghizadehUVa) (e.g. `neuraloperator-no-physics` or `gino-flood`). Create it at https://github.com/new (leave "Initialize with README" unchecked if you're pushing this repo).

## Step 1: Commit your changes

Ensure all changes you want in the release are committed:

```bash
git status
git add -A
git commit -m "Prepare release: GINO flood, WV configs, and project customizations"
```

(Or commit in smaller steps; then proceed when `main` is ready to release.)

## Step 2: Add your GitHub repo as a remote and push

Using your repo [gino-flood-uq](https://github.com/MehdiTaghizadehUVa/gino-flood-uq):

```bash
git remote add github https://github.com/MehdiTaghizadehUVa/gino-flood-uq.git
git push -u github main
```

## Step 3: Create and push a version tag

Pick a version (e.g. first release `1.0.0` or next after upstream `1.0.3`). Ensure `neuralop/__init__.py` has the same `__version__` if you want consistency.

```bash
# Create an annotated tag (recommended)
git tag -a v1.0.3 -m "Release v1.0.3: GINO flood and WV pluvial configs"

# Push the tag to your GitHub repo
git push github tag v1.0.3
# Or: git push myorigin v1.0.3
```

## Step 4: Create a GitHub Release (optional but recommended)

1. Open your repo on GitHub: `https://github.com/YOUR_USERNAME/YOUR_REPO`.
2. Go to **Releases** → **Create a new release**.
3. Choose the tag you just pushed (e.g. `v1.0.3`).
4. Set the release title (e.g. `Release v1.0.3`).
5. Add release notes (e.g. highlights, configs added, links to docs).
6. Publish the release.

## Quick reference: your remotes after setup

| Remote   | URL                              | Use              |
|----------|----------------------------------|------------------|
| `origin` | https://github.com/neuraloperator/neuraloperator.git | Upstream (pull only) |
| `github` | https://github.com/MehdiTaghizadehUVa/gino-flood-uq.git | Your repo (push releases) |

## Bumping version for the next release

1. Update `neuralop/__init__.py`: set `__version__ = 'x.y.z'`.
2. Commit, push to your remote, then tag and create a new release as above.
