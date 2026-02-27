#!/usr/bin/env bash
# Switch the 'github' remote to SSH so pushes work without a password on Rivanna.
# Run this after you've set up an SSH key and added it to GitHub (see docs/RIVANNA_GIT_PUSH_SETUP.md).

set -e
cd "$(dirname "$0")/.."

GITHUB_REPO="${GITHUB_REPO:-MehdiTaghizadehUVa/gino-flood-uq}"
SSH_URL="git@github.com:${GITHUB_REPO}.git"

if ! git remote get-url github &>/dev/null; then
  git remote add github "$SSH_URL"
  echo "Added remote github -> $SSH_URL"
else
  git remote set-url github "$SSH_URL"
  echo "Set remote github to $SSH_URL"
fi

echo "Test with: git push github main"
