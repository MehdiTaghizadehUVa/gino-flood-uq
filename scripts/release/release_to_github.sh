#!/usr/bin/env bash
# Add your GitHub repo as a remote and push main + optional tag.
# Edit GITHUB_REPO below, then run: ./scripts/release/release_to_github.sh

set -e
cd "$(dirname "$0")/.."

# --- EDIT THIS: your GitHub username and repository name ---
GITHUB_REPO="${GITHUB_REPO:-MehdiTaghizadehUVa/gino-flood-uq}"

if [[ "$GITHUB_REPO" == *"YOUR_USERNAME"* ]] || [[ "$GITHUB_REPO" == *"YOUR_REPO"* ]]; then
  echo "Edit GITHUB_REPO in this script or set it when running:"
  echo "  GITHUB_REPO=username/repo ./scripts/release/release_to_github.sh"
  exit 1
fi

REMOTE_NAME="github"
URL="https://github.com/${GITHUB_REPO}.git"

if git remote get-url "$REMOTE_NAME" &>/dev/null; then
  echo "Remote '$REMOTE_NAME' already exists: $(git remote get-url $REMOTE_NAME)"
else
  git remote add "$REMOTE_NAME" "$URL"
  echo "Added remote: $REMOTE_NAME -> $URL"
fi

echo "Pushing main to $REMOTE_NAME..."
git push -u "$REMOTE_NAME" main

echo "Done. To create a release tag, run:"
echo "  git tag -a v1.0.3 -m 'Release v1.0.3'"
echo "  git push $REMOTE_NAME v1.0.3"
echo "Then create a GitHub Release from the tag in the repo's Releases page."
