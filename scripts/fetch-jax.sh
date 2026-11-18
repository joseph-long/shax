#!/usr/bin/env bash
# Fetch the pinned upstream JAX checkout used as shax's test oracle.
# Usage: scripts/fetch-jax.sh [--force]
# To try a new pin without touching vendor/jax (see port-upstream.py --rebase):
#   JAX_TAG=jax-vX.Y.Z JAX_DEST=vendor/jax-new scripts/fetch-jax.sh
set -euo pipefail

JAX_TAG="${JAX_TAG:-jax-v0.12.0.1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${JAX_DEST:-$REPO_ROOT/vendor/jax}"
[[ "$DEST" = /* ]] || DEST="$REPO_ROOT/$DEST"

if [[ -d "$DEST/.git" ]]; then
  current="$(git -C "$DEST" describe --tags --exact-match 2>/dev/null || echo unknown)"
  if [[ "$current" == "$JAX_TAG" && "${1:-}" != "--force" ]]; then
    echo "vendor/jax already at $JAX_TAG"
    exit 0
  fi
  if [[ "${1:-}" != "--force" ]]; then
    echo "vendor/jax is at '$current', expected '$JAX_TAG'; rerun with --force to replace it" >&2
    exit 1
  fi
  rm -rf "$DEST"
fi

mkdir -p "$(dirname "$DEST")"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$JAX_TAG" \
  https://github.com/jax-ml/jax "$DEST"
echo "Fetched $JAX_TAG into $DEST"
