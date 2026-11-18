#!/usr/bin/env bash
# Build shax and the jax/jaxlib shim wheels into dist/ (pure Python, any platform).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
mkdir -p "$ROOT/dist"
for d in "$ROOT" "$ROOT/packaging/jax-shim" "$ROOT/packaging/jaxlib-shim"; do
  "$UV" build --wheel -q -o "$ROOT/dist" "$d"
done
ls "$ROOT/dist"
