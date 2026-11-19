#!/usr/bin/env bash
# Export notebook.py as an offline marimo WASM app: code cells shown, outputs
# embedded (computed natively at export time) so they show on load.
#
#   examples/optax-wasm/export.sh [OUT_DIR [MARIMO_ARGS...]]   # default: /tmp/optax-wasm
#
# Needs `marimo` (0.25.x) on PATH, and `uv` (on PATH, or $UV, default
# ~/.local/bin/uv). Builds the wheels in dist/ first if they're
# missing. Extra arguments go to marimo; `--no-execute` skips embedding the
# outputs (the browser then computes them all on load, ~12 s).
# Check the result with scripts/check-wasm-export.py OUT_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-/tmp/optax-wasm}"
shift || true

if ! command -v uv > /dev/null; then
  # marimo uses uv to resolve the notebook's local wheels, and --execute runs
  # the notebook in a uv sandbox built from the PEP 723 header (so `jax` is
  # shax; marimo's own environment may have real jax or none).
  UV="${UV:-$HOME/.local/bin/uv}"
  [[ -x "$UV" ]] || { echo "export.sh: needs uv (on PATH, or set UV)" >&2; exit 1; }
  PATH="$(dirname "$UV"):$PATH"
fi

if ! compgen -G "$ROOT/dist/shax-*.whl" > /dev/null; then
  "$ROOT/packaging/build-wheels.sh"
fi
# --mode run: read-only app whose widgets re-run cells in the browser;
# --show-code: show the code cells (run mode hides them by default);
# --offline: bundle Pyodide and every wheel, no CDN; --execute: embed outputs.
# A later --no-execute in "$@" overrides --execute.
marimo export html-wasm "$HERE/notebook.py" -o "$OUT" \
  --mode run --show-code --offline --execute "$@"
