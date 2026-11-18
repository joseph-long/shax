#!/usr/bin/env bash
# Export notebook.py as an offline marimo WASM app with its code cells shown.
#
#   examples/optax-wasm/export.sh [OUT_DIR]   # default: /tmp/optax-wasm
#
# Needs `marimo` (0.25.x) on PATH. Builds the wheels in dist/ first if they're
# missing. Check the result with scripts/check-wasm-export.py OUT_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-/tmp/optax-wasm}"
if ! compgen -G "$ROOT/dist/shax-*.whl" > /dev/null; then
  "$ROOT/packaging/build-wheels.sh"
fi
# --mode run: read-only app; --show-code: show the code cells (run mode hides
# them by default); --offline: bundle Pyodide and every wheel, no CDN.
marimo export html-wasm "$HERE/notebook.py" -o "$OUT" \
  --mode run --show-code --offline
