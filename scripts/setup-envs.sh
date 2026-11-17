#!/usr/bin/env bash
# Create the two test environments:
#   .venv         shax (editable) + test deps, and NO jax/jaxlib
#   .venv-jaxref  real jax[cpu] at the pinned version + test deps
# Usage: scripts/setup-envs.sh [shax|jaxref]   (default: both)
#
# Uses uv (default ~/.local/bin/uv; override with UV=) with uv-managed Python
# only, never system/module interpreters.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
UV="${UV:-$HOME/.local/bin/uv}"
JAX_VERSION="${JAX_VERSION:-0.12.0.1}"
# Pyodide 314.x runs CPython 3.14.
PY="${PY:-3.14}"
# Pin shax's env to the versions bundled with Pyodide 314.0.8 so local results
# match the deployment target (see AGENTS.md "Pyodide target").
PYODIDE_PINS=(numpy==2.4.6 scipy==1.18.0 ml_dtypes==0.5.4)

make_venv() {
  local dir="$1"
  [[ -d "$dir" ]] || "$UV" venv -q --python-preference only-managed -p "$PY" "$dir"
  "$dir/bin/python" -I -c 'import sys; sys.exit(sys.version_info.releaselevel != "final")' \
    || { echo "ERROR: $dir uses a pre-release Python ($("$dir/bin/python" -V)); update uv (\`$UV self update\`) and recreate it" >&2; exit 1; }
}

want="${1:-both}"
if [[ "$want" == both || "$want" == shax ]]; then
  make_venv .venv
  VIRTUAL_ENV=.venv "$UV" pip install -q -r harness/requirements-test.txt "${PYODIDE_PINS[@]}" -e .
  if .venv/bin/python -I -c 'import importlib.util as u,sys; sys.exit(not (u.find_spec("jax") or u.find_spec("jaxlib")))'; then
    echo "ERROR: jax or jaxlib got installed into .venv; remove it" >&2; exit 1
  fi
  echo ".venv ready ($(.venv/bin/python -V))"
fi
if [[ "$want" == both || "$want" == jaxref ]]; then
  make_venv .venv-jaxref
  VIRTUAL_ENV=.venv-jaxref "$UV" pip install -q -r harness/requirements-test.txt "jax[cpu]==$JAX_VERSION"
  echo ".venv-jaxref ready ($(.venv-jaxref/bin/python -V))"
fi
