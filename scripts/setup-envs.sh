#!/usr/bin/env bash
# Create the two test environments:
#   .venv         shax (editable) + test deps, and NO jax/jaxlib
#   .venv-jaxref  real jax[cpu] at the pinned version + test deps
# Usage: scripts/setup-envs.sh [shax|jaxref]   (default: both)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
JAX_VERSION="${JAX_VERSION:-0.12.0.1}"
PY="${PY:-3.13}"
command -v uv >/dev/null || { module load uv 2>/dev/null || true; }

want="${1:-both}"
if [[ "$want" == both || "$want" == shax ]]; then
  [[ -d .venv ]] || uv venv -q -p "$PY" .venv
  VIRTUAL_ENV=.venv uv pip install -q -r harness/requirements-test.txt -e .
  if .venv/bin/python -I -c 'import importlib.util as u,sys; sys.exit(not (u.find_spec("jax") or u.find_spec("jaxlib")))'; then
    echo "ERROR: jax or jaxlib got installed into .venv; remove it" >&2; exit 1
  fi
  echo ".venv ready"
fi
if [[ "$want" == both || "$want" == jaxref ]]; then
  [[ -d .venv-jaxref ]] || uv venv -q -p "$PY" .venv-jaxref
  VIRTUAL_ENV=.venv-jaxref uv pip install -q -r harness/requirements-test.txt "jax[cpu]==$JAX_VERSION"
  echo ".venv-jaxref ready"
fi
