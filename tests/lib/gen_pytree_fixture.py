"""Regenerate fixtures/pytree_cases.json from real jaxlib.

Run with the reference env (real jaxlib 0.12.0.1), from the repo root:

  .venv-jaxref/bin/python tests/lib/gen_pytree_fixture.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jaxlib  # noqa: E402
from jaxlib._jax import pytree as P  # noqa: E402

import _pytree_cases  # noqa: E402


def main():
  out = {}
  for name, thunk in _pytree_cases.make_cases(P):
    assert name not in out, name
    out[name] = _pytree_cases.run_case(thunk)
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                      "pytree_cases.json")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(dict(jaxlib_version=jaxlib.__version__, cases=out), f,
              ensure_ascii=False, indent=0, sort_keys=True)
  print(f"wrote {len(out)} cases to {path}")


if __name__ == "__main__":
  main()
