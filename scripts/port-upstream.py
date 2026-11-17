#!/usr/bin/env python3
"""Copy an upstream JAX module into shax with mechanical import rewrites.

  scripts/port-upstream.py jax/_src/config.py            # -> shax/jax/_src/config.py
  scripts/port-upstream.py jax/_src/config.py --diff     # show diff vs. current port

Rewrites `from jax...` / `import jax...` to their `shax.jax` equivalents and
adds a provenance header. Hand edits after that should be marked `# SHAX:` so
`--diff` stays reviewable when the upstream pin moves.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "jax"
TAG = "jax-v0.12.0.1"

HEADER = ("# SHAX: ported from {src} ({tag}); Apache-2.0, see "
          "shax/jax/_src/LICENSE-JAX.\n# SHAX: hand edits are marked "
          "`SHAX:`; regenerate a diff with scripts/port-upstream.py {src} --diff\n")

_RULES = [
    # from jax._src.x import y  /  from jax import y  /  from jax.x import y
    (re.compile(r"^(\s*)from jax(\.[\w.]+)? import ", re.M),
     lambda m: f"{m.group(1)}from shax.jax{m.group(2) or ''} import "),
    # import jax._src.x as y  /  import jax.numpy as jnp
    (re.compile(r"^(\s*)import jax\.([\w.]+) as (\w+)", re.M),
     lambda m: f"{m.group(1)}import shax.jax.{m.group(2)} as {m.group(3)}"),
    # import jax
    (re.compile(r"^(\s*)import jax\s*$", re.M),
     lambda m: f"{m.group(1)}from shax import jax"),
]


def rewrite(src_rel: str, text: str) -> str:
  for pat, repl in _RULES:
    text = pat.sub(repl, text)
  leftover = re.findall(r"^\s*import jax\.[\w.]+\s*$", text, re.M)
  if leftover:
    print(f"warning: bare dotted imports need a hand edit: {leftover}",
          file=sys.stderr)
  return HEADER.format(src=src_rel, tag=TAG) + text


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("src", help="path relative to vendor/jax, e.g. jax/_src/config.py")
  p.add_argument("--dest", help="default: shax/<src>")
  p.add_argument("--diff", action="store_true",
                 help="print diff between mechanical port and current file")
  p.add_argument("--force", action="store_true", help="overwrite existing dest")
  a = p.parse_args()
  src = UPSTREAM / a.src
  dest = ROOT / (a.dest or ("shax/" + a.src))
  ported = rewrite(a.src, src.read_text())
  if a.diff:
    cur = dest.read_text() if dest.exists() else ""
    sys.stdout.writelines(difflib.unified_diff(
        ported.splitlines(True), cur.splitlines(True),
        f"mechanical/{a.src}", str(dest.relative_to(ROOT))))
    return
  if dest.exists() and not a.force:
    sys.exit(f"{dest} exists (use --force to overwrite, --diff to compare)")
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text(ported)
  print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
  main()
