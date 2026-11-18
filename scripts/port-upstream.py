#!/usr/bin/env python3
"""Copy an upstream JAX module into shax with mechanical import rewrites.

  scripts/port-upstream.py jax/_src/config.py            # -> shax/jax/_src/config.py
  scripts/port-upstream.py jax/_src/config.py --diff     # show diff vs. current port
  scripts/port-upstream.py --status                      # every ported file + hand-edit hunks
  scripts/port-upstream.py --rebase vendor/jax-new --new-tag jax-vX.Y.Z [src ...]

--rebase moves ported files to a new upstream: for each one it three-way
merges (git merge-file) the current file (ours) with the mechanical port of
the new upstream (theirs), using the mechanical port of the current pin as the
base, so `SHAX:` hand edits carry over. Conflicts are left marked in place.
See "Moving the upstream pin" in AGENTS.md.

Rewrites `from jax...` / `import jax...` to their `shax.jax` equivalents and
adds a provenance header. Hand edits after that should be marked `# SHAX:` so
`--diff` stays reviewable when the upstream pin moves.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import subprocess
import sys
import tempfile

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
    # from jaxlib.mlir.x import y: shax's MLIR stubs live where upstream
    # re-exports jaxlib's (jax._src.lib.mlir)
    (re.compile(r"^(\s*)from jaxlib\.mlir(\.[\w.]+)? import ", re.M),
     lambda m: f"{m.group(1)}from shax.jax._src.lib.mlir{m.group(2) or ''} import "),
    # opt_einsum is vendored (Pyodide doesn't bundle it)
    (re.compile(r"^(\s*)import opt_einsum\s*$", re.M),
     lambda m: f"{m.group(1)}from shax._vendor import opt_einsum"),
    (re.compile(r"^(\s*)from opt_einsum(\.[\w.]+)? import ", re.M),
     lambda m: f"{m.group(1)}from shax._vendor.opt_einsum{m.group(2) or ''} import "),
]


def rewrite(src_rel: str, text: str, tag: str = TAG) -> str:
  for pat, repl in _RULES:
    text = pat.sub(repl, text)
  leftover = re.findall(r"^\s*import jax\.[\w.]+\s*$", text, re.M)
  if leftover:
    print(f"warning: bare dotted imports need a hand edit: {leftover}",
          file=sys.stderr)
  return HEADER.format(src=src_rel, tag=tag) + text


_PORTED_RE = re.compile(r"^# SHAX: ported from (\S+) \(")


def ported_files() -> dict[str, Path]:
  """Upstream path -> shax file, for every file with a provenance header."""
  out = {}
  for f in sorted((ROOT / "shax").rglob("*.py")):
    with open(f) as fh:
      m = _PORTED_RE.match(fh.readline())
    if m:
      out[m.group(1)] = f
  return out


def _hunks(a: str, b: str) -> int:
  return sum(1 for l in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                             lineterm="", n=0)
             if l.startswith("@@"))


def cmd_status() -> None:
  files = ported_files()
  clean = 0
  for src, dest in files.items():
    up = UPSTREAM / src
    if not up.exists():
      print(f"  missing upstream  {src}")
      continue
    n = _hunks(rewrite(src, up.read_text()), dest.read_text())
    if n:
      print(f"  {n:3d} hunks        {dest.relative_to(ROOT)}")
    else:
      clean += 1
  print(f"{len(files)} ported files, {clean} identical to the mechanical port")


def cmd_rebase(new_root: Path, new_tag: str, srcs: list[str]) -> int:
  files = ported_files()
  if srcs:
    files = {s: files[s] for s in srcs}
  conflicted, gone = [], []
  for src, dest in files.items():
    old_up, new_up = UPSTREAM / src, new_root / src
    if not new_up.exists():
      gone.append(src)
      continue
    base = rewrite(src, old_up.read_text()) if old_up.exists() else ""
    theirs = rewrite(src, new_up.read_text(), new_tag)
    with tempfile.TemporaryDirectory() as d:
      paths = []
      for name, text in (("base", base), ("theirs", theirs)):
        pth = Path(d) / name
        pth.write_text(text)
        paths.append(str(pth))
      r = subprocess.run(
          ["git", "merge-file", "--diff3", "-L", "shax", "-L", f"upstream {TAG}",
           "-L", f"upstream {new_tag}", str(dest), paths[0], paths[1]],
          capture_output=True, text=True)
    if r.returncode < 0 or r.returncode > 127:
      sys.exit(f"git merge-file failed on {dest}: {r.stderr}")
    if r.returncode:
      conflicted.append((dest, r.returncode))
  for dest, n in conflicted:
    print(f"  CONFLICT ({n})  {dest.relative_to(ROOT)}")
  for src in gone:
    print(f"  gone upstream  {src}  (still ported as {files[src].relative_to(ROOT)})")
  print(f"rebased {len(files) - len(gone)} files onto {new_tag}: "
        f"{len(conflicted)} with conflicts. Next: resolve them, set TAG in this "
        "script and JAX_TAG in fetch-jax.sh, refetch, rerun the baselines.")
  return 1 if conflicted else 0


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("src", nargs="*",
                 help="path(s) relative to vendor/jax, e.g. jax/_src/config.py")
  p.add_argument("--dest", help="default: shax/<src>")
  p.add_argument("--diff", action="store_true",
                 help="print diff between mechanical port and current file")
  p.add_argument("--force", action="store_true", help="overwrite existing dest")
  p.add_argument("--status", action="store_true",
                 help="list ported files and their hand-edit hunks")
  p.add_argument("--rebase", metavar="NEW_UPSTREAM",
                 help="three-way merge ported files onto this upstream checkout")
  p.add_argument("--new-tag", help="tag of the --rebase checkout")
  a = p.parse_args()
  if a.status:
    return cmd_status()
  if a.rebase:
    if not a.new_tag:
      p.error("--rebase needs --new-tag")
    return cmd_rebase(Path(a.rebase).resolve(), a.new_tag, a.src)
  if len(a.src) != 1:
    p.error("give exactly one src to port or --diff")
  (src_rel,) = a.src
  src = UPSTREAM / src_rel
  dest = ROOT / (a.dest or ("shax/" + src_rel))
  ported = rewrite(src_rel, src.read_text())
  if a.diff:
    cur = dest.read_text() if dest.exists() else ""
    sys.stdout.writelines(difflib.unified_diff(
        ported.splitlines(True), cur.splitlines(True),
        f"mechanical/{src_rel}", str(dest.relative_to(ROOT))))
    return
  if dest.exists() and not a.force:
    sys.exit(f"{dest} exists (use --force to overwrite, --diff to compare)")
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text(ported)
  print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
  sys.exit(main())
