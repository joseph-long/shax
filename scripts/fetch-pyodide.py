#!/usr/bin/env python3
"""Fetch a pinned Pyodide runtime plus the wheels the test harness needs, so
`scripts/jaxtest run --target pyodide` works offline.

  python3 -I scripts/fetch-pyodide.py           # -> vendor/pyodide/<version>/

Layout: the npm `pyodide` package's files (pyodide.mjs, pyodide.asm.wasm,
python_stdlib.zip, pyodide-lock.json, ...) at the top level, with the Pyodide
wheels from the CDN next to them (where `loadPackage` looks for them), and
pure-Python wheels from PyPI in `pypi/`. Every download is checked against
the npm integrity hash, the lock file's sha256 or PyPI's sha256.

Needs only `requests` (run with the system python3; curl is blocked here).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import requests

PYODIDE_VERSION = "314.0.8"
# The CPython ABI + platform tag of wheels built for this Pyodide.
PYODIDE_WHEEL_TAG = "cp314-cp314-pyemscripten_2026_0_wasm32"
# Pyodide packages to load (their dependencies are added from the lock file).
PYODIDE_PACKAGES = ["numpy", "scipy", "ml-dtypes", "pytest", "micropip", "pillow",
                    "mpmath", "cloudpickle"]
# Pure-Python test deps Pyodide doesn't bundle; pinned to match .venv.
PYPI_WHEELS = {"absl-py": "2.5.1", "hypothesis": "6.170.15",
               "sortedcontainers": "2.4.0"}

ROOT = Path(__file__).resolve().parent.parent
CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"


def get(url: str) -> bytes:
  r = requests.get(url, timeout=120)
  r.raise_for_status()
  return r.content


def check_sha256(data: bytes, want: str, what: str) -> None:
  got = hashlib.sha256(data).hexdigest()
  if got != want:
    sys.exit(f"sha256 mismatch for {what}: got {got}, want {want}")


def fetch_runtime(dest: Path) -> None:
  meta = requests.get(f"https://registry.npmjs.org/pyodide/{PYODIDE_VERSION}",
                      timeout=60).json()["dist"]
  data = get(meta["tarball"])
  algo, b64 = meta["integrity"].split("-", 1)
  if base64.b64encode(hashlib.new(algo, data).digest()).decode() != b64:
    sys.exit("npm integrity mismatch for the pyodide package")
  with tarfile.open(fileobj=io.BytesIO(data)) as t:
    for m in t.getmembers():
      if not m.isfile() or not m.name.startswith("package/"):
        continue
      out = dest / m.name[len("package/"):]
      out.parent.mkdir(parents=True, exist_ok=True)
      out.write_bytes(t.extractfile(m).read())
  print(f"pyodide {PYODIDE_VERSION} runtime -> {dest}")


def fetch_pyodide_wheels(dest: Path) -> list[str]:
  lock = json.loads((dest / "pyodide-lock.json").read_text())
  pkgs = lock["packages"]
  todo, names = list(PYODIDE_PACKAGES), []
  while todo:
    n = todo.pop()
    if n not in names:
      names.append(n)
      todo += pkgs[n]["depends"]
  for n in sorted(names):
    p = pkgs[n]
    out = dest / p["file_name"]
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == p["sha256"]:
      continue
    data = get(CDN + p["file_name"])
    check_sha256(data, p["sha256"], p["file_name"])
    out.write_bytes(data)
    print(f"  {p['file_name']}")
  return sorted(names)


def _pick_wheel(name: str, urls: list[dict]) -> dict:
  """A pure-Python wheel, else one built for this Pyodide's ABI."""
  wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
  for suffix in ("-none-any.whl", f"-{PYODIDE_WHEEL_TAG}.whl"):
    for u in wheels:
      if u["filename"].endswith(suffix):
        return u
  sys.exit(f"no wheel for {name} usable in Pyodide {PYODIDE_VERSION}")


def fetch_pypi_wheels(dest: Path) -> list[str]:
  out_dir = dest / "pypi"
  out_dir.mkdir(exist_ok=True)
  files = []
  for name, version in PYPI_WHEELS.items():
    info = requests.get(f"https://pypi.org/pypi/{name}/{version}/json",
                        timeout=60).json()
    whl = _pick_wheel(name, info["urls"])
    out = out_dir / whl["filename"]
    if not (out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()
            == whl["digests"]["sha256"]):
      data = get(whl["url"])
      check_sha256(data, whl["digests"]["sha256"], whl["filename"])
      out.write_bytes(data)
      print(f"  pypi/{whl['filename']}")
    files.append(f"pypi/{whl['filename']}")
  return files


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dest", default=str(ROOT / "vendor" / "pyodide" / PYODIDE_VERSION))
  dest = Path(ap.parse_args().dest)
  dest.mkdir(parents=True, exist_ok=True)
  if not (dest / "pyodide.mjs").exists():
    fetch_runtime(dest)
  packages = fetch_pyodide_wheels(dest)
  wheels = fetch_pypi_wheels(dest)
  # What the Node runner loads (scripts/pyodide-pytest.mjs).
  (dest / "shax-manifest.json").write_text(json.dumps(
      {"pyodide_version": PYODIDE_VERSION, "packages": packages,
       "pypi_wheels": wheels}, indent=1) + "\n")
  print(f"ok: {len(packages)} pyodide packages, {len(wheels)} PyPI wheels")


if __name__ == "__main__":
  main()
