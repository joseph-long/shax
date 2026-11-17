"""pytest plugin: run the vendored JAX test suite against shax (or real JAX).

Loaded with `-p shax_harness.plugin`; normally invoked through scripts/jaxtest.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tomllib

import pytest

from shax_harness import aliases

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPO_ROOT / "vendor" / "jax"
SKIPS_FILE = REPO_ROOT / "harness" / "skips.toml"

_OUTCOME_RANK = {"passed": 0, "skipped": 1, "xfailed": 1, "xpassed": 2,
                 "failed": 3, "error": 4}


def pytest_addoption(parser):
  group = parser.getgroup("shax")
  group.addoption("--shax-target", choices=aliases.TARGETS,
                  default=os.environ.get("SHAX_TARGET", "shax"),
                  help="what `import jax` should resolve to")
  group.addoption("--shax-report", default=None,
                  help="write per-test outcomes as JSON to this path")


def pytest_configure(config):
  target = config.getoption("shax_target")
  aliases.install(target, UPSTREAM_ROOT)
  config._shax_target = target
  config._shax_skips = _load_skips(target)
  config._shax_results = {}
  config._shax_collection_errors = {}


def _load_skips(target):
  if not SKIPS_FILE.exists():
    return []
  data = tomllib.loads(SKIPS_FILE.read_text())
  # Inside Pyodide the target is still "shax" (same code); entries for
  # "pyodide" cover what WASM can't do (threads, >2 GiB, 32-bit C long).
  active = {target} | ({"pyodide"} if sys.platform == "emscripten" else set())
  out = []
  for entry in data.get("skip", []):
    if active & set(entry.get("targets", ["shax"])):
      out.append((entry["nodeid"], entry["reason"], entry.get("exact", False)))
  return out


def pytest_collection_modifyitems(config, items):
  skips = config._shax_skips
  if not skips:
    return
  for item in items:
    for prefix, reason, exact in skips:
      if item.nodeid == prefix if exact else item.nodeid.startswith(prefix):
        item.add_marker(pytest.mark.skip(reason=f"harness-skip: {reason}"))
        break


def _message(report) -> str:
  if report.skipped:
    lr = report.longrepr
    msg = lr[2] if isinstance(lr, tuple) else str(lr)
    return msg.removeprefix("Skipped: ")
  crash = getattr(report.longrepr, "reprcrash", None)
  if crash is not None:
    msg = crash.message
  else:
    # e.g. collection ImportErrors: use the last "E   ..." exception line.
    text = report.longreprtext
    elines = [l[1:].strip() for l in text.splitlines() if l.startswith("E ")]
    msg = elines[-1] if elines else text
  return msg.strip().splitlines()[0][:300] if msg.strip() else ""


def pytest_runtest_logreport(report):
  config = _CONFIG
  if config is None:
    return
  if report.when == "call":
    outcome = report.outcome
    if hasattr(report, "wasxfail"):
      outcome = "xfailed" if report.skipped else "xpassed"
  elif report.failed:
    outcome = "error"
  elif report.skipped:
    outcome = "skipped"
  else:
    return
  prev = config._shax_results.get(report.nodeid)
  if prev is None or _OUTCOME_RANK[outcome] > _OUTCOME_RANK[prev["outcome"]]:
    entry = {"outcome": outcome}
    if outcome != "passed":
      entry["message"] = _message(report)
    config._shax_results[report.nodeid] = entry


def pytest_collectreport(report):
  if _CONFIG is not None and report.failed:
    _CONFIG._shax_collection_errors[report.nodeid] = _message(report)


_CONFIG = None


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
  global _CONFIG
  _CONFIG = session.config


def pytest_sessionfinish(session):
  config = session.config
  path = config.getoption("shax_report")
  if not path or hasattr(config, "workerinput"):  # xdist workers don't write
    return
  env_keys = ("JAX_NUM_GENERATED_CASES", "JAX_SKIP_SLOW_TESTS",
              "JAX_ENABLE_X64", "JAX_PLATFORMS")
  report = {
      "target": config._shax_target,
      "env": {k: os.environ.get(k) for k in env_keys},
      "collection_errors": dict(sorted(config._shax_collection_errors.items())),
      "results": dict(sorted(config._shax_results.items())),
  }
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(report, indent=1) + "\n")
