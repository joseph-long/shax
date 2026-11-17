"""Import aliasing so the unmodified JAX test suite can import shax as `jax`.

Targets:
  jax-upstream  no aliasing at all (real JAX, upstream test_util).
  jax-ported    real JAX, but `jax._src.test_util` -> `shax_harness.test_util`.
                Used to validate the port: results must match jax-upstream.
  shax          `jax` / `jax.<anything>` -> `shax.jax.<anything>` (shax mirrors
                JAX's module layout, including `_src`), except the test-only
                modules in TEST_ONLY, which are injected as
                `shax.jax._src.<name>` (shax itself never ships them).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

TARGETS = ("shax", "jax-ported", "jax-upstream")

PORTED_TEST_UTIL = "shax_harness.test_util"

# Test-only `jax._src` modules. Value: a module name to alias to, or None to
# load vendor/jax's own pure-Python source file for it.
TEST_ONLY: dict[str, str | None] = {
    "test_util": PORTED_TEST_UTIL,
    "test_loader": None,
    "test_warning_util": None,
    "lax_reference": None,
    "internal_test_util": None,  # package; submodules load from vendor too
}


class HarnessImportError(ModuleNotFoundError):
  pass


class _AliasLoader(importlib.abc.Loader):
  """Make `alias` refer to the already-importable module `target`."""

  def __init__(self, target: str):
    self.target = target
    self._orig_spec = None

  def create_module(self, spec):
    module = importlib.import_module(self.target)
    self._orig_spec = module.__spec__
    return module

  def exec_module(self, module):
    # The import system overwrites __spec__ with the alias spec; restore it so
    # the module's own identity (reload, pickling, relative imports) is intact.
    module.__spec__ = self._orig_spec


class AliasFinder(importlib.abc.MetaPathFinder):

  def __init__(self, target: str, upstream_root: Path):
    if target not in TARGETS:
      raise ValueError(f"unknown target {target!r}")
    self.target = target
    self.upstream_src = upstream_root / "jax" / "_src"

  def _alias(self, fullname: str, target: str):
    if importlib.util.find_spec(target) is None:
      raise HarnessImportError(
          f"shax harness: shax does not provide {fullname!r} (expected {target!r})",
          name=fullname)
    return importlib.util.spec_from_loader(fullname, _AliasLoader(target))

  def find_spec(self, fullname, path=None, target=None):
    if self.target == "jax-upstream":
      return None
    if self.target == "jax-ported":
      if fullname == "jax._src.test_util":
        return self._alias(fullname, PORTED_TEST_UTIL)
      return None

    # shax target. `from jax._src import X` looks X up as `shax.jax._src.X`
    # (the parent's real name), so test-only modules are matched under both.
    for prefix in ("jax._src.", "shax.jax._src."):
      if fullname.startswith(prefix):
        leaf = fullname[len(prefix):]
        if leaf.split(".")[0] in TEST_ONLY:
          if prefix == "jax._src.":
            return self._alias(fullname, "shax." + fullname)
          if TEST_ONLY.get(leaf) is not None:
            return self._alias(fullname, TEST_ONLY[leaf])
          src = self.upstream_src.joinpath(*leaf.split("."))
          if src.is_dir():
            return importlib.util.spec_from_file_location(
                fullname, src / "__init__.py",
                submodule_search_locations=[str(src)])
          return importlib.util.spec_from_file_location(
              fullname, src.with_suffix(".py"))
    if fullname == "jax" or fullname.startswith("jax."):
      return self._alias(fullname, "shax." + fullname)
    return None


def install(target: str, upstream_root: Path) -> None:
  """Install the finder for `target`. Must run before anything imports jax."""
  _preflight(target)
  if target != "jax-upstream":
    sys.meta_path.insert(0, AliasFinder(target, upstream_root))
  _postflight(target, upstream_root)


def _preflight(target: str) -> None:
  if "jax" in sys.modules:
    raise RuntimeError("shax harness: `jax` was imported before the harness "
                       "could install its import hooks")
  if target == "shax":
    for real in ("jax", "jaxlib"):
      if importlib.util.find_spec(real) is not None:
        raise RuntimeError(
            f"shax harness: real `{real}` is importable in this environment; "
            "run the shax target in an env without jax/jaxlib (.venv)")


def _postflight(target: str, upstream_root: Path) -> None:
  import jax  # noqa: PLC0415
  is_shax = bool(getattr(jax, "__shax__", False))
  if is_shax != (target == "shax"):
    raise RuntimeError(f"shax harness: target {target!r} but `jax` is {jax!r}")
  if not is_shax:
    jax_file = Path(jax.__file__).resolve()
    if upstream_root.resolve() in jax_file.parents:
      raise RuntimeError(
          f"shax harness: `jax` resolved to the vendored source tree "
          f"({jax_file}); run pytest from the repo root, not vendor/jax")
  if target == "jax-ported":
    from jax._src import test_util  # noqa: PLC0415
    if test_util.__name__ != PORTED_TEST_UTIL:
      raise RuntimeError("shax harness: ported test_util was not picked up")
