"""Import aliasing so the unmodified JAX test suite can import shax as `jax`.

Targets:
  jax-upstream  no aliasing at all (real JAX, upstream test_util).
  jax-ported    real JAX, but `jax._src.test_util` -> `shax_harness.test_util`.
                Used to validate the port: results must match jax-upstream.
  shax          `jax` / `jax.<public>`   -> `shax.jax.<public>`
                `jax._src.test_util`     -> `shax_harness.test_util`
                `jax._src` / `jax._src.<x>` -> `shax_harness.jax_src.<x>` stand-ins,
                  except UPSTREAM_PURE names, which load vendor/jax's own
                  pure-Python source under the stand-in name
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

TARGETS = ("shax", "jax-ported", "jax-upstream")

# Pure-Python upstream modules (stdlib/absl/numpy + other jax._src names we
# provide) that are loaded straight from vendor/jax rather than reimplemented.
UPSTREAM_PURE = frozenset({
    "deprecations",
    "public_test_util",
    "test_loader",
    "test_warning_util",
})

PORTED = {"jax._src.test_util": "shax_harness.test_util"}
_STANDIN_PREFIX = "shax_harness.jax_src."


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
          f"shax harness: no stand-in for {fullname!r} (expected {target!r})",
          name=fullname)
    return importlib.util.spec_from_loader(fullname, _AliasLoader(target))

  def find_spec(self, fullname, path=None, target=None):
    if self.target == "jax-upstream":
      return None
    if fullname in PORTED:
      return self._alias(fullname, PORTED[fullname])
    if self.target != "shax":
      return None
    # Upstream pure-Python modules live under their stand-in name, because
    # `from jax._src import X` resolves X via the parent's real __name__
    # (shax_harness.jax_src), bypassing the `jax._src.X` alias.
    if fullname.startswith(_STANDIN_PREFIX):
      rest = fullname[len(_STANDIN_PREFIX):]
      if rest in UPSTREAM_PURE:
        return importlib.util.spec_from_file_location(
            fullname, self.upstream_src / f"{rest}.py")
      return None
    if fullname != "jax" and not fullname.startswith("jax."):
      return None

    if fullname == "jax._src":
      return self._alias(fullname, "shax_harness.jax_src")
    if fullname.startswith("jax._src."):
      rest = fullname[len("jax._src."):]
      return self._alias(fullname, _STANDIN_PREFIX + rest)
    return self._alias(fullname, "shax." + fullname)


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
    if test_util.__name__ != PORTED["jax._src.test_util"]:
      raise RuntimeError("shax harness: ported test_util was not picked up")
