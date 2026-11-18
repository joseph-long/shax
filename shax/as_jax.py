"""Make `import jax` (and `jax.<anything>`) resolve to shax.

  import shax.as_jax  # noqa: F401  (or: from shax.as_jax import install; install())
  import jax, optax   # jax is shax.jax

Installs a `sys.meta_path` finder that aliases `jax` / `jax.X` to the already
importable `shax.jax` / `shax.jax.X`, so both names give the *same* module
objects (one config, one set of caches). The `jax` shim wheel
(packaging/jax-shim) calls this from its `jax/__init__.py`, which is how
libraries that `import jax` (optax, chex, ...) find shax in Pyodide.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

__all__ = ["install"]


class _AliasLoader(importlib.abc.Loader):

  def __init__(self, target: str):
    self.target = target
    self._orig_spec = None

  def create_module(self, spec):
    module = importlib.import_module(self.target)
    self._orig_spec = module.__spec__
    return module

  def exec_module(self, module):
    # The import system sets __spec__ to the alias's spec; restore the real
    # one so reload/pickling/relative imports keep the module's identity.
    module.__spec__ = self._orig_spec


class _JaxAliasFinder(importlib.abc.MetaPathFinder):

  def find_spec(self, fullname, path=None, target=None):
    if fullname != "jax" and not fullname.startswith("jax."):
      return None
    real = "shax." + fullname
    if importlib.util.find_spec(real) is None:
      return None  # let the import fail normally (ModuleNotFoundError)
    return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


def install() -> None:
  """Alias `jax` to `shax.jax`. Idempotent. Refuses if real jax is loaded."""
  jax = sys.modules.get("jax")
  if jax is not None and not (getattr(jax, "__shax__", False)
                              or getattr(jax, "__shax_shim__", False)):
    raise RuntimeError("shax: the real `jax` is already imported")
  if not any(isinstance(f, _JaxAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _JaxAliasFinder())
  import shax.jax  # noqa: PLC0415
  sys.modules["jax"] = shax.jax


install()
