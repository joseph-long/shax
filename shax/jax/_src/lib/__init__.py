"""Pure-Python stand-in for `jax._src.lib` (the jaxlib C++/XLA bindings).

Only the pieces that shax's ported modules and JAX's tests need are provided,
with the same names and (as far as JAX relies on them) the same semantics.
"""
import re
import types

from shax.jax._src.lib import _jax
from shax.jax._src.lib import utils
from shax.jax._src.lib import weakref_lru_cache
from shax.jax._src.lib._jax import guard_lib, jax_jit
from shax.jax._src.lib import xla_client
from shax.jax._src.lib import _profiler
from shax.jax._src.lib import ifrt_proxy
from shax.jax._src.lib.xla_client import Device
from shax.jax._src.lib import hlo, jax_mlir_ext, _profile_data  # MLIR/profiler stubs

# Optional jaxlib components that shax doesn't have. Upstream checks these for
# None before use.
# jaxlib's _xla (XLA compiler hooks such as HLO-module transforms): calling
# anything raises NotImplementedError.
_xla = _jax._UnsupportedNamespace("jaxlib._xla")
lapack = None
gpu_linalg = gpu_solver = gpu_sparse = gpu_prng = gpu_rnn = gpu_triton = None
cuda_versions = None
tpu = None
jaxlib = None
cuda_path = None
# jaxlib's FFI types for Python callbacks (no FFI kernels in shax).
from shax.jax._src.lib import ffi  # noqa: E402


class _NoKernels(types.SimpleNamespace):
  """A jaxlib kernel module with no custom-call targets to register."""

  def registrations(self):
    return {}


cpu_sparse = _NoKernels()

# SHAX: copied from upstream jax/_src/lib/__init__.py (shax pins its own
# version below rather than reading jaxlib's).
# Checks the jaxlib version before importing anything else.
# Returns the jaxlib version string.
def check_jaxlib_version(jax_version: str, jaxlib_version: str,
                         minimum_jaxlib_version: str) -> tuple[int, ...]:
  # Regex to match a dotted version prefix 0.1.23.456.789 of a PEP440 version.
  # PEP440 allows a number of non-numeric suffixes, which we allow also.
  # We currently do not allow an epoch.
  version_regex = re.compile(r"[0-9]+(?:\.[0-9]+)*")
  def _parse_version(v: str) -> tuple[int, ...]:
    m = version_regex.match(v)
    if m is None:
      raise ValueError(f"Unable to parse jaxlib version '{v}'")
    return tuple(int(x) for x in m.group(0).split('.'))

  _jax_version = _parse_version(jax_version)
  _minimum_jaxlib_version = _parse_version(minimum_jaxlib_version)
  _jaxlib_version = _parse_version(jaxlib_version)

  if _jaxlib_version < _minimum_jaxlib_version:
    msg = (f'jaxlib is version {jaxlib_version}, but this version '
           f'of jax requires version >= {minimum_jaxlib_version}.')
    raise RuntimeError(msg)

  if _jaxlib_version > _jax_version:
    raise RuntimeError(
        f'jaxlib version {jaxlib_version} is newer than and '
        f'incompatible with jax version {jax_version}. Please '
        'update your jax and/or jaxlib packages.')
  return _jaxlib_version


# shax emulates the jaxlib that matches the JAX version it tracks.
version_str = "0.12.0.1"
version = (0, 12, 0, 1)
# Match the real jaxlib 0.12.0.1 values so upstream version gates take the
# same code paths they do under real JAX.
jaxlib_extension_version = 505
ifrt_version = 76

__all__ = ["_jax", "guard_lib", "jax_jit", "xla_client", "utils", "weakref_lru_cache", "Device",
           "version", "version_str", "jaxlib_extension_version", "ifrt_version"]
