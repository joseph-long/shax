"""Pure-Python stand-in for `jax._src.lib` (the jaxlib C++/XLA bindings).

Only the pieces that shax's ported modules and JAX's tests need are provided,
with the same names and (as far as JAX relies on them) the same semantics.
"""
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

# shax emulates the jaxlib that matches the JAX version it tracks.
version_str = "0.12.0.1"
version = (0, 12, 0, 1)
# Match the real jaxlib 0.12.0.1 values so upstream version gates take the
# same code paths they do under real JAX.
jaxlib_extension_version = 505
ifrt_version = 76

__all__ = ["_jax", "guard_lib", "jax_jit", "xla_client", "utils", "weakref_lru_cache", "Device",
           "version", "version_str", "jaxlib_extension_version", "ifrt_version"]
