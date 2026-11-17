"""Pure-Python stand-ins for pieces of jaxlib's `_jax` extension module."""
from __future__ import annotations

import enum
import threading
import types
import weakref
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")


# --- config -----------------------------------------------------------------
# Mirrors jaxlib/config.cc: each Config has a global value and an optional
# thread-local override; `trace_context()` snapshots the values of configs
# that affect tracing (used in jit cache keys).

class _Unset:
  __slots__ = ()
  def __repr__(self):
    return "<unset>"

unset = _Unset()

_tls = threading.local()
_trace_context_configs: list["Config"] = []


def _locals() -> dict:
  d = getattr(_tls, "values", None)
  if d is None:
    d = _tls.values = {}
  return d


class Config(Generic[_T]):
  # No __slots__: upstream's State subclass declares its own slots and also
  # needs a __dict__-free layout only for speed, which shax doesn't need.

  def __init__(self, name: str, value: _T, *, include_in_jit_key: bool = False,
               include_in_trace_context: bool = False):
    self._cname = name
    self._global = value
    self._include_in_jit_key = include_in_jit_key
    self._include_in_trace_context = include_in_trace_context  # independent, as in config.cc
    if self._include_in_trace_context:
      _trace_context_configs.append(self)

  @property
  def value(self) -> _T:
    v = _locals().get(id(self), unset)
    return self._global if v is unset else v

  def get_global(self) -> _T:
    return self._global

  def set_global(self, value: _T) -> None:
    self._global = value

  def get_local(self) -> Any:
    return _locals().get(id(self), unset)

  def set_local(self, value: Any) -> None:
    if value is unset:
      _locals().pop(id(self), None)
    else:
      _locals()[id(self)] = value

  def swap_local(self, value: Any) -> Any:
    prev = self.get_local()
    self.set_local(value)
    return prev


def trace_context() -> tuple:
  return tuple(_hashable(c.value) for c in _trace_context_configs)


def trace_context_names() -> tuple:
  return tuple(c._cname for c in _trace_context_configs)


def _hashable(v):
  try:
    hash(v)
    return v
  except TypeError:
    return id(v)


config = types.SimpleNamespace(
    Config=Config, unset=unset, trace_context=trace_context,
    trace_context_names=trace_context_names)


# --- guard_lib (transfer / GC guards): stored but never enforced by shax -----

class TransferGuardLevel(enum.Enum):
  ALLOW = 0
  LOG = 1
  DISALLOW = 2
  LOG_EXPLICIT = 3
  DISALLOW_EXPLICIT = 4


class GarbageCollectionGuardLevel(enum.Enum):
  ALLOW = 0
  LOG = 1
  FATAL = 2


class _GuardState:
  def __init__(self):
    self.host_to_device = None
    self.device_to_device = None
    self.device_to_host = None
    self.garbage_collect_array = None
    self.explicit_device_put = False
    self.explicit_device_get = False


_guard_global = _GuardState()
_guard_tls = threading.local()


def _thread_local_guard_state() -> _GuardState:
  s = getattr(_guard_tls, "state", None)
  if s is None:
    s = _guard_tls.state = _GuardState()
  return s


guard_lib = types.SimpleNamespace(
    TransferGuardLevel=TransferGuardLevel,
    GarbageCollectionGuardLevel=GarbageCollectionGuardLevel,
    global_state=lambda: _guard_global,
    thread_local_state=_thread_local_guard_state,
    update_thread_guard_global_state=lambda val: None,
)


# --- jax_jit: the C++ fast path reads these; shax reads config directly -----

jax_jit = types.SimpleNamespace(
    set_enable_x64_state=lambda state: None,
    set_disable_jit_state=lambda state: None,
    set_post_hook_state=lambda state: None,
    # Filled in below from _pjit.py (import-order).
    ArgumentSignature=None,
    parse_arguments=None,
)


# --- runtime traceback plumbing (no runtime to send tracebacks to) ----------

class RuntimeTracebackMode(enum.Enum):
  OFF = 0
  ON = 1
  FULL = 2


def set_send_traceback_to_runtime_global(mode) -> None:
  pass


def set_send_traceback_to_runtime_thread_local(mode) -> None:
  pass


# --- traceback filtering -----------------------------------------------------

_exclude_paths: set[str] = set()


def add_exclude_path(path: str) -> None:
  _exclude_paths.add(path)


# --- value canonicalization: a port of jaxlib/py_values.cc CanonicalizeValue --
# Handlers are looked up by exact type, then along the MRO (and cached for the
# exact type). Python scalars become weakly-typed Typed{Int,Float,Complex};
# NumPy arrays/scalars become TypedNdArray with the dtype canonicalized for the
# x64 mode; typed values, bools and registered types pass through or use their
# handler.

_invalid_input_exception: type[Exception] | None = None
_valid_dtypes = None
_canonicalize_handlers: dict[type, Any] = {}
_typed: dict[str, Any] = {}


def _identity(x):
  return x


def set_invalid_input_exception(exc: type[Exception]) -> None:
  global _invalid_input_exception
  _invalid_input_exception = exc


def set_valid_dtypes(dtypes) -> None:
  global _valid_dtypes
  _valid_dtypes = dtypes


def register_canonicalize_value_handler(typ: type, handler) -> None:
  _canonicalize_handlers[typ] = _identity if handler is None else handler


def _x64() -> bool:
  from shax.jax._src import config  # noqa: PLC0415  (import cycle)
  return config.enable_x64.value


def _canonicalize_dtype(dtype):
  import numpy as np  # noqa: PLC0415
  if _x64():
    return dtype
  kind, size = dtype.kind, dtype.itemsize
  if kind == "i" and size == 8:
    return np.dtype("int32")
  if kind == "u" and size == 8:
    return np.dtype("uint32")
  if kind == "f" and size == 8:
    return np.dtype("float32")
  if kind == "c" and size == 16:
    return np.dtype("complex64")
  return dtype


def _check_valid_dtype(dtype) -> None:
  if _valid_dtypes is None:
    raise RuntimeError("valid_dtypes is not set")
  if dtype not in _valid_dtypes:
    raise TypeError(f"Dtype {dtype} is not a valid JAX array type. Only arrays "
                    "of numeric types are supported by JAX.")


def _canonicalize_int(x):
  import numpy as np  # noqa: PLC0415
  v = int(x)
  if _x64():
    if not -2**63 <= v < 2**63:
      raise OverflowError(f"Python int {x} too large to convert to int64")
    return _typed["int"](x, np.dtype("int64"))
  if not -2**31 <= v < 2**31:
    raise OverflowError(f"Python int {x} too large to convert to int32")
  return _typed["int"](x, np.dtype("int32"))


def _canonicalize_float(x):
  import numpy as np  # noqa: PLC0415
  return _typed["float"](x, np.dtype("float64" if _x64() else "float32"))


def _canonicalize_complex(x):
  import numpy as np  # noqa: PLC0415
  return _typed["complex"](x, np.dtype("complex128" if _x64() else "complex64"))


def _canonicalize_numpy_array(x):
  raw = x.dtype
  _check_valid_dtype(raw)
  canon = _canonicalize_dtype(raw)
  if canon == raw:
    return _typed["ndarray"](x)
  return _typed["ndarray"](x.astype(canon))


def _canonicalize_numpy_scalar(x):
  import numpy as np  # noqa: PLC0415
  raw = np.asarray(x).dtype
  _check_valid_dtype(raw)
  canon = _canonicalize_dtype(raw)
  if canon == raw:
    return _typed["ndarray"](np.asarray(x))
  return _typed["ndarray"](np.asarray(x, dtype=canon))


def set_typed_int_type(t) -> None:
  _typed["int"] = t
  register_canonicalize_value_handler(int, _canonicalize_int)
  register_canonicalize_value_handler(t, None)


def set_typed_float_type(t) -> None:
  _typed["float"] = t
  register_canonicalize_value_handler(float, _canonicalize_float)
  register_canonicalize_value_handler(t, None)


def set_typed_complex_type(t) -> None:
  _typed["complex"] = t
  register_canonicalize_value_handler(complex, _canonicalize_complex)
  register_canonicalize_value_handler(t, None)


class WeakKeyWeakValueCache:
  """jaxlib's WeakKeyWeakValueCache (weak_key_weak_value_cache.h): memoizes
  `fn(x)` by the identity of `x`, holding both `x` and the result weakly.

  jaxlib canonicalizes NumPy arrays through one of these, so the same array
  canonicalizes to the same TypedNdArray while both are alive. Tracing relies
  on that: constants are deduplicated by `id()` (`DynamicJaxprTrace.new_const`).
  """

  def __init__(self, fn):
    self._fn = fn
    self._entries: dict = {}  # id(key) -> (weakref(key), weakref(value))
    self._lock = threading.Lock()

  def _evict(self, key_id, key_ref):
    with self._lock:
      entry = self._entries.get(key_id)
      if entry is not None and entry[0] is key_ref:
        del self._entries[key_id]

  def __call__(self, x):
    key_id = id(x)
    with self._lock:
      entry = self._entries.get(key_id)
    if entry is not None and entry[0]() is x:
      value = entry[1]()
      if value is not None:
        return value
    value = self._fn(x)
    try:
      key_ref = weakref.ref(x, lambda r, i=key_id: self._evict(i, r))
      value_ref = weakref.ref(value)
    except TypeError:  # not weak-referenceable: don't cache
      return value
    with self._lock:
      self._entries[key_id] = (key_ref, value_ref)
    return value


def set_typed_ndarray_type(t) -> None:
  import numpy as np  # noqa: PLC0415
  _typed["ndarray"] = t
  register_canonicalize_value_handler(
      np.ndarray, WeakKeyWeakValueCache(_canonicalize_numpy_array))
  register_canonicalize_value_handler(t, None)
  register_canonicalize_value_handler(np.generic, _canonicalize_numpy_scalar)


# InitCanonicalizeValueHandlers: bools pass through (PyArray, i.e. ArrayImpl,
# registers itself when array.py is imported).
register_canonicalize_value_handler(bool, None)


def canonicalize_value(x):
  typ = type(x)
  handler = _canonicalize_handlers.get(typ)
  if handler is None:
    for base in typ.__mro__[1:]:
      handler = _canonicalize_handlers.get(base)
      if handler is not None:
        _canonicalize_handlers[typ] = handler
        break
  if handler is not None:
    return handler(x)
  if hasattr(x, "__jax_array__"):
    raise ValueError(
        "Triggering __jax_array__() during abstractification is no longer"
        " supported. To avoid this error, either explicitly convert your object"
        " using jax.numpy.array(), or register your object as a pytree.")
  if _invalid_input_exception is None:
    raise RuntimeError("invalid_input_exception not initialized")
  raise _invalid_input_exception(
      f"Argument '{x}' of type {typ} is not a valid JAX type.")


# --- PartitionSpec canonicalization (jaxlib partition_spec.cc) ---------------

_pspec_unconstrained = None


def set_pspec_unconstrained(value) -> None:
  global _pspec_unconstrained
  _pspec_unconstrained = value


def canonicalize_partition(partition):
  if isinstance(partition, list):
    partition = tuple(partition)
  if isinstance(partition, tuple):
    if len(partition) == 0:
      return None
    if len(partition) == 1:
      return partition[0]
  return partition


def canonicalize_partitions(partitions: tuple) -> tuple:
  return tuple(canonicalize_partition(p) for p in partitions)


class JaxRuntimeError(RuntimeError):
  pass


def batched_copy_to_host_async(arrays) -> None:
  # shax arrays already live on the host; only jaxlib's checks remain.
  for x in arrays:
    if x.is_deleted():
      raise JaxRuntimeError("INVALID_ARGUMENT: Array has been deleted.")


# --- tracebacks: implemented in xla_client.py; jaxlib exposes them on _jax ----
from shax.jax._src.lib.xla_client import (  # noqa: E402
    Frame, Traceback, TracebackScope, set_tracebacks_enabled, tracebacks_enabled)


# --- tracer class registration (C++ uses it for fast isinstance checks) -------

_tracer_class = None


def set_tracer_class(cls) -> None:
  global _tracer_class
  _tracer_class = cls


# --- unsupported jaxlib features ----------------------------------------------
from shax.jax._src.lib import _unsupported as _u  # noqa: E402

Client = None  # shax's client is xla_client.Client; isinstance checks use that
CompileOnlyPyClient = _u.cls("_jax.CompileOnlyPyClient")
CpuCollectives = _u.cls("_jax.CpuCollectives")
DistributedRuntimeClient = _u.cls("_jax.DistributedRuntimeClient")
DistributedRuntimeService = _u.cls("_jax.DistributedRuntimeService")
TransferServerInterfaceFactory = _u.cls("_jax.TransferServerInterfaceFactory")
for _name in ("create_preemption_sync_manager", "cuda_array_interface_to_buffer",
              "get_c_api_client", "get_distributed_runtime_client",
              "get_distributed_runtime_service", "hlo_module_cost_analysis",
              "initialize_pjrt_plugin", "json_to_pprof_profile",
              "make_gloo_tcp_collectives", "make_mpi_collectives",
              "make_transfer_server_interface_factory"):
  globals()[_name] = _u.function(f"_jax.{_name}")


def pjrt_plugin_loaded(name) -> bool:
  return False


def pjrt_plugin_initialized(name) -> bool:
  return False


class _UnsupportedNamespace(types.SimpleNamespace):
  """A jaxlib submodule shax lacks: every attribute is a function that raises
  NotImplementedError when called (so importing/referencing works)."""

  def __init__(self, qualname: str):
    super().__init__()
    object.__setattr__(self, "_qualname", qualname)

  def __getattr__(self, name):
    if name.startswith("__"):
      raise AttributeError(name)
    return _u.function(f"{self._qualname}.{name}")


# jaxlib's MLIR helpers (StableHLO serialization, shape refinement, ...).
mlir = _UnsupportedNamespace("jaxlib.mlir")
ifrt_programs = _UnsupportedNamespace("jaxlib.ifrt_programs")

# --- jit dispatch (pure Python; see _pjit.py) ----------------------------------
from shax.jax._src.lib._pjit import (  # noqa: E402
    PjitFunction, PjitFunctionCache, pjit, shard_args)
from shax.jax._src.lib import _pjit as _pjit_mod  # noqa: E402
jax_jit.ArgumentSignature = _pjit_mod.ArgumentSignature
jax_jit.parse_arguments = _pjit_mod.parse_arguments
ResultHandler = object  # only used in type annotations


# --- misc names jaxlib exposes on _jax -----------------------------------------
from shax.jax._src.lib.xla_client import Device, Memory  # noqa: E402

# DLPack: TODO(shax) -- NumPy implements __dlpack__, so this could be supported.
dlpack_version = (1, 3)
buffer_to_dlpack_managed_tensor = _u.function("_jax.buffer_to_dlpack_managed_tensor")
dlpack_managed_tensor_to_buffer = _u.function("_jax.dlpack_managed_tensor_to_buffer")

# SHAX: XLA's ApproxTopKReductionOutputSize, ported in the NumPy backend.
def approx_top_k_reduction_output_size(*args, **kwargs):
  from shax.jax._src.shax_backend.control import approx_top_k_reduction_output_size as f
  return f(*args, **kwargs)
