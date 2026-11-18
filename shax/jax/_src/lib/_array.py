"""Pure-Python stand-in for jaxlib's C++ `ArrayImpl` core.

Upstream `jax/_src/array.py` defines `ArrayImpl` in Python and merges it onto
jaxlib's C++ class with `use_cpp_class(xc.ArrayImpl)`; methods marked
`@use_cpp_method()` come from C++. This class supplies those methods, storing
the value as one read-only NumPy array on shax's single CPU device.
"""
from __future__ import annotations

import gc
import inspect
import itertools
import os
import sys
import weakref

import numpy as np

from shax.jax._src import basearray


# Every ArrayImpl, for Client.live_arrays() (jaxlib tracks these per client).
# Arrays are unhashable, so key by creation order.
_LIVE: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_next_id = itertools.count()


def live_arrays() -> list:
  return [a for a in list(_LIVE.values()) if not a._deleted]


# Element types XLA's PEP3118FormatDescriptorForPrimitiveType knows.
_PEP3118_DTYPES = frozenset(map(np.dtype, (
    np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
    np.uint32, np.uint64, np.float16, np.float32, np.float64, np.complex64,
    np.complex128)))


def _xla_type_name(dtype: np.dtype) -> str:
  """XLA's PrimitiveType name for an ml_dtypes type (bfloat16 -> BF16)."""
  name = dtype.name
  if name == "bfloat16":
    return "BF16"
  for prefix, xla in (("float", "F"), ("uint", "U"), ("int", "S")):
    if name.startswith(prefix):
      bits, _, rest = name[len(prefix):].partition("_")
      return f"{xla}{bits}{rest.upper()}"
  return name.upper()


# py_array.cc PyArray_tp_finalize: arrays freed by the cycle collector
# (rather than reference counting) are reported under
# jax_array_garbage_collection_guard. A gc callback marks collections.
_collecting = False


def _gc_callback(phase, info):
  global _collecting
  _collecting = phase == "start"


gc.callbacks.append(_gc_callback)


def _report_gc_deleted(arr) -> None:
  from shax.jax._src.lib import _jax  # noqa: PLC0415
  level = _jax._thread_local_guard_state().garbage_collect_array
  if level is None:
    level = _jax._guard_global.garbage_collect_array
  if level is None or level is _jax.GarbageCollectionGuardLevel.ALLOW:
    return
  tb = arr.__dict__.get("_traceback")
  msg = ("`jax.Array` was deleted by the Python garbage collector instead of "
         "reference counting. Break the reference cycle that delays the "
         "deletion of this `jax.Array` to avoid hogging memory. Traceback: \n"
         + (str(tb) if tb is not None else "not available"))
  if level is _jax.GarbageCollectionGuardLevel.FATAL:
    sys.stderr.write(f"Fatal Python error: {msg}\n")
    sys.stderr.flush()
    os.abort()
  sys.stderr.write(f"RuntimeError: {msg}\n")


def _get_traceback():
  from shax.jax._src.lib.xla_client import Traceback  # noqa: PLC0415
  global _get_traceback
  _get_traceback = Traceback.get_traceback
  return Traceback.get_traceback()


def _invalid_argument(msg: str):
  from shax.jax._src.lib._jax import JaxRuntimeError  # noqa: PLC0415
  return JaxRuntimeError(f"INVALID_ARGUMENT: {msg}")


class ArrayImpl(basearray.Array):

  def __init__(self, aval, sharding, arrays, committed: bool,
               _skip_checks: bool = False):
    if not isinstance(arrays, (list, tuple)):
      # nanobind: `arrays` binds to a std::vector; upstream turns this into
      # its own error message.
      raise TypeError(
          f"ArrayImpl(): incompatible `arrays` argument: {type(arrays)}")
    self._aval = aval
    self._sharding = sharding
    self._committed = committed
    if len(arrays) != 1:
      raise ValueError(
          f"shax arrays live on a single device; got {len(arrays)} shards")
    (buf,) = arrays
    if isinstance(buf, ArrayImpl):
      buf = buf._value
    buf = np.asarray(buf)
    if buf.dtype != aval.dtype:
      buf = buf.astype(aval.dtype)
    if buf.flags.writeable:
      # Copy rather than freeze in place: the caller may still own `buf`.
      buf = buf.copy()
      buf.flags.writeable = False
    self._npy_value = buf
    self._deleted = False
    # jaxlib records where each array was created (its IFRT user context).
    self._traceback = _get_traceback()
    _LIVE[next(_next_id)] = self

  def __del__(self):
    if _collecting:
      _report_gc_deleted(self)

  @property
  def traceback(self):
    return None if self._deleted else self._traceback

  @property
  def aval(self):
    # Read-only, as in jaxlib (jaxlib_extension_version >= 490).
    return self._aval

  def platform(self) -> str:
    if self._deleted:
      raise _invalid_argument("platform() called on deleted or donated buffer")
    return "cpu"

  def is_ready(self) -> bool:
    self._check_if_deleted()
    return True

  def __buffer__(self, flags: int) -> memoryview:
    """py_array.cc PyArray_bf_getbuffer: a read-only view of the CPU buffer."""
    if self._deleted:
      raise BufferError("INVALID_ARGUMENT: Array has been deleted.")
    buf = self._npy_value
    if buf.dtype not in _PEP3118_DTYPES:
      raise BufferError(
          f"INVALID_ARGUMENT: Buffers of type {_xla_type_name(buf.dtype)} are "
          "not supported by the Python buffer protocol.")
    if flags & inspect.BufferFlags.WRITABLE:
      raise BufferError("INVALID_ARGUMENT: XLA buffers are read-only.")
    return memoryview(buf)

  # jaxlib exposes the per-device buffers as single-device arrays; with one
  # device there's one, sharing this array's buffer.
  @property
  def _arrays(self):
    if self._deleted:
      return None  # py_array.cc PyArray::arrays
    return [self._fully_replicated_shard()]

  def _fully_replicated_shard(self):
    if self._deleted:
      raise ValueError(
          "FullyReplicatedShard() called on deleted or donated buffer")
    shard = self.__dict__.get("_s_fully_replicated")
    if shard is None:
      from shax.jax._src.lib import xla_client  # noqa: PLC0415
      sharding = self._sharding
      if isinstance(sharding, xla_client.SingleDeviceSharding):
        shard = self
      else:
        (device,) = sharding._internal_device_list
        shard = type(self)(
            self.aval,
            xla_client.SingleDeviceSharding(
                device, memory_kind=sharding.memory_kind),
            [self._npy_value], self._committed)
      self.__dict__["_s_fully_replicated"] = shard
    return shard

  def delete(self):
    self._deleted = True
    self._npy_value = None

  def is_deleted(self) -> bool:
    return self._deleted

  def _replace_with(self, o):
    """py_array.cc ReplaceWithAlias: make this array alias `o`'s buffer.
    Used to write a jit's updated `Ref` values back into the Ref's array."""
    if o is self:
      raise _invalid_argument("Unable to replace an Array with itself.")
    for what, a, b in (("dtype", self.aval.dtype, o.aval.dtype),
                       ("shape", self.aval.shape, o.aval.shape),
                       ("sharding", self._sharding, o._sharding),
                       ("committed", self._committed, o._committed),
                       ("weak_type", self.aval.weak_type, o.aval.weak_type)):
      if a != b:
        raise _invalid_argument(
            f"Unable to replace an Array with an Array of different {what}.")
    if self._deleted or o._deleted:
      raise _invalid_argument("Unable to replace an Array when one or both arrays "
                       "have been deleted.")
    self._npy_value = o._npy_value  # read-only, so sharing it is safe
    self.__dict__.pop("_s_fully_replicated", None)

  @property
  def _pjrt_layout(self):
    self._check_if_deleted()
    from shax.jax._src.lib._layout import default_pjrt_layout  # noqa: PLC0415
    return default_pjrt_layout(self.aval)

  def block_until_ready(self):
    self._check_if_deleted()
    return self

  def _single_device_array_to_np_array_did_copy(self):
    return self._to_np_array_did_copy()

  def _to_np_array_did_copy(self):
    self._check_if_deleted()
    if self.__dict__.get("_s_token_buffer"):
      # jaxlib: tokens are XLA token-typed buffers with no host value.
      raise RuntimeError(
          "Cannot convert a token-shape buffer to a numpy array.")
    return self._npy_value, False

  def _copy_single_device_array_to_host_async(self):
    pass

  def unsafe_buffer_pointer(self) -> int:
    self._check_if_deleted()
    return self._npy_value.ctypes.data

  @property
  def __cuda_array_interface__(self):
    raise AttributeError("__cuda_array_interface__ is only defined for GPU buffers.")

  def on_device_size_in_bytes(self) -> int:
    self._check_if_deleted()
    return self._npy_value.nbytes
