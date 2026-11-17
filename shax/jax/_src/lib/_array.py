"""Pure-Python stand-in for jaxlib's C++ `ArrayImpl` core.

Upstream `jax/_src/array.py` defines `ArrayImpl` in Python and merges it onto
jaxlib's C++ class with `use_cpp_class(xc.ArrayImpl)`; methods marked
`@use_cpp_method()` come from C++. This class supplies those methods, storing
the value as one read-only NumPy array on shax's single CPU device.
"""
from __future__ import annotations

import itertools
import weakref

import numpy as np

from shax.jax._src import basearray


# Every ArrayImpl, for Client.live_arrays() (jaxlib tracks these per client).
# Arrays are unhashable, so key by creation order.
_LIVE: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_next_id = itertools.count()


def live_arrays() -> list:
  return [a for a in list(_LIVE.values()) if not a._deleted]


class ArrayImpl(basearray.Array):

  def __init__(self, aval, sharding, arrays, committed: bool,
               _skip_checks: bool = False):
    self.aval = aval
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
    _LIVE[next(_next_id)] = self

  # jaxlib exposes the per-device buffers as single-device arrays; with one
  # device there's one, sharing this array's buffer.
  @property
  def _arrays(self):
    self._check_if_deleted()
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
      raise ValueError("Unable to replace an Array with itself.")
    for what, a, b in (("dtype", self.aval.dtype, o.aval.dtype),
                       ("shape", self.aval.shape, o.aval.shape),
                       ("sharding", self._sharding, o._sharding),
                       ("committed", self._committed, o._committed),
                       ("weak_type", self.aval.weak_type, o.aval.weak_type)):
      if a != b:
        raise ValueError(
            f"Unable to replace an Array with an Array of different {what}.")
    if self._deleted or o._deleted:
      raise ValueError("Unable to replace an Array when one or both arrays "
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
    raise AttributeError("__cuda_array_interface__ is only defined for GPU arrays")

  def on_device_size_in_bytes(self) -> int:
    self._check_if_deleted()
    return self._npy_value.nbytes
