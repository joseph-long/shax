"""Pure-Python stand-in for `jaxlib._jax.ffi` (jaxlib/ffi.cc's Python bindings).

Only the types a Python FFI callback (`jax.experimental.buffer_callback`)
sees: `Buffer`, a view of a CPU buffer; `ExecutionContext`; and
`ExecutionStage`. shax has no XLA FFI, so nothing here registers kernels.
"""
from __future__ import annotations

import enum

import numpy as np

__all__ = ["Buffer", "ExecutionContext", "ExecutionStage"]


class Buffer:
  """A CPU buffer passed to an FFI callback (ffi.cc `PyFfiAnyBuffer`)."""

  __slots__ = ("_array", "_writeable")

  def __init__(self, array: np.ndarray, writeable: bool):
    self._array = array
    self._writeable = writeable

  @property
  def dtype(self) -> np.dtype:
    return self._array.dtype

  @property
  def ndim(self) -> int:
    return self._array.ndim

  @property
  def shape(self) -> tuple:
    return self._array.shape

  @property
  def writeable(self) -> bool:
    return self._writeable

  def __array__(self, dtype=None, copy=None):
    if dtype is not None:
      raise ValueError("dtype parameter is not supported by Buffer.__array__.")
    if copy is not None and copy:
      raise ValueError("Buffer.__array__ with copy=True is not supported.")
    # As in jaxlib, the array is a view of the buffer, writable even for
    # inputs (ffi.cc leaves the flag alone): the callback must not write them.
    return self._array

  @property
  def __cuda_array_interface__(self):
    from shax.jax._src.lib._jax import JaxRuntimeError  # noqa: PLC0415
    raise JaxRuntimeError(
        "UNIMPLEMENTED: Buffer.__cuda_array_interface__ is only supported on "
        "CUDA and ROCm.")

  def __dlpack__(self, stream=None, max_version=None, dl_device=None,
                 copy=None):
    del stream, dl_device
    if copy is not None and copy:
      raise ValueError("Buffer.__dlpack__ with copy=True is not supported.")
    if max_version is None:
      return self._array.__dlpack__()
    return self._array.__dlpack__(max_version=max_version)

  def __dlpack_device__(self) -> tuple:
    return (1, 0)  # kDLCPU, device 0


class ExecutionStage(enum.Enum):
  INSTANTIATE = 0
  PREPARE = 1
  INITIALIZE = 2
  EXECUTE = 3


class ExecutionContext:
  """The FFI call context (ffi.cc `PyFfiContext`); always the EXECUTE stage."""

  __slots__ = ("_stage",)

  def __init__(self, stage: ExecutionStage = ExecutionStage.EXECUTE):
    self._stage = stage

  @property
  def stage(self) -> ExecutionStage:
    return self._stage

  @property
  def stream(self) -> int:
    from shax.jax._src.lib._jax import JaxRuntimeError  # noqa: PLC0415
    raise JaxRuntimeError("INTERNAL: XLA FFI GPU context is not available")
