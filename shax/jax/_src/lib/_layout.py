"""Pure-Python stand-ins for jaxlib's `xla_client.Layout` (an XLA layout) and
`PjRtLayout`, and the default CPU layouts shax reports for arrays and
executables.

On CPU, XLA's default layout is row-major (minor_to_major = ndim-1, ..., 0),
untiled, with the element size in bits set only for sub-byte dtypes (int4,
float4_e2m1fn, ...). PRNG key arrays report their physical (uint32) layout.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Layout:
  """`xla_client.Layout`: minor_to_major, tiling, element size in bits."""

  def __init__(self, minor_to_major: Sequence[int],
               tiling: Sequence[Sequence[int]] = (),
               element_size_in_bits: int = 0):
    self._minor_to_major = tuple(int(d) for d in minor_to_major)
    self._tiling = [tuple(int(t) for t in tile) for tile in tiling]
    self._element_size_in_bits = int(element_size_in_bits)

  def minor_to_major(self) -> tuple[int, ...]:
    return self._minor_to_major

  def tiling(self) -> list[tuple[int, ...]]:
    return list(self._tiling)

  def element_size_in_bits(self) -> int:
    return self._element_size_in_bits

  def _key(self):
    return (self._minor_to_major, tuple(self._tiling),
            self._element_size_in_bits)

  def __eq__(self, other):
    return isinstance(other, Layout) and self._key() == other._key()

  def __hash__(self):
    return hash(self._key())

  def to_string(self) -> str:
    s = "{" + ",".join(map(str, self._minor_to_major))
    attrs = "".join(f"T({','.join(map(str, t))})" for t in self._tiling)
    if self._element_size_in_bits:
      attrs += f"E({self._element_size_in_bits})"
    if attrs:
      s += ":" + attrs
    return s + "}"

  __str__ = to_string


class PjRtLayout:
  """`xla_client.PjRtLayout`: a wrapper around an XLA layout."""

  def __init__(self, xla_layout: Layout):
    self._layout = xla_layout

  def _xla_layout(self) -> Layout:
    return self._layout

  def __eq__(self, other):
    return isinstance(other, PjRtLayout) and self._layout == other._layout

  def __hash__(self):
    return hash(self._layout)

  def __str__(self):
    return str(self._layout)

  __repr__ = __str__

  def __reduce__(self):
    return (PjRtLayout, (self._layout,))


def default_pjrt_layout(aval) -> PjRtLayout | None:
  """XLA:CPU's default layout for values of type `aval` (None for tokens)."""
  from shax.jax._src import core, dtypes  # noqa: PLC0415
  if not isinstance(aval, core.ShapedArray):
    return None
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    aval = core.physical_aval(aval)
  dt = np.dtype(aval.dtype)
  bits = dtypes.itemsize_bits(dt)
  elem_bits = bits if bits < 8 else 0
  return PjRtLayout(Layout(tuple(range(aval.ndim - 1, -1, -1)), (), elem_bits))
