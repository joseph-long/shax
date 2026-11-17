"""Data-placement primitives on shax's single device.

`device_put` and `reshard` change where/how a value is laid out (device,
sharding, memory kind). shax has one device and one memory, so on values they
are identities; the result's type comes from the primitive's abstract eval.
"""
from __future__ import annotations

import numpy as np

from shax.jax._src import dispatch
from shax.jax._src import pjit
from shax.jax._src.shax_backend import register


@register(dispatch.device_put_p)
def _device_put(*xs, devices, srcs, copy_semantics):
  return [np.array(x) for x in xs]


@register(pjit.reshard_p)
def _reshard(x, *, dst_sharding, concrete_mesh):
  return np.array(x)
