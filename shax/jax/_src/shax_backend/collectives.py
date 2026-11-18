"""Eager `shard_map` (and so `pmap`) on shax's single device.

Upstream evaluates an eager `shard_map` with `ShardMapTrace`, which runs each
primitive as `jit(shard_map(prim))` and relies on XLA to compile that. shax
evaluates jaxprs eagerly, so that path would re-enter itself forever. shax
only ever has one device, so every mesh has size 1. Each shard is then the
whole value, and the body can run directly on the global values while the
mesh axes are bound (size 1) in the axis env.

Collectives over size-1 axes are trivial (psum = identity, axis_index = 0,
all_gather adds a unit dimension, ...). Their upstream eager impls raise
("must be called under shard_map"), which is right *outside* a shard_map. So
the size-1 impls below wrap the upstream ones and only apply inside
`eval_body` (called from the `# SHAX:` path in shard_map._shard_map_impl).
"""
from __future__ import annotations

import contextlib
import threading

from shax.jax._src import core
from shax.jax._src import config
from shax.jax._src.lax import lax
from shax.jax._src.lax import parallel

_state = threading.local()


def _active() -> bool:
  return getattr(_state, "depth", 0) > 0


@contextlib.contextmanager
def _body_context():
  _state.depth = getattr(_state, "depth", 0) + 1
  try:
    yield
  finally:
    _state.depth -= 1


def eval_body(trace, jaxpr, consts, args):
  """Evaluate a traced shard_map body jaxpr on a size-1 mesh.

  The caller binds the mesh axes (axis env, manual abstract mesh).
  """
  # The body's vars carry manual-axis (vma) types; the global values don't,
  # so eval_jaxpr's enable_checks typecheck doesn't apply here.
  args = [_unmeshed(a) for a in args]
  with (_body_context(), core.set_current_trace(trace),
        config.enable_checks(False)):
    return core.eval_jaxpr(jaxpr, consts, *args)


def _unmeshed(x):
  """`x` without its mesh sharding: on a size-1 mesh the shard is the value.

  Values sharded on an Explicit mesh can't enter the manual-mesh body
  ("Closing over inputs to shard_map ... `Explicit` axes"); the shard each
  device sees has no mesh type.
  """
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  from shax.jax._src.lib import xla_client  # noqa: PLC0415
  if not isinstance(x, ArrayImpl):
    return x
  aval = x.aval
  if not isinstance(aval, core.ShapedArray) or aval.sharding.mesh.empty:
    return x
  return ArrayImpl(
      core.ShapedArray(aval.shape, aval.dtype, weak_type=aval.weak_type),
      xla_client.SingleDeviceSharding(x.sharding._device_assignment[0]),
      list(x._arrays), committed=x.committed)


def _override(prim, size1_impl):
  upstream = prim.impl

  def impl(*args, **params):
    if _active():
      return size1_impl(*args, **params)
    return upstream(*args, **params)

  prim.def_impl(impl)


def _named(axes):
  return tuple(a for a in axes if not isinstance(a, int))


def _positional(axes):
  return tuple(a for a in axes if isinstance(a, int))


def _allreduce(reducer):
  def impl(arg, *, axes, axis_index_groups=None):
    pos = _positional(axes)
    return reducer(arg, pos) if pos else arg
  return impl


_override(parallel.psum_p, _allreduce(lax.reduce_sum))
_override(parallel.pmax_p, _allreduce(lax.reduce_max))
_override(parallel.pmin_p, _allreduce(lax.reduce_min))
_override(parallel.psum_invariant_p, _allreduce(lax.reduce_sum))
for _p in (parallel.unreduced_psum_p, parallel.unreduced_pmax_p,
           parallel.unreduced_pmin_p):
  _override(_p, lambda x, **_: x)

# Type-level casts between varying/invariant/reduced: identities on values.
for _p in (core.pvary_p, parallel.preduced_p, parallel.vary_unreduced_cast_p,
           core.reduced_vary_cast_p):
  _override(_p, lambda x, **_: x)


def _ppermute(x, *, axis_name, perm):
  # The only permutation of one device is [(0, 0)]; with no pairs the result
  # is zeros (devices that receive nothing get zeros).
  return x if perm else lax.full_like(x, 0)


_override(parallel.ppermute_p, _ppermute)
_override(parallel.pbroadcast_p, lambda x, **_: x)


def _all_gather(x, *, all_gather_dimension, axis_size, tiled, **_):
  assert axis_size == 1, axis_size
  return x if tiled else lax.expand_dims(x, (all_gather_dimension,))


_override(parallel.all_gather_p, _all_gather)
_override(parallel.all_gather_invariant_p, _all_gather)
_override(parallel.all_gather_reduced_p, _all_gather)


def _reduce_scatter(x, *, scatter_dimension, axis_size, tiled, **_):
  assert axis_size == 1, axis_size
  return x if tiled else lax.squeeze(x, (scatter_dimension,))


_override(parallel.reduce_scatter_p, _reduce_scatter)
_override(parallel.unreduced_reduce_scatter_p, _reduce_scatter)


def _all_to_all(x, *, split_axis, concat_axis, axis_name, axis_index_groups,
                tiled):
  # Splitting into one chunk and concatenating one chunk is the identity.
  return x


_override(parallel.all_to_all_p, _all_to_all)
