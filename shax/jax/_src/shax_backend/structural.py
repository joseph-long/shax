"""NumPy implementations of shape, structural, indexing and sorting primitives.

See shax_backend/__init__.py for the calling convention. Semantics follow XLA
on CPU (https://openxla.org/xla/operation_semantics): gathers clamp or fill,
scatters clamp or drop whole out-of-bounds windows, dynamic slices clamp.
"""
from __future__ import annotations

import numpy as np

from shax.jax._src import core
from shax.jax._src import dtypes
from shax.jax._src.lax import lax
from shax.jax._src.lax import slicing
from shax.jax._src.shax_backend import register, to_numpy

GatherScatterMode = slicing.GatherScatterMode

_NATIVE_FLOATS = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))
_NATIVE_INTS = tuple(np.dtype(t) for t in (np.int8, np.int16, np.int32, np.int64,
                                           np.uint8, np.uint16, np.uint32, np.uint64))


def _is_float(dt) -> bool:
  return dtypes.issubdtype(dt, np.floating)


def _is_complex(dt) -> bool:
  return dtypes.issubdtype(dt, np.complexfloating)


def _compute_dtype(x: np.ndarray) -> np.ndarray:
  """Upcast ml_dtypes values (bfloat16, float8_*, int4, ...) to native types."""
  dt = x.dtype
  if _is_float(dt) and dt not in _NATIVE_FLOATS:
    return x.astype(np.float32)
  if dtypes.issubdtype(dt, np.integer) and dt not in _NATIVE_INTS:
    return x.astype(np.int32 if dtypes.issubdtype(dt, np.signedinteger) else np.uint32)
  return x


# ---------------------------------------------------------------------------
# Shape / structural primitives

@register(lax.broadcast_in_dim_p)
def _broadcast_in_dim(x, *, shape, broadcast_dimensions, sharding):
  x = np.asarray(x)
  in_shape = [1] * len(shape)
  for i, d in enumerate(broadcast_dimensions):
    in_shape[d] = x.shape[i]
  return np.broadcast_to(x.reshape(in_shape), tuple(shape)).copy()


@register(lax.concatenate_p)
def _concatenate(*xs, dimension):
  return np.concatenate([np.asarray(x) for x in xs], axis=dimension)


@register(lax.copy_p)
def _copy(x):
  return np.array(x, copy=True)


@register(lax.empty_p)
def _empty(*, shape, dtype, out_sharding):
  return np.zeros(shape, dtype)


@register(lax.empty2_p)
def _empty2(*, dtype, memory_space):
  return np.zeros((), dtype)


@register(lax.iota_p)
def _iota(*, dtype, shape, dimension, sharding):
  n = shape[dimension]
  r = np.arange(n, dtype=np.int64).astype(dtype)
  bshape = [1] * len(shape)
  bshape[dimension] = n
  return np.broadcast_to(r.reshape(bshape), tuple(shape)).copy()


@register(lax.pad_p)
def _pad(x, padding_value, *, padding_config):
  x = np.asarray(x)
  pv = np.asarray(padding_value, dtype=x.dtype)
  for axis, (lo, hi, interior) in enumerate(padding_config):
    n = x.shape[axis]
    # Interior padding.
    if interior > 0 and n > 1:
      new_n = n + (n - 1) * interior
      shp = list(x.shape)
      shp[axis] = new_n
      y = np.full(shp, pv, dtype=x.dtype)
      idx = [slice(None)] * x.ndim
      idx[axis] = slice(None, None, interior + 1)
      y[tuple(idx)] = x
      x = y
    # Edge padding (possibly negative).
    n = x.shape[axis]
    start = max(-lo, 0)
    stop = n - max(-hi, 0)
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(start, max(stop, start))
    x = x[tuple(idx)]
    parts = []
    if lo > 0:
      shp = list(x.shape)
      shp[axis] = lo
      parts.append(np.full(shp, pv, dtype=x.dtype))
    parts.append(x)
    if hi > 0:
      shp = list(x.shape)
      shp[axis] = hi
      parts.append(np.full(shp, pv, dtype=x.dtype))
    if len(parts) > 1:
      x = np.concatenate(parts, axis=axis)
    # If negative padding removed more than the array held, the size can only
    # be zero (abstract eval guarantees a non-negative size).
  return x


@register(lax.reshape_p)
def _reshape(x, *, new_sizes, dimensions, sharding):
  x = np.asarray(x)
  if dimensions is not None:
    x = np.transpose(x, dimensions)
  return np.reshape(x, tuple(new_sizes))


@register(lax.rev_p)
def _rev(x, *, dimensions):
  x = np.asarray(x)
  if not dimensions:
    return x
  return np.flip(x, axis=tuple(dimensions)).copy()


@register(lax.split_p)
def _split(x, *, sizes, axis):
  x = np.asarray(x)
  out = []
  start = 0
  for s in sizes:
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(start, start + s)
    out.append(x[tuple(idx)])
    start += s
  return out


@register(lax.squeeze_p)
def _squeeze(x, *, dimensions):
  x = np.asarray(x)
  return np.squeeze(x, axis=tuple(dimensions)) if dimensions else x


@register(lax.stack_p)
def _stack(*xs, axis):
  return np.stack([np.asarray(x) for x in xs], axis=axis)


@register(lax.unstack_p)
def _unstack(x, *, axis):
  x = np.asarray(x)
  return [np.take(x, i, axis=axis) for i in range(x.shape[axis])]


@register(lax.tile_p)
def _tile(x, *, reps):
  return np.tile(np.asarray(x), tuple(reps))


@register(lax.transpose_p)
def _transpose(x, *, permutation):
  return np.transpose(np.asarray(x), tuple(permutation))


@register(lax.optimization_barrier_p, raw=True)
def _optimization_barrier(*args):
  from shax.jax._src.state.types import AbstractRef  # noqa: PLC0415
  return [a for a in args if not isinstance(core.typeof(a), AbstractRef)]


# ---------------------------------------------------------------------------
# Slicing

@register(slicing.slice_p)
def _slice(x, *, start_indices, limit_indices, strides):
  x = np.asarray(x)
  strides = strides or (1,) * x.ndim
  return x[tuple(slice(s, l, st) for s, l, st in
                 zip(start_indices, limit_indices, strides))]


def _clamped_starts(shape, sizes, starts):
  out = []
  for n, sz, s in zip(shape, sizes, starts):
    s = int(np.asarray(s).astype(np.int64)) if np.asarray(s).dtype != np.uint64 \
        else min(int(np.asarray(s)), n)
    out.append(min(max(s, 0), n - sz))
  return out


@register(slicing.dynamic_slice_p)
def _dynamic_slice(x, *start_indices, slice_sizes):
  x = np.asarray(x)
  starts = _clamped_starts(x.shape, slice_sizes, start_indices)
  return x[tuple(slice(s, s + sz) for s, sz in zip(starts, slice_sizes))].copy()


@register(slicing.dynamic_update_slice_p)
def _dynamic_update_slice(x, update, *start_indices):
  out = np.array(x, copy=True)
  update = np.asarray(update)
  starts = _clamped_starts(out.shape, update.shape, start_indices)
  out[tuple(slice(s, s + sz) for s, sz in zip(starts, update.shape))] = update
  return out


def _to_int64(indices):
  idx = np.asarray(indices)
  if idx.dtype == np.uint64:
    # Values beyond int64 are out of bounds either way; saturate.
    return np.minimum(idx, np.iinfo(np.int64).max).astype(np.int64)
  return idx.astype(np.int64)


def _window_index_arrays(starts, nbatch, window_dims, window_sizes, ndim):
  """Index arrays of shape B + W addressing `operand` for each window element.

  starts: int64 array of shape B + (ndim,), the window start per operand dim.
  window_dims: operand dims (in order) that carry a window dimension.
  """
  nw = len(window_dims)
  out = []
  for d in range(ndim):
    s = starts[..., d].reshape(starts.shape[:nbatch] + (1,) * nw)
    if d in window_dims:
      j = window_dims.index(d)
      shp = [1] * (nbatch + nw)
      shp[nbatch + j] = window_sizes[j]
      s = s + np.arange(window_sizes[j], dtype=np.int64).reshape(shp)
    out.append(s)
  return out


def _batch_iota(batch_shape, dim):
  shp = [1] * len(batch_shape)
  shp[dim] = batch_shape[dim]
  return np.broadcast_to(
      np.arange(batch_shape[dim], dtype=np.int64).reshape(shp), batch_shape)


@register(slicing.gather_p)
def _gather(operand, indices, *, dimension_numbers, slice_sizes, unique_indices,
            indices_are_sorted, mode, fill_value):
  dn = dimension_numbers
  operand = np.asarray(operand)
  idx = _to_int64(indices)
  batch_shape = idx.shape[:-1]
  nbatch = len(batch_shape)
  ndim = operand.ndim
  slice_sizes = tuple(int(s) for s in slice_sizes)

  window_dims = [d for d in range(ndim) if d not in dn.collapsed_slice_dims
                 and d not in dn.operand_batching_dims]
  window_sizes = [slice_sizes[d] for d in window_dims]
  out_rank = nbatch + len(dn.offset_dims)
  batch_out_dims = [o for o in range(out_rank) if o not in dn.offset_dims]
  # Output shape, then permutation from intermediate (B + W) to output layout.
  out_shape = [0] * out_rank
  perm = [0] * out_rank
  for j, o in enumerate(dn.offset_dims):
    out_shape[o] = window_sizes[j]
    perm[o] = nbatch + j
  for j, o in enumerate(batch_out_dims):
    out_shape[o] = batch_shape[j]
    perm[o] = j
  out_shape = tuple(out_shape)

  if int(np.prod(out_shape, dtype=np.int64)) == 0:
    return np.zeros(out_shape, operand.dtype)

  starts = np.zeros(batch_shape + (ndim,), np.int64)
  for k, d in enumerate(dn.start_index_map):
    starts[..., d] = idx[..., k]
  for od, sd in zip(dn.operand_batching_dims, dn.start_indices_batching_dims):
    starts[..., od] = _batch_iota(batch_shape, sd)

  upper = np.array([operand.shape[d] - slice_sizes[d] for d in range(ndim)],
                   np.int64)
  in_bounds = None
  if mode == GatherScatterMode.FILL_OR_DROP:
    sim = list(dn.start_index_map)
    if sim:
      in_bounds = np.all((idx >= 0) & (idx <= upper[sim]), axis=-1)
    else:
      in_bounds = np.ones(batch_shape, bool)
  starts = np.clip(starts, 0, np.maximum(upper, 0))

  index_arrays = _window_index_arrays(starts, nbatch, window_dims, window_sizes,
                                      ndim)
  res = operand[tuple(index_arrays)]
  res = np.broadcast_to(res, batch_shape + tuple(window_sizes))
  if in_bounds is not None:
    fill = np.asarray(fill_value).astype(operand.dtype) if fill_value is not None \
        else np.zeros((), operand.dtype)
    mask = in_bounds.reshape(batch_shape + (1,) * len(window_sizes))
    res = np.where(mask, res, fill)
  return np.transpose(res, perm).copy()


def _scatter_targets(operand_shape, indices, updates, dn, mode):
  """Returns (flat operand indices, flat update values) of applied updates."""
  ndim = len(operand_shape)
  idx = _to_int64(indices)
  batch_shape = idx.shape[:-1]
  nbatch = len(batch_shape)
  updates = np.asarray(updates)
  uwd = list(dn.update_window_dims)
  update_scatter_dims = [d for d in range(updates.ndim) if d not in uwd]
  # Rearrange updates to B + W.
  upd = np.transpose(updates, update_scatter_dims + uwd)
  window_dims = [d for d in range(ndim) if d not in dn.inserted_window_dims
                 and d not in dn.operand_batching_dims]
  window_sizes = [updates.shape[d] for d in uwd]
  full_sizes = [1] * ndim
  for d, s in zip(window_dims, window_sizes):
    full_sizes[d] = s

  starts = np.zeros(batch_shape + (ndim,), np.int64)
  for k, d in enumerate(dn.scatter_dims_to_operand_dims):
    starts[..., d] = idx[..., k]
  for od, sd in zip(dn.operand_batching_dims, dn.scatter_indices_batching_dims):
    starts[..., od] = _batch_iota(batch_shape, sd)
  upper = np.array([operand_shape[d] - full_sizes[d] for d in range(ndim)],
                   np.int64)
  if mode == GatherScatterMode.CLIP:
    starts = np.clip(starts, 0, np.maximum(upper, 0))
    valid = np.ones(batch_shape, bool)
  else:
    valid = np.all((starts >= 0) & (starts <= upper), axis=-1)

  index_arrays = _window_index_arrays(starts, nbatch, window_dims, window_sizes,
                                      ndim)
  full = batch_shape + tuple(window_sizes)
  index_arrays = [np.broadcast_to(a, full)[valid] for a in index_arrays]
  vals = upd[valid]
  if vals.size == 0:
    return np.zeros(0, np.int64), vals.reshape(-1)
  if ndim == 0:
    flat = np.zeros(vals.size, np.int64)
  else:
    flat = np.ravel_multi_index(tuple(a.reshape(-1) for a in index_arrays),
                                operand_shape)
  return flat, vals.reshape(-1)


def _scatter_impl(ufunc_at, operand, indices, updates, *, update_jaxpr,
                  update_consts, dimension_numbers, indices_are_sorted,
                  unique_indices, mode):
  out = np.array(operand, copy=True, order="C")
  flat_idx, vals = _scatter_targets(out.shape, indices, updates,
                                    dimension_numbers, mode)
  if flat_idx.size == 0:
    return out
  flat = out.reshape(-1)
  vals = vals.astype(out.dtype)
  if ufunc_at is None:
    flat[flat_idx] = vals
    return out
  cdt = _compute_dtype(flat[:0]).dtype
  if cdt != flat.dtype:
    tmp = flat.astype(cdt)
    ufunc_at(tmp, flat_idx, vals.astype(cdt))
    flat[...] = tmp.astype(out.dtype)
  else:
    ufunc_at(flat, flat_idx, vals)
  return out


def _make_scatter(ufunc):
  def impl(operand, indices, updates, **params):
    return _scatter_impl(None if ufunc is None else ufunc.at,
                         operand, indices, updates, **params)
  return impl


register(slicing.scatter_add_p)(_make_scatter(np.add))
register(slicing.scatter_sub_p)(_make_scatter(np.subtract))
register(slicing.scatter_mul_p)(_make_scatter(np.multiply))
register(slicing.scatter_min_p)(_make_scatter(np.minimum))
register(slicing.scatter_max_p)(_make_scatter(np.maximum))


def _is_replace_jaxpr(jaxpr) -> bool:
  """True if `jaxpr` is `lambda x, y: y` (plain scatter)."""
  if jaxpr is None:
    return True
  jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
  return (not jaxpr.eqns and len(jaxpr.invars) == 2 and len(jaxpr.outvars) == 1
          and jaxpr.outvars[0] is jaxpr.invars[1])


@register(slicing.scatter_p)
def _scatter(operand, indices, updates, *, update_jaxpr, update_consts,
             dimension_numbers, indices_are_sorted, unique_indices, mode):
  if _is_replace_jaxpr(update_jaxpr):
    return _scatter_impl(None, operand, indices, updates,
                         update_jaxpr=update_jaxpr, update_consts=update_consts,
                         dimension_numbers=dimension_numbers,
                         indices_are_sorted=indices_are_sorted,
                         unique_indices=unique_indices, mode=mode)
  # General combiner: out[i] = f(out[i], u) applied once per update, in rounds
  # so that duplicate targets are combined sequentially.
  out = np.array(operand, copy=True, order="C")
  flat_idx, vals = _scatter_targets(out.shape, indices, updates,
                                    dimension_numbers, mode)
  flat = out.reshape(-1)
  vals = vals.astype(out.dtype)
  jaxpr = getattr(update_jaxpr, "jaxpr", update_jaxpr)
  consts = list(update_consts)

  def combine(xs, ys):
    try:
      res, = core.eval_jaxpr(jaxpr, consts, xs, ys)
      res = np.asarray(to_numpy(res))
      if res.shape == xs.shape:
        return res
    except Exception:  # noqa: BLE001  (fall back to elementwise evaluation)
      pass
    return np.array([np.asarray(to_numpy(core.eval_jaxpr(jaxpr, consts, x, y)[0]))
                     for x, y in zip(xs, ys)], dtype=out.dtype).reshape(xs.shape)

  remaining = np.arange(flat_idx.size)
  while remaining.size:
    _, first = np.unique(flat_idx[remaining], return_index=True)
    take = remaining[first]
    tgt = flat_idx[take]
    flat[tgt] = np.asarray(combine(flat[tgt], vals[take])).astype(out.dtype)
    remaining = np.delete(remaining, first)
  return out


# ---------------------------------------------------------------------------
# Sorting

def _float_sort_keys(x):
  """Keys realizing -inf < ... < -0 == 0 < ... < inf < NaN (all NaNs equal)."""
  x = _compute_dtype(x)
  nan = np.isnan(x)
  v = np.where(nan, np.zeros((), x.dtype), x)
  v = np.where(v == 0, np.zeros((), x.dtype), v)
  return [nan, v]   # most significant first


def _sort_keys(x):
  dt = x.dtype
  if _is_complex(dt):
    return _float_sort_keys(np.real(x)) + _float_sort_keys(np.imag(x))
  if _is_float(dt):
    return _float_sort_keys(x)
  return [_compute_dtype(x)]


@register(lax.sort_p)
def _sort(*operands, dimension, is_stable, num_keys):
  ops = [np.moveaxis(np.asarray(o), dimension, -1) for o in operands]
  keys = []
  for o in ops[:num_keys]:
    keys.extend(_sort_keys(o))
  if ops[0].size == 0:
    order = np.zeros(ops[0].shape, np.intp)
  else:
    # np.lexsort sorts by the last key first and is stable.
    order = np.lexsort(keys[::-1], axis=-1)
  return [np.moveaxis(np.take_along_axis(o, order, axis=-1), -1, dimension)
          for o in ops]


def _total_order_key(x):
  """Integer key ordering floats as -NaN < -inf < ... < -0 < 0 < ... < NaN."""
  x = _compute_dtype(x)
  if _is_float(x.dtype):
    i = x.astype(np.float64).view(np.int64)
    return i ^ ((i >> 63) & np.int64(0x7FFFFFFFFFFFFFFF))
  return x


@register(lax.top_k_p)
def _top_k(operand, *, k, axis, is_stable):
  x = np.moveaxis(np.asarray(operand), axis, -1)
  key = _total_order_key(x)
  # Descending, ties broken by lower index: stable ascending sort of the
  # reversed array, then reverse.
  rev = key[..., ::-1]
  order = np.argsort(rev, axis=-1, kind="stable")[..., ::-1]
  order = (x.shape[-1] - 1 - order)[..., :k]
  vals = np.take_along_axis(x, order, axis=-1)
  return [np.moveaxis(vals, -1, axis),
          np.moveaxis(order.astype(np.int32), -1, axis)]


def _argminmax(fn, x, axes, index_dtype):
  axis, = axes
  x = np.asarray(x)
  if _is_float(x.dtype) and x.dtype not in _NATIVE_FLOATS:
    x = x.astype(np.float32)
  elif x.dtype not in _NATIVE_FLOATS:
    x = _compute_dtype(x)
  return fn(x, axis=axis).astype(index_dtype)


@register(lax.argmax_p)
def _argmax(x, *, axes, index_dtype):
  return _argminmax(np.argmax, x, axes, index_dtype)


@register(lax.argmin_p)
def _argmin(x, *, axes, index_dtype):
  return _argminmax(np.argmin, x, axes, index_dtype)
