"""NumPy implementations of reductions, contractions, convolution, windowed and
cumulative ops, FFT and linear algebra primitives (see shax_backend/__init__.py).

Conventions follow XLA on CPU: low-precision floats (float16, bfloat16, float8)
are computed in float32 and cast back; integer sums/products wrap; max/min
propagate NaN; linalg failures produce NaN-filled outputs like the LAPACK
lowerings in jax/_src/lax/linalg.py.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from shax.jax._src import core
from shax.jax._src import dtypes as jdt
from shax.jax._src.lax import lax
from shax.jax._src.lax import convolution
from shax.jax._src.lax import fft as lax_fft
from shax.jax._src.lax import linalg
from shax.jax._src.lax import windowed_reductions as wr
from shax.jax._src.lax.control_flow import loops
from shax.jax._src.shax_backend import register, to_numpy


# ---------------------------------------------------------------------------
# dtype helpers

def _is_float(dt):
  return jdt.issubdtype(dt, np.floating)


def _is_complex(dt):
  return jdt.issubdtype(dt, np.complexfloating)


def _is_int(dt):
  return jdt.issubdtype(dt, np.integer)


def _is_bool(dt):
  return np.dtype(dt) == np.bool_


def _is_std_int(dt):
  return np.dtype(dt) in (np.dtype(t) for t in (
      np.int8, np.int16, np.int32, np.int64,
      np.uint8, np.uint16, np.uint32, np.uint64))


def _float_compute(dt):
  """Compute dtype for floats: float32 unless the dtype is float64."""
  return np.dtype(np.float64) if np.dtype(dt) == np.float64 else np.dtype(np.float32)


def _acc_dtype(dt):
  """Accumulation dtype for sums/products/dots (results wrap on cast back)."""
  dt = np.dtype(dt)
  if _is_bool(dt):
    return dt
  if _is_complex(dt):
    return dt
  if _is_float(dt):
    return _float_compute(dt)
  if _is_int(dt):
    return np.dtype(np.uint64) if jdt.issubdtype(dt, np.unsignedinteger) else np.dtype(np.int64)
  return dt


def _minmax_dtype(dt):
  """Compute dtype for max/min-style ops (exact; no overflow possible)."""
  dt = np.dtype(dt)
  if _is_float(dt):
    return _float_compute(dt)
  if _is_int(dt) and not _is_std_int(dt):
    return np.dtype(np.uint64) if jdt.issubdtype(dt, np.unsignedinteger) else np.dtype(np.int64)
  return dt


def _max_identity(dt):
  dt = np.dtype(dt)
  if _is_bool(dt):
    return False
  if _is_float(dt):
    return -np.inf if jdt.supports_inf(dt) else float(jdt.finfo(dt).min)
  if _is_int(dt):
    return int(jdt.iinfo(dt).min)
  raise TypeError(dt)


def _min_identity(dt):
  dt = np.dtype(dt)
  if _is_bool(dt):
    return True
  if _is_float(dt):
    return np.inf if jdt.supports_inf(dt) else float(jdt.finfo(dt).max)
  if _is_int(dt):
    return int(jdt.iinfo(dt).max)
  raise TypeError(dt)


def _upcast(x, dt):
  x = np.asarray(x)
  return x if x.dtype == dt else x.astype(dt)


# ---------------------------------------------------------------------------
# Evaluating reducer jaxprs elementwise over arrays

def _jaxpr_fn(jaxpr, consts=()):
  del consts  # Jaxprs carry their own consts.
  return core.jaxpr_as_fun(jaxpr)


def _vectorized(fn, n_out):
  """Lift a scalar reducer to flat 1-D NumPy arrays (via vmap)."""
  from shax.jax._src import api  # noqa: PLC0415
  vfn = api.vmap(fn)

  def apply(*args):
    args = [np.asarray(a) for a in args]
    shape = args[0].shape
    size = math.prod(shape)
    if size == 0:
      return [np.empty(shape, a.dtype) for a in args[:n_out]]
    flat = [a.reshape(size) for a in args]
    outs = vfn(*flat)
    return [np.asarray(to_numpy(o)).reshape(shape) for o in outs]
  return apply


# ---------------------------------------------------------------------------
# Reductions

def _reduce_with(op, x, axes, identity, compute_dtype):
  x = np.asarray(x)
  out_dtype = x.dtype
  xc = _upcast(x, compute_dtype)
  init = np.asarray(identity).astype(compute_dtype) if identity is not None else None
  if init is None:
    r = op.reduce(xc, axis=tuple(axes))
  else:
    r = op.reduce(xc, axis=tuple(axes), initial=init)
  return np.asarray(r).astype(out_dtype)


@register(lax.reduce_sum_p)
def _reduce_sum(x, *, axes, out_sharding=None):
  x = np.asarray(x)
  dt = _acc_dtype(x.dtype)
  if _is_bool(x.dtype):
    return np.logical_or.reduce(x, axis=tuple(axes))
  return np.asarray(np.sum(_upcast(x, dt), axis=tuple(axes), dtype=dt)).astype(x.dtype)


@register(lax.reduce_prod_p)
def _reduce_prod(x, *, axes, out_sharding=None):
  x = np.asarray(x)
  dt = _acc_dtype(x.dtype)
  if _is_bool(x.dtype):
    return np.logical_and.reduce(x, axis=tuple(axes))
  return np.asarray(np.prod(_upcast(x, dt), axis=tuple(axes), dtype=dt)).astype(x.dtype)


def _complex_lex_reduce(x, axes, is_max):
  # XLA compares complex numbers lexicographically (real, then imag).
  x = np.asarray(x)
  axes = tuple(axes)
  keep = [d for d in range(x.ndim) if d not in axes]
  xt = np.transpose(x, keep + list(axes))
  out_shape = xt.shape[:len(keep)]
  xt = xt.reshape(out_shape + (math.prod(xt.shape[len(keep):]),))
  if xt.shape[-1] == 0:
    v = -np.inf if is_max else np.inf
    return np.full(out_shape, complex(v, 0), x.dtype)
  re, im = xt.real, xt.imag
  order = np.lexsort((im, re), axis=-1)
  idx = order[..., -1] if is_max else order[..., 0]
  return np.take_along_axis(xt, idx[..., None], -1)[..., 0]


@register(lax.reduce_max_p)
def _reduce_max(x, *, axes, out_sharding=None):
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return _complex_lex_reduce(x, axes, True)
  return _reduce_with(np.maximum, x, axes, _max_identity(x.dtype),
                      _minmax_dtype(x.dtype))


@register(lax.reduce_min_p)
def _reduce_min(x, *, axes, out_sharding=None):
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return _complex_lex_reduce(x, axes, False)
  return _reduce_with(np.minimum, x, axes, _min_identity(x.dtype),
                      _minmax_dtype(x.dtype))


def _bitwise_reduce(logical, bitwise, identity_bits, x, axes):
  x = np.asarray(x)
  if _is_bool(x.dtype):
    return logical.reduce(x, axis=tuple(axes))
  dt = x.dtype if _is_std_int(x.dtype) else (
      np.dtype(np.uint64) if jdt.issubdtype(x.dtype, np.unsignedinteger)
      else np.dtype(np.int64))
  xc = _upcast(x, dt)
  init = np.zeros((), dt)
  if identity_bits:
    init = np.bitwise_not(init)
  return np.asarray(bitwise.reduce(xc, axis=tuple(axes), initial=init)).astype(x.dtype)


@register(lax.reduce_and_p)
def _reduce_and(x, *, axes):
  return _bitwise_reduce(np.logical_and, np.bitwise_and, True, x, axes)


@register(lax.reduce_or_p)
def _reduce_or(x, *, axes):
  return _bitwise_reduce(np.logical_or, np.bitwise_or, False, x, axes)


@register(lax.reduce_xor_p)
def _reduce_xor(x, *, axes):
  return _bitwise_reduce(np.logical_xor, np.bitwise_xor, False, x, axes)


@register(lax.reduce_p)
def _reduce(*args, computation, jaxpr, dimensions):
  n = len(args) // 2
  operands = [np.asarray(a) for a in args[:n]]
  inits = [np.asarray(a) for a in args[n:]]
  f = _vectorized(_jaxpr_fn(jaxpr), n)
  dims = tuple(dimensions)
  x0 = operands[0]
  keep = [d for d in range(x0.ndim) if d not in dims]
  out_shape = tuple(x0.shape[d] for d in keep)
  red = math.prod(x0.shape[d] for d in dims)
  xs = [np.transpose(x, keep + list(dims)).reshape(out_shape + (red,))
        for x in operands]
  # Pairwise tree reduction along the last axis.
  while xs[0].shape[-1] > 1:
    m = xs[0].shape[-1]
    h = m // 2
    combined = f(*[x[..., :h] for x in xs], *[x[..., h:2 * h] for x in xs])
    if m % 2:
      xs = [np.concatenate([c, x[..., 2 * h:]], axis=-1)
            for c, x in zip(combined, xs)]
    else:
      xs = combined
  accs = [np.broadcast_to(i.astype(x.dtype), out_shape).copy()
          for i, x in zip(inits, operands)]
  if red == 0:
    return accs
  return f(*accs, *[x[..., 0] for x in xs])


# ---------------------------------------------------------------------------
# Cumulative reductions

def _cumulative(ufunc_accumulate, x, axis, reverse, compute_dtype):
  x = np.asarray(x)
  if x.shape[axis] == 0:
    return x
  xc = _upcast(x, compute_dtype)
  if reverse:
    xc = np.flip(xc, axis)
  r = ufunc_accumulate(xc, axis=axis)
  if reverse:
    r = np.flip(r, axis)
  return np.asarray(r).astype(x.dtype)


@register(loops.cumsum_p)
def _cumsum(x, *, axis, reverse):
  x = np.asarray(x)
  return _cumulative(np.add.accumulate, x, axis, reverse, _acc_dtype(x.dtype))


@register(loops.cumprod_p)
def _cumprod(x, *, axis, reverse):
  x = np.asarray(x)
  return _cumulative(np.multiply.accumulate, x, axis, reverse, _acc_dtype(x.dtype))


def _complex_lex_accumulate(is_max):
  def acc(x, axis):
    x = np.moveaxis(x, axis, -1).copy()
    for i in range(1, x.shape[-1]):
      a, b = x[..., i - 1], x[..., i]
      if is_max:
        take_a = (a.real > b.real) | ((a.real == b.real) & (a.imag > b.imag))
      else:
        take_a = (a.real < b.real) | ((a.real == b.real) & (a.imag < b.imag))
      x[..., i] = np.where(take_a, a, b)
    return np.moveaxis(x, -1, axis)
  return acc


@register(loops.cummax_p)
def _cummax(x, *, axis, reverse):
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return _cumulative(_complex_lex_accumulate(True), x, axis, reverse, x.dtype)
  return _cumulative(np.maximum.accumulate, x, axis, reverse, _minmax_dtype(x.dtype))


@register(loops.cummin_p)
def _cummin(x, *, axis, reverse):
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return _cumulative(_complex_lex_accumulate(False), x, axis, reverse, x.dtype)
  return _cumulative(np.minimum.accumulate, x, axis, reverse, _minmax_dtype(x.dtype))


def _logaddexp_accumulate(x, axis):
  if _is_complex(x.dtype):
    x = np.moveaxis(x, axis, -1).copy()
    for i in range(1, x.shape[-1]):
      a, b = x[..., i - 1], x[..., i]
      m = np.where(a.real > b.real, a, b)
      m = np.where(np.isfinite(m.real), m, 0)
      x[..., i] = m + np.log(np.exp(a - m) + np.exp(b - m))
    return np.moveaxis(x, -1, axis)
  return np.logaddexp.accumulate(x, axis=axis)


@register(loops.cumlogsumexp_p)
def _cumlogsumexp(x, *, axis, reverse):
  x = np.asarray(x)
  dt = x.dtype if _is_complex(x.dtype) else _float_compute(x.dtype)
  return _cumulative(_logaddexp_accumulate, x, axis, reverse, dt)


# ---------------------------------------------------------------------------
# Windowed reductions

def _pad_dilate(x, fill, padding, base_dilation=None):
  """Interior-dilate `x` by `base_dilation` and edge-pad (possibly negative)
  with `fill`, XLA style."""
  x = np.asarray(x)
  if base_dilation is not None and any(d != 1 for d in base_dilation):
    shape = tuple((s - 1) * d + 1 if s > 0 else 0
                  for s, d in zip(x.shape, base_dilation))
    y = np.full(shape, fill, dtype=x.dtype)
    y[tuple(slice(None, None, d) for d in base_dilation)] = x
    x = y
  pos = [(max(lo, 0), max(hi, 0)) for lo, hi in padding]
  if any(lo or hi for lo, hi in pos):
    shape = tuple(s + lo + hi for s, (lo, hi) in zip(x.shape, pos))
    y = np.full(shape, fill, dtype=x.dtype)
    y[tuple(slice(lo, lo + s) for s, (lo, _) in zip(x.shape, pos))] = x
    x = y
  if any(lo < 0 or hi < 0 for lo, hi in padding):
    idx = tuple(slice(-lo if lo < 0 else 0, s + hi if hi < 0 else s)
                for s, (lo, hi) in zip(x.shape, padding))
    x = x[idx]
  return x


def _window_geometry(padded_shape, window_dimensions, window_strides,
                     window_dilation):
  out = []
  for p, w, s, d in zip(padded_shape, window_dimensions, window_strides,
                        window_dilation):
    eff = (w - 1) * d + 1 if w > 0 else 0
    out.append(max(0, (p - eff) // s + 1) if p >= eff else 0)
  return tuple(out)


def _window_slices(out_shape, window_dimensions, window_strides,
                   window_dilation):
  """Yield (offset, slice tuple) for each window element."""
  for off in itertools.product(*(range(w) for w in window_dimensions)):
    yield off, tuple(
        slice(k * d, k * d + (o - 1) * s + 1, s) if o > 0 else slice(0, 0)
        for k, d, o, s in zip(off, window_dilation, out_shape, window_strides))


def _reduce_window_monoid(op, identity, compute_dtype, x, window_dimensions,
                          window_strides, padding, base_dilation,
                          window_dilation):
  x = np.asarray(x)
  out_dtype = x.dtype
  xc = _upcast(x, compute_dtype)
  ident = np.asarray(identity).astype(compute_dtype)
  xp = _pad_dilate(xc, ident, padding, base_dilation)
  out_shape = _window_geometry(xp.shape, window_dimensions, window_strides,
                               window_dilation)
  acc = np.full(out_shape, ident, dtype=compute_dtype)
  if math.prod(out_shape):
    for _, sl in _window_slices(out_shape, window_dimensions, window_strides,
                                window_dilation):
      acc = op(acc, xp[sl])
  return acc.astype(out_dtype)


_RW_PARAMS = ("window_dimensions", "window_strides", "padding",
              "base_dilation", "window_dilation")


@register(wr.reduce_window_sum_p)
def _reduce_window_sum(x, **params):
  x = np.asarray(x)
  if _is_bool(x.dtype):
    return _reduce_window_monoid(np.logical_or, False, x.dtype, x, **params)
  return _reduce_window_monoid(np.add, 0, _acc_dtype(x.dtype), x, **params)


def _lex_max(a, b):
  take_b = (b.real > a.real) | ((b.real == a.real) & (b.imag > a.imag))
  return np.where(take_b | np.isnan(b.real) | np.isnan(b.imag), b, a)


def _lex_min(a, b):
  take_b = (b.real < a.real) | ((b.real == a.real) & (b.imag < a.imag))
  return np.where(take_b | np.isnan(b.real) | np.isnan(b.imag), b, a)


@register(wr.reduce_window_max_p)
def _reduce_window_max(x, **params):
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return _reduce_window_monoid(_lex_max, complex(-np.inf, 0), x.dtype, x,
                                 **params)
  return _reduce_window_monoid(np.maximum, _max_identity(x.dtype),
                               _minmax_dtype(x.dtype), x, **params)


@register(wr.reduce_window_min_p)
def _reduce_window_min(x, **params):
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return _reduce_window_monoid(_lex_min, complex(np.inf, 0), x.dtype, x,
                                 **params)
  return _reduce_window_monoid(np.minimum, _min_identity(x.dtype),
                               _minmax_dtype(x.dtype), x, **params)


@register(wr.reduce_window_p)
def _reduce_window(*args, jaxpr, consts, window_dimensions, window_strides,
                   padding, base_dilation, window_dilation):
  n = len(args) // 2
  operands = [np.asarray(a) for a in args[:n]]
  inits = [np.asarray(a).astype(o.dtype) for a, o in zip(args[n:], operands)]
  f = _vectorized(_jaxpr_fn(jaxpr, consts), n)
  padded = [_pad_dilate(x, i, padding, base_dilation)
            for x, i in zip(operands, inits)]
  out_shape = _window_geometry(padded[0].shape, window_dimensions,
                               window_strides, window_dilation)
  accs = [np.full(out_shape, i, dtype=x.dtype) for x, i in zip(operands, inits)]
  if math.prod(out_shape):
    for _, sl in _window_slices(out_shape, window_dimensions, window_strides,
                                window_dilation):
      accs = f(*accs, *[p[sl] for p in padded])
  return accs


def _select_positions(operand, select_fn, window_dimensions, window_strides,
                      padding, base_dilation=None, window_dilation=None):
  """For each window, the flat index (into the padded operand) of the element
  chosen by `select_fn(a, b) -> bool (keep a)`, scanning in window order and
  skipping padding/dilation holes. Returns (out_shape, flat operand indices),
  with -1 for windows containing no operand element."""
  nd = operand.ndim
  base_dilation = base_dilation or (1,) * nd
  window_dilation = window_dilation or (1,) * nd
  # Positions of real (non-padding) operand elements in the padded frame.
  coords = np.arange(operand.size, dtype=np.int64).reshape(operand.shape)
  pc = _pad_dilate(coords, -1, padding, base_dilation)
  pv = _pad_dilate(operand, np.zeros((), operand.dtype), padding, base_dilation)
  out_shape = _window_geometry(pc.shape, window_dimensions, window_strides,
                               window_dilation)
  best_idx = np.full(out_shape, -1, dtype=np.int64)
  best_val = np.zeros(out_shape, dtype=operand.dtype)
  if not math.prod(out_shape):
    return out_shape, best_idx
  for _, sl in _window_slices(out_shape, window_dimensions, window_strides,
                              window_dilation):
    ci, cv = pc[sl], pv[sl]
    valid = ci >= 0
    have = best_idx >= 0
    keep_old = np.asarray(select_fn(best_val, cv)).astype(bool)
    take = valid & (~have | ~keep_old)
    best_idx = np.where(take, ci, best_idx)
    best_val = np.where(take, cv, best_val)
  return out_shape, best_idx


def _select_prim_fn(select_prim, dtype):
  if select_prim is lax.ge_p:
    return np.greater_equal
  if select_prim is lax.le_p:
    return np.less_equal
  if select_prim is lax.gt_p:
    return np.greater
  if select_prim is lax.lt_p:
    return np.less
  raise NotImplementedError(f"shax: select_prim {select_prim}")


@register(wr.select_and_scatter_add_p)
def _select_and_scatter_add(source, operand, *, select_prim, window_dimensions,
                            window_strides, padding):
  source = np.asarray(source)
  operand = np.asarray(operand)
  cdt = _minmax_dtype(operand.dtype) if not _is_complex(operand.dtype) else operand.dtype
  op = _upcast(operand, cdt)
  _, idx = _select_positions(op, _select_prim_fn(select_prim, cdt),
                             window_dimensions, window_strides, padding)
  adt = _acc_dtype(source.dtype)
  out = np.zeros(operand.size, dtype=adt)
  valid = idx >= 0
  np.add.at(out, idx[valid], _upcast(source, adt)[valid])
  return out.reshape(operand.shape).astype(source.dtype)


@register(wr.select_and_scatter_p)
def _select_and_scatter(operand, source, init_value, *, select_jaxpr,
                        select_consts, scatter_jaxpr, scatter_consts,
                        window_dimensions, window_strides, padding):
  operand = np.asarray(operand)
  source = np.asarray(source)
  sel = _vectorized(_jaxpr_fn(select_jaxpr, select_consts), 1)
  sca = _vectorized(_jaxpr_fn(scatter_jaxpr, scatter_consts), 1)
  _, idx = _select_positions(operand, lambda a, b: sel(a, b)[0],
                             window_dimensions, window_strides, padding)
  out = np.full(operand.size, np.asarray(init_value), dtype=operand.dtype)
  idx_f = idx.reshape(-1)
  src_f = source.reshape(-1)
  # Scatter sequentially in window order (combination order matters for
  # non-commutative scatter computations).
  for j in np.nonzero(idx_f >= 0)[0]:
    i = idx_f[j]
    out[i] = sca(out[i:i + 1], src_f[j:j + 1])[0][0]
  return out.reshape(operand.shape)


@register(wr.select_and_gather_add_p)
def _select_and_gather_add(tangents, operand, *, select_prim,
                           window_dimensions, window_strides, padding,
                           base_dilation, window_dilation):
  tangents = np.asarray(tangents)
  operand = np.asarray(operand)
  cdt = _minmax_dtype(operand.dtype)
  op = _upcast(operand, cdt)
  # Padding behaves like -inf/+inf in the reference implementation, i.e. it
  # is never selected over a real element.
  _, idx = _select_positions(op, _select_prim_fn(select_prim, cdt),
                             window_dimensions, window_strides, padding,
                             base_dilation, window_dilation)
  t = tangents.reshape(-1)
  valid = idx >= 0
  out = np.zeros(idx.shape, dtype=tangents.dtype)
  out[valid] = t[idx[valid]]
  return out


# ---------------------------------------------------------------------------
# Dot products

def _dot_compute_dtype(lhs, rhs, preferred_element_type):
  dts = [np.dtype(lhs.dtype), np.dtype(rhs.dtype)]
  if preferred_element_type is not None:
    target = np.dtype(preferred_element_type)
  elif dts[0] == dts[1]:
    target = dts[0]
  else:
    target = None  # apply_primitive casts to the abstract-eval dtype
  allt = dts + ([target] if target is not None else [])
  wide = any(d in (np.dtype(np.float64), np.dtype(np.complex128)) for d in allt)
  if any(_is_complex(d) for d in allt):
    return np.dtype(np.complex128 if wide else np.complex64), target
  if any(_is_float(d) for d in allt):
    return np.dtype(np.float64 if wide else np.float32), target
  if all(_is_bool(d) or jdt.issubdtype(d, np.unsignedinteger) for d in allt):
    return np.dtype(np.uint64), target
  return np.dtype(np.int64), target


def _dot_general_np(lhs, rhs, dimension_numbers, preferred_element_type):
  lhs = np.asarray(lhs)
  rhs = np.asarray(rhs)
  (lc, rc), (lb, rb) = dimension_numbers
  lc, rc, lb, rb = map(tuple, (lc, rc, lb, rb))
  cdt, target = _dot_compute_dtype(lhs, rhs, preferred_element_type)
  lf = [d for d in range(lhs.ndim) if d not in lc and d not in lb]
  rf = [d for d in range(rhs.ndim) if d not in rc and d not in rb]
  bshape = tuple(lhs.shape[d] for d in lb)
  lfs = tuple(lhs.shape[d] for d in lf)
  rfs = tuple(rhs.shape[d] for d in rf)
  k = math.prod(lhs.shape[d] for d in lc)
  B, M, N = math.prod(bshape), math.prod(lfs), math.prod(rfs)
  a = _upcast(np.transpose(lhs, lb + tuple(lf) + lc), cdt).reshape(B, M, k)
  b = _upcast(np.transpose(rhs, rb + rc + tuple(rf)), cdt).reshape(B, k, N)
  if k == 0:
    out = np.zeros((B, M, N), cdt)
  elif _is_float(cdt) or _is_complex(cdt):
    out = np.matmul(a, b)
  else:
    out = np.einsum("bmk,bkn->bmn", a, b)
  out = out.reshape(bshape + lfs + rfs)
  if target is None:
    return out
  if _is_bool(target):
    return out != 0
  return out.astype(target)


@register(lax.dot_general_p)
def _dot_general(lhs, rhs, *, dimension_numbers, precision,
                 preferred_element_type, out_sharding=None):
  return _dot_general_np(lhs, rhs, dimension_numbers, preferred_element_type)


@register(lax.ragged_dot_general_p, raw=True)
def _ragged_dot_general(lhs, rhs, group_sizes, **params):
  return lax._ragged_dot_general_impl(lhs, rhs, group_sizes, **params)


def _register_scaled_dot():
  import importlib  # noqa: PLC0415
  # (`shax.jax._src.lax.scaled_dot` the attribute is the function.)
  sd = importlib.import_module("shax.jax._src.lax.scaled_dot")

  @register(sd.scaled_dot_p)
  def _scaled_dot(lhs, rhs, lhs_scale, rhs_scale, *, dimension_numbers,
                  preferred_element_type=None):
    (lc, rc), _ = dimension_numbers

    def bcast(scale, shape, cdims):
      scale = np.asarray(scale)
      for i in cdims:
        if scale.shape[i] != shape[i]:
          scale = np.repeat(scale, shape[i] // scale.shape[i], axis=i)
      return scale
    bf16 = jdt.bfloat16
    ls = bcast(lhs_scale, np.shape(lhs), lc).astype(bf16)
    rs = bcast(rhs_scale, np.shape(rhs), rc).astype(bf16)
    l = (np.asarray(lhs).astype(bf16).astype(np.float32) * ls.astype(np.float32)).astype(bf16)
    r = (np.asarray(rhs).astype(bf16).astype(np.float32) * rs.astype(np.float32)).astype(bf16)
    return _dot_general_np(l, r, dimension_numbers,
                           preferred_element_type or bf16)


_register_scaled_dot()


def _register_scaled_matmul():
  import importlib  # noqa: PLC0415
  try:
    sm = importlib.import_module("shax.jax._src.cudnn.scaled_matmul_stablehlo")
  except ImportError:  # pragma: no cover
    return

  @register(sm._scaled_matmul_p)
  def _scaled_matmul(a, b, a_scales, b_scales, *, preferred_element_type):
    # Mirrors upstream `_scaled_matmul_cpu_lowering_impl`.
    p = np.dtype(preferred_element_type)
    f = _float_compute(p) if _is_float(p) else p

    def scale(x, s):
      Bx, R, K = x.shape
      Ks = s.shape[-1]
      x = np.asarray(x).astype(p).astype(f).reshape(Bx, R, Ks, K // Ks)
      s = np.asarray(s).astype(p).astype(f).reshape(Bx, R, Ks, 1)
      return (x * s).astype(p).reshape(Bx, R, K)
    out = _dot_general_np(scale(a, a_scales), scale(b, b_scales),
                          (((2,), (2,)), ((0,), (0,))), p)
    return [out]


_register_scaled_matmul()


# ---------------------------------------------------------------------------
# Convolution

@register(convolution.conv_general_dilated_p)
def _conv_general_dilated(lhs, rhs, *, window_strides, padding, lhs_dilation,
                          rhs_dilation, dimension_numbers, feature_group_count,
                          batch_group_count, precision,
                          preferred_element_type, **_):
  lhs = np.asarray(lhs)
  rhs = np.asarray(rhs)
  dn = dimension_numbers
  if dn is None:
    nd = lhs.ndim
    dn = convolution.ConvDimensionNumbers(
        tuple(range(nd)), tuple(range(nd)), tuple(range(nd)))
  lhs_spec, rhs_spec, out_spec = dn
  cdt, target = _dot_compute_dtype(lhs, rhs, preferred_element_type)
  x = _upcast(np.transpose(lhs, lhs_spec), cdt)   # N, C, spatial...
  w = _upcast(np.transpose(rhs, rhs_spec), cdt)   # O, I, spatial...
  nsp = x.ndim - 2
  zero = np.zeros((), cdt)
  x = _pad_dilate(x, zero, [(0, 0), (0, 0)] + [tuple(p) for p in padding],
                  (1, 1) + tuple(lhs_dilation))
  N, C = x.shape[:2]
  O, I = w.shape[:2]
  ksp = w.shape[2:]
  G, Bg = feature_group_count, batch_group_count
  eff_k = tuple((k - 1) * d + 1 if k > 0 else 0 for k, d in zip(ksp, rhs_dilation))
  out_sp = tuple(max(0, (p - k) // s + 1) if p >= k else 0
                 for p, k, s in zip(x.shape[2:], eff_k, window_strides))
  Nb = N // Bg
  Cg = C // G
  Og = O // (G * Bg)
  xg = x.reshape((Bg, Nb, G, Cg) + x.shape[2:])
  wg = w.reshape((Bg, G, Og, I) + ksp)
  acc = np.zeros((Bg, Nb, G, Og) + out_sp, dtype=cdt)
  if math.prod(out_sp) and math.prod(ksp) and Cg:
    for off in itertools.product(*(range(k) for k in ksp)):
      sl = tuple(slice(o * d, o * d + (n - 1) * s + 1, s)
                 for o, d, n, s in zip(off, rhs_dilation, out_sp, window_strides))
      xs = xg[(slice(None),) * 4 + sl]          # Bg, Nb, G, Cg, out_sp
      ws = wg[(slice(None),) * 4 + off]         # Bg, G, Og, I
      xs2 = np.moveaxis(xs, 3, -1)              # Bg, Nb, G, out_sp..., Cg
      # result: Bg, Nb, G, out_sp..., Og
      r = np.matmul(xs2[..., None, :],
                    np.swapaxes(ws, -1, -2).reshape(
                        (Bg, 1, G) + (1,) * nsp + (I, Og)))[..., 0, :]
      acc += np.moveaxis(r, -1, 3)
  # -> Nb, (Bg, G, Og), out_sp
  out = np.moveaxis(acc, 0, 1).reshape((Nb, Bg * G * Og) + out_sp)
  inv = np.argsort(out_spec)
  out = np.transpose(out, inv)
  if target is None:
    return out
  if _is_bool(target):
    return out != 0
  return out.astype(target)


# ---------------------------------------------------------------------------
# FFT

@register(lax_fft.fft_p)
def _fft(x, *, fft_type, fft_lengths):
  x = np.asarray(x)
  k = len(fft_lengths)
  axes = tuple(range(x.ndim - k, x.ndim))
  FT = lax_fft.FftType
  if k == 0:
    return x
  if x.size == 0 or any(n == 0 for n in fft_lengths):
    if fft_type == FT.RFFT:
      shape = x.shape[:-1] + (fft_lengths[-1] // 2 + 1,)
      return np.zeros(shape, np.result_type(x.dtype, np.complex64))
    if fft_type == FT.IRFFT:
      return np.zeros(x.shape[:-k] + tuple(fft_lengths), x.real.dtype)
    return np.zeros(x.shape, x.dtype)
  if fft_type == FT.FFT:
    return np.fft.fftn(x, axes=axes)
  if fft_type == FT.IFFT:
    return np.fft.ifftn(x, axes=axes)
  if fft_type == FT.RFFT:
    return np.fft.rfftn(x, s=tuple(fft_lengths), axes=axes)
  if fft_type == FT.IRFFT:
    return np.fft.irfftn(x, s=tuple(fft_lengths), axes=axes)
  raise ValueError(fft_type)


# ---------------------------------------------------------------------------
# Linear algebra helpers

def _scipy_lapack():
  try:
    import scipy.linalg.lapack as lapack  # noqa: PLC0415
    return lapack
  except ImportError:
    return None


def _need_scipy(what):
  lap = _scipy_lapack()
  if lap is None:
    raise NotImplementedError(f"shax: {what} requires scipy")
  return lap


def _la_dtype(dt):
  """LAPACK compute dtype for `dt`."""
  dt = np.dtype(dt)
  if _is_complex(dt):
    return dt
  if dt == np.float64:
    return dt
  return np.dtype(np.float32)


def _lapack_fn(lap, name, dt):
  prefix = {np.dtype(np.float32): "s", np.dtype(np.float64): "d",
            np.dtype(np.complex64): "c", np.dtype(np.complex128): "z"}[np.dtype(dt)]
  return getattr(lap, prefix + name)


def _batch_apply(fn, arrays, core_ndims, out_dtypes):
  """Apply fn(*matrices) -> tuple of arrays over leading batch dims.

  `core_ndims` gives the non-batch rank of each input. fn handles its own
  failures (returning NaN-filled outputs, as the LAPACK lowerings do)."""
  arrays = [np.asarray(a) for a in arrays]
  batch = arrays[0].shape[:arrays[0].ndim - core_ndims[0]]
  results = None
  for idx in np.ndindex(*batch):
    outs = fn(*[a[idx] for a in arrays])
    if results is None:
      results = [np.empty(batch + np.shape(o), dtype=dt)
                 for o, dt in zip(outs, out_dtypes)]
    for r, o in zip(results, outs):
      r[idx] = o
  if results is None:
    # Empty batch: evaluate on a dummy element to find the core shapes.
    dummies = [np.zeros(a.shape[a.ndim - c:], a.dtype)
               for a, c in zip(arrays, core_ndims)]
    outs = fn(*dummies)
    results = [np.empty(batch + np.shape(o), dtype=dt)
               for o, dt in zip(outs, out_dtypes)]
  return results


def _nan_like(x):
  return np.full(np.shape(x), np.nan, dtype=np.asarray(x).dtype)


# ---------------------------------------------------------------------------
# Cholesky

@register(linalg.cholesky_p)
def _cholesky(x):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)

  def one(a):
    try:
      return (np.linalg.cholesky(a),)
    except np.linalg.LinAlgError:
      return (np.full(a.shape, np.nan, dtype=dt),)
  if x.shape[-1] == 0:
    return x
  return _batch_apply(one, [_upcast(x, dt)], [2], [dt])[0].astype(x.dtype)


@register(linalg.cholesky_update_p)
def _cholesky_update(r, w):
  r = np.asarray(r)
  dt = _la_dtype(r.dtype)
  R = _upcast(r, dt).copy()
  z = _upcast(np.asarray(w), dt).copy()
  n = z.shape[0]
  for k in range(n):
    x, y = R[k, k], z[k]
    if y == 0:
      c, s = dt.type(1), dt.type(0)
    else:
      d = max(abs(x), abs(y))
      xs, ys = x / d, y / d
      rh = 1 / np.sqrt(xs * xs + ys * ys)
      c, s = xs * rh, -ys * rh
    row_k = c * R[k, :] - s * z
    z = c * z + s * R[k, :]
    R[k, :] = row_k
  return R.astype(r.dtype)


# ---------------------------------------------------------------------------
# Eigendecompositions

@register(linalg.eig_p)
def _eig(x, *, compute_left_eigenvectors, compute_right_eigenvectors,
         enable_eigvec_derivs, implementation):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  cdt = np.result_type(dt, np.complex64)
  n = x.shape[-1]

  def one(a):
    outs = []
    try:
      if not np.all(np.isfinite(a)):
        raise np.linalg.LinAlgError
      if compute_left_eigenvectors:
        import scipy.linalg  # noqa: PLC0415
        w, vl, vr = scipy.linalg.eig(a, left=True, right=True)
      elif compute_right_eigenvectors:
        w, vr = np.linalg.eig(a)
        vl = None
      else:
        w = np.linalg.eigvals(a)
        vl = vr = None
    except (np.linalg.LinAlgError, ValueError):
      w = np.full((n,), np.nan, cdt)
      vl = vr = np.full((n, n), np.nan, cdt)
    outs.append(np.asarray(w).astype(cdt))
    if compute_left_eigenvectors:
      outs.append(np.asarray(vl).astype(cdt))
    if compute_right_eigenvectors:
      outs.append(np.asarray(vr).astype(cdt))
    return outs
  ndt = 1 + compute_left_eigenvectors + compute_right_eigenvectors
  return _batch_apply(one, [_upcast(x, dt)], [2], [cdt] * ndt)


@register(linalg.eigh_p)
def _eigh(x, *, lower, sort_eigenvalues, subset_by_index, algorithm):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  rdt = np.finfo(dt).dtype
  n = x.shape[-1]
  lo, hi = (0, n) if subset_by_index is None else subset_by_index

  def one(a):
    try:
      if not np.all(np.isfinite(a)):
        raise np.linalg.LinAlgError
      w, v = np.linalg.eigh(a, UPLO="L" if lower else "U")
    except np.linalg.LinAlgError:
      w = np.full((n,), np.nan, rdt)
      v = np.full((n, n), np.nan, dt)
    return v[:, lo:hi], w[lo:hi]
  v, w = _batch_apply(one, [_upcast(x, dt)], [2], [dt, rdt])
  return v.astype(x.dtype), w


@register(linalg.schur_p)
def _schur(x, *, compute_schur_vectors, sort_eig_vals, select_callable):
  if sort_eig_vals:
    raise NotImplementedError(
        "The sort feature of LAPACK's gees routine is not implemented.")
  import scipy.linalg  # noqa: PLC0415
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  output = "complex" if _is_complex(dt) else "real"

  def one(a):
    try:
      t, z = scipy.linalg.schur(a, output=output)
    except (np.linalg.LinAlgError, ValueError):
      t = z = np.full(a.shape, np.nan, dt)
    return (t, z) if compute_schur_vectors else (t,)
  n_out = 2 if compute_schur_vectors else 1
  return _batch_apply(one, [_upcast(x, dt)], [2], [dt] * n_out)


@register(linalg.hessenberg_p)
def _hessenberg(x):
  lap = _need_scipy("hessenberg")
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  gehrd = _lapack_fn(lap, "gehrd", dt)
  n = x.shape[-1]

  def one(a):
    if n == 0:
      return a, np.zeros((0,), dt)
    h, tau, info = gehrd(a)
    if info != 0:
      return _nan_like(h), np.full((n - 1,), np.nan, dt)
    return h, np.asarray(tau)[:n - 1]
  return _batch_apply(one, [_upcast(x, dt)], [2], [dt, dt])


@register(linalg.tridiagonal_p)
def _tridiagonal(x, *, lower):
  lap = _need_scipy("tridiagonal")
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  rdt = np.finfo(dt).dtype
  name = "hetrd" if _is_complex(dt) else "sytrd"
  fn = _lapack_fn(lap, name, dt)
  n = x.shape[-1]

  def one(a):
    c, d, e, tau, info = fn(a, lower=int(lower))
    if info != 0:
      return (_nan_like(c), np.full((n,), np.nan, rdt),
              np.full((n - 1,), np.nan, rdt), np.full((n - 1,), np.nan, dt))
    return c, d, e, np.asarray(tau)[:n - 1]
  return _batch_apply(one, [_upcast(x, dt)], [2], [dt, rdt, rdt, dt])


# ---------------------------------------------------------------------------
# SVD

@register(linalg.svd_p)
def _svd(x, *, full_matrices, compute_uv, subset_by_index, algorithm=None):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  rdt = np.finfo(dt).dtype
  m, n = x.shape[-2:]
  k = min(m, n)
  lo, hi = (0, k) if subset_by_index is None else subset_by_index
  hi = min(hi, k)
  use_gesvd = algorithm is not None and getattr(algorithm, "name", None) == "QR"

  def one(a):
    if m == 0 or n == 0:
      s = np.zeros((0,), rdt)
      if full_matrices:
        u, vt = np.eye(m, dtype=dt), np.eye(n, dtype=dt)
      else:
        u, vt = np.zeros((m, 0), dt), np.zeros((0, n), dt)
      return (s, u, vt) if compute_uv else (s,)
    try:
      if not np.all(np.isfinite(a)):
        raise np.linalg.LinAlgError
      if use_gesvd and _scipy_lapack() is not None:
        import scipy.linalg  # noqa: PLC0415
        res = scipy.linalg.svd(a, full_matrices=full_matrices,
                               compute_uv=compute_uv, lapack_driver="gesvd")
      else:
        res = np.linalg.svd(a, full_matrices=full_matrices,
                            compute_uv=compute_uv)
      if compute_uv:
        u, s, vt = res
      else:
        s = res
    except (np.linalg.LinAlgError, ValueError):
      s = np.full((k,), np.nan, rdt)
      um = m if full_matrices else k
      vn = n if full_matrices else k
      u = np.full((m, um), np.nan, dt)
      vt = np.full((vn, n), np.nan, dt)
    s = s[lo:hi]
    if not compute_uv:
      return (s,)
    if not full_matrices:
      u, vt = u[:, lo:hi], vt[lo:hi, :]
    return s, u, vt
  outs = _batch_apply(one, [_upcast(x, dt)], [2],
                      [rdt, dt, dt] if compute_uv else [rdt])
  if compute_uv:
    outs[1] = outs[1].astype(x.dtype)
    outs[2] = outs[2].astype(x.dtype)
  return outs


# ---------------------------------------------------------------------------
# QR family (Householder conventions follow LAPACK xLARFG)

def _larfg(alpha, x):
  """Returns (beta, tau, v) with H = I - tau [1; v][1; v]^H, H^H [alpha; x] = [beta; 0]."""
  xnorm = np.linalg.norm(x) if x.size else 0.0
  is_c = np.iscomplexobj(x) or np.iscomplexobj(alpha)
  alphr = np.real(alpha)
  alphi = np.imag(alpha) if is_c else 0.0
  if xnorm == 0 and alphi == 0:
    return alpha, np.zeros((), np.asarray(alpha).dtype), np.zeros_like(x)
  beta = -math.copysign(math.sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm),
                        alphr)
  if is_c:
    tau = complex((beta - alphr) / beta, -alphi / beta)
  else:
    tau = (beta - alphr) / beta
  v = x / (alpha - beta)
  return beta, tau, v


def _geqrf_one(a):
  a = a.copy()
  m, n = a.shape
  k = min(m, n)
  taus = np.zeros((k,), a.dtype)
  for i in range(k):
    beta, tau, v = _larfg(a[i, i], a[i + 1:, i])
    a[i, i] = beta
    a[i + 1:, i] = v
    taus[i] = tau
    if i + 1 < n and tau != 0:
      vv = np.concatenate([np.ones((1,), a.dtype), v])
      sub = a[i:, i + 1:]
      # Apply H^H = I - conj(tau) v v^H from the left.
      a[i:, i + 1:] = sub - np.conj(tau) * np.outer(vv, vv.conj() @ sub)
  return a, taus


def _geqp3_one(a, jpvt):
  """Column-pivoted QR (LAPACK xGEQP3 conventions, 1-based jpvt)."""
  a = a.copy()
  m, n = a.shape
  k = min(m, n)
  jpvt = np.asarray(jpvt)
  perm = np.arange(n)
  # Move fixed columns (jpvt != 0) to the front.
  nfxd = 0
  for j in range(n):
    if jpvt[j] != 0:
      if j != nfxd:
        a[:, [nfxd, j]] = a[:, [j, nfxd]]
        perm[[nfxd, j]] = perm[[j, nfxd]]
      nfxd += 1
  taus = np.zeros((k,), a.dtype)
  for i in range(k):
    if i >= nfxd:
      norms = np.linalg.norm(a[i:, i:], axis=0)
      p = i + int(np.argmax(norms))
      if p != i:
        a[:, [i, p]] = a[:, [p, i]]
        perm[[i, p]] = perm[[p, i]]
    beta, tau, v = _larfg(a[i, i], a[i + 1:, i])
    a[i, i] = beta
    a[i + 1:, i] = v
    taus[i] = tau
    if i + 1 < n and tau != 0:
      vv = np.concatenate([np.ones((1,), a.dtype), v])
      sub = a[i:, i + 1:]
      a[i:, i + 1:] = sub - np.conj(tau) * np.outer(vv, vv.conj() @ sub)
  return a, (perm + 1).astype(jpvt.dtype), taus


def _orgqr_one(a, taus):
  m, n = a.shape
  k = taus.shape[0]
  q = np.eye(m, n, dtype=a.dtype)
  for i in reversed(range(k)):
    v = np.zeros((m,), a.dtype)
    v[i] = 1
    v[i + 1:] = a[i + 1:, i]
    q[i:, :] = q[i:, :] - taus[i] * np.outer(v[i:], v[i:].conj() @ q[i:, :])
  return q


@register(linalg.geqrf_p)
def _geqrf(x):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  return [o.astype(x.dtype) for o in
          _batch_apply(lambda a: _geqrf_one(a), [_upcast(x, dt)], [2], [dt, dt])]


@register(linalg.geqp3_p)
def _geqp3(x, jpvt, *, use_magma=None):
  x = np.asarray(x)
  jpvt = np.asarray(jpvt)
  dt = _la_dtype(x.dtype)
  lap = _scipy_lapack()

  def one(a, jp):
    if lap is not None and not np.any(jp) and a.size:
      qr, jo, tau, _, info = _lapack_fn(lap, "geqp3", dt)(a)
      return qr, np.asarray(jo).astype(jp.dtype), np.asarray(tau)[:min(a.shape)]
    return _geqp3_one(a, jp)
  outs = _batch_apply(one, [_upcast(x, dt), jpvt], [2, 1], [dt, jpvt.dtype, dt])
  return [outs[0].astype(x.dtype), outs[1], outs[2].astype(x.dtype)]


@register(linalg.householder_product_p)
def _householder_product(a, taus):
  a = np.asarray(a)
  dt = _la_dtype(a.dtype)
  return _batch_apply(lambda a, t: (_orgqr_one(a, t),), [_upcast(a, dt), _upcast(taus, dt)], [2, 1],
                      [dt])[0].astype(a.dtype)


@register(linalg.ormqr_p)
def _ormqr(a, taus, c, *, left, transpose):
  a = np.asarray(a)
  dt = _la_dtype(a.dtype)

  def one(a, taus, c):
    m = a.shape[0]
    k = taus.shape[0]
    c = c.copy()
    t = np.conj(taus) if transpose else taus
    order = range(k - 1, -1, -1) if (left != transpose) else range(k)
    for i in order:
      v = np.zeros((m,), a.dtype)
      v[i] = 1
      v[i + 1:] = a[i + 1:, i]
      if left:
        c = c - t[i] * np.outer(v, v.conj() @ c)
      else:
        c = c - t[i] * np.outer(c @ v, v.conj())
    return (c,)
  return _batch_apply(one, [_upcast(a, dt), _upcast(taus, dt), _upcast(c, dt)],
                      [2, 1, 2], [dt])[0].astype(np.asarray(c).dtype)


@register(linalg.qr_p)
def _qr(x, *, pivoting, full_matrices, use_magma=None):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  m, n = x.shape[-2:]
  kq = m if full_matrices else min(m, n)
  lap = _scipy_lapack()

  def one(a):
    if m == 0 or n == 0:
      q = np.eye(m, kq, dtype=dt)
      r = np.zeros((kq, n), dt)
      return (q, r, np.zeros((n,), np.int32)) if pivoting else (q, r)
    if pivoting:
      jp = np.zeros((n,), np.int32)
      if lap is not None:
        r, p, taus, _, _ = _lapack_fn(lap, "geqp3", dt)(a)
        taus = np.asarray(taus)[:min(m, n)]
      else:
        r, p, taus = _geqp3_one(a, jp)
      p = np.asarray(p).astype(np.int32) - 1
    else:
      r, taus = _geqrf_one(a)
    if m < n:
      q = _orgqr_one(r[:m, :m], taus)
    elif full_matrices:
      qin = np.zeros((m, m), dt)
      qin[:, :n] = r
      q = _orgqr_one(qin, taus)
    else:
      q = _orgqr_one(r, taus)
      r = r[:n, :n]
    r = np.triu(r)
    return (q, r, p) if pivoting else (q, r)
  dts = [dt, dt, np.dtype(np.int32)] if pivoting else [dt, dt]
  outs = _batch_apply(one, [_upcast(x, dt)], [2], dts)
  outs[0] = outs[0].astype(x.dtype)
  outs[1] = outs[1].astype(x.dtype)
  return outs


# ---------------------------------------------------------------------------
# LU

def _cabs1(x):
  return np.abs(x.real) + np.abs(x.imag) if np.iscomplexobj(x) else np.abs(x)


def _getrf_one(a):
  a = a.copy()
  m, n = a.shape
  k = min(m, n)
  piv = np.zeros((k,), np.int32)
  for j in range(k):
    p = j + int(np.argmax(_cabs1(a[j:, j]))) if m - j > 0 else j
    if np.isnan(_cabs1(a[j:, j])).any():
      p = j + int(np.argmax(np.isnan(_cabs1(a[j:, j]))))
    piv[j] = p
    if a[p, j] != 0:
      if p != j:
        a[[j, p], :] = a[[p, j], :]
      a[j + 1:, j] = a[j + 1:, j] / a[j, j]
    if j + 1 < n:
      a[j + 1:, j + 1:] -= np.outer(a[j + 1:, j], a[j, j + 1:])
  return a, piv


def _pivots_to_perm(piv, m):
  perm = np.arange(m, dtype=np.int32)
  for i, p in enumerate(piv):
    if 0 <= p < m:
      perm[[i, p]] = perm[[p, i]]
  return perm


@register(linalg.lu_p)
def _lu(x):
  x = np.asarray(x)
  dt = _la_dtype(x.dtype)
  m = x.shape[-2]

  def one(a):
    lu, piv = _getrf_one(a)
    return lu, piv, _pivots_to_perm(piv, m)
  outs = _batch_apply(one, [_upcast(x, dt)], [2],
                      [dt, np.dtype(np.int32), np.dtype(np.int32)])
  outs[0] = outs[0].astype(x.dtype)
  return outs


@register(linalg.lu_pivots_to_permutation_p)
def _lu_pivots_to_permutation(pivots, *, permutation_size):
  pivots = np.asarray(pivots)
  return _batch_apply(lambda p: (_pivots_to_perm(p, permutation_size),),
                      [pivots], [1], [np.dtype(np.int32)])[0]


# ---------------------------------------------------------------------------
# Triangular / tridiagonal solves, symmetric product

@register(linalg.triangular_solve_p)
def _triangular_solve(a, b, *, left_side, lower, transpose_a, conjugate_a,
                      unit_diagonal):
  a = np.asarray(a)
  b = np.asarray(b)
  dt = _la_dtype(np.result_type(a.dtype, b.dtype) if a.dtype != b.dtype else a.dtype)
  A = _upcast(a, dt)
  B = _upcast(b, dt)
  A = np.tril(A) if lower else np.triu(A)
  n = A.shape[-1]
  if unit_diagonal:
    A = A.copy()
    A[..., np.arange(n), np.arange(n)] = 1
  if conjugate_a:
    A = np.conj(A)
  low = lower
  if transpose_a:
    A = np.swapaxes(A, -1, -2)
    low = not low
  if not left_side:
    # X A = B  <=>  A^T X^T = B^T
    A = np.swapaxes(A, -1, -2)
    B = np.swapaxes(B, -1, -2)
    low = not low
  A = np.broadcast_to(A, B.shape[:-2] + A.shape[-2:])
  X = np.zeros(B.shape, dtype=dt)
  order = range(n) if low else range(n - 1, -1, -1)
  for i in order:
    if low:
      r = B[..., i, :] - np.einsum("...j,...jk->...k", A[..., i, :i], X[..., :i, :])
    else:
      r = B[..., i, :] - np.einsum("...j,...jk->...k", A[..., i, i + 1:], X[..., i + 1:, :])
    X[..., i, :] = r / A[..., i, i][..., None]
  if not left_side:
    X = np.swapaxes(X, -1, -2)
  return X.astype(b.dtype)


def _gtsv(dl, d, du, b, perturb):
  """Batched LAPACK xGTSV (Gaussian elimination with partial pivoting).

  dl, d, du: (batch, n); b: (batch, n, k). dl[:, 0] and du[:, n-1] unused."""
  d = d.copy()
  b = b.copy()
  n = d.shape[-1]
  sub = dl[:, 1:].copy()            # sub[i] = A[i+1, i]
  sup = du[:, :-1].copy()           # sup[i] = A[i, i+1]
  sup2 = np.zeros_like(sub)         # second superdiagonal from interchanges
  bad = np.zeros(d.shape[0], bool)
  if perturb:
    scale = np.maximum(np.max(_cabs1(d), axis=-1, initial=0),
                       np.maximum(np.max(_cabs1(sub), axis=-1, initial=0),
                                  np.max(_cabs1(sup), axis=-1, initial=0)))
    tiny = np.finfo(np.finfo(d.dtype).dtype).eps * np.where(scale > 0, scale, 1)
  for i in range(n - 1):
    swap = _cabs1(d[:, i]) < _cabs1(sub[:, i])
    # No interchange.
    di = d[:, i]
    if perturb:
      di = np.where(di == 0, tiny.astype(d.dtype), di)
      d[:, i] = np.where(swap, d[:, i], di)
    fact_n = sub[:, i] / di
    # Interchange.
    fact_s = d[:, i] / np.where(swap, sub[:, i], 1)
    bad |= ~swap & (di == 0)
    # Updates
    temp = d[:, i + 1].copy()
    new_d_i = np.where(swap, sub[:, i], d[:, i])
    new_d_ip1 = np.where(swap, sup[:, i] - fact_s * temp, temp - fact_n * sup[:, i])
    if i < n - 2:
      new_sup2 = np.where(swap, sup[:, i + 1], 0)
      new_sup_ip1 = np.where(swap, -fact_s * sup[:, i + 1], sup[:, i + 1])
      sup[:, i + 1] = new_sup_ip1
      sup2[:, i] = new_sup2
    new_sup_i = np.where(swap, temp, sup[:, i])
    sup[:, i] = new_sup_i
    d[:, i] = new_d_i
    d[:, i + 1] = new_d_ip1
    bi = b[:, i, :].copy()
    bip1 = b[:, i + 1, :].copy()
    sw = swap[:, None]
    b[:, i, :] = np.where(sw, bip1, bi)
    b[:, i + 1, :] = np.where(sw, bi - fact_s[:, None] * bip1,
                              bip1 - fact_n[:, None] * bi)
  if perturb:
    d = np.where(d == 0, tiny[:, None].astype(d.dtype), d)
  bad |= np.any(d == 0, axis=-1)
  x = b
  x[:, n - 1, :] = x[:, n - 1, :] / d[:, n - 1, None]
  if n > 1:
    x[:, n - 2, :] = (x[:, n - 2, :] - sup[:, n - 2, None] * x[:, n - 1, :]) / d[:, n - 2, None]
  for i in range(n - 3, -1, -1):
    x[:, i, :] = (x[:, i, :] - sup[:, i, None] * x[:, i + 1, :]
                  - sup2[:, i, None] * x[:, i + 2, :]) / d[:, i, None]
  if not perturb:
    x[bad] = np.nan
  return x


@register(linalg.tridiagonal_solve_p)
def _tridiagonal_solve(dl, d, du, b, *, perturb_singular):
  b = np.asarray(b)
  dt = _la_dtype(b.dtype)
  batch = b.shape[:-2]
  n, k = b.shape[-2:]
  if n == 0:
    return b
  flat = lambda v: _upcast(v, dt).reshape((-1, n))
  x = _gtsv(flat(dl), flat(d), flat(du), _upcast(b, dt).reshape((-1, n, k)),
            perturb_singular)
  return x.reshape(b.shape).astype(b.dtype)


@register(linalg.symmetric_product_p)
def _symmetric_product(a, c, *, alpha, beta):
  a = np.asarray(a)
  c = np.asarray(c)
  dt = _la_dtype(a.dtype)
  A = _upcast(a, dt)
  r = alpha * np.matmul(A, np.swapaxes(A, -1, -2)) + beta * _upcast(c, dt)
  return r.astype(c.dtype)
