"""NumPy implementations of elementwise lax primitives and special functions.

Semantics follow XLA on CPU (see shax_backend/__init__.py for the calling
convention). Narrow floats (float16, bfloat16, float8_*, ...) are computed in
float32 and rounded back by the backend, which is what XLA:CPU does for most
ops. Sub-byte integers (int4, uint2, ...) are computed in a wider integer type
and wrapped back to their width.
"""
from __future__ import annotations

import math

import numpy as np

from shax.jax._src import core
from shax.jax._src import dtypes
from shax.jax._src.lax import lax
from shax.jax._src.lax import special
from shax.jax._src.shax_backend import register as _backend_register


_FTZ_DTYPES = ("float32", "float64", "bfloat16", "complex64", "complex128")
_HW_FLOATS = ("float32", "float64", "complex64", "complex128")


def _ftz(a, dtypes_=_FTZ_DTYPES):
  """Flush denormals to (signed) zero, as XLA:CPU does (FTZ + DAZ)."""
  a = np.asarray(a)
  if a.dtype.name not in dtypes_:
    return a
  if _is_complex(a.dtype):
    tiny = np.finfo(a.dtype).tiny
    re = np.where(np.abs(a.real) < tiny, np.copysign(0, a.real), a.real)
    im = np.where(np.abs(a.imag) < tiny, np.copysign(0, a.imag), a.imag)
    out = np.empty(a.shape, a.dtype)
    out.real, out.imag = re, im
    return out
  w = a.astype(np.float32) if a.dtype.name == "bfloat16" else a
  tiny = np.finfo(np.float32 if a.dtype.name == "bfloat16" else a.dtype).tiny
  flush = np.abs(w) < tiny
  if not np.any(flush):
    return a
  return np.where(flush, np.copysign(np.zeros((), w.dtype), w).astype(a.dtype), a)


def register(prim, *, cast: bool = True, ftz: bool | tuple = False):
  """Register an impl; inexact results are cast back to the operand's dtype.

  Impls compute narrow floats in float32 and float32 transcendentals in
  float64; rounding back here (inside the backend's np.errstate) keeps
  overflow-to-inf casts from warning. With `ftz`, denormal float32/float64/
  bfloat16 inputs and outputs are flushed to zero like XLA:CPU arithmetic.
  """
  ftz_dtypes = ftz if isinstance(ftz, tuple) else _FTZ_DTYPES
  flush = lambda a: _ftz(a, ftz_dtypes)  # noqa: E731

  def decorator(fn):
    def impl(*args, **params):
      if ftz:
        args = [flush(a) for a in args]
      out = fn(*args, **params)
      if not cast or not args:
        return flush(out) if ftz else out
      x_dt = np.asarray(args[0]).dtype
      if params.get("out_dtype") is not None:
        x_dt = np.dtype(params["out_dtype"])
      out = np.asarray(out)
      if _is_float(out.dtype) and _is_float(x_dt):
        out = out.astype(x_dt) if out.dtype != x_dt else out
      elif _is_complex(out.dtype) and _is_complex(x_dt):
        out = out.astype(x_dt) if out.dtype != x_dt else out
      elif _is_float(out.dtype) and _is_complex(x_dt):
        rdt = np.finfo(x_dt).dtype
        out = out.astype(rdt) if out.dtype != rdt else out
      return flush(out) if ftz else out
    impl.__name__ = getattr(fn, "__name__", "impl")
    _backend_register(prim)(impl)
    return fn
  return decorator


# ---------------------------------------------------------------------------
# dtype helpers

_NATIVE_FLOATS = (np.dtype(np.float32), np.dtype(np.float64))


def _is_float(dt) -> bool:
  return dtypes.issubdtype(dt, np.floating)


def _is_complex(dt) -> bool:
  return dtypes.issubdtype(dt, np.complexfloating)


def _is_int(dt) -> bool:
  return dtypes.issubdtype(dt, np.integer)


def _is_signed(dt) -> bool:
  return dtypes.issubdtype(dt, np.signedinteger)


def _nbits(dt) -> int:
  return dtypes.itemsize_bits(np.dtype(dt))


def _uint_for_bits(n):
  return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[max(n, 8)]


def _int_for_bits(n):
  return {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[max(n, 8)]


def _c(x):
  """C-contiguous copy/view of `x` that keeps 0-d shapes (unlike ascontiguousarray)."""
  x = np.asarray(x)
  return x if x.flags.c_contiguous else x.copy(order="C")


def _up(x):
  """Promote narrow / non-native floats to float32 (or keep f32/f64/complex)."""
  x = np.asarray(x)
  dt = x.dtype
  if _is_float(dt) and dt not in _NATIVE_FLOATS:
    return x.astype(np.float32)
  return x


def _up_int(x):
  """Sub-byte integers -> int8/uint8 so NumPy arithmetic works."""
  x = np.asarray(x)
  dt = x.dtype
  if _is_int(dt) and _nbits(dt) < 8:
    return x.astype(np.int8 if _is_signed(dt) else np.uint8)
  return x


def _wrap_int(v, dt):
  """Wrap integer values `v` into dtype `dt` (two's complement)."""
  dt = np.dtype(dt)
  n = _nbits(dt)
  if n >= 8:
    return np.asarray(v).astype(dt)
  v = np.asarray(v).astype(np.int64) & ((1 << n) - 1)
  if _is_signed(dt):
    v = np.where(v >= (1 << (n - 1)), v - (1 << n), v)
  return v.astype(dt)


def _unop(prim, fn, *, upcast=True):
  @register(prim, ftz=True)
  def impl(x, **params):
    del params
    x = _up(x) if upcast else np.asarray(x)
    return fn(x)
  return impl


# ---------------------------------------------------------------------------
# conversions


def _float_to_int(x, dt):
  """XLA float->int conversion: truncate, saturate, NaN -> 0."""
  dt = np.dtype(dt)
  info = dtypes.iinfo(dt)
  x = np.asarray(x)
  if x.dtype not in _NATIVE_FLOATS:
    x = x.astype(np.float32)
  lo, hi = int(info.min), int(info.max)
  nan = np.isnan(x)
  t = np.trunc(np.where(nan, 0, x))
  too_hi = t >= float(hi + 1)  # hi + 1 is a power of two, exactly representable
  too_lo = t < float(lo)
  safe = np.where(too_hi | too_lo, 0, t)
  if _nbits(dt) < 8:
    out = safe.astype(np.int64)
  else:
    out = safe.astype(np.int64 if _is_signed(dt) or _nbits(dt) < 64 else np.uint64)
  out = np.where(too_hi, hi, np.where(too_lo, lo, out))
  return _wrap_int(out, dt) if _nbits(dt) < 8 else out.astype(dt)


_FLOAT0 = np.dtype([('float0', np.void, 0)])  # == dtypes.float0


def _convert(x, new_dtype):
  new_dtype = np.dtype(new_dtype)
  x = np.asarray(x)
  old = x.dtype
  if old == new_dtype:
    return x
  if old == _FLOAT0:  # the trivial tangent space holds only zeros
    return np.zeros(x.shape, new_dtype)
  if new_dtype == np.bool_:
    return x != 0
  if _is_complex(old) and not _is_complex(new_dtype):
    x = np.real(x)
    old = x.dtype
  if _is_int(new_dtype):
    if _is_float(old):
      return _float_to_int(x, new_dtype)
    return _wrap_int(_up_int(x), new_dtype)
  if _is_int(old) and _nbits(old) < 8:
    x = _up_int(x)
  if (_is_float(old) and old not in _NATIVE_FLOATS and old != np.float16
      and _is_float(new_dtype) and new_dtype not in _NATIVE_FLOATS):
    # Between two ml_dtypes: go through float32 (exact for the source).
    x = x.astype(np.float32)
  return x.astype(new_dtype)


@register(lax.convert_element_type_p, cast=False)
def _convert_element_type(x, *, new_dtype, weak_type, sharding):
  del weak_type, sharding
  x = np.asarray(x)
  new_dtype = np.dtype(new_dtype)
  # Hardware conversions on XLA:CPU see denormals as zero (and flush f64->f32
  # results); bit-level ones (to/from bfloat16, to complex) keep them.
  src, dst = x.dtype.name, new_dtype.name
  if src == dst:
    return x  # a same-type convert is a no-op in XLA (no denormal flush)
  if (src == "float64" or src == "complex128"
      or (src in ("float32", "complex64")
          and dst != "bfloat16" and not _is_complex(new_dtype))
      or (src == "bfloat16" and dst == "bool")):
    x = _ftz(x)
  out = _convert(x, new_dtype)
  if src in ("float64", "complex128") and dst in ("float32", "bfloat16", "complex64"):
    out = _ftz(out)  # the f64 -> f32 step flushes
  return out


@register(core.stage_p, cast=False)
def _stage(x):
  # Identity: materializes a literal/constant as an Array.
  return np.asarray(x)


def _to_bits(x):
  """Bit patterns of `x` as an unsigned integer array of the dtype's width."""
  x = _c(x)
  n = _nbits(x.dtype)
  if n >= 8:
    return x.view(_uint_for_bits(n))
  if _is_int(x.dtype):
    return (_up_int(x).astype(np.int16) & ((1 << n) - 1)).astype(np.uint8)
  return x.view(np.uint8) & ((1 << n) - 1)


def _from_bits(bits, dt):
  dt = np.dtype(dt)
  n = _nbits(dt)
  if n >= 8:
    return _c(bits.astype(_uint_for_bits(n))).view(dt)
  bits = bits.astype(np.uint8) & ((1 << n) - 1)
  if _is_int(dt):
    return _wrap_int(bits, dt)
  return bits.view(dt)


@register(lax.bitcast_convert_type_p, cast=False)
def _bitcast_convert_type(x, *, new_dtype):
  new_dtype = np.dtype(new_dtype)
  x = np.asarray(x)
  old_n, new_n = _nbits(x.dtype), _nbits(new_dtype)
  if old_n == new_n:
    if old_n >= 8:
      return _c(x).view(new_dtype)
    return _from_bits(_to_bits(x), new_dtype)
  bits = _to_bits(x).astype(np.uint64)
  if old_n > new_n:
    k = old_n // new_n
    shifts = np.arange(k, dtype=np.uint64) * np.uint64(new_n)
    parts = (bits[..., None] >> shifts) & np.uint64((1 << new_n) - 1)
    return _from_bits(parts, new_dtype)
  k = new_n // old_n
  shifts = np.arange(k, dtype=np.uint64) * np.uint64(old_n)
  combined = np.bitwise_or.reduce(bits << shifts, axis=-1)
  return _from_bits(np.asarray(combined, dtype=np.uint64), new_dtype)


@register(lax.reduce_precision_p)
def _reduce_precision(x, *, exponent_bits, mantissa_bits):
  x = np.asarray(x)
  dt = x.dtype
  if dt in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
    w = x
  elif dt == dtypes.bfloat16:
    w = x.astype(np.float32)  # same exponent width; exact
  else:
    w = x.astype(np.float32)
  info = np.finfo(w.dtype)
  src_m, src_e, nbits = int(info.nmant), int(info.nexp), _nbits(w.dtype)
  ut = _uint_for_bits(nbits)
  one = ut(1)
  bits = _c(w).view(ut).copy()
  if mantissa_bits < src_m:
    sh = ut(src_m - mantissa_bits)
    last_bit_mask = one << sh
    base_bias = (last_bit_mask >> one) - one
    last_bit = (bits & last_bit_mask) >> sh
    bits = bits + last_bit + base_bias
    bits = bits & ~(last_bit_mask - one)
  if exponent_bits < src_e:
    exp_mask = ut(((1 << src_e) - 1) << src_m)
    sign_mask = ut(1 << (nbits - 1))
    bias = (1 << (src_e - 1)) - 1
    red_bias = (1 << (exponent_bits - 1)) - 1
    red_max = bias + red_bias
    red_min = bias - red_bias
    x_exp = bits & exp_mask
    overflow = x_exp > ut(red_max << src_m)
    underflow = x_exp <= ut(red_min << src_m)
    signed_zero = bits & sign_mask
    signed_inf = signed_zero | exp_mask
    bits = np.where(overflow, signed_inf, np.where(underflow, signed_zero, bits))
  out = np.asarray(bits, dtype=ut).view(w.dtype)
  isnan = np.isnan(w)
  return np.where(isnan, w, out)


# ---------------------------------------------------------------------------
# arithmetic


def _int_binop(fn):
  def impl(x, y):
    x, y = np.asarray(x), np.asarray(y)
    dt = x.dtype
    if _is_int(dt) and _nbits(dt) < 8:
      return _wrap_int(fn(x.astype(np.int16), y.astype(np.int16)), dt)
    if not (_is_int(dt) or dt == np.bool_):
      x, y = _up(x), _up(y)
    return fn(x, y)
  return impl


@register(lax.add_p, ftz=True)
def _add(x, y, **params):
  del params
  if np.asarray(x).dtype == np.bool_:
    return np.logical_or(x, y)
  return _int_binop(np.add)(x, y)


@register(lax.sub_p, ftz=True)
def _sub(x, y, **params):
  del params
  return _int_binop(np.subtract)(x, y)


@register(lax.mul_p, ftz=True)
def _mul(x, y, *, out_dtype=None):
  if out_dtype is not None and np.dtype(out_dtype) != np.asarray(x).dtype:
    # Multiply in the (wider) output type; the wrapper then keeps that type.
    out_dtype = np.dtype(out_dtype)
    return _mul(_convert(x, out_dtype), _convert(y, out_dtype))
  if np.asarray(x).dtype == np.bool_:
    return np.logical_and(x, y)
  return _int_binop(np.multiply)(x, y)


def _int_divrem(x, y):
  """C-style truncating quotient and remainder with XLA's edge cases."""
  x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
  dt = x.dtype
  small = _nbits(dt) < 8
  if small:
    x, y = x.astype(np.int16), y.astype(np.int16)
  zero = y == 0
  if _is_signed(dt):
    lo = int(dtypes.iinfo(dt).min)
    ovf = (x == lo) & (y == -1)
  else:
    ovf = np.zeros(x.shape, bool)
  safe_y = np.where(zero | ovf, 1, y).astype(x.dtype)
  r = np.fmod(x, safe_y)
  q = (x - r) // safe_y
  if _is_signed(dt):
    q = np.where(zero, -1, np.where(ovf, x, q))
  else:
    q = np.where(zero, np.array(-1).astype(x.dtype) if not small else (1 << _nbits(dt)) - 1, q)
  r = np.where(zero, x, np.where(ovf, 0, r))
  if small:
    return _wrap_int(q, dt), _wrap_int(r, dt)
  return q.astype(dt), r.astype(dt)


@register(lax.div_p, ftz=True)
def _div(x, y, **params):
  del params
  x = np.asarray(x)
  if _is_int(x.dtype):
    return _int_divrem(x, y)[0]
  return np.true_divide(_up(x), _up(y))


@register(lax.rem_p)
def _rem(x, y, **params):
  del params
  x = np.asarray(x)
  if _is_int(x.dtype):
    return _int_divrem(x, y)[1]
  return np.fmod(_up(x), _up(_ftz(y)))  # XLA:CPU flushes a denormal divisor


@register(lax.neg_p)
def _neg(x, **params):
  del params
  x = np.asarray(x)
  if _is_int(x.dtype):
    if _nbits(x.dtype) < 8:
      return _wrap_int(-x.astype(np.int16), x.dtype)
    return np.negative(x)
  return np.negative(_up(x))


@register(lax.abs_p)
def _abs(x, **params):
  del params
  x = np.asarray(x)
  if _is_int(x.dtype):
    if _nbits(x.dtype) < 8:
      return _wrap_int(np.abs(x.astype(np.int16)), x.dtype)
    return np.abs(x)
  if _is_complex(x.dtype):
    return np.abs(x)  # hypot-based, overflow safe
  return np.abs(_up(x))


@register(lax.sign_p, ftz=True)
def _sign(x, **params):
  del params
  x = np.asarray(x)
  if _is_int(x.dtype):
    if _nbits(x.dtype) < 8:
      return _wrap_int(np.sign(x.astype(np.int16)), x.dtype)
    return np.sign(x)
  if _is_complex(x.dtype):
    # XLA: x / |x| (so infinite inputs give nan), 0 at 0.
    a = np.abs(x)
    safe = np.where(a == 0, 1, a)
    out = _mkc(x.real / safe, x.imag / safe).astype(x.dtype)
    return np.where(a == 0, np.zeros_like(x), out)
  x = _up(x)
  return np.where(x == 0, x, np.sign(x))


@register(lax.floor_p, ftz=True)
def _floor(x, **params):
  return np.floor(_up(x))


@register(lax.ceil_p, ftz=True)
def _ceil(x, **params):
  return np.ceil(_up(x))


@register(lax.round_p, ftz=True)
def _round(x, *, rounding_method):
  x = _up(x)
  if lax.RoundingMethod(rounding_method) == lax.RoundingMethod.TO_NEAREST_EVEN:
    return np.rint(x)
  t = np.trunc(x)
  frac = x - t  # exact
  return t + np.where(np.abs(frac) >= 0.5, np.copysign(np.ones_like(x), x), 0).astype(x.dtype)


@register(lax.is_finite_p)
def _is_finite(x):
  return np.isfinite(_up(x))


@register(lax.nextafter_p)
def _nextafter(x, y):
  x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
  dt = x.dtype
  if dt in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
    return np.nextafter(x, y)
  # Bit-level nextafter for IEEE-like ml_dtypes (e.g. bfloat16).
  ut = _uint_for_bits(_nbits(dt))
  xb = _c(x).view(ut)
  xf, yf = x.astype(np.float32), y.astype(np.float32)
  sign_mask = ut(1 << (_nbits(dt) - 1))
  away = (xf < yf) == (xf > 0)  # moving away from zero
  nb = np.where(away, xb + ut(1), xb - ut(1)).astype(ut)
  tiny = np.where(yf < 0, sign_mask | ut(1), ut(1)).astype(ut)
  nb = np.where(xf == 0, tiny, nb).astype(ut)
  out = _c(nb).view(dt)
  out = np.where(xf == yf, y, out)
  out = np.where(np.isnan(xf) | np.isnan(yf), np.array(np.nan, dt), out)
  return out


def _up64(x):
  """Promote to float64/complex128 for transcendental functions.

  XLA's float32 transcendentals are close to correctly rounded; evaluating in
  double and rounding once matches them better than NumPy's float32 loops.
  """
  x = np.asarray(x)
  if _is_complex(x.dtype):
    return x.astype(np.complex128)
  if _is_float(x.dtype):
    return x.astype(np.float64)
  return x


def _mkc(re, im):
  """Complex array from parts without inf * 0 = nan artifacts."""
  re, im = np.broadcast_arrays(np.asarray(re), np.asarray(im))
  out = np.empty(re.shape, np.result_type(re.dtype, np.complex64))
  out.real = re
  out.imag = im
  return out


def _nan_c(dt):
  return np.array(complex(np.nan, np.nan), dt)


def _cexp(z):
  # e^(a+bi) = e^a (cos b + i sin b); imag is exactly 0 when b == 0. Unlike
  # C99, XLA gives nan + nan i for infinite b.
  a, b = z.real, z.imag
  ea = np.exp(a)
  re = ea * np.cos(b)
  im = np.where(b == 0, b, ea * np.sin(b))
  re = np.where(b == 0, ea, re)
  out = _mkc(re, im)
  return np.where(np.isinf(b), _nan_c(out.dtype), out)


def _cexpm1(z):
  a, b = z.real, z.imag
  em1 = np.expm1(a)
  sinh = np.sin(0.5 * b)
  re = em1 * np.cos(b) - 2 * sinh * sinh
  im = np.where(b == 0, b, np.exp(a) * np.sin(b))
  re = np.where(b == 0, em1, re)
  out = _mkc(re, im)
  return np.where(np.isinf(b), _nan_c(out.dtype), out)


def _clog1p(z):
  a, b = z.real, z.imag
  m = np.maximum(np.abs(1 + a), np.abs(b))
  small = a * (2 + a) + b * b
  use_small = (np.abs(a) < 0.5) & (np.abs(b) < 0.5)
  safe_m = np.where(m == 0, 1, m)
  q = np.minimum(np.abs(1 + a), np.abs(b)) / safe_m
  q = np.where(np.isinf(m), 0, q)
  big = np.log(m) + 0.5 * np.log1p(q * q)
  re = np.where(use_small, 0.5 * np.log1p(small), big)
  im = np.arctan2(b, 1 + a)
  return _mkc(re, im)


def _both_inf(fn):
  def impl(z):
    out = fn(z)
    bad = np.isinf(z.real) & np.isinf(z.imag)
    return np.where(bad, _nan_c(out.dtype), out)
  return impl


def _ctan(z):
  out = np.tan(z)
  return np.where(np.isinf(z.real) & ~np.isinf(z.imag), _nan_c(out.dtype), out)


_COMPLEX_OVERRIDES = {
    np.exp: _cexp,
    np.exp2: lambda z: _cexp(z * np.log(2)),
    np.expm1: _cexpm1,
    np.log1p: _clog1p,
    np.sin: _both_inf(np.sin), np.cos: _both_inf(np.cos),
    np.sinh: _both_inf(np.sinh), np.cosh: _both_inf(np.cosh),
    np.tan: _ctan,
}


def _transcendental(prim, fn):
  cfn = _COMPLEX_OVERRIDES.get(fn, fn)
  @register(prim, ftz=True)
  def impl(x, **params):
    del params
    if fn is np.exp2 and _is_float(np.asarray(x).dtype):
      # JAX lowers exp2(x) as exp(x * log(2)) computed in x's dtype.
      xd = np.asarray(x).dtype
      up = _up(x)
      ln2 = _up(np.array(np.log(2), xd))  # the constant is rounded to xd too
      x = (up * ln2).astype(xd)
      return np.exp(_up64(x))
    x = _up64(x)
    return cfn(x) if _is_complex(x.dtype) else fn(x)
  return impl


for _p, _f in [
    (lax.exp_p, np.exp), (lax.exp2_p, np.exp2), (lax.log_p, np.log),
    (lax.log2_p, np.log2), (lax.expm1_p, np.expm1), (lax.log1p_p, np.log1p),
    (lax.tanh_p, np.tanh), (lax.sin_p, np.sin), (lax.cos_p, np.cos),
    (lax.tan_p, np.tan), (lax.asin_p, np.arcsin), (lax.acos_p, np.arccos),
    (lax.atan_p, np.arctan), (lax.sinh_p, np.sinh), (lax.cosh_p, np.cosh),
    (lax.asinh_p, np.arcsinh), (lax.acosh_p, np.arccosh),
    (lax.atanh_p, np.arctanh), (lax.cbrt_p, np.cbrt),
]:
  _transcendental(_p, _f)

_unop(lax.sqrt_p, np.sqrt)


@register(lax.rsqrt_p, ftz=True)
def _rsqrt(x, **params):
  x = _up(x)
  return np.ones((), x.dtype) / np.sqrt(x)


@register(lax.logistic_p, ftz=True)
def _logistic(x, **params):
  x = _up(x)
  if _is_complex(x.dtype):
    return 1 / (1 + np.exp(-x))
  e = np.exp(-np.abs(x))
  one = np.ones((), x.dtype)
  return np.where(x >= 0, one / (one + e), e / (one + e))


@register(lax.atan2_p, ftz=True)
def _atan2(x, y):
  x, y = _up(x), _up(y)
  if _is_complex(x.dtype) or _is_complex(y.dtype):
    # atan2(x, y) = -i * log((y + i x) / sqrt(x^2 + y^2))
    return -1j * np.log((y + 1j * x) / np.sqrt(x * x + y * y))
  return np.arctan2(x, y)


@register(lax.square_p, ftz=True)
def _square(x, **params):
  x = np.asarray(x)
  if _is_int(x.dtype):
    return _int_binop(np.multiply)(x, x)
  if _is_complex(x.dtype):
    a, b = x.real, x.imag
    re = np.where((np.abs(a) == np.abs(b)) & np.isfinite(a), 0,
                  (a - b) * (a + b)).astype(a.dtype)
    out = np.empty(x.shape, x.dtype)
    out.real = re
    out.imag = (a * b) * 2
    return out
  x = _up(x)
  return x * x


@register(lax.one_minus_square_p, ftz=True)
def _one_minus_square(x, **params):
  x = np.asarray(x)
  if _is_int(x.dtype):
    one = np.ones((), x.dtype)
    return _int_binop(np.multiply)(_int_binop(np.add)(one, x),
                                   _int_binop(np.subtract)(one, x))
  x = _up(x)
  one = np.ones((), x.dtype)
  return (one + x) * (one - x)


@register(lax.polynomial_p, ftz=True)
def _polynomial(x, *coeffs, unroll=None):
  del unroll
  x = _up(x)
  cs = [_up(c) for c in coeffs]
  acc = cs[-1]
  for c in reversed(cs[:-1]):
    acc = acc * x + c
  return np.broadcast_to(acc, np.broadcast_shapes(x.shape, *(c.shape for c in cs)))


@register(lax.pow_p, ftz=True)
def _pow(x, y):
  x = _up(x)
  y = np.asarray(y)
  y = y.astype(x.dtype) if y.dtype != x.dtype else _up(y)
  if _is_complex(x.dtype):
    out = np.power(x, y)
    xz = x == 0
    yz = y == 0
    out = np.where(xz & yz, np.ones((), x.dtype), out)
    out = np.where(xz & ~yz & (y.real > 0) & (y.imag == 0),
                   np.zeros((), x.dtype), out)
    return out
  return np.power(x, y)


@register(lax.integer_pow_p, ftz=True)
def _integer_pow(x, *, y):
  x = np.asarray(x)
  dt = x.dtype
  if _is_int(dt):
    mul = _int_binop(np.multiply)
  else:
    x = _up(x)
    mul = np.multiply
  if y == 0:
    return np.ones_like(x)
  recip = y < 0
  y = -y if recip else y
  acc = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else mul(acc, x)
    y >>= 1
    if y > 0:
      x = mul(x, x)
  return np.ones((), acc.dtype) / acc if recip else acc


def _minmax(is_max):
  def impl(x, y):
    x, y = np.asarray(x), np.asarray(y)
    dt = x.dtype
    if _is_complex(dt):
      rx, ry = x.real, y.real
      if is_max:
        which = np.where(rx == ry, x.imag > y.imag, rx > ry)
      else:
        which = np.where(rx == ry, x.imag < y.imag, rx < ry)
      return np.where(which, x, y)
    if _is_int(dt) or dt == np.bool_:
      return np.maximum(x, y) if is_max else np.minimum(x, y)
    x, y = _up(x), _up(y)
    out = np.maximum(x, y) if is_max else np.minimum(x, y)
    # XLA orders -0 < +0.
    eq = x == y
    if is_max:
      pick = np.where(np.signbit(x), y, x)
    else:
      pick = np.where(np.signbit(x), x, y)
    return np.where(eq, pick, out)
  return impl


register(lax.max_p, ftz=_HW_FLOATS)(_minmax(True))
register(lax.min_p, ftz=_HW_FLOATS)(_minmax(False))


@register(lax.clamp_p, ftz=_HW_FLOATS)
def _clamp(lo, x, hi):
  mx, mn = _minmax(True), _minmax(False)
  return mn(mx(np.asarray(x), np.asarray(lo)), np.asarray(hi))


@register(lax.select_n_p)
def _select_n(which, *cases):
  which = np.asarray(which)
  if len(cases) == 1:
    return cases[0]
  shape = np.broadcast_shapes(which.shape, *(np.shape(c) for c in cases))
  if which.dtype == np.bool_:
    return np.where(which, cases[1], cases[0]).reshape(shape)
  # XLA lowers to a tree of selects, so out-of-range indices clamp.
  which = np.clip(which, 0, len(cases) - 1)
  out = np.broadcast_to(cases[0], shape)
  for i in range(1, len(cases)):
    out = np.where(which == i, cases[i], out)
  return out


@register(lax.complex_p)
def _complex(x, y):
  x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
  ct = np.complex128 if x.dtype == np.float64 else np.complex64
  out = np.empty(x.shape, ct)
  out.real = x
  out.imag = y
  return out


@register(lax.conj_p)
def _conj(x, *, input_dtype=None):
  x = np.asarray(x)
  if not _is_complex(x.dtype):
    return x.astype(np.complex128 if x.dtype == np.float64 else np.complex64)
  return np.conj(x)


@register(lax.real_p)
def _real(x):
  return np.real(x)


@register(lax.imag_p)
def _imag(x):
  return np.imag(x)


# ---------------------------------------------------------------------------
# bitwise / integer


def _bitwise(fn, bool_fn):
  def impl(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype == np.bool_:
      return bool_fn(x, y)
    dt = x.dtype
    if _nbits(dt) < 8:
      return _wrap_int(fn(_up_int(x), _up_int(y)), dt)
    return fn(x, y)
  return impl


register(lax.and_p)(_bitwise(np.bitwise_and, np.logical_and))
register(lax.or_p)(_bitwise(np.bitwise_or, np.logical_or))
register(lax.xor_p)(_bitwise(np.bitwise_xor, np.logical_xor))


@register(lax.not_p)
def _not(x):
  x = np.asarray(x)
  if x.dtype == np.bool_:
    return np.logical_not(x)
  if _nbits(x.dtype) < 8:
    return _wrap_int(np.invert(_up_int(x)), x.dtype)
  return np.invert(x)


def _shift(kind):
  def impl(x, y):
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    dt = x.dtype
    n = _nbits(dt)
    mask = np.uint64((1 << n) - 1) if n < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
    xb = _to_bits(x).astype(np.uint64)
    yb = _to_bits(y).astype(np.uint64)  # negative shifts are huge unsigned
    oob = yb >= np.uint64(n)
    s = np.where(oob, np.uint64(0), yb).astype(np.uint64)
    if kind == "left":
      r = np.where(oob, np.uint64(0), (xb << s) & mask)
    elif kind == "logical":
      r = np.where(oob, np.uint64(0), xb >> s)
    else:
      xs = xb.astype(np.int64) if n == 64 else xb.view(np.int64)
      if n < 64:
        xs = np.where(xs >= (1 << (n - 1)), xs - (1 << n), xs)
      r = np.where(oob, np.where(xs < 0, -1, 0), xs >> s.astype(np.int64))
      r = r.astype(np.int64).view(np.uint64) & mask
    return _from_bits(np.asarray(r, dtype=np.uint64), dt)
  return impl


register(lax.shift_left_p)(_shift("left"))
register(lax.shift_right_logical_p)(_shift("logical"))
register(lax.shift_right_arithmetic_p)(_shift("arith"))


@register(lax.population_count_p)
def _population_count(x):
  x = np.asarray(x)
  if _is_signed(x.dtype) and _nbits(x.dtype) < 8:
    # XLA counts bits of the sign-extended 8-bit storage value.
    return _wrap_int(np.bitwise_count(_up_int(x).view(np.uint8)), x.dtype)
  return _from_bits(np.bitwise_count(_to_bits(x)).astype(np.uint64), x.dtype)


@register(lax.clz_p)
def _clz(x):
  x = np.asarray(x)
  n = _nbits(x.dtype)
  b = _to_bits(x).astype(np.uint64)
  out = np.full(b.shape, n, np.int64)
  for i in range(n):
    out = np.where((out == n) & ((b >> np.uint64(n - 1 - i)) & np.uint64(1)).astype(bool),
                   i, out)
  return _from_bits(out.astype(np.uint64), x.dtype)


@register(lax.mulhi_p)
def _mulhi(x, y):
  x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
  dt = x.dtype
  n = _nbits(dt)
  signed = _is_signed(dt)
  if n <= 32:
    wide = np.int64 if signed else np.uint64
    p = _up_int(x).astype(wide) * _up_int(y).astype(wide)
    return _wrap_int(p >> wide(n), dt)
  # 64-bit: exact via Python ints.
  xs, ys = x.ravel().tolist(), y.ravel().tolist()
  hi = [((a * b) >> 64) for a, b in zip(xs, ys)]
  hi = [(h + (1 << 64)) % (1 << 64) for h in hi]
  return np.array(hi, np.uint64).view(np.int64 if signed else np.uint64).reshape(x.shape)


# ---------------------------------------------------------------------------
# comparisons


def _cmp(fn):
  def impl(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if _is_float(x.dtype):
      x, y = _up(x), _up(y)
    elif _is_int(x.dtype) and _nbits(x.dtype) < 8:
      x, y = _up_int(x), _up_int(y)
    return fn(x, y)
  return impl


for _p, _f in [(lax.eq_p, np.equal), (lax.ne_p, np.not_equal),
               (lax.ge_p, np.greater_equal), (lax.gt_p, np.greater),
               (lax.le_p, np.less_equal), (lax.lt_p, np.less)]:
  register(_p, ftz=True)(_cmp(_f))


def _total_order_key(x):
  """Signed integer key whose order is IEEE totalOrder on floats."""
  x = np.asarray(x)
  n = _nbits(x.dtype)
  st = _int_for_bits(n)
  k = _c(x).view(st) if n >= 8 else _to_bits(x).astype(np.int8)
  return np.where(k < 0, k ^ st(np.iinfo(st).max), k)


def _cmp_to(fn):
  def impl(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if _is_complex(x.dtype):
      kx = (_total_order_key(x.real), _total_order_key(x.imag))
      ky = (_total_order_key(y.real), _total_order_key(y.imag))
      if fn is np.equal:
        return (kx[0] == ky[0]) & (kx[1] == ky[1])
      lt = (kx[0] < ky[0]) | ((kx[0] == ky[0]) & (kx[1] < ky[1]))
      return lt if fn is np.less else lt | ((kx[0] == ky[0]) & (kx[1] == ky[1]))
    if _is_float(x.dtype):
      return fn(_total_order_key(x), _total_order_key(y))
    return _cmp(fn)(x, y)
  return impl


register(lax.eq_to_p)(_cmp_to(np.equal))
register(lax.le_to_p)(_cmp_to(np.less_equal))
register(lax.lt_to_p)(_cmp_to(np.less))


# ---------------------------------------------------------------------------
# special functions


def _scipy_special():
  try:
    import scipy.special as sc  # noqa: PLC0415
  except ImportError:  # pragma: no cover
    return None
  return sc


def _vectorize_math(fn, *args):
  return np.vectorize(fn, otypes=[np.float64])(*args)


def _lgamma_np(x):
  sc = _scipy_special()
  if sc is not None:
    return sc.gammaln(x)
  return _vectorize_math(
      lambda v: math.inf if v <= 0 and v == math.floor(v) else math.lgamma(v), x)


def _digamma_np(x):
  sc = _scipy_special()
  out = sc.psi(x)
  out = np.where(x == 0, -np.inf, out)
  return np.where((x < 0) & (x == np.floor(x)), np.nan, out)


@register(special.lgamma_p)
def _lgamma(x):
  x = _up(x)
  # scipy gives lgamma(-inf) = -inf; XLA (and |Gamma| -> 0 being log -> +inf
  # along the poles) gives +inf.
  return np.where(np.isneginf(x), np.inf, _lgamma_np(x))


@register(special.digamma_p)
def _digamma(x):
  return _digamma_np(_up(x))


@register(special.erf_p)
def _erf(x):
  x = _up(x)
  sc = _scipy_special()
  return sc.erf(x) if sc is not None else _vectorize_math(math.erf, x)


@register(special.erfc_p)
def _erfc(x, **params):
  x = _up(x)
  sc = _scipy_special()
  return sc.erfc(x) if sc is not None else _vectorize_math(math.erfc, x)


@register(special.erf_inv_p)
def _erf_inv(x):
  return _scipy_special().erfinv(_up(x))


@register(special.bessel_i0e_p)
def _bessel_i0e(x):
  return _scipy_special().i0e(_up(x))


@register(special.bessel_i1e_p)
def _bessel_i1e(x):
  return _scipy_special().i1e(_up(x))


def _zeta_np(x, q):
  sc = _scipy_special()
  out = sc.zeta(x, q)
  out = np.where(x == 1, np.inf, out)
  out = np.where(x < 1, np.nan, out)
  return out


@register(special.zeta_p)
def _zeta(x, q):
  return _zeta_np(_up(x), _up(q))


@register(special.polygamma_p)
def _polygamma(m, x):
  m, x = np.broadcast_arrays(_up(m), _up(x))
  sc = _scipy_special()
  is_int = m == np.floor(m)
  mm = np.where(is_int & (m >= 0), m, 1)
  fact = sc.gamma(mm + 1)
  sign = np.where(np.fmod(mm, 2) == 0, -1.0, 1.0)
  out = sign * fact * _zeta_np(mm + 1, x)
  out = np.where(m == 0, _digamma_np(x), out)
  return np.where(is_int & (m >= 0), out, np.nan)


@register(special.regularized_incomplete_beta_p)
def _regularized_incomplete_beta(a, b, x):
  a, b, x = _up(a), _up(b), _up(x)
  out = _scipy_special().betainc(a, b, x)
  bad = (a <= 0) | (b <= 0) | (x < 0) | (x > 1)
  return np.where(bad, np.nan, out)


# igamma family: NumPy ports of JAX's series / continued-fraction algorithms
# (jax/_src/lax/special.py), which XLA lowers directly, so numerics match.

_VALUE, _DERIVATIVE = 0, 1


def _igamma_series(ax, x, a, enabled, dt, mode):
  eps = np.finfo(dt).eps
  r = a.copy()
  c = np.ones_like(a)
  ans = np.ones_like(a)
  dc_da = np.zeros_like(a)
  dans_da = np.zeros_like(a)
  enabled = enabled.copy()
  while np.any(enabled):
    r_n = r + 1
    dc_da_n = dc_da * (x / r_n) - (c * x) / (r_n * r_n)
    dans_da_n = dans_da + dc_da_n
    c_n = c * (x / r_n)
    ans_n = ans + c_n
    if mode == _VALUE:
      cond = enabled & (c_n / ans_n > eps)
    else:
      cond = enabled & (np.abs(dc_da_n / dans_da_n) > eps)
    r = np.where(enabled, r_n, r)
    c = np.where(enabled, c_n, c)
    ans = np.where(enabled, ans_n, ans)
    dc_da = np.where(enabled, dc_da_n, dc_da)
    dans_da = np.where(enabled, dans_da_n, dans_da)
    enabled = cond
  if mode == _VALUE:
    return (ans * ax) / a
  dlogax_da = np.log(x) - _digamma_np(a + 1).astype(dt)
  return ax * (ans * dlogax_da + dans_da) / a


def _igammac_cf(ax, x, a, enabled, dt, mode):
  eps = np.finfo(dt).eps
  y = 1 - a
  z = x + y + 1
  c = dt.type(0)
  pkm2 = np.ones_like(x)
  qkm2 = x.copy()
  pkm1 = x + 1
  qkm1 = z * x
  ans = pkm1 / qkm1
  t = np.ones_like(x)
  dpkm2_da = np.zeros_like(x)
  dqkm2_da = np.zeros_like(x)
  dpkm1_da = np.zeros_like(x)
  dqkm1_da = -x
  dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1
  enabled = enabled.copy()
  while c < 2000 and np.any(enabled):
    c = c + dt.type(1)
    y_n = y + 1
    z_n = z + 2
    yc = y_n * c
    pk = pkm1 * z_n - pkm2 * yc
    qk = qkm1 * z_n - qkm2 * yc
    nz = qk != 0
    r = pk / qk
    t_n = np.where(nz, np.abs((ans - r) / r), 1)
    ans_n = np.where(nz, r, ans)
    dpk_da = dpkm1_da * z_n - pkm1 - dpkm2_da * yc + pkm2 * c
    dqk_da = dqkm1_da * z_n - qkm1 - dqkm2_da * yc + qkm2 * c
    dans_da_new = np.where(nz, (dpk_da - ans_n * dqk_da) / qk, dans_da)
    grad_cond = np.where(nz, np.abs(dans_da_new - dans_da), 1)
    n_pkm2, n_pkm1, n_qkm2, n_qkm1 = pkm1, pk, qkm1, qk
    n_dpkm2, n_dqkm2, n_dpkm1, n_dqkm1 = dpkm1_da, dqkm1_da, dpk_da, dqk_da
    rescale = np.abs(pk) > 1 / eps
    n_pkm2, n_pkm1, n_qkm2, n_qkm1, n_dpkm2, n_dqkm2, n_dpkm1, n_dqkm1 = (
        np.where(rescale, v * eps, v) for v in
        (n_pkm2, n_pkm1, n_qkm2, n_qkm1, n_dpkm2, n_dqkm2, n_dpkm1, n_dqkm1))
    if mode == _VALUE:
      cond = enabled & (t_n > eps)
    else:
      cond = enabled & (grad_cond > eps)
    sel = lambda new, old: np.where(enabled, new, old).astype(dt)  # noqa: E731
    ans, t, y, z = sel(ans_n, ans), sel(t_n, t), sel(y_n, y), sel(z_n, z)
    pkm1, qkm1 = sel(n_pkm1, pkm1), sel(n_qkm1, qkm1)
    pkm2, qkm2 = sel(n_pkm2, pkm2), sel(n_qkm2, qkm2)
    dpkm2_da, dqkm2_da = sel(n_dpkm2, dpkm2_da), sel(n_dqkm2, dqkm2_da)
    dpkm1_da, dqkm1_da = sel(n_dpkm1, dpkm1_da), sel(n_dqkm1, dqkm1_da)
    dans_da = sel(dans_da_new, dans_da)
    enabled = cond
  if mode == _VALUE:
    return ans * ax
  dlogax_da = np.log(x) - _digamma_np(a).astype(dt)
  return ax * (ans * dlogax_da + dans_da)


def _igamma_args(a, x):
  a, x = np.broadcast_arrays(_up(a), _up(x))
  dt = np.result_type(a, x)
  return a.astype(dt), x.astype(dt), dt


def _ax(a, x, dt):
  ax = a * np.log(x) - x - _lgamma_np(a).astype(dt)
  underflow = ax < -np.log(np.finfo(dt).max)
  return np.exp(ax).astype(dt), underflow


@register(special.igamma_p)
def _igamma(a, x):
  a, x, dt = _igamma_args(a, x)
  is_nan = np.isnan(a) | np.isnan(x)
  x_inf, a_inf = x == np.inf, a == np.inf
  a_zero, x_zero = a == 0, x == 0
  domain = (x < 0) | (a < 0) | (a_zero & x_zero) | (a_inf & x_inf) | is_nan
  use_c = (x >= 1) & (x > a)
  ax, underflow = _ax(a, x, dt)
  enabled = ~(x_zero | domain | underflow | x_inf | a_inf)
  out = np.where(use_c, 1 - _igammac_cf(ax, x, a, enabled & use_c, dt, _VALUE),
                 _igamma_series(ax, x, a, enabled & ~use_c, dt, _VALUE))
  out = np.where(a_inf, 0, out)
  out = np.where(x_zero, 0, out)
  out = np.where(x_inf, 1, out)
  return np.where(domain, np.nan, out)


@register(special.igammac_p)
def _igammac(a, x):
  a, x, dt = _igamma_args(a, x)
  is_nan = np.isnan(a) | np.isnan(x)
  x_inf, a_inf = x == np.inf, a == np.inf
  a_zero, x_zero = a == 0, x == 0
  domain = (x < 0) | (a < 0) | (a_zero & x_zero) | (a_inf & x_inf) | is_nan
  use_ig = (x < 1) | (x < a)
  ax, underflow = _ax(a, x, dt)
  enabled = ~(domain | underflow | x_inf | a_zero | a_inf)
  out = np.where(use_ig, 1 - _igamma_series(ax, x, a, enabled & use_ig, dt, _VALUE),
                 _igammac_cf(ax, x, a, enabled & ~use_ig, dt, _VALUE))
  out = np.where(a_inf, 1, out)
  out = np.where(x_inf | a_zero, 0, out)
  return np.where(domain, np.nan, out)


@register(special.igamma_grad_a_p)
def _igamma_grad_a(a, x):
  a, x, dt = _igamma_args(a, x)
  is_nan = np.isnan(a) | np.isnan(x)
  x_zero = x == 0
  domain = (x < 0) | (a <= 0)
  use_c = (x > 1) & (x > a)
  ax, underflow = _ax(a, x, dt)
  enabled = ~(x_zero | domain | underflow | is_nan)
  out = np.where(use_c, -_igammac_cf(ax, x, a, enabled & use_c, dt, _DERIVATIVE),
                 _igamma_series(ax, x, a, enabled & ~use_c, dt, _DERIVATIVE))
  out = np.where(x_zero, 0, out)
  return np.where(domain | is_nan, np.nan, out)
