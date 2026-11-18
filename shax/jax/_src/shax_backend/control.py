"""NumPy implementations of control flow, PRNG, callback, state and misc
primitives (see shax_backend/__init__.py).

Higher-order primitives (cond, while, scan) are registered with `raw=True` and
evaluate their sub-jaxprs eagerly with `eval_jaxpr_compiled` (XLA compiles
the whole loop even when called eagerly), so the body's own
primitives dispatch back through the backend. Values with extended dtypes
(PRNG key arrays) are handled through their physical (uint32) base arrays.

Bit-exactness notes:
- threefry2x32 / threefry4x32 / philox{2,4}x32 follow the reference algorithms
  in shax/jax/_src/random/*.py with wrapping uint32 arithmetic.
- rng_bit_generator follows XLA's RngBitGeneratorExpander + lib/prng.cc
  (Apache-2.0, The OpenXLA Authors): RNG_DEFAULT is Philox on CPU.
"""
from __future__ import annotations

import math

import numpy as np

from shax.jax._src import core
from shax.jax._src import dispatch
from shax.jax._src import dtypes
from shax.jax._src import earray
from shax.jax._src import effects
from shax.jax._src.lax import lax
from shax.jax._src.lax import ann
from shax.jax._src.lax import parallel
from shax.jax._src.lax.control_flow import conditionals
from shax.jax._src.lax.control_flow import loops
from shax.jax._src.random import prng
from shax.jax._src.random import core as random_core
from shax.jax._src.random import threefry2x32 as _tf2
from shax.jax._src.random import threefry4x32 as _tf4
from shax.jax._src.random import philox2x32 as _ph2
from shax.jax._src.random import philox4x32 as _ph4
from shax.jax._src import callback as callback_lib
from shax.jax._src import compute_on as compute_on_lib
from shax.jax._src import ffi
from shax.jax._src import xla_metadata as xla_metadata_lib
from shax.jax._src.state import primitives as state_primitives
from shax.jax._src.state import discharge as state_discharge
from shax.jax._src.shax_backend import (
    register, to_numpy, from_numpy, new_token, ext_physical, ext_wrap,
    eval_jaxpr_compiled)


# ---------------------------------------------------------------------------
# Helpers for values that may carry extended (PRNG key) dtypes.

def _is_key_aval(aval) -> bool:
  """Extended-dtype aval (PRNG keys, custom edtypes): stored physically."""
  return (isinstance(aval, core.ShapedArray)
          and dtypes.issubdtype(aval.dtype, dtypes.extended))


def _phys(x) -> np.ndarray:
  """Host value of `x`; PRNG key arrays become their uint32 base arrays."""
  if isinstance(x, (prng.PRNGKeyArray, earray.EArray)) or (
      not isinstance(x, (np.ndarray, np.generic)) and _is_key_aval(core.typeof(x))):
    return to_numpy(ext_physical(x))
  return np.asarray(to_numpy(x))


def _wrap(val, aval):
  """Wrap a host value (physical for keys) as a JAX value of type `aval`."""
  if _is_key_aval(aval):
    if not isinstance(val, (np.ndarray, np.generic)) and _is_key_aval(core.typeof(val)):
      return val
    phys = core.physical_aval(aval)
    return ext_wrap(np.asarray(val, dtype=phys.dtype).reshape(phys.shape), aval)
  if isinstance(aval, core.ShapedArray):
    if type(val).__name__ == "ArrayImpl" and val.aval == aval:
      return val
    return from_numpy(_phys(val), aval)
  return val  # tokens, refs, ...


# ---------------------------------------------------------------------------
# Control flow

def _compiled_fun(closed_jaxpr):
  return lambda *args: eval_jaxpr_compiled(
      closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


@register(conditionals.cond_p, raw=True)
def _cond(index, *args, branches, **_):
  i = int(np.asarray(to_numpy(index)))
  i = min(max(i, 0), len(branches) - 1)
  return _compiled_fun(branches[i])(*args)


@register(xla_metadata_lib.xla_metadata_call_p, raw=True)
def _xla_metadata_call(*args, jaxpr, **_):
  # The metadata only annotates the HLO; run the body.
  return _compiled_fun(jaxpr)(*args)


@register(compute_on_lib.compute_on_p, raw=True)
def _compute_on(*args, jaxpr, out_memory_spaces, **_):
  # shax computes everything on its one CPU device; outputs land in the
  # requested memory spaces (by their avals).
  if dispatch.jaxpr_has_primitive(jaxpr, "compute_on"):
    raise ValueError("Nesting `compute_on` with different compute types is "
                     "not allowed.")
  outs = _compiled_fun(jaxpr)(*args)
  return [_wrap(_phys(o), a.update(memory_space=s))
          if isinstance(a, core.ShapedArray) else o
          for o, a, s in zip(outs, jaxpr.out_avals, out_memory_spaces)]


@register(conditionals.platform_index_p)
def _platform_index(*, platforms, **_):
  default = None
  for i, ps in enumerate(platforms):
    if ps is None:
      default = i
    elif "cpu" in ps:
      return np.int32(i)
  if default is None:
    raise NotImplementedError(
        f"translation rule for platform_index(platforms={platforms}) not "
        "found for platform cpu")
  return np.int32(default)


def _bcast_select(pred, new, old):
  pred = pred.reshape(pred.shape + (1,) * (new.ndim - pred.ndim))
  return np.where(pred, new, old)


@register(loops.while_p, raw=True)
def _while(*args, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts):
  cconsts = args[:cond_nconsts]
  bconsts = args[cond_nconsts:cond_nconsts + body_nconsts]
  carry = list(args[cond_nconsts + body_nconsts:])
  cond_f = _compiled_fun(cond_jaxpr)
  body_f = _compiled_fun(body_jaxpr)
  batched = bool(cond_jaxpr.out_avals[0].shape)
  cond_ordered = bool(effects.ordered_effects.filter_in(cond_jaxpr.effects))
  while True:
    pred = _phys(cond_f(*cconsts, *carry)[0])
    if not batched:
      if not bool(pred):
        return carry
      carry = list(body_f(*bconsts, *carry))
    elif cond_ordered:
      # Batched predicate with ordered effects in cond: like upstream's
      # lowering, run the body on all lanes while any is live (no select).
      if not pred.any():
        return carry
      carry = list(body_f(*bconsts, *carry))
    else:
      # Batched predicate (from vmap): run while any lane is live, and only
      # update the live lanes. Like XLA's lowering (reduce_or in the cond
      # block; the body re-evaluates cond on the old carry and selects), so
      # effects in cond happen the same number of times.
      if not pred.any():
        return carry
      new = body_f(*bconsts, *carry)
      body_pred = _phys(cond_f(*cconsts, *carry)[0])
      carry = [_wrap(_bcast_select(body_pred, _phys(n), _phys(c)), aval)
               for n, c, aval in zip(new, carry, body_jaxpr.out_avals)]


@register(loops.scan_p, raw=True)
def _scan(*args, reverse, length, ft_in, ft_out, jaxpr, **_):
  consts, carry, xs = map(list, ft_in.update(args).unpack())
  _, y_avals = ft_out.update(jaxpr.out_avals).unpack()
  y_avals = list(y_avals)
  f = _compiled_fun(jaxpr)
  xs_np = [_phys(x) for x in xs]
  x_avals = [core.typeof(x) for x in xs]
  x_avals = [a.update(shape=a.shape[1:]) for a in x_avals]
  ys = []
  for it in range(length):
    i = length - 1 - it if reverse else it
    x_i = [_wrap(xn[i], xa) for xn, xa in zip(xs_np, x_avals)]
    out = f(*consts, *carry, *x_i)
    carry, y = map(list, ft_out.update(out).unpack())
    ys.append([_phys(v) for v in y])
  if reverse:
    ys.reverse()
  stacked = []
  for k, aval in enumerate(y_avals):
    phys = core.physical_aval(aval) if _is_key_aval(aval) else aval
    if ys:
      arr = np.stack([np.asarray(y[k], dtype=phys.dtype) for y in ys])
    else:
      arr = np.zeros((0, *phys.shape), dtype=phys.dtype)
    stacked.append(_wrap(arr, aval.update(shape=(length, *aval.shape))))
  return [*carry, *stacked]


# ---------------------------------------------------------------------------
# PRNG hash functions (wrapping uint32 arithmetic)

_U32 = np.uint32


def _u32(*xs):
  return [np.array(a, dtype=_U32, copy=True) for a in np.broadcast_arrays(*xs)]


def _rotl(x, d):
  d = int(d)
  return (x << _U32(d)) | (x >> _U32(32 - d))


def _mulhilo(m, x):
  prod = x.astype(np.uint64) * np.uint64(m)
  return (prod & np.uint64(0xFFFFFFFF)).astype(_U32), (prod >> np.uint64(32)).astype(_U32)


def threefry2x32(k1, k2, x1, x2):
  k1, k2, x0, x1 = _u32(k1, k2, x1, x2)
  ks = [k1, k2, k1 ^ k2 ^ _U32(0x1BD11BDA)]
  rotations = ((13, 15, 26, 6), (17, 29, 16, 24))
  x0 += ks[0]
  x1 += ks[1]
  tmp = np.empty_like(x1)
  for i in range(5):
    for r in rotations[i % 2]:
      # x0 += x1; x1 = rotl(x1, r) ^ x0   (in place: these arrays are big)
      x0 += x1
      np.right_shift(x1, _U32(32 - r), out=tmp)
      np.left_shift(x1, _U32(r), out=x1)
      x1 |= tmp
      x1 ^= x0
    x0 += ks[(i + 1) % 3]
    x1 += ks[(i + 2) % 3]
    x1 += _U32(i + 1)
  return [x0, x1]


@register(_tf2.threefry2x32_p)
def _threefry2x32(k1, k2, x1, x2):
  return threefry2x32(k1, k2, x1, x2)


@register(_tf4.threefry4x32_p)
def _threefry4x32(k0, k1, k2, k3, x0, x1, x2, x3):
  k0, k1, k2, k3, x0, x1, x2, x3 = _u32(k0, k1, k2, k3, x0, x1, x2, x3)
  rots = _tf4._ROTATIONS_32X4
  ks = [k0, k1, k2, k3, k0 ^ k1 ^ k2 ^ k3 ^ _U32(0x1BD11BDA)]
  x0 = x0 + ks[0]; x1 = x1 + ks[1]; x2 = x2 + ks[2]; x3 = x3 + ks[3]
  for rnd in range(20):
    r0, r1 = rots[rnd % 8]
    if rnd % 2 == 0:
      x0 = x0 + x1; x1 = _rotl(x1, r0); x1 = x0 ^ x1
      x2 = x2 + x3; x3 = _rotl(x3, r1); x3 = x2 ^ x3
    else:
      x0 = x0 + x3; x3 = _rotl(x3, r0); x3 = x0 ^ x3
      x2 = x2 + x1; x1 = _rotl(x1, r1); x1 = x2 ^ x1
    if (rnd & 3) == 3:
      j = rnd // 4
      x0 = x0 + ks[(1 + j) % 5]
      x1 = x1 + ks[(2 + j) % 5]
      x2 = x2 + ks[(3 + j) % 5]
      x3 = x3 + ks[(4 + j) % 5] + _U32(1 + j)
  return [x0, x1, x2, x3]


@register(_ph2.philox2x32_p)
def _philox2x32(k0, x0, x1):
  k0, x0, x1 = _u32(k0, x0, x1)
  for rnd in range(10):
    if rnd > 0:
      k0 = k0 + _U32(0x9E3779B9)
    lo, hi = _mulhilo(0xD256D193, x0)
    x0, x1 = hi ^ x1 ^ k0, lo
  return [x0, x1]


def _philox4x32_rounds(k0, k1, x0, x1, x2, x3):
  for rnd in range(10):
    if rnd > 0:
      k0 = k0 + _U32(0x9E3779B9)
      k1 = k1 + _U32(0xBB67AE85)
    lo0, hi0 = _mulhilo(0xD2511F53, x0)
    lo1, hi1 = _mulhilo(0xCD9E8D57, x2)
    x0, x1, x2, x3 = hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0
  return [x0, x1, x2, x3]


@register(_ph4.philox4x32_p)
def _philox4x32(k0, k1, x0, x1, x2, x3):
  return _philox4x32_rounds(*_u32(k0, k1, x0, x1, x2, x3))


@register(prng.iota_2x32_shape_p)
def _iota_2x32_shape(*, shape):
  n = math.prod(shape)
  if n <= 2**32:
    # The usual case: build the halves directly (a uint64 count array is 8
    # bytes/element, too much for 1e8-element draws in wasm32's 4 GiB).
    return [np.zeros(shape, _U32), np.arange(n, dtype=_U32).reshape(shape)]
  counts = np.arange(n, dtype=np.uint64).reshape(shape)
  return [(counts >> np.uint64(32)).astype(_U32), counts.astype(_U32)]


@register(random_core.random_clone_p, raw=True)
def _random_clone(x):
  # A new array (as the jit jax runs eagerly would return), same contents.
  aval = core.typeof(x)
  if _is_key_aval(aval):
    return ext_wrap(_phys(x), aval)
  return from_numpy(_phys(x), aval) if isinstance(aval, core.ShapedArray) else x


# ---------------------------------------------------------------------------
# rng_bit_generator: XLA RngBitGeneratorExpander (lib/prng.cc) in NumPy.

_M64 = (1 << 64) - 1


def _split_u64(v):
  v = np.asarray(v, dtype=np.uint64)
  return v.astype(_U32), (v >> np.uint64(32)).astype(_U32)


def _join_u64(lo, hi):
  return lo.astype(np.uint64) | (hi.astype(np.uint64) << np.uint64(32))


def _philox_bits32(key, counter_lo, counter_hi, n):
  """n uint32 Philox outputs, plus the advanced 128-bit counter."""
  nv4 = -(-n // 4)
  j = np.arange(nv4, dtype=np.uint64)
  lo = (np.uint64(counter_lo) + j)
  hi = np.where(lo < np.uint64(counter_lo), np.uint64((counter_hi + 1) & _M64),
                np.uint64(counter_hi))
  w0, w1 = _split_u64(lo)
  w2, w3 = _split_u64(hi)
  k0, k1 = _split_u64(np.uint64(key))
  out = _philox4x32_rounds(np.broadcast_to(k0, w0.shape), np.broadcast_to(k1, w0.shape),
                           w0, w1, w2, w3)
  new_lo = (counter_lo + nv4) & _M64
  new_hi = (counter_hi + (1 if new_lo < counter_lo else 0)) & _M64
  return out, nv4, new_lo, new_hi


def _interleave(results, n):
  return np.stack(results, axis=1).reshape(-1)[:n]


def _philox_generate(key, counter_lo, counter_hi, shape, bits):
  n = math.prod(shape)
  if bits == 64:
    out, _, new_lo, new_hi = _philox_bits32(key, counter_lo, counter_hi, 2 * n)
    b0 = _join_u64(out[0], out[1])
    b1 = _join_u64(out[2], out[3])
    vals = _interleave([b0, b1], n)
  else:
    out, _, new_lo, new_hi = _philox_bits32(key, counter_lo, counter_hi, n)
    vals = _interleave(out, n)
  return vals.reshape(shape), new_lo, new_hi


def _threefry_generate(key, state, shape, bits):
  k0, k1 = _split_u64(np.uint64(key))
  if bits == 64:
    n = math.prod(shape)
    inp = (np.uint64(state) + np.arange(n, dtype=np.uint64)).reshape(shape)
    x0, x1 = threefry2x32(k0, k1, *_split_u64(inp))
    return _join_u64(x0, x1), (state + n) & _M64
  dims = list(shape)
  if not dims:
    half = [1]
    split = None
  else:
    evens = [i for i, d in enumerate(dims) if d % 2 == 0]
    split = evens[0] if evens else int(np.argmax(dims))
    half = list(dims)
    half[split] = -(-dims[split] // 2)
  nh = math.prod(half)
  inp = (np.uint64(state) + np.arange(nh, dtype=np.uint64)).reshape(half)
  x0, x1 = threefry2x32(k0, k1, *_split_u64(inp))
  new_state = (state + nh) & _M64
  if split is None:
    return x0.reshape(()), new_state
  res = np.stack([x0, x1], axis=split + 1)
  full = list(dims)
  full[split] = half[split] * 2
  res = res.reshape(full)
  res = res[tuple(slice(0, d) for d in dims)]
  return res, new_state


@register(lax.rng_bit_generator_p)
def _rng_bit_generator(key, *, shape, dtype, algorithm, out_sharding=None):
  key = np.asarray(key)
  dtype = np.dtype(dtype)
  if key.dtype == np.uint32:
    st = _join_u64(key.reshape(2, 2)[:, 0], key.reshape(2, 2)[:, 1])
  else:
    st = key.astype(np.uint64)
  st = [int(s) for s in st]
  shape = tuple(shape)
  bits = dtype.itemsize * 8 if dtype.kind == "u" else 32
  gen_bits = 64 if bits == 64 else 32
  if algorithm == lax.RandomAlgorithm.RNG_THREE_FRY:
    vals, new = _threefry_generate(st[0], st[1], shape, gen_bits)
    new_state = [st[0], new]
  else:  # RNG_DEFAULT is Philox on CPU.
    # A length-2 state is reversed into a 128-bit counter (lo=st[1],
    # hi=st[0]) and only the low word of the new counter is kept.
    vals, new_lo, _ = _philox_generate(st[0], st[1], st[0], shape, gen_bits)
    new_state = [st[0], new_lo]
  new_state = np.array(new_state, dtype=np.uint64)
  if key.dtype == np.uint32:
    lo, hi = _split_u64(new_state)
    new_key = np.stack([lo, hi], axis=1).reshape(4)
  else:
    new_key = new_state
  if dtype.kind == "u":
    vals = vals.astype(dtype)  # truncates for 8/16 bits
  else:
    vals = vals.astype(_U32).astype(dtype)
  return [new_key, vals]


_rng_uniform_state = np.random.default_rng(0)


@register(lax.rng_uniform_p)
def _rng_uniform(a, b, *, shape):
  a = np.asarray(a)
  dt = a.dtype
  if dtypes.issubdtype(dt, np.integer):
    lo, hi = int(a), int(b)
    if hi <= lo:
      return np.full(shape, lo, dtype=dt)
    return _rng_uniform_state.integers(lo, hi, size=shape).astype(dt)
  u = _rng_uniform_state.random(size=shape)
  return (np.float64(a) + u * (np.float64(b) - np.float64(a))).astype(
      np.float32 if dt.itemsize < 4 else dt)


# ---------------------------------------------------------------------------
# Extended dtypes

@register(lax.to_edtype_p, raw=True)
def _to_edtype(x, *, edtype):
  aval = core.typeof(x)
  phys = x if type(x).__name__ == "ArrayImpl" else _wrap(_phys(x), aval)
  n = aval.ndim - core.physical_element_aval(edtype).ndim
  return ext_wrap(phys, aval.update(shape=aval.shape[:n], dtype=edtype,
                                    weak_type=False))


@register(lax.from_edtype_p, raw=True)
def _from_edtype(x, *, dtype):
  return ext_physical(x)


# ---------------------------------------------------------------------------
# Callbacks

def _check_callback_outs(outs, result_avals):
  outs = list(outs)
  if len(outs) != len(result_avals):
    raise RuntimeError(
        "Mismatched number of outputs from callback. "
        "Expected: {}, Actual: {}".format(len(result_avals), len(outs)))
  outs = [dtypes.canonicalize_value(np.asarray(a)) for a in outs]
  for i, (val, aval) in enumerate(zip(outs, result_avals)):
    if val.shape != aval.shape:
      raise RuntimeError(
          f"Incorrect output shape for return value #{i}: "
          f"Expected: {aval.shape}, Actual: {val.shape}")
    if val.dtype != aval.dtype:
      raise RuntimeError(
          f"Incorrect output dtype for return value #{i}: "
          f"Expected: {aval.dtype}, Actual: {val.dtype}")
  return [np.asarray(v) for v in outs]


def _run_callback(name, callback, args, result_avals):
  # Upstream's impls `device_put` the operands to the CPU device first; ours
  # already are CPU-resident JAX arrays (raw impl), so call directly under the
  # same mesh contexts (the callback may run inside a mesh context).
  from shax.jax._src import mesh as mesh_lib  # noqa: PLC0415
  from shax.jax._src import sharding_impls  # noqa: PLC0415
  args = [a if isinstance(a, (core.Tracer, prng.PRNGKeyArray, earray.EArray))
          or type(a).__name__ == "ArrayImpl"
          else _wrap(np.asarray(a), core.typeof(a)) for a in args]
  with (sharding_impls._internal_use_concrete_mesh(mesh_lib.empty_concrete_mesh),
        mesh_lib.use_abstract_mesh(mesh_lib.empty_abstract_mesh)):
    try:
      outs = callback(*args)
    except BaseException:
      callback_lib.logger.exception(f"jax.{name} failed")
      raise
  return [from_numpy(v, a) for v, a in
          zip(_check_callback_outs(outs, result_avals), result_avals)]


@register(callback_lib.pure_callback_p, raw=True)
def _pure_callback(*args, callback, result_avals, **_):
  return _run_callback("pure_callback", callback, args, result_avals)


@register(callback_lib.io_callback_p, raw=True)
def _io_callback(*args, callback, result_avals, **_):
  return _run_callback("io_callback", callback, args, result_avals)


# ---------------------------------------------------------------------------
# State (Ref) primitives: evaluated with the upstream discharge rules on the
# Ref's buffer, then the buffer is replaced (as XLA would after donation).

def _ref_impl(ref):
  return ref._refs if isinstance(ref, core.Ref) else ref


@register(state_primitives.get_p, raw=True)
def _get(ref, *idx, tree):
  r = _ref_impl(ref)
  return state_discharge._get_discharge(r._buf, idx, tree)


@register(state_primitives.swap_p, raw=True)
def _swap(ref, val, *idx, tree):
  r = _ref_impl(ref)
  old, new = state_discharge._swap_discharge(r._buf, val, idx, tree)
  r._buf = _wrap(new, core.typeof(r._buf))
  return old


@register(state_primitives.addupdate_p, raw=True)
def _addupdate(ref, val, *idx, tree):
  r = _ref_impl(ref)
  new = state_discharge._addupdate_discharge(r._buf, val, idx, tree)
  r._buf = _wrap(new, core.typeof(r._buf))
  return []


# ---------------------------------------------------------------------------
# Tokens

@register(lax.create_token_p, raw=True)
def _create_token(*_):
  return new_token()


@register(lax.after_all_p, raw=True)
def _after_all(*_):
  return new_token()


# ---------------------------------------------------------------------------
# Misc

@register(ffi.ffi_call_p)
def _ffi_call(*args, **params):
  target = params.get("target_name", "<unknown>")
  raise NotImplementedError(
      f"shax: can't run XLA FFI call '{target}': shax has no XLA runtime")


@register(ann.approx_top_k_p)
def _approx_top_k(operand, *, k, reduction_dimension, recall_target, is_max_k,
                  reduction_input_size_override, aggregate_to_topk):
  # On CPU XLA uses an exact (sort-based) fallback; with aggregate_to_topk=False
  # the output keeps the top `out_size` elements in sorted order.
  x = np.asarray(operand)
  dim = reduction_dimension % x.ndim
  if aggregate_to_topk:
    out_size = k
  else:
    out_size = approx_top_k_reduction_output_size(
        x.shape[dim], x.ndim, k, recall_target, aggregate_to_topk,
        reduction_input_size_override)[0]
  xf = np.moveaxis(x, dim, -1)
  key = xf.astype(np.float32) if xf.dtype.itemsize < 4 else xf
  if is_max_k:
    order = np.argsort(-key, axis=-1, kind="stable")
  else:
    order = np.argsort(key, axis=-1, kind="stable")
  idx = order[..., :out_size]
  vals = np.take_along_axis(xf, idx, axis=-1)
  return [np.moveaxis(vals, -1, dim), np.moveaxis(idx.astype(np.int32), -1, dim)]


def approx_top_k_reduction_output_size(input_size, rank, top_k, recall_target,
                                       aggregate_to_topk,
                                       input_size_override=-1):
  """Port of XLA's ApproxTopKReductionOutputSize (approx_topk_shape.cc)."""
  if aggregate_to_topk:
    return top_k, -1
  tpu_tiling = 1024 if rank == 1 else 128
  if input_size <= tpu_tiling:
    return input_size, 0
  if input_size_override >= 0:
    if input_size > input_size_override:
      raise ValueError(
          f"reduction_input_size_override: {input_size_override} should be "
          f"greater equals to operands[reduction_dim]: {input_size}")
  logical_input_size = input_size_override if input_size_override >= 0 else input_size
  log2_ceil = lambda v: max(int(v) - 1, 0).bit_length()
  cdiv = lambda a, b: -(-a // b)
  if top_k == 1:
    return tpu_tiling, log2_ceil(cdiv(logical_input_size, tpu_tiling))
  if recall_target <= 0.0 or recall_target > 1.0:
    raise ValueError(f"recall_target should be in (0, 1], got {recall_target}")
  if recall_target == 1.0:
    return input_size, 0
  m = int((1.0 - top_k) / math.log(float(recall_target)))
  m = min(max(m, tpu_tiling), logical_input_size)
  log2_reduction = (logical_input_size // m).bit_length() - 1
  if log2_reduction <= 0:
    return input_size, 0
  log2_reduction = min(log2_reduction, log2_ceil(input_size // tpu_tiling))
  return (cdiv(cdiv(input_size, tpu_tiling), 1 << log2_reduction) * tpu_tiling,
          log2_reduction)


@register(lax.polynomial_p)
def _polynomial(x, *coeffs, unroll=None):
  acc = np.asarray(coeffs[-1])
  for c in reversed(coeffs[:-1]):
    acc = acc * x + c
  return np.broadcast_to(acc, np.broadcast_shapes(np.shape(x), *map(np.shape, coeffs)))


@register(parallel.axis_index_p)
def _axis_index(*, axis_name):
  # Only reachable eagerly for a bound axis of a size-1 mesh: index 0.
  return np.int32(0)


# Concrete EArrays only come out of this backend, so registering here is early
# enough. Upstream adds extended-dtype values inside compiled code with the
# dtype's `_rules.add` (mlir.add_jaxvals_lowering); do the same eagerly.
def _add_earrays(x, y):
  dt = core.typeof(x).dtype
  return dt._rules.add(dt, x, y)


from shax.jax._src import ad_util  # noqa: E402
ad_util.raw_jaxval_adders[earray.EArray] = _add_earrays
