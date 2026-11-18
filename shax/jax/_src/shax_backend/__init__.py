"""shax's execution backend: evaluates JAX primitives with NumPy.

This replaces XLA compilation and dispatch. Two entry points are wired into
ported upstream code (both marked `# SHAX:` there):

- `apply_primitive(prim, *args, **params)`: the eager impl of every primitive
  (upstream `dispatch.apply_primitive`, which would jit the single primitive).
- `run_jaxpr(closed_jaxpr, args)`: the impl of `jit` (upstream
  `pjit._pjit_call_impl_python`, which would lower and compile).

Primitive implementations live in sibling modules and register themselves with
`register(prim)`. They receive NumPy arrays (or NumPy scalars) and the
primitive's params, and return NumPy values. `apply_primitive` then casts the
results to the dtypes the primitive's abstract evaluation says JAX produces and
checks the shapes, so impls may rely on NumPy's own promotion and stay short.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib
import threading
from typing import Any

import numpy as np

_impls: dict[Any, Callable[..., Any]] = {}
_loaded = False

# Modules defining impls; imported on first use (they import lax etc.).
_IMPL_MODULES = (
    "shax.jax._src.shax_backend.lax_impls",
    "shax.jax._src.shax_backend.placement",
    "shax.jax._src.shax_backend.elementwise",
    "shax.jax._src.shax_backend.structural",
    "shax.jax._src.shax_backend.numerics",
    "shax.jax._src.shax_backend.control",
    "shax.jax._src.shax_backend.prng_jit",
)


_raw: set = set()


def register(prim, *, raw: bool = False):
  """Decorator: register `fn` as the NumPy implementation of `prim`.

  With `raw=True` the impl receives the original JAX values (ArrayImpls etc.)
  instead of NumPy arrays and must return JAX values (or NumPy values, which
  are wrapped). Use it for higher-order primitives (cond, while, scan, ...)
  that evaluate sub-jaxprs with `core.eval_jaxpr`.
  """
  def decorator(fn):
    _impls[prim] = fn
    if raw:
      _raw.add(prim)
    else:
      _raw.discard(prim)
    return fn
  return decorator


_load_lock = threading.RLock()


def _load() -> None:
  # Thread-safe: another thread must not see a half-populated impl table.
  global _loaded
  if _loaded:
    return
  with _load_lock:
    if not _loaded:
      for name in _IMPL_MODULES:
        importlib.import_module(name)
      _loaded = True


def to_numpy(x) -> Any:
  """Host value of a concrete JAX value (ArrayImpl, TypedNdArray, scalar)."""
  from shax.jax._src import core  # noqa: PLC0415
  if isinstance(x, np.ndarray) and type(x) is np.ndarray:
    return x
  val = getattr(x, "_value", None)
  if val is not None and not isinstance(x, core.Tracer):
    return val
  if hasattr(x, "aval") and not isinstance(x, core.Tracer):
    return np.asarray(x, dtype=x.aval.dtype)
  return np.asarray(x)


def new_token():
  """A fresh concrete `core.Token` (what XLA's token result handler yields)."""
  from shax.jax._src import core  # noqa: PLC0415
  return from_numpy(None, core.abstract_token)


def from_numpy(val, aval) -> Any:
  """Wrap a NumPy result as an ArrayImpl with exactly `aval`'s type."""
  from shax.jax._src import core, dtypes  # noqa: PLC0415
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  from shax.jax._src.sharding_impls import SingleDeviceSharding  # noqa: PLC0415
  from shax.jax._src import xla_bridge  # noqa: PLC0415
  if isinstance(aval, core.AbstractToken):
    # Like jax's token result handler: a Token wrapping a placeholder array.
    if isinstance(val, core.Token):
      return val
    buf = from_numpy(np.zeros((0,), np.bool_), core.get_token_aval())
    buf._s_token_buffer = True
    # No host value: in upstream ArrayImpl, `_npy_value` is a host-copy
    # cache, and a set cache would let `token._buf._value` bypass the
    # "Cannot convert a token-shape buffer" error.
    buf._npy_value = None
    return core.Token(buf)
  if not isinstance(aval, core.ShapedArray):
    raise NotImplementedError(f"shax: can't materialize a value of type {aval}")
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    raise NotImplementedError(f"shax: extended dtype {aval.dtype} not supported yet")
  if aval.dtype == dtypes.float0:
    # Like jax's array result handler: float0 values never live on device.
    from shax.jax._src import literals  # noqa: PLC0415
    return literals.TypedNdArray(np.zeros(aval.shape, dtypes.float0), aval=aval)
  arr = np.asarray(val)
  if arr.dtype != aval.dtype:
    arr = arr.astype(aval.dtype)
  if arr.shape != aval.shape:
    raise AssertionError(
        f"shax backend produced shape {arr.shape}, expected {aval.str_short()}")
  # As XLA places an output by its type: e.g. a device_put to
  # MemorySpace.Host yields an array in "pinned_host" memory.
  space = getattr(aval, "memory_space", core.MemorySpace.Device)
  sharding = SingleDeviceSharding(
      xla_bridge.devices()[0],
      memory_kind=(None if space == core.MemorySpace.Device
                   else core.mem_space_to_kind(space)))
  return ArrayImpl(aval, sharding, [arr], committed=False, _skip_checks=True)


def apply_primitive(prim, *args, **params):
  """Eager application (dispatch.apply_primitive): like jax's per-primitive
  jit, check the outputs for nan/inf under debug_nans/debug_infs."""
  out = _apply_impl(prim, *args, **params)
  from shax.jax._src import dispatch  # noqa: PLC0415
  if dispatch.needs_check_special():
    from shax.jax._src import api_util, core  # noqa: PLC0415
    outs = out if prim.multiple_results else [out]
    try:
      dispatch.check_special(prim.name, [
          o for o in outs if isinstance(core.typeof(o), core.ShapedArray)])
    except api_util.InternalFloatingPointError as e:
      raise FloatingPointError(
          f"invalid value ({e.ty}) encountered in {prim.name}") from None
  return out


def _apply_impl(prim, *args, **params):
  _load()
  impl = _impls.get(prim)
  if impl is None and (lf := _lowered_fun(prim)) is None:
    raise NotImplementedError(
        f"shax: primitive '{prim}' has no NumPy implementation yet")
  from shax.jax._src import core  # noqa: PLC0415
  in_avals = [core.typeof(a) for a in args]
  out_avals, _ = prim.abstract_eval(*in_avals, **params)
  if impl is None:
    # SHAX: like jax's per-primitive jit, run the primitive's `lower_fun`
    # lowering (a traceable function on physical values).
    return _eval_via_lowering(prim, lf, args, params, out_avals)
  # SHAX: PRNG-key (extended dtype) values: run non-raw impls on the physical
  # base arrays, vmapped over the key element dims (see `_apply_physical`).
  if prim not in _raw and any(map(_is_ext, (
      *in_avals, *(out_avals if prim.multiple_results else [out_avals])))):
    return _apply_physical(prim, args, params, in_avals, out_avals)
  if prim in _raw:
    out = impl(*args, **params)
    wrap = _wrap_raw
  else:
    np_args = [to_numpy(a) for a in args]
    with np.errstate(all="ignore"):
      out = impl(*np_args, **params)
    wrap = from_numpy
  if prim.multiple_results:
    return [wrap(o, a) for o, a in zip(out, out_avals, strict=True)]
  return wrap(out, out_avals)


def _is_ext(aval) -> bool:
  from shax.jax._src import core, dtypes  # noqa: PLC0415
  return (isinstance(aval, core.ShapedArray)
          and dtypes.issubdtype(aval.dtype, dtypes.extended))


def _lowered_fun(prim):
  """`(fun, multiple_results)` of prim's CPU `mlir.lower_fun` lowering, if any."""
  from shax.jax._src.interpreters import mlir  # noqa: PLC0415
  for table in (mlir._platform_specific_lowerings.get("cpu", {}), mlir._lowerings):
    entry = table.get(prim)
    if entry is not None:
      return getattr(entry.rule, "_shax_lower_fun", None)
  return None


_lowering_cache: dict = {}

# Primitives with a Python impl that compiled evaluation runs differently
# (as their lowering would), registered with `register_jit_impl`.
_jit_impls: dict = {}


def register_jit_impl(prim):
  def deco(fn):
    _jit_impls[prim] = fn
    return fn
  return deco


def _compiled_call(key, fn, *args):
  """`fn(*args)` as compiled code: trace once per (key, arg types), then
  evaluate the jaxpr with `eval_jaxpr_compiled` (no eager op dispatch)."""
  from shax.jax._src import api, core  # noqa: PLC0415
  k = (key, tuple(core.typeof(a) for a in args))
  closed = _lowering_cache.get(k)
  if closed is None:
    closed = api.make_jaxpr(fn)(*args)
    if len(_lowering_cache) > 4096:
      _lowering_cache.clear()
    _lowering_cache[k] = closed
  (out,) = eval_jaxpr_compiled(closed.jaxpr, closed.consts, *args)
  return out


def _eval_via_lowering(prim, lf, args, params, out_avals):
  """Evaluate `prim` with its lowering function on physical values."""
  # Like `mlir.lower_fun`, `fun` sees the logical values; in XLA everything is
  # physical, so it may return physical arrays for extended-dtype outputs.
  from shax.jax._src import api, core  # noqa: PLC0415
  fun, multiple_results = lf
  # Trace like `lower_fun` does (so `fun` can use array methods on abstract
  # values of any dtype), then evaluate the jaxpr as compiled code would.
  # Cached per (prim, params, input types), as XLA compiles a lowering once.
  try:
    key = (prim, tuple(sorted(params.items())),
           tuple(core.typeof(a) for a in args))
    hash(key)
  except TypeError:
    key = None
  closed = _lowering_cache.get(key) if key is not None else None
  if closed is None:
    closed = api.make_jaxpr(lambda *xs: fun(*xs, **params))(*args)
    if key is not None:
      if len(_lowering_cache) > 4096:
        _lowering_cache.clear()
      _lowering_cache[key] = closed
  out = eval_jaxpr_compiled(closed.jaxpr, closed.consts, *args)
  out = out if multiple_results else out[0]
  outs = list(out) if multiple_results else [out]
  avals = list(out_avals) if prim.multiple_results else [out_avals]
  res = [(o if _is_ext(core.typeof(o)) else ext_wrap(o, a)) if _is_ext(a)
         else _wrap_raw(o, a) for o, a in zip(outs, avals, strict=True)]
  return res if prim.multiple_results else res[0]


def _default_sharding():
  from shax.jax._src import xla_bridge  # noqa: PLC0415
  from shax.jax._src.sharding_impls import SingleDeviceSharding  # noqa: PLC0415
  return SingleDeviceSharding(xla_bridge.devices()[0])


def ext_physical(x):
  """The physical (base) array of an extended-dtype value."""
  from shax.jax._src import core, earray  # noqa: PLC0415
  from shax.jax._src.interpreters import pxla  # noqa: PLC0415
  from shax.jax._src.random import prng  # noqa: PLC0415
  if isinstance(x, prng.PRNGKeyArray):
    return x._base_array
  if isinstance(x, earray.EArray):
    return x._data
  if (not isinstance(x, core.Tracer) and _is_ext(core.typeof(x))
      and (handler := pxla.shard_arg_handlers.get(type(x))) is not None):
    # A user-defined at-rest type (e.g. lax_test's FooArray): its shard_arg
    # handler is how jax gets the physical buffer.
    return handler([x], [_default_sharding()], [None], [None])[0]
  return x


def ext_wrap(phys, aval):
  """Wrap a physical array (ArrayImpl or NumPy) as a value of extended `aval`.

  PRNG keys become `PRNGKeyArray`s; other extended dtypes become `EArray`s
  (what upstream result handlers produce for at-rest extended arrays).
  """
  from shax.jax._src import core, dtypes, earray  # noqa: PLC0415
  from shax.jax._src.random import prng  # noqa: PLC0415
  if not (type(phys).__name__ == "ArrayImpl"):
    phys = from_numpy(np.asarray(phys), core.physical_aval(aval))
  if dtypes.issubdtype(aval.dtype, dtypes.prng_key):
    return prng.PRNGKeyArray(aval.dtype._impl, phys)
  return earray.EArray(aval, phys)


def _ext_jit_output(x, aval):
  """A jit's extended-dtype output, as jax's result handler would make it."""
  from shax.jax._src import earray  # noqa: PLC0415
  rules = aval.dtype._rules
  if (isinstance(x, earray.EArray)
      and hasattr(rules, "global_sharded_result_handler")):
    return rules.global_sharded_result_handler(
        aval, _default_sharding(), False)(x._data)
  return x


def _apply_physical(prim, args, params, in_avals, out_avals):
  """Apply `prim` to extended-dtype values via their physical arrays.

  A structural op on e.g. PRNG keys equals the same op on the base arrays
  vmapped over the trailing element dims (what XLA lowering does via
  physical types).
  """
  from shax.jax._src import api, core  # noqa: PLC0415
  outs = list(out_avals) if prim.multiple_results else [out_avals]
  nd = next(core.physical_element_aval(a.dtype).ndim
            for a in (*in_avals, *outs) if _is_ext(a))
  if prim.name in ("eq", "ne"):
    # Keys compare equal iff all their key words do (XLA lowering: compare
    # the physical arrays, then reduce_and / reduce_or the element dims).
    x, y = (to_numpy(ext_physical(a)) for a in args)
    x, y = np.broadcast_arrays(x, y)
    res = x == y if prim.name == "eq" else x != y
    axes = tuple(range(res.ndim - nd, res.ndim))
    res = res.all(axes) if prim.name == "eq" else res.any(axes)
    return from_numpy(res, out_avals)
  if prim.name == "empty":
    pa = core.physical_aval(outs[0])
    return ext_wrap(np.zeros(pa.shape, pa.dtype), outs[0])
  if not any(map(_is_ext, in_avals)):
    raise NotImplementedError(
        f"shax: primitive '{prim}' producing extended dtype from non-extended inputs")
  phys_args = [ext_physical(a) for a in args]
  in_axes = tuple(-1 if _is_ext(a) else None for a in in_avals)
  out_axes = [-1 if _is_ext(a) else None for a in outs]
  f = lambda *xs: prim.bind(*xs, **params)
  if not prim.multiple_results:
    out_axes = out_axes[0]
  for _ in range(nd):
    f = api.vmap(f, in_axes=in_axes, out_axes=out_axes)
  res = f(*phys_args)
  res_list = list(res) if prim.multiple_results else [res]
  res_list = [ext_wrap(r, a) if _is_ext(a) else r
              for r, a in zip(res_list, outs, strict=True)]
  return res_list if prim.multiple_results else res_list[0]


def _wrap_raw(x, aval):
  if _is_ext(aval):
    return x  # SHAX: extended-dtype values (keys, EArrays) pass through
  if type(x).__name__ == "ArrayImpl" and x.aval == aval:
    return x
  if not isinstance(aval, __import__("shax.jax._src.core", fromlist=["core"]).ShapedArray):
    return x  # tokens, refs, etc. pass through
  return from_numpy(to_numpy(x), aval)


_jit_state = threading.local()


def eval_jaxpr_compiled(jaxpr, consts, *args) -> list[Any]:
  """`core.eval_jaxpr`, but the way compiled code runs a jaxpr.

  Inside a jit, real JAX executes XLA code, never the primitives' Python
  `impl` rules (which users and tests may swap out, e.g. to count calls). So
  first-order primitives with a NumPy implementation run it directly;
  everything else (higher-order primitives, custom rules, effects) binds as
  in `core.eval_jaxpr`.
  """
  from shax.jax._src import config, core, source_info_util  # noqa: PLC0415
  from shax.jax._src.lib.utils import foreach  # noqa: PLC0415
  _load()

  # SHAX: values produced by first-order NumPy impls stay plain ndarrays in
  # `env`, typed by their jaxpr variable's aval (what XLA's buffers are);
  # they're wrapped as ArrayImpls only when something else reads them. That
  # skips a per-op abstract_eval and ArrayImpl, most of a step's cost.
  raw_vals: set = set()  # vars whose env value is a raw ndarray

  def read(v):
    if isinstance(v, core.Literal):
      return v.val
    val = env[v]
    if v in raw_vals:
      val = env[v] = from_numpy(val, v.aval)
      raw_vals.discard(v)
    return val

  def read_np(v):
    return to_numpy(v.val if isinstance(v, core.Literal) else env[v])

  checks = config.enable_checks.value

  def write(v, val):
    if checks:
      assert core.typecheck(v.aval, val), (v.aval, core.typeof(val), val)
    env[v] = val

  env: dict = {}
  foreach(write, jaxpr.all_invars, [*consts, *args])
  lu = core.last_used(jaxpr)
  eval_trace = isinstance(core.trace_ctx.trace, core.EvalTrace)
  from shax.jax._src import pjit as pjit_lib  # noqa: PLC0415
  for eqn in jaxpr.eqns:
    prim = eqn.primitive
    fast = _fast_impl(prim)
    direct = fast is not None and eval_trace
    name_stack = (source_info_util.current_name_stack()
                  + eqn.source_info.name_stack)
    with (source_info_util.user_context(eqn.source_info.traceback,
                                        name_stack=name_stack),
          eqn.ctx.manager):
      if (prim is pjit_lib.jit_p and eval_trace
          and not eqn.params["jaxpr"].effects):
        # SHAX: a nested jit inside compiled code is inlined, as XLA does;
        # dispatching it again (caches, fast path) would only add overhead.
        inner = eqn.params["jaxpr"]
        ans = eval_jaxpr_compiled(inner.jaxpr, inner.consts,
                                  *map(read, eqn.invars))
        foreach(write, eqn.outvars, ans)
        core.clean_up_dead_vars(eqn, env, lu)
        continue
      if direct and _raw_eligible(eqn):
        outs = _run_raw(fast, prim, list(map(read_np, eqn.invars)), eqn)
        for v, o in zip(eqn.outvars, outs, strict=True):
          env[v] = o
          raw_vals.add(v)
        core.clean_up_dead_vars(eqn, env, lu)
        continue
      if eval_trace and prim in _jit_impls:
        ans = _jit_impls[prim](*map(read, eqn.invars), **eqn.params)
      elif direct and prim in _impls:  # no per-op nan/inf checks inside a jit
        ans = _apply_impl(prim, *map(read, eqn.invars), **eqn.params)
      elif (prim not in _impls and (eval_trace or "impl" not in vars(prim))
            and (lf := _lowered_fun(prim)) is not None):
        # SHAX: no NumPy impl, but a `lower_fun` lowering: that's what XLA
        # runs, so compiled code uses it even if the primitive has a Python
        # impl rule (which would dispatch eager ops, e.g. random_bits).
        in_vals = list(map(read, eqn.invars))
        out_avals = (eqn.out_avals if hasattr(eqn, "out_avals") else
                     [v.aval for v in eqn.outvars])
        ans = _eval_via_lowering(
            prim, lf, in_vals, eqn.params,
            out_avals if prim.multiple_results else out_avals[0])
      else:
        ans = prim.bind(*map(read, eqn.invars),
                        **prim.get_bind_params(eqn.params))
    if prim.multiple_results:
      foreach(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
    core.clean_up_dead_vars(eqn, env, lu)
  return list(map(read, jaxpr.outvars))


# Impls used only on the raw-ndarray path inside a jit (all operands plain
# arrays): primitives whose eager dispatch must keep upstream's impl because
# they also take other value types (e.g. add_any on float0 or keys).
_raw_only_impls: dict = {}


def _fast_impl(prim):
  """The NumPy impl that compiled evaluation may call directly, if any."""
  if prim in _raw:
    return None
  return _impls.get(prim) or _raw_only_impls.get(prim)


_plain_cache: dict = {}


def _plain_aval(aval) -> bool:
  try:
    return _plain_cache[aval]
  except KeyError:
    pass
  except TypeError:  # unhashable aval
    return _plain_aval_uncached(aval)
  r = _plain_cache[aval] = _plain_aval_uncached(aval)
  return r


def _plain_aval_uncached(aval) -> bool:
  from shax.jax._src import core, dtypes  # noqa: PLC0415
  return (type(aval) is core.ShapedArray
          and not dtypes.issubdtype(aval.dtype, dtypes.extended)
          and aval.dtype != dtypes.float0
          and getattr(aval, "memory_space", core.MemorySpace.Device)
              == core.MemorySpace.Device)


def _raw_eligible(eqn) -> bool:
  """Can this eqn's NumPy impl run on/produce plain ndarrays?"""
  return (all(_plain_aval(v.aval) for v in eqn.invars)
          and all(_plain_aval(v.aval) for v in eqn.outvars))


def _run_raw(impl, prim, np_args, eqn) -> list:
  with np.errstate(all="ignore"):
    out = impl(*np_args, **eqn.params)
  outs = out if prim.multiple_results else [out]
  res = []
  for o, v in zip(outs, eqn.outvars, strict=True):
    aval = v.aval
    o = np.asarray(o)
    if o.dtype != aval.dtype:
      o = o.astype(aval.dtype)
    if o.shape != aval.shape:
      raise AssertionError(
          f"shax backend produced shape {o.shape}, expected {aval.str_short()}")
    res.append(o)
  return res


def evaluating_jit() -> bool:
  """True while `run_jaxpr` evaluates a jit's jaxpr (where real JAX would run
  compiled code, so eager-only restrictions don't apply)."""
  return getattr(_jit_state, "depth", 0) > 0


def run_jaxpr(closed_jaxpr, args: Sequence[Any]) -> list[Any]:
  """Evaluate a jit's closed jaxpr eagerly; outputs become ArrayImpls."""
  from shax.jax._src import core  # noqa: PLC0415
  _jit_state.depth = getattr(_jit_state, "depth", 0) + 1
  try:
    outs = eval_jaxpr_compiled(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)
  finally:
    _jit_state.depth -= 1
  # Like XLA, a jit never hands back one of its inputs or constants as an
  # output buffer (absent donation): those get copied.
  external = {id(x) for x in (*args, *closed_jaxpr.consts)}
  result = []
  for o, aval in zip(outs, closed_jaxpr.out_avals, strict=True):
    if isinstance(o, core.Tracer):
      result.append(o)
    elif _is_ext(aval):  # SHAX: extended dtypes (keys, EArrays, ...)
      result.append(_ext_jit_output(o, aval))
    elif (type(o).__name__ == "ArrayImpl" and o.aval == aval
          and id(o) not in external):
      result.append(o)
    elif isinstance(aval, core.AbstractToken):
      result.append(o if isinstance(o, core.Token) and id(o) not in external
                    else from_numpy(None, aval))
    else:
      result.append(from_numpy(np.array(to_numpy(o)), aval))
  return result
