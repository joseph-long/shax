"""Pure-Python stand-ins for jaxlib's jit dispatch (pjit.cc, py_values.cc).

jaxlib's `PjitFunction` keeps a C++ fast path keyed on argument signatures and
falls back to the Python `cache_miss` on a miss. shax always calls
`cache_miss`: upstream's Python-level caches (`_infer_params`, jaxpr and
lowering caches) still prevent re-tracing, which is the behaviour JAX's tests
observe. Values move to the "device" by wrapping NumPy arrays in ArrayImpl.
"""
from __future__ import annotations

from collections import OrderedDict
import functools
import inspect
import weakref
from typing import Any

import numpy as np

kPjitFunctionPickleVersion = 1


class PjitFunctionCache:
  """Per-(function, global cache key) signature tables, as in pjit.cc.

  jaxlib keys on a weak reference to the function and drops the entries when
  it dies; shax does the same where the function is weak-referenceable.
  """
  _all: "weakref.WeakSet[PjitFunctionCache]" = weakref.WeakSet()

  def __init__(self, capacity: int = 4096):
    self._capacity = capacity
    self._weak: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    self._strong: dict = {}
    PjitFunctionCache._all.add(self)

  def _executables(self, fun, global_cache_key) -> OrderedDict | None:
    """pjit.cc Lookup: the table shared by jits of `fun` with this key.
    None for AOT functions (`fun is None`): pjit.cc's DefaultCache() gives
    each of those a fresh table of its own."""
    if fun is None:
      return None
    try:
      per_fun = self._weak.get(fun)
      if per_fun is None:
        per_fun = self._weak[fun] = {}
    except TypeError:  # not weak-referenceable: hold it strongly (never
      try:             # by id(), which a later object can reuse)
        per_fun = self._strong.setdefault(fun, {})
      except TypeError:  # nor hashable
        return None
    try:
      table = per_fun.get(global_cache_key)
      if table is None:
        table = per_fun[global_cache_key] = OrderedDict()
    except TypeError:
      table = OrderedDict()
    return table

  def _evict(self, table) -> None:
    while len(table) > self._capacity:
      table.popitem(last=False)

  def size(self) -> int:
    return len(self._weak) + len(self._strong)

  def capacity(self) -> int:
    return self._capacity

  def clear(self) -> None:
    self._weak.clear()
    self._strong.clear()

  @staticmethod
  def clear_all() -> None:
    for c in list(PjitFunctionCache._all):
      c.clear()

  def __getstate__(self):
    return {"version": kPjitFunctionPickleVersion, "capacity": self._capacity}

  def __setstate__(self, state):
    if state["version"] != kPjitFunctionPickleVersion:
      raise ValueError(
          f"Invalid PjitFunction pickle version, got {state['version']}, "
          f"expected {kPjitFunctionPickleVersion}")
    self._capacity = state["capacity"]
    self._weak = weakref.WeakKeyDictionary()
    self._strong = {}
    PjitFunctionCache._all.add(self)


_MISSING = object()


def _usable_fastpath(fastpath_data):
  """The fastpath data shax can run directly, else None (fall back)."""
  if fastpath_data is None:
    return None
  ex = fastpath_data.xla_executable
  module = getattr(ex, "module", None)
  if module is None or module.input_output_aliases and any(
      a is not None for a in module.input_output_aliases):
    return None  # donation: let the Python path consume the inputs
  return fastpath_data


def _run_fastpath(fp, flat):
  """jaxlib's PjitFunction fast path: run the cached executable on the
  kept arguments and build outputs from the stored avals/shardings.
  Returns _MISSING to fall back to cache_miss."""
  from shax.jax._src import config, dispatch  # noqa: PLC0415  (cycle)
  from shax.jax._src import shax_backend  # noqa: PLC0415
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  if config.no_execution.value or dispatch.needs_check_special():
    return _MISSING
  args = [*fp.const_args, *flat]
  kept = [a for a, k in zip(args, fp.kept_var_bitvec, strict=True) if k]
  shardings = fp.in_shardings  # pjit.cc: indexed by kept (post-DCE) position
  bufs = []
  for x, s in zip(kept, shardings, strict=True):
    if type(x) is ArrayImpl:
      if x.is_deleted():
        return _MISSING  # let the Python path raise the right error
      bufs.append(x)
    else:  # NumPy arrays and scalars: put them on the device
      bufs.append(batched_device_put(
          _aval_of(x), s, [np.asarray(x)], list(s._addressable_device_assignment),
          False))
  module = fp.xla_executable.module
  jaxpr = module.jaxpr
  # Arguments and outputs here are flattened with the dispatch registry, so
  # extended-dtype values (PRNG keys) appear as their physical arrays; the
  # jaxpr works on the logical values.
  ins = [shax_backend.ext_wrap(b, a) if shax_backend._is_ext(a) else b
         for b, a in zip(bufs[module.num_const_args:], jaxpr.in_avals,
                         strict=True)]
  outs = [shax_backend.ext_physical(o) if shax_backend._is_ext(a) else o
          for o, a in zip(shax_backend.run_jaxpr(jaxpr, ins), jaxpr.out_avals,
                          strict=True)]
  res = []
  for o, aval, s, committed in zip(outs, fp.out_avals, fp.out_shardings,
                                   fp.out_committed, strict=True):
    if type(o) is not ArrayImpl or o.aval != aval:
      return _MISSING
    if o.sharding != s or o._committed != committed:
      o = ArrayImpl(o.aval, s, [o._value], committed, _skip_checks=True)
    res.append(o)
  return fp.out_pytree_def.unflatten(res)


def _aval_of(x):
  from shax.jax._src import core  # noqa: PLC0415
  return core.shaped_abstractify(x)


_DEFAULT_CACHE = None


def _default_cache() -> PjitFunctionCache:
  global _DEFAULT_CACHE
  if _DEFAULT_CACHE is None:
    _DEFAULT_CACHE = PjitFunctionCache()
  return _DEFAULT_CACHE


_PY_SCALARS = (bool, int, float, complex)


def _arg_signature(x, x64):
  """PyArgSignatureOfValue (py_values.cc): (dtype, shape, weak_type, sharding,
  committed), or None for values jaxlib's fast path can't handle."""
  from shax.jax._src import dtypes  # noqa: PLC0415  (import cycle)
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  from shax.jax._src.lib import _jax  # noqa: PLC0415
  t = type(x)
  if t is ArrayImpl:
    aval = x.aval
    return (aval.dtype, aval.shape, aval.weak_type, x._sharding, x._committed)
  if t is _jax._typed.get("float") or t is _jax._typed.get("int") or (
      t is _jax._typed.get("complex")):
    return (np.dtype(x.dtype), (), True)  # typed_scalar_handler: no squash
  if t is _jax._typed.get("ndarray"):  # typed_ndarray_handler
    return (np.dtype(x.dtype), x.shape, bool(x.weak_type))
  if t in _PY_SCALARS:
    if t is bool:
      return (np.dtype(bool), (), True)
    big = {int: np.int64, float: np.float64, complex: np.complex128}[t]
    small = {int: np.int32, float: np.float32, complex: np.complex64}[t]
    return (np.dtype(big if x64 else small), (), True)
  if isinstance(x, (np.ndarray, np.generic)) and not isinstance(x, np.ma.MaskedArray):
    try:
      dtype = dtypes.canonicalize_dtype(x.dtype)
    except Exception:  # noqa: BLE001
      return None
    weak = bool(getattr(getattr(x, "aval", None), "weak_type", False))
    return (dtype, np.shape(x), weak)
  return None


class PjitFunction:
  """Callable returned by `jax.jit`. Instances get `functools.wraps`'d, and
  upstream attaches `lower`/`trace`/`eval_shape`/`clear_cache` to the type.

  Internal state lives in slots, not `__dict__`: `functools.wraps(fun)` copies
  `fun.__dict__` onto the new PjitFunction, and when `fun` is itself jitted that
  would overwrite our cache_miss with the inner function's (jaxlib keeps this
  state in the C++ object for the same reason).
  """
  __slots__ = ("_s_function_name", "_s_fun", "_s_cache_miss", "_s_static_argnums",
               "_s_static_argnames", "_s_global_cache_key", "_s_cache",
               "_s_registry", "_s_shard_arg_fallback", "_s_own_table",
               "__dict__", "__weakref__")

  def __init__(self, function_name, fun, cache_miss, static_argnums,
               static_argnames, global_cache_key, pytree_registry,
               shard_arg_fallback, cache=None):
    self._s_function_name = function_name
    self._s_fun = fun
    self._s_cache_miss = cache_miss
    self._s_static_argnums = list(static_argnums)
    self._s_static_argnames = list(static_argnames)
    self._s_global_cache_key = global_cache_key
    self._s_cache = cache if cache is not None else _default_cache()
    self._s_registry = pytree_registry
    self._s_shard_arg_fallback = shard_arg_fallback
    self._s_own_table = None

  def _executables(self):
    if self._s_own_table is not None:
      return self._s_own_table
    table = self._s_cache._executables(self._s_fun, self._s_global_cache_key)
    if table is None:  # pjit.cc DefaultCache(): private to this function
      table = self._s_own_table = OrderedDict()
    return table

  def __call__(self, *args, **kwargs):
    # pjit.cc: compute the call signature; unsupported arguments (tracers,
    # unknown types) fall back to cache_miss without creating an entry. On a
    # miss, cache_miss runs and its fastpath data (if any) is stored; on a
    # hit with fastpath data, the executable runs directly (no Python
    # cache_miss, so no "jit_cpp_cache_miss" event), as jaxlib does.
    from shax.jax._src import config  # noqa: PLC0415  (import cycle)
    if config.disable_jit.value:  # pjit.cc: GetDisableJit()
      if self._s_fun is None:
        raise ValueError(
            "Disable jit is not supported in the AOT path since the function "
            f"is not available for ({self._s_function_name})")
      return self._s_fun(*args, **kwargs)
    signature, flat = self._call_signature(args, kwargs)
    if signature is None:
      return self._s_cache_miss(*args, **kwargs)[0]
    executables = self._executables()
    entry = executables.get(signature, _MISSING)
    if entry is _MISSING:
      executables[signature] = None  # falls back to Python until populated
      self._s_cache._evict(executables)
      try:
        out = self._s_cache_miss(*args, **kwargs)
      except BaseException:
        raise  # entry stays, falling back to Python (pjit.cc)
      if len(out) > 2 and out[2]:
        executables.pop(signature, None)  # remove_cache
      elif signature in executables:
        executables[signature] = _usable_fastpath(out[1])
      return out[0]
    executables.move_to_end(signature)
    if entry is not None:
      result = _run_fastpath(entry, flat)
      if result is not _MISSING:
        return result
    return self._s_cache_miss(*args, **kwargs)[0]

  def _call_signature(self, args, kwargs):
    try:
      kwnames = tuple(kwargs)
      arg_sig, flat = parse_arguments(
          args, tuple(kwargs.values()), kwnames, self._s_static_argnums,
          self._s_static_argnames, self._s_registry)
      hash(arg_sig)
    except Exception:  # noqa: BLE001  (pjit.cc: ParseArguments failed)
      return None, None
    from shax.jax._src import config  # noqa: PLC0415  (import cycle)
    x64 = config.enable_x64.value
    dyn = []
    for x in flat:
      s = _arg_signature(x, x64)
      if s is None:
        return None, None
      dyn.append(s)
    try:
      signature = (arg_sig, tuple(dyn), config.trace_context())
      hash(signature)
      return signature, flat
    except TypeError:
      return None, None

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    return functools.partial(self, obj)

  @property
  def __signature__(self):
    return inspect.signature(self._s_fun)

  @property
  def _cache_miss(self):
    return self._s_cache_miss

  def _cache_size(self) -> int:
    return len(self._executables())

  def _clear_cache(self) -> None:
    self._executables().clear()

  def __repr__(self):
    try:  # pjit.cc PjitFunction_tp_repr: never fails
      return f"<PjitFunction of {self.__wrapped__!r}>"
    except Exception:  # noqa: BLE001
      return "<PjitFunction>"

  def __getstate__(self):
    state = {"version": kPjitFunctionPickleVersion,
             "function_name": self._s_function_name}
    if self._s_fun is not None:
      state["fun"] = self._s_fun
    state.update(cache_miss=self._s_cache_miss,
                 static_argnums=list(self._s_static_argnums),
                 static_argnames=list(self._s_static_argnames),
                 global_cache_key=self._s_global_cache_key,
                 pytree_registry=self._s_registry,
                 shard_arg_fallback=self._s_shard_arg_fallback,
                 cache=self._s_cache)
    return state

  def __setstate__(self, state):
    if state["version"] != kPjitFunctionPickleVersion:
      raise ValueError(
          f"Invalid PjitFunction pickle version, got {state['version']}, "
          f"expected {kPjitFunctionPickleVersion}. Pickling/Unpickling jitted "
          "functions using different JAX versions is not supported.")
    PjitFunction.__init__(
        self, state["function_name"], state.get("fun"), state["cache_miss"],
        state["static_argnums"], state["static_argnames"],
        state["global_cache_key"], state["pytree_registry"],
        state["shard_arg_fallback"], state["cache"])


def pjit(function_name, fun, cache_miss, static_argnums, static_argnames,
         global_cache_key, pytree_registry, shard_arg_fallback, cache=None):
  return PjitFunction(function_name, fun, cache_miss, static_argnums,
                      static_argnames, global_cache_key, pytree_registry,
                      shard_arg_fallback, cache)


def shard_args(shardings, layouts, copy_semantics, args, fallback=None):
  """Put each arg on the device for its sharding (pjit.cc `shard_args`).

  NumPy arrays are handled here; everything else goes to `fallback` in one
  batch, as jaxlib does.
  """
  n = len(args)
  if not (len(shardings) == len(layouts) == len(copy_semantics) == n):
    raise ValueError("shard_args arguments must have the same length.")
  from shax.jax._src import core  # noqa: PLC0415  (import cycle)
  from shax.jax._src.lib import xla_client  # noqa: PLC0415
  results: list[Any] = [None] * n
  rest = []
  for i, (x, s, l, cs) in enumerate(zip(args, shardings, layouts, copy_semantics)):
    if isinstance(x, np.ndarray) and l is None:
      aval = core.shaped_abstractify(x)
      results[i] = xla_client.batched_device_put(
          aval, s, [x], list(s._addressable_device_assignment), True)
    else:
      rest.append((i, x, s, l, cs))
  if rest:
    if fallback is None:
      raise TypeError(f"shard_args: no handler for {type(rest[0][1])}")
    idx, xs, ss, ls, css = zip(*rest)
    outs = fallback(list(ss), list(ls), list(css), list(xs))
    for i, o in zip(idx, outs):
      results[i] = o
  return results


# --- xla_client functions -------------------------------------------------------

def batched_device_put(aval, sharding, xs, devices, committed=True,
                       force_copy=False, host_buffer_semantics=None,
                       enable_x64=None):
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  return ArrayImpl(aval, sharding, [np.asarray(xs[0])], committed,
                   _skip_checks=True)


def array_result_handler(aval, sharding, committed, *args, **kwargs):
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415

  def handler(bufs):
    if isinstance(bufs, (list, tuple)):
      (buf,) = bufs
    else:
      buf = bufs
    return ArrayImpl(aval, sharding, [buf], committed, _skip_checks=True)

  handler.wrap = lambda f: f  # jaxlib ResultHandler API; unused by shax
  return handler


def batched_copy_array_to_devices_with_sharding(arrays, devices, shardings,
                                                copy_semantics):
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  return [ArrayImpl(a.aval, s, [a._value], a._committed, _skip_checks=True)
          for a, s in zip(arrays, shardings)]


def reorder_shards(x, dst_sharding, copy_semantics):
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  return ArrayImpl(x.aval, dst_sharding, [x._value], x._committed,
                   _skip_checks=True)


# --- jax_jit.parse_arguments / ArgumentSignature (port of jaxlib/jax_jit.cc) ----

class ArgumentSignature:
  __slots__ = ("static_args", "static_arg_names", "dynamic_arg_names",
               "dynamic_arg_treedefs", "_hash")

  def __init__(self, static_args, static_arg_names, dynamic_arg_names,
               dynamic_arg_treedefs):
    self.static_args = tuple(static_args)
    self.static_arg_names = tuple(static_arg_names)
    self.dynamic_arg_names = tuple(dynamic_arg_names)
    self.dynamic_arg_treedefs = tuple(dynamic_arg_treedefs)
    self._hash = None

  def __hash__(self):
    if self._hash is None:
      hashes = []
      for a in self.static_args:
        try:
          hashes.append(hash(a))
        except TypeError as e:
          raise ValueError(
              "Non-hashable static arguments are not supported. An error "
              "occurred while trying to hash an object of type "
              f"{type(a)}, {a}. The error was:\n{e}\n") from e
      self._hash = hash((self.dynamic_arg_treedefs, len(self.dynamic_arg_names),
                         len(self.static_args), len(self.static_arg_names),
                         self.dynamic_arg_names, tuple(hashes),
                         self.static_arg_names))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, ArgumentSignature):
      return False
    if self.dynamic_arg_treedefs != other.dynamic_arg_treedefs:
      return False
    if self.dynamic_arg_names != other.dynamic_arg_names:
      return False
    if self.static_arg_names != other.static_arg_names:
      return False
    if len(self.static_args) != len(other.static_args):
      return False
    for a, b in zip(self.static_args, other.static_args):
      try:
        if type(a) is not type(b) or not bool(a == b):
          return False
      except Exception as e:
        raise ValueError(
            "static arguments should be comparable using __eq__.The following "
            "error was raised when comparing two objects of types "
            f"{type(a)} and {type(b)}. The error was:\n{e}") from e
    return True

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return ("static args (positional + keyword): "
            f"[{','.join(str(a) for a in self.static_args)}], "
            "static arg keyword names: "
            f"[{','.join(str(a) for a in self.static_arg_names)}], "
            "dynamic arg signatures (positional + keyword): "
            f"[{','.join(str(a) for a in self.dynamic_arg_names)}], "
            "dynamic arg shardings: "
            f"[{'| '.join(str(t) for t in self.dynamic_arg_treedefs)}]")

  __str__ = __repr__


def parse_arguments(positional_args, keyword_args, kwnames, static_argnums,
                    static_argnames, pytree_registry):
  """Parses the arguments to a function as jax.jit would.

  Returns an ArgumentSignature and the flattened dynamic arguments.
  """
  static_args, static_arg_names = [], []
  dynamic_arg_names, treedefs = [], []
  flat: list = []
  n = len(positional_args)
  static_set = {(t if t >= 0 else t + n) for t in static_argnums}
  for i, arg in enumerate(positional_args):
    if i in static_set:
      static_args.append(arg)
    else:
      leaves, treedef = pytree_registry.flatten(arg)
      flat.extend(leaves)
      treedefs.append(treedef)
  if keyword_args:
    static_names = set(static_argnames)
    for name, arg in sorted(zip(kwnames, keyword_args), key=lambda kv: kv[0]):
      if name in static_names:
        static_arg_names.append(name)
        static_args.append(arg)
      else:
        dynamic_arg_names.append(name)
        leaves, treedef = pytree_registry.flatten(arg)
        flat.extend(leaves)
        treedefs.append(treedef)
  return (ArgumentSignature(static_args, static_arg_names, dynamic_arg_names,
                            treedefs), flat)
