"""shax's stand-ins for lowering and compilation (the AOT path).

Upstream lowers a jaxpr to a StableHLO module (`mlir.lower_jaxpr_to_module`),
compiles it with XLA (`pxla._cached_compilation`), and runs the executable
(`pxla.ExecuteReplicated`). shax keeps upstream's object model around those
steps (`stages.Lowered`/`Compiled`, `pxla.MeshComputation`/`MeshExecutable`)
and swaps in three objects at the bottom, each wired in by a `# SHAX:` edit:

- `ShaxModule`: what "lowering" returns instead of an `ir.Module`. It keeps the
  jaxpr. It has no StableHLO, so asking for MLIR/HLO text raises
  `NotImplementedError` (which `Lowered.compiler_ir("hlo")` and
  `Compiled.as_text()` turn into `None`, as JAX allows).
- `ShaxExecutable`: what "compiling" returns instead of an XLA
  `LoadedExecutable`.
- `ShaxExecute`: the `ExecuteReplicated` that runs it, by evaluating the
  jaxpr with the NumPy backend (`shax_backend.run_jaxpr`).
"""
from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
import re
import weakref
from typing import Any

import numpy as np

NO_HLO_MESSAGE = (
    "shax: no StableHLO/HLO; shax doesn't lower to XLA. Use jax.make_jaxpr "
    "to inspect the program")


def _no_hlo(*args, **kwargs):
  raise NotImplementedError(NO_HLO_MESSAGE)


class _NoHloOperation:
  """`ir.Module.operation` stand-in: printing/serializing has nothing to show."""
  print = staticmethod(_no_hlo)
  write_bytecode = staticmethod(_no_hlo)
  get_asm = staticmethod(_no_hlo)
  verify = staticmethod(lambda: True)

  @property
  def attributes(self):
    raise NotImplementedError(NO_HLO_MESSAGE)


class ShaxModule:
  """The result of "lowering" a jaxpr in shax: the jaxpr itself.

  Stands in for the `ir.Module` that upstream's `MeshComputation` holds.
  """

  def __init__(self, name: str, jaxpr, *, num_const_args: int,
               in_avals: Sequence[Any], out_avals: Sequence[Any],
               ordered_effects: Sequence[Any], platforms: Sequence[str],
               input_output_aliases=None):
    self.name = name
    # The jaxpr to run (const args still inlined), as a shallow copy: this
    # module is the value of `pxla._cached_lowering_to_hlo`, which is keyed
    # weakly on the jaxpr passed in; referencing that key would keep the cache
    # entry (and the jaxpr's constants) alive forever.
    self.jaxpr = copy.copy(jaxpr)
    self.num_const_args = num_const_args
    self.in_avals = tuple(in_avals)
    self.out_avals = tuple(out_avals)
    self.ordered_effects = list(ordered_effects)
    self.platforms = tuple(platforms)
    self.input_output_aliases = input_output_aliases
    self.operation = _NoHloOperation()

  def __str__(self):
    return (f"shax module {self.name} (shax doesn't lower to StableHLO; this "
            f"is the jaxpr it runs):\n{self.jaxpr}")

  def __repr__(self):
    return f"<ShaxModule {self.name}>"

  as_hlo_module = as_hlo_text = as_serialized_hlo_module_proto = _no_hlo


def lower_jaxpr_to_module(module_name, jaxpr, *, num_const_args, in_avals,
                          out_avals, ordered_effects, platforms,
                          input_output_aliases, dim_vars):
  """The tail of `mlir.lower_jaxpr_to_module`, minus MLIR."""
  from shax.jax._src import util  # noqa: PLC0415
  from shax.jax._src.interpreters import mlir  # noqa: PLC0415
  _check_lowering_rules(jaxpr.jaxpr, platforms)
  module = ShaxModule(module_name, jaxpr, num_const_args=num_const_args,
                      in_avals=in_avals, out_avals=out_avals,
                      ordered_effects=ordered_effects, platforms=platforms,
                      input_output_aliases=input_output_aliases)
  util.test_event("mlir.collect_lowered_jaxprs", jaxpr, module)
  return mlir.LoweringResult(
      module, [], [], mlir.ShapePolyLoweringState(dim_vars, platforms))


def _lowering_time_checks(eqn) -> None:
  """Errors that upstream lowering *rules* raise, which shax (running no
  lowering rules) reproduces here."""
  from shax.jax._src import checkify, config  # noqa: PLC0415
  if eqn.primitive is checkify.check_p:
    # checkify.check_lowering_rule (cpu): a check that was never
    # functionalized can't be staged, unless XLA runtime errors are enabled.
    if not eqn.params["debug"] and not config.xla_runtime_errors.value:
      raise checkify.functionalization_error


def _check_lowering_rules(jaxpr, platforms) -> None:
  """Fail like upstream's lowering when a primitive has no MLIR lowering rule
  for the platform (`mlir._get_lowering_rules` / `lower_per_platform`).

  shax never runs lowering rules, but whether one is registered is part of
  JAX's observable behaviour: jit of a primitive that only has an `impl`
  and an abstract eval fails at lowering time.
  """
  from shax.jax._src import core, pjit  # noqa: PLC0415
  from shax.jax._src.interpreters import mlir  # noqa: PLC0415
  # Only jaxprs that upstream lowers as they are (the top level, nested jits
  # and closed calls). Higher-order primitives' lowering rules may rewrite
  # their bodies first (e.g. cond/scan/while discharge Ref effects), so a
  # rule-less primitive there needn't be an error.
  plain_calls = {pjit.jit_p, core.closed_call_p, core.call_p}
  seen: set = set()

  def check(j):
    for eqn in j.eqns:
      prim = eqn.primitive
      if prim not in seen:
        seen.add(prim)
        if prim not in mlir._lowerings:
          missing = [p for p in platforms
                     if prim not in mlir._platform_specific_lowerings[p]]
          if missing:
            where = (f"platform {missing[0]}" if len(platforms) == 1
                     else f"platforms {missing}")
            raise NotImplementedError(
                f"MLIR translation rule for primitive '{prim}' not found "
                f"for {where}")
      _lowering_time_checks(eqn)
      if prim in plain_calls:
        for sub in core.jaxprs_in_params(eqn.params):
          check(sub)

  check(jaxpr)


# --- compilation ---------------------------------------------------------------

_INT_RE = re.compile(r"\s*[+-]?\d+\s*\Z")
_INT_RANGES = {"int32": (-2**31, 2**31 - 1), "int64": (-2**63, 2**63 - 1),
               "uint32": (0, 2**32 - 1), "uint64": (0, 2**64 - 1)}


def _option_str(v) -> str:
  if isinstance(v, bool):
    return str(int(v))
  if isinstance(v, float):
    return f"{v:g}"
  return str(v)


def _option_ok(kind: str, enum_values, v) -> bool:
  """Whether jaxlib accepts `v` for a DebugOptions field of type `kind`.

  Mirrors what real jaxlib 0.12 accepts (checked with `.venv-jaxref`).
  """
  if kind == "bool":
    return (isinstance(v, bool) or (isinstance(v, str) and v in ("True", "False"))
            or (isinstance(v, (int, float)) and v in (1, 1.0))
            or (isinstance(v, float) and v == 0.0))
  if kind in _INT_RANGES:
    lo, hi = _INT_RANGES[kind]
    if isinstance(v, str) and _INT_RE.match(v):
      v = int(v)
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi
  if kind in ("float", "double"):
    if isinstance(v, str):
      try:
        float(v)
      except ValueError:
        return False
      return True
    return isinstance(v, float)
  if kind in ("string", "repeated_string", "bytes"):
    return isinstance(v, str)
  if kind in ("enum", "repeated_enum"):
    if isinstance(v, int) and not isinstance(v, bool):
      return True
    if not isinstance(v, str):
      return False
    names = v.split(",") if kind == "repeated_enum" else [v]
    return all(_INT_RE.match(n) or (n.strip() in enum_values if enum_values
                                    else bool(n.strip()))
               for n in names)
  return True  # message fields: not validated


def check_compiler_options(compiler_options_kvs) -> None:
  """Validate `compiler_options` as jaxlib does when compiling.

  The options JAX applies itself (`compiler.get_compile_options`) are handled
  as upstream does; the rest must name `xla.DebugOptions` fields.
  """
  from shax.jax._src import compiler  # noqa: PLC0415
  from shax.jax._src.lib import _jax  # noqa: PLC0415
  from shax.jax._src.lib._xla_debug_options import DEBUG_OPTIONS  # noqa: PLC0415
  for name, v in compiler_options_kvs:
    if name == "individually_defined_output_indices" or name == "fdo_profile":
      continue
    if name in ("optimization_level", "memory_fitting_level"):
      compiler.CompilerEffortLevel(v)  # ValueError, as upstream
      continue
    if v is None or not isinstance(v, (str, bool, int, float)):
      raise TypeError(
          "set_env_option_overrides(): incompatible function arguments; values "
          f"must be str | bool | int | float, got {type(v).__name__}")
    field = DEBUG_OPTIONS.get(name)
    if field is None:
      raise _jax.JaxRuntimeError(
          f"INVALID_ARGUMENT: No such compile option: '{name}'")
    kind, enum_values = field
    if not _option_ok(kind, enum_values, v):
      shown = kind[len("repeated_"):] if kind.startswith("repeated_") else kind
      raise _jax.JaxRuntimeError(
          f"INVALID_ARGUMENT: While setting option {name}, '{_option_str(v)}' "
          f"is not a valid {shown} value.")


@dataclasses.dataclass(frozen=True)
class CompiledMemoryStats:
  """Like jaxlib's `CompiledMemoryStats`. shax knows argument and output sizes;
  it generates no code and has no temporaries it can account for."""
  generated_code_size_in_bytes: int = 0
  argument_size_in_bytes: int = 0
  output_size_in_bytes: int = 0
  alias_size_in_bytes: int = 0
  temp_size_in_bytes: int = 0
  host_generated_code_size_in_bytes: int = 0
  host_argument_size_in_bytes: int = 0
  host_output_size_in_bytes: int = 0
  host_alias_size_in_bytes: int = 0
  host_temp_size_in_bytes: int = 0
  serialized_buffer_assignment_proto: bytes = b""

  def __str__(self):
    fields = ", ".join(f"{f.name}={getattr(self, f.name)}"
                       for f in dataclasses.fields(self)
                       if f.name != "serialized_buffer_assignment_proto")
    return f"CompiledMemoryStats({fields})"


def _nbytes(avals) -> int:
  from shax.jax._src import core  # noqa: PLC0415
  total = 0
  for a in avals:
    if isinstance(a, core.ShapedArray):
      pa = core.physical_aval(a)
      total += int(np.prod(pa.shape, dtype=np.int64)) * np.dtype(pa.dtype).itemsize
  return total


_LIVE_EXECUTABLES: "weakref.WeakSet[ShaxExecutable]" = weakref.WeakSet()


def live_executables() -> list:
  """`Client.live_executables()`: the executables not yet garbage collected."""
  return list(_LIVE_EXECUTABLES)


class ShaxExecutable:
  """Stands in for an XLA `LoadedExecutable`: a `ShaxModule`, "compiled"."""

  def __init__(self, module: ShaxModule, name: str, devices,
               compiler_options_kvs):
    # A copy: `pxla._cached_compilation` is keyed weakly on `module`, and this
    # executable is the cached value, so it must not reference the key.
    self.module = copy.copy(module)
    self.name = name
    self._devices = list(devices)
    self.compiler_options_kvs = compiler_options_kvs
    self.fingerprint = None
    _LIVE_EXECUTABLES.add(self)

  def __repr__(self):
    return f"<ShaxExecutable {self.name}>"

  def local_devices(self):
    return self._devices

  def get_hlo_text(self):
    raise NotImplementedError(NO_HLO_MESSAGE)

  def hlo_modules(self):
    raise NotImplementedError(NO_HLO_MESSAGE)

  def get_parameter_layouts(self):
    from shax.jax._src.lib._layout import default_pjrt_layout  # noqa: PLC0415
    tokens = [default_pjrt_layout(None)] * len(self.module.ordered_effects)
    return [*tokens, *map(default_pjrt_layout, self.module.in_avals)]

  def get_output_layouts(self):
    from shax.jax._src.lib._layout import default_pjrt_layout  # noqa: PLC0415
    tokens = [default_pjrt_layout(None)] * len(self.module.ordered_effects)
    return [*tokens, *map(default_pjrt_layout, self.module.out_avals)]

  def cost_analysis(self):
    return {}  # shax has no cost model

  def get_compiled_memory_stats(self):
    m = self.module
    return CompiledMemoryStats(argument_size_in_bytes=_nbytes(m.in_avals),
                               output_size_in_bytes=_nbytes(m.out_avals))

  def execute_sharded(self, *args, **kwargs):
    raise NotImplementedError("shax: ShaxExecutable runs through ShaxExecute")

  def delete(self):
    pass

  def is_deleted(self):
    return False


def compile_module(module: ShaxModule, name, devices, compiler_options_kvs):
  """Stand-in for `compiler.compile_or_get_cached`."""
  from shax.jax._src import dispatch  # noqa: PLC0415
  with dispatch.log_elapsed_time(
      "Finished XLA compilation of {fun_name} in {elapsed_time:.9f} sec",
      fun_name=name, event=dispatch.BACKEND_COMPILE_EVENT):
    check_compiler_options(compiler_options_kvs)
    return ShaxExecutable(module, name, devices, compiler_options_kvs)


# --- execution -----------------------------------------------------------------

def _is_ext(aval) -> bool:
  from shax.jax._src import core, dtypes  # noqa: PLC0415
  return (isinstance(aval, core.ShapedArray)
          and dtypes.issubdtype(aval.dtype, dtypes.extended))


def _donated_inputs(module: ShaxModule, input_bufs) -> list[int]:
  """Indices of the inputs XLA would consume (donated and aliased to an
  output), checking, as PjRt does, that none is also passed elsewhere."""
  aliases = module.input_output_aliases
  if not aliases:
    return []
  donated = [i for i, a in enumerate(aliases)
             if a is not None and i < len(input_bufs)
             and hasattr(input_bufs[i], "delete")
             and not isinstance(input_bufs[i], np.ndarray)]
  if donated:
    first_use: dict[int, int] = {}
    for j, b in enumerate(input_bufs):
      first_use.setdefault(id(b), j)
    for i in donated:
      b = input_bufs[i]
      for j, other in enumerate(input_bufs):
        if j != i and other is b:
          from shax.jax._src.lib import _jax  # noqa: PLC0415
          raise _jax.JaxRuntimeError(
              "INVALID_ARGUMENT: Attempt to use a buffer that was previously "
              "donated in the same call to Execute() (flattened argument "
              f"{max(i, j)}, replica 0, partition 0, first use: {min(i, j)}). "
              "Toy example for this bug: `f(donate(a), a)`.")
  return donated


def _make_execute_class():
  from shax.jax._src import config, core, dispatch, dtypes  # noqa: PLC0415
  from shax.jax._src.interpreters import pxla  # noqa: PLC0415
  from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
  from shax.jax._src import shax_backend  # noqa: PLC0415

  class ShaxExecute(pxla.ExecuteReplicated):
    """`ExecuteReplicated` for a `ShaxExecutable`: evaluates the jaxpr.

    Takes the same arguments as upstream's executable (const args, then the
    kept args, then mutable inputs); the const args are inlined in the jaxpr
    as literals, so they are dropped. Effects run in program order as the
    jaxpr is evaluated, so no runtime tokens are threaded. Donation is ignored.
    """
    __slots__ = ("out_avals", "out_shardings", "committed")

    def __init__(self, *args, out_avals, out_shardings, committed, **kwargs):
      super().__init__(*args, **kwargs)
      self.out_avals = out_avals
      self.out_shardings = out_shardings
      self.committed = committed

    def __call__(self, *args):
      if config.no_execution.value:
        raise RuntimeError(
            f"JAX tried to execute function {self.name}, but the no_execution "
            "config option is set")
      args = [x for i, x in enumerate(args) if i in self.kept_var_idx]
      if self.mut:
        args = [*args, *self.mut.in_mut]
      input_bufs = self.in_handler(args)
      module = self.xla_executable.module
      # shard_args turns PRNG key arrays into their physical buffers (what an
      # XLA executable takes); the jaxpr wants the key arrays themselves.
      # Likewise tokens: shard_args unwraps a core.Token to its buffer.
      input_bufs = [x if _is_ext(a) or isinstance(a, core.AbstractToken)
                    else b for x, b, a in
                    zip(args, input_bufs, module.in_avals, strict=True)]
      # (Check and consume the caller's arrays: in jaxlib, shard_args with
      # REUSE_INPUT shares their buffers.)
      nested = shax_backend.evaluating_jit()
      donated = _donated_inputs(module, args)
      outs = shax_backend.run_jaxpr(module.jaxpr,
                                    input_bufs[module.num_const_args:])
      # Donated buffers that XLA would alias to an output are consumed.
      for i in donated:
        args[i].delete()
        if input_bufs[i] is not args[i] and hasattr(input_bufs[i], "delete"):
          input_bufs[i].delete()
      out = [self._handle_out(o, a, s) for o, a, s in
             zip(outs, self.out_avals, self.out_shardings, strict=True)]
      if dispatch.needs_check_special() and not nested:
        # As upstream's result handlers: raises InternalFloatingPointError,
        # which pjit catches to re-run de-optimized and find the culprit.
        # A jit called while evaluating another jit's jaxpr would be inlined
        # into the outer executable by XLA, so only outer outputs are checked.
        for o in out:
          if type(o) is ArrayImpl:
            dispatch.check_special_array(self.name, o)
      if self.mut is None:
        return out
      out_ = []
      for i, o in zip(self.mut.out_mut, out):
        if i is not None:
          try: args[i]._refs._buf._replace_with(o)
          except AttributeError: pass
        else:
          out_.append(o)
      return out_

    def _handle_out(self, o, aval, sharding):
      # Give outputs the executable's output sharding and committedness, as
      # the result handlers of upstream's executable do.
      if (type(o) is not ArrayImpl or not isinstance(aval, core.ShapedArray)
          or dtypes.issubdtype(aval.dtype, dtypes.extended)
          or not hasattr(sharding, "_device_assignment")):
        return o
      if o.sharding == sharding and o._committed == self.committed:
        return o
      return ArrayImpl(o.aval, sharding, [o._value], self.committed,
                       _skip_checks=True)

  return ShaxExecute


_execute_cls = None


def make_execute(unloaded) -> Any:
  """`UnloadedMeshExecutable.build_unsafe_call` for a `ShaxExecutable`."""
  global _execute_cls
  from shax.jax._src.interpreters import pxla  # noqa: PLC0415
  if _execute_cls is None:
    _execute_cls = _make_execute_class()
  handle_args = pxla.InputsHandler(unloaded.input_shardings,
                                   unloaded.dispatch_in_layouts)
  return _execute_cls(
      unloaded.xla_executable, unloaded.name, unloaded.backend, handle_args,
      None, unloaded.unordered_effects, unloaded.ordered_effects,
      unloaded.keepalive, bool(unloaded.host_callbacks),
      unloaded.kept_var_idx, unloaded.mut, unloaded.pgle_profiler,
      out_avals=unloaded.output_avals,
      out_shardings=unloaded.output_shardings,
      committed=unloaded.committed)
