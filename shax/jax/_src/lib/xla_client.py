"""Stand-in for jaxlib.xla_client: devices, the shax client, and the few
XLA types (sharding, layout) that JAX's Python layers reference."""
from __future__ import annotations


class Device:  # SHAX: type name matches jaxlib (repr says CpuDevice)
  """The single host CPU device that shax computes on."""

  platform = "cpu"
  device_kind = "cpu"
  client = None

  def __init__(self, id: int = 0):
    self.id = id
    self.process_index = 0
    self.host_id = 0
    self.task_id = 0
    self.local_hardware_id = id

  def __repr__(self):
    return f"CpuDevice(id={self.id})"

  def __str__(self):
    return f"cpu:{self.id}"

  def default_memory(self):
    return Memory(self, "device")

  def addressable_memories(self):
    return [Memory(self, k) for k in ("device", "pinned_host", "unpinned_host")]

  def memory(self, kind: str):
    if kind not in ("device", "pinned_host", "unpinned_host"):
      raise ValueError(f"Could not find memory addressable by device {self!r}. "
                       f"Device {self!r} can address the following memory kinds: "
                       "device, pinned_host, unpinned_host. Got memory kind: "
                       f"{kind}")
    return Memory(self, kind)

  def memory_stats(self):
    return None

  def live_buffers(self):
    return []


class Memory:
  def __init__(self, device: Device, kind: str):
    self._device = device
    self.kind = kind

  def addressable_by_devices(self):
    return [self._device]

  def __eq__(self, other):
    return (isinstance(other, Memory) and other.kind == self.kind
            and other._device is self._device)

  def __hash__(self):
    return hash((id(self._device), self.kind))

  def __str__(self):
    return f"CPU_DEVICE_{self._device.id}" if self.kind == "device" else self.kind

  __repr__ = __str__


class ArrayMeta(type):
  """Port of jaxlib's ArrayMeta (py_array.cc): Tracers whose aval is an array
  type count as instances of `jax.Array`."""

  def __instancecheck__(cls, x):
    if type.__instancecheck__(cls, x):
      return True
    if cls is Array:
      from shax.jax._src.lib import _jax  # noqa: PLC0415  (import cycle)
      tracer = _jax._tracer_class
      if tracer is not None and isinstance(x, tracer):
        fn = getattr(x, "_is_traced_array", None)
        if fn is not None:
          return bool(fn())
    return False


class Array(metaclass=ArrayMeta):
  """Base of jax.Array (jaxlib's C++ `Array`); jax._src.basearray merges its
  Python definition onto this class with `use_cpp_class(xc.Array)`."""
  __slots__ = ("__weakref__",)


class Client:
  """shax's only backend client: NumPy on the host CPU."""

  platform = "cpu"
  platform_version = "cpu"
  runtime_type = "shax"
  is_c_api = False

  def __init__(self, num_devices: int | None = None):
    # shax computes on one device; requests for more are recorded but ignored
    # (tests that need several devices skip themselves).
    self._devices = [Device(0)]
    for d in self._devices:
      d.client = self

  def devices(self):
    return list(self._devices)

  def live_arrays(self):
    from shax.jax._src.lib import _array  # noqa: PLC0415
    return _array.live_arrays()

  def local_devices(self):
    return list(self._devices)

  def _get_all_devices(self):
    return list(self._devices)

  def device_count(self) -> int:
    return len(self._devices)

  def local_device_count(self) -> int:
    return len(self._devices)

  def process_index(self) -> int:
    return 0

  def get_default_layout(self, dtype, shard_shape, device):
    from shax.jax._src import core  # noqa: PLC0415
    from shax.jax._src.lib._layout import default_pjrt_layout  # noqa: PLC0415
    return default_pjrt_layout(core.ShapedArray(tuple(shard_shape), dtype))

  def __repr__(self):
    return "<shax CPU client>"


def make_cpu_client(num_devices: int | None = None, **kwargs) -> Client:
  return Client(num_devices)


# --- sharding-related types ---------------------------------------------------

class DeviceList(tuple):
  """Immutable, hashable sequence of devices (jaxlib's DeviceList)."""

  def __new__(cls, devices=()):
    return super().__new__(cls, tuple(devices))

  @property
  def is_fully_addressable(self) -> bool:
    return True

  @property
  def addressable_device_list(self) -> "DeviceList":
    return self

  @property
  def default_memory_kind(self) -> str:
    return "device"

  @property
  def memory_kinds(self) -> tuple:
    return ("device", "pinned_host", "unpinned_host")

  @property
  def process_indices(self) -> set:
    return {d.process_index for d in self}

  @property
  def device_kind(self) -> str:
    return self[0].device_kind if self else ""

  def python_device_iterator(self):
    return iter(self)


# Base classes that jaxlib implements in C++; `use_cpp_class(None)` keeps the
# Python class, so these are None.
Sharding = None
PmapSharding = None
# NamedSharding, SingleDeviceSharding, GSPMDSharding: see __getattr__ (_sharding.py)


class _Unsupported:
  """Placeholder for XLA-only types; constructing one is an error in shax."""

  def __init__(self, *args, **kwargs):
    raise NotImplementedError(
        f"shax: {type(self).__name__} (XLA sharding/layout) is not supported")


class OpSharding(_Unsupported):
  class Type:
    REPLICATED = "REPLICATED"
    MAXIMAL = "MAXIMAL"
    TUPLE = "TUPLE"
    OTHER = "OTHER"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"
    UNREDUCED = "UNREDUCED"

  class ShardGroupType:
    AS = "AS"
    LIKE = "LIKE"


class HloSharding:
  """Minimal HloSharding: enough for single-device and replicated shardings.

  Tiled shardings record their tile-assignment dims so replication checks work;
  anything needing real XLA sharding protos is unsupported.
  """

  def __init__(self, kind: str = "replicated", dims=(), devices=None,
               subgroup_types=()):
    self._kind = kind
    self._dims = tuple(dims)
    self._devices = None if devices is None else tuple(devices)
    self._subgroup_types = list(subgroup_types)

  @classmethod
  def replicate(cls):
    return cls("replicated")

  @classmethod
  def manual(cls):
    return cls("manual")

  @classmethod
  def unknown(cls):
    return cls("unknown")

  @classmethod
  def iota_tile(cls, dims, reshape_dims=(), transpose_perm=(), subgroup_types=()):
    return cls("tiled", dims, None, subgroup_types)

  @classmethod
  def subgroup_with_device_ordering(cls, tile_assignment, subgroup_types=()):
    import numpy as np  # noqa: PLC0415
    ta = np.asarray(tile_assignment)
    return cls("tiled", ta.shape, ta.ravel().tolist(), subgroup_types)

  @classmethod
  def from_proto(cls, proto):
    raise NotImplementedError("shax: HloSharding.from_proto is not supported")

  def is_replicated(self) -> bool:
    if self._kind == "replicated":
      return True
    return self._kind == "tiled" and all(d == 1 for d in self._dims[
        :len(self._dims) - len(self._subgroup_types)] or self._dims)

  def is_manual(self) -> bool:
    return self._kind == "manual"

  def is_unknown(self) -> bool:
    return self._kind == "unknown"

  def is_unreduced(self) -> bool:
    return False

  def is_maximal(self) -> bool:
    return False

  def is_tiled(self) -> bool:
    return self._kind == "tiled"

  def tuple_elements(self):
    return []

  def num_devices(self) -> int:
    import math  # noqa: PLC0415
    return math.prod(self._dims) if self._dims else 1

  def num_dimensions(self) -> int:
    return len(self._dims)

  def tile_assignment_dimensions(self):
    return list(self._dims) if self._kind == "tiled" else [1]

  def tile_assignment_devices(self):
    if self._devices is not None:
      return list(self._devices)
    return list(range(self.num_devices()))

  def subgroup_types(self):
    return list(self._subgroup_types)

  def replicate_on_last_tile_dim(self) -> bool:
    return OpSharding.Type.REPLICATED in self._subgroup_types

  def to_proto(self):
    raise NotImplementedError("shax: HloSharding.to_proto is not supported")

  def _key(self):
    if self.is_replicated():
      return ("replicated",)
    return (self._kind, self._dims, self._devices, tuple(self._subgroup_types))

  def __eq__(self, other):
    return isinstance(other, HloSharding) and self._key() == other._key()

  def __hash__(self):
    return hash(self._key())

  def __repr__(self):
    if self.is_replicated():
      return "{replicated}"
    if self._kind == "manual":
      return "{manual}"
    return f"{{devices=[{','.join(map(str, self._dims))}]}}"


from shax.jax._src.lib._layout import Layout  # noqa: E402


class DeviceTopology(_Unsupported):
  pass


class Executable(_Unsupported):
  pass


class LoadedExecutable(_Unsupported):
  pass


class CompileOptions(_Unsupported):
  pass


from shax.jax._src.lib._layout import PjRtLayout  # noqa: E402




# ---------------------------------------------------------------------------
# Tracebacks: a pure-Python port of jaxlib/traceback.cc (jaxlib 0.12.0.1).
#
# A Traceback is an immutable sequence of (code object, lasti) pairs, innermost
# frame first. Equality and hashing are by code-object identity and lasti, like
# the C++ type. jaxlib's `add_exclude_path` does not affect Traceback (it only
# filters the runtime call-location tracebacks in call_location.cc), so neither
# does this.

import sys as _sys
import threading as _threading
import types as _types

_MAX_FRAMES = 512
_tracebacks_enabled = True
_traceback_tls = _threading.local()


def tracebacks_enabled() -> bool:
  return _tracebacks_enabled


def set_tracebacks_enabled(value: bool, /) -> None:
  global _tracebacks_enabled
  _tracebacks_enabled = bool(value)


def _runtime_error(msg):
  # jaxlib raises JaxRuntimeError (a RuntimeError subclass); use shax's if it
  # has one.
  try:
    from shax.jax._src.lib import _jax
  except ImportError:
    _jax = None
  exc = getattr(_jax, "JaxRuntimeError", None) or RuntimeError
  return exc(msg)


def _addr2line(code, lasti: int) -> int:
  # PyCode_Addr2Line.
  if lasti < 0:
    return code.co_firstlineno
  for start, end, line in code.co_lines():
    if start <= lasti < end:
      return -1 if line is None else line
  return -1


def _addr2location(code, lasti: int):
  # PyCode_Addr2Location.
  if lasti < 0:
    return (code.co_firstlineno, 0, code.co_firstlineno, 0)
  # Past the end, CPython's address-range walk stops at the last entry.
  pos = (None, None, None, None)
  for i, pos in enumerate(code.co_positions()):
    if i == lasti // 2:
      break
  line, end_line, col, end_col = (-1 if v is None else v for v in pos)
  return (line, col, end_line, end_col)


class Frame:
  """One decoded traceback frame (jaxlib `_jax.Frame`)."""
  __slots__ = ("_file_name", "_function_name", "_function_start_line",
               "_line_num")

  def __init__(self, file_name: str, function_name: str,
               function_start_line: int, line_num: int, /):
    if not isinstance(file_name, str) or not isinstance(function_name, str):
      raise TypeError("Frame(): file_name and function_name must be str")
    self._file_name = file_name
    self._function_name = function_name
    self._function_start_line = int(function_start_line)
    self._line_num = int(line_num)

  file_name = property(lambda self: self._file_name)
  function_name = property(lambda self: self._function_name)
  function_start_line = property(lambda self: self._function_start_line)
  line_num = property(lambda self: self._line_num)

  def __repr__(self):
    return f"{self._function_name};{self._file_name}:{self._line_num}"


def _decode_frame(code, lasti) -> Frame:
  return Frame(code.co_filename, code.co_qualname, code.co_firstlineno,
               _addr2line(code, lasti))


def _no_columns_linetable(code) -> bytes:
  # Like PyCode_NewEmpty: every instruction maps to co_firstlineno, with no
  # column info (location-table entry kind 13 = NO_COLUMNS, line delta 0).
  units = len(code.co_code) // 2
  out = bytearray()
  while units > 0:
    n = min(units, 8)
    out += bytes([0x80 | (13 << 3) | (n - 1), 0])
    units -= n
  return bytes(out)


_frame_template = compile("__f = __getframe()", "", "exec")
_frame_template = _frame_template.replace(
    co_linetable=_no_columns_linetable(_frame_template))


def _fake_frame(file_name: str, name: str, lineno: int):
  """A real frame object for an empty code object (PyCode_NewEmpty+PyFrame_New)."""
  try:
    code = _frame_template.replace(co_filename=file_name, co_name=name,
                                   co_qualname=name, co_firstlineno=lineno)
  except (ValueError, OverflowError):
    code = _frame_template.replace(co_filename=file_name, co_name=name,
                                   co_qualname=name)
  ns = {"__getframe": _sys._getframe}
  exec(code, ns)
  frame = ns["__f"]
  ns.clear()
  return frame


def _python_traceback(entries):
  """entries: (file_name, name, lineno), innermost first."""
  tb = None
  for file_name, name, lineno in entries:
    tb = _types.TracebackType(tb, _fake_frame(file_name, name, lineno), 0,
                              lineno)
  return tb


class Traceback:
  """A captured Python stack (jaxlib `_jax.Traceback`), innermost frame first."""
  __slots__ = ("_codes", "_lastis")

  def __init__(self, *args, **kwargs):
    if args or kwargs:
      raise TypeError("Traceback() takes no arguments")
    self._codes = ()
    self._lastis = ()

  @classmethod
  def _make(cls, codes, lastis):
    tb = object.__new__(cls)
    tb._codes = tuple(codes)
    tb._lastis = tuple(lastis)
    return tb

  @staticmethod
  def get_traceback():
    """Returns a :class:`Traceback` for the current thread.

    If traceback collection is disabled, returns ``None``.
    """
    if not _tracebacks_enabled:
      return None
    stop = getattr(_traceback_tls, "stop_frame", None)
    codes = []
    lastis = []
    f = _sys._getframe(1)
    while f is not None and len(codes) < _MAX_FRAMES:
      if stop is not None and f is stop:
        break
      codes.append(f.f_code)
      lastis.append(f.f_lasti)
      f = f.f_back
    return Traceback._make(codes, lastis)

  @property
  def frames(self) -> list[Frame]:
    return [_decode_frame(c, l) for c, l in zip(self._codes, self._lastis)]

  def raw_frames(self):
    return (list(self._codes), list(self._lastis))

  def __add__(self, other):
    if type(other) is not Traceback:
      return NotImplemented
    return Traceback._make(self._codes + other._codes,
                           self._lastis + other._lastis)

  def _key(self):
    return (tuple(map(id, self._codes)), self._lastis)

  def __eq__(self, other):
    if type(other) is not Traceback:
      return False
    return (len(self._codes) == len(other._codes)
            and self._lastis == other._lastis
            and all(a is b for a, b in zip(self._codes, other._codes)))

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self._key())

  def __str__(self):
    return "\n".join(f"{fr.file_name}:{fr.line_num} ({fr.function_name})"
                     for fr in self.frames)

  def as_python_traceback(self):
    return _python_traceback(
        (c.co_filename, c.co_name, _addr2line(c, l))
        for c, l in zip(self._codes, self._lastis))

  @staticmethod
  def traceback_from_frames(frames):
    """Creates a traceback from a list of frames."""
    return _python_traceback(
        (f.file_name, f.function_name, f.line_num) for f in frames)

  @staticmethod
  def code_addr2line(code, lasti):
    """Python wrapper around the Python C API function PyCode_Addr2Line"""
    if not isinstance(code, _types.CodeType):
      raise _runtime_error("code argument must be a code object")
    return _addr2line(code, lasti)

  @staticmethod
  def code_addr2location(code, lasti):
    """Python wrapper around the Python C API function PyCode_Addr2Location"""
    if not isinstance(code, _types.CodeType):
      raise _runtime_error("code argument must be a code object")
    return _addr2location(code, lasti)


class TracebackScope:
  """Truncates tracebacks captured inside the scope at the caller's frame."""
  __slots__ = ("_old", "_entered")

  def __init__(self):
    self._old = None
    self._entered = False

  def __enter__(self):
    if self._entered:
      raise ValueError("Cannot enter TracebackScope recursively.")
    self._old = getattr(_traceback_tls, "stop_frame", None)
    self._entered = True
    _traceback_tls.stop_frame = _sys._getframe(1)
    return self

  def __exit__(self, *args):
    _traceback_tls.stop_frame = self._old
    self._old = None
    self._entered = False



# --- unsupported XLA client features / execution backend TODOs -----------------
import enum as _enum
from shax.jax._src.lib import _unsupported as _u


class ArrayCopySemantics(_enum.Enum):
  ALWAYS_COPY = 0
  REUSE_INPUT = 1
  DONATE_INPUT = 2


class AutotuneCacheMode(_enum.Enum):
  UNSPECIFIED = 0
  UPDATE = 1
  READ = 2


DeviceAssignment = _u.cls("xla_client.DeviceAssignment")
XlaComputation = _u.cls("xla_client.XlaComputation")
for _name in ("get_topology_for_devices", "initialize_pjrt_plugin",
              "load_pjrt_plugin_dynamically", "load_pjrt_plugin_with_c_api",
              "make_c_api_client", "make_c_api_device_topology",
              "make_tfrt_tpu_c_api_device_topology",
              "register_custom_call_as_batch_partitionable",
              "register_custom_call_target", "register_custom_type"):
  globals()[_name] = _u.function(f"xla_client.{_name}")


def pjrt_plugin_loaded(name) -> bool:
  return False


def pjrt_plugin_initialized(name) -> bool:
  return False


def batched_block_until_ready(arrays) -> None:
  for a in arrays:
    a.block_until_ready()


def __getattr__(name):
  # Built lazily to avoid import cycles (they construct ArrayImpl).
  if name == "ArrayImpl":
    from shax.jax._src.lib._array import ArrayImpl  # noqa: PLC0415
    globals()["ArrayImpl"] = ArrayImpl
    return ArrayImpl
  if name in ("NamedSharding", "SingleDeviceSharding", "GSPMDSharding"):
    from shax.jax._src.lib import _sharding  # noqa: PLC0415
    globals()[name] = getattr(_sharding, name)
    return globals()[name]
  if name == "XlaRuntimeError":
    return _xla_runtime_error()
  if name in ("array_result_handler", "batched_copy_array_to_devices_with_sharding",
              "batched_device_put", "reorder_shards"):
    from shax.jax._src.lib import _pjit  # noqa: PLC0415
    globals()[name] = getattr(_pjit, name)
    return globals()[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

Shape = _u.cls("xla_client.Shape")
for _name in ("encode_inspect_sharding_callback", "register_custom_call_partitioner"):
  globals()[_name] = _u.function(f"xla_client.{_name}")


def _xla_runtime_error():
  from shax.jax._src.lib._jax import JaxRuntimeError  # noqa: PLC0415
  return JaxRuntimeError


# Registrations of XLA custom calls/types/partitioners are accepted and ignored:
# upstream registers some at import time, and shax never lowers to XLA, so
# invoking a custom call fails elsewhere with a clear error.
def _ignore_registration(*args, **kwargs) -> None:
  return None


register_custom_call_target = _ignore_registration
register_custom_type = _ignore_registration
register_custom_call_as_batch_partitionable = _ignore_registration
register_custom_call_partitioner = _ignore_registration
