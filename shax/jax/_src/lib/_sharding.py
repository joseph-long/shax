"""Pure-Python port of jaxlib/sharding.cc: the C++ cores of NamedSharding,
SingleDeviceSharding and GSPMDSharding. Upstream Python classes are merged onto
these with `use_cpp_class`, exactly as with jaxlib's C++ classes."""
from __future__ import annotations

from shax.jax._src import sharding as jsharding
from shax.jax._src.lib.xla_client import DeviceList, HloSharding

_VALID_MEMORY_KINDS = ("device", "pinned_host", "unpinned_host")


def _check_and_canonicalize_memory_kind(memory_kind, device_list):
  if memory_kind is not None:
    supported = device_list.memory_kinds
    if memory_kind in supported:
      return memory_kind
    if len(device_list.addressable_device_list) == 0:
      return memory_kind
    kind = device_list.addressable_device_list[0].device_kind
    raise ValueError(
        f"Could not find memory addressable by device {kind}. Device {kind} "
        f"can address the following memory kinds: {', '.join(supported)}. "
        f"Got memory kind: {memory_kind}")
  return device_list.default_memory_kind


class NamedSharding(jsharding.Sharding):

  def __init__(self, mesh, spec, *, memory_kind=None, _logical_device_ids=None):
    self._mesh = mesh
    self._spec = spec
    self.__memory_kind = memory_kind
    self.__logical_device_ids = _logical_device_ids
    idl = mesh._internal_device_list
    self.__internal_device_list = idl
    if idl is not None:
      self.__memory_kind = _check_and_canonicalize_memory_kind(memory_kind, idl)
    elif memory_kind is not None and memory_kind not in _VALID_MEMORY_KINDS:
      raise ValueError(f"Got invalid memory kind: {memory_kind}. Valid memory "
                       f"kinds are: {', '.join(_VALID_MEMORY_KINDS)}")
    from shax.jax._src.named_sharding import check_pspec  # noqa: PLC0415
    check_pspec(mesh, spec)

  mesh = property(lambda self: self._mesh)
  spec = property(lambda self: self._spec)
  _memory_kind = property(lambda self: self.__memory_kind)
  _logical_device_ids = property(lambda self: self.__logical_device_ids)

  @property
  def _internal_device_list(self):
    if self.__internal_device_list is None:
      raise ValueError("Mesh has no devices (it is an AbstractMesh)")
    return self.__internal_device_list

  def __eq__(self, other):
    if other is None or not isinstance(other, NamedSharding):
      return False
    return self is other or (
        self.mesh == other.mesh and self.spec == other.spec
        and self._memory_kind == other._memory_kind
        and self._logical_device_ids == other._logical_device_ids)

  def __hash__(self):
    h = self.__dict__.get("_NamedSharding__hash")
    if h is None:
      h = hash((self.mesh, self.spec, self._memory_kind, self._logical_device_ids))
      self.__dict__["_NamedSharding__hash"] = h
    return h


class SingleDeviceSharding(jsharding.Sharding):

  def __init__(self, device, *, memory_kind=None):
    self.__device = device
    self.__internal_device_list = DeviceList((device,))
    self.__memory_kind = _check_and_canonicalize_memory_kind(
        memory_kind, self.__internal_device_list)

  _device = property(lambda self: self.__device)
  _memory_kind = property(lambda self: self.__memory_kind)
  _internal_device_list = property(lambda self: self.__internal_device_list)


class GSPMDSharding(jsharding.Sharding):

  def __init__(self, devices, op_sharding, *, memory_kind=None):
    if not isinstance(devices, DeviceList):
      devices = DeviceList(tuple(devices))
    if len(devices) == 0:
      raise ValueError("Devices given to GSPMDSharding must not be empty")
    if not isinstance(op_sharding, HloSharding):
      raise NotImplementedError("shax: GSPMDSharding needs an HloSharding")
    self.__devices = devices
    self.__hlo_sharding = op_sharding
    self.__memory_kind = _check_and_canonicalize_memory_kind(memory_kind, devices)

  _devices = property(lambda self: self.__devices)
  _hlo_sharding = property(lambda self: self.__hlo_sharding)
  _memory_kind = property(lambda self: self.__memory_kind)
  _internal_device_list = property(lambda self: self.__devices)
