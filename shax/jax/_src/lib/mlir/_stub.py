"""Permissive stand-ins for MLIR Python bindings.

shax never lowers to MLIR/StableHLO, but upstream JAX modules reference MLIR
types and dialect ops at import time (type annotations, `register_lowering`
rules, module-level tables). Every attribute of a stub module is a stub class;
stub classes can be subclassed, called, and have arbitrary attributes, and
calling one returns a stub instance that behaves the same way. Nothing here
computes anything; if a lowering rule ever actually runs in shax, that's a bug
upstream of here.
"""
from __future__ import annotations

import types


class _StubMeta(type):
  def __getattr__(cls, name):
    if name.startswith("__") and name.endswith("__"):
      raise AttributeError(name)
    return _make(f"{cls.__qualname__}.{name}")

  def __call__(cls, *args, **kwargs):
    return super().__call__()

  def __repr__(cls):
    return f"<shax mlir stub {cls.__qualname__}>"


class Stub(metaclass=_StubMeta):
  def __init__(self, *args, **kwargs):
    pass

  def __getattr__(self, name):
    if name.startswith("__") and name.endswith("__"):
      raise AttributeError(name)
    return _make(f"{type(self).__qualname__}.{name}")()

  def __call__(self, *args, **kwargs):
    return type(self)()

  def __iter__(self):
    return iter(())

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def __repr__(self):
    return f"<shax mlir stub instance {type(self).__qualname__}>"


def _make(qualname: str) -> type:
  cls = _StubMeta(qualname.rsplit(".", 1)[-1], (Stub,), {})
  cls.__qualname__ = qualname
  return cls


class StubModule(types.ModuleType):
  def __getattr__(self, name):
    if name.startswith("__") and name.endswith("__"):
      raise AttributeError(name)
    value = _make(f"{self.__name__.rsplit('.', 1)[-1]}.{name}")
    setattr(self, name, value)
    return value


def install(module_name: str) -> None:
  """Turn the (otherwise empty) module `module_name` into a stub module."""
  import sys  # noqa: PLC0415
  sys.modules[module_name].__class__ = StubModule
