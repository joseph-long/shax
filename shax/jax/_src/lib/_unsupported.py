"""Placeholders for jaxlib features shax doesn't have (profiler, PjRt plugins,
distributed runtime, GPU/TPU libraries, FFI, ...). Importing and referencing
them works; calling or instantiating one raises NotImplementedError."""
from __future__ import annotations


def function(qualname: str):
  def unsupported(*args, **kwargs):
    raise NotImplementedError(f"shax: {qualname} is not supported")
  unsupported.__name__ = qualname.rsplit(".", 1)[-1]
  unsupported.__qualname__ = qualname
  return unsupported


def cls(qualname: str) -> type:
  def __init__(self, *args, **kwargs):
    raise NotImplementedError(f"shax: {qualname} is not supported")
  name = qualname.rsplit(".", 1)[-1]
  return type(name, (), {"__init__": __init__, "__qualname__": qualname})


def backend_todo(qualname: str):
  """Execution-backend pieces shax will implement; not done yet."""
  def todo(*args, **kwargs):
    raise NotImplementedError(f"shax backend TODO: {qualname}")
  todo.__name__ = qualname.rsplit(".", 1)[-1]
  return todo
