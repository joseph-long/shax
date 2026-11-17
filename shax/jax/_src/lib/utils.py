"""Stand-in for jaxlib.utils (C++ helpers used by jax._src.util)."""
from __future__ import annotations

import threading


def safe_zip(*args):
  if not args:
    raise TypeError("safe_zip requires at least 1 argument.")
  return list(zip(*args, strict=True))


def safe_map(f, *args):
  args = [list(a) for a in args]
  n = len(args[0])
  for a in args[1:]:
    if len(a) != n:
      raise ValueError(f"safe_map() argument {args.index(a) + 1} is "
                       f"length {len(a)}, but argument 1 is length {n}")
  return list(map(f, *args))


def foreach(f, *args):
  safe_map(f, *args)
  return None


def topological_sort(parents_attr: str, end_nodes) -> list:
  """Port of jaxlib/utils.cc TopologicalSort (same traversal order, which
  tests depend on): parents come before children; end nodes are deduplicated."""
  ends, seen = [], set()
  for n in end_nodes:
    if id(n) not in seen:
      seen.add(id(n))
      ends.append(n)
  if not ends:
    return []
  stack = list(ends)
  child_counts: dict[int, int] = {}
  while stack:
    node = stack.pop()
    count = child_counts.get(id(node), 0)
    if count == 0:
      stack.extend(getattr(node, parents_attr))
    child_counts[id(node)] = count + 1
  for n in ends:
    child_counts[id(n)] -= 1
  childless = [n for n in ends if child_counts[id(n)] == 0]
  out = []
  while childless:
    node = childless.pop()
    out.append(node)
    for parent in getattr(node, parents_attr):
      if child_counts[id(parent)] == 1:
        childless.append(parent)
      else:
        child_counts[id(parent)] -= 1
  out.reverse()
  return out


class Mutex:
  """Reader/writer mutex API backed by a single re-entrant lock."""

  def __init__(self):
    self._lock = threading.RLock()

  def lock(self): self._lock.acquire()
  def unlock(self): self._lock.release()
  writer_lock = lock
  writer_unlock = unlock
  reader_lock = lock
  reader_unlock = unlock
  def assert_held(self): pass
  def assert_reader_held(self): pass

  def __enter__(self):
    self.lock()
    return self

  def __exit__(self, *exc):
    self.unlock()


def install_failure_signal_handler(*args, **kwargs) -> None:
  pass


def absl_set_min_log_level(level: int) -> None:
  pass  # no C++ absl logging in shax
