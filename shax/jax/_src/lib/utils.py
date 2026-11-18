"""Stand-in for jaxlib.utils (C++ helpers used by jax._src.util)."""
from __future__ import annotations

import threading


_SEQ = (list, tuple)
_MAX_DRAIN = 10  # utils.cc kMaxDrainCount


def _drain(it) -> int | None:
  """Count what's left in `it`, giving up (None) at _MAX_DRAIN."""
  count = 0
  try:
    for _ in it:
      count += 1
      if count >= _MAX_DRAIN:
        return None
  except Exception:  # noqa: BLE001  (utils.cc clears errors while draining)
    pass
  return count


def _zip_length_error(iterators, n, mismatch, arg_is_longer):
  """utils.cc SetSafeZipLengthError."""
  lengths, capped = [], []
  for j, it in enumerate(iterators):
    if arg_is_longer:
      consumed = n + 1 if j == mismatch else n
    else:
      consumed = n + 1 if j < mismatch else n
    rest = _drain(it)
    if rest is None:
      lengths.append(f"{consumed + _MAX_DRAIN}+")
      capped.append(str(j))
    else:
      lengths.append(str(consumed + rest))
  msg = (f"safe_zip() argument {mismatch} has length {lengths[mismatch]} but "
         f"argument 0 has length {lengths[0]}. All lengths: "
         f"({', '.join(lengths)})")
  if capped:
    msg += f" (Note: length capped for argument(s) {', '.join(capped)})"
  return ValueError(msg)


def safe_zip(*args):
  """Port of jaxlib/utils.cc SafeZip: zip to a list, requiring equal lengths."""
  if not args:
    raise TypeError("safe_zip requires at least 1 argument")
  if all(type(a) in _SEQ for a in args):
    n = len(args[0])
    if all(len(a) == n for a in args):
      return list(zip(*args))
  iterators = [iter(a) for a in args]
  out = []
  sentinel = object()
  first, rest = iterators[0], iterators[1:]
  while True:
    v = next(first, sentinel)
    if v is not sentinel:
      tup = [v]
      for i, it in enumerate(rest, 1):
        w = next(it, sentinel)
        if w is sentinel:
          raise _zip_length_error(iterators, len(out), i, False)
        tup.append(w)
      out.append(tuple(tup))
    else:
      for i, it in enumerate(rest, 1):
        if next(it, sentinel) is not sentinel:
          raise _zip_length_error(iterators, len(out), i, True)
      return out


def safe_map(*args):
  """Port of jaxlib/utils.cc SafeMap: map to a list, requiring equal lengths."""
  if len(args) < 2:
    raise TypeError("safe_map requires at least 2 arguments")
  f, seqs = args[0], args[1:]
  if all(type(a) in _SEQ for a in seqs):
    n = len(seqs[0])
    if all(len(a) == n for a in seqs):
      return list(map(f, *seqs))
  iterators = [iter(a) for a in seqs]
  out = []
  sentinel = object()
  first, rest = iterators[0], iterators[1:]
  while True:
    v = next(first, sentinel)
    if v is not sentinel:
      vals = [v]
      for i, it in enumerate(rest, 1):
        w = next(it, sentinel)
        if w is sentinel:
          raise ValueError(
              f"safe_map() argument {i + 1} is shorter than argument 1")
        vals.append(w)
      out.append(f(*vals))
    else:
      for i, it in enumerate(rest, 1):
        if next(it, sentinel) is not sentinel:
          raise ValueError(
              f"safe_map() argument {i + 1} is longer than argument 1")
      return out


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
