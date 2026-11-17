"""Tests for the pure-Python Traceback in shax/jax/_src/lib/xla_client.py.

Behaviour was checked against real jaxlib 0.12.0.1 `_jax.Traceback` (same
CPython): code_addr2line/code_addr2location agree on every code unit (and
out-of-range/negative offsets) of several stdlib modules; frames, str, equality,
TracebackScope truncation, as_python_traceback/traceback_from_frames output
under `traceback.format_tb` all match. These tests pin that behaviour without
needing jax installed.
"""
import importlib.util
import os
import sys
import traceback as pytb
import types

import pytest


def _load_xc():
  # Load directly from the file so the test doesn't depend on all of shax.jax
  # importing (xla_client itself only needs the stdlib).
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                      os.pardir, "shax", "jax", "_src", "lib", "xla_client.py")
  spec = importlib.util.spec_from_file_location("_shax_xla_client", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


xc = _load_xc()
Traceback = xc.Traceback


def _inner():
  return Traceback.get_traceback(), sys._getframe()


def _outer():
  return _inner()


def test_get_traceback_frames():
  tb, frame = _outer()
  codes, lastis = tb.raw_frames()
  assert codes[0] is _inner.__code__
  assert codes[1] is _outer.__code__
  assert codes[2] is test_get_traceback_frames.__code__
  f0 = tb.frames[0]
  assert f0.file_name == __file__
  assert f0.function_name == "_inner"
  assert f0.function_start_line == _inner.__code__.co_firstlineno
  assert f0.line_num == _inner.__code__.co_firstlineno + 1
  assert repr(f0) == f"_inner;{__file__}:{f0.line_num}"
  # Python frames agree on the code/line of the outer frames (still suspended
  # at the same call, though f_lasti has moved on).
  f = frame.f_back
  for code, lasti, fr in zip(codes[1:2], lastis[1:2], tb.frames[1:2]):
    assert f.f_code is code and f.f_lineno == fr.line_num
    assert Traceback.code_addr2line(code, lasti) == fr.line_num
    f = f.f_back
  assert str(tb).splitlines()[0] == f"{__file__}:{f0.line_num} (_inner)"


def test_equality_hash_add():
  tbs = [Traceback.get_traceback() for _ in range(2)]
  assert tbs[0] == tbs[1] and hash(tbs[0]) == hash(tbs[1])
  other = Traceback.get_traceback()
  assert other != tbs[0]
  assert tbs[0] != 1 and not (tbs[0] == 1)
  with pytest.raises(TypeError):
    tbs[0] < tbs[1]
  both = tbs[0] + other
  assert len(both.frames) == len(tbs[0].frames) + len(other.frames)
  with pytest.raises(TypeError):
    tbs[0] + 1
  empty = Traceback()
  assert str(empty) == "" and empty.frames == [] and empty.raw_frames() == ([], [])
  assert empty.as_python_traceback() is None
  assert bool(empty)
  with pytest.raises(TypeError, match="takes no arguments"):
    Traceback(1)


def test_enable_disable():
  assert xc.tracebacks_enabled()
  xc.set_tracebacks_enabled(False)
  try:
    assert Traceback.get_traceback() is None
  finally:
    xc.set_tracebacks_enabled(True)
  assert Traceback.get_traceback() is not None


def test_traceback_scope():
  def g():
    with xc.TracebackScope() as s:
      assert isinstance(s, xc.TracebackScope)
      a = [f.function_name for f in Traceback.get_traceback().frames]
      def h():
        return [f.function_name for f in Traceback.get_traceback().frames]
      b = h()
    c = Traceback.get_traceback().frames[0].function_name
    return a, b, c
  assert g() == ([], ["test_traceback_scope.<locals>.g.<locals>.h"],
                 "test_traceback_scope.<locals>.g")


def test_python_tracebacks():
  tb, _ = _outer()
  pt = tb.as_python_traceback()
  assert isinstance(pt, types.TracebackType)
  # Outermost frame first, innermost last.
  entries = []
  while pt is not None:
    entries.append((pt.tb_frame.f_code.co_filename, pt.tb_frame.f_code.co_name,
                    pt.tb_lineno))
    pt = pt.tb_next
  assert entries[-1] == (__file__, "_inner", tb.frames[0].line_num)
  assert entries[-2][1] == "_outer"

  x = Traceback.traceback_from_frames(
      [xc.Frame("a.py", "fn", 1, 2), xc.Frame("b.py", "g", 3, 4)])
  assert pytb.format_tb(x) == ['  File "b.py", line 4, in g\n',
                               '  File "a.py", line 2, in fn\n']
  assert x.tb_frame.f_code.co_firstlineno == 4 and x.tb_lasti == 0
  # Negative line numbers (pallas passes -1 for unknown start lines) are OK.
  Traceback.traceback_from_frames([xc.Frame("a.py", "fn", -1, -1)])


def _all_codes(code):
  yield code
  for c in code.co_consts:
    if isinstance(c, types.CodeType):
      yield from _all_codes(c)


def test_code_addr2line_and_location():
  with pytest.raises(Exception, match="code argument must be a code object"):
    Traceback.code_addr2line(1, 0)
  with pytest.raises(Exception, match="code argument must be a code object"):
    Traceback.code_addr2location("x", 0)
  src = open(pytb.__file__).read()
  for code in _all_codes(compile(src, pytb.__file__, "exec")):
    lines = {}
    for start, end, line in code.co_lines():
      for off in range(start, end, 2):
        lines[off] = -1 if line is None else line
    positions = list(code.co_positions())
    for off in range(0, len(code.co_code), 2):
      assert Traceback.code_addr2line(code, off) == lines[off]
      l, el, c, ec = (-1 if v is None else v for v in positions[off // 2])
      assert Traceback.code_addr2location(code, off) == (l, c, el, ec)
    first = code.co_firstlineno
    assert Traceback.code_addr2line(code, -1) == first
    assert Traceback.code_addr2location(code, -1) == (first, 0, first, 0)
