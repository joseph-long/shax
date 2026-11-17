"""Differential tests: shax's pure-Python _pretty_printer vs real jaxlib.

Expected outputs come from fixtures/pretty_printer_cases.json, generated from
real jaxlib 0.12.0.1 by gen_pretty_printer_fixture.py (see its docstring).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _pp_docgen  # noqa: E402


def _load_pp():
  # Load the module straight from its file so these tests don't depend on the
  # rest of `shax.jax` importing cleanly (importing `shax.jax._src.lib` runs
  # `shax/jax/__init__.py`). The module itself is pure stdlib.
  import importlib.util
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                      os.pardir, "shax", "jax", "_src", "lib",
                      "_pretty_printer.py")
  spec = importlib.util.spec_from_file_location("_shax_pretty_printer", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


pp = _load_pp()

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                        "pretty_printer_cases.json")
with open(_FIXTURE, encoding="utf-8") as _f:
  _DOCS = json.load(_f)["docs"]


@pytest.mark.parametrize("i", range(len(_DOCS)))
def test_against_jaxlib(i):
  case = _DOCS[i]
  spec = case["spec"]
  assert _pp_docgen.doc_repr(pp, spec) == case["repr"]
  for r in case["renders"]:
    text, sm = _pp_docgen.render(pp, spec, r["opts"])
    assert text == r["text"], r["opts"]
    assert sm == r["source_map"], r["opts"]


def test_api_surface():
  assert [c.name for c in pp.Color] == [
      "BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE",
      "RESET"]
  assert pp.Color.RED.value == 31
  assert [i.name for i in pp.Intensity] == ["DIM", "NORMAL", "BRIGHT"]
  assert [o.name for o in pp.OutputFormat] == ["TEXT", "HTML"]
  with pytest.raises(TypeError, match="abstract"):
    pp.Doc()
  with pytest.raises(TypeError, match="abstract"):
    pp.TextDoc()
  d = pp.text("a")
  assert isinstance(d, pp.Doc) and type(d) is pp.TextDoc
  assert repr(d + pp.brk()) == 'concat(text("a"), break(" "))'
  assert d != pp.text("a")
  with pytest.raises(TypeError):
    d + 1
  with pytest.raises(TypeError):
    pp.text(1)
  with pytest.raises(TypeError):
    pp.concat(iter([d]))
  with pytest.raises(TypeError):
    pp.nest(1.5, d)
  assert repr(pp.concat((d, d))) == 'concat(text("a"), text("a"))'
  # ints convert implicitly to enums, like nanobind does.
  assert repr(pp.color(d, 31)) == 'color(text("a"), red, None, None)'
  with pytest.raises(TypeError):
    d._format(80, use_color=False, output_format=pp.OutputFormat.TEXT,
              separable_lines=False, annotation_prefix=" # ", source_map=())
  with pytest.raises(AttributeError):
    d.zz = 1


def test_negative_indent_break_raises():
  doc = pp.nest(-3, pp.concat([pp.text("a"), pp.brk()]))
  with pytest.raises(ValueError):
    doc._format(80, use_color=False, output_format=pp.OutputFormat.TEXT,
                separable_lines=False, annotation_prefix=" # ",
                source_map=None)


def test_upstream_wrapper():
  # The ported jax._src.pretty_printer wraps this module via use_cpp_class.
  try:
    from shax.jax._src import pretty_printer as upp
  except ImportError as e:
    pytest.skip(f"shax.jax does not import yet: {e}")
  doc = upp.group(upp.join(upp.brk(), [upp.text("x"), upp.keyword(upp.text("y"))]))
  assert str(doc) == "x y"
  assert doc.format(1, use_color=False) == "x\ny"
  assert doc.format(80, use_color=True) == "x \x1b[34;1my\x1b[39;22m"
  assert upp.Doc.__module__ == "shax.jax._src.lib._pretty_printer"
