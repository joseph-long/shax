"""Pure-Python stand-in for jaxlib's C++ ``_pretty_printer`` extension.

A line-by-line port of ``jaxlib/_pretty_printer.cc`` (jaxlib 0.12.0.1): a
Wadler-Lindig pretty printer with annotations, ANSI/HTML colors and source maps.

Faithfulness notes:
  * The C++ code measures text in UTF-8 *bytes* (``std::string::size()``), for
    line fitting, annotation padding and source-map column offsets. We do the
    same so that output is identical for non-ASCII text.
  * Source-map "source" identity is pointer identity in C++ (``is`` here), with
    a distinct "null" source (``_NULL``) that is not ``None``.
"""
from __future__ import annotations

from collections.abc import Sequence
import enum

__all__ = [
    "BreakDoc", "Color", "ColorDoc", "ConcatDoc", "Doc", "GroupDoc",
    "Intensity", "NestDoc", "NilDoc", "OutputFormat", "SourceMapDoc",
    "TextDoc", "brk", "color", "concat", "group", "nest", "nil",
    "source_map", "text",
]


class Color(enum.Enum):
  BLACK = 30
  RED = 31
  GREEN = 32
  YELLOW = 33
  BLUE = 34
  MAGENTA = 35
  CYAN = 36
  WHITE = 37
  RESET = 39


class Intensity(enum.Enum):
  DIM = 2
  NORMAL = 22
  BRIGHT = 1


class OutputFormat(enum.Enum):
  TEXT = 0
  HTML = 1


_RESET = 39
_NORMAL = 22
_DIM = 2
# ColorState = (foreground, background, intensity), as raw ints.
_DEFAULT_COLORS = (_RESET, _RESET, _NORMAL)
_ANNOTATION_COLORS = (_RESET, _RESET, _DIM)

_FLAT = 0
_BREAK = 1


class _Null:
  """Stand-in for a null ``nb::object`` (distinct from ``None``)."""
  __slots__ = ()


_NULL = _Null()


def _nbytes(s: str) -> int:
  return len(s) if s.isascii() else len(s.encode("utf-8"))


def _check_str(name, value, optional=False):
  if optional and value is None:
    return
  if not isinstance(value, str):
    raise TypeError(f"{name}: expected str, got {type(value).__name__}")


def _check_doc(fn, value):
  if not isinstance(value, Doc):
    raise TypeError(
        f"{fn}(): incompatible function arguments; expected Doc, got "
        f"{type(value).__name__}")


def _check_int(fn, value):
  if isinstance(value, bool) or not isinstance(value, int):
    import operator
    try:
      return operator.index(value)
    except TypeError:
      raise TypeError(
          f"{fn}(): incompatible function arguments; expected int, got "
          f"{type(value).__name__}") from None
  return value


def _new(cls):
  return object.__new__(cls)


class Doc:
  """A pretty-printer document (abstract)."""
  __slots__ = ("_num_annotations",)

  def __new__(cls, *args, **kwargs):
    raise TypeError("Doc is abstract and cannot be instantiated")

  def __repr__(self) -> str:
    return self._repr()

  def __add__(self, other):
    if not isinstance(other, Doc):
      raise TypeError(
          "__add__(): incompatible function arguments. The following argument "
          "types are supported:\n    1. (self, other: Doc) -> Doc\n\n"
          f"Invoked with types: {type(self).__module__}.{type(self).__name__}, "
          f"{type(other).__name__}")
    return _make_concat([self, other])

  def _format(self, width, *, use_color, output_format, separable_lines,
              annotation_prefix, source_map):
    width = _check_int("_format", width)
    if not isinstance(output_format, OutputFormat):
      raise TypeError("_format(): output_format must be an OutputFormat")
    _check_str("annotation_prefix", annotation_prefix)
    if source_map is not None and not isinstance(source_map, list):
      raise TypeError("_format(): source_map must be a list or None")
    return _format(self, width, bool(use_color), output_format,
                   bool(separable_lines), annotation_prefix, source_map)

  def _repr(self) -> str:
    raise NotImplementedError


class NilDoc(Doc):
  __slots__ = ()

  def _repr(self):
    return "nil"


class TextDoc(Doc):
  __slots__ = ("_text", "_nbytes", "_annotation", "_anchor", "_href")

  def _repr(self):
    result = f'text("{self._text}"'
    if self._annotation is not None:
      result += f', annotation="{self._annotation}"'
    if self._anchor is not None:
      result += f', anchor="{self._anchor}"'
    if self._href is not None:
      result += f', href="{self._href}"'
    return result + ")"


class ConcatDoc(Doc):
  __slots__ = ("_children",)

  def _repr(self):
    return "concat(" + ", ".join(c._repr() for c in self._children) + ")"


class BreakDoc(Doc):
  __slots__ = ("_text", "_nbytes")

  def _repr(self):
    return f'break("{self._text}")'


class GroupDoc(Doc):
  __slots__ = ("_child",)

  def _repr(self):
    return f"group({self._child._repr()})"


class NestDoc(Doc):
  __slots__ = ("_n", "_child")

  def _repr(self):
    return f"nest({self._n}, {self._child._repr()})"


class SourceMapDoc(Doc):
  __slots__ = ("_child", "_source")

  def _repr(self):
    return f"source({self._child._repr()}, {self._source!r})"


class ColorDoc(Doc):
  __slots__ = ("_child", "_foreground", "_background", "_intensity")

  def _repr(self):
    fg = "None" if self._foreground is None else self._foreground.name.lower()
    bg = "None" if self._background is None else self._background.name.lower()
    it = "None" if self._intensity is None else self._intensity.name.lower()
    return f"color({self._child._repr()}, {fg}, {bg}, {it})"


# ---------------------------------------------------------------------------
# Factory functions


def nil() -> Doc:
  """An empty document."""
  d = _new(NilDoc)
  d._num_annotations = 0
  return d


def text(text: str, annotation: str | None = None, anchor: str | None = None,
         href: str | None = None) -> Doc:
  """Literal text."""
  _check_str("text", text)
  _check_str("annotation", annotation, optional=True)
  _check_str("anchor", anchor, optional=True)
  _check_str("href", href, optional=True)
  d = _new(TextDoc)
  d._text = text
  d._nbytes = _nbytes(text)
  d._annotation = annotation
  d._anchor = anchor
  d._href = href
  d._num_annotations = 0 if annotation is None else 1
  return d


def _make_concat(children: list) -> Doc:
  d = _new(ConcatDoc)
  d._children = children
  d._num_annotations = sum(c._num_annotations for c in children)
  return d


def concat(children: Sequence[Doc]) -> Doc:
  """Concatenation of documents."""
  if not (hasattr(children, "__len__") and hasattr(children, "__getitem__")):
    raise TypeError(
        "concat(): incompatible function arguments; expected a sequence of "
        f"Doc, got {type(children).__name__}")
  children = list(children)
  for c in children:
    _check_doc("concat", c)
  return _make_concat(children)


def brk(text: str = " ") -> Doc:
  """A break.

Prints either as a newline or as `text`, depending on the enclosing group.
"""
  _check_str("brk", text)
  d = _new(BreakDoc)
  d._text = text
  d._nbytes = _nbytes(text)
  d._num_annotations = 0
  return d


def group(arg: Doc, /) -> Doc:
  """Layout alternative groups.

Prints the group with its breaks as their text (typically spaces) if the
entire group would fit on the line when printed that way. Otherwise, breaks
inside the group as printed as newlines.
"""
  _check_doc("group", arg)
  d = _new(GroupDoc)
  d._child = arg
  d._num_annotations = arg._num_annotations
  return d


def nest(n: int, child: Doc) -> Doc:
  """Increases the indentation level by `n`."""
  n = _check_int("nest", n)
  _check_doc("nest", child)
  d = _new(NestDoc)
  d._n = n
  d._child = child
  d._num_annotations = child._num_annotations
  return d


def _opt_enum(enum_cls, value):
  if value is None or isinstance(value, enum_cls):
    return value
  if isinstance(value, int) and not isinstance(value, bool):
    try:
      return enum_cls(value)
    except ValueError:
      pass
  raise TypeError(
      f"color(): incompatible function arguments; expected "
      f"{enum_cls.__name__} | None, got {type(value).__name__}")


def color(child: Doc, foreground: Color | None = None,
          background: Color | None = None,
          intensity: Intensity | None = None) -> Doc:
  """ANSI colors.

Overrides the foreground/background/intensity of the text for the child doc.
Requires use_colors=True to be set when printing; otherwise does nothing.
"""
  _check_doc("color", child)
  d = _new(ColorDoc)
  d._child = child
  d._foreground = _opt_enum(Color, foreground)
  d._background = _opt_enum(Color, background)
  d._intensity = _opt_enum(Intensity, intensity)
  d._num_annotations = child._num_annotations
  return d


def source_map(doc: Doc, source: object) -> Doc:
  """Source mapping.

A source map associates a region of the pretty-printer's text output with a
source location that produced it. For the purposes of the pretty printer a
``source`` may be any object: we require only that we can compare sources for
equality. A text region to source object mapping can be populated as a side
output of the ``format`` method.
"""
  _check_doc("source_map", doc)
  d = _new(SourceMapDoc)
  d._child = doc
  d._source = source
  d._num_annotations = doc._num_annotations
  return d


# ---------------------------------------------------------------------------
# Layout


def _fits(doc: Doc, width: int) -> bool:
  agenda = [doc]
  while width >= 0 and agenda:
    d = agenda.pop()
    t = type(d)
    if t is TextDoc or t is BreakDoc:
      width -= d._nbytes
    elif t is ConcatDoc:
      agenda.extend(reversed(d._children))
    elif t is NilDoc:
      pass
    else:  # Group, Nest, SourceMap, Color
      agenda.append(d._child)
  return width >= 0


def _sparse(doc: Doc) -> bool:
  # Faithful port of the C++ Sparse(); note that, as written upstream, the
  # counter only decreases so this always returns True.
  if doc._num_annotations == 0:
    return True
  agenda = [doc]
  num_annotations = 0
  seen_break = False
  while agenda:
    d = agenda.pop()
    t = type(d)
    if t is TextDoc:
      if d._annotation is not None:
        if num_annotations >= 1 and seen_break:
          return False
        num_annotations -= 1
    elif t is BreakDoc:
      seen_break = True
    elif t is ConcatDoc:
      agenda.extend(reversed(d._children))
    elif t is NilDoc:
      pass
    else:
      agenda.append(d._child)
  return True


def _escape_html(s: str) -> str:
  return (s.replace("&", "&amp;").replace('"', "&quot;")
          .replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;"))


def _html_span_opening_tag(state) -> str:
  if state == _DEFAULT_COLORS:
    return ""
  fg, bg, it = state
  classes = []
  if fg != _RESET:
    classes.append(f"ansi-fg-{fg}")
  if bg != _RESET:
    classes.append(f"ansi-bg-{bg + 10}")
  if it != _NORMAL:
    classes.append(f"ansi-intensity-{it}")
  return '<span class="' + " ".join(classes) + '">'


class _FormatState:
  __slots__ = ("width", "agenda", "parts", "nbytes", "k", "line_annotations",
               "color", "source_map", "line_source_map", "source_start",
               "source", "lines", "html", "separable_lines")

  def append(self, s: str) -> None:
    if s:
      self.parts.append(s)
      self.nbytes += _nbytes(s)

  def update_color(self, update) -> str:
    cur = self.color
    if cur is None or cur == update:
      return ""
    if self.html:
      result = ""
      if cur != _DEFAULT_COLORS:
        result += "</span>"
      if update != _DEFAULT_COLORS:
        result += _html_span_opening_tag(update)
    else:
      codes = []
      if cur[0] != update[0]:
        codes.append(str(update[0]))
      if cur[1] != update[1]:
        codes.append(str(update[1] + 10))
      if cur[2] != update[2]:
        codes.append(str(update[2]))
      result = "\033[" + ";".join(codes) + "m"
    self.color = update
    return result

  def end_line(self, next_indent: int) -> None:
    if self.source_map is not None:
      self.append(self.update_color(_DEFAULT_COLORS))
      pos = self.nbytes
      if self.source_start != pos and self.source is not _NULL:
        self.line_source_map.append((self.source_start, pos, self.source))
      self.source_map.append(self.line_source_map)
      self.line_source_map = []
      self.source_start = next_indent
    if self.separable_lines or self.line_annotations:
      self.append(self.update_color(_DEFAULT_COLORS))
    self.lines.append(("".join(self.parts), self.k, self.line_annotations))


def _format(doc: Doc, width: int, use_color: bool, output_format: OutputFormat,
            separable_lines: bool, annotation_prefix: str,
            source_map: list | None) -> str:
  st = _FormatState()
  st.color = _DEFAULT_COLORS if use_color else None
  st.html = output_format is OutputFormat.HTML
  html = st.html
  st.separable_lines = separable_lines
  st.width = width
  st.source_start = 0
  st.source_map = source_map
  st.line_source_map = []
  st.source = _NULL
  st.parts = []
  st.nbytes = 0
  st.k = 0
  st.line_annotations = []
  st.lines = []

  # Agendum: (indent, mode, doc, color, source)
  agenda = [(0, _BREAK, doc, _DEFAULT_COLORS, _NULL)]
  st.agenda = agenda
  have_source_map = source_map is not None
  while agenda:
    indent, mode, d, color_, source = agenda.pop()
    if have_source_map and source is not st.source:
      st.append(st.update_color(_DEFAULT_COLORS))
      pos = st.nbytes
      if st.source_start != pos and st.source is not _NULL:
        st.line_source_map.append((st.source_start, pos, st.source))
      st.source = source
      st.source_start = pos

    t = type(d)
    if t is TextDoc:
      st.append(st.update_color(color_))
      has_link = html and (d._anchor is not None or d._href is not None)
      if has_link:
        link = "<a"
        if d._anchor is not None:
          link += f' id="{d._anchor}"'
        if d._href is not None:
          link += f' href="{d._href}"'
        st.append(link + ">")
      st.append(_escape_html(d._text) if html else d._text)
      if has_link:
        st.append("</a>")
      if d._annotation is not None:
        st.line_annotations.append(d._annotation)
      st.k += d._nbytes
    elif t is ConcatDoc:
      src = st.source
      for c in reversed(d._children):
        agenda.append((indent, mode, c, color_, src))
    elif t is BreakDoc:
      if mode == _BREAK:
        st.end_line(indent)
        if indent < 0:
          # std::string(negative, ' ') throws in C++.
          raise ValueError("basic_string::_M_create")
        st.parts = [" " * indent] if indent else []
        st.nbytes = indent
        st.line_annotations = []
        st.k = indent
      else:
        st.append(st.update_color(color_))
        st.append(_escape_html(d._text) if html else d._text)
        st.k += d._nbytes
    elif t is GroupDoc:
      fits = _fits(d, st.width - st.k) and _sparse(d)
      agenda.append((indent, _FLAT if fits else _BREAK, d._child, color_,
                     st.source))
    elif t is NestDoc:
      agenda.append((indent + d._n, mode, d._child, color_, st.source))
    elif t is SourceMapDoc:
      agenda.append((indent, mode, d._child, color_, d._source))
    elif t is ColorDoc:
      fg, bg, it = color_
      if d._foreground is not None:
        fg = d._foreground.value
      if d._background is not None:
        bg = d._background.value
      if d._intensity is not None:
        it = d._intensity.value
      agenda.append((indent, mode, d._child, (fg, bg, it), st.source))
    # NilDoc: nothing.

  st.end_line(0)

  max_width = 0
  for line in st.lines:
    max_width = max(max_width, line[1])

  out = []
  for i, (line_text, line_width, annotations) in enumerate(st.lines):
    if i > 0:
      out.append("\n")
    out.append(line_text)
    for j, ann in enumerate(annotations):
      if j > 0:
        out.append("\n")
      if use_color:
        out.append(st.update_color(_ANNOTATION_COLORS))
      padding = (max_width - line_width) if j == 0 else max_width
      out.append(" " * padding)
      out.append(annotation_prefix)
      out.append(ann)
      if use_color:
        out.append(st.update_color(_DEFAULT_COLORS))
  if use_color:
    out.append(st.update_color(_DEFAULT_COLORS))
  return "".join(out)


for _cls in (Color, Intensity, OutputFormat, Doc, NilDoc, TextDoc, ConcatDoc,
             BreakDoc, GroupDoc, NestDoc, SourceMapDoc, ColorDoc):
  _cls.__module__ = __name__
del _cls
