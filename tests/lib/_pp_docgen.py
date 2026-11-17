"""Shared helpers for the pretty-printer differential tests.

Docs are described by JSON-able "spec" trees so the same doc can be built with
real ``jaxlib._pretty_printer`` (to generate the fixture) and with shax's
pure-Python port (to check against it). Only stdlib is used here.

Spec forms:
  ["nil"]
  ["text", text, annotation|None, anchor|None, href|None]
  ["brk", text]
  ["concat", [spec, ...]]
  ["group", spec]
  ["nest", n, spec]
  ["color", spec, fg|None, bg|None, intensity|None]   (enum member names)
  ["src", pool_index, spec]
"""
from __future__ import annotations

import random

TEXTS = ["a", "b", "x", "foo", "bar", "baz", "f32[3,4]", "=", ":", "{", "}",
         "(", ")", "lambda", "let", "in", "", "é", "漢字", "<b>", "a&b",
         "'q'", '"dq"', "longer_identifier_name", "   ", "x" * 30]
BRK_TEXTS = [" ", " ", " ", "", ", ", ";", "é", "<>"]
ANNOTATIONS = ["ann", "f32[]", "note é", "<html>"]
COLORS = ["BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN",
          "WHITE", "RESET"]
INTENSITIES = ["DIM", "NORMAL", "BRIGHT"]
POOL_SIZE = 4


def random_spec(rng: random.Random, depth: int = 0, max_depth: int = 5):
  leaf = depth >= max_depth or rng.random() < 0.25 + 0.1 * depth
  if leaf:
    r = rng.random()
    if r < 0.05:
      return ["nil"]
    if r < 0.35:
      return ["brk", rng.choice(BRK_TEXTS)]
    ann = rng.choice(ANNOTATIONS) if rng.random() < 0.12 else None
    anchor = f"id{rng.randrange(9)}" if rng.random() < 0.08 else None
    href = f"#h{rng.randrange(9)}" if rng.random() < 0.08 else None
    return ["text", rng.choice(TEXTS), ann, anchor, href]
  kind = rng.choices(["concat", "group", "nest", "color", "src"],
                     weights=[6, 3, 3, 2, 2])[0]
  sub = lambda: random_spec(rng, depth + 1, max_depth)  # noqa: E731
  if kind == "concat":
    return ["concat", [sub() for _ in range(rng.randrange(0, 6))]]
  if kind == "group":
    return ["group", sub()]
  if kind == "nest":
    return ["nest", rng.choice([0, 1, 2, 2, 4, 7]), sub()]
  if kind == "color":
    opt = lambda xs: rng.choice(xs) if rng.random() < 0.6 else None  # noqa: E731
    return ["color", sub(), opt(COLORS), opt(COLORS), opt(INTENSITIES)]
  return ["src", rng.randrange(POOL_SIZE), sub()]


def jaxpr_like_spec(n_eqns: int, seed: int):
  """Something shaped like a printed jaxpr (nested groups/nests/annotations)."""
  rng = random.Random(seed)
  def var(i):
    return ["color", ["text", f"v{i}", None, None, None], None, None, None]
  def typ(i):
    return ["color", ["text", f"f32[{i},{i + 1}]", None, None, None],
            "MAGENTA", None, "NORMAL"]
  eqns = []
  for i in range(n_eqns):
    lhs = ["concat", [var(i), ["text", ":", None, None, None], typ(i)]]
    args = []
    for j in range(rng.randrange(1, 4)):
      if j:
        args.append(["brk", " "])
      args.append(var(rng.randrange(i + 1)))
    params = ["nest", 2, ["concat", [
        ["brk", ""], ["text", "axes=(0,)", None, None, None], ["brk", " "],
        ["text", "keepdims=False", None, None, None]]]]
    ann = "src.py:%d" % i if rng.random() < 0.3 else None
    rhs = ["concat", [
        ["color", ["text", f"prim_{i}", ann, None, None], "BLUE", None, "BRIGHT"],
        ["group", ["concat", [["text", "[", None, None, None], params,
                              ["text", "]", None, None, None]]]],
        ["brk", " "], ["nest", 2, ["group", ["concat", args]]]]]
    eqn = ["src", i % POOL_SIZE, ["group", ["concat", [
        lhs, ["brk", " "], ["text", "=", None, None, None], ["brk", " "],
        ["nest", 4, rhs]]]]]
    eqns.append(eqn)
  body = []
  for i, e in enumerate(eqns):
    body.append(["brk", " "] if i == 0 else ["brk", "; "])
    body.append(e)
  return ["group", ["concat", [
      ["text", "{ ", None, None, None],
      ["color", ["text", "lambda ", None, None, None], "BLUE", None, "BRIGHT"],
      ["text", "; ", None, None, None],
      ["text", "a:f32[]", None, None, None],
      ["text", ". ", None, None, None],
      ["color", ["text", "let", None, None, None], "BLUE", None, "BRIGHT"],
      ["nest", 2, ["concat", body]],
      ["brk", " "],
      ["color", ["text", "in ", None, None, None], "BLUE", None, "BRIGHT"],
      ["text", "(v0,) }", None, None, None]]]]


def make_pool():
  # Fresh, distinct objects (identity matters to the source map).
  return [tuple(["s", i]) for i in range(POOL_SIZE)]


def build(pp, spec, pool):
  k = spec[0]
  if k == "nil":
    return pp.nil()
  if k == "text":
    return pp.text(spec[1], spec[2], spec[3], spec[4])
  if k == "brk":
    return pp.brk(spec[1])
  if k == "concat":
    return pp.concat([build(pp, c, pool) for c in spec[1]])
  if k == "group":
    return pp.group(build(pp, spec[1], pool))
  if k == "nest":
    return pp.nest(spec[1], build(pp, spec[2], pool))
  if k == "color":
    fg = None if spec[2] is None else pp.Color[spec[2]]
    bg = None if spec[3] is None else pp.Color[spec[3]]
    it = None if spec[4] is None else pp.Intensity[spec[4]]
    return pp.color(build(pp, spec[1], pool), fg, bg, it)
  if k == "src":
    return pp.source_map(build(pp, spec[2], pool), pool[spec[1]])
  raise ValueError(k)


def render(pp, spec, opts):
  """Build and format `spec`; returns (text, source_map or None)."""
  pool = make_pool()
  doc = build(pp, spec, pool)
  sm = [] if opts["source_map"] else None
  out = doc._format(
      opts["width"], use_color=opts["use_color"],
      output_format=pp.OutputFormat[opts["output_format"]],
      separable_lines=opts["separable_lines"],
      annotation_prefix=opts["annotation_prefix"], source_map=sm)
  if sm is not None:
    def idx(src):
      for i, p in enumerate(pool):
        if p is src:
          return i
      return repr(src)
    sm = [[[a, b, idx(s)] for a, b, s in line] for line in sm]
  return out, sm


def doc_repr(pp, spec):
  return repr(build(pp, spec, make_pool()))


def make_cases():
  """The deterministic list of (spec, [opts, ...]) groups."""
  rng = random.Random(20251116)
  cases = []
  def opts(width, use_color=False, fmt="TEXT", sep=False, prefix=" # ",
           sm=False):
    return dict(width=width, use_color=use_color, output_format=fmt,
                separable_lines=sep, annotation_prefix=prefix, source_map=sm)

  # Random docs, each under several random configurations.
  for _ in range(300):
    spec = random_spec(rng)
    group = []
    cases.append((spec, group))
    for _ in range(6):
      group.append(opts(
          rng.randrange(0, 121), use_color=rng.random() < 0.5,
          fmt=rng.choice(["TEXT", "HTML"]), sep=rng.random() < 0.3,
          prefix=rng.choice([" # ", "  -- ", ""]), sm=rng.random() < 0.5))
  # A few docs at every width 0..120 (layout fitting boundaries).
  for seed in range(4):
    spec = random_spec(random.Random(seed), max_depth=6)
    cases.append((spec, [opts(w) for w in range(0, 121)]))
  for n, seed in [(3, 0), (8, 1), (20, 2)]:
    spec = jaxpr_like_spec(n, seed)
    group = []
    cases.append((spec, group))
    for w in list(range(0, 121, 3)) + [200]:
      group.append(opts(w))
      group.append(opts(w, use_color=True, sm=True))
      group.append(opts(w, use_color=True, fmt="HTML", sep=True, sm=True))
  return cases
