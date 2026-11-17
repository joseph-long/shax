"""Shared case definitions for the pytree differential tests.

`make_cases(P)` takes a pytree module (real ``jaxlib._jax.pytree`` or shax's
``shax/jax/_src/lib/pytree.py``) and returns a list of ``(name, thunk)``
pairs. `run_case(thunk)` runs one and returns a normalized string describing
the result, the exception (type + message + cause type) and any warnings, so
the two implementations can be compared textually.

Used by gen_pytree_fixture.py (under the real-JAX env) and test_pytree.py.
"""
import collections
import dataclasses
import pickle
import re
import sys
import warnings

THIS = sys.modules[__name__]


# ---------------------------------------------------------------------------
# Node types used in the cases. Reprs are deterministic.
# ---------------------------------------------------------------------------

NT = collections.namedtuple("NT", "a b")
NT2 = collections.namedtuple("NT2", "a b")
NT3 = collections.namedtuple("NT", "a b c")
Empty = collections.namedtuple("Empty", ())


class NTSub(NT):
  pass


class FakeNT(tuple):
  _fields = ("a", "b")


class Obj:
  """Custom node: children are `c`, aux data is `aux`."""

  def __init__(self, *c, aux="aux"):
    self.c = c
    self.aux = aux

  def __repr__(self):
    return f"Obj{self.c}[{self.aux!r}]"

  def __eq__(self, other):
    return (type(self) is type(other) and self.c == other.c
            and self.aux == other.aux)

  __hash__ = None


class ObjK(Obj):
  """Custom node registered with keys."""


class ObjOther(Obj):
  """Registered with the same functions as Obj but a distinct type."""


class Unregistered(Obj):
  pass


def _obj_flatten(o):
  return (list(o.c), o.aux)


def _obj_unflatten(aux, c):
  return Obj(*c, aux=aux)


def _objk_flatten(o):
  return (o.c, o.aux)


def _objk_unflatten(aux, c):
  return ObjK(*c, aux=aux)


def _objk_flatten_keys(o):
  return ([(f"k{i}", c) for i, c in enumerate(o.c)], o.aux)


@dataclasses.dataclass
class DC:
  x: object
  y: object
  m: object = "meta"


@dataclasses.dataclass
class DC2:
  x: object
  y: object
  m: object = "meta"


@dataclasses.dataclass
class DCMeta:
  m: object


class Bad:
  """Custom nodes whose flatten functions misbehave (selected by `mode`)."""

  def __init__(self, mode):
    self.mode = mode

  def __repr__(self):
    return f"Bad({self.mode!r})"


_BAD_FLATTEN = {
    "str": lambda: "hello",
    "triple": lambda: (1, 2, 3),
    "noniter": lambda: (7, 7),
    "list": lambda: [[1, 2], 7],
    "ok": lambda: ([1, 2], 7),
}

_BAD_KEYS = {
    "str": lambda: "hello",
    "triple": lambda: (1, 2, 3),
    "noniter": lambda: (7, 7),
    "list": lambda: [[("a", 1)], 7],
    "notpair": lambda: ([1], 7),
    "listpair": lambda: ([["a", 1]], 7),
    "triplepair": lambda: ([("a", 1, 2)], 7),
    "ntpair": lambda: ([NT("a", 1)], 7),
    "gen": lambda: ((p for p in [("a", 1), ("b", 2)]), 7),
    "ok": lambda: ([("a", 1), ("b", 2)], 7),
}


class BadK(Bad):
  pass


class RaisingEq:
  def __init__(self, v):
    self.v = v

  def __eq__(self, other):
    raise RuntimeError("eq boom")

  def __ne__(self, other):
    raise RuntimeError("ne boom")

  def __hash__(self):
    return 0

  def __repr__(self):
    return "RaisingEq"


class AmbiguousBool:
  """Like a numpy array: comparisons return something with no truth value."""

  def __ne__(self, other):
    return self

  def __eq__(self, other):
    return self

  def __bool__(self):
    raise ValueError("The truth value is ambiguous")

  def __hash__(self):
    return 0

  def __repr__(self):
    return "AmbiguousBool"


class BadLt:
  def __lt__(self, other):
    raise RuntimeError("lt boom")

  def __hash__(self):
    return 1

  def __repr__(self):
    return "BadLt"


class Box:
  def __init__(self, x):
    self.x = x

  def __repr__(self):
    return f"Box({self.x!r})"


class MyDict(dict):
  pass


class MyList(list):
  pass


class MyTuple(tuple):
  pass


def _register_all(R):
  R.register_node(Obj, _obj_flatten, _obj_unflatten)
  R.register_node(ObjOther, _obj_flatten, _obj_unflatten)
  R.register_node(ObjK, _objk_flatten, _objk_unflatten, _objk_flatten_keys)
  R.register_node(Bad, lambda o: _BAD_FLATTEN[o.mode](), lambda a, c: None)
  R.register_node(BadK, lambda o: ((), 0), lambda a, c: None,
                  lambda o: _BAD_KEYS[o.mode]())
  R.register_node(Box, lambda o: ((), o.x), lambda a, c: Box(a))
  R.register_node(NT2, lambda o: ((o.a,), o.b), lambda b, a: NT2(a[0], b))
  R.register_dataclass_node(DC, ["x", "y"], ["m"])
  R.register_dataclass_node(DC2, ("x", "y"), ("m",))
  R.register_dataclass_node(DCMeta, [], ["m"])


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------

_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")
_MODS = re.compile(r"jaxlib\._jax\.pytree|_shax_pytree|shax\.jax\._src\.lib\.pytree")


def normalize(s):
  return _MODS.sub("<pytree>", _ADDR.sub(" at 0x?", s))


def run_case(thunk):
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
      out = "OK " + repr(thunk())
    except Exception as e:  # pylint: disable=broad-except
      cause = e.__cause__
      out = f"EXC {type(e).__name__}: {e}"
      if cause is not None:
        out += f" | cause={type(cause).__name__}: {cause}"
  for wi in w:
    out += f" | WARN {wi.category.__name__}: {wi.message}"
  return normalize(out)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _trees():
  return [
      ("none", None),
      ("leaf_int", 1),
      ("leaf_str", "foo"),
      ("leaf_obj", Unregistered(1)),
      ("empty_tuple", ()),
      ("empty_list", []),
      ("empty_dict", {}),
      ("tuple1", (1,)),
      ("tuple_none", (None,)),
      ("list_tuple", [()]),
      ("tuple2", (1, 2)),
      ("nested", ((1, "foo"), ["bar", (3, None, 7)])),
      ("dict", {"b": 1, "a": [2, None, (3,)], "c": NT(4, 5)}),
      ("dict_ab", {"a": 1, "b": 2}),
      ("dict_ac", {"a": 1, "c": 2}),
      ("dict_int_keys", {3: "x", 1: "y", 2: [1]}),
      ("dict_mixed_num", {1.5: 1, 2: 3, True: 4}),
      ("dict_tuple_keys", {(1, "b"): 1, (1, "a"): 2}),
      ("dict_strange_keys", {"a\n": None, "b'": (), "é": 1}),
      ("nested_dict", {"z": {"b": 2, "a": 1}, "y": [{"q": 1}]}),
      ("nt", NT(1, 2)),
      ("nt_nested", NT((1, NT(2, None)), {"baz": 3})),
      ("nt2_custom", NT2(1, "hi")),
      ("nt3", NT3(1, 2, 3)),
      ("nt_empty", Empty()),
      ("nt_sub", NTSub(1, 2)),
      ("fake_nt", FakeNT((1, 2))),
      ("obj", Obj(1, [2, 3], aux="zz")),
      ("obj_none_aux", Obj(1, aux=None)),
      ("obj_list_aux", Obj(4, aux=[4, "foo"])),
      ("obj_other", ObjOther(1, [2, 3], aux="zz")),
      ("objk", ObjK(1, [2, 3], aux="q")),
      ("objk_nested", [ObjK(Obj(1, aux=2), {"x": 3})]),
      ("box", Box("a'b")),
      ("box_type", Box(NT)),
      ("box_tuple", Box((2, 3))),
      ("dc", DC([1, 2], 3, "m")),
      ("dc_nested", DC(DC(1, 2), (3,), ("t",))),
      ("dc2", DC2([1, 2], 3, "m")),
      ("dc_meta", DCMeta(5)),
      ("mydict", MyDict(a=1)),
      ("mylist", MyList([1])),
      ("mytuple", MyTuple((1,))),
      ("ordereddict", collections.OrderedDict(b=1, a=2)),
      ("defaultdict", collections.defaultdict(int, b=1)),
      ("deep", [[[[[[1]]]]], ((((2,),),),)]),
      ("wide", list(range(30))),
  ]


def make_cases(P):
  """Returns a list of (name, thunk)."""
  R = P.PyTreeRegistry()
  _register_all(R)
  R.__module__ = __name__
  R.__name__ = "_REG"
  THIS._REG = R
  R2 = P.PyTreeRegistry()
  _register_all(R2)
  RN = P.PyTreeRegistry(enable_none=False)
  RN.register_node(type(None), lambda o: ((), None), lambda a, c: None)
  td = lambda x, r=R: r.flatten(x)[1]
  leaf = td(1)

  cases = []

  def add(name, fn):
    cases.append((name, fn))

  trees = _trees()

  # --- per-tree operations --------------------------------------------------
  for name, tree in trees:
    add(f"flatten/{name}", lambda t=tree: R.flatten(t))
    add(f"flatten_with_path/{name}", lambda t=tree: R.flatten_with_path(t))
    add(f"str/{name}", lambda t=tree: str(td(t)))
    add(f"counts/{name}", lambda t=tree: (td(t).num_leaves, td(t).num_nodes))
    add(f"node_data/{name}", lambda t=tree: td(t).node_data())
    add(f"children/{name}", lambda t=tree: td(t).children())
    add(f"roundtrip/{name}",
        lambda t=tree: td(t).unflatten(R.flatten(t)[0]))
    add(f"flatten_up_to_self/{name}", lambda t=tree: td(t).flatten_up_to(t))
    add(f"one_level/{name}", lambda t=tree: R.flatten_one_level(t))
    add(f"one_level_keys/{name}",
        lambda t=tree: R.flatten_one_level_with_keys(t))
    add(f"is_node/{name}", lambda t=tree: R.is_node(type(t)))
    add(f"all_leaves/{name}", lambda t=tree: P.all_leaves(R, [t]))
    add(f"serialize/{name}", lambda t=tree: td(t).serialize_using_proto().hex())
    add(f"proto_roundtrip/{name}",
        lambda t=tree: str(P.PyTreeDef.deserialize_using_proto(
            R, td(t).serialize_using_proto())))
    add(f"pickle_eq/{name}",
        lambda t=tree: pickle.loads(pickle.dumps(td(t))) == td(t))
    add(f"pickle_str/{name}",
        lambda t=tree: str(pickle.loads(pickle.dumps(td(t)))))
    add(f"getstate/{name}", lambda t=tree: td(t).__getstate__()[1])
    add(f"walk/{name}", lambda t=tree: td(t).walk(
        lambda c, d: ("N", c, d), lambda x: ("L", x), R.flatten(t)[0]))
    add(f"compose_leaf/{name}", lambda t=tree: str(leaf.compose(td(t))))
    add(f"compose/{name}",
        lambda t=tree: str(td(t).compose(td((1, [2])))))
    add(f"compose_counts/{name}", lambda t=tree: (
        td(t).compose(td((1, [2]))).num_leaves,
        td(t).compose(td((1, [2]))).num_nodes,
        [c.num_leaves for c in td(t).compose(td((1, [2]))).children()]))
    add(f"from_children/{name}", lambda t=tree: str(
        P.PyTreeDef.from_node_data_and_children(
            R, td(t).node_data(), td(t).children())))
    add(f"from_children_eq/{name}", lambda t=tree: (
        P.PyTreeDef.from_node_data_and_children(
            R, td(t).node_data(), td(t).children()) == td(t)))
    add(f"treedef_tuple/{name}",
        lambda t=tree: str(P.treedef_tuple(R, [td(t), leaf, td(t)])))
    add(f"other_registry_eq/{name}", lambda t=tree: td(t) == td(t, R2))
    add(f"none_leaf_registry/{name}", lambda t=tree: RN.flatten(t))
    add(f"hash_consistent/{name}",
        lambda t=tree: hash(td(t)) == hash(td(t, R2)) if td(t) == td(t, R2)
        else "neq")
    add(f"from_iterable_tree/{name}", lambda t=tree: td(t).from_iterable_tree(
        td(t).walk(lambda c, d: c, lambda x: x, R.flatten(t)[0])))
    add(f"is_leaf_lists/{name}",
        lambda t=tree: R.flatten(t, lambda x: isinstance(x, list)))
    add(f"is_leaf_none/{name}", lambda t=tree: R.flatten(t, lambda x: None))
    add(f"is_leaf_path/{name}", lambda t=tree: R.flatten_with_path(
        t, lambda kp, x: len(kp) >= 2))

  # --- pairwise flatten_up_to and equality ----------------------------------
  for n1, t1 in trees:
    for n2, t2 in trees:
      add(f"up_to/{n1}/{n2}", lambda a=t1, b=t2: td(a).flatten_up_to(b))
      add(f"eq/{n1}/{n2}", lambda a=t1, b=t2: (td(a) == td(b), td(a) != td(b)))

  # --- specific flatten_up_to errors ---------------------------------------
  pairs = [
      ([1, 2], [7]), ((1,), (7, 8)), ({"a": 1}, {"a": 7, "b": 8}),
      ({"a": 1}, {"b": 7}), ([1], {"a": 7}), ([1], (7,)), ((1,), [7]),
      ((1,), {"b": 7}), ([NT(1, 2)], [(1, 2)]), ([NT(1, 2)], [NT2(1, 2)]),
      ([Obj(1)], [(1,)]), (None, [2]), (NT(1, 2), FakeNT((1, 2))),
      (Obj(1, 2), Obj(1, 2, 3)), (Obj(1, 2), Obj(1, 2, aux="other")),
      (Obj(1, 2), ObjOther(1, 2)), (DC(1, 2), DC(1, 2, "n")),
      (DC(1, 2), DC2(1, 2)), (DC(1, [2]), DC(1, 2)), ((1, [2]), ([1], (2,))),
      ({"a": 1}, {"a": 1, 2: 3}), ({1: 1, 2: 2}, {1.0: 3, 2: 4}),
      ({"a": 1}, collections.OrderedDict(a=1)), ([1, 2], MyList([1, 2])),
      ((1, 2), NT(1, 2)), (Box(RaisingEq(1)), Box(RaisingEq(2))),
      (Box(1), Box(AmbiguousBool())),
  ]
  for i, (a, b) in enumerate(pairs):
    add(f"up_to_err/{i}", lambda a=a, b=b: td(a).flatten_up_to(b))

  # --- equality metadata errors --------------------------------------------
  add("eq_raise", lambda: td(Box(RaisingEq(1))) == td(Box(RaisingEq(2))))
  add("eq_raise_nested",
      lambda: td([1, Box(RaisingEq(1))]) == td([1, Box(RaisingEq(2))]))
  add("eq_ambiguous", lambda: td(Box(AmbiguousBool())) == td(Box(1)))
  add("eq_dc_ambiguous",
      lambda: td(DC(1, 2, AmbiguousBool())) == td(DC(1, 2, 1)))
  add("eq_nt_vs_tuple", lambda: td(NT(1, 2)) == td((1, 2)))
  add("eq_dict_keys", lambda: td({"a": 1}) == td({"b": 1}))
  add("eq_int_float_keys", lambda: td({1: 1}) == td({1.0: 1}))
  add("eq_non_treedef", lambda: (td(1) == 5, td(1) != 5, td(1) == None))  # noqa: E711
  add("eq_none_leaf_reg", lambda: RN.flatten(None)[1] == R.flatten(None)[1])
  add("eq_dc_other_type", lambda: td(DC(1, 2)) == td(DC2(1, 2)))
  add("eq_obj_other_type", lambda: td(Obj(1)) == td(ObjOther(1)))
  add("hash_box_unhashable", lambda: isinstance(hash(td(Box([1]))), int))

  # --- dict key sorting ------------------------------------------------------
  add("unsortable", lambda: R.flatten({"b": 1, 2: 3}))
  add("unsortable_nested", lambda: R.flatten([{"a": 1, 2: 3}]))
  add("bad_lt", lambda: R.flatten({BadLt(): 1, 2: 2}))
  add("unsortable_one_level", lambda: R.flatten_one_level({"a": 1, 2: 3}))
  add("dict_unflatten_order",
      lambda: list(td({"b": 1, "a": 2}).unflatten([5, 6]).keys()))

  # --- leaf predicate --------------------------------------------------------
  for name, ret in [("int", 1), ("zero", 0), ("list", []), ("str", "x"),
                    ("np_like", collections.OrderedDict())]:
    add(f"pred_bad/{name}", lambda r=ret: R.flatten([1, [2]], lambda x: r))
  add("pred_true_root", lambda: R.flatten([1], lambda x: True))
  add("pred_path_true_root", lambda: R.flatten_with_path(5, lambda kp, x: True))
  add("pred_raises", lambda: R.flatten([1], lambda x: 1 / 0))
  add("pred_not_callable", lambda: R.flatten([1], 5))
  add("pred_one_arg_with_path", lambda: R.flatten_with_path([1], lambda x: 1))
  add("pred_sees_keys", lambda: (lambda seen: (R.flatten_with_path(
      {"a": [ObjK(1, 2), Obj(3)], "b": DC(1, 2)},
      lambda kp, x: seen.append(kp)), seen))([]))
  add("pred_sees_nodes", lambda: (lambda seen: (R.flatten(
      {"a": [ObjK(1, 2), Obj(3)], "b": DC(1, 2)},
      lambda x: seen.append(repr(x))), seen))([]))

  # --- bad custom flatten functions ----------------------------------------
  for mode in _BAD_FLATTEN:
    add(f"bad_flatten/{mode}", lambda m=mode: R.flatten(Bad(m)))
    add(f"bad_flatten_path/{mode}", lambda m=mode: R.flatten_with_path(Bad(m)))
    add(f"bad_one_level/{mode}", lambda m=mode: R.flatten_one_level(Bad(m)))
    add(f"bad_one_level_keys/{mode}",
        lambda m=mode: R.flatten_one_level_with_keys(Bad(m)))
  for mode in _BAD_KEYS:
    add(f"badk_flatten/{mode}", lambda m=mode: R.flatten(BadK(m)))
    add(f"badk_flatten_path/{mode}",
        lambda m=mode: R.flatten_with_path(BadK(m)))
    add(f"badk_one_level_keys/{mode}",
        lambda m=mode: R.flatten_one_level_with_keys(BadK(m)))

  # --- iterator leaves warn --------------------------------------------------
  iters = {
      "generator": lambda: (x for x in [1]),
      "zip": lambda: zip([1], [2]),
      "list_iter": lambda: iter([1]),
      "tuple_iter": lambda: iter((1,)),
      "set_iter": lambda: iter({1}),
      "map": lambda: map(int, [1]),
      "filter": lambda: filter(None, [1]),
      "enumerate": lambda: enumerate([1]),
      "reversed": lambda: reversed([1]),
      "dict_keys_iter": lambda: iter({1: 2}),
      "dict_values_iter": lambda: iter({1: 2}.values()),
      "dict_items_iter": lambda: iter({1: 2}.items()),
      "dict_values": lambda: {1: 2}.values(),
      "dict_keys": lambda: {1: 2}.keys(),
      "range": lambda: range(3),
      "range_iter": lambda: iter(range(3)),
  }
  for name, mk in iters.items():
    add(f"iter_leaf/{name}", lambda mk=mk: type(R.flatten([mk()])[0][0]).__name__)
    add(f"iter_leaf_pred/{name}",
        lambda mk=mk: len(R.flatten([mk()], lambda x: x is not None and
                                    not isinstance(x, list))[0]))

  # --- unflatten / walk ------------------------------------------------------
  t2 = td([1, 2])
  add("unflatten_too_few", lambda: t2.unflatten([1]))
  add("unflatten_too_many", lambda: t2.unflatten([1, 2, 3]))
  add("unflatten_not_iterable", lambda: t2.unflatten(5))
  add("unflatten_none", lambda: t2.unflatten(None))
  add("unflatten_dict", lambda: t2.unflatten({1: 2, 3: 4}))
  add("unflatten_str", lambda: t2.unflatten("ab"))
  add("unflatten_gen_consumption", lambda: (lambda g: (t2.unflatten(g), next(g)))(
      iter(range(10))))
  add("unflatten_leaf_extra", lambda: td(None).unflatten([1]))
  add("unflatten_obj_children_type", lambda: td(Obj(1, 2)).unflatten([[1], 2]))
  add("unflatten_dc", lambda: td(DC(1, 2)).unflatten([3, 4]))
  add("walk_too_few", lambda: t2.walk(lambda a, b: a, lambda x: x, [1]))
  add("walk_too_many", lambda: t2.walk(lambda a, b: a, lambda x: x, [1, 2, 3]))
  add("walk_f_leaf_none", lambda: t2.walk(lambda a, b: a, None, [1, 2]))
  add("walk_not_callable", lambda: t2.walk(5, len, [1, 2]))
  add("walk_gen_consumption", lambda: (lambda g: (
      t2.walk(lambda a, b: a, lambda x: x, g), next(g)))(iter(range(10))))
  add("walk_dict_data_is_copy", lambda: (lambda t: (
      t.walk(lambda c, d: d.append("X") or d, lambda x: x, [1]), str(t)))(
          td({"a": 1})))

  # --- from_iterable_tree ---------------------------------------------------
  add("fit_leaf", lambda: td(1).from_iterable_tree(5))
  add("fit_none_child", lambda: td([1, (2, None)]).from_iterable_tree(
      [3, [4, []]]))
  add("fit_none_child_bad", lambda: td([1, (2, None)]).from_iterable_tree(
      [3, [4, [1]]]))
  add("fit_not_iterable", lambda: td([1, None]).from_iterable_tree([1, 2]))
  add("fit_arity", lambda: td([1, 2]).from_iterable_tree([3]))
  add("fit_dict", lambda: td({"a": 1, "b": 2}).from_iterable_tree([3, 4]))
  add("fit_custom", lambda: td(Obj(1, 2)).from_iterable_tree([3, 4]))

  # --- compose / treedef_tuple / from_node_data errors -----------------------
  add("compose_other_registry", lambda: leaf.compose(R2.flatten(1)[1]))
  add("compose_not_treedef", lambda: leaf.compose(5))
  add("treedef_tuple_empty", lambda: str(P.treedef_tuple(R, [])))
  add("treedef_tuple_other_registry",
      lambda: P.treedef_tuple(R2, [R.flatten(1)[1]]))
  add("treedef_tuple_tuple_arg", lambda: P.treedef_tuple(R, (leaf,)))
  add("treedef_tuple_bad_elem", lambda: P.treedef_tuple(R, [1]))
  add("treedef_tuple_eq", lambda: P.treedef_tuple(R, [leaf]) == td((1,)))
  add("treedef_tuple_counts", lambda: (lambda t: (t.num_leaves, t.num_nodes))(
      P.treedef_tuple(R, [td([1, 2]), td(None)])))
  fnd = P.PyTreeDef.from_node_data_and_children
  for name, nd, ch in [
      ("dict", (dict, ["a", "b"]), [leaf, leaf]),
      ("dict_tuple_keys", (dict, ("a",)), [leaf]),
      ("dict_mismatch", (dict, ["a", "b"]), [leaf]),
      ("dict_none_keys", (dict, None), []),
      ("dict_str_keys", (dict, "ab"), [leaf, leaf]),
      ("leaf", None, []),
      ("leaf_children", None, [leaf, leaf]),
      ("unknown_type", (int, None), [leaf]),
      ("not_type", (5, None), []),
      ("list_node_data", [list, None], [leaf]),
      ("tuple1", (tuple,), []),
      ("int", 5, []),
      ("children_not_treedef", (list, None), [1]),
      ("children_not_iterable", (list, None), 5),
      ("children_iter", (list, None), iter([leaf])),
      ("nt", (NT, None), [leaf, leaf]),
      ("nt_data_ignored", (NT, 5), [leaf, leaf]),
      ("nt2_registered_custom", (NT2, "b"), [leaf]),
      ("tuple_data_ignored", (tuple, 5), [leaf, leaf]),
      ("custom", (Obj, "zz"), [leaf, leaf]),
      ("custom_other_registry_children", (Obj, "zz"), [R2.flatten(1)[1]]),
      ("dc", (DC, ("m",)), [leaf, leaf]),
      ("dc_wrong_arity", (DC, ("m",)), [leaf, leaf, leaf]),
      ("none", (type(None), None), []),
  ]:
    add(f"from_node_data/{name}",
        lambda nd=nd, ch=ch: str(fnd(R, nd, ch)))
  add("from_node_data/none_disabled",
      lambda: str(fnd(P.PyTreeRegistry(enable_none=False),
                      (type(None), None), [])))
  add("from_node_data/none_custom", lambda: str(fnd(RN, (type(None), None), [])))
  add("from_node_data/dict_mismatch_node_data",
      lambda: fnd(R, (dict, ["a", "b"]), [leaf]).node_data())
  add("from_node_data/kwargs", lambda: str(fnd(
      registry=R, node_data=(list, None), children=[leaf])))
  add("from_node_data/on_instance",
      lambda: str(leaf.from_node_data_and_children(R, (list, None), [])))

  # --- registry construction / registration ---------------------------------
  for kw in ["enable_none", "enable_tuple", "enable_namedtuple", "enable_list",
             "enable_dict"]:
    add(f"registry_disable/{kw}", lambda kw=kw: (lambda r: (
        r.flatten([None, (1,), NT(1, 2), {"a": 1}]),
        [r.is_node(t) for t in (type(None), tuple, NT, list, dict)]))(
            P.PyTreeRegistry(**{kw: False})))
  add("registry_positional", lambda: P.PyTreeRegistry(True, False).flatten((1,)))
  add("registry_int_flag", lambda: P.PyTreeRegistry(enable_none=1))
  add("register_dup", lambda: R.register_node(Obj, len, len))
  add("register_dup_builtin", lambda: R.register_node(tuple, len, len))
  add("register_dup_dc", lambda: R.register_dataclass_node(DC, ["x"], []))
  add("register_dup_dc_over_custom",
      lambda: R.register_dataclass_node(Obj, ["x"], []))
  add("register_not_callable", lambda: P.PyTreeRegistry().register_node(
      int, 5, 6))
  add("register_keys_not_callable", lambda: P.PyTreeRegistry().register_node(
      int, len, len, 5))
  add("register_dc_str_fields", lambda: P.PyTreeRegistry()
      .register_dataclass_node(int, "x", []))
  add("register_dc_int_fields", lambda: P.PyTreeRegistry()
      .register_dataclass_node(int, ["x"], [1]))
  add("register_tuple_when_disabled", lambda: (lambda r: (
      r.register_node(tuple, lambda o: (o, None), lambda a, c: tuple(c)),
      r.flatten((1, 2)), r.flatten(NT(1, 2))))(
          P.PyTreeRegistry(enable_tuple=False)))
  add("dc_missing_attr", lambda: (lambda r: (
      r.register_dataclass_node(Box, ["nope"], []), r.flatten(Box(1))))(
          P.PyTreeRegistry()))
  add("is_node_misc", lambda: [R.is_node(x) for x in (
      5, int, str, NT, NTSub, FakeNT, Obj, DC, MyDict, type(None), [1])])
  add("reduce_unnamed", lambda: P.PyTreeRegistry().__reduce__())
  add("reduce_named", lambda: R.__reduce__())
  add("default_registry_identity",
      lambda: P.default_registry() is P._default_registry)
  add("version", lambda: P.version)
  add("all_leaves_misc", lambda: [
      P.all_leaves(R, [1, 2]), P.all_leaves(R, [1, None]),
      P.all_leaves(R, iter([1, NT(1, 2)])), P.all_leaves(R, [Obj()]),
      P.all_leaves(R, []), P.all_leaves(R, "abc")])
  add("all_leaves_not_iterable", lambda: P.all_leaves(R, 5))

  # --- PyTreeDef misc ----------------------------------------------------------
  add("no_constructor", lambda: P.PyTreeDef())
  add("node_data_dict_is_copy", lambda: (lambda t: (
      t.node_data()[1].append("X"), str(t)))(td({"a": 1})))
  add("children_eq", lambda: td(((1, 2, 3), (4,))).children() == [
      td((0, 0, 0)), td((7,))])
  add("num_props", lambda: (type(td(1)).num_leaves.__class__.__name__,
                            type(td(1)).num_nodes.__class__.__name__))

  # --- pickling ----------------------------------------------------------------
  def mk(state):
    t = P.PyTreeDef.__new__(P.PyTreeDef)
    t.__setstate__(state)
    return t

  st = td([Obj(aux=1), 1]).__getstate__()
  add("setstate_ok", lambda: str(mk(st)))
  add("setstate_other_registry", lambda: str(mk((R2, st[1]))))
  add("setstate_3tuple", lambda: mk((R, st[1], 3)))
  add("setstate_bad_node", lambda: mk((R, [(0, 0, None, None, 1)])))
  add("setstate_leaf_data", lambda: mk((R, [(0, 0, 5, None, 1, 1)])))
  add("setstate_leaf_custom", lambda: mk((R, [(0, 0, None, Obj, 1, 1)])))
  add("setstate_dc_kind_for_custom", lambda: str(mk((R, [(7, 0, (), Obj, 0, 1)]))))
  add("setstate_counts_trusted", lambda: mk(
      (R, [(0, 0, None, None, 1, 1), (5, 1, ["a"], None, 7, 9)])).num_leaves)
  add("setstate_dict_arity_mismatch",
      lambda: str(mk((R, [(5, 1, ["a", "b"], None, 0, 1)]))))
  add("setstate_empty", lambda: (lambda t: (t.num_leaves, t.num_nodes))(
      mk((R, []))))
  add("setstate_empty_repr", lambda: repr(mk((R, []))))
  add("setstate_empty_node_data", lambda: mk((R, [])).node_data())
  add("setstate_empty_unflatten", lambda: mk((R, [])).unflatten([]))
  add("setstate_empty_children", lambda: mk((R, [])).children())
  add("setstate_partial", lambda: repr(mk((R, st[1][:2]))))
  for k in ["SequenceKey", "DictKey", "GetAttrKey", "FlattenedIndexKey"]:
    add(f"key_setstate_2tuple/{k}", lambda k=k: getattr(P, k).__new__(
        getattr(P, k)).__setstate__((1, 2)))

  # --- proto deserialization ------------------------------------------------
  protos = {
      "empty": "",
      "garbage": "ffff",
      "two_leaves": "0a021001" * 2,
      "type9": "0a021009",
      "type0": "0a021000",
      "leaf_arity1": "0a0408011001",
      "none_arity1": "0a0408011003",
      "dict_no_strings": "0a021001" + "0a09080110051a030a0100",
      "dict_key_oob": "0a021001" + "0a09080110051a030a0105" + "120161",
      "dict_no_keys": "0a021001" + "0a0408011005" + "120161",
      "dict_arity2_keys1": "0a021001" + "0a09080210051a030a0100" + "120161",
      "dict_dup_key": "0a021001" * 2 + "0a0a080210051a040a020000" + "120161",
      "dict_unsorted": "0a021001" * 2 + "0a0a080210051a040a020001" + "120162"
                       + "120161",
      "dict_unpacked_ids": "0a021001" * 2 + "0a0a080210051a04" + "0800" +
                           "0801" + "120161" + "120162",
      "tuple_nested": "0a021001" + "0a0408011004" * 2,
      "tuple_arity0": "0a0408001004",
      "tuple_alone": "0a0408011004",
      "leaf_then_none_arity1": "0a021001" + "0a0408011003",
      "truncated": "0a021001" + "0a0608011001",
      "leaf_leaf_arity1": "0a021001" + "0a0408011001",
      "none": "0a021003",
      "wire_mismatch": "0a0210031801",
      "malformed_late": "0a021001" * 2 + "0a0408031002",
      "too_many_roots": "0a021001" * 3 + "0a0408021002",
      "invalid_type_beats_arity": "0a021001" + "0a0408031002" + "0a021009",
      "unknown_top_field": "0a021001" + "2a00",
      "unknown_dictkeys_field": "0a021001" * 2 + "0a0408021002" + "1a00",
      "string_first": "12026162" + "0a021001",
      "bad_utf8": "1202ffff" + "0a021001",
      "big_arity": "0a021001" + "0a0808ffffffff0f1002",
      "spec_example": "0a0408011002",
  }
  for name, hx in protos.items():
    add(f"proto/{name}", lambda hx=hx: (lambda t: (
        t.num_leaves, t.num_nodes, run_case(lambda: repr(t))))(
            P.PyTreeDef.deserialize_using_proto(R, bytes.fromhex(hx))))
  add("proto/str_data", lambda: P.PyTreeDef.deserialize_using_proto(R, "abc"))
  add("proto/dict_disabled_registry", lambda: str(
      P.PyTreeDef.deserialize_using_proto(
          P.PyTreeRegistry(enable_dict=False), td({"a": 1})
          .serialize_using_proto())))
  add("proto/serialize_none_custom",
      lambda: RN.flatten(None)[1].serialize_using_proto())
  add("proto/serialize_int_keys", lambda: td({1: 2}).serialize_using_proto())
  add("proto/serialize_interning", lambda: td(
      {"b": 1, "a": 2, "c": {"b": 3, "d": 4}}).serialize_using_proto().hex())

  # --- keys ---------------------------------------------------------------------
  SK, DK, GK, FK = P.SequenceKey, P.DictKey, P.GetAttrKey, P.FlattenedIndexKey
  keys = [SK(1), SK(idx=0), SK(-3), DK("a"), DK(key=1), DK(None), DK((1, "x")),
          DK("it's"), GK("bar"), GK(name="it's"), FK(1), FK(key=-2)]
  for i, k in enumerate(keys):
    add(f"key_repr/{i}", lambda k=k: (repr(k), str(k)))
    add(f"key_pickle/{i}", lambda k=k: pickle.loads(pickle.dumps(k)) == k)
    add(f"key_getstate/{i}", lambda k=k: k.__getstate__())
    add(f"key_attrs/{i}", lambda k=k: [getattr(k, a) for a in k.__match_args__])
    add(f"key_hash_eq/{i}", lambda k=k: hash(k) == hash(pickle.loads(
        pickle.dumps(k))))
  add("key_hash_values", lambda: [hash(SK(1)), hash(SK(5)), hash(SK(-1)),
                                  hash(SK(-2)), hash(FK(1)), hash(FK(-1)),
                                  hash(GK("q")) == hash("q"),
                                  hash(DK((1, 2))) == hash((1, 2))])
  add("key_match_args", lambda: [SK.__match_args__, DK.__match_args__,
                                 GK.__match_args__, FK.__match_args__])
  add("key_eq_cross", lambda: [SK(1) == FK(1), DK("a") == GK("a"),
                               GK("a") == "a", SK(1) == 1, SK(1) == SK(True),
                               SK(1) != SK(2), DK([1]) == DK([1])])
  add("key_lt", lambda: SK(1) < SK(2))
  add("key_dict_unhashable", lambda: hash(DK([1])))
  add("key_dict_raising_eq", lambda: DK(RaisingEq(1)) == DK(RaisingEq(1)))
  add("key_ctor_errors", lambda: [run_case(f) for f in (
      lambda: SK("a"), lambda: SK(1.0), lambda: SK(2**31), lambda: SK(2**31 - 1),
      lambda: SK(-2**31), lambda: SK(True), lambda: GK(1), lambda: GK(None),
      lambda: FK("a"), lambda: DK(), lambda: SK())])
  add("key_set_attr", lambda: setattr(SK(1), "foo", 1))

  def match(k):
    match k:
      case P.SequenceKey(idx=idx):
        return ("seq", idx)
      case P.DictKey(key=key):
        return ("dict", key)
      case P.GetAttrKey(name=name):
        return ("attr", name)
      case P.FlattenedIndexKey(key=key):
        return ("flat", key)
    return None

  add("key_pattern_match", lambda: [match(k) for k in keys])
  def match_positional(k):
    match k:
      case P.SequenceKey(idx):
        return ("seq", idx)
      case P.DictKey(key):
        return ("dict", key)
      case P.GetAttrKey(name):
        return ("attr", name)
      case P.FlattenedIndexKey(key):
        return ("flat", key)
    return None

  add("key_pattern_positional", lambda: [match_positional(k) for k in keys])

  return cases
