"""Pure-Python stand-in for jaxlib's C++ pytree extension (``jaxlib._jax.pytree``).

This mirrors the behaviour of the nanobind/C++ implementation shipped with
jaxlib 0.12.0.1 as observed from Python: flattening order, treedef equality,
``repr`` strings, pickling, the protobuf serialization format, and the exact
exception types and messages that JAX's tests assert on.

Internally a ``PyTreeDef`` is a post-order list of nodes, exactly like the C++
``traversal_`` vector. Each node is a tuple
``(kind, arity, node_data, custom, num_leaves, num_nodes)`` where ``custom`` is
the ``_Registration`` for custom/dataclass nodes and ``None`` otherwise.
"""

from __future__ import annotations

import builtins
import warnings

version = 3

# Node kinds; the numeric values match the C++ ``PyTreeKind`` enum (they are
# visible through ``PyTreeDef.__getstate__``).
_LEAF = 0
_NONE = 1
_TUPLE = 2
_NAMEDTUPLE = 3
_LIST = 4
_DICT = 5
_CUSTOM = 6
_DATACLASS = 7
_FROZENDICT = 8

_DICT_KINDS = (_DICT, _FROZENDICT)
_KIND, _ARITY, _DATA, _CUSTOMREG, _NLEAVES, _NNODES = range(6)

_NoneType = type(None)
# Python >= 3.15 only.
_frozendict = getattr(builtins, "frozendict", None)

# Iterator types that are flattened as leaves with a DeprecationWarning
# (mirrors the PyGen_Check / PyXxx_Type checks in pytree.cc).
_WARN_ITER_TYPES = tuple({
    type((lambda: (yield))()),         # generator
    type(zip()), type(iter([])), type(iter(set())), type(map(len, ())),
    type(filter(None, ())), type(enumerate(())), type(reversed(())),
    type(iter(())), type(iter({})), type(iter({}.values())),
    type(iter({}.items())),
})
_DICT_VALUES_TYPE = type({}.values())


# ---------------------------------------------------------------------------
# nanobind-style argument errors
# ---------------------------------------------------------------------------

def _type_name(tp) -> str:
  mod = getattr(tp, "__module__", None)
  name = getattr(tp, "__qualname__", None) or tp.__name__
  if mod in (None, "builtins"):
    return name
  return f"{mod}.{name}"


def _incompatible(fname, sig, args, kwargs=None) -> TypeError:
  """Builds the TypeError nanobind raises when argument conversion fails."""
  parts = [_type_name(type(a)) for a in args]
  if kwargs:
    parts.append("kwargs = { " + ", ".join(
        f"{k}: {_type_name(type(v))}" for k, v in kwargs.items()) + " }")
  return TypeError(
      f"{fname}(): incompatible function arguments. The following argument "
      f"types are supported:\n    1. {sig}\n\n"
      f"Invoked with types: {', '.join(parts)}")


def _jax_runtime_error(msg):
  try:
    from shax.jax._src.lib import _jax  # pylint: disable=g-import-not-at-top
    cls = getattr(_jax, "JaxRuntimeError", None)
  except Exception:  # pragma: no cover - import cycles during bootstrap
    cls = None
  if cls is None:
    cls = JaxRuntimeError
  return cls(msg)


class JaxRuntimeError(RuntimeError):
  """Fallback for ``jaxlib._jax.JaxRuntimeError`` when ``_jax`` lacks it."""


_SENTINEL = object()


def _try_iter(x):
  try:
    return iter(x)
  except TypeError:
    return None


def _is_iterable(x) -> bool:
  try:
    iter(x)
  except TypeError:
    return False
  return True


def _is_int32(x) -> bool:
  if isinstance(x, float):
    return False
  try:
    v = x.__index__()
  except Exception:
    return False
  return -2**31 <= v < 2**31


# ---------------------------------------------------------------------------
# Key path entries
# ---------------------------------------------------------------------------

_SIG_SEQKEY = "__init__(self, idx: int) -> None"
_SIG_DICTKEY = "__init__(self, key: object | None) -> None"
_SIG_ATTRKEY = "__init__(self, name: str) -> None"
_SIG_FLATKEY = "__init__(self, key: int) -> None"
_SIG_SETSTATE = "__setstate__(self, arg: tuple, /) -> None"


class SequenceKey:
  __slots__ = ("_idx",)
  __match_args__ = ("idx",)

  def __init__(self, idx: int) -> None:
    if not _is_int32(idx):
      raise _incompatible("__init__", _SIG_SEQKEY, (self, idx))
    self._idx = idx.__index__()

  @property
  def idx(self) -> int:
    return self._idx

  def __str__(self):
    return f"[{self._idx}]"

  def __repr__(self):
    return f"SequenceKey(idx={self._idx})"

  def __eq__(self, other):
    return isinstance(other, SequenceKey) and self._idx == other._idx

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self._idx + 1)

  def __getstate__(self):
    return (self._idx,)

  def __setstate__(self, state):
    if not isinstance(state, tuple):
      raise _incompatible("__setstate__", _SIG_SETSTATE, (self, state))
    if len(state) != 1:
      raise _jax_runtime_error(
          "Malformed pickled SequenceKey, expected 1-tuple")
    if not _is_int32(state[0]):
      raise RuntimeError("std::bad_cast")
    self._idx = state[0].__index__()


class DictKey:
  __slots__ = ("_key",)
  __match_args__ = ("key",)

  def __init__(self, key: object) -> None:
    self._key = key

  @property
  def key(self):
    return self._key

  def __str__(self):
    return f"[{self._key!r}]"

  def __repr__(self):
    return f"DictKey(key={self._key!r})"

  def __eq__(self, other):
    return isinstance(other, DictKey) and bool(self._key == other._key)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self._key)

  def __getstate__(self):
    return (self._key,)

  def __setstate__(self, state):
    if not isinstance(state, tuple):
      raise _incompatible("__setstate__", _SIG_SETSTATE, (self, state))
    if len(state) != 1:
      raise _jax_runtime_error("Malformed pickled DictKey, expected 1-tuple")
    self._key = state[0]


class GetAttrKey:
  __slots__ = ("_name",)
  __match_args__ = ("name",)

  def __init__(self, name: str) -> None:
    if not isinstance(name, str):
      raise _incompatible("__init__", _SIG_ATTRKEY, (self, name))
    self._name = str(name)

  @property
  def name(self) -> str:
    return self._name

  def __str__(self):
    return f".{self._name}"

  def __repr__(self):
    return f"GetAttrKey(name='{self._name}')"

  def __eq__(self, other):
    return isinstance(other, GetAttrKey) and self._name == other._name

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self._name)

  def __getstate__(self):
    return (self._name,)

  def __setstate__(self, state):
    if not isinstance(state, tuple):
      raise _incompatible("__setstate__", _SIG_SETSTATE, (self, state))
    if len(state) != 1:
      raise _jax_runtime_error(
          "Malformed pickled GetAttrKey, expected 1-tuple")
    self._name = str(state[0])


class FlattenedIndexKey:
  __slots__ = ("_key",)
  __match_args__ = ("key",)

  def __init__(self, key: int) -> None:
    if not _is_int32(key):
      raise _incompatible("__init__", _SIG_FLATKEY, (self, key))
    self._key = key.__index__()

  @property
  def key(self) -> int:
    return self._key

  def __str__(self):
    return f"[<flat index {self._key}>]"

  def __repr__(self):
    return f"FlattenedIndexKey(key={self._key})"

  def __eq__(self, other):
    return isinstance(other, FlattenedIndexKey) and self._key == other._key

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self._key + 42)

  def __getstate__(self):
    return (self._key,)

  def __setstate__(self, state):
    if not isinstance(state, tuple):
      raise _incompatible("__setstate__", _SIG_SETSTATE, (self, state))
    if len(state) != 1:
      raise _jax_runtime_error(
          "Malformed pickled FlattenedIndexKey, expected 1-tuple")
    if not _is_int32(state[0]):
      raise RuntimeError("std::bad_cast")
    self._key = state[0].__index__()


SequenceKey.__init__.__doc__ = _SIG_SEQKEY
DictKey.__init__.__doc__ = _SIG_DICTKEY
GetAttrKey.__init__.__doc__ = _SIG_ATTRKEY
FlattenedIndexKey.__init__.__doc__ = _SIG_FLATKEY


# ---------------------------------------------------------------------------
# Helpers shared by flatten / flatten_up_to / flatten_one_level
# ---------------------------------------------------------------------------

def _sorted_dict_keys(d) -> list:
  keys = list(d.keys())
  try:
    keys.sort()
  except Exception as e:
    raise ValueError(
        "Comparator raised exception while sorting pytree dictionary keys."
    ) from e
  return keys


def _call_to_iterable(reg, obj):
  out = reg.to_iterable(obj)
  if not isinstance(out, tuple) or len(out) != 2:
    raise ValueError(
        "The to_iterable function for a custom PyTree node should return a "
        f"(children, aux_data) tuple, got {out!r}")
  it = _try_iter(out[0])
  if it is None:
    raise ValueError(
        "The to_iterable function for a custom PyTree node should return a "
        "(children, aux_data) tuple where 'children' is iterable, got "
        f"{out!r}")
  return out, it


def _call_to_iterable_with_keys(reg, obj):
  """Returns (list of (key, child) pairs, aux_data)."""
  out = reg.to_iterable_with_keys(obj)
  if not isinstance(out, tuple) or len(out) != 2:
    raise ValueError(
        "The to_iterable_with_keys function for a custom PyTree node should "
        f"return a (key_leaf_pairs, aux_data) tuple, got {out!r}")
  it = _try_iter(out[0])
  if it is None:
    raise ValueError(
        "The to_iterable_with_keys function for a custom PyTree node should "
        "return a (key_leaf_pairs, aux_data) tuple where 'key_leaf_pairs' is "
        f"iterable, got {out!r}")
  pairs = []
  for child in it:
    if not isinstance(child, tuple) or len(child) != 2:
      # (sic) The C++ message really is truncated like this.
      raise ValueError(
          "The to_iterable_with_keys function for a custom PyTree node should "
          "return a (key_leaf_pairs, aux_data) tuple where 'child"
          f"{child!r}")
    pairs.append((child[0], child[1]))
  return pairs, out[1]


def _node_type(node):
  kind = node[_KIND]
  if kind in (_CUSTOM, _DATACLASS):
    return node[_CUSTOMREG].type
  return {_NONE: _NoneType, _TUPLE: tuple, _LIST: list, _DICT: dict,
          _NAMEDTUPLE: node[_DATA], _LEAF: None}.get(kind)


def _set_num_leaves_and_nodes(nodes):
  """Port of PyTreeDef::SetNumLeavesAndNumNodes: recomputes the counts."""
  out = []
  starts = []  # (num_leaves, index) at the start of each pending subtree
  num_leaves = 0
  for i, node in enumerate(nodes):
    kind, arity = node[_KIND], node[_ARITY]
    start = (num_leaves, i)
    if kind == _LEAF:
      num_leaves += 1
    if arity == 0:
      starts.append(start)
    else:
      if arity < 0:
        raise ValueError(
            f"Malformed PyTreeDef: node {i} has negative arity ({arity}).")
      if arity > len(starts):
        raise ValueError(
            f"Malformed PyTreeDef: node {i} has arity {arity}, exceeding the "
            f"number of subtrees preceding it ({len(starts)}).")
      del starts[len(starts) - (arity - 1):]
    out.append((kind, arity, node[_DATA], node[_CUSTOMREG],
                num_leaves - starts[-1][0], i + 1 - starts[-1][1]))
  return out


def _keys_equal(lhs, rhs) -> bool:
  """Port of IsSortedPyDictKeysEqual."""
  if len(lhs) != len(rhs):
    return False
  for a, b in zip(lhs, rhs):
    if a != b:
      return False
  return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class _Registration:
  __slots__ = ("kind", "type", "to_iterable", "from_iterable",
               "to_iterable_with_keys", "data_fields", "meta_fields")

  def __init__(self, kind, typ, to_iterable=None, from_iterable=None,
               to_iterable_with_keys=None, data_fields=(), meta_fields=()):
    self.kind = kind
    self.type = typ
    self.to_iterable = to_iterable
    self.from_iterable = from_iterable
    self.to_iterable_with_keys = to_iterable_with_keys
    self.data_fields = data_fields
    self.meta_fields = meta_fields


_SIG_REG_INIT = (
    "__init__(self, enable_none: bool = True, enable_tuple: bool = True, "
    "enable_namedtuple: bool = True, enable_list: bool = True, "
    "enable_dict: bool = True) -> None")
_SIG_FLATTEN = (
    "flatten(self, tree: object | None, leaf_predicate: "
    "typing.Callable[[typing.Any], bool] | None = None) -> "
    "tuple[list[typing.Any], PyTreeDef]")
_SIG_FLATTEN_WITH_PATH = (
    "flatten_with_path[KeyPath: tuple[Any, ...]](self, tree: object | None, "
    "leaf_predicate: typing.Callable[[typing.Any, typing.Any], bool] | None "
    "= None) -> tuple[list[tuple[KeyPath, typing.Any]], PyTreeDef]")
_SIG_REGISTER_NODE = (
    "register_node[T, Children: Iterable[Any], AuxData: Hashable, "
    "KeyLeafPairs: Iterable[tuple[Any, Any]]](self, type: type[T], "
    "to_iterable: typing.Callable[[T], tuple[Children, AuxData]], "
    "from_iterable: typing.Callable[[AuxData, Children], T], "
    "to_iterable_with_keys: typing.Callable[[T], tuple[KeyLeafPairs, "
    "AuxData]] | None = None) -> typing.Any")
_SIG_REGISTER_DATACLASS = (
    "register_dataclass_node(self, type: type, data_fields: "
    "typing.Sequence[str], meta_fields: typing.Sequence[str], /) -> "
    "typing.Any")


def _is_str_sequence(x) -> bool:
  return isinstance(x, (list, tuple)) and all(isinstance(s, str) for s in x)


class PyTreeRegistry:
  """Registry of pytree node types (pure-Python port of the C++ class)."""

  def __init__(self, *args, **kwargs) -> None:
    names = ("enable_none", "enable_tuple", "enable_namedtuple",
             "enable_list", "enable_dict")
    flags = dict.fromkeys(names, True)
    ok = len(args) <= len(names) and all(k in names for k in kwargs) and not (
        set(names[:len(args)]) & set(kwargs))
    if ok:
      flags.update(zip(names, args))
      flags.update(kwargs)
      ok = all(type(v) is bool for v in flags.values())
    if not ok:
      raise _incompatible("__init__", _SIG_REG_INIT, (self,) + args, kwargs)
    enable_none = flags["enable_none"]
    enable_tuple = flags["enable_tuple"]
    enable_list = flags["enable_list"]
    enable_dict = flags["enable_dict"]
    regs = {}
    if enable_none:
      regs[id(_NoneType)] = _Registration(_NONE, _NoneType)
    if enable_tuple:
      regs[id(tuple)] = _Registration(_TUPLE, tuple)
    if enable_list:
      regs[id(list)] = _Registration(_LIST, list)
    if enable_dict:
      regs[id(dict)] = _Registration(_DICT, dict)
      if _frozendict is not None:
        regs[id(_frozendict)] = _Registration(_FROZENDICT, _frozendict)
    # NB: like the C++ implementation, `enable_namedtuple` has no effect:
    # namedtuples are always recognized heuristically.
    self._registrations = regs

  def __reduce__(self):
    return self.__name__

  # -- type classification -------------------------------------------------

  def _kind_of(self, obj):
    reg = self._registrations.get(id(type(obj)))
    if reg is not None:
      return reg.kind, reg
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
      return _NAMEDTUPLE, None
    return _LEAF, None

  def is_node(self, type) -> bool:  # pylint: disable=redefined-builtin
    if id(type) in self._registrations:
      return True
    return (isinstance(type, builtins.type) and issubclass(type, tuple)
            and hasattr(type, "_fields"))

  # -- registration ----------------------------------------------------------

  def register_node(self, type, to_iterable, from_iterable,  # pylint: disable=redefined-builtin
                    to_iterable_with_keys=None):
    if (not callable(to_iterable) or not callable(from_iterable)
        or (to_iterable_with_keys is not None
            and not callable(to_iterable_with_keys))):
      args = (self, type, to_iterable, from_iterable)
      if to_iterable_with_keys is not None:
        args += (to_iterable_with_keys,)
      raise _incompatible("register_node", _SIG_REGISTER_NODE, args)
    if id(type) in self._registrations:
      raise ValueError(
          f"Duplicate custom PyTreeDef type registration for {type!r}.")
    self._registrations[id(type)] = _Registration(
        _CUSTOM, type, to_iterable, from_iterable, to_iterable_with_keys)

  def register_dataclass_node(self, type, data_fields, meta_fields, /):  # pylint: disable=redefined-builtin
    if not _is_str_sequence(data_fields) or not _is_str_sequence(meta_fields):
      raise _incompatible("register_dataclass_node", _SIG_REGISTER_DATACLASS,
                          (self, type, data_fields, meta_fields))
    if id(type) in self._registrations:
      raise ValueError(
          "Duplicate custom dataclass PyTreeDef type registration for "
          f"{type!r}.")
    self._registrations[id(type)] = _Registration(
        _DATACLASS, type, data_fields=tuple(data_fields),
        meta_fields=tuple(meta_fields))

  # -- flattening ------------------------------------------------------------

  def flatten(self, tree, leaf_predicate=None):
    if leaf_predicate is not None and not callable(leaf_predicate):
      raise _incompatible("flatten", _SIG_FLATTEN,
                          (self, tree, leaf_predicate))
    leaves = []
    nodes = []
    self._flatten(tree, leaves, nodes, leaf_predicate, None)
    return leaves, PyTreeDef._make(self, nodes)

  def flatten_with_path(self, tree, leaf_predicate=None):
    if leaf_predicate is not None and not callable(leaf_predicate):
      raise _incompatible("flatten_with_path", _SIG_FLATTEN_WITH_PATH,
                          (self, tree, leaf_predicate))
    leaves = []
    nodes = []
    self._flatten(tree, leaves, nodes, leaf_predicate, [])
    return leaves, PyTreeDef._make(self, nodes)

  def _flatten(self, handle, leaves, nodes, pred, keypath):
    """Recursive post-order flatten. `keypath` is a list or None."""
    start_nodes = len(nodes)
    start_leaves = len(leaves)
    if pred is not None:
      if keypath is not None:
        o = pred(tuple(keypath), handle)
      else:
        o = pred(handle)
      if o is None or o is False:
        known_leaf = False
      elif o is True:
        known_leaf = True
      else:
        raise ValueError(
            f"is_leaf predicate returned a non-boolean value {o!r}; "
            "expected a boolean")
      if known_leaf:
        leaves.append(handle if keypath is None else (tuple(keypath), handle))
        nodes.append((_LEAF, 0, None, None, 1, 1))
        return

    reg = self._registrations.get(id(type(handle)))
    if reg is not None:
      kind = reg.kind
    elif isinstance(handle, tuple) and hasattr(handle, "_fields"):
      kind = _NAMEDTUPLE
    else:
      kind = _LEAF

    if kind == _LEAF:
      if isinstance(handle, _WARN_ITER_TYPES) or isinstance(
          handle, _DICT_VALUES_TYPE):
        warnings.warn(
            f"Python iterable type '{type(handle).__name__}' is treated as a "
            "leaf in PyTree. Use is_leaf to suppress this warning if you want "
            "this behavior. In a future version of JAX, this will become an "
            "error.", DeprecationWarning, stacklevel=1)
      leaves.append(handle if keypath is None else (tuple(keypath), handle))
      nodes.append((_LEAF, 0, None, None, 1, 1))
      return

    recurse = self._flatten
    node_data = None
    custom = None
    if kind == _NONE:
      arity = 0
    elif kind == _TUPLE or kind == _LIST:
      arity = len(handle)
      if keypath is None:
        for child in handle:
          recurse(child, leaves, nodes, pred, None)
      else:
        for i, child in enumerate(handle):
          keypath.append(SequenceKey(i))
          recurse(child, leaves, nodes, pred, keypath)
          keypath.pop()
    elif kind == _DICT or kind == _FROZENDICT:
      keys = _sorted_dict_keys(handle)
      for k in keys:
        if keypath is None:
          recurse(handle[k], leaves, nodes, pred, None)
        else:
          keypath.append(DictKey(k))
          recurse(handle[k], leaves, nodes, pred, keypath)
          keypath.pop()
      arity = len(keys)
      node_data = keys
    elif kind == _NAMEDTUPLE:
      arity = len(handle)
      node_data = type(handle)
      if keypath is None:
        for child in handle:
          recurse(child, leaves, nodes, pred, None)
      else:
        fields = _namedtuple_fields(handle)
        for i, child in enumerate(handle):
          keypath.append(GetAttrKey(fields[i]))
          recurse(child, leaves, nodes, pred, keypath)
          keypath.pop()
    elif kind == _CUSTOM:
      custom = reg
      arity = 0
      if keypath is not None and reg.to_iterable_with_keys is not None:
        pairs, node_data = _call_to_iterable_with_keys(reg, handle)
        for k, child in pairs:
          arity += 1
          keypath.append(k)
          recurse(child, leaves, nodes, pred, keypath)
          keypath.pop()
      else:
        out, it = _call_to_iterable(reg, handle)
        node_data = out[1]
        for child in it:
          if keypath is None:
            recurse(child, leaves, nodes, pred, None)
          else:
            keypath.append(FlattenedIndexKey(arity))
            recurse(child, leaves, nodes, pred, keypath)
            keypath.pop()
          arity += 1
    else:  # _DATACLASS
      custom = reg
      node_data = tuple(getattr(handle, f) for f in reg.meta_fields)
      arity = len(reg.data_fields)
      for f in reg.data_fields:
        child = getattr(handle, f)
        if keypath is None:
          recurse(child, leaves, nodes, pred, None)
        else:
          keypath.append(GetAttrKey(f))
          recurse(child, leaves, nodes, pred, keypath)
          keypath.pop()
    nodes.append((kind, arity, node_data, custom,
                  len(leaves) - start_leaves, len(nodes) - start_nodes + 1))

  def _flatten_one_level(self, x, with_keys):
    kind, reg = self._kind_of(x)
    if kind == _NONE:
      return ((), None)
    if kind == _TUPLE or kind == _LIST:
      if with_keys:
        return (tuple((SequenceKey(i), c) for i, c in enumerate(x)), None)
      return (x, None)
    if kind == _DICT or kind == _FROZENDICT:
      keys = _sorted_dict_keys(x)
      if with_keys:
        values = tuple((DictKey(k), x[k]) for k in keys)
      else:
        values = tuple(x[k] for k in keys)
      return (values, tuple(keys))
    if kind == _NAMEDTUPLE:
      if with_keys:
        fields = _namedtuple_fields(x)
        return ([(GetAttrKey(fields[i]), c) for i, c in enumerate(x)],
                type(x))
      return (list(x), type(x))
    if kind == _CUSTOM:
      if with_keys:
        if reg.to_iterable_with_keys is not None:
          return _call_to_iterable_with_keys(reg, x)
        out, it = _call_to_iterable(reg, x)
        return ([(FlattenedIndexKey(i), c) for i, c in enumerate(it)], out[1])
      out, _ = _call_to_iterable(reg, x)
      return (out[0], out[1])
    if kind == _DATACLASS:
      leaves = []
      for f in reg.data_fields:
        v = getattr(x, f)
        leaves.append((GetAttrKey(f), v) if with_keys else v)
      aux = tuple(getattr(x, f) for f in reg.meta_fields)
      return (leaves, aux)
    return None

  def flatten_one_level(self, tree):
    return self._flatten_one_level(tree, False)

  def flatten_one_level_with_keys(self, tree):
    return self._flatten_one_level(tree, True)


def _namedtuple_fields(x):
  fields = getattr(x, "_fields")
  if not isinstance(fields, tuple) or len(fields) != len(x):
    raise ValueError(
        "A namedtuple's _fields attribute should have the same size as the "
        "tuple.")
  return [str(f) for f in fields]


def _hashable(x) -> bool:
  try:
    hash(x)
  except TypeError:
    return False
  return True


PyTreeRegistry.__init__.__doc__ = _SIG_REG_INIT
PyTreeRegistry.flatten.__doc__ = _SIG_FLATTEN
PyTreeRegistry.flatten_with_path.__doc__ = _SIG_FLATTEN_WITH_PATH
PyTreeRegistry.register_node.__doc__ = _SIG_REGISTER_NODE
PyTreeRegistry.register_dataclass_node.__doc__ = _SIG_REGISTER_DATACLASS
PyTreeRegistry.flatten_one_level.__doc__ = (
    "flatten_one_level(self, tree: object | None) -> "
    "tuple[typing.Iterable[typing.Any], typing.Any] | None")
PyTreeRegistry.flatten_one_level_with_keys.__doc__ = (
    "flatten_one_level_with_keys[KeyLeafPair: tuple[Any, Any]](self, tree: "
    "object | None) -> tuple[typing.Iterable[KeyLeafPair], typing.Any] | None")
PyTreeRegistry.is_node.__doc__ = "is_node(self, type: type) -> bool"


# ---------------------------------------------------------------------------
# PyTreeDef
# ---------------------------------------------------------------------------

_SIG_UNFLATTEN = (
    "unflatten(self, arg: typing.Iterable[typing.Any], /) -> typing.Any")
_SIG_WALK = (
    "walk[T](self, __f_node: typing.Callable[[typing.Any, typing.Any], "
    "typing.Any], __f_leaf: typing.Callable[[T], typing.Any] | None, leaves: "
    "typing.Iterable[typing.Any], /) -> typing.Any")
_SIG_FROM_NODE_DATA = (
    "from_node_data_and_children(self, registry: PyTreeRegistry, node_data: "
    "tuple[type, Any] | None, children: typing.Iterable[PyTreeDef]) -> "
    "PyTreeDef")
_SIG_DESERIALIZE = (
    "deserialize_using_proto(registry: {mod}.PyTreeRegistry, data: bytes) -> "
    "{mod}.PyTreeDef")
_SIG_COMPOSE = (
    "compose(self, arg: {mod}.PyTreeDef, /) -> {mod}.PyTreeDef")


_NONE_PREFIX_MSG = (
    "\n\nIn previous releases of JAX, flatten-up-to used to consider None to "
    "be a tree-prefix of non-None values. To obtain the previous behavior, "
    "you can usually write:\n"
    "  jax.tree.map(lambda x, y: None if x is None else f(x, y), a, b, "
    "is_leaf=lambda x: x is None)")


class PyTreeDef:
  """A pytree structure: a post-order traversal of nodes."""

  __slots__ = ("_registry", "_nodes", "__weakref__")

  def __init__(self, *args, **kwargs):
    raise TypeError(f"{__name__}.PyTreeDef: no constructor defined!")

  @classmethod
  def _make(cls, registry, nodes):
    self = object.__new__(cls)
    self._registry = registry
    self._nodes = nodes
    return self

  # -- basic properties ------------------------------------------------------

  @property
  def num_leaves(self) -> int:
    return self._nodes[-1][_NLEAVES] if self._nodes else 0

  @property
  def num_nodes(self) -> int:
    return len(self._nodes)

  def children(self):
    nodes = self._nodes
    if not nodes:
      return []
    root = nodes[-1]
    children = [None] * root[_ARITY]
    pos = len(nodes) - 1
    for i in range(root[_ARITY] - 1, -1, -1):
      n = nodes[pos - 1][_NNODES]
      if pos < n:
        raise RuntimeError("children() walked off start of array")
      children[i] = PyTreeDef._make(self._registry, nodes[pos - n:pos])
      pos -= n
    if pos != 0:
      raise RuntimeError("pos != 0 at end of PyTreeDef::Children")
    return children

  def node_data(self):
    """Returns None if a leaf-pytree, else (type, node_data)"""
    if not self._nodes:
      raise RuntimeError("empty PyTreeDef traversal.")
    node = self._nodes[-1]
    kind = node[_KIND]
    if kind == _LEAF:
      return None
    if kind == _DICT:
      return (dict, list(node[_DATA]))
    if kind == _FROZENDICT:
      return (_frozendict, list(node[_DATA]))
    if kind in (_CUSTOM, _DATACLASS):
      return (node[_CUSTOMREG].type, node[_DATA])
    return (_node_type(node), None)

  # -- unflattening ----------------------------------------------------------

  @staticmethod
  def _make_node(node, children):
    if len(children) != node[_ARITY]:
      raise RuntimeError("Node arity mismatch.")
    kind = node[_KIND]
    if kind == _NONE:
      return None
    if kind == _TUPLE:
      return tuple(children)
    if kind == _LIST:
      return list(children)
    if kind == _NAMEDTUPLE:
      return node[_DATA](*children)
    if kind == _DICT or kind == _FROZENDICT:
      d = {}
      keys = node[_DATA]
      if len(keys) < len(children):
        raise RuntimeError("Number of keys and entries does not match.")
      for i, c in enumerate(children):
        d[keys[i]] = c
      return d if kind == _DICT else _frozendict(d)
    if kind == _CUSTOM:
      return node[_CUSTOMREG].from_iterable(node[_DATA], tuple(children))
    if kind == _DATACLASS:
      reg = node[_CUSTOMREG]
      kwargs = {}
      for f, v in zip(reg.meta_fields, node[_DATA]):
        kwargs[f] = v
      for f, v in zip(reg.data_fields, children):
        kwargs[f] = v
      return reg.type(**kwargs)
    raise RuntimeError("MakeNode not implemented for leaves.")

  def unflatten(self, arg, /):
    try:
      it = iter(arg)
    except TypeError:
      raise _incompatible("unflatten", _SIG_UNFLATTEN, (self, arg)) from None
    agenda = []
    count = 0
    make_node = self._make_node
    for node in self._nodes:
      arity = node[_ARITY]
      if len(agenda) < arity:
        raise RuntimeError("Too few elements for TreeDef node.")
      if node[_KIND] == _LEAF:
        x = next(it, _SENTINEL)
        if x is _SENTINEL:
          raise ValueError(
              f"Too few leaves for PyTreeDef; expected {self.num_leaves}, "
              f"got {count}")
        agenda.append(x)
        count += 1
      else:
        if arity:
          children = agenda[-arity:]
          del agenda[-arity:]
        else:
          children = []
        agenda.append(make_node(node, children))
    for _ in it:
      raise ValueError(
          f"Too many leaves for PyTreeDef; expected {self.num_leaves}.")
    if len(agenda) != 1:
      raise RuntimeError("PyTreeDef traversal did not yield a singleton.")
    return agenda[0]

  def walk(self, f_node, f_leaf, leaves, /):
    """Walk pytree, calling f_node(node, node_data) at nodes, and f_leaf at leaves"""
    if not callable(f_node) or not callable(f_leaf) or not _is_iterable(leaves):
      raise _incompatible("walk", _SIG_WALK, (self, f_node, f_leaf, leaves))
    it = iter(leaves)
    agenda = []
    for node in self._nodes:
      if node[_KIND] == _LEAF:
        leaf = next(it, _SENTINEL)
        if leaf is _SENTINEL:
          raise ValueError("Too few leaves for PyTreeDef")
        agenda.append(f_leaf(leaf))
      else:
        arity = node[_ARITY]
        if len(agenda) < arity:
          raise RuntimeError("Too few elements for custom type.")
        if arity:
          children = tuple(agenda[-arity:])
          del agenda[-arity:]
        else:
          children = ()
        kind = node[_KIND]
        if kind in _DICT_KINDS:
          data = list(node[_DATA])
        elif kind in (_NAMEDTUPLE, _CUSTOM, _DATACLASS):
          data = node[_DATA]
        else:
          data = None
        agenda.append(f_node(children, data))
    for _ in it:
      raise ValueError("Too many leaves for PyTreeDef")
    if len(agenda) != 1:
      raise RuntimeError("PyTreeDef traversal did not yield a singleton.")
    return agenda[0]

  def from_iterable_tree(self, arg, /):
    pos = [len(self._nodes) - 1]
    make_node = self._make_node

    def helper(xs):
      if pos[0] < 0:
        raise ValueError("Tree structures did not match.")
      node = self._nodes[pos[0]]
      pos[0] -= 1
      if node[_KIND] == _LEAF:
        return xs
      ys = list(xs)
      if len(ys) != node[_ARITY]:
        raise ValueError("Arity mismatch between trees")
      for j in range(node[_ARITY] - 1, -1, -1):
        ys[j] = helper(ys[j])
      return make_node(node, ys)

    out = helper(arg)
    if pos[0] != -1:
      raise ValueError("Tree structures did not match.")
    return out

  # -- flatten_up_to ---------------------------------------------------------

  def flatten_up_to(self, tree):
    xs = tree
    nodes = self._nodes
    num_leaves = self.num_leaves
    leaves = [None] * num_leaves
    leaf = num_leaves - 1
    agenda = [xs]
    pos = len(nodes) - 1
    registry = self._registry
    while agenda:
      if pos < 0:
        raise ValueError(
            f"Tree structures did not match: {xs!r} vs {self}")
      node = nodes[pos]
      pos -= 1
      obj = agenda.pop()
      kind = node[_KIND]
      if kind == _LEAF:
        if leaf < 0:
          raise RuntimeError("Leaf count mismatch.")
        leaves[leaf] = obj
        leaf -= 1
      elif kind == _NONE:
        if obj is not None:
          raise ValueError(f"Expected None, got {obj!r}." + _NONE_PREFIX_MSG)
      elif kind == _TUPLE:
        if type(obj) is not tuple:
          raise ValueError(f"Expected tuple, got {obj!r}.")
        if len(obj) != node[_ARITY]:
          raise ValueError(
              f"Tuple arity mismatch: {len(obj)} != {node[_ARITY]}; "
              f"tuple: {obj!r}.")
        agenda.extend(obj)
      elif kind == _LIST:
        if type(obj) is not list:
          raise ValueError(f"Expected list, got {obj!r}.")
        if len(obj) != node[_ARITY]:
          raise ValueError(
              f"List arity mismatch: {len(obj)} != {node[_ARITY]}; "
              f"list: {obj!r}.")
        agenda.extend(obj)
      elif kind == _DICT or kind == _FROZENDICT:
        tname = "dict" if kind == _DICT else "frozendict"
        if type(obj) is not (dict if kind == _DICT else _frozendict):
          raise ValueError(f"Expected {tname}, got {obj!r}.")
        keys = _sorted_dict_keys(obj)
        expected = node[_DATA]
        if not _keys_equal(keys, expected):
          raise ValueError(
              f"{'Dict' if kind == _DICT else tname} key mismatch; expected "
              f"keys: {list(expected)!r}; present keys: {keys!r}.")
        for k in keys:
          agenda.append(obj[k])
      elif kind == _NAMEDTUPLE:
        if not isinstance(obj, tuple) or not hasattr(obj, "_fields"):
          raise ValueError(f"Expected named tuple, got {obj!r}.")
        if len(obj) != node[_ARITY]:
          raise ValueError(
              f"Named tuple arity mismatch: {len(obj)} != {node[_ARITY]}; "
              f"tuple: {obj!r}.")
        if type(obj) != node[_DATA]:
          raise ValueError(
              f"Named tuple type mismatch: expected type: {node[_DATA]!r}, "
              f"tuple: {obj!r}.")
        agenda.extend(obj)
      elif kind == _CUSTOM:
        reg = node[_CUSTOMREG]
        if registry._registrations.get(id(type(obj))) is not reg:
          raise ValueError(
              f"Custom node type mismatch: expected type: {reg.type!r}, "
              f"value: {obj!r}.")
        out, it = _call_to_iterable(reg, obj)
        if node[_DATA] != out[1]:
          raise ValueError(
              f"Mismatch custom node data: {node[_DATA]!r} != {out[1]!r}; "
              f"value: {obj!r}.")
        arity = 0
        for child in it:
          arity += 1
          agenda.append(child)
        if arity != node[_ARITY]:
          raise ValueError(
              f"Custom type arity mismatch: {arity} != {node[_ARITY]}; "
              f"value: {obj!r}.")
      elif kind == _DATACLASS:
        reg = node[_CUSTOMREG]
        if registry._registrations.get(id(type(obj))) is not reg:
          raise ValueError(
              "Custom dataclass node type mismatch: expected type: "
              f"{reg.type!r}, value: {obj!r}.")
        meta = tuple(getattr(obj, f) for f in reg.meta_fields)
        if node[_DATA] != meta:
          raise ValueError(
              f"Mismatch custom dataclass node data: {node[_DATA]!r} != "
              f"{meta!r}; value: {obj!r}.")
        if len(reg.data_fields) != node[_ARITY]:
          raise ValueError(
              "Custom type arity mismatch: "
              f"{len(reg.data_fields)} != {node[_ARITY]}; value: {obj!r}.")
        for f in reg.data_fields:
          agenda.append(getattr(obj, f))
      else:
        raise RuntimeError("Unknown node kind.")
    if pos != -1 or leaf != -1:
      raise ValueError(f"Tree structures did not match: {xs!r} vs {self}")
    return leaves

  # -- composition -----------------------------------------------------------

  def compose(self, arg, /):
    if not isinstance(arg, PyTreeDef):
      raise _incompatible(
          "compose", _SIG_COMPOSE.format(mod=__name__), (self, arg))
    if arg._registry is not self._registry:
      raise ValueError(
          "PyTree registries of PyTreeDefs passed to Compose() must match.")
    nodes = []
    for n in self._nodes:
      if n[_KIND] == _LEAF:
        nodes.extend(arg._nodes)
      else:
        nodes.append(n)
    return PyTreeDef._make(self._registry, _set_num_leaves_and_nodes(nodes))

  @staticmethod
  def from_node_data_and_children(registry, node_data, children):
    """Reconstructs a pytree from `node_data()` and `children()`."""
    if (not isinstance(registry, PyTreeRegistry)
        or not (node_data is None or (isinstance(node_data, (tuple, list))
                                      and len(node_data) == 2))
        or not _is_iterable(children)):
      raise _incompatible("from_node_data_and_children", _SIG_FROM_NODE_DATA,
                          (registry, node_data, children))
    nodes = []
    num_leaves = 0
    arity = 0
    for c in children:
      if not isinstance(c, PyTreeDef):
        raise RuntimeError("std::bad_cast")
      nodes.extend(c._nodes)
      num_leaves += c.num_leaves
      arity += 1
    if node_data is None:
      nodes.append((_LEAF, arity, None, None, num_leaves + 1, len(nodes) + 1))
      return PyTreeDef._make(registry, nodes)
    typ, data = node_data[0], node_data[1]
    custom = None
    if issubclass(typ, tuple) and hasattr(typ, "_fields"):
      kind = _NAMEDTUPLE
      data = typ
    else:
      reg = registry._registrations.get(id(typ))
      if reg is None:
        raise RuntimeError(f"Could not find type: {typ!r}.")
      kind = reg.kind
      if kind in (_CUSTOM, _DATACLASS):
        custom = reg
      elif kind in _DICT_KINDS:
        if not isinstance(data, (list, tuple)):
          raise RuntimeError("std::bad_cast")
        data = list(data)
      else:
        data = None
    nodes.append((kind, arity, data, custom, num_leaves, len(nodes) + 1))
    return PyTreeDef._make(registry, nodes)

  # -- comparison / hashing / printing --------------------------------------

  def _equals(self, other):
    a_nodes, b_nodes = self._nodes, other._nodes
    if len(a_nodes) != len(b_nodes):
      return False
    for a, b in zip(a_nodes, b_nodes):
      kind = a[_KIND]
      a_dict = kind in _DICT_KINDS
      b_dict = b[_KIND] in _DICT_KINDS
      if (kind != b[_KIND] or a[_ARITY] != b[_ARITY]
          or (a_dict and len(a[_DATA]) != len(b[_DATA]))):
        return False
      ca, cb = a[_CUSTOMREG], b[_CUSTOMREG]
      if kind == _CUSTOM:
        if (ca is None) != (cb is None):
          return False
        if ca is not None and (ca.from_iterable is not cb.from_iterable
                               or ca.type is not cb.type):
          return False
      if kind in (_NAMEDTUPLE, _CUSTOM, _DATACLASS):
        try:
          ne = bool(a[_DATA] != b[_DATA])
        except Exception as e:
          msg = ("Exception raised while checking equality of metadata "
                 "fields of pytree")
          if ca is not None:
            msg += f" node of type {ca.type!r}"
          msg += (". Make sure that metadata fields are hashable and have "
                  "simple equality semantics. (Note: arrays cannot be passed "
                  "as metadata fields!)")
          raise ValueError(msg) from e
        if ne:
          return False
      if a_dict and not _keys_equal(a[_DATA], b[_DATA]):
        return False
    return True

  def __eq__(self, other):
    if other is None:  # nanobind rejects None for the `object` argument.
      raise _incompatible("__eq__", "__eq__(self, arg: object, /) -> bool",
                          (self, other))
    return isinstance(other, PyTreeDef) and self._equals(other)

  def __ne__(self, other):
    if other is None:
      raise _incompatible("__ne__", "__ne__(self, arg: object, /) -> bool",
                          (self, other))
    return isinstance(other, PyTreeDef) and not self._equals(other)

  def __hash__(self):
    parts = []
    for n in self._nodes:
      c = n[_CUSTOMREG]
      if n[_KIND] == _CUSTOM and c is not None:
        parts.append((n[_KIND], n[_ARITY], id(c.type), id(c.from_iterable)))
      else:
        # NB: the C++ hash mixes in the registration pointer for dataclass
        # nodes, but equality ignores it; we keep hash consistent with eq.
        parts.append((n[_KIND], n[_ARITY]))
    return hash(tuple(parts))

  def _to_string(self):
    agenda = []
    for node in self._nodes:
      kind, arity = node[_KIND], node[_ARITY]
      if len(agenda) < arity:
        raise RuntimeError("Too few elements for container.")
      if kind == _LEAF:
        agenda.append("*")
        continue
      children = agenda[len(agenda) - arity:]
      del agenda[len(agenda) - arity:]
      if kind == _NONE:
        rep = "None"
      elif kind == _TUPLE:
        rep = "(" + ", ".join(children) + ("," if arity == 1 else "") + ")"
      elif kind == _LIST:
        rep = "[" + ", ".join(children) + "]"
      elif kind in _DICT_KINDS:
        keys = node[_DATA]
        if len(keys) != arity:
          raise RuntimeError("Number of keys and entries does not match.")
        rep = ("{" if kind == _DICT else "frozendict({") + ", ".join(
            f"{k!r}: {c}" for k, c in zip(keys, children)) + (
                "}" if kind == _DICT else "})")
      else:
        if kind == _NAMEDTUPLE:
          name = "namedtuple"
          data = node[_DATA].__name__
        else:
          name = node[_CUSTOMREG].type.__name__
          data = str(node[_DATA])
        rep = f"CustomNode({name}[{data}], [{', '.join(children)}])"
      agenda.append(rep)
    if len(agenda) != 1:
      raise RuntimeError("PyTreeDef traversal did not yield a singleton.")
    return f"PyTreeDef({agenda[0]})"

  def __repr__(self):
    return self._to_string()

  __str__ = __repr__

  # -- pickling --------------------------------------------------------------

  def __getstate__(self):
    out = []
    for n in self._nodes:
      kind = n[_KIND]
      data = n[_DATA]
      if kind in _DICT_KINDS:
        data = list(data)
      elif kind not in (_NAMEDTUPLE, _CUSTOM, _DATACLASS):
        data = None
      custom = None if n[_CUSTOMREG] is None else n[_CUSTOMREG].type
      out.append((kind, n[_ARITY], data, custom, n[_NLEAVES], n[_NNODES]))
    return (self._registry, out)

  def __setstate__(self, state, /):
    if not isinstance(state, tuple):
      raise RuntimeError("std::bad_cast")
    if len(state) != 2:
      raise _jax_runtime_error("Malformed pickled PyTreeDef, expected 2-tuple")
    registry, node_states = state
    if not isinstance(registry, PyTreeRegistry) or not isinstance(
        node_states, list):
      raise RuntimeError("std::bad_cast")
    self._registry = registry
    self._nodes = nodes = []
    for ns in node_states:
      if not isinstance(ns, tuple):
        raise RuntimeError("std::bad_cast")
      if len(ns) != 6:
        raise _jax_runtime_error("Malformed pickled PyTreeDef")
      kind, arity, data, custom_type, num_leaves, num_nodes = ns
      if kind not in range(_LEAF, _FROZENDICT + 1):
        # The C++ code reinterprets out-of-range values; we refuse them.
        raise _jax_runtime_error("Malformed pickled PyTreeDef")
      if kind in _DICT_KINDS:
        if not isinstance(data, (list, tuple)):
          raise RuntimeError("std::bad_cast")
        data = list(data)
      elif kind not in (_NAMEDTUPLE, _CUSTOM, _DATACLASS):
        if data is not None:
          raise _jax_runtime_error("Malformed pickled PyTreeDef")
      custom = None
      if kind in (_CUSTOM, _DATACLASS):
        if custom_type is not None:
          custom = registry._registrations.get(id(custom_type))
        if custom is None:
          raise _jax_runtime_error(
              f"Unknown custom type in pickled PyTreeDef: {custom_type!r}")
      elif custom_type is not None:
        raise _jax_runtime_error("Malformed pickled PyTreeDef")
      nodes.append((kind, arity, data, custom, num_leaves, num_nodes))

  # -- protobuf serialization ------------------------------------------------

  def serialize_using_proto(self) -> bytes:
    interned = {}
    node_msgs = []
    for node in self._nodes:
      kind = node[_KIND]
      if kind == _LEAF:
        ptype = 1
      elif kind == _LIST:
        ptype = 2
      elif kind == _NONE:
        ptype = 3
      elif kind == _TUPLE:
        ptype = 4
      elif kind == _DICT:
        ptype = 5
      else:
        raise ValueError(
            "User-defined nodes are not supported when serializing pytrees "
            "as protocol buffers. You should either convert the user-defined "
            "nodes to another type or use pickle instead.")
      msg = bytearray()
      if node[_ARITY]:
        msg += b"\x08" + _encode_varint(node[_ARITY])
      msg += b"\x10" + _encode_varint(ptype)
      if kind == _DICT:
        ids = bytearray()
        for k in node[_DATA]:
          if not isinstance(k, str):
            raise ValueError(
                "Only string keys are supported in proto pytree serialization.")
          if k not in interned:
            interned[k] = len(interned)
          ids += _encode_varint(interned[k])
        if ids:
          keys_msg = b"\x0a" + _encode_varint(len(ids)) + bytes(ids)
          msg += b"\x1a" + _encode_varint(len(keys_msg)) + keys_msg
      node_msgs.append(bytes(msg))
    out = bytearray()
    for m in node_msgs:
      out += b"\x0a" + _encode_varint(len(m)) + m
    for s in interned:
      b = s.encode("utf-8")
      out += b"\x12" + _encode_varint(len(b)) + b
    return bytes(out)

  @staticmethod
  def deserialize_using_proto(registry, data):
    if not isinstance(registry, PyTreeRegistry) or not isinstance(data, bytes):
      raise _incompatible(
          "deserialize_using_proto", _SIG_DESERIALIZE.format(mod=__name__),
          (registry, data))
    try:
      proto_nodes, strings = _parse_pytreedef_proto(data)
    except _ProtoError:
      raise _jax_runtime_error("Could not deserialize PyTreeDefProto.") from None
    kind_map = {1: _LEAF, 2: _LIST, 3: _NONE, 4: _TUPLE, 5: _DICT}
    nodes = []
    for arity, ptype, str_ids in proto_nodes:
      # `arity` is a proto uint32 stored in a C++ int.
      if arity >= 2**31:
        arity -= 2**32
      kind = kind_map.get(ptype)
      if kind is None:
        raise ValueError("Malformed pytree proto (invalid node type)")
      node_data = None
      if kind == _DICT:
        keys = []
        for i in str_ids:
          if i >= len(strings):
            raise ValueError("Malformed pytree proto (dict_key out of range).")
          keys.append(strings[i])
        if arity < 0 or len(keys) != arity:
          raise ValueError(
              f"Malformed pytree proto (dict node has arity {arity}, which "
              f"does not match its number of keys {len(keys)}).")
        node_data = keys
      nodes.append((kind, arity, node_data, None, 0, 0))
    return PyTreeDef._make(registry, _set_num_leaves_and_nodes(nodes))

PyTreeDef.unflatten.__doc__ = _SIG_UNFLATTEN
PyTreeDef.flatten_up_to.__doc__ = (
    "flatten_up_to(self, tree: object | None) -> list")
PyTreeDef.children.__doc__ = (
    f"children(self) -> list[{__name__}.PyTreeDef]")
PyTreeDef.serialize_using_proto.__doc__ = (
    "serialize_using_proto(self) -> bytes")


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format codec for PyTreeDefProto
# ---------------------------------------------------------------------------

class _ProtoError(Exception):
  pass


def _encode_varint(v: int) -> bytes:
  v &= (1 << 64) - 1
  out = bytearray()
  while True:
    b = v & 0x7F
    v >>= 7
    if v:
      out.append(b | 0x80)
    else:
      out.append(b)
      return bytes(out)


def _read_varint(buf, pos):
  result = 0
  shift = 0
  while True:
    if pos >= len(buf) or shift >= 70:
      raise _ProtoError()
    b = buf[pos]
    pos += 1
    result |= (b & 0x7F) << shift
    shift += 7
    if not b & 0x80:
      return result & ((1 << 64) - 1), pos


def _iter_fields(buf):
  """Yields (field_number, wire_type, value) for a serialized message."""
  pos = 0
  n = len(buf)
  while pos < n:
    tag, pos = _read_varint(buf, pos)
    field, wt = tag >> 3, tag & 7
    if field == 0:
      raise _ProtoError()
    if wt == 0:
      val, pos = _read_varint(buf, pos)
    elif wt == 1:
      if pos + 8 > n:
        raise _ProtoError()
      val, pos = buf[pos:pos + 8], pos + 8
    elif wt == 2:
      length, pos = _read_varint(buf, pos)
      if pos + length > n:
        raise _ProtoError()
      val, pos = buf[pos:pos + length], pos + length
    elif wt == 5:
      if pos + 4 > n:
        raise _ProtoError()
      val, pos = buf[pos:pos + 4], pos + 4
    else:
      raise _ProtoError()
    yield field, wt, val


def _parse_pytreedef_proto(data: bytes):
  nodes = []
  strings = []
  for field, wt, val in _iter_fields(data):
    if field == 1:
      if wt != 2:
        raise _ProtoError()
      arity, ptype, str_ids = 0, 0, []
      for f2, wt2, v2 in _iter_fields(val):
        if f2 == 1:
          if wt2 != 0:
            raise _ProtoError()
          arity = v2
        elif f2 == 2:
          if wt2 != 0:
            raise _ProtoError()
          ptype = v2 & 0xFFFFFFFF
          if ptype >= 2**31:
            ptype -= 2**32
        elif f2 == 3:
          if wt2 != 2:
            raise _ProtoError()
          for f3, wt3, v3 in _iter_fields(v2):
            if f3 == 1:
              if wt3 == 0:
                str_ids.append(v3 & 0xFFFFFFFF)
              elif wt3 == 2:
                p = 0
                while p < len(v3):
                  sid, p = _read_varint(v3, p)
                  str_ids.append(sid & 0xFFFFFFFF)
              else:
                raise _ProtoError()
      nodes.append((arity, ptype, str_ids))
    elif field == 2:
      if wt != 2:
        raise _ProtoError()
      try:
        strings.append(bytes(val).decode("utf-8"))
      except UnicodeDecodeError:
        raise _ProtoError() from None
  return nodes, strings


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

_default_registry = PyTreeRegistry()


def default_registry() -> PyTreeRegistry:
  return _default_registry


_SIG_TREEDEF_TUPLE = (
    "treedef_tuple(registry: PyTreeRegistry, arg0: typing.Sequence[PyTreeDef]"
    ", /) -> PyTreeDef")
_SIG_ALL_LEAVES = (
    "all_leaves(arg0: {mod}.PyTreeRegistry, arg1: collections.abc.Iterable, "
    "/) -> bool")


def treedef_tuple(registry, treedefs, /):
  if not isinstance(registry, PyTreeRegistry) or not isinstance(treedefs, list):
    raise _incompatible("treedef_tuple", _SIG_TREEDEF_TUPLE,
                        (registry, treedefs))
  nodes = []
  for t in treedefs:
    if not isinstance(t, PyTreeDef):
      raise RuntimeError("std::bad_cast")
    if t._registry is not registry:
      raise ValueError(
          "PyTree registries of PyTreeDefs passed to Tuple() must match.")
    nodes.extend(t._nodes)
  nodes.append((_TUPLE, len(treedefs), None, None, 0, 0))
  return PyTreeDef._make(registry, _set_num_leaves_and_nodes(nodes))


def all_leaves(registry, iterable, /) -> bool:
  if not isinstance(registry, PyTreeRegistry) or not _is_iterable(iterable):
    raise _incompatible("all_leaves", _SIG_ALL_LEAVES.format(mod=__name__),
                        (registry, iterable))
  for x in iterable:
    if registry._kind_of(x)[0] != _LEAF:
      return False
  return True


default_registry.__doc__ = (
    f"default_registry() -> {__name__}.PyTreeRegistry")
treedef_tuple.__doc__ = _SIG_TREEDEF_TUPLE
all_leaves.__doc__ = _SIG_ALL_LEAVES.format(mod=__name__)
