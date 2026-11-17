"""Stand-in for jaxlib.weakref_lru_cache: LRU caches keyed partly by weakref.

Semantics follow jaxlib: the cache key includes `cache_context_fn()` (the
trace context), entries keyed on a weakly-referenced argument are dropped when
that object dies, and `cache_info()` reports hits/misses/maxsize/currsize.
"""
from __future__ import annotations

from collections import OrderedDict, namedtuple
import functools
import threading
import weakref

_Info = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class _LRU:
  def __init__(self, maxsize):
    self.maxsize = maxsize
    self.data: OrderedDict = OrderedDict()
    self.hits = self.misses = 0
    self.lock = threading.RLock()

  def lookup(self, key, compute):
    with self.lock:
      if key in self.data:
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key]
      self.misses += 1
    value = compute()
    with self.lock:
      self.data[key] = value
      self.data.move_to_end(key)
      if self.maxsize is not None:
        while len(self.data) > self.maxsize:
          self.data.popitem(last=False)
    return value

  def info(self):
    return _Info(self.hits, self.misses, self.maxsize, len(self.data))

  def clear(self):
    with self.lock:
      self.data.clear()
      self.hits = self.misses = 0


def _kw_key(kwargs):
  return tuple(sorted(kwargs.items())) if kwargs else ()


class StrongLRUCache:
  WeakrefLRUCacheInfo = _Info

  def __init__(self, fn, cache_context_fn=None, maxsize=2048, explain=None,
               num_shards=None):
    self._fn = fn
    self._context = cache_context_fn
    self._lru = _LRU(maxsize)
    functools.update_wrapper(self, fn)

  def __call__(self, *args, **kwargs):
    ctx = self._context() if self._context is not None else None
    key = (ctx, args, _kw_key(kwargs))
    return self._lru.lookup(key, lambda: self._fn(*args, **kwargs))

  def cache_info(self): return self._lru.info()
  def cache_clear(self): self._lru.clear()
  def cache_keys(self): return [k[1] for k in self._lru.data]

  def __get__(self, obj, objtype=None):
    return self if obj is None else functools.partial(self, obj)


def strong_lru_cache(fn, cache_context_fn=None, maxsize=2048, explain=None,
                     num_shards=None):
  return StrongLRUCache(fn, cache_context_fn, maxsize, explain, num_shards)


class _WeakKey:
  """Hashes/compares by the referent while it is alive."""
  __slots__ = ("ref", "hash", "__weakref__")

  def __init__(self, obj, callback=None):
    self.ref = weakref.ref(obj, callback)
    self.hash = hash(obj)

  def __hash__(self):
    return self.hash

  def __eq__(self, other):
    if not isinstance(other, _WeakKey):
      return NotImplemented
    a, b = self.ref(), other.ref()
    return a is not None and b is not None and (a is b or a == b)


class WeakrefLRUCache:
  WeakrefLRUCacheInfo = _Info

  def __init__(self, cache_context_fn, fn, maxsize=2048, explain=None,
               num_weakrefs=1):
    self._fn = fn
    self._context = cache_context_fn
    self._maxsize = maxsize
    self._explain = explain
    self._num_weakrefs = num_weakrefs
    # One LRU per weakly-referenced key tuple, dropped when a referent dies.
    self._per_weak: dict = {}
    self._hits = self._misses = 0
    self._lock = threading.RLock()
    functools.update_wrapper(self, fn)

  def _remove(self, dead_ref):
    # Weakref callback: `dead_ref` is the weakref.ref whose referent died.
    with self._lock:
      for k in [k for k in self._per_weak
                if any(w.ref is dead_ref for w in k)]:
        del self._per_weak[k]

  def __call__(self, *args, **kwargs):
    n = self._num_weakrefs
    weak_args, rest = args[:n], args[n:]
    wkey = tuple(_WeakKey(a, self._remove) for a in weak_args)
    ctx = self._context()
    key = (ctx, rest, _kw_key(kwargs))
    return self._call(wkey, key, args, kwargs)

  def _call(self, wkey, key, args, kwargs):
    with self._lock:
      lru = self._per_weak.get(wkey)
      if lru is None:
        lru = self._per_weak[wkey] = _LRU(self._maxsize)
      if key in lru.data:
        self._hits += 1
        lru.data.move_to_end(key)
        return lru.data[key]
      self._misses += 1
      explainer = self._explain() if self._explain is not None else None
      miss_keys = self._get_keys() if explainer is not None else None
    if explainer is not None:
      # jaxlib: explainer(miss_keys, *args, **kwargs) before computing.
      explainer(miss_keys, *args, **kwargs)
    value = self._fn(*args, **kwargs)
    with self._lock:
      lru.data[key] = value
      if self._maxsize is not None:
        while len(lru.data) > self._maxsize:
          lru.data.popitem(last=False)
    return value

  def cache_info(self):
    size = sum(len(l.data) for l in self._per_weak.values())
    return _Info(self._hits, self._misses, self._maxsize, size)

  def cache_clear(self):
    with self._lock:
      self._per_weak.clear()
      self._hits = self._misses = 0

  def _get_keys(self):
    """jaxlib GetKeys(): (weakref [or tuple of them], context, args, kwargs)."""
    out = []
    for wkey, lru in list(self._per_weak.items()):
      wr = wkey[0].ref if len(wkey) == 1 else tuple(w.ref for w in wkey)
      for ctx, rest, kw in list(lru.data):
        out.append((wr, ctx, rest, dict(kw)))
    return out

  def cache_keys(self):
    out = []
    for wkey, lru in self._per_weak.items():
      objs = tuple(w.ref() for w in wkey)
      out.extend((*objs, *k[1:]) for k in lru.data)
    return out

  def evict_weakref(self, obj):
    with self._lock:
      for k in [k for k in self._per_weak
                if any(w.ref() is obj for w in k)]:
        del self._per_weak[k]


def weakref_lru_cache(cache_context_fn, fn, maxsize=2048, explain=None,
                      num_weakrefs=1):
  return WeakrefLRUCache(cache_context_fn, fn, maxsize, explain, num_weakrefs)


class MultiWeakrefLRUCache(WeakrefLRUCache):
  """jaxlib's MultiWeakrefLRUCache: args are flattened with `registry`; leaves
  whose exact type is in `weak_types` (a live set, read at call time), or that
  are callable, are held weakly (the entry dies with any of them); the other
  leaves, the treedef and the sorted keyword names form the strong key.

  Holding the weak leaves strongly instead would leak: e.g. a cached Jaxpr
  keeps its consts (device arrays) alive forever.
  """

  def __init__(self, cache_context_fn, fn, *, maxsize=2048, explain=None,
               registry=None, weak_types=None):
    super().__init__(cache_context_fn, fn, maxsize, explain)
    self._registry = registry
    self._weak_types = weak_types if weak_types is not None else set()

  def __call__(self, *args, **kwargs):
    weak, strong, treedefs = [], [], []

    def process(value):
      if self._registry is not None:
        leaves, treedef = self._registry.flatten(value)
      else:
        leaves, treedef = [value], None
      treedefs.append(treedef)
      for leaf in leaves:
        if type(leaf) in self._weak_types or callable(leaf):
          weak.append(leaf)
        else:
          strong.append(leaf)

    for a in args:
      process(a)
    kwnames = tuple(sorted(kwargs))
    for name in kwnames:
      process(kwargs[name])
    wkey = tuple(_WeakKey(w, self._remove) for w in weak)
    key = (self._context(), kwnames, tuple(strong), tuple(treedefs))
    return self._call(wkey, key, args, kwargs)


def multi_weakref_lru_cache(cache_context_fn, fn, *, maxsize=2048, explain=None,
                            registry=None, weak_types=None):
  return MultiWeakrefLRUCache(cache_context_fn, fn, maxsize=maxsize,
                              explain=explain, registry=registry,
                              weak_types=weak_types)


class WeakValueInterner:
  def __init__(self, fn):
    self._fn = fn
    self._table: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    self._lock = threading.RLock()
    functools.update_wrapper(self, fn)

  def __call__(self, *args, **kwargs):
    key = (args, _kw_key(kwargs))
    with self._lock:
      v = self._table.get(key)
      if v is None:
        v = self._fn(*args, **kwargs)
        self._table[key] = v
      return v


def weak_value_interner(fn):
  return WeakValueInterner(fn)
