"""Tests for shax/jax/_src/lib/weakref_lru_cache.py.

jaxlib's WeakrefLRUCache drops entries when the weakly-referenced argument
dies. JAX relies on that: trace_to_jaxpr caches jaxprs keyed weakly on the
traced function, and those jaxprs can hold tracers of an enclosing trace as
consts. If the entry outlives the function, the tracer leak checker
(`jax.checking_leaks`, on in checkify tests) reports a leaked trace.
"""
import gc

from shax.jax._src.lib.weakref_lru_cache import weakref_lru_cache


class _Fn:
  def __call__(self):
    pass


def test_entries_evicted_when_weak_key_dies():
  cache = weakref_lru_cache(lambda: None, lambda f, x: x * 2)
  f = _Fn()
  assert cache(f, 1) == 2
  assert cache(f, 2) == 4
  assert cache(f, 2) == 4
  info = cache.cache_info()
  assert (info.hits, info.misses, info.currsize) == (1, 2, 2)
  del f
  gc.collect()
  assert cache.cache_info().currsize == 0


def test_entries_kept_per_weak_key():
  cache = weakref_lru_cache(lambda: None, lambda f, x: x)
  f, g = _Fn(), _Fn()
  cache(f, 1)
  cache(g, 1)
  del f
  gc.collect()
  assert cache.cache_info().currsize == 1
  assert cache(g, 1) == 1
  assert cache.cache_info().hits == 1
