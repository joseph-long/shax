"""Regression tests for jaxlib stand-ins whose bugs gave silently wrong results.

- PjitFunction tables: pjit.cc gives each function-less (AOT, `Compiled`)
  PjitFunction a private table (DefaultCache()). Sharing one let every
  `jit(f).lower(x).compile()(y)` with the same signature run the first
  executable ever compiled.
- ArrayImpl._replace_with: how a jit writes updated `Ref` values back. When it
  was missing, upstream's `except AttributeError` dropped every write.
- WeakKeyWeakValueCache: jaxlib canonicalizes a NumPy array to the same
  TypedNdArray while both live; tracing dedups constants by `id()` on that.
"""
import gc

import numpy as np

from shax import jax
import shax.jax.numpy as jnp
from shax.jax._src.lib._jax import WeakKeyWeakValueCache


def test_aot_compiled_functions_do_not_share_executables():
  for k in range(4):
    def f(x, k=k):
      return x + k
    assert float(jax.jit(f).lower(1.).compile()(4.)) == 4. + k
    del f
    gc.collect()


def test_jit_writes_back_refs():
  closed = jax.new_ref(jnp.zeros(3, jnp.float32))

  def f():
    closed[...] = jnp.ones(3, jnp.float32)

  jax.jit(f)()
  np.testing.assert_array_equal(closed[...], np.ones(3, np.float32))

  arg = jax.new_ref(jnp.zeros(3, jnp.float32))

  @jax.jit
  def g(r):
    r[...] = r[...] + 2

  g(arg)
  g(arg)  # the second call takes the cached fast path
  np.testing.assert_array_equal(arg[...], np.full(3, 4, np.float32))


def test_weak_key_weak_value_cache():
  calls = []

  def fn(x):
    calls.append(x)
    return np.array(x)  # a new object each call

  cache = WeakKeyWeakValueCache(fn)
  a = np.arange(3)
  v1 = cache(a)
  assert cache(a) is v1 and len(calls) == 1  # same key, value alive: memoized
  b = np.arange(3)
  assert cache(b) is not v1  # keyed by identity, not equality
  del v1
  gc.collect()
  cache(a)  # value died: recomputed
  assert len(calls) == 3


def test_make_jaxpr_dedups_numpy_constants():
  c = np.ones(3, np.float32)
  jaxpr = jax.make_jaxpr(lambda: (jnp.array(c), jnp.sum(c), c))()
  assert len(jaxpr.consts) == 1
