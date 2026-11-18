"""PRNG primitives inside a jit, the way XLA runs them.

Their lowerings (prng.py `random_*_lowering`) are `mlir.lower_fun` of the
key implementation's function on the *physical* key data: XLA compiles that
once and runs it. Their Python `impl` rules instead run the same function
eagerly, dispatching one jitted op per jnp call (visible as
`jit_cpp_cache_miss` events, and slow). So inside compiled evaluation shax
traces the physical function once per signature and evaluates the jaxpr.
"""
from __future__ import annotations

from shax.jax._src.random import prng
from shax.jax._src.shax_backend import _compiled_call, register_jit_impl


@register_jit_impl(prng.random_seed_p)
def _seed(seeds, *, impl):
  base = _compiled_call(("seed", impl),
                        lambda s: prng.random_seed_impl_base(s, impl=impl),
                        seeds)
  return prng.PRNGKeyArray(impl, base)


@register_jit_impl(prng.random_split_p)
def _split(keys, *, shape):
  impl, ndim = keys._impl, keys.ndim
  base = _compiled_call(
      ("split", impl, ndim, shape),
      lambda b: prng.random_split_impl_base(impl, b, ndim, shape=shape),
      keys._base_array)
  return prng.PRNGKeyArray(impl, base)


@register_jit_impl(prng.random_fold_in_p)
def _fold_in(keys, msgs):
  impl, kshape = keys._impl, keys.shape
  base = _compiled_call(
      ("fold_in", impl, kshape),
      lambda b, m: prng.random_fold_in_impl_base(impl, b, m, kshape),
      keys._base_array, msgs)
  return prng.PRNGKeyArray(impl, base)


@register_jit_impl(prng.random_bits_p)
def _bits(keys, *, bit_width, shape):
  impl, ndim = keys._impl, keys.ndim
  return _compiled_call(
      ("bits", impl, ndim, bit_width, shape),
      lambda b: prng.random_bits_impl_base(impl, b, ndim,
                                           bit_width=bit_width, shape=shape),
      keys._base_array)
