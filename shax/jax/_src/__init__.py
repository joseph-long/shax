"""shax internals. Mirrors the layout of `jax._src` so JAX's tests (which import
private modules) and ported upstream code can find things by the same names.
Implementations are shax's own, built on NumPy; some modules are ports of
upstream JAX source (Apache-2.0, see LICENSE-JAX) and say so in their header.
"""
