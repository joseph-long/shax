# The `jax` shim: importing it replaces sys.modules["jax"] with shax.jax and
# aliases every `jax.X` to `shax.jax.X` (see shax/as_jax.py).
__shax_shim__ = True  # tells shax.as_jax this half-imported `jax` is the shim
import shax.as_jax  # noqa: E402, F401
