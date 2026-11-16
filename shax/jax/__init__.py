"""shax's imitation of the public `jax` namespace."""

# Marker checked by the test harness (and anything else that needs to know).
__shax__ = True

# The JAX release whose public API (and test suite) shax tracks.
__version__ = "0.12.0.1"


def default_backend() -> str:
  return "cpu"
