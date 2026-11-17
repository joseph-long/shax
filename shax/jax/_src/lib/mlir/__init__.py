"""Stub of jaxlib's MLIR bindings (see _stub.py). shax never lowers to MLIR."""
from shax.jax._src.lib.mlir import _stub
from shax.jax._src.lib.mlir import ir, passmanager, dialects  # noqa: F401
_stub.install(__name__)
