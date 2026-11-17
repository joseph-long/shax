"""Compute backends for opt_einsum."""

# Backends
from shax._vendor.opt_einsum.backends.cupy import to_cupy
from shax._vendor.opt_einsum.backends.dispatch import (
    build_expression,
    evaluate_constants,
    get_func,
    has_backend,
    has_einsum,
    has_tensordot,
)
from shax._vendor.opt_einsum.backends.tensorflow import to_tensorflow
from shax._vendor.opt_einsum.backends.theano import to_theano
from shax._vendor.opt_einsum.backends.torch import to_torch

__all__ = [
    "get_func",
    "has_einsum",
    "has_tensordot",
    "build_expression",
    "evaluate_constants",
    "has_backend",
    "to_tensorflow",
    "to_theano",
    "to_cupy",
    "to_torch",
]
