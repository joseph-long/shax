# `shax` - a shim for JAX code on unsupported platforms

```python
from shax import jax
from shax.jax import numpy as jnp
```

## Like JAX, but slower

The `jaxlib` dependency is tricky to package, and probably won't be packaged in Pyodide any time soon.

NumPy, however, is very well supported. Since `jax.numpy` imitates `numpy`'s API, we can use `numpy` to imitate JAX imitating NumPy.

## Validation

The portions of the JAX test suite that do not depend on real GPU or TPU hardware, or the existence of particular compiled JAX-specific libraries, can be run against shax by

TODO

## Usage

If you have a Marimo notebook and export with

```
marimo export html-wasm ./example_jax.py --offline -o ./example_jax.html
```

it should be able to run your JAX code on CPU with `shax`. How, exactly, is to be determined.

## Non-goals

There's no attempt here to make a performant JAX alternative. Just get browser/WASM execution using well-supported parts of the scientific software stack.
