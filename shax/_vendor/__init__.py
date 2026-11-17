"""Third-party pure-Python packages shax vendors so it works offline in Pyodide.

opt_einsum 3.4.0 (MIT, see opt_einsum/LICENSE): upstream `jnp.einsum` uses it
for contraction paths, and Pyodide doesn't bundle it. Taken unmodified from
the PyPI sdist (sha256 96ca72f1b886d148241348783498194c577fa30a8faac108586b14f1ba4473ac),
without its tests, except:
  - `opt_einsum` imports are rewritten to `shax._vendor.opt_einsum`;
  - edits marked `# SHAX:` make its `jax` backend use shax.jax, never real jax.
To update, re-copy from a new sdist and reapply those two edits.
"""
