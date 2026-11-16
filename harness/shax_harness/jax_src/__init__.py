"""Stand-ins for `jax._src.*` modules imported by the JAX test suite.

Under the `shax` harness target, `jax._src.<name>` resolves to
`shax_harness.jax_src.<name>`. Each stand-in should mirror the signatures of
the real `jax._src.<name>` names that tests (and shax_harness.test_util) use,
implemented on top of shax. A missing stand-in makes the importing test file
fail at collection, which the harness reports.
"""
