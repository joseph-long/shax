"""`shax.as_jax`: `import jax` gives shax (what the jax shim wheel does)."""
import subprocess
import sys
import textwrap


def run(code: str):
  return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                        capture_output=True, text=True)


def test_import_jax_is_shax_and_shares_modules():
  r = run("""
      import shax.as_jax
      import jax, jax.numpy as jnp
      from jax._src import core
      import shax.jax, shax.jax.numpy, shax.jax._src.core
      assert jax is shax.jax and jnp is shax.jax.numpy
      assert core is shax.jax._src.core
      assert float(jax.grad(lambda x: x ** 2)(3.)) == 6.
      import importlib; importlib.reload(jnp)  # identity kept (__spec__ restored)
  """)
  assert r.returncode == 0, r.stderr


def test_missing_submodule_is_module_not_found():
  r = run("""
      import shax.as_jax
      try:
        import jax.no_such_module
      except ModuleNotFoundError:
        pass
      else:
        raise AssertionError("expected ModuleNotFoundError")
  """)
  assert r.returncode == 0, r.stderr


def test_refuses_when_real_jax_already_imported():
  r = run("""
      import sys, types
      sys.modules["jax"] = types.ModuleType("jax")  # stands in for real jax
      try:
        import shax.as_jax
      except RuntimeError as e:
        assert "real `jax`" in str(e)
      else:
        raise AssertionError("expected RuntimeError")
  """)
  assert r.returncode == 0, r.stderr
