"""Self-tests for the harness import aliasing (run in .venv, the shax env):

  PYTHONPATH=harness .venv/bin/python -m pytest -p no:cacheprovider harness/tests
"""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(code: str):
  prelude = textwrap.dedent(f"""
      import sys
      from pathlib import Path
      sys.path.insert(0, {str(ROOT / 'harness')!r})
      from shax_harness import aliases
      aliases.install('shax', Path({str(ROOT / 'vendor' / 'jax')!r}))
  """)
  return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)],
                        cwd=ROOT, capture_output=True, text=True)


def test_jax_is_shax():
  r = run("""
      import jax, shax.jax
      assert jax is shax.jax and jax.__shax__
      assert jax.__spec__.name == 'shax.jax', jax.__spec__
  """)
  assert r.returncode == 0, r.stderr


def test_upstream_pure_module_loads_from_vendor():
  r = run("""
      from jax._src import test_warning_util as t
      assert t.__file__.endswith('vendor/jax/jax/_src/test_warning_util.py'), t.__file__
  """)
  assert r.returncode == 0, r.stderr


def test_missing_stand_in_is_reported():
  r = run("""
      from shax_harness.aliases import HarnessImportError
      try:
        import jax._src.definitely_not_provided
      except HarnessImportError as e:
        assert 'shax does not provide' in str(e)
      else:
        raise AssertionError('expected HarnessImportError')
  """)
  assert r.returncode == 0, r.stderr


def test_import_forms_agree():
  r = run("""
      import jax._src.test_warning_util as a
      from jax._src import test_warning_util as b
      import sys
      assert a is b is sys.modules['shax.jax._src.test_warning_util']
  """)
  assert r.returncode == 0, r.stderr


def test_opt_einsum_is_vendored_copy():
  r = run("""
      import opt_einsum, shax._vendor.opt_einsum as vendored
      from opt_einsum import contract
      assert opt_einsum is vendored, opt_einsum
      assert contract is vendored.contract
      from jax._src import lax_reference  # imports opt_einsum
  """)
  assert r.returncode == 0, r.stderr
