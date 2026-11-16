# AGENTS.md — working notes for agents on `shax`

`shax` is a pure-Python + NumPy (+ SciPy where needed) imitation of JAX, so JAX
code can run where `jaxlib` can't (Pyodide/WASM, e.g. Marimo `html-wasm` exports).
Speed is a non-goal. Correctness is measured by running JAX's own test suite.

## Ground rules

- **shax must never import `jax` or `jaxlib`.** Runtime deps: Python stdlib, `numpy`,
  and (optionally, lazily) `scipy`. Anything else must also be available in Pyodide.
- **Never edit `vendor/jax/`.** It's a pristine upstream checkout used as the test
  oracle. Adapt in the harness (aliases, ported `test_util`, skip lists), not in
  the tests.
- **Pinned upstream:** JAX `jax-v0.12.0.1` (shallow clone at `vendor/jax`, gitignored).
  If the pin moves, re-run the reference baseline before trusting any comparison.
- **The real-JAX reference venv (`.venv-jaxref`) is the oracle for harness bugs.**
  If a test fails under shax, first check it passes with the same harness under
  real JAX. A test that fails under real JAX + our harness is a harness bug, not
  a shax bug.
- **Don't game the suite.** Don't special-case test names, sniff `PYTEST_CURRENT_TEST`,
  or weaken `assertAllClose` tolerances in the ported `test_util`. Skips belong in
  the harness skip list with a reason (e.g. "needs sharding", "needs HLO text").
- Prefer behaviour that matches JAX bit-for-bit where JAX specifies it (dtype
  promotion, x32 canonicalization, PRNG/threefry outputs, error types/messages
  that tests assert on). Approximate where JAX itself is approximate.
- Commit only when the user asks.

## Design decisions (agreed with the user)

- **Real tracing and transformations are in scope:** `jit`, `grad`/`vjp`/`jvp`/
  `linearize`, `vmap`, control flow (`lax.cond`/`scan`/`while_loop`/`fori_loop`),
  `custom_jvp`/`custom_vjp`. The public API must behave like JAX; internals may
  diverge where that's simpler (no XLA, no MLIR, no levels if not needed).
- **Architecture reference:** `vendor/jax/docs/autodidax.md` (primitives + interpreters
  for eval/jvp/vmap, jaxprs, `jit` as staging, `linearize`/`vjp` via partial eval +
  transpose, `cond`) and `vendor/jax/docs/autodidax2_part1.md` (simpler
  "current interpreter" context design). Expected shape: a set of `lax`-level
  primitives with NumPy impl rules; `jax.numpy` written on top of `lax`, as in
  JAX, so every `jnp` function is automatically traceable/differentiable/batchable.
  `jit` traces to a jaxpr, caches on (pytree structure, avals, static args), and
  evaluates the jaxpr with NumPy.
- **Arrays are immutable, JAX-style.** `shax` `Array` wraps a read-only NumPy
  array; no in-place mutation; updates go through `x.at[idx].set/add/mul/min/max/
  get/apply` with JAX's scatter/gather semantics (out-of-bounds gather clamps by
  default, out-of-bounds scatter updates are dropped, `mode=` options, duplicate
  indices accumulate for `add`, unspecified order for `set`).
- **JAX test source is pinned by a script:** `scripts/fetch-jax.sh` clones
  `jax-v0.12.0.1` into `vendor/jax`. No submodule.

## Layout

```
shax/                        # the shim; `from shax import jax` / `shax.jax.numpy`
  jax/__init__.py            #   sets `__shax__ = True` (harness checks it) and `__version__` = emulated JAX
harness/
  shax_harness/
    plugin.py                # pytest plugin: installs aliasing in pytest_configure; writes JSON report
    aliases.py               # sys.meta_path finder (see "How the harness works")
    test_util.py             # port of jax/_src/test_util.py; changes marked `# SHAX:`
    jax_src/                 # stand-ins for jax._src.<x> under the shax target (mostly TODO)
  tests/test_aliases.py      # harness self-tests (run in .venv)
  pytest.ini                 # mirrors vendor/jax/pyproject.toml pytest options
  requirements-test.txt      # test deps for both venvs
  subsets/<name>.txt         # named selections: paths / node ids relative to vendor/jax
  skips.toml                 # node-id-prefix skips with reasons, per target
  baselines/<target>/<subset>.json   # accepted outcome per node id (the ratchet; commit these)
scripts/
  fetch-jax.sh               # clone vendor/jax at the pinned tag
  setup-envs.sh              # create .venv (shax, no jax) and .venv-jaxref (real jax)
  jaxtest                    # runner: run / compare / summary
vendor/jax/                  # upstream checkout (gitignored)
.jaxtest/                    # run reports (gitignored)
```

## Running tests

```
scripts/fetch-jax.sh && scripts/setup-envs.sh          # once
scripts/jaxtest run --subset smoke                     # shax target (default), -n 8
scripts/jaxtest run --target jax-ported --subset smoke # real JAX + our ported test_util
scripts/jaxtest run tests/fft_test.py -- -k testFftn   # ad hoc; after `--` goes to pytest
scripts/jaxtest run --subset smoke --update-baseline   # accept current results (ratchet)
scripts/jaxtest compare .jaxtest/jax-upstream-smoke.json .jaxtest/jax-ported-smoke.json
PYTHONPATH=harness .venv/bin/python -m pytest -p no:cacheprovider harness/tests
```

`run` prints per-file passed/failed/error/skipped/collection-error counts, the
most common failure messages and skip reasons, and regressions vs the baseline
(exit status 1 if any). Defaults: `JAX_NUM_GENERATED_CASES=10`,
`JAX_SKIP_SLOW_TESTS=1` (`--slow` to include), `--x64` for `JAX_ENABLE_X64=1`,
`-j 1` to disable xdist (needed for `pdb`/`-s`).

## How the harness works (and its gotchas)

- Targets: `jax-upstream` (real JAX, untouched), `jax-ported` (real JAX, but
  `jax._src.test_util` → our port), `shax` (`jax`/`jax.*` → `shax.jax.*`,
  `jax._src.*` → `shax_harness.jax_src.*`, `jax._src.test_util` → our port).
- **Invariant:** `jax-ported` must match `jax-upstream` exactly on every subset we
  use (`scripts/jaxtest compare ...`). Check it after any change to `test_util.py`.
  Smoke: 1573 passed / 57 skipped on both, 0 differences.
- The port's shax-only branches (`if _SHAX:`) never run under real JAX, so keep
  them small: public-API equivalents, or `_shax_unsupported(...)`, which raises
  `SkipTest("shax-unsupported: ...")` so those skips are visible in reports.
- `from jax._src import X` resolves X via the parent module's *real* name
  (`shax_harness.jax_src.X`), not via the `jax._src.X` alias. So stand-ins must
  live at `shax_harness/jax_src/X.py`, and upstream pure-Python modules
  (`UPSTREAM_PURE` in aliases.py: test_loader, test_warning_util, deprecations,
  public_test_util) are loaded under the stand-in name.
- Never run pytest/python with cwd inside `vendor/jax`: the source tree `jax/`
  shadows the installed one, and `vendor/jax/jax/_src/typing.py` etc. shadow the
  stdlib. The runner uses cwd = repo root; the plugin refuses a vendored `jax`.
- `harness/pytest.ini` keeps upstream's `filterwarnings = error`, so shax must not
  leak NumPy `RuntimeWarning`s (overflow, invalid, divide) that real JAX wouldn't
  emit; use `np.errstate` inside shax.
- `--noconftest` is on: vendor/jax/conftest.py (doctest namespace + accelerator
  xdist env vars) is not needed.
- Stand-ins in `jax_src/` should re-export from shax wherever shax has the real
  thing (e.g. `jax_src/dtypes.py` → `shax.jax._src.dtypes` or similar), and only
  hold test-only glue themselves.
- Event counters: upstream test_util counts JAX events through
  `util.test_event_listener(name)` (`"trace_to_jaxpr"`, `"jit_cpp_cache_miss"`,
  `"lower_jaxpr_to_module"`, ...). shax should call that hook with the same names
  where it has a matching concept (tracing, jit cache misses) so cache tests work.

## Phases

0. Harness — **done**: runner, aliasing, ported test_util validated against
   upstream on `smoke`, shax target reports collection errors.
1. `smoke` subset (tree_util, dtypes, einsum, fft): config, dtypes (incl.
   `ml_dtypes` types), tree_util, core Array + primitives + eval, `jit` tracing.
2. `numpy` subset; immutable Array + `.at[]` scatter/gather.
3. Transformations: jvp/linearize/vjp/grad, vmap, control flow, custom_jvp/vjp
   (`api_test`, `lax_autodiff_test`, `lax_control_flow_test`, `custom_api_test`).
4. `lax`, `random` (threefry bit-exact), `linalg`, `scipy`, `nn`; then the same
   harness inside Pyodide.

## Environment (Flatiron workstation)

- `module load uv` (python 3.13 via uv). Create venvs with `uv venv -p 3.13 <dir>`.
- Reference env: `uv pip install "jax[cpu]==0.12.0.1" absl-py hypothesis pytest pytest-xdist mpmath scipy cloudpickle filelock flatbuffers portpicker rich matplotlib`.
- Always run JAX tests with `JAX_PLATFORMS=cpu` in the reference env (the machine has a GPU).
- Use `pytest -n 8` or so for big files; there are 32 cores but the machine is shared.
- Shared-filesystem rules from `/etc/claude-code/CLAUDE.md` apply: any recursive
  `find`/`grep -r` must be scoped (e.g. `vendor/jax/tests`), use `timeout 120`, and a max depth.

## Facts about the JAX test suite (v0.12.0.1)

- ~175 entries in `vendor/jax/tests/`; ~155k lines of `*_test.py`. Tests are
  `absltest`/`parameterized` classes deriving from `jtu.JaxTestCase`, run fine under pytest.
- **Every test file imports private modules.** Counts of top-level test files:
  `jax._src.test_util as jtu` (163), `jax._src.config` (79), `core` (36), `jax._src.lib` (29),
  `interpreters` (29), `dtypes` (24), `lax` (22), `util` (18). So the harness must
  supply `jax._src.test_util`, `jax._src.config`, `jax._src.dtypes`, and a few small
  utilities; modules it can't supply make the file fail at collection — record that,
  don't paper over it.
- Upstream `test_util.py` (~2200 lines) imports XLA/MLIR internals (`xla_bridge`,
  `mlir`, `hlo`, `compilation_cache`, `cloud_tpu_init`, ...). The port in
  `harness/shax_harness/jtu.py` keeps the public surface tests use and drops/stubs those.
  Most-used `jtu` names: `sample_product`, `rand_*`, `dtypes`, `JaxTestCase`,
  `test_device_matches`, `run_on_devices`/`skip_on_devices`, `ignore_warning`,
  `check_grads`, `with_config`, `create_mesh`/`with_explicit_mesh` (sharding; skip).
- `JaxTestCase.setUp` enters config contexts: `jax_enable_checks=True`,
  `jax_numpy_dtype_promotion='strict'`, `jax_numpy_rank_promotion='raise'`,
  `jax_legacy_prng_key='error'`; RNG seeded from `adler32(test method name)`.
  shax's config must accept these.
- `_CompileAndCheck` asserts that the second call of a `jax.jit`-ed function does
  **not** re-run Python — shax's `jit` must really trace and cache by
  (shapes, dtypes, static args), not be an identity wrapper.
- `_CheckAgainstNumpy` / `assertAllClose` check dtypes too, so x32 defaults matter
  (`JAX_ENABLE_X64` unset → float32/int32 canonicalization).
- Knobs (env vars): `JAX_NUM_GENERATED_CASES` (default 10; cases sampled from
  `sample_product` deterministically), `JAX_SKIP_SLOW_TESTS`, `JAX_ENABLE_X64`,
  `JAX_TEST_DUT`.
- `vendor/jax/pyproject.toml` sets `filterwarnings = ["error", ...]` and
  `--import-mode=importlib`; our `harness/pytest.ini` should mirror those and
  set `rootdir` explicitly so the upstream config isn't picked up by accident.
- Upstream CPU CI: `pytest -n auto --tb=short --maxfail=20 tests examples` with
  `JAX_SKIP_SLOW_TESTS=true` (`ci/run_pytest_cpu.sh`).

Collected test counts under real JAX (default 10 generated cases):

| file | tests | | file | tests |
|---|---|---|---|---|
| lax_numpy_test | 3348 | | lax_test | 2614 |
| lax_numpy_operators_test | 1628 | | api_test | 1015 |
| lax_numpy_reducers_test | 1047 | | linalg_test | 1000 |
| lax_numpy_indexing_test | 390 | | dtypes_test | 865 |
| lax_numpy_ufuncs_test | 215 | | lax_autodiff_test | 745 |
| lax_numpy_einsum_test | 176 | | tree_util_test | 478 |
| lax_numpy_setops_test | 155 | | nn_test | 443 |
| fft_test | 111 | | random_test | 254 |
| lax_scipy_test | 121 | | | |

Out of scope (don't target): `pjit`, `shard_map`, `pmap`, `multi_device`, `memories`,
`layout`, `pallas/`, `mosaic*`, `export*`, `compilation_cache`, `ffi`, `profiler*`,
`multiprocess*`, `distributed*`, `cudnn*`/`fused_attention*`/GPU/TPU-specific files,
`xla_*`, `jax_to_ir`, `filecheck/`.

## Workflow

1. Pick a subset; run it with `--target jax-ported` first to confirm the harness is sound.
2. Run with `--target shax`; compare to the baseline; fix shax; repeat.
3. When results improve, update the baseline (the ratchet). Any test that goes
   pass → fail is a regression and must be fixed or explained before moving on.
4. Report pass/fail/skip/error-at-collection counts per file when summarizing progress.
