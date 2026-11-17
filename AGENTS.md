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
- **Architecture: port upstream's pure-Python layers, replace the backend**
  (decided 2026-11-16). JAX's tracing/transformation code (`core`,
  `interpreters/{ad,batching,partial_eval}`, `lax` op definitions, `numpy`,
  `tree_util`, `dtypes`, `config`, ...) is pure Python and nearly free of XLA
  (core.py: 2 refs; ad/batching: 0). shax ports these with
  `scripts/port-upstream.py` (imports rewritten to `shax.jax`, provenance
  header, hand edits marked `# SHAX:`) and keeps them close to upstream. shax's
  own code is the **backend**:
  - `shax/jax/_src/lib/`: pure-Python stand-ins for jaxlib C++ (`_jax.config`,
    `pytree`, `weakref_lru_cache`, `utils`, `canonicalize_value`, ...), matching
    jaxlib behaviour observable from Python, checked differentially against
    real jaxlib in `.venv-jaxref` (tests in `tests/lib/`).
  - MLIR/HLO lowering: stubbed so `register_lowering` calls are accepted and
    ignored; nothing is ever lowered.
  - Execution: an immutable NumPy-backed `Array`; a NumPy impl rule per `lax`
    primitive; `jit`/`pjit` = trace to a jaxpr, cache, evaluate the jaxpr with
    the NumPy impls.
  When porting, prefer adding a stand-in under `lib/` or a small `# SHAX:` edit
  over restructuring upstream code. Don't port modules nobody imports.
  `vendor/jax/docs/autodidax.md` is still a good map of how the pieces fit.
- **Licensing:** ported files are Apache-2.0 (JAX's license, copied to
  `shax/jax/_src/LICENSE-JAX`); keep upstream copyright headers.
- **Arrays are immutable, JAX-style.** `shax` `Array` wraps a read-only NumPy
  array; no in-place mutation; updates go through `x.at[idx].set/add/mul/min/max/
  get/apply` with JAX's scatter/gather semantics (out-of-bounds gather clamps by
  default, out-of-bounds scatter updates are dropped, `mode=` options, duplicate
  indices accumulate for `add`, unspecified order for `set`).
- **JAX test source is pinned by a script:** `scripts/fetch-jax.sh` clones
  `jax-v0.12.0.1` into `vendor/jax`. No submodule.

## Pyodide target (checked 2026-11-16, Pyodide 314.0.8 released 2026-11-13)

Found via `https://cdn.jsdelivr.net/pyodide/v<ver>/full/pyodide-lock.json`.
- Python **3.14**; bundled compiled wheels: numpy **2.4.6**, scipy **1.18.0**,
  **ml_dtypes 0.5.4** (JAX requires `ml_dtypes>=0.5.0`). So shax uses `ml_dtypes`
  as-is for bfloat16/float8/int4 etc. No shim is needed.
- **Not bundled**, all pure Python and fetchable by micropip from PyPI:
  - `opt_einsum`: upstream `jnp.einsum` imports it, so shax currently depends on
    it. Replacing it with a small `np.einsum_path`-based shim is an open TODO,
    because offline WASM exports can't fetch from PyPI.
  - `absl-py`, `hypothesis`: only needed to run the test suite in Pyodide.
- Bundled: `pytest`, `mpmath`, `cloudpickle`.
- `.venv` mirrors these versions (Python 3.14, numpy/scipy/ml_dtypes pins in
  `scripts/setup-envs.sh`) so local runs match the deployment target. Don't use
  NumPy features newer than 2.4.

## Layout

```
shax/                        # the shim; `from shax import jax` / `shax.jax.numpy`
  jax/                       #   ported public `jax` package (`__init__` sets `__shax__ = True`)
    _src/                    #   ported `jax._src` modules (same names as upstream)
      LICENSE-JAX            #     Apache-2.0 license of the ported code
      lib/                   #     pure-Python stand-ins for jaxlib (see "shax internals")
      shax_backend/          #     shax's NumPy execution backend (shax-original code)
tests/lib/                   # differential tests of lib/ stand-ins vs real jaxlib (+ fixtures)
harness/
  shax_harness/
    plugin.py                # pytest plugin: installs aliasing in pytest_configure; writes JSON report
    aliases.py               # sys.meta_path finder (see "How the harness works")
    test_util.py             # port of jax/_src/test_util.py; changes marked `# SHAX:`
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
  port-upstream.py           # copy an upstream module into shax/ with import rewrites
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
scripts/jaxtest accept .jaxtest/shax-core.json --subset core  # reviewed report -> baseline
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
  `jax._src.test_util` → our port), `shax` (`jax`/`jax.*`, including `jax._src.*`,
  → `shax.jax.*`; test-only modules in `TEST_ONLY` in aliases.py are injected as
  `shax.jax._src.<name>`: our `test_util` port, plus upstream's own
  `test_loader`/`test_warning_util` loaded from vendor/jax).
- **Invariant:** `jax-ported` must match `jax-upstream` exactly on every subset we
  use (`scripts/jaxtest compare ...`). Check it after any change to `test_util.py`.
  Smoke: 1573 passed / 57 skipped on both, 0 differences.
- The port's shax-only branches (`if _SHAX:`) never run under real JAX, so keep
  them small: public-API equivalents, or `_shax_unsupported(...)`, which raises
  `SkipTest("shax-unsupported: ...")` so those skips are visible in reports.
- `from jax._src import X` resolves X via the parent module's *real* name
  (`shax.jax._src.X`), not via the `jax._src.X` alias, so the finder matches
  test-only modules under both names.
- Never run pytest/python with cwd inside `vendor/jax`: the source tree `jax/`
  shadows the installed one, and `vendor/jax/jax/_src/typing.py` etc. shadow the
  stdlib. The runner uses cwd = repo root; the plugin refuses a vendored `jax`.
- `harness/pytest.ini` keeps upstream's `filterwarnings = error`, so shax must not
  leak NumPy `RuntimeWarning`s (overflow, invalid, divide) that real JAX wouldn't
  emit; use `np.errstate` inside shax.
- `--noconftest` is on: vendor/jax/conftest.py (doctest namespace + accelerator
  xdist env vars) is not needed.
- Event counters: upstream test_util counts JAX events through
  `util.test_event_listener(name)` (`"trace_to_jaxpr"`, `"jit_cpp_cache_miss"`,
  `"lower_jaxpr_to_module"`, ...). shax should call that hook with the same names
  where it has a matching concept (tracing, jit cache misses) so cache tests work.

## shax internals

- **Porting:** `scripts/port-upstream.py jax/_src/foo.py` writes
  `shax/jax/_src/foo.py` with imports rewritten to `shax.jax` and a provenance
  header; `--diff` shows hand edits vs the mechanical port. Hand edits are marked
  `# SHAX:` and should be rare and small. Most upstream modules are ported
  unmodified. Missing modules show up as `ModuleNotFoundError: shax.jax._src.X`:
  port them.
- **jaxlib's C++ source is vendored** at `vendor/jax/jaxlib/*.cc` (e.g.
  `pytree.cc`, `py_values.cc`, `pjit.cc`, `jax_jit.cc`, `sharding.cc`,
  `py_array.cc`, `_pretty_printer.cc`). Use it as the spec when writing a
  `lib/` stand-in, and check behaviour against real jaxlib in `.venv-jaxref`.
- **`shax/jax/_src/lib/`** mirrors `jax._src.lib`, as pure Python:
  - `_jax.py`: config state, canonicalize_value, partition-spec helpers, and the
    tracer-class hook; re-exports `Traceback`, the jit dispatch pieces, and
    placeholders.
  - `pytree.py`, `_pretty_printer.py`, `weakref_lru_cache.py`, `utils.py`.
  - `xla_client.py`: `Device`, the `Client`, `DeviceList`, `HloSharding`,
    `Traceback`, `ArrayMeta`/`Array`, and lazily-built `ArrayImpl` / sharding
    classes from `_array.py` / `_sharding.py`.
  - `_pjit.py`: `PjitFunction`, `parse_arguments`, `shard_args`,
    `batched_device_put`.
  - `mlir/`: permissive stubs.
  - `_unsupported.py`: helpers for things shax will never do (profiler, plugins,
    distributed, FFI), which raise only when called.

  Upstream merges Python classes onto C++ ones with `use_cpp_class`; shax's
  stand-ins are Python "C++ classes" used the same way (e.g. `ArrayImpl`,
  `NamedSharding`).
- **`shax/jax/_src/shax_backend/`** is the execution backend.
  - `dispatch.apply_primitive` → `shax_backend.apply_primitive` (eager ops).
  - **jit/AOT (`shax_backend/aot.py`)**: jit goes through upstream's real
    lower → compile → execute pipeline (`stages.Lowered`/`Compiled`,
    `pxla.MeshExecutable`), so lowering/compile events, caches and
    `.lower().compile()` behave as in JAX. Three `# SHAX:` edits swap the
    bottom layer: `mlir.lower_jaxpr_to_module` returns a `ShaxModule`
    (holds the jaxpr; asking for HLO/StableHLO text raises
    `NotImplementedError`), `pxla._cached_compilation` returns a
    `ShaxExecutable`, and `ShaxExecute` (an `ExecuteReplicated`) runs it
    with `shax_backend.run_jaxpr`.
  - Primitive impls are NumPy functions registered with `@register(prim)` in
    the modules listed in `_IMPL_MODULES`. Results are cast and checked against
    `prim.abstract_eval`; `raw=True` impls get JAX values (for higher-order
    primitives).
  - `jit` traces once (upstream's Python caches), then evaluates the cached jaxpr
    eagerly with `eval_jaxpr_compiled`. That works like compiled code: a
    first-order primitive with a NumPy impl runs it directly, never the
    primitive's Python `impl` rule (tests swap those out to count calls).
  - `lib/_pjit.py` `PjitFunction` emulates pjit.cc's C++ fast path: the call
    signature (parse_arguments + per-arg dtype/shape/weak_type/sharding/
    committed + `config.trace_context()`) keys a per-function table. A miss
    runs Python `cache_miss` and stores its fastpath data. A hit runs the
    cached executable directly (`_run_fastpath`), so no `jit_cpp_cache_miss`
    event fires, as in JAX (tests count those). Fast-path args/outputs are
    flattened with the *dispatch* registry: PRNG keys appear as their
    physical uint32 arrays, and `in_shardings` is indexed by kept (post-DCE)
    position. Donation, nan checks, `no_execution` and anything unexpected
    fall back to `cache_miss`.
  - `evaluating_jit()` is true inside `run_jaxpr`. Use it to scope
    eager-only restrictions to truly eager calls; inside a jit, real JAX runs
    compiled code.
  - shax has one device, so every mesh has size 1. Eager `shard_map`/`pmap`
    (`_shax_size1_shard_map_impl` in shard_map.py) traces the body like
    staging does (same rank/vma errors), then evaluates it on the global
    values. Upstream's `ShardMapTrace` path would recurse, because it relies
    on compiling `jit(shard_map(prim))`. Collectives over size-1 axes
    (`collectives.py`) override upstream impls only inside that body.
    Outside it, the upstream "must be under shard_map" errors stand.
  - Result conventions, as in jax's result handlers: float0 outputs are
    `TypedNdArray` zeros, not device arrays. Tokens are `core.Token` around
    a placeholder buffer (`new_token()`), which refuses NumPy conversion.
  - The backend's lazy `_load()` is thread-safe; keep it that way.
- **Arrays:** `ArrayImpl` holds one read-only NumPy array on the single
  `CpuDevice(id=0)`; inputs are copied, never frozen in place.
  `_fully_replicated_shard()`/`_arrays` give a single-device view of the same
  buffer.
- **`lib/weakref_lru_cache.py`** must evict entries when the weak key dies,
  and must call `explain` on misses like jaxlib. `MultiWeakrefLRUCache` holds
  `weak_types` leaves (and callables) weakly. A cached *value* must never
  strongly reference its own weak key (e.g. a lowering result holding the
  jaxpr), or the entry never dies and the jaxpr's const arrays leak
  (`jax.live_arrays()`, lax_control_flow_test::test_cond_memory_leak). Otherwise cached jaxprs that
  hold outer tracers trip the leak checker, and `explain_cache_misses` logs
  nothing (tests in `tests/lib/test_weakref_lru_cache.py`).
- **Debugging tip:** set `JAX_TRACEBACK_FILTERING=off`, or JAX hides shax's
  internal frames from tracebacks.
- **Can't support:** user-defined primitives that only have an MLIR lowering
  and `impl = dispatch.apply_primitive` (e.g. api_test `tokentest`). shax
  can't run lowering rules.

## Phases

0. Harness: **done**. Runner, aliasing, and the ported test_util, validated
   against upstream on `smoke`.
1. `smoke` + `core` subsets: **baselined 2026-11-17**
   (`harness/baselines/shax/{smoke,core}.json`).
   - smoke: 1555 passed / 1 failed / 47 skipped.
   - core (38 files): 19701 passed / 102 failed / 877 skipped / 0 collection
     errors. Most of what's left needs HLO/StableHLO text (~50 tests, which
     could go to skips.toml as "needs HLO text"), user primitives with only
     an MLIR lowering, and a few cache/event-count tests.
   Phases 2–4 below were largely absorbed into `core`; the remaining work is
   the leftover core failures, the out-of-core files, and Pyodide.
2. `numpy` subset; immutable Array + `.at[]` scatter/gather.
3. Transformations: jvp/linearize/vjp/grad, vmap, control flow, custom_jvp/vjp
   (`api_test`, `lax_autodiff_test`, `lax_control_flow_test`, `custom_api_test`).
4. `lax`, `random` (threefry bit-exact), `linalg`, `scipy`, `nn`; then the same
   harness inside Pyodide.

## Environment (Flatiron workstation)

- **Do not use the Lmod `module` system for anything in this project** (user's
  instruction). Use `uv` from `~/.local/bin/uv` (0.13.8 as of 2026-11-16) with
  uv-managed Pythons only. This overrides the module/venv advice in the
  cluster-wide instructions.
- Build envs with `scripts/setup-envs.sh`. Both venvs use uv-managed **CPython
  3.14** (Pyodide's version; currently 3.14.10). The script refuses pre-release
  interpreters, because numpy 2.4 wheels fail to load in 3.14 alphas.
- Always run JAX tests with `JAX_PLATFORMS=cpu` in the reference env (the machine has a GPU).
- Use `pytest -n 8` or so for big files; there are 32 cores but the machine is shared.
- **Network:** `curl` is blocked in this environment; Python `requests` works
  (system `python3`, not the venvs). Use e.g.
  `python3 -I -c 'import requests; print(requests.get(URL, timeout=30).text)'`.
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
  `harness/shax_harness/test_util.py` keeps the public surface tests use and drops/stubs those.
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
