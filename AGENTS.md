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
- **Licensing:** shax is Apache-2.0, matching JAX (`LICENSE`; decided
  2026-11-17). Ported files keep their upstream copyright headers, with
  JAX's license also at `shax/jax/_src/LICENSE-JAX`. Vendored opt_einsum
  stays MIT (`shax/_vendor/opt_einsum/LICENSE`); the package's SPDX
  expression is `Apache-2.0 AND MIT`.
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
  - `opt_einsum`: upstream `jnp.einsum` uses it, and offline WASM exports
    can't fetch from PyPI, so shax **vendors** opt_einsum 3.4.0 (MIT) in
    `shax/_vendor/opt_einsum`. `port-upstream.py` rewrites `opt_einsum`
    imports to it. Provenance and the two `# SHAX:` edits (its `jax` backend
    uses shax.jax) are in `shax/_vendor/__init__.py`. Nothing in shax may
    import a site-packages `opt_einsum`.
  - `absl-py`, `hypothesis`: only needed to run the test suite in Pyodide.
- Bundled: `pytest`, `mpmath`, `cloudpickle`.
- `.venv` mirrors these versions (Python 3.14, numpy/scipy/ml_dtypes pins in
  `scripts/setup-envs.sh`) so local runs match the deployment target. Don't use
  NumPy features newer than 2.4.

## Layout

```
shax/                        # the shim; `from shax import jax` / `shax.jax.numpy`
  _vendor/opt_einsum/        #   vendored opt_einsum 3.4.0 (MIT; see _vendor/__init__.py)
  jax/                       #   ported public `jax` package (`__init__` sets `__shax__ = True`)
    _src/                    #   ported `jax._src` modules (same names as upstream)
      LICENSE-JAX            #     Apache-2.0 license of the ported code
      lib/                   #     pure-Python stand-ins for jaxlib (see "shax internals")
      shax_backend/          #     shax's NumPy execution backend (shax-original code)
  as_jax.py                  #   `import shax.as_jax` makes `jax`/`jax.X` alias shax.jax(.X)
packaging/                   # `jax` / `jaxlib` shim wheels (+ build-wheels.sh -> dist/)
examples/optax-wasm/         # marimo notebook: jax.grad + optax, offline html-wasm export
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
  skips.toml                 # node-id-prefix (or `exact = true`) skips with reasons, per target
  baselines/<target>/<subset>[-x64][-slow].json  # accepted outcome per node id (the ratchet; commit these)
scripts/
  fetch-jax.sh               # clone vendor/jax at the pinned tag
  setup-envs.sh              # create .venv (shax, no jax) and .venv-jaxref (real jax)
  jaxtest                    # runner: run / compare / summary
  port-upstream.py           # port a module; --status; --rebase onto a new upstream tag
  fetch-pyodide.py           # pinned Pyodide runtime + wheels -> vendor/pyodide/<ver> (hash-checked)
  pyodide-pytest.mjs         # Node: run pytest inside Pyodide with the repo mounted
  check-wasm-export.py       # load a marimo html-wasm export offline in headless Chromium
vendor/jax/                  # upstream checkout (gitignored)
vendor/pyodide/              # Pyodide runtime + wheels (gitignored)
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
python3 -I scripts/fetch-pyodide.py && scripts/jaxtest run --target pyodide --subset smoke -j 4
PYTHONPATH=harness .venv/bin/python -m pytest -p no:cacheprovider harness/tests
```

`run` prints per-file passed/failed/error/skipped/collection-error counts, the
most common failure messages and skip reasons, and regressions vs the baseline
(exit status 1 if any). Defaults: `JAX_NUM_GENERATED_CASES=10`,
`JAX_SKIP_SLOW_TESTS=1` (`--slow` to include), `--x64` for `JAX_ENABLE_X64=1`,
`-j 1` to disable xdist (needed for `pdb`/`-s`). `--x64` and `--slow` runs
generate different cases, so each configuration has its own baseline
(`core-x64.json`, `core-slow.json`); `accept` picks it from the report's env.
A skip entry matches node ids by prefix; add `exact = true` when the test's
name is a prefix of other tests (e.g. `test_composite` vs `test_composite_grad`).

**Pyodide target** (`--target pyodide`): runs shax and the harness inside
Pyodide 314.0.8 under Node (`node` from the user's micromamba env is on PATH).
The repo is mounted at its real path, and shax is imported from source.
WASM has no subprocesses, so there's no xdist: `-j N` shards the test files
across N Node processes by file size and merges their reports
(`.jaxtest/pyodide-<label>.shardK.{json,log}`). Its baselines live under
`harness/baselines/pyodide/`. Differences from the native `shax` target are
Pyodide bugs to fix (or explain), since the code is the same. Inside Pyodide
the plugin also applies skips.toml entries with `targets = ["pyodide"]`
(platform limits: no threads; the wasm32 heap; NumPy on wasm32 having a
32-bit C `long` and returning garbage `frexp` exponents for inf/nan). Note
that `sample_product` case lists can differ on wasm32 (`np.long` is int32),
so a case index like `testFoo35` can name a different case there. Mind
memory in the backend: a full-size uint64 temporary per op is 8x a 1e8-element
bool draw, more than wasm32 can allocate.

## Goalpost: JAX code in an offline marimo WASM export (met 2026-11-17)

`examples/optax-wasm/notebook.py` (jit + value_and_grad + optax.adam)
exported with `marimo export html-wasm --offline` runs in headless Chromium
with all network blocked (`scripts/check-wasm-export.py`), with the same
result as natively. See `examples/optax-wasm/README.md`. Libraries that
`import jax` get shax through the `jax` shim wheel (`packaging/jax-shim`,
version `0.12.0.1+shax`, which runs `shax.as_jax`); the metadata-only
`jaxlib` shim satisfies their `jaxlib` requirement. marimo 0.25.2 uses
Pyodide **314.0.0** (numpy 2.4.3), older than the harness's 314.0.8.
optax 0.2.9 (adam, adamw + schedules + clipping, sgd/nesterov, lbfgs,
masked) and chex assertions match real JAX on CPU.

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
  - **Compiled evaluation is the hot path** (`eval_jaxpr_compiled`; an optax
    step is ~50 primitives). Inside it: first-order NumPy impls pass plain
    ndarrays between equations, typed from the jaxpr's `outvar.aval` (no
    per-op `abstract_eval` or ArrayImpl; wrapped lazily when something else
    reads them); nested `jit` eqns are inlined (as XLA does); primitives
    whose lowering is `lower_fun` of Python code (e.g. PRNG
    `random_bits`/`split`/`fold_in`/`seed`, `shax_backend/prng_jit.py`)
    are traced once per signature and evaluated, never run eagerly (eager
    ops would dispatch jits, firing `jit_cpp_cache_miss`). `add_any` has a
    raw-path-only impl. Profile with an optax loop before adding per-op
    work here.
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
    fall back to `cache_miss`. `disable_jit` calls the Python function
    first, as pjit.cc does. Function-less (AOT `Compiled`) PjitFunctions get
    a private table (pjit.cc `DefaultCache()`); sharing one ran the wrong
    executable. Never key a cache on `id()` of an object that can die.
    `_arg_signature` mirrors py_values.cc, including the `literals.Typed*`
    types: a value it doesn't recognise silently disables the fast path.
  - **Lowering-time errors** (`aot._check_lowering_rules`): JAX fails at
    lowering when a primitive has no MLIR rule for the platform, and some
    lowering rules raise (e.g. checkify's unfunctionalized `check`). shax
    reproduces both at "lowering" without running the rules. It checks the
    top level and nested jit/closed_call bodies only, because
    cond/scan/while lowering discharges Ref effects first.
  - **Prefer upstream's expansions to hand-written NumPy:** if a primitive's
    CPU lowering is `mlir.lower_fun(impl)`, leaving it without a NumPy impl
    makes the backend evaluate that lax expansion, which is what XLA runs,
    boundary cases included (`regularized_incomplete_beta_p`: a SciPy impl
    got NaN where JAX defines 0/1).
  - Jit outputs are placed by their aval's `memory_space` (e.g. a
    `device_put` to `MemorySpace.Host` yields `memory_kind="pinned_host"`).
  - Ref writes inside a jit come back through `ArrayImpl._replace_with`
    (py_array.cc ReplaceWithAlias). Upstream's write-back swallows
    AttributeError, so a missing method there drops writes silently.
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
    Body arguments lose their mesh sharding (`collectives._unmeshed`): a
    value sharded on an Explicit mesh (e.g. a remat residual) can't enter
    the manual-mesh body, and on a size-1 mesh the shard is the value.
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
- **`lib/_jax.py` canonicalization:** NumPy arrays canonicalize through a
  `WeakKeyWeakValueCache` (as in jaxlib), so the same array yields the same
  TypedNdArray while both live; tracing dedups constants by `id()`.
- **`erf_inv`** (every `jax.random.normal`) uses XLA's float32 Giles
  polynomial (<= 2 ulp from JAX), not SciPy; f64 uses SciPy if present,
  else Newton steps. Keep common ops SciPy-free: in a WASM export SciPy is
  a large extra download.
- **XLA quirks worth matching:** a bool holding a non-0/1 byte converts to
  unsigned <=8-bit types by keeping the byte (masked), which `arr.view()`
  relies on; to anything else it converts to 0/1. XLA:CPU flushes float
  denormals in arithmetic (`_ftz`), including scatter-add/sub/mul combiners
  (only the touched elements).
- **jaxlib C++ is invisible in tracebacks:** `traceback_util` excludes the
  whole `lib/` and `shax_backend/` directories, and `PjitFunction.__call__`
  strips `lib/` frames from exceptions passing through it (jaxlib's C++ call
  adds none; errors_test checks the filtered frames).
- **Loggers keep upstream names:** ported modules log to
  `logging.getLogger(__name__.removeprefix("shax."))` (a port rule), so
  `jax_debug_log_modules` and handlers on the `jax` logger work.
- **jaxlib behaviours emulated in `lib/`:** transfer guards
  (`_jax.apply_transfer_guard`, checked where host values are put on the
  device: `batched_device_put`, the jit fast path, eager primitives),
  the array GC guard (`ArrayImpl.__del__` during a gc pass), creation
  tracebacks on arrays/executables, the buffer protocol (`__buffer__`),
  DLPack via NumPy (ml_dtypes types such as bfloat16/float8 travel as
  same-width uints, with the capsule's DLDataType code retagged via ctypes
  as jaxlib tags them; `_jax._dl_dtype`), FFI
  `Buffer`s for `buffer_callback`, and jaxlib's `safe_map`/`safe_zip` errors.
- **Pallas** (`jax.experimental.pallas`) is ported; `interpret=True`
  kernels run (the CPU lowering's HLO interpreter, via `register_jit_impl`).
  Mosaic/Triton lowering is unavailable, as on CPU JAX.
- **Eager higher-order primitives with Ref arguments** (e.g. `scan` over a
  `new_ref`) run as a jit, as upstream's dispatch does, so Ref writes are
  discharged and written back.
- **Debugging tip:** set `JAX_TRACEBACK_FILTERING=off`, or JAX hides shax's
  internal frames from tracebacks.
- **Can't support** (skipped in `harness/skips.toml`, with reasons):
  - anything that reads HLO/StableHLO text or MLIR objects, or uses
    `jax.export`;
  - user-defined primitives that only have an MLIR lowering and
    `impl = dispatch.apply_primitive` (e.g. api_test `tokentest`), since
    shax can't run lowering rules;
  - tests that count Python stack frames, because jaxlib's C++ adds none;
  - tests that run `import jax` in a subprocess.
  `jax.extend` is ported, but its `mlir`, `pallas`, `xla` and
  `ifrt_programs` parts raise `NotImplementedError` when used
  (`_jax._UnsupportedNamespace`).

## Phases

0. Harness: **done**. Runner, aliasing, and the ported test_util, validated
   against upstream on `smoke`.
1. `smoke` + `core` subsets: **done 2026-11-17**, 0 failures
   (`harness/baselines/shax/{smoke,core,core-x64,core-slow}.json`).
   - core `--slow`: 19786 passed / 0 failed / 894 skipped.
   - smoke: 1555 passed / 0 failed / 48 skipped.
   - core (38 files): 19728 passed / 0 failed / 952 skipped / 0 collection
     errors. Skips with harness reasons: HLO/StableHLO text, MLIR-only
     user primitives, XLA-only features, stack-frame counting.
   - core `--x64`: 20363 passed / 1 failed / 976 skipped. The failure,
     scipy_optimize_test `TestBFGS::test_fixes4594`, is badly conditioned
     BFGS whose outcome hinges on XLA's dot rounding (JAX's own `H @ g`
     differs per row for a symmetric H and equal g, i.e. FMA order); shax
     doesn't emulate Eigen's kernel order.
   - Pyodide (`harness/baselines/pyodide/{smoke,core}.json`): smoke matches
     native exactly. Core: 19713 passed / 0 failed / 967 skipped; the 15 tests
     that pass natively but are skipped there are WASM limits (`targets =
     ["pyodide"]` in skips.toml).
   Phases 2–4 below were largely absorbed into `core`; the remaining work is
   the out-of-core files.
   - `extended` (74 in-scope files outside core, incl. `roofline`, `sparse`,
     `key_reuse`, `hijax`, `mutable_array`, `api_test`-adjacent files):
     4289 passed / 0 failed / 807 skipped (2026-11-18; includes
     `shape_poly_test::DimExprTest`, the rest of that file is `jax.export`).
     `--x64`: 4480 / 0 / 785; `--slow`: 4337 / 0 / 759. Pyodide: 4258 / 0 / 825.
   - Not in a subset on purpose: `documentation_coverage_test` and
     `package_structure_test` walk packages by `jax.__name__` (`shax.jax`
     under shax), so they test package identity, not behaviour. A direct
     `dir()` diff of every public module vs real JAX shows the stable API
     (jax, numpy, lax, nn, random, scipy, tree_util, ...) complete; gaps are
     GPU/TPU/Mosaic experimental modules and `jnp.int1`/`uint1` (ml_dtypes
     0.5.4, as in Pyodide, predates them; upstream gates them on it).
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

## Moving the upstream pin

shax tracks one JAX tag (`TAG` in `scripts/port-upstream.py`, `JAX_TAG` in
`scripts/fetch-jax.sh`). `scripts/port-upstream.py --status` lists every
ported file (found by its `# SHAX: ported from` header) and how many hunks it
differs from the mechanical port; most files are identical, so a re-port is
mostly mechanical. To move to a new tag:

1. `JAX_TAG=jax-vX.Y.Z JAX_DEST=vendor/jax-new scripts/fetch-jax.sh`
2. `scripts/port-upstream.py --rebase vendor/jax-new --new-tag jax-vX.Y.Z`
   three-way merges each ported file: base = mechanical port of the current
   pin, ours = the shax file, theirs = mechanical port of the new tag. Hand
   edits carry over; conflicts are left as diff3 markers, and files deleted
   upstream are listed.
3. Resolve conflicts. Re-check the `lib/` stand-ins against the new
   `vendor/jax/jaxlib/*.cc` (and `jaxlib_extension_version` checks in
   upstream code), and port any newly imported modules
   (`ModuleNotFoundError: shax.jax._src.X`).
4. Bump `TAG` and `JAX_TAG`, replace `vendor/jax` with the new checkout,
   rebuild `.venv-jaxref` for the new jax/jaxlib, and port
   `harness/shax_harness/test_util.py` from the new upstream `test_util.py`.
5. Re-run `jax-upstream` vs `jax-ported` (must match), then the shax
   subsets. Expect test-list churn; compare to the old baselines, then
   accept new ones.

## Workflow

1. Pick a subset; run it with `--target jax-ported` first to confirm the harness is sound.
2. Run with `--target shax`; compare to the baseline; fix shax; repeat.
3. When results improve, update the baseline (the ratchet). Any test that goes
   pass → fail is a regression and must be fixed or explained before moving on.
4. Report pass/fail/skip/error-at-collection counts per file when summarizing progress.
