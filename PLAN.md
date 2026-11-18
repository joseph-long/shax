# shax: status and plan (as of 2026-11-18)

Pick-up notes. Ground rules, architecture and harness details live in
`AGENTS.md`; this file is the state of the work and what's next.

## Where things stand

Repo moved from `/mnt/home/jlong/devel/shax` (GPFS) to
`/home/jlong/devel/shax` (workstation-local NVMe) on 2026-11-18 because the
GPFS home hit its file-count quota. History intact (`git fsck` clean); last
commit `10217cd`. Both venvs were rebuilt here with `scripts/setup-envs.sh`
(delete them first: the script keeps an existing venv dir).

Test results (all with 0 regressions against their baselines):

| run | passed | failed | skipped |
|---|---|---|---|
| smoke (shax) | 1555 | 0 | 48 |
| smoke (jax-ported, real JAX) | 1573 | 0 | 57 |
| core | 19728 | 0 | 952 |
| core `--x64` | 20363 | 1 | 976 |
| core `--slow` | 19786 | 0 | 894 |
| core, Pyodide | 19713 | 0 | 967 |
| extended (74 files + `DimExprTest`) | 4289 | 0 | 807 |
| extended `--x64` | 4480 | 0 | 785 |
| extended `--slow` | 4337 | 0 | 759 |
| extended, Pyodide | 4258 | 0 | 825 |

- core `--x64`'s one failure is `scipy_optimize_test TestBFGS::test_fixes4594`
  (ill-conditioned BFGS that hinges on XLA's dot rounding; see AGENTS.md).
- Unit tests (`tests/`, 6544 passed) and harness self-tests (5 passed).
- All of the above re-run after the move on 2026-11-18 (except core
  `--x64`/`--slow`), plus smoke on every target, the wheel build and the
  WASM check; 0 regressions.

Goalpost met: `examples/optax-wasm` (jit + value_and_grad + optax.adam)
exported with `marimo export html-wasm --mode run --show-code --offline`
runs in headless Chromium with the network blocked, loss 0.009453 (same as
native), ~12 s to render, ~56 MB. Code cells are shown: `examples/optax-wasm/export.sh`
bakes in `--show-code`, and `check-wasm-export.py` fails if no code cells are visible.

Performance: optax step ~1.1 ms natively (`eval_jaxpr_compiled` raw path
skips per-eqn config contexts; dtype predicates memoized; `_ftz` returns
after one ufunc when nothing is below `tiny`).

## Plan (original steps)

1. HLO skips: done.
2. Fix failures: done for smoke/core/extended.
3. `--x64` / `--slow` runs: done for core and extended.
4. Expand coverage: in progress (see below).
5. Vendor opt_einsum: done (`shax/_vendor/opt_einsum`).
6. Run the harness in Pyodide: done (core and extended baselines).
7. License and docs: done (Apache-2.0; AGENTS.md, READMEs).

## Next steps

1. **Re-verify after the move**: done 2026-11-18 (see above).
2. **Watch a flaky test**: `api_test.py::DCETest::
   test_dce_jaxpr_scan_nontrivial_fixedpoint_carry_2` failed once (a doubled
   `sin` eqn in a freshly traced scan body). Unmodified upstream tracing
   code; not reproduced in 6 api_test runs or 5000 traces. If it recurs,
   suspect trace-time caching (`lib/weakref_lru_cache.py`, const dedup by
   `id()` in `lib/_jax.py` canonicalization).
3. **marimo edit mode**: resolved 2026-11-18. Edit-mode exports work, but
   marimo doesn't run their cells on load (not even with `auto_instantiate =
   true`); the user clicks run. Clicking run-all in headless Chromium
   produces `#shax-result`. For outputs on load, use `--execute` (embeds
   natively computed outputs) with `--mode run`: outputs show in ~1 s and
   widgets re-run cells in the browser. In edit mode `--execute` outputs
   show but widgets do nothing until the reader clicks run. Noted in
   `examples/optax-wasm/README.md`.
4. **Coverage expansion (step 4)**. Files outside every subset, by status:
   - Excluded on purpose: `documentation_coverage_test`,
     `package_structure_test` (they test package identity: `jax.__name__`
     is `shax.jax`). A direct `dir()` diff vs real JAX shows the stable API
     complete.
   - Need HLO/XLA (likely skip): `source_mapper_test`, `sourcemap_test`,
     `custom_partitioning_test`, `pgle_test`, `overlap_test`.
   - Checked 2026-11-18: `ragged_collective_test` and `scaled_dot_test`
     skip every test under real JAX on CPU too (nothing to gain).
     `shape_poly_test`: 2453 pass under real JAX, but all except
     `DimExprTest` go through `jax.export` (StableHLO). `DimExprTest`
     (symbolic-dimension algebra, 130 pass + 1 export skip, also in
     Pyodide/x64) is now in `extended`.
   - Left: `pytorch_interoperability_test` (needs torch),
     `magma_linalg_test` (GPU).
   - Out-of-scope list in AGENTS.md still applies (pjit, multi-device,
     pallas/, export, ...), though Pallas interpret mode does run.
5. **API gaps** (from the `dir()` diff): `jnp.int1`/`uint1` (need ml_dtypes
   >= 0.6; Pyodide ships 0.5.4, upstream gates on it, so fine as is);
   experimental `mesh_utils`, `shard_map` (deprecated alias), `layout`,
   `profiler`, `custom_partitioning`, `array_serialization` are unported.
   Port the pure-Python ones if something imports them.
6. **Performance** (non-goal, but the WASM demo benefits): `_ftz` sped up
   2026-11-18 (`.any()` instead of `np.any`, early exit; ~17% off a step).
   Remaining per-op costs: `_run_raw`, `from_numpy` (per-call imports,
   creation tracebacks via `get_traceback`). Profile with an optax loop
   (`cProfile` over `step`; optax/chex installed `--no-deps` outside
   `.venv`) before changing the hot path.
7. **Upstream pin**: when moving off `jax-v0.12.0.1`, follow "Moving the
   upstream pin" in AGENTS.md.

## Environment reminders

- `~/.local/bin/uv` only; no Lmod. `curl` blocked; use Python `requests`.
- Recursive traversals under `/mnt/home` need `timeout 120` and
  `-maxdepth 7`.
- GPFS home is over its file-count quota; the repo lives on local NVMe now.
