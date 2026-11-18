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
| extended (74 files) | 4159 | 0 | 806 |
| extended `--x64` | 4350 | 0 | 784 |
| extended `--slow` | 4207 | 0 | 758 |
| extended, Pyodide | 4128 | 0 | 824 |

- core `--x64`'s one failure is `scipy_optimize_test TestBFGS::test_fixes4594`
  (ill-conditioned BFGS that hinges on XLA's dot rounding; see AGENTS.md).
- Unit tests (`tests/`, 6544 passed) and harness self-tests (5 passed) were
  last run before the move; re-run them in the new location.

Goalpost met: `examples/optax-wasm` (jit + value_and_grad + optax.adam)
exported with `marimo export html-wasm --mode run --show-code --offline`
runs in headless Chromium with the network blocked, loss 0.009453 (same as
native), ~12 s to render, ~56 MB. Code cells are shown: `examples/optax-wasm/export.sh`
bakes in `--show-code`, and `check-wasm-export.py` fails if no code cells are visible.

Performance: optax step ~1.6 ms natively (`eval_jaxpr_compiled` raw path
skips per-eqn config contexts; dtype predicates memoized).

## Plan (original steps)

1. HLO skips: done.
2. Fix failures: done for smoke/core/extended.
3. `--x64` / `--slow` runs: done for core and extended.
4. Expand coverage: in progress (see below).
5. Vendor opt_einsum: done (`shax/_vendor/opt_einsum`).
6. Run the harness in Pyodide: done (core and extended baselines).
7. License and docs: done (Apache-2.0; AGENTS.md, READMEs).

## Next steps

1. **Re-verify after the move**: `PYTHONPATH=harness .venv/bin/python -m
   pytest -p no:cacheprovider tests harness/tests`, then core natively and in
   Pyodide (`-j 10`; core takes ~40 min because of slow `sqrtm` tests).
   Rebuild `dist/` (`packaging/build-wheels.sh`) and re-run the WASM check.
2. **Watch a flaky test**: `api_test.py::DCETest::
   test_dce_jaxpr_scan_nontrivial_fixedpoint_carry_2` failed once (a doubled
   `sin` eqn in a freshly traced scan body). Unmodified upstream tracing
   code; not reproduced in 6 api_test runs or 5000 traces. If it recurs,
   suspect trace-time caching (`lib/weakref_lru_cache.py`, const dedup by
   `id()` in `lib/_jax.py` canonicalization).
3. **marimo edit mode**: `--mode edit` export didn't produce `#shax-result`
   within 600 s in `scripts/check-wasm-export.py`. Unknown whether edit-mode
   WASM doesn't auto-run cells or something fails. Look at a screenshot /
   console log if editable in-browser code is wanted.
4. **Coverage expansion (step 4)**. Files outside every subset, by status:
   - Excluded on purpose: `documentation_coverage_test`,
     `package_structure_test` (they test package identity: `jax.__name__`
     is `shax.jax`). A direct `dir()` diff vs real JAX shows the stable API
     complete.
   - Need HLO/XLA (likely skip): `source_mapper_test`, `sourcemap_test`,
     `custom_partitioning_test`, `pgle_test`, `overlap_test`.
   - Larger candidates worth a look: `shape_poly_test` (mostly
     `jax.export`, probably skip), `ragged_collective_test`, `scaled_dot_test`
     (skips itself under pytest), `pytorch_interoperability_test` (needs
     torch), `magma_linalg_test` (GPU).
   - Out-of-scope list in AGENTS.md still applies (pjit, multi-device,
     pallas/, export, ...), though Pallas interpret mode does run.
5. **API gaps** (from the `dir()` diff): `jnp.int1`/`uint1` (need ml_dtypes
   >= 0.6; Pyodide ships 0.5.4, upstream gates on it, so fine as is);
   experimental `mesh_utils`, `shard_map` (deprecated alias), `layout`,
   `profiler`, `custom_partitioning`, `array_serialization` are unported.
   Port the pure-Python ones if something imports them.
6. **Performance** (non-goal, but the WASM demo benefits): next hotspot is
   `elementwise._ftz` (~17% of an optax step; five ufunc calls per op).
   Profile with an optax loop (`cProfile` over `step`) before changing the
   hot path.
7. **Upstream pin**: when moving off `jax-v0.12.0.1`, follow "Moving the
   upstream pin" in AGENTS.md.

## Environment reminders

- `~/.local/bin/uv` only; no Lmod. `curl` blocked; use Python `requests`.
- Recursive traversals under `/mnt/home` need `timeout 120` and
  `-maxdepth 7`.
- GPFS home is over its file-count quota; the repo lives on local NVMe now.
