# jax + optax in the browser, offline (marimo WASM export)

`notebook.py` fits a cubic with `jax.jit`, `jax.value_and_grad` and
`optax.adam`. Exported with marimo's `html-wasm --offline`, it runs entirely
in the browser (Pyodide) with no network access: `jax` there is shax.

How it gets `jax` without jaxlib:

- `dist/shax-*.whl`: shax itself.
- `dist/jax-0.12.0.1+shax-*.whl`: a shim distribution named `jax`
  (`packaging/jax-shim`) whose `import jax` installs `shax.as_jax`, so `jax`
  and every `jax.X` *are* `shax.jax` / `shax.jax.X`.
- `dist/jaxlib-0.12.0.1+shax-*.whl`: metadata-only `jaxlib`
  (`packaging/jaxlib-shim`), so packages declaring `jaxlib` (optax, chex)
  install. It contains no modules.

The notebook's PEP 723 header lists them, with `[tool.uv.sources]` pointing
at `../../dist/`; marimo's export copies them into `public/wheels/`.

## Build, export, check

```
pip install "marimo==0.25.2" playwright       # in some venv; uv on PATH or at ~/.local/bin/uv; then:
playwright install chromium                   # the offline export resolves packages in a headless browser
examples/optax-wasm/export.sh /tmp/optax-wasm # builds dist/ if needed, then exports
python scripts/check-wasm-export.py /tmp/optax-wasm --screenshot /tmp/optax-wasm.png
python -m http.server --directory /tmp/optax-wasm   # to try it yourself
```

`export.sh` runs `marimo export html-wasm notebook.py --mode run --show-code
--offline --execute` (extra arguments go to marimo, e.g. `--no-execute`).
Run mode hides code cells unless `--show-code` is given, so the export would
show only prose and widgets; the check script fails if no code cells are
visible (`--allow-no-code` to skip that). `--execute` embeds outputs computed
at export time, so readers see them on load (below).

`check-wasm-export.py` serves the export on localhost, loads it in headless
Chromium with all non-localhost requests blocked, and waits for the
notebook's `#shax-result` element (final loss in `data-loss`), and counts
the visible code cells. It strips the outputs `--execute` embedded from the
served page, so the result must come from shax running in the browser;
`--keep-preview` serves the page as readers get it. As of
2026-11-17 (marimo 0.25.2, Pyodide 314.0.0) it renders in ~12 s with loss
0.009453, the same as running the notebook natively. The export is ~55 MB
(Pyodide runtime, numpy, ml_dtypes, marimo and its deps, optax, absl-py,
shax).

Notes:

- marimo 0.25.2 pins Pyodide **314.0.0** (numpy 2.4.3), not the 314.0.8 that
  `scripts/jaxtest --target pyodide` uses.
- The browser serves the export over HTTP; opening `index.html` from
  `file://` doesn't work (Pyodide fetches its files).
- **Outputs on load (`--execute`, the default in `export.sh`)**: marimo
  runs the notebook natively at export time, in a uv sandbox built from the
  PEP 723 header (so `jax` there is shax too), and embeds the outputs in
  `index.html`. In run mode they appear within ~1 s, before Pyodide has
  loaded (~5 s, plus package installs), and widgets still work: moving a
  slider re-runs the dependent cells in the browser. With `--no-execute`,
  outputs appear only after the browser has run every cell (~12 s).
- marimo needs `uv` for any export of this notebook (it resolves the local
  wheels with it); `export.sh` falls back to `$UV` / `~/.local/bin/uv`.
- `--mode edit` exports work (editable code in the browser), but marimo
  doesn't run their cells on load, even with `[tool.marimo.runtime]
  auto_instantiate = true` in the script header: the user clicks run
  (bottom right). With `--execute` they show the embedded outputs, but the
  cells stay stale, so moving a slider changes nothing until the reader
  clicks run. Checked 2026-11-18 in headless Chromium (marimo 0.25.1).
- SciPy isn't needed: shax only loads it lazily for a few special functions.
