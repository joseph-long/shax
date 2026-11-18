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
packaging/build-wheels.sh                     # -> dist/
pip install "marimo==0.25.2" playwright       # in some venv; then:
playwright install chromium                   # the offline export resolves packages in a headless browser
marimo export html-wasm examples/optax-wasm/notebook.py -o /tmp/optax-wasm --mode run --offline
python scripts/check-wasm-export.py /tmp/optax-wasm --screenshot /tmp/optax-wasm.png
python -m http.server --directory /tmp/optax-wasm   # to try it yourself
```

`check-wasm-export.py` serves the export on localhost, loads it in headless
Chromium with all non-localhost requests blocked, and waits for the
notebook's `#shax-result` element (final loss in `data-loss`). As of
2026-11-17 (marimo 0.25.2, Pyodide 314.0.0) it renders in ~12 s with loss
0.009453, the same as running the notebook natively. The export is ~55 MB
(Pyodide runtime, numpy, ml_dtypes, marimo and its deps, optax, absl-py,
shax).

Notes:

- marimo 0.25.2 pins Pyodide **314.0.0** (numpy 2.4.3), not the 314.0.8 that
  `scripts/jaxtest --target pyodide` uses.
- The browser serves the export over HTTP; opening `index.html` from
  `file://` doesn't work (Pyodide fetches its files).
- SciPy isn't needed: shax only loads it lazily for a few special functions.
