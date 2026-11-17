"""Regenerate fixtures/pretty_printer_cases.json from real jaxlib.

Run with the reference env (real jaxlib 0.12.0.1), from the repo root:

  .venv-jaxref/bin/python tests/lib/gen_pretty_printer_fixture.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxlib import _pretty_printer as pp  # noqa: E402
import jaxlib  # noqa: E402

import _pp_docgen  # noqa: E402


def main():
  out = []
  for spec, group in _pp_docgen.make_cases():
    renders = []
    for opts in group:
      text, sm = _pp_docgen.render(pp, spec, opts)
      renders.append(dict(opts=opts, text=text, source_map=sm))
    out.append(dict(spec=spec, repr=_pp_docgen.doc_repr(pp, spec),
                    renders=renders))
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                      "pretty_printer_cases.json")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(dict(jaxlib_version=jaxlib.__version__, docs=out), f,
              ensure_ascii=False, separators=(",", ":"))
  print(f"wrote {len(out)} docs, "
        f"{sum(len(d['renders']) for d in out)} renders to {path}")


if __name__ == "__main__":
  main()
