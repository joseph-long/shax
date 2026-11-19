#!/usr/bin/env python3
"""Check that a `marimo export html-wasm --offline` export runs fully offline.

  python scripts/check-wasm-export.py <export-dir> [--selector '#shax-result'] [--timeout 600]

Serves <export-dir> on localhost, opens it in headless Chromium (Playwright)
with every non-localhost request blocked, and waits for `--selector` to
appear in the page. Prints its attributes, how many code cells are visible,
any blocked requests and browser console errors. Exit status 0 only if the
element appeared, code cells are visible (unless --allow-no-code; exports need
`--show-code` for that) and nothing tried to reach the network.

Outputs embedded with `--execute` are stripped from the served page (its
`"session"` preview is replaced by null), so the selector can only appear once
the browser has run the notebook itself. `--keep-preview` serves the page as
is, to check what readers see on load.

Needs `playwright` and a Chromium (`playwright install chromium`; set
PLAYWRIGHT_BROWSERS_PATH if it lives elsewhere). See examples/optax-wasm.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import sys
import threading
import time


_SESSION = '"session": '


def strip_preview(html: str) -> tuple[str, bool]:
  """Replace the page's embedded `--execute` outputs ("session") with null."""
  i = html.find(_SESSION)
  if i < 0:
    return html, False
  start = i + len(_SESSION)
  obj, end = json.JSONDecoder().raw_decode(html, start)
  if obj is None:
    return html, False
  return html[:start] + "null" + html[end:], True


def serve(directory: str, keep_preview: bool,
          stripped: list[str]) -> tuple[http.server.ThreadingHTTPServer, str]:
  class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
      pass

    def do_GET(self):
      path = self.translate_path(self.path)
      if os.path.isdir(path):  # "/" serves index.html
        path = os.path.join(path, "index.html")
      if keep_preview or not path.endswith(".html"):
        return super().do_GET()
      try:
        with open(path, encoding="utf-8") as f:
          html, changed = strip_preview(f.read())
      except OSError:
        return super().do_GET()
      if changed:
        stripped.append(self.path)
      body = html.encode()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

  handler = functools.partial(Handler, directory=directory)
  httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("export_dir")
  ap.add_argument("--selector", default="#shax-result")
  ap.add_argument("--timeout", type=float, default=600, help="seconds")
  ap.add_argument("--screenshot", help="save a full-page PNG here when done")
  ap.add_argument("--allow-no-code", action="store_true",
                  help="don't fail when no code cells are visible")
  ap.add_argument("--keep-preview", action="store_true",
                  help="serve outputs embedded by --execute instead of stripping them")
  a = ap.parse_args()

  from playwright.sync_api import sync_playwright  # noqa: PLC0415

  stripped: list[str] = []
  httpd, url = serve(a.export_dir, a.keep_preview, stripped)
  blocked: list[str] = []
  console_errors: list[str] = []
  t0 = time.monotonic()
  with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1100, "height": 900})

    def route(r):
      if r.request.url.startswith(url) or r.request.url.startswith("data:"):
        r.continue_()
      else:
        blocked.append(r.request.url)
        r.abort()

    page.route("**/*", route)
    page.on("console", lambda m: m.type == "error" and console_errors.append(m.text))
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
    page.goto(url)
    found = None
    n_code = 0
    try:
      el = page.wait_for_selector(a.selector, timeout=a.timeout * 1000,
                                  state="attached")
      found = el.evaluate(
          "e => Object.fromEntries([...e.attributes].map(x => [x.name, x.value]))")
      page.wait_for_timeout(1500)  # let the remaining outputs paint
      # Visible CodeMirror editors: marimo renders each shown code cell as one.
      n_code = page.locator(".cm-editor:visible").count()
      if a.screenshot:
        page.screenshot(path=a.screenshot, full_page=True)
    except Exception as e:  # noqa: BLE001
      print(f"selector {a.selector!r} not found: {type(e).__name__}")
      text = page.inner_text("body")[-3000:]
      print("--- page text (tail) ---\n" + text)
    browser.close()
  httpd.shutdown()

  print(f"elapsed: {time.monotonic() - t0:.1f}s")
  for u in sorted(set(stripped)):
    print(f"stripped embedded outputs from {u} (the browser computed these)")
  if found is not None:
    print(f"found {a.selector}: {found}")
    print(f"visible code cells: {n_code}")
    if not n_code and not a.allow_no_code:
      print("NO CODE CELLS VISIBLE (export with --show-code, or pass --allow-no-code)")
  for u in blocked:
    print(f"BLOCKED (tried the network): {u}")
  for m in console_errors[:20]:
    print(f"console error: {m}")
  code_ok = n_code > 0 or a.allow_no_code
  return 0 if found is not None and code_ok and not blocked else 1


if __name__ == "__main__":
  sys.exit(main())
