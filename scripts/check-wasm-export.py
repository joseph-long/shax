#!/usr/bin/env python3
"""Check that a `marimo export html-wasm --offline` export runs fully offline.

  python scripts/check-wasm-export.py <export-dir> [--selector '#shax-result'] [--timeout 600]

Serves <export-dir> on localhost, opens it in headless Chromium (Playwright)
with every non-localhost request blocked, and waits for `--selector` to
appear in the page. Prints its attributes, any blocked requests and browser
console errors. Exit status 0 only if the element appeared and nothing tried
to reach the network.

Needs `playwright` and a Chromium (`playwright install chromium`; set
PLAYWRIGHT_BROWSERS_PATH if it lives elsewhere). See examples/optax-wasm.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import sys
import threading
import time


def serve(directory: str) -> tuple[http.server.ThreadingHTTPServer, str]:
  class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
      pass

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
  a = ap.parse_args()

  from playwright.sync_api import sync_playwright  # noqa: PLC0415

  httpd, url = serve(a.export_dir)
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
    try:
      el = page.wait_for_selector(a.selector, timeout=a.timeout * 1000,
                                  state="attached")
      found = el.evaluate(
          "e => Object.fromEntries([...e.attributes].map(x => [x.name, x.value]))")
      if a.screenshot:
        page.wait_for_timeout(1500)  # let the remaining outputs paint
        page.screenshot(path=a.screenshot, full_page=True)
    except Exception as e:  # noqa: BLE001
      print(f"selector {a.selector!r} not found: {type(e).__name__}")
      text = page.inner_text("body")[-3000:]
      print("--- page text (tail) ---\n" + text)
    browser.close()
  httpd.shutdown()

  print(f"elapsed: {time.monotonic() - t0:.1f}s")
  if found is not None:
    print(f"found {a.selector}: {found}")
  for u in blocked:
    print(f"BLOCKED (tried the network): {u}")
  for m in console_errors[:20]:
    print(f"console error: {m}")
  return 0 if found is not None and not blocked else 1


if __name__ == "__main__":
  sys.exit(main())
