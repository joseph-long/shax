// Run pytest inside Pyodide (Node), for `scripts/jaxtest run --target pyodide`.
//
//   node scripts/pyodide-pytest.mjs <pyodide-dir> <repo-root> -- <pytest args...>
//
// <pyodide-dir> is what scripts/fetch-pyodide.py made (runtime + wheels +
// shax-manifest.json); everything loads from it, offline. The repo is mounted
// (NODEFS) at its real absolute path, so paths in the pytest arguments and
// the report are the same as on the host. shax and the harness are imported
// from the mounted source tree, not installed. JAX_* / SHAX_* environment
// variables are passed through. Exits with pytest's exit code.
import { readFileSync } from "node:fs";
import path from "node:path";

const argv = process.argv.slice(2);
const sep = argv.indexOf("--");
if (sep !== 2) {
  console.error("usage: pyodide-pytest.mjs <pyodide-dir> <repo-root> -- <pytest args>");
  process.exit(2);
}
const pyodideDir = path.resolve(argv[0]);
const repoRoot = path.resolve(argv[1]);
const pytestArgs = argv.slice(3);

const manifest = JSON.parse(
  readFileSync(path.join(pyodideDir, "shax-manifest.json"), "utf8"));
const { loadPyodide } = await import(path.join(pyodideDir, "pyodide.mjs"));

const env = {};
for (const [k, v] of Object.entries(process.env)) {
  if (k.startsWith("JAX_") || k.startsWith("SHAX_")) env[k] = v;
}
const pyodide = await loadPyodide({
  indexURL: pyodideDir + "/",
  env: { ...env, HOME: "/home/pyodide" },
});
// Pass output bytes straight through (the default batches by line, which
// breaks pytest's progress dots).
const raw = (stream) => ({ write: (buf) => { stream.write(buf); return buf.length; } });
pyodide.setStdout(raw(process.stdout));
pyodide.setStderr(raw(process.stderr));

const FS = pyodide.FS;
FS.mkdirTree(repoRoot);
FS.mount(FS.filesystems.NODEFS, { root: repoRoot }, repoRoot);
FS.mkdirTree("/tmp");

await pyodide.loadPackage(manifest.packages, { messageCallback: () => {} });
await pyodide.loadPackage(
  manifest.pypi_wheels.map((w) => path.join(pyodideDir, w)),
  { messageCallback: () => {} });

pyodide.globals.set("_repo_root", repoRoot);
pyodide.globals.set("_pytest_args", pyodide.toPy(pytestArgs));
const code = await pyodide.runPythonAsync(`
import os, sys
os.chdir(_repo_root)
sys.path[:0] = [os.path.join(_repo_root, "harness"), _repo_root]
import pytest
_exit = int(pytest.main(list(_pytest_args)))
sys.stdout.flush(); sys.stderr.flush()
_exit
`);
process.exit(code);
