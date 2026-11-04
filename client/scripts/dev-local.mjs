import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const python = process.env.PYTHON_BIN || path.join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const apiPort = process.env.DGILIT_API_PORT || "8000";
const webPort = process.env.PORT || "5173";
const apiOrigin = process.env.DGILIT_API_ORIGIN || `http://127.0.0.1:${apiPort}`;
if (!existsSync(python) && !process.env.PYTHON_BIN) {
  console.error("Create the isolated API environment first: python3 -m venv .venv && .venv/bin/python -m pip install -r api/requirements.txt");
  process.exit(1);
}
const children = [];
let stopping = false;
async function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) if (child.exitCode === null) child.kill("SIGTERM");
  await Promise.all(children.map(child => new Promise(resolve => {
    if (child.exitCode !== null) return resolve();
    const timer = setTimeout(() => { child.kill("SIGKILL"); resolve(); }, 4000);
    child.once("exit", () => { clearTimeout(timer); resolve(); });
  })));
  process.exit(code);
}
function start(command, args) {
  const child = spawn(command, args, { cwd: root, stdio: "inherit", env: { ...process.env, DGILIT_API_ORIGIN: apiOrigin } });
  children.push(child);
  child.on("error", error => { console.error(error.message); void stop(1); });
  child.on("exit", code => { if (!stopping) void stop(code || 0); });
  return child;
}
process.on("SIGINT", () => void stop());
process.on("SIGTERM", () => void stop());
start(python, ["-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", apiPort]);
let ready = false;
for (let attempt = 0; attempt < 40 && !stopping; attempt++) {
  try { await fetch(`${apiOrigin}/api/health`, { signal: AbortSignal.timeout(1500) }); ready = true; break; }
  catch { await new Promise(resolve => setTimeout(resolve, 250)); }
}
if (!ready) { console.error("The API did not start. Check its log above."); await stop(1); }
else if (!stopping) {
  console.log(`\ndgiLIT evidence browser: http://127.0.0.1:${webPort}\nRead-only API: ${apiOrigin}\n`);
  start(process.execPath, ["scripts/run-framework.mjs", "dev", "--hostname", "127.0.0.1", "--port", webPort]);
}
