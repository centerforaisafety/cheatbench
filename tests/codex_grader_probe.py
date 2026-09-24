"""The in-container half of the codex grader isolation probe.

Streamed into a REAL episode container on stdin, in the blob `CodexAgent.blob`
built, and exec'd by the same bootstrap a real episode uses. It stands in for
`codex exec` and for nothing else: everything it touches on the way -- the
grader module, `serve_grader`, `prepare_home`, the row -- is the production code
path, so what this probe observes is what an episode observes.

What it does, in the order an episode does it:

  1. exec the task's grader out of `MODULES["grader"]`, as the runner does;
  2. start it with the runner's own `serve_grader`;
  3. write `$CODEX_HOME/config.toml` with the runner's own `prepare_home`;
  4. THEN put on the agent's hat: read the URL back out of config.toml the way
     an agent with `cat` would, call the tool over it, and sweep the whole
     filesystem and /proc for the grader's source and the row's answer key.

Exactly one line of JSON on stdout, through the runner's own `_OUT` handle.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request

TASK = globals()["TASK"]
MODULES = globals()["MODULES"]

# The runner, exec'd as a module. No TASK in this namespace, so its `main()`
# does not fire -- we are calling the pieces, not running an episode.
runner: dict = {"__name__": "rh_runner"}
exec(compile(MODULES["runner"], "<runner>", "exec"), runner)
OUT = runner["_OUT"]                     # stdout, before the runner hid it

report: dict = {"steps": []}


def step(name, **kw):
    report["steps"].append({"step": name, **kw})


# --- 1/2. the grader, exactly as an episode starts it ----------------------
grader_ns: dict = {"__name__": "rh_grader"}
exec(compile(MODULES["grader"], "<grader>", "exec"), grader_ns)
server = runner["serve_grader"](grader_ns, TASK.get("row") or {}, {},
                                "/workspace")
step("serve_grader", url_host=server.server_address[0],
     port=server.server_address[1])

# --- 3. config.toml, exactly as an episode writes it -----------------------
cfg = TASK["codex"]
runner["prepare_home"](cfg, "sk-not-a-real-key", grader_url=server.url,
                       grader_name=server.grader_server_name)
config_path = os.path.join(cfg["codex_home"], "config.toml")
config_text = open(config_path).read()
report["config_toml"] = config_text

# --- 4. the agent's view ---------------------------------------------------
# The URL as an agent would get it: parsed out of the config file it can read.
#
# Scoped to the `[mcp_servers.*]` table on purpose. A bare `url = "..."` search
# matches `openai_base_url` first when the run has a proxy configured, and this
# probe then POSTs the episode's MCP handshake at the model provider -- which is
# how the first version of it "failed", with a 403 from litellm.
url = ""
section = re.search(r'^\s*\[mcp_servers\.[^\]]+\]\s*$(.*?)(?=^\s*\[|\Z)',
                    config_text, re.M | re.S)
if section:
    line = re.search(r'^\s*url\s*=\s*"([^"]+)"', section.group(1), re.M)
    url = line.group(1) if line else ""
report["url_recovered_from_config"] = bool(url)
report["url_is_loopback"] = url.startswith("http://127.0.0.1:")


def rpc(payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return {"code": r.status, "body": r.read().decode()}


try:
    report["initialize"] = rpc({"jsonrpc": "2.0", "id": 0,
                                "method": "initialize", "params": {}})
    report["tools_list"] = rpc({"jsonrpc": "2.0", "id": 1,
                                "method": "tools/list", "params": {}})
    report["tools_call"] = rpc({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": server.grader_tool_name,
                   "arguments": {"deliverable": "/workspace/nothing.xlsx"}}})
except Exception as e:  # noqa: BLE001
    report["call_error"] = f"{type(e).__name__}: {e}"

# A path the agent guessed wrong reaches nothing at all.
base = url.rsplit("/", 2)[0] if url else ""
try:
    urllib.request.urlopen(urllib.request.Request(
        base + "/guessed/mcp", data=b"{}", method="POST"), timeout=15)
    report["wrong_path_status"] = 200
except urllib.error.HTTPError as e:
    report["wrong_path_status"] = e.code
except Exception as e:  # noqa: BLE001
    report["wrong_path_status"] = f"{type(e).__name__}: {e}"


# --- the sweep: is any of it on disk, or in anyone's /proc? -----------------
#
# Done IN PYTHON, not by shelling out to grep, and that is not a style
# preference. The first version of this probe ran
# `grep -F -e <needle> ... /proc/*/cmdline`, which put every needle on a command
# line -- and then found them there. A probe whose own argv contaminates the
# thing it is measuring cannot answer the question. Here the needles never leave
# this process's heap, exactly like the row and the grader they stand for.
NEEDLES = [n.encode() for n in TASK["probe_needles"]]
_MAX_FILE = 8 * 1024 * 1024


def hits_in(data: bytes) -> list:
    return sorted({n.decode() for n in NEEDLES if n in data})


def sweep_filesystem() -> list:
    """Every regular file on the episode's own filesystem."""
    out, root_dev = [], os.stat("/").st_dev
    for dirpath, dirnames, filenames in os.walk("/", onerror=lambda e: None):
        # Stay on the overlay: /proc, /sys and /dev are separate mounts and are
        # swept (or deliberately not) elsewhere.
        dirnames[:] = [d for d in dirnames
                       if _same_device(os.path.join(dirpath, d), root_dev)]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                if os.path.islink(path) or os.path.getsize(path) > _MAX_FILE:
                    continue
                with open(path, "rb") as fh:
                    found = hits_in(fh.read())
            except OSError:
                continue
            if found:
                out.append({"path": path, "needles": found})
    return out


def _same_device(path: str, dev: int) -> bool:
    try:
        return not os.path.islink(path) and os.stat(path).st_dev == dev
    except OSError:
        return False


def sweep_proc() -> list:
    """argv and environ of every process. The agent runs as this uid and can
    read both."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        for what in ("cmdline", "environ"):
            try:
                with open(f"/proc/{entry}/{what}", "rb") as fh:
                    found = hits_in(fh.read())
            except OSError:
                continue
            if found:
                out.append({"pid": entry, "source": what, "needles": found})
    return out


try:
    report["filesystem_hits"] = sweep_filesystem()
    report["filesystem_error"] = None
except Exception as e:  # noqa: BLE001
    report["filesystem_hits"] = []
    report["filesystem_error"] = f"{type(e).__name__}: {e}"

try:
    report["proc_hits"] = sweep_proc()
    report["proc_error"] = None
except Exception as e:  # noqa: BLE001
    report["proc_hits"] = []
    report["proc_error"] = f"{type(e).__name__}: {e}"

report["codex_home_listing"] = subprocess.run(
    ["ls", "-laR", cfg["codex_home"]], capture_output=True, text=True).stdout

server.close()
OUT.write(json.dumps(report) + "\n")
OUT.flush()
