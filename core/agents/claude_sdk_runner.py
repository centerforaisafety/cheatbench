"""The Claude Agent SDK episode runner that executes INSIDE the container.

This file lives in the repo and is *never* baked into the image. `core/trial.py`
reads its source, packs it with `core/stage.py`, the task's optional `grader.py`
and the task data into one JSON blob, and feeds that blob to the container on
**stdin**.
A neutral one-liner in argv execs it.

Why stdin and nothing else:

  * The agent's Bash tool runs as the same uid as this process, so it can read
    `/proc/<pid>/cmdline` and `/proc/<pid>/environ` of its own ancestors. Task
    data therefore cannot ride in argv or in the environment -- and note that
    popping a variable out of `os.environ` later does NOT change `/proc`, which
    reports the exec-time environment block for the life of the process.
  * It cannot ride in a file either: the agent has Read/Bash on the container
    filesystem.
  * stdin is consumed by this process before the agent exists and is never
    re-readable, so the row (which holds the answer key and the gold's path)
    only ever exists as heap memory of a process the agent cannot ptrace into.

It also puts the agent's own binary in place (`install_agent` below). The image
is the task environment and carries no agent: an adapter brings its own runtime,
so one image serves Claude, Codex and anything else we add.

Contract: exactly ONE line of JSON on stdout (the whole episode record).
Everything else -- SDK chatter, tracebacks, live turn streaming -- goes to
stderr, so the host can parse stdout without heuristics.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# stdout hygiene, established before anything else can print.
#
# fd 1 is duplicated to a private handle and then *replaced* by fd 2, so any
# stray write to stdout -- ours, the SDK's, or an inherited child's -- lands on
# stderr instead of corrupting the single JSON line the host parses.
# ---------------------------------------------------------------------------
_OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)


def _log(msg: str) -> None:
    print(f"[runner] {msg}", file=sys.stderr, flush=True)


# Live turn streaming. The host tees these to messages.jsonl / turns.log as they
# arrive, so an episode that is killed or times out still leaves its turns
# behind.
MSG_PREFIX = "\x1eRHMSG "


def _emit_msg(rec: dict) -> None:
    try:
        sys.stderr.write(MSG_PREFIX + json.dumps(rec) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - streaming is best effort, never fatal
        pass


# ---------------------------------------------------------------------------
# transcript handling. NO TRUNCATION anywhere: full tool inputs, tool results,
# thinking and text, however long.
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """SDK messages are dataclasses; json.dumps cannot serialize them."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"_type": type(obj).__name__,
                **{f.name: _jsonable(getattr(obj, f.name))
                   for f in dataclasses.fields(obj)}}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _elide_images(content):
    """Replace image payloads in a tool result with a note.

    A `Read` of a staged PNG returns the picture base64'd. That payload is
    worthless in a log and bloats every artefact it touches. The picture itself
    is NOT elided anywhere the agent can see -- only in what we persist. Text is
    never touched: this is payload, not content.
    """
    if isinstance(content, list):
        out = []
        for it in content:
            if isinstance(it, dict) and it.get("type") == "image":
                src = it.get("source") or {}
                n = len(src.get("data") or "")
                out.append({"type": "text",
                            "text": f"<image returned to the model, {n} b64 chars, elided>"})
            else:
                out.append(it)
        return out
    return content


# The redacted-thinking envelope, by either the SDK's class name or the raw API
# block type. OpenRouter reuses it to pass PLAIN reasoning from the models it
# proxies; see the host-side converter in claude_sdk.py, which decodes it.
_REDACTED_TYPES = {"RedactedThinkingBlock", "redacted_thinking"}
_OPENROUTER_REASONING = "openrouter.reasoning:"


def _is_redacted_thinking(b: dict) -> bool:
    return bool(_REDACTED_TYPES & {b.get("_type"), b.get("type")})


def _slim(rec: dict) -> dict | None:
    """Strip a transcript message to what a reader (or the judge) needs."""
    t = rec.get("_type")
    if t == "StreamEvent":
        return None
    if t == "SystemMessage":
        if rec.get("subtype") in {"api_retry", "api_error"}:
            d = rec.get("data") or {}
            # Keep retry telemetry, not request headers, bodies, raw exceptions
            # or a provider's possibly credential-bearing error object.
            safe = {k: d[k] for k in ("retryAttempt", "maxRetries", "retryInMs",
                                      "attempt", "max_retries", "retry_delay_ms",
                                      "api_error_status", "status_code")
                    if type(d.get(k)) in (int, float)}
            return {"_type": t, "subtype": rec["subtype"], "data": safe}
        if rec.get("subtype") != "init":
            return None
        d = rec.get("data") or {}
        return {"_type": t, "subtype": "init",
                "data": {"tools": d.get("tools"),
                         "mcp_servers": d.get("mcp_servers")}}
    if t == "ResultMessage":
        return {"_type": t, "subtype": rec.get("subtype"),
                "num_turns": rec.get("num_turns"),
                "total_cost_usd": rec.get("total_cost_usd"),
                "usage": rec.get("usage"),
                "result": rec.get("result")}
    blocks = []
    for b in rec.get("content") or []:
        if not isinstance(b, dict):
            continue
        bt = b.get("_type")
        if bt == "TextBlock":
            if (b.get("text") or "").strip():
                blocks.append({"_type": bt, "text": b["text"]})
        elif bt == "ThinkingBlock":
            # `signature` is a multi-KB base64 blob with nothing to read in it,
            # so it is dropped -- but the BLOCK is kept whatever `thinking`
            # holds. Under `display: "omitted"` (the Opus 5 default) the field
            # arrives empty, and a transcript that dropped those blocks read as
            # if the model had never reasoned. It had; the text was withheld.
            # Those are different facts and the transcript must be able to tell
            # them apart. The renderer shows nothing for an empty one.
            blocks.append({"_type": bt, "thinking": b.get("thinking") or ""})
        elif _is_redacted_thinking(b):
            # Anthropic's encrypted-reasoning envelope. `data` is kept ONLY when
            # it is OpenRouter's reuse of the envelope to pass plain reasoning
            # through (`openrouter.reasoning:<b64>`), which the host-side
            # converter decodes; genuine ciphertext is unreadable and is dropped
            # for the same reason `signature` is, leaving the block itself as
            # the record that reasoning happened here.
            out_block = {"_type": b.get("_type") or "redacted_thinking"}
            data = b.get("data")
            if isinstance(data, str) and data.startswith(_OPENROUTER_REASONING):
                out_block["data"] = data
            blocks.append(out_block)
        elif bt == "ToolUseBlock":
            blocks.append({"_type": bt, "id": b.get("id"),
                           "name": b.get("name"), "input": b.get("input")})
        elif bt == "ToolResultBlock":
            blocks.append({"_type": bt, "tool_use_id": b.get("tool_use_id"),
                           "is_error": b.get("is_error"),
                           "content": _elide_images(b.get("content"))})
    if not blocks:
        return None
    out = {"_type": t, "content": blocks}
    if rec.get("error"):
        out["error"] = rec["error"]
    return out


def _count_tool_uses(transcript: list[dict]) -> int:
    n = 0
    for msg in transcript:
        if msg.get("_type") != "AssistantMessage":
            continue
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("_type") == "ToolUseBlock":
                n += 1
    return n


def _final_text(transcript: list[dict], result_msg: dict | None) -> str:
    if result_msg and result_msg.get("result"):
        return str(result_msg["result"])
    for msg in reversed(transcript):
        if msg.get("_type") != "AssistantMessage":
            continue
        texts = [b.get("text", "") for b in (msg.get("content") or [])
                 if isinstance(b, dict) and b.get("_type") == "TextBlock"]
        if any(t.strip() for t in texts):
            return "\n".join(texts)
    return ""


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
WORKDIR = "/workspace"

# The formats a deliverable can take. An agent that produced its own work writes
# one of these to its cwd; we ship the bytes back to the host so a later pass can
# diff them against whatever was staged.
DELIVERABLE_EXTS = (".xlsx", ".xls", ".xlsm", ".docx", ".doc", ".pptx", ".ppt",
                    ".pdf", ".csv", ".svg")

# Don't ship an unbounded amount back on stdout.
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024


def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple = ()) -> list:
    """Export office files and exact task-requested outputs for the host.

    `skip_dirs` are the top-level directories OUR staging created, so nothing we
    planted is ever re-exported as if the agent had made it. Best effort: a read
    error on one file is skipped, not fatal, since the episode still owes the
    host its JSON line.
    """
    import base64
    import stat

    # Task rows may request exact additional outputs (e.g. a ZIP and message).
    # These names do not broaden collection to other archives or text files.
    requested = {p for p in include_files if isinstance(p, str)
                 and p and not os.path.isabs(p)
                 and not any(part in ("", ".", "..") for part in p.split("/"))}
    out, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in skip_dirs:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            relative = os.path.relpath(os.path.join(dirpath, name), root)
            if (os.path.splitext(name)[1].lower() not in DELIVERABLE_EXTS
                    and relative not in requested):
                continue
            q = os.path.join(dirpath, name)
            try:
                # No symlink files; os.walk already excludes symlink directories.
                info = os.lstat(q)
                if not stat.S_ISREG(info.st_mode):
                    continue
                remaining = _DELIVERABLE_MAX_BYTES - total
                if info.st_size > remaining:
                    continue
                with open(q, "rb") as f:
                    data = f.read(remaining + 1)
            except OSError:
                continue
            if total + len(data) > _DELIVERABLE_MAX_BYTES:
                _log(f"deliverable export cap hit; skipping {q}")
                continue
            total += len(data)
            out.append({"name": os.path.relpath(q, root),
                        "b64": base64.b64encode(data).decode()})
    return out


# ---------------------------------------------------------------------------
# the adapter's runtime
# ---------------------------------------------------------------------------
# The image is the task environment and carries no agent binary: baking
# `@anthropic-ai/claude-code` into it welded one vendor to the sandbox and would
# have meant a second image the day we run Codex. The adapter says how to get
# its own runtime instead (core/agents/installed.py: INSTALL_CHECK / INSTALL), and this is
# where that runs -- inside the episode's own overlay, after staging and before
# the agent exists.
#
# `check || install` is the whole contract, and the check is what makes a
# pre-baked image an optional CACHE rather than a requirement: on an image that
# already has the binary this costs one PATH lookup.
INSTALL_TIMEOUT_S = 600


# Task-authored installation commands receive an environment without model
# credentials or routing variables. Keep package-manager and proxy settings.
_SECRET_ENV_SUBSTRINGS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD",
                          "CREDENTIAL", "AUTH", "BASE_URL")
_SECRET_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_", "CLAUDE_", "CODEX_",
                        "GEMINI_", "GOOGLE_", "VERTEX_", "XAI_", "GROK_",
                        "KIMI_", "MOONSHOT_", "MUSE_", "LITELLM_", "AZURE_",
                        "AWS_", "HF_", "HUGGINGFACE_", "HUGGING_FACE_")


def task_install_env() -> dict:
    """`os.environ` with the model credential removed, for a task's own shell."""
    out = {}
    for k, v in os.environ.items():
        up = k.upper()
        if any(s in up for s in _SECRET_ENV_SUBSTRINGS):
            continue
        if up.startswith(_SECRET_ENV_PREFIXES):
            continue
        out[k] = v
    return out


def install_agent(spec: dict | None,
                  env: dict | None = None) -> dict | None:
    """Put the adapter's binary in place. Returns what happened, or None.

    Never raises: the host is owed exactly one JSON line whatever happens here,
    and an install failure must surface as a recorded status rather than as a
    container that died with no record.

    `env` is the environment the install shell gets, and None means "inherit
    ours". That is what the ADAPTER's own runtime install wants -- it is the
    harness's own command in the harness's own environment. A TASK's `install:`
    is handed `task_install_env()` instead.
    """
    if not spec:
        return None
    import subprocess

    t0 = time.time()
    out = {"name": spec.get("name"), "status": "present",
           # The pin that was ASKED for, next to the version that actually ran.
           # Empty means "latest at episode time", which is a legitimate answer
           # and a different one from "we did not look".
           "pinned_version": spec.get("pinned_version") or "",
           "version": ""}

    def _sh(cmd: str, timeout: float):
        # Output is captured, never inherited: fd 1 is already pointed at
        # stderr, but a child that wrote to it directly would still be one more
        # thing between us and the single JSON line on stdout.
        #
        # `env` is None for the adapter's own runtime install, which means
        # "inherit ours"; a TASK's `install:` arrives with the credentials
        # already stripped out of it.
        return subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout, env=env)

    def _version() -> str:
        """What actually ended up installed. Never fatal.

        Read AFTER the install, down both branches, because the point is to
        record the build that produced this episode -- a cached binary that was
        already there is exactly as much of a reproducibility fact as one we
        just fetched.
        """
        cmd = spec.get("version_cmd")
        if not cmd:
            return ""
        try:
            res = _sh(cmd, 60)
            return (res.stdout or "").strip().split("\n")[0] if \
                res.returncode == 0 else ""
        except Exception:  # noqa: BLE001
            return ""

    try:
        if _sh(spec.get("check") or "true", 60).returncode == 0:
            out["version"] = _version()
            out["seconds"] = round(time.time() - t0, 1)
            _log(f"{out['name']}: runtime already present "
                 f"({out['seconds']}s to check), version "
                 f"{out['version'] or 'unknown'}")
            return out
        res = _sh(spec["install"], INSTALL_TIMEOUT_S)
        out["status"] = "installed" if res.returncode == 0 else "failed"
        if res.returncode != 0:
            # NO TRUNCATION: whatever npm said, in full.
            out["error"] = (res.stderr or res.stdout or "").strip()
        else:
            out["version"] = _version()
    except Exception as e:  # noqa: BLE001
        out["status"] = "failed"
        out["error"] = f"{type(e).__name__}: {e}"

    out["seconds"] = round(time.time() - t0, 1)
    _log(f"{out['name']}: runtime {out['status']} in {out['seconds']}s, "
         f"version {out['version'] or 'unknown'}")
    if out.get("error"):
        _log(f"{out['name']}: install said: {out['error']}")
    return out




async def run(task: dict, grader, install: dict | None = None,
              forwarder_mod: dict | None = None, observer=None) -> dict:
    """Run one episode and return the record the host will finish assembling.

    `install` is what `install_agent` reported, carried through to the record.
    `forwarder_mod` is the exec'd core/agents/forwarder.py namespace, present
    only when the model entry set `extra_body`.
    """
    from claude_agent_sdk import ClaudeAgentOptions, query

    model = task["model"]
    timeout_s = int(task.get("timeout_s") or 3600)

    # The task's grader, if it ships one, lives in an in-process MCP server. Its
    # implementation is never written to the container filesystem and never
    # appears in argv or environ, so no shell command the agent can run reaches
    # it -- only its verdicts cross back.
    #
    # A task that ships none arrives here as `({}, [], None)`: no servers, no
    # extra allowed tools, and no grader state to report.
    mcp, extra_allowed, grader_state = grader

    # An explicit empty list is MCP-only, not a request to restore defaults.
    tools = list(task["tools"]) if task.get("tools") is not None else ["Bash", "Read", "Write"]
    # Held shut on top of our own defaults. This is how a task's
    # `web_search: false` reaches the SDK: the web tools are already absent from
    # `tools`, and naming them here too means the SDK refuses them outright.
    blocked = ["Task", "TodoWrite", "Edit", "NotebookEdit"]
    for name in task.get("disallowed_tools") or []:
        if name not in blocked:
            blocked.append(name)
    child_env = {str(k): str(v) for k, v in (task.get("env") or {}).items()}
    # The API key arrives via the container runtime and is already in os.environ
    # under the name(s) the adapter declared (`routing.cli_key_env`; the CLI
    # reads ANTHROPIC_API_KEY); pass it explicitly so ClaudeAgentOptions.env is
    # self-contained.
    routing = dict(task.get("routing") or {})
    for name in [*(routing.get("cli_key_env") or []), "ANTHROPIC_API_KEY"]:
        if os.environ.get(name):
            child_env.setdefault(name, os.environ[name])
    api_key = next((child_env[n] for n in (routing.get("cli_key_env") or
                                           ["ANTHROPIC_API_KEY"])
                    if child_env.get(n)), "")

    # PER-MODEL ROUTING. `api_base_url` already sits in child_env as
    # ANTHROPIC_BASE_URL (core/agents/claude_sdk.py CLI_BASE_URL_ENV) when the
    # entry set one. A non-empty `extra_body` is the opt-in for the forwarder:
    # the CLI is pointed at a loopback listener that merges the body into every
    # JSON request and forwards it to the real URL with the real key. The
    # upstream is logged host-only; bodies and keys never are.
    forwarder = None
    if routing.get("extra_body"):
        upstream = routing.get("api_base_url") or child_env.get("ANTHROPIC_BASE_URL") or ""
        if not upstream:
            raise RuntimeError("extra_body is set but there is no api_base_url "
                               "to forward to: the forwarder needs an explicit "
                               "upstream (set api_base_url on the model entry)")
        forwarder = forwarder_mod["Forwarder"](
            base_url=upstream, api_key=api_key,
            extra_body=routing["extra_body"], log=_log).start()
        child_env["ANTHROPIC_BASE_URL"] = forwarder.url
        _log(f"forwarder on {forwarder.url} -> {forwarder.upstream.base_url} "
             f"(extra_body keys: {sorted(routing['extra_body'])})")
    _log(f"routing: base_url={child_env.get('ANTHROPIC_BASE_URL') or 'vendor default'} "
         f"key_env={routing.get('cli_key_env') or ['ANTHROPIC_API_KEY']} "
         f"forwarder={'on' if forwarder else 'off'}")

    opts_kwargs: dict[str, Any] = dict(
        model=model,
        cwd=WORKDIR,
        tools=tools,
        allowed_tools=[*tools, *extra_allowed],
        disallowed_tools=blocked,
        permission_mode=task.get("permission_mode") or "bypassPermissions",
        # Exactly the servers the task's grader built. strict_mcp_config keeps
        # any operator server out: this dict is the whole list.
        mcp_servers=mcp,
        strict_mcp_config=True,
        setting_sources=[],
        # One Read of a staged image blows past the 1 MiB default and kills the
        # message reader mid-episode, which reads as a model failure.
        max_buffer_size=256 * 1024 * 1024,
        env=child_env,
    )
    if task.get("max_turns"):
        opts_kwargs["max_turns"] = task["max_turns"]
    # Keys that carry the task's `tools:` policy into the SDK. NOTHING in
    # configs/models.yaml -- not a model's `generation_config:`, not an
    # `agents:` entry -- may widen or rewrite the policy the task declared;
    # otherwise a model could hand itself Edit/web tools while run.json still
    # records the task's policy as enforced. Reject any collision loudly, naming
    # the model and the key. Everything else (thinking, effort, env, ...) passes
    # through unchanged.
    _TOOL_POLICY_KEYS = {"tools", "allowed_tools", "disallowed_tools"}
    for k, v in (task.get("model_options") or {}).items():
        if k in _TOOL_POLICY_KEYS:
            raise RuntimeError(
                f"model {model!r}: config key {k!r} collides with the task's "
                f"tool policy; configs/models.yaml may not override "
                f"{sorted(_TOOL_POLICY_KEYS)}. Remove {k!r} from this model's "
                f"generation_config (or from its agents: entry) -- the task's "
                f"tools: policy is the only source of the tool set.")
        if k == "env":                    # merge: the API key must survive
            opts_kwargs["env"] = {**opts_kwargs["env"],
                                  **{a: str(b) for a, b in v.items()}}
        else:
            opts_kwargs[k] = v

    # message_delta stream events are the only per-turn usage the SDK exposes;
    # ResultMessage (which carries total_cost_usd) never arrives when the host
    # kills the episode at the cap, so without this every capped episode reports
    # $0. _slim() drops StreamEvents, so the transcript is unchanged.
    opts_kwargs["include_partial_messages"] = True

    # Use the CLI the adapter installed (the `claude` on PATH, whose version the
    # install step records as agent_version), not the SDK's bundled copy. The
    # SDK's _find_cli() PREFERS its bundled binary, which for claude-agent-sdk
    # 0.2.135 is Claude Code 2.1.227; api.anthropic.com rejects that for
    # claude-fable-5-1 ("version 2.1.251 or newer is required"), while the
    # gateway never enforced the gate. cli_path overrides the lookup.
    if "cli_path" not in opts_kwargs:
        installed_cli = shutil.which("claude")
        if installed_cli:
            opts_kwargs["cli_path"] = installed_cli

    options = ClaudeAgentOptions(**opts_kwargs)

    # The prompt was rendered on the host; the container never sees the task
    # folder, the prompt templates or data.jsonl.
    content = task["content"]

    async def _input():
        yield {"type": "user", "message": {"role": "user", "content": content}}

    transcript: list[dict] = []
    stream_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0}
    result_msg: dict | None = None
    init_msg: dict | None = None
    error: str | None = None
    t0 = time.time()

    try:
        async with asyncio.timeout(timeout_s):
            async for message in query(prompt=_input(), options=options):
                raw = _jsonable(message)
                if raw.get("_type") == "StreamEvent":
                    ev = raw.get("event") or {}
                    # message_delta carries the FINAL usage for that turn;
                    # message_start repeats it and would double the input legs.
                    if ev.get("type") == "message_delta":
                        u = ev.get("usage") or {}
                        for _k in ("input_tokens", "output_tokens",
                                   "cache_creation_input_tokens",
                                   "cache_read_input_tokens"):
                            stream_usage[_k] += u.get(_k) or 0
                rec = _slim(raw)
                if rec is None:
                    continue
                transcript.append(rec)
                _emit_msg(rec)
                if observer is not None:
                    observer.observe(rec)
                if rec.get("_type") == "SystemMessage" and rec.get("subtype") == "init":
                    init_msg = rec
                elif rec.get("_type") == "ResultMessage":
                    result_msg = rec
    except TimeoutError:
        error = f"timeout after {timeout_s}s"
    except Exception as e:  # noqa: BLE001 - one bad episode must not kill the run
        error = f"{type(e).__name__}: {e}"
        # When the CLI reports an API failure it does so as a ResultMessage
        # whose subtype is "success" and whose `result` is the actual message
        # ("API Error: Connection lost mid-response ..."). The SDK then raises
        # with the SUBTYPE, so the exception reads "returned an error result:
        # success" and carries nothing to classify. We already hold the
        # ResultMessage -- put its text on the record so the failure can be
        # typed instead of landing as `unclassified`.
        res = str((result_msg or {}).get("result") or "")
        if res.startswith("API Error") and res not in error:
            error = f"{error} -- {res}"

    wall = time.time() - t0
    final_text = _final_text(transcript, result_msg)

    init_data = (init_msg or {}).get("data") or {}

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()),
            tuple(task.get("deliverable_files") or ()))
    except Exception as e:  # noqa: BLE001
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    forwarder_calls = None
    if forwarder is not None:
        forwarder.close()
        forwarder_calls = list(forwarder.calls)

    return {
        "id": task["id"],
        "model": model,
        "messages": transcript,
        # The grader's own dict, exactly as the grader kept it.
        "grader_state": grader_state,
        "deliverables": deliverables,
        "final_text": final_text,
        "n_turns": (result_msg or {}).get("num_turns"),
        "n_tool_calls": _count_tool_uses(transcript),
        "wall_time": round(wall, 1),
        "cost_usd": (result_msg or {}).get("total_cost_usd"),
        "stream_usage": stream_usage,
        "usage": (result_msg or {}).get("usage"),
        "result_subtype": (result_msg or {}).get("subtype"),
        "terminal_reason": (result_msg or {}).get("terminal_reason"),
        "session_id": (result_msg or {}).get("session_id"),
        "init_tools": init_data.get("tools"),
        "init_mcp_servers": init_data.get("mcp_servers"),
        "install": install,
        # One summary per request the forwarder relayed (method, path, status,
        # sizes, seconds -- never a body), or None when no forwarder ran.
        "forwarder_calls": forwarder_calls,
        "error": error,
    }


def main(task: dict, modules: dict) -> None:
    """Stage the row's files, install the agent, build the grader, run it."""
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass

    # The harness code that travels with us. It is exec'd out of the stdin blob,
    # so it exists only in this process's memory.
    #
    # `stage` is always there. `grader` is the task's OPTIONAL grader module and
    # arrives as "" when the task ships none -- one convention, set in
    # `Task.grader_src()` and carried through `Agent.blob()`. Empty is treated
    # exactly like absent so a dropped key cannot masquerade as a grader.
    ns: dict[str, dict] = {}
    grader_src = (modules.get("grader") or "").strip()
    # `forwarder` travels only when the model entry set `extra_body`.
    forwarder_src = (modules.get("forwarder") or "").strip()
    for name in ("stage", *(("grader",) if grader_src else ()),
                 *(("forwarder",) if forwarder_src else ())):
        g: dict = {"__name__": f"rh_{name}"}
        exec(compile(modules[name], f"<{name}>", "exec"), g)
        ns[name] = g

    staged: dict = {}
    try:
        # Before the agent exists, so its very first `ls` already sees them.
        staged = ns["stage"]["write"](task.get("files") or [], WORKDIR)
        for dest, q in staged.items():
            _log(f"staged {dest} ({os.path.getsize(q)} bytes, "
                 f"mode {oct(os.stat(q).st_mode & 0o777)})")
    except Exception as e:  # noqa: BLE001 - still owe the host one JSON line
        _log(f"staging failed: {type(e).__name__}: {e}")

    # After staging, before the agent exists. `check || install`, and on an
    # image that already carries the binary it is only the check.
    install = install_agent(task.get("install"))

    # Install task dependencies after the agent runtime, without credentials.
    # Failure is recorded but does not prevent the episode from running.
    task_install = install_agent(task.get("task_install"),
                                 env=task_install_env())

    # (mcp_servers, extra_allowed_tool_names, grader_state) -- see run().
    if not grader_src:
        # NO GRADER: `grader_state` is None, not `{}`. Whether this task ships
        # one is already recorded by its `grader:` key, so the record does not
        # need to restate it.
        _log("no grader module in this episode's payload: task ships none")
        grader = ({}, [], None)
    else:
        try:
            grader = ns["grader"]["make_grader"](task.get("row") or {}, staged,
                                                 WORKDIR)
        except Exception as e:  # noqa: BLE001
            _log(f"grader construction failed: {type(e).__name__}: {e}")
            grader = ({}, [], {"error": f"{type(e).__name__}: {e}"})

    observer = None
    try:
        observer_factory = ns.get("grader", {}).get("make_observer")
        run_options = {"forwarder_mod": ns.get("forwarder")}
        if observer_factory is not None:
            observer = observer_factory(task.get("row") or {}, staged, WORKDIR, grader[2])
            run_options["observer"] = observer
        record = asyncio.run(run(task, grader, install, **run_options))
    except Exception as e:  # noqa: BLE001 - still owe the host one JSON line
        import traceback

        traceback.print_exc(file=sys.stderr)
        record = {"id": task.get("id"), "model": task.get("model"),
                  "messages": [], "grader_state": grader[2],
                  "final_text": "", "install": install,
                  "error": f"{type(e).__name__}: {e}"}
    if observer is not None:
        try:
            observer.finish()
        except Exception as e:
            record["error"] = f"{record.get('error') or ''} observer finish failed: {e}".strip()
    record["task_install"] = task_install
    _OUT.write(json.dumps(record) + "\n")
    _OUT.flush()


# Executed only when the bootstrap supplies TASK; importing this module (e.g.
# for a syntax check, or to read its source) does nothing.
_TASK = globals().get("TASK")
if _TASK is not None:
    main(_TASK, globals().get("MODULES") or {})
