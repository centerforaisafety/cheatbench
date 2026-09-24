#!/usr/bin/env python3
"""Render native Harbor transcripts for the archive builder.

Each step is (kind, text, paired): kind is 'in' for an issued tool call with
arguments, 'out' for an observation, or 'say' for agent prose/reasoning.
Paired observations belong to the preceding call; terminal screens may contain
multiple commands and remain unpaired. This module never assigns verdicts.

Native ACP/provider logs take precedence over potentially lossy Harbor ATIF.
Unknown, empty or degraded recordings raise instead of becoming clean episodes.
See _REGISTRY for format detection order and tests/test_render.py for fixtures.
"""
from __future__ import annotations

import glob
import json
import os
import re

__all__ = [
    "Step", "FORMATS", "detect", "render", "render_as", "sources",
    "RenderError", "UnknownTranscriptFormat", "EmptyTranscript", "DegradedTranscript",
    "UnjudgeableTranscript",
]

# A rendered step, exactly as cheat.grade() wants it.
Step = tuple  # (kind: str, text: str, paired: bool)

# Per-step byte caps. These match tasks/software_engineering/build.py's archive caps, so a
# live episode rendered here and the same episode rendered into
# results/*/trajectories.txt.gz grade identically. Pass clip=False for fidelity.
IN_CAP = 8000
OUT_CAP = 20000
SAY_CAP = 20000


# --------------------------------------------------------------------------- errors

class RenderError(Exception):
    """Base: this episode could not be turned into a trustworthy step stream."""


class UnknownTranscriptFormat(RenderError):
    """No renderer claimed this directory. Raised INSTEAD of returning []."""


class EmptyTranscript(RenderError):
    """A renderer claimed the directory and produced nothing.

    Usually a run that died before its first tool call, or a transcript still
    being written. Either way the caller must decide -- scoring it as a clean
    episode is the bug this module exists to prevent.
    """


class UnjudgeableTranscript(RenderError):
    """A recognised format that structurally cannot carry the evidence we grade.

    Distinct from UnknownTranscriptFormat: we know exactly what this file is and
    know it is not enough. Separating the two keeps "nobody has taught the
    renderer about this harness yet" from "this harness does not record what we
    would need".
    """


class DegradedTranscript(RenderError):
    """The chosen source exists but has been stripped of the evidence we grade on.

    The live case: harbor's ATIF conversion of an ACP session records
    `function_name` and throws the arguments away, so every command renders as a
    bare tool name. Loud beats silent.
    """


# --------------------------------------------------------------------------- helpers

def _agent_dir(episode_dir: str) -> str:
    """The directory holding the transcripts.

    Accepts either an instance dir (the normal case, `<inst>/agent/...`) or an
    agent dir directly, so fixtures and ad-hoc dirs work without a wrapper.
    """
    a = os.path.join(episode_dir, "agent")
    return a if os.path.isdir(a) else episode_dir


def _clip(s, cap):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    if cap is None or len(s) <= cap:
        return s
    return s[:cap] + "\n... [%d bytes elided]" % (len(s) - cap)


def _jsonl(path):
    """Yield parsed objects from a JSONL file, skipping the harness's own noise.

    Every one of these files is polluted: kimi-code.txt interleaves raw command
    stdout with its chat log, grok-build.txt starts with ANSI-coloured tracing
    lines, codex.txt ends mid-write when a container is killed. Skipping
    non-objects is not sloppiness, it is the format.
    """
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _dumps(o):
    try:
        return json.dumps(o, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(o)


class _Out:
    """Coalesce adjacent streamed prose chunks without adding separators."""

    def __init__(self, clip=True):
        self.steps = []
        self.clip = clip
        self._buf = []
        self._tag = None

    def say_chunk(self, text, tag="say"):
        if not text:
            return
        if tag != self._tag:
            self.flush()
            self._tag = tag
        self._buf.append(text)

    def flush(self):
        if self._buf:
            t = "".join(self._buf)
            self._buf = []
            if t.strip():
                self.steps.append(("say", _clip(t, SAY_CAP if self.clip else None), True))
        self._tag = None

    def say(self, text):
        self.flush()
        if text and str(text).strip():
            self.steps.append(("say", _clip(text, SAY_CAP if self.clip else None), True))

    def cmd(self, text, paired=True):
        self.flush()
        if text and str(text).strip():
            self.steps.append(("in", _clip(text, IN_CAP if self.clip else None), paired))

    def result(self, text, paired=True):
        self.flush()
        if text and str(text).strip():
            self.steps.append(("out", _clip(text, OUT_CAP if self.clip else None), paired))

    def done(self):
        self.flush()
        return self.steps


def _texts(content):
    """Flatten the several content-block shapes these protocols use into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    out = []
    for c in content or []:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            inner = c.get("content")
            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                out.append(inner["text"])
            elif isinstance(inner, str):
                out.append(inner)
            elif isinstance(c.get("text"), str):
                out.append(c["text"])
    return "\n".join(x for x in out if x)


# --------------------------------------------------------------------------- claude-code

def _render_claude(path, clip=True):
    """Claude Code stream-JSONL, and the raw ~/.claude session JSONL it mirrors.

    One event per line; the payload is `message.content`, a list of typed blocks.
    tool_use carries the tool NAME and its full input, and the matching
    tool_result follows it, so pairing is structural.
    """
    o = _Out(clip)
    for ev in _jsonl(path):
        blocks = (ev.get("message") or {}).get("content")
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "thinking":
                o.say(b.get("thinking") or "")
            elif t == "text":
                o.say(b.get("text") or "")
            elif t == "tool_use":
                # "<Tool> <json args>". The tool NAME has to stay glued to the
                # arguments so the archive retains the actual tool identity.
                o.cmd("%s %s" % (b.get("name") or "tool", _dumps(b.get("input") or {})))
            elif t == "tool_result":
                o.result(_texts(b.get("content")))
    return o.done()


# --------------------------------------------------------------------------- codex

def _render_codex(path, clip=True):
    """Codex JSONL: `item.completed` envelopes around typed items."""
    o = _Out(clip)
    for ev in _jsonl(path):
        if ev.get("type") != "item.completed":
            continue
        it = ev.get("item") or {}
        t = it.get("type")
        if t == "agent_message":
            o.say(it.get("text") or "")
        elif t == "reasoning":
            txt = it.get("text") or it.get("summary") or ""
            if isinstance(txt, list):
                txt = "\n".join(str(x) for x in txt)
            o.say(txt)
        elif t == "command_execution":
            o.cmd("bash %s" % (it.get("command") or ""))
            o.result(it.get("aggregated_output") or "")
        elif t in ("file_change", "patch_apply"):
            o.cmd("%s %s" % (t, _dumps(it)))
    return o.done()


# --------------------------------------------------------------------------- terminus-2

_EPNUM = re.compile(r"(\d+)$")


def _render_terminus(episode_dir, clip=True):
    """terminus-2 episode dirs: agent/episode-N/{prompt,response}.txt.

    prompt.txt is the terminal SCREEN handed to the model, response.txt its JSON
    reply ({analysis, plan, commands:[{keystrokes}], task_complete}).

    Both step kinds are emitted unpaired. A screen is not the paired result of
    the command above it: one episode can fire four commands and the next screen
    shows only what fits. Preserve raw keystrokes and avoid invented pairing.
    """
    a = _agent_dir(episode_dir)
    eps = [d for d in glob.glob(os.path.join(a, "episode-*")) if _EPNUM.search(d)]
    eps.sort(key=lambda d: int(_EPNUM.search(d).group(1)))
    o = _Out(clip)
    for ed in eps:
        pr = os.path.join(ed, "prompt.txt")
        if os.path.exists(pr):
            o.result(open(pr, errors="replace").read(), paired=False)
        rp = os.path.join(ed, "response.txt")
        if not os.path.exists(rp):
            continue
        raw = open(rp, errors="replace").read()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            # A refusal or a truncated generation. Keep it as prose rather than
            # dropping the episode: it is still what the agent said.
            o.say(raw)
            continue
        try:
            d = json.loads(m.group(0))
        except ValueError:
            o.say(raw)
            continue
        o.say(d.get("analysis") or "")
        o.say(d.get("plan") or "")
        for c in d.get("commands") or []:
            if isinstance(c, dict) and c.get("keystrokes"):
                o.cmd(c["keystrokes"], paired=False)
        if d.get("task_complete"):
            o.say("<task_complete>")
    return o.done()


# --------------------------------------------------------------------------- ACP

def _acp_tool_name(call_id):
    """gemini-cli encodes the tool name in the id: `google_web_search__call_4563598`.

    Preserve the actual tool identity alongside the human-readable title.
    """
    if isinstance(call_id, str) and "__call" in call_id:
        return call_id.split("__call", 1)[0]
    return ""


def _render_acp(path, clip=True):
    """ACP protocol log: agent/acp-events.jsonl, one JSON-RPC notification per line.

    The two ACP agents put the command in OPPOSITE places, and handling only one
    of them is exactly how every dsh episode came out as the word "bash":

        gemini-cli   title = the command / the query; rawInput absent
        deepseek-hs  title = the bare tool name ("bash", "read", "edit");
                     rawInput = {"command": "git show <sha>", ...}

    So the rendered command line is kind + title + locations + rawInput, all of
    them, always. Each half is absent for the other agent, so concatenating is
    purely additive and never duplicates.

    `locations[]` carries the untruncated path that a `read` title elides
    ("lib/.../executor/module_common.py").

    Results pair structurally: every result names its own toolCallId.

    A call's command text is taken from the FIRST event that carries it,
    whatever kind of event that is, and then deduped by toolCallId. Three event
    shapes can be the first, and keying on any one of them loses commands:

      request_permission   payload.tool_call, not payload.update. This is the
                           agent asking to run the command, so it is both the
                           earliest and the most faithful position for the 'in'.
      tool_call            the ordinary case.
      tool_call_update     a call that was never announced separately -- it
                           goes straight to `completed`, carrying its title.

    Across the work-test ACP logs 599 of 4017 calls (15%) have NO tool_call or
    in_progress event at all; 587 of those appear first as request_permission.
    An earlier version of this renderer only accepted in_progress or a
    first-seen tool_call and silently dropped every one of them -- including a
    `wget https://github.com/.../protoc-3.20.3-linux-x86_64.zip` whose loss
    turned a network act into no act at all. Dropping a command instead of
    judging it is the exact failure this module exists to prevent, so the rule
    is: take it from wherever it appears, emit it once.
    """
    o = _Out(clip)
    seen = set()

    def emit_cmd(call):
        """Render one tool call's command line, at most once per toolCallId."""
        cid = call.get("toolCallId")
        if cid in seen:
            return
        parts = [call.get("kind") or "", _acp_tool_name(cid), call.get("title") or ""]
        locs = " ".join(l.get("path", "") for l in (call.get("locations") or [])
                        if isinstance(l, dict))
        if locs and locs not in parts[2]:
            parts.append(locs)
        raw = call.get("rawInput")
        if isinstance(raw, dict) and raw:
            parts.append(_dumps(raw))
        cmd = " ".join(p for p in parts if p).strip()
        if cmd:
            # Only a call we could actually render counts as seen: a bare
            # envelope must not block the later event that has the text.
            o.cmd(cmd)
            seen.add(cid)

    for ev in _jsonl(path):
        payload = ev.get("payload") or {}

        if ev.get("event_type") == "request_permission":
            call = payload.get("tool_call")
            if isinstance(call, dict):
                emit_cmd(call)
            continue

        u = payload.get("update") or {}
        upd = u.get("sessionUpdate")

        if upd in ("agent_message_chunk", "agent_thought_chunk"):
            o.say_chunk(_texts(u.get("content")), upd)
            continue
        if upd not in ("tool_call", "tool_call_update"):
            continue

        emit_cmd(u)
        if u.get("status") == "completed":
            o.result(_texts(u.get("content")))
    return o.done()


# --------------------------------------------------------------------------- kimi-code

def _render_kimi(path, clip=True):
    """kimi-code.txt: OpenAI-style chat JSONL, one message per line.

        {"role":"assistant","content":...,"tool_calls":[{"function":{name,arguments}}]}
        {"role":"tool","tool_call_id":...,"content":...}

    Name and arguments both survive and results carry their tool_call_id, so
    this is as judgeable as claude-code. The file also contains raw command
    stdout on non-JSON lines (the CLI tees it); _jsonl drops those, and nothing
    is lost -- the same bytes come back in the role:"tool" message.
    """
    o = _Out(clip)
    for d in _jsonl(path):
        role = d.get("role")
        if role == "assistant":
            o.say(_texts(d.get("content")))
            for c in d.get("tool_calls") or []:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, (dict, list)):
                    args = _dumps(args)
                o.cmd("%s %s" % (fn.get("name") or "tool", args or ""))
        elif role == "tool":
            o.result(_texts(d.get("content")))
    return o.done()


# --------------------------------------------------------------------------- muse-code

_MUSE_NOISE = ("output", "original_output_bytes", "original_output_tokens")


def _render_muse(path, clip=True):
    """muse-code.jsonl: an event-sourced session log, not a chat log.

    Nearly all of it is lifecycle bookkeeping. The judgeable pair lives in one
    `tool.result` event whose `payload.text` holds BOTH the command and its
    output, with the tool name in `payload.correlation_facts.tool_name`. So a
    single event yields an 'in' and an 'out'.

    `payload.text` has two shapes:
      JSON  bash and web_search -> {"command"/"query", "output"/"results", ...}
      prose every other tool    -> "Read text file `x.py`.\\n1|..." /
                                   "wrote 5519 bytes to /app/..." / a diff
    For the prose shape the first line is the only input-ish text there is, so it
    becomes the 'in' and the body becomes the 'out'.

    Muse records tool results rather than full inputs. Edited file contents can
    appear in observations; the audit must not assume that every observation
    was explicitly supplied as an input. The archive cannot restore missing data.
    """
    o = _Out(clip)
    pending_say = []
    for d in _jsonl(path):
        pt = d.get("payload_type")
        pay = d.get("payload") or {}

        if pt == "run.output.delta":
            o.say_chunk(pay.get("text") or "", "delta")
            pending_say.append(pay.get("text") or "")
            continue
        if pt == "run.terminal.completed":
            t = pay.get("text") or ""
            if t and t not in "".join(pending_say):
                o.say(t)
            continue
        if pt != "tool.result":
            continue

        tool = (pay.get("correlation_facts") or {}).get("tool_name") or "tool"
        text = pay.get("text") or ""
        inner = None
        if text.lstrip().startswith("{"):
            try:
                inner = json.loads(text)
            except ValueError:
                inner = None

        if isinstance(inner, dict):
            cmd = inner.get("command") or inner.get("query") or inner.get("path") or ""
            if not cmd:
                cmd = _dumps({k: v for k, v in inner.items() if k not in _MUSE_NOISE})
            o.cmd("%s %s" % (tool, cmd))
            body = inner.get("output")
            if body is None and "results" in inner:
                body = _dumps(inner["results"])
            o.result(body if isinstance(body, str) else (_dumps(body) if body else ""))
        else:
            head = text.split("\n", 1)[0]
            o.cmd("%s %s" % (tool, head))
            o.result(text)
    return o.done()


# --------------------------------------------------------------------------- grok-build

def _grok_bytes(v):
    """rawOutput.{output,stdout} arrive as an array of byte values."""
    if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
        try:
            return bytes(v).decode("utf-8", "replace")
        except (ValueError, TypeError):
            return ""
    return v if isinstance(v, str) else ""


def _grok_result_text(d):
    """Everything this update carries, content and rawOutput both.

    They are not redundant: a grep update's `content` says "found 9 matches"
    while `rawOutput.stdout` holds the matched LINES -- which is the evidence.
    Keeping only one of them threw away either the summary or the substance.
    """
    parts = []
    t = _texts(d.get("content"))
    if t.strip():
        parts.append(t)
    ro = d.get("rawOutput")
    if isinstance(ro, dict):
        for k in ("output", "stdout", "output_for_prompt", "summary_for_prompt", "text"):
            s = _grok_bytes(ro.get(k))
            if s and s.strip():
                parts.append(s)
                break
        else:
            if ro:
                parts.append(_dumps(ro))
    elif isinstance(ro, str) and ro.strip():
        parts.append(ro)
    keep = [p for p in parts if not any(p != q and p in q for q in parts)]
    return "\n".join(dict.fromkeys(keep))


def _render_grok(path, clip=True):
    """grok-build.txt: xAI's grok CLI streams JSONL close to the ACP shape.

    `tool_call` carries `toolName` + `rawInput` (the command lives THERE, not in
    `title`, which is just the tool name again). Results arrive as
    `tool_call_update`s sharing a toolCallId.

    Two things this has to get right:
      * `text`/`thought` events are ONE TOKEN each ({"type":"text","data":"I'll"}),
        so they are coalesced with no separator (_Out.say_chunk).
      * a result STREAMS as several updates on one id -- first a description,
        then an empty envelope, then the output growing to its final value.
        Emitting one 'out' per update puts a description where the result
        belongs. Buffer by id, keep the longest, and emit on status=completed.

    KNOWN WEAKNESS: grok batches several tool_calls before any result arrives, so
    an 'out' is not always adjacent to its 'in'. Attribution is weaker here than
    for claude-code or ACP; it is still far better than the empty stream.
    """
    o = _Out(clip)
    pending = {}
    for d in _jsonl(path):
        t = d.get("type")
        if t in ("text", "thought"):
            o.say_chunk(d.get("data") or "", t)
        elif t == "tool_call":
            name = d.get("toolName") or d.get("title") or "tool"
            raw = d.get("rawInput")
            body = _dumps(raw) if raw else ""
            locs = " ".join(l.get("path", "") for l in (d.get("locations") or [])
                            if isinstance(l, dict))
            o.cmd(" ".join(x for x in (name, body, locs) if x))
        elif t == "tool_call_update":
            cid = d.get("toolCallId")
            out = _grok_result_text(d)
            if out and len(out) >= len(pending.get(cid) or ""):
                pending[cid] = out
            if d.get("status") == "completed":
                o.result(pending.pop(cid, ""))
    return o.done()


# --------------------------------------------------------------------------- gemini-cli

def _render_gemini_cli(path, clip=True):
    """gemini-cli.txt: the CLI's own stdout, which is narration and warnings only.

    A real capture reads:

        Ripgrep is not available. Falling back to GrepTool.
        I will use `grep_search` to find where `RedisCacheConfig` is defined.
        Error executing tool grep_search: Path is not a directory: /app/...

    The tool calls and their results never appear -- they go to the model over
    the API. This file cannot establish which calls were actually issued, so
    it is refused instead of being treated as a complete trajectory.

    Recognising the file by name and refusing is the whole point: the caller
    learns that this cell cannot be scored, rather than that it was honest.
    """
    raise UnjudgeableTranscript(
        "%s is a gemini-cli stdout scrape: it records the agent's narration and "
        "the CLI's warnings, never the commands or their results. Any step stream "
        "built from it lacks issued tool calls. Judge from agent/acp-events.jsonl (run the agent "
        "over ACP) or exclude it; do not treat it as a clean episode." % path)


# --------------------------------------------------------------------------- ATIF

def _render_atif(path, clip=True):
    """trajectory.json: harbor's harness-neutral ATIF re-encoding.

    steps[] with source system|user|agent. An agent step carries
    reasoning_content, message, tool_calls[{tool_call_id, function_name,
    arguments}] and observation.results[{source_call_id, content}] -- the
    results of that step's OWN calls. They are matched by id so each 'out'
    follows its own 'in'; the raw order of observation.results does not track
    the order of tool_calls.

    system/user steps are the task instruction and harness reminders, not agent
    conduct, so they are skipped: grading an agent on the prompt it was handed
    produces false positives on every route.

    Raises DegradedTranscript when every tool call has empty arguments. That is
    what an ACP session looks like after ATIF conversion, and it is the shape
    that renders 65 commands as 65 bare tool names. This module reaches ATIF
    only as a last resort, so if it is degraded there is nothing better on disk
    and the caller needs to hear about it.
    """
    with open(path, errors="replace") as f:
        try:
            doc = json.load(f)
        except ValueError as e:
            raise RenderError("%s is not valid JSON: %s" % (path, e))
    if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
        raise RenderError("%s has no ATIF steps[] array" % path)

    o = _Out(clip)
    n_calls = n_empty = 0
    for st in doc["steps"]:
        if not isinstance(st, dict) or st.get("source") != "agent":
            continue
        o.say(st.get("reasoning_content") or "")
        msg = st.get("message")
        if isinstance(msg, str):
            o.say(msg)
        results = {}
        obs = st.get("observation")
        if isinstance(obs, dict):
            for r in obs.get("results") or []:
                if isinstance(r, dict):
                    results.setdefault(r.get("source_call_id"), _texts(r.get("content")))
        used = set()
        for tc in st.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            n_calls += 1
            args = tc.get("arguments")
            if not args:
                n_empty += 1
            o.cmd("%s %s" % (tc.get("function_name") or "tool",
                             _dumps(args) if args else ""))
            cid = tc.get("tool_call_id")
            if cid in results:
                used.add(cid)
                o.result(results[cid])
        for cid, body in results.items():
            if cid not in used:
                o.result(body)

    if n_calls and n_empty == n_calls:
        raise DegradedTranscript(
            "%s records %d tool call(s) and the arguments of every one of them are "
            "empty -- ATIF conversion dropped the payloads, so every command would "
            "render as a bare tool name. The command text survives only in the raw "
            "protocol log (agent/acp-events.jsonl for an ACP run); this episode dir "
            "does not have one." % (path, n_calls))
    return o.done()


# --------------------------------------------------------------------------- registry

def _file(*names):
    """Locator for 'agent/<name> exists and is non-empty'."""
    def locate(agent_dir):
        for n in names:
            p = os.path.join(agent_dir, n)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return p
        return None
    return locate


def _locate_terminus(agent_dir):
    for d in sorted(glob.glob(os.path.join(agent_dir, "episode-*"))):
        if os.path.isdir(d) and _EPNUM.search(d):
            return agent_dir
    return None


def _locate_session(agent_dir):
    """The biggest session log; a run can leave several, most of them stubs."""
    cand = [p for p in glob.glob(os.path.join(agent_dir, "sessions", "projects", "*", "*.jsonl"))
            if os.path.getsize(p) > 0]
    return max(cand, key=os.path.getsize) if cand else None


# (name, locate(agent_dir) -> source path or None, render(source, clip) -> steps)
# ORDER IS THE POINT. See the module docstring for why acp precedes atif.
_REGISTRY = (
    ("acp", _file("acp-events.jsonl"), _render_acp),
    ("claude-code", _file("claude-code.txt"), _render_claude),
    ("codex", _file("codex.txt"), _render_codex),
    ("kimi-code", _file("kimi-code.txt"), _render_kimi),
    ("muse-code", _file("muse-code.jsonl"), _render_muse),
    ("grok-build", _file("grok-build.txt"), _render_grok),
    ("gemini-cli", _file("gemini-cli.txt"), _render_gemini_cli),
    ("terminus-2", _locate_terminus, _render_terminus),
    ("claude-session", _locate_session, _render_claude),
    ("atif", _file("trajectory.json"), _render_atif),
)

FORMATS = tuple(name for name, _, _ in _REGISTRY)


# --------------------------------------------------------------------------- public API

def _inventory(episode_dir, limit=40):
    """What we actually saw, for the refusal message. Names, not guesses."""
    rows = []
    for label, d in (("", episode_dir), ("agent/", os.path.join(episode_dir, "agent"))):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names[:limit]:
            p = os.path.join(d, n)
            try:
                tag = "dir" if os.path.isdir(p) else "%dB" % os.path.getsize(p)
            except OSError:
                tag = "?"
            rows.append("%s%s (%s)" % (label, n, tag))
        if len(names) > limit:
            rows.append("%s... and %d more" % (label, len(names) - limit))
    return rows


def sources(episode_dir):
    """-> [(format, source_path)] for every renderer that claims this dir.

    Diagnostic: shows what detect() chose and what it passed over. An ACP run
    lists both ('acp', .../acp-events.jsonl) and ('atif', .../trajectory.json).
    """
    a = _agent_dir(episode_dir)
    out = []
    for name, locate, _ in _REGISTRY:
        try:
            p = locate(a)
        except OSError:
            p = None
        if p:
            out.append((name, p))
    return out


def detect(episode_dir):
    """-> the format name for this episode dir. Raises UnknownTranscriptFormat.

    Never returns None and never guesses. First match in _REGISTRY order wins.
    """
    found = sources(episode_dir)
    if found:
        return found[0][0]
    raise UnknownTranscriptFormat(
        "no renderer matched %s.\n"
        "Known formats, in detection order: %s.\n"
        "Files seen:\n  %s\n"
        "If this is a new harness, add a renderer to tasks/software_engineering/render.py. "
        "Returning an empty step stream instead would score this episode as CLEAN."
        % (episode_dir, ", ".join(FORMATS),
           "\n  ".join(_inventory(episode_dir)) or "(nothing)"))


def render_as(fmt, episode_dir, clip=True):
    """Render with an explicitly named format, bypassing detection."""
    for name, locate, fn in _REGISTRY:
        if name != fmt:
            continue
        src = locate(_agent_dir(episode_dir))
        if not src:
            raise UnknownTranscriptFormat(
                "format %r found no source under %s" % (fmt, episode_dir))
        return fn(src, clip)
    raise ValueError("unknown format %r; known: %s" % (fmt, ", ".join(FORMATS)))


def render(episode_dir, clip=True, allow_empty=False):
    """-> [(kind, text, paired), ...] for one episode directory.

    Raises UnknownTranscriptFormat if nothing claims the directory, and
    EmptyTranscript if a renderer claimed it and produced no steps. Missing
    evidence must not become a clean episode.

    clip=True caps each step at the character limits used by build_archive.py.
    allow_empty=True is for callers doing their own bookkeeping over many dirs.
    """
    fmt = detect(episode_dir)
    steps = render_as(fmt, episode_dir, clip=clip)
    if not steps and not allow_empty:
        src = dict(sources(episode_dir)).get(fmt)
        raise EmptyTranscript(
            "%s detected as %r but rendered 0 steps (source: %s). A run that died "
            "before its first tool call cannot establish behavior, so this is "
            "raised rather than returned."
            % (episode_dir, fmt, src))
    return steps


# --------------------------------------------------------------------------- cli

def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="episode (instance) directories")
    ap.add_argument("--dump", action="store_true", help="print every rendered step")
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    for d in a.dirs:
        d = d.rstrip("/")
        try:
            fmt = detect(d)
            steps = render(d, clip=not a.no_clip)
        except RenderError as e:
            rc = 1
            print("%-60s REFUSED %s: %s" % (os.path.basename(d)[:60],
                                            type(e).__name__, str(e).split("\n")[0]))
            continue
        kinds = {k: sum(1 for s in steps if s[0] == k) for k in ("in", "out", "say")}
        print("%-60s %-14s steps=%4d in=%3d out=%3d say=%3d"
              % (os.path.basename(d)[:60], fmt, len(steps),
                 kinds["in"], kinds["out"], kinds["say"]))
        if a.dump:
            for i, (k, t, p) in enumerate(steps, 1):
                print("  [%d] %s%s %s" % (i, k.upper(), "" if p else "*",
                                          t.replace("\n", "\\n")[:200]))
    return rc


if __name__ == "__main__":
    raise SystemExit(_main())
