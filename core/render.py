"""OURS: no Harbor counterpart -- Harbor writes ATIF out as `trajectory.json` and
serves it to a web viewer (`src/harbor/viewer/`); nothing in it turns a
trajectory into the text an LLM reads, which is all this file does.

ONE renderer: an ATIF `Trajectory` as the judge's ordered log.

There used to be a renderer per adapter, because each adapter kept its vendor's
own record -- the Claude SDK's message dataclasses for one, the lines of a Codex
rollout file for the other -- and only the adapter could read its own shape.
That put the vendor's data model on the critical path of the MEASUREMENT: a
third agent meant a third renderer and a third chance for the judge to be handed
a subtly different log.

Now every adapter converts its native output to ATIF (`Agent.to_trajectory`, see
`core/trajectory/`) and this module is the only thing that turns a trajectory
into text. `render_trajectory` takes a `Trajectory` and nothing else, so it
cannot branch on which agent produced it -- there is no argument it could branch
on.

THE SHAPE IS THE CONTRACT. `prompts/judge.md` is written against one log format
and one only:

    [init] agent=... version=... model=... tools=[...]
    THINKING: ...
    ASSISTANT: ...
    [7] TOOL_CALL Bash: command=...
        -> RESULT[Bash]: ...
    FINAL MESSAGE: ...

Tool calls are numbered in issue order across the whole trajectory, so "call 14"
in a verdict lines up with the log.

NO TRUNCATION, ever. A clipped command hides the tail of a wordlist, a clipped
assistant message hides the reasoning a verdict turns on, and an elided middle
hides exactly where the interesting behaviour happens. Base64 blobs are elided
because they are payload rather than content, and blank runs are collapsed;
nothing textual is ever cut. This rule is load-bearing -- a clipped command has
previously hidden the behaviour a verdict turned on.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .trajectory import ContentPart, Trajectory

# Anything that looks like a base64 payload (a long unbroken run of the
# alphabet) is worthless to a reader and expensive to send.
_B64_RUN = re.compile(r"[A-Za-z0-9+/=]{200,}")

# The argument that decides a verdict goes first, so a reader (and the judge)
# sees `command=` before a wall of options. Keyed by tool name across every
# adapter: Claude's `Bash` and Codex's `exec` are the same question. This is a
# presentation hint, not adapter dispatch -- a name that is not in the table
# simply renders in its natural key order.
_PRIORITY = {
    # claude-sdk
    "WebFetch": ["url", "prompt"],
    "WebSearch": ["query"],
    "Bash": ["command", "description"],
    "Read": ["file_path"],
    "Write": ["file_path", "content"],
    "Grep": ["pattern", "path"],
    "Glob": ["pattern", "path"],
    # codex
    "shell": ["command", "workdir"],
    "exec": ["cmd", "command", "input", "workdir"],
    "unified_exec": ["cmd", "command", "input", "workdir"],
    "local_shell": ["command", "workdir"],
    "apply_patch": ["input", "patch"],
    "web_search": ["query", "queries", "url"],
    "web_search_call": ["query", "queries", "url"],
}


def scrub(text: Any) -> str:
    """Elide base64 blobs and collapse blank runs. Text is never truncated."""
    if not isinstance(text, str):
        text = str(text)
    text = _B64_RUN.sub(
        lambda m: f"<base64 blob, {len(m.group(0))} chars, elided>", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def tool_input(name: str, inp: Any) -> str:
    """Tool arguments, with the fields that decide the verdict shown first."""
    if not isinstance(inp, dict):
        return scrub(inp)
    priority = _PRIORITY.get(name, [])
    keys = [k for k in priority if k in inp] + [k for k in inp if k not in priority]
    parts = []
    for k in keys:
        v = inp[k]
        v = v if isinstance(v, str) else json.dumps(v, default=str)
        parts.append(f"{k}={scrub(v)}")
    return "  ".join(parts) if parts else "{}"


def tool_result(content: Any) -> str:
    """What came back, flattened. Image payloads are named, never sent."""
    if isinstance(content, list):
        chunks = []
        for item in content:
            # ATIF multimodal parts, which is what a converted trajectory holds.
            if isinstance(item, ContentPart):
                chunks.append(item.text or "" if item.type == "text"
                              else f"<{item.type} returned to the model, elided>")
            elif isinstance(item, dict):
                if item.get("type") == "image":
                    chunks.append("<image returned to the model, elided>")
                elif "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(json.dumps(item, default=str))
            else:
                chunks.append(str(item))
        content = "\n".join(chunks)
    return scrub(content)


def message_text(message: Any) -> str:
    """An ATIF `Step.message`, which is a string or a list of ContentParts."""
    if isinstance(message, str):
        return message
    chunks = []
    for part in message or []:
        if isinstance(part, ContentPart):
            chunks.append(part.text or "" if part.type == "text"
                          else f"<{part.type}: {part.source.path}>")
        elif isinstance(part, dict):
            chunks.append(part.get("text")
                          or f"<{part.get('type')} content, elided>")
    return "\n".join(c for c in chunks if c)


def render_steps(traj: Trajectory, *, start_seq: int = 0) -> list[str]:
    """The step lines of a trajectory, without the `[init]`/`FINAL` envelope.

    Split out from `render_trajectory` because the live turn log is a TAIL: it
    renders one streamed record at a time and must not repeat the header on
    every block or close every block with a final message. Same lines, same
    scrubbing, same shape -- only the envelope differs.
    """
    lines: list[str] = []
    seq = start_seq
    names: dict[str, str] = {}
    prev_ctx: tuple | None = None

    for step in traj.steps:
        # Model / effort, announced only when it CHANGES. An agent that switches
        # model or reasoning effort mid-episode is a fact a verdict may turn on;
        # repeating an unchanged line on every step is noise.
        if step.source == "agent":
            ctx = (step.model_name, step.reasoning_effort)
            if any(c is not None for c in ctx):
                if prev_ctx is not None and ctx != prev_ctx:
                    lines.append(f"[context] model={ctx[0]} "
                                 f"reasoning_effort={ctx[1]}")
                prev_ctx = ctx

        # Extended thinking is its OWN line and is never folded into the
        # message: the judge reads what the model said and what it privately
        # reasoned as different evidence.
        if step.reasoning_content and step.reasoning_content.strip():
            lines.append(f"THINKING: {scrub(step.reasoning_content.strip())}")

        text = message_text(step.message).strip()
        if text:
            if step.source == "agent":
                lines.append(f"ASSISTANT: {scrub(text)}")
            else:
                lines.append(f"    -> USER/SYSTEM[{step.source}]: {scrub(text)}")

        for call in step.tool_calls or []:
            names[call.tool_call_id] = call.function_name
            lines.append(f"[{seq}] TOOL_CALL {call.function_name}: "
                         f"{tool_input(call.function_name, call.arguments)}")
            seq += 1

        if step.observation:
            for result in step.observation.results:
                name = names.get(result.source_call_id or "", "tool")
                err = " (ERROR)" if (result.extra or {}).get("is_error") else ""
                lines.append(f"    -> RESULT[{name}]{err}: "
                             f"{tool_result(result.content)}")

    return lines


def render_trajectory(traj: Trajectory) -> str:
    """A whole ATIF trajectory as the judge's ordered log.

    The ONLY renderer. It takes a `Trajectory` and nothing else, so it has no
    way to know -- and no way to care -- which adapter produced it.
    """
    init = [f"agent={traj.agent.name}", f"version={traj.agent.version}"]
    if traj.agent.model_name:
        init.append(f"model={traj.agent.model_name}")
    for key, value in (traj.agent.extra or {}).items():
        init.append(f"{key}={value}")
    lines = ["[init] " + " ".join(init)]

    lines += render_steps(traj)

    # The agent's closing answer, when its runtime reports one separately from
    # the last assistant step (both of ours do). ATIF has no dedicated field for
    # it, so it lives in the root `extra`, which is exactly what that field is
    # for -- and reading it here keeps this renderer adapter-blind.
    final = (traj.extra or {}).get("final_text")
    if final:
        lines.append(f"FINAL MESSAGE: {scrub(str(final))}")

    return "\n".join(lines)
