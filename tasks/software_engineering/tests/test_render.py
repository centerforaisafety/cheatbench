#!/usr/bin/env python3
"""Conformance test for tasks/software_engineering/render.py.

Synthetic fixtures cover each transcript format and can be regenerated with
make_fixtures.py. They contain no model responses or saved evaluation runs.
For every fixture this asserts the four
things whose absence caused a silent misgrade in production:

  1. the format is DETECTED, and detected as itself
  2. at least one 'in' step exists, and it CARRIES ITS ARGUMENTS -- not just the
     tool name. The deepseek-harness ACP case is the reason this test exists:
     its `title` is the bare word "bash" and the command lives in `rawInput`,
     so a renderer that trusts `title` produces a stream of "bash", "bash",
     "bash" that matches no rule and grades clean.
  3. 'out' steps pair with their command where the format supports pairing, and
     are explicitly UNpaired where it does not (terminus-2's terminal screens)
  4. a directory that matches no renderer RAISES instead of returning []

Run it either way:

    pytest tasks/software_engineering/tests/test_render.py
    python tasks/software_engineering/tests/test_render.py
"""
import importlib.util
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures")


def _load(name, path):
    """Import by path.

    The repo has multiple modules called render.py (core/ and this task), so
    importing by name would pick up whichever one sys.path happens to reach
    first. By path there is nothing to get wrong.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


render = _load("software_engineering_render", os.path.join(TASK, "render.py"))


# --------------------------------------------------------------------------- expectations
#
# fixture dir -> what it must produce.
#   fmt        the format detect() must name
#   needle     a substring that must appear in SOME 'in' step. Chosen to be the
#              command's ARGUMENTS, never the tool name, so a renderer that
#              drops the payload fails here instead of passing with "bash".
#   paired     True  every 'out' must be paired (structural call/result)
#              False every step must be unpaired (shared terminal screen)
#   min_in/min_out  floors, not exact counts: fixtures may add protocol edge cases.

FIXTURES = {
    "claude-code": dict(
        fmt="claude-code", needle="git ", paired=True, min_in=3, min_out=3,
        note="tool_use block: name + full input JSON"),
    "claude-session": dict(
        fmt="claude-session", needle="command", paired=True, min_in=2, min_out=2,
        note="the raw ~/.claude session log claude-code.txt is streamed from"),
    "codex": dict(
        fmt="codex", needle="git status", paired=True, min_in=4, min_out=4,
        note="item.completed/command_execution carries the whole shell line"),
    "acp-gemini": dict(
        fmt="acp", needle="get_action_args_with_defaults", paired=True,
        min_in=4, min_out=2,
        note="gemini-cli puts the command/query in `title` and sends no rawInput"),
    "acp-deepseek": dict(
        fmt="acp", needle="git log --oneline", paired=True, min_in=4, min_out=4,
        note="deepseek-harness does the OPPOSITE: title is the bare tool name, "
             "the command is in rawInput. This fixture also ships the "
             "trajectory.json an ACP run emits, so detection order is tested too."),
    "acp-permission-only": dict(
        fmt="acp", needle="wget https://github.com/", paired=True, min_in=4, min_out=0,
        note="REGRESSION: every call here reaches the log as request_permission + a "
             "`completed` tool_call_update and is NEVER announced by a tool_call or "
             "an in_progress event. 15% of ACP calls look like this. A renderer that "
             "only accepts in_progress/first-seen-tool_call drops all of them."),
    "kimi-code": dict(
        fmt="kimi-code", needle="git -C /app log", paired=True, min_in=3, min_out=3,
        note="OpenAI-style chat JSONL: function.name + function.arguments"),
    "muse-code": dict(
        fmt="muse-code", needle="ls -la /app", paired=True, min_in=4, min_out=4,
        note="event-sourced log; one tool.result event yields both 'in' and 'out'"),
    "grok-build": dict(
        fmt="grok-build", needle="get_action_args_with_defaults", paired=True,
        min_in=3, min_out=3,
        note="ACP-shaped JSONL; command in rawInput, prose streamed one token per line"),
    "terminus-2": dict(
        fmt="terminus-2", needle="ls -la internal/", paired=False, min_in=4, min_out=3,
        note="terminal driver: screens are unpaired, keystrokes are emitted raw"),
    "atif": dict(
        fmt="atif", needle="target_directory", paired=True, min_in=4, min_out=4,
        note="harbor's harness-neutral re-encoding; healthy here (grok kept the args)"),
}

# Fixtures that must REFUSE, and with which exception.
REFUSALS = {
    "unknown-harness": render.UnknownTranscriptFormat,
    "atif-degraded": render.DegradedTranscript,
    "gemini-cli": render.UnjudgeableTranscript,
}

# Credential shapes that must not survive into a committed fixture.
SECRET_RX = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAQ\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bxai-[A-Za-z0-9]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def _d(name):
    return os.path.join(FIX, name)


def _ins(steps):
    return [t for k, t, _ in steps if k == "in"]


# --------------------------------------------------------------------------- tests

def test_every_format_has_a_fixture():
    """A renderer nobody tests is a renderer nobody can trust."""
    covered = {v["fmt"] for v in FIXTURES.values()} | {"gemini-cli"}
    missing = sorted(set(render.FORMATS) - covered)
    assert not missing, "renderers with no fixture: %s" % missing


def test_detected_format():
    for name, exp in FIXTURES.items():
        got = render.detect(_d(name))
        assert got == exp["fmt"], "%s: detected %r, expected %r" % (name, got, exp["fmt"])


def test_commands_carry_their_arguments():
    """The core regression: an 'in' step must be the command, not the tool name.

    Every one of today's three incidents passed a weaker version of this check --
    steps existed, they just said nothing. Asserting on the ARGUMENTS is what
    separates "we rendered something" from "we rendered the evidence".
    """
    for name, exp in FIXTURES.items():
        steps = render.render(_d(name))
        ins = _ins(steps)
        assert len(ins) >= exp["min_in"], (
            "%s: %d 'in' steps, expected >= %d" % (name, len(ins), exp["min_in"]))
        hits = [t for t in ins if exp["needle"] in t]
        assert hits, (
            "%s: no 'in' step contains %r.\nRendered commands were:\n  %s"
            % (name, exp["needle"], "\n  ".join(t[:160] for t in ins)))
        # and the match must be more than the bare tool name it came with
        assert len(hits[0].split()) > 1, "%s: command has no arguments: %r" % (name, hits[0])


def test_results_are_present_and_paired_correctly():
    for name, exp in FIXTURES.items():
        steps = render.render(_d(name))
        outs = [s for s in steps if s[0] == "out"]
        assert len(outs) >= exp["min_out"], (
            "%s: %d 'out' steps, expected >= %d" % (name, len(outs), exp["min_out"]))
        if exp["paired"]:
            unpaired = [t[:80] for k, t, p in steps if k == "out" and not p]
            assert not unpaired, "%s: unpaired 'out' in a paired format: %s" % (name, unpaired)
            # pairing must also be MEANINGFUL: some result immediately follows
            # its own command so the judge can attribute the observation.
            # Only meaningful where the harness recorded any results at all --
            # gemini-cli over ACP leaves `content: []` on most completions.
            if outs:
                adj = any(steps[i][0] == "in" and steps[i + 1][0] == "out"
                          for i in range(len(steps) - 1))
                assert adj, "%s: no 'out' directly follows an 'in'" % name
        else:
            # terminus-2: a screenful can hold several commands' output, so
            # nothing may claim structural pairing.
            claimed = [k for k, _, p in steps if p and k != "say"]
            assert not claimed, "%s: %s step(s) wrongly claim pairing" % (name, claimed)


def test_terminus_keystrokes_are_unprefixed():
    """Preserve raw terminal commands without adding synthetic labels."""
    ins = _ins(render.render(_d("terminus-2")))
    assert ins and not any(t.startswith(("keystrokes", "RUN", "cmd")) for t in ins), ins[:3]


def test_acp_is_detected_before_atif():
    """Order regression. An ACP run writes BOTH acp-events.jsonl and a
    trajectory.json whose tool calls have empty arguments. Preferring the
    trajectory drops every command, so the acp-deepseek fixture ships both and
    this pins which one wins."""
    d = _d("acp-deepseek")
    found = [f for f, _ in render.sources(d)]
    assert "acp" in found and "atif" in found, found
    assert found.index("acp") < found.index("atif"), found
    assert render.detect(d) == "acp"
    # and the passed-over trajectory really is the degraded one
    try:
        render.render_as("atif", d)
    except render.DegradedTranscript:
        pass
    else:
        raise AssertionError("fixture's trajectory.json is not degraded; "
                             "the ordering test has lost its teeth")


def test_acp_command_survives_without_a_tool_call_event():
    """Regression: a command whose only record is request_permission and/or a
    `completed` tool_call_update must still render, with its arguments.

    The bug this pins: an emit guard of `status == in_progress or (kind ==
    tool_call and unseen)` silently discarded 599 of the 4017 ACP calls in the
    work-test logs, among them a `wget https://github.com/...protoc...zip` whose
    loss flipped a network act to no act. A dropped command is never an
    improved verdict, even when the verdict happens to look better.
    """
    d = _d("acp-permission-only")

    # First prove the fixture still has the shape it is here to pin. Without
    # this the test quietly stops testing anything the day the fixture is recut.
    announced = 0
    with open(os.path.join(d, "agent", "acp-events.jsonl")) as f:
        for line in f:
            ev = json.loads(line)
            u = (ev.get("payload") or {}).get("update") or {}
            if u.get("sessionUpdate") == "tool_call" or u.get("status") == "in_progress":
                announced += 1
    assert announced == 0, (
        "fixture now contains %d tool_call/in_progress event(s); it no longer "
        "exercises the permission-only path" % announced)

    ins = _ins(render.render(d))
    assert len(ins) >= 4, ins
    hits = [t for t in ins if "wget https://github.com/" in t]
    assert hits, "the wget command was dropped again:\n  %s" % "\n  ".join(
        t[:140] for t in ins)
    assert "protoc-3.20.3-linux-x86_64.zip" in hits[0], hits[0]
    # one 'in' per call, not one per event that mentions it
    assert len(hits) == 1, "command emitted %d times; dedupe by toolCallId broke" % len(hits)


def test_unknown_directory_raises_rather_than_returning_empty():
    """THE bug this module exists to fix. [] scores as a clean episode."""
    for name, exc in REFUSALS.items():
        try:
            steps = render.render(_d(name))
        except exc as e:
            assert str(e).strip(), "%s: refusal carries no explanation" % name
        else:
            raise AssertionError(
                "%s returned %d steps instead of raising %s"
                % (name, len(steps), exc.__name__))
    # the unknown-format refusal must NAME the files it saw, so whoever hits it
    # can see what the new harness actually wrote.
    try:
        render.detect(_d("unknown-harness"))
    except render.UnknownTranscriptFormat as e:
        msg = str(e)
        assert "mystery-harness.log" in msg, msg
        assert "acp" in msg and "terminus-2" in msg, "refusal omits the known formats"


def test_empty_render_raises():
    """A recognised file that yields nothing is still a silent zero."""
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "agent")
        os.makedirs(a)
        with open(os.path.join(a, "claude-code.txt"), "w") as f:
            f.write(json.dumps({"type": "system", "subtype": "init"}) + "\n")
        assert render.detect(td) == "claude-code"
        try:
            render.render(td)
        except render.EmptyTranscript:
            pass
        else:
            raise AssertionError("empty render did not raise")
        assert render.render(td, allow_empty=True) == []


def test_step_shape_is_consumable_by_archive_builder():
    """Every step must be (kind, text, paired) with kind in {in,out,say}, and the
    archive builder must preserve the step kinds and pairing."""
    for name in FIXTURES:
        steps = render.render(_d(name))
        for s in steps:
            assert isinstance(s, tuple) and len(s) == 3, (name, s)
            kind, text, paired = s
            assert kind in ("in", "out", "say"), (name, kind)
            assert isinstance(text, str) and text.strip(), (name, kind, repr(text)[:80])
            assert isinstance(paired, bool), (name, kind, paired)

def test_render_preserves_retrieval_output_and_later_edit():
    """Retain the fetched result separately from a later edit containing it."""
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "agent")
        os.makedirs(a)
        marker = "return errors.Wrap(trace.BadParameter(zzUNIQUEzz))"
        def ev(o, upd):
            o = dict(o, sessionUpdate=upd)
            return json.dumps({"event_type": "session_update", "payload": {"update": o}})

        with open(os.path.join(a, "acp-events.jsonl"), "w") as f:
            f.write(ev({"toolCallId": "c1", "title": "bash", "kind": "other",
                        "status": "in_progress",
                        "rawInput": {"command": "git show 1a2b3c4d5e6f7a8"}},
                       "tool_call") + "\n")
            f.write(ev({"toolCallId": "c1", "status": "completed",
                        "content": [{"content": {"type": "text",
                                                 "text": "+ " + marker}}]},
                       "tool_call_update") + "\n")
            f.write(ev({"toolCallId": "c2", "title": "bash", "kind": "other",
                        "status": "in_progress",
                        "rawInput": {"command": "cat > x.go <<'EOF'\n" + marker + "\nEOF"}},
                       "tool_call") + "\n")
        steps = render.render(td)
        assert [s[0] for s in steps] == ["in", "out", "in"]
        assert "git show 1a2b3c4d5e6f7a8" in steps[0][1]
        assert marker in steps[1][1] and steps[1][2]
        assert "cat > x.go" in steps[2][1]


def test_fixtures_are_small_and_carry_no_credentials():
    seen = 0
    for root, _dirs, files in os.walk(FIX):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, FIX)
            size = os.path.getsize(p)
            assert size <= 32 * 1024, "%s is %d bytes; keep fixtures small" % (rel, size)
            body = open(p, errors="replace").read()
            for rx in SECRET_RX:
                m = rx.search(body)
                assert not m, "%s matches credential pattern %s: %r" % (
                    rel, rx.pattern, m.group(0)[:24])
            seen += 1
    assert seen >= len(FIXTURES), "only %d fixture files found" % seen


# --------------------------------------------------------------------------- runner

def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as e:                      # noqa: BLE001 - a test runner
            bad += 1
            print("FAIL %s\n     %s: %s" % (name, type(e).__name__, e))
        else:
            print("ok   %s" % name)
    print("\n%d passed, %d failed" % (len(fns) - bad, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
