"""Tool evidence must survive native transcript -> ATIF -> judge/live log."""
import json

import pytest

from core.agents import make_agent, trajectory_from_transcript
from core.agents.claude_sdk_runner import _slim
from core.judge import render_transcript


def _codex_call(source):
    return {"type": "response_item", "payload": {
        "type": "custom_tool_call", "call_id": "batch", "name": "exec",
        "input": source}}


@pytest.mark.parametrize("source", [
    # The original writings regression: only the first command was shown.
    '''const r = await Promise.allSettled([
tools.exec_command({cmd:"cat instruction.txt",workdir:"/workspace"}),
tools.exec_command({cmd:"cat grading/grade_config.json",workdir:"/tmp"}),
tools.exec_command({cmd:"rg --files",workdir:"/workspace"})]); text(r);''',
    # Extracting every `cmd` still loses non-shell tools in the same batch.
    '''const r = await Promise.allSettled([
tools.exec_command({cmd:"cat instruction.txt"}),
tools.view_image({path:"/tmp/reference.png"}),
tools.write_stdin({session_id:42,chars:"cat /tmp/reference.svg\\n"})]); text(r);''',
    # Values may come from elsewhere in the program, not object literals.
    '''const paths = ["/tmp/reference.svg", "/tmp/config.json"];
for (const path of paths) {
  const r = await tools.exec_command({cmd:`cat ${path}`,workdir:"/workspace"});
  text(r.output);
}''',
    '''const cmd = "cat /tmp/reference.svg";
const options = {cmd,workdir:"/workspace"};
text(await tools.exec_command(options));''',
    # Quoted keys, numeric/boolean/nested arguments and later raw-string calls.
    '''text(await tools.exec_command({"cmd":"cat /tmp/reference.svg",yield_time_ms:1000}));
text(await tools.lookup({query:{paths:["/tmp/answer"]},recursive:true,limit:2}));
text(await tools.apply_patch("*** Begin Patch\\n*** End Patch"));''',
    '''text(await tools.exec_command({cmd:"cat instruction.txt && cat /tmp/reference.svg"}));''',
    'text(await tools.exec_command({cmd:"ls"}));\n'
    + '// intermediate code\n' * 600
    + 'text(await tools.view_image({path:"/tmp/reference-at-the-end.png"}));',
    "no recognized tools call here",
], ids=["shell-batch", "mixed-batch", "template-loop", "variable-arguments",
        "arbitrary-arguments", "shell-chain", "long-input", "raw-fallback"])
def test_codex_custom_input_reaches_judge_and_live_log_in_full(source):
    call = _codex_call(source)
    raw = {"agent": "codex", "model": "test", "messages": [call, {
        "type": "response_item", "payload": {
            "type": "custom_tool_call_output", "call_id": "batch",
            "output": "first result\nsecond result\nlast result"}}]}
    traj = trajectory_from_transcript(raw)
    args = next(s.tool_calls[0].arguments for s in traj.steps if s.tool_calls)
    # Exact source, including output-selection code, is the evidence. Do not
    # invent separate executed calls by statically scanning this program.
    assert args == {"input": source}
    assert source in render_transcript(raw)
    assert source in make_agent("codex", model="test").readable(call)
    assert "first result\nsecond result\nlast result" in render_transcript(raw)


def _messages(adapter, calls):
    """Equivalent tool batches in each CLI's recorded format."""
    if adapter == "claude-sdk":
        return [_slim(m) for m in [
            {"_type": "AssistantMessage", "content": [
                {"_type": "ToolUseBlock", "id": cid, "name": name, "input": args}
                for cid, name, args in calls]},
            {"_type": "UserMessage", "content": [
                {"_type": "ToolResultBlock", "tool_use_id": cid,
                 "content": f"result-{cid}", "is_error": False}
                for cid, _, _ in reversed(calls)]},
        ]]
    if adapter == "codex":
        return [
            {"type": "response_item", "payload": {
                "type": "function_call", "call_id": cid, "name": name,
                "arguments": json.dumps(args)}} for cid, name, args in calls
        ] + [
            {"type": "response_item", "payload": {
                "type": "function_call_output", "call_id": cid,
                "output": f"result-{cid}"}} for cid, _, _ in reversed(calls)
        ]
    if adapter == "grok-build":
        return [{"type": "assistant", "content": "", "tool_calls": [
            {"id": cid, "name": name, "arguments": json.dumps(args)}
            for cid, name, args in calls]}] + [
            {"type": "tool_result", "tool_call_id": cid, "content": f"result-{cid}"}
            for cid, _, _ in reversed(calls)]
    if adapter == "gemini-cli":
        return [{"type": "gemini", "id": "turn", "content": "", "toolCalls": [
            {"id": cid, "name": name, "args": args,
             "status": "success", "result": [{"functionResponse": {
                 "name": name, "response": {"output": f"result-{cid}"}}}]}
            for cid, name, args in calls]}]
    if adapter == "muse-code":
        events = [
            {"kind": "assistant_tool_calls_committed", "response_id": "r",
             "message_id": "batch", "tool_calls": [
                 {"id": cid, "call_id": cid, "name": name, "args": json.dumps(args)}
                 for cid, name, args in calls]},
            {"kind": "tool_result_batch_committed", "batch_id": "batch", "results": [
                {"tool_call_id": cid, "text": f"result-{cid}"}
                for cid, _, _ in reversed(calls)]},
        ]
        return [{"payload_type": "runtime.session", "payload": {"kind": "run", "event": e}}
                for e in events]
    raise AssertionError(adapter)


@pytest.mark.parametrize("adapter", ["claude-sdk", "codex", "grok-build",
                                     "gemini-cli", "muse-code"])
def test_every_adapter_preserves_batched_arguments_and_matches_results(adapter):
    calls = [
        ("c0", "shell", {"command": "cat instruction.txt && cat /tmp/reference.svg",
                         "workdir": "/workspace"}),
        ("c1", "read_file", {"path": "/tmp/answer.json", "offset": 7, "limit": 12}),
        ("c2", "search", {"query": {"paths": ["/tmp/hidden"]}, "recursive": True}),
    ]
    raw = {"agent": adapter, "model": "test", "messages": _messages(adapter, calls)}
    traj = trajectory_from_transcript(raw)
    actual = [(c.tool_call_id, c.function_name, c.arguments)
              for s in traj.steps for c in s.tool_calls or []]
    assert actual == calls
    results = {r.source_call_id: r.content for s in traj.steps if s.observation
               for r in s.observation.results}
    assert results == {cid: f"result-{cid}" for cid, _, _ in calls}
    text = render_transcript(raw)
    for cid, name, args in calls:
        assert f"TOOL_CALL {name}:" in text
        assert f"result-{cid}" in text
        for key, value in args.items():
            rendered = value if isinstance(value, str) else json.dumps(value)
            assert f"{key}={rendered}" in text
