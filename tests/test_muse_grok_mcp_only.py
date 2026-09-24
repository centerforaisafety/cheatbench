"""MCP-only native tool selection, transport gate, and real-CLI canaries."""
import copy
import json
import os
from pathlib import Path
import subprocess
import pytest

from core.agents import make_agent, forwarder
from core.agents import muse_code_runner as muse, grok_build_runner as grok
from core.sandbox import native_inference as bridge

ROOT = Path(__file__).resolve().parents[1]
POLICY = {"mcp_only": True, "web_search": False, "web_fetch": False}


@pytest.mark.parametrize("adapter,key", [("muse-code", "muse"), ("grok-build", "grok")])
def test_restricted_payload_and_normal_defaults(adapter, key):
    agent = make_agent(adapter, model="test-model", api_base_url="https://gateway.example/v1")
    assert agent.apply_tool_policy(POLICY) == []
    assert agent.setup() == []
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="unchanged", files=[],
        skip_dirs=(), stage_src="", grader_src="test", timeout_s=3600, env={}))
    task = payload["task"]
    assert task[key]["mcp_only"] and task["content"] == "unchanged"
    assert task["row"]["restricted_tools"] and task["row"]["episode_timeout_s"] == 3600
    assert "support" in payload["modules"]
    if key == "muse": assert "--disable-write" not in task[key]["cli_flags"]
    routed = json.loads(bridge.reroute_blob(json.dumps(payload), "http://127.0.0.1:8000/v1"))["task"]
    assert routed[key]["base_url"] == "http://127.0.0.1:8000/v1"
    assert routed["content"] == task["content"]
    assert agent.apply_tool_policy({"web_search": False, "web_fetch": False})


def test_muse_request_retains_settings_and_nine_schemas():
    chess = {"type": "namespace", "name": "mcp__chess", "tools": [
        {"type": "function", "name": n, "parameters": {"type": "object"}} for n in sorted(muse.MCP_NAMES)]}
    doc = {"model": "same", "input": "same", "reasoning": {"effort": "high"},
           "tools": [chess, {"type": "namespace", "name": "muse", "tools": [{"name": "write_todos"}]}]}
    actual, names = muse.restricted_request(doc)
    assert actual == {**doc, "tools": [chess]}
    assert set(names) == {"mcp__chess__" + n for n in muse.MCP_NAMES}
    bad = copy.deepcopy(doc); bad["tools"][0]["tools"].pop()
    with pytest.raises(ValueError): muse.restricted_request(bad)


@pytest.mark.parametrize("name", ["bash", "read_file", "write_file", "write_todos", "subagent_spawn", "workflow"])
def test_muse_response_blocks_native_execution(name):
    for payload in ({"item": {"type": "function_call", "name": name, "arguments": "{}"}},
                    {"response": {"output": [{"type": "function_call", "name": name, "arguments": "{}"}]}}):
        with pytest.raises(ValueError, match="blocked"): muse.checked_response(payload)


@pytest.mark.parametrize("game", ["chess", "go"])
def test_muse_fragmented_stream_preserves_reasoning_and_normalizes_mcp_identity(game):
    event = {"type": "response.output_item.added", "item": {"type": "function_call",
        "namespace": "mcp__" + game, "name": "read_notes", "arguments": "{}"}}
    frames = [b'event: response.reasoning_text.delta\ndata: {"type":"response.reasoning_text.delta","delta":"keep this"}\n\n',
              ("event: response.output_item.added\ndata: " + json.dumps(event) + "\n\n").encode()]
    wire = b"".join(frames)
    class Response:
        status = 200
        def header(self, name): return "text/event-stream" if name == "Content-Type" else str(len(wire))
        def iter_chunks(self, size=65536):
            yield from (wire[i:i+3] for i in range(0, len(wire), 3))
    response = Response(); muse.guard_response(response, forwarder._SSEFrames, game)
    result = b"".join(response.iter_chunks())
    assert b'keep this' in result and ('mcp__' + game + '__read_notes').encode() in result
    assert response.header("Content-Length") is None


@pytest.mark.parametrize("game", ["chess", "go"])
def test_muse_gateway_dot_names_map_to_native_dispatch_ids(game):
    for name in muse.MCP_NAMES:
        event = {"item": {"type": "function_call", "name": "mcp__" + game + "." + name, "arguments": "{}"}}
        assert muse.checked_response(event, game)["item"]["name"] == "mcp__" + game + "__" + name
        done = {"type": "response.function_call_arguments.done", "name": "mcp__" + game + "." + name, "arguments": "{}"}
        assert muse.checked_response(done, game)["name"] == "mcp__" + game + "__" + name


def test_restricted_muse_preserves_normal_install_policy():
    agent = make_agent("muse-code", model="muse-spark", version="latest", api_base_url="https://gateway.example/v1")
    agent.apply_tool_policy(POLICY)
    install = agent.install()
    assert install["check"] == "false"
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="test", files=[], skip_dirs=(),
        stage_src="", timeout_s=300, env={}, install=install))
    assert payload["task"]["install"] == install


def test_grok_meta_tools_only_on_grok_bridge():
    declarations = [{"type": "function", "function": {"name": n}} for n in ("search_tool", "use_tool", "bash")]
    assert bridge.restrict_tools({"tools": declarations})["tools"] == []
    assert bridge.restrict_tools({"tools": declarations}, grok_mcp=True)["tools"] == declarations[:2]
    summary = grok.summarise_stream([{"type": "available_commands", "tools": ["search_tool", "use_tool"]},
        {"type": "available_commands", "tools": ["search_tool", "use_tool", "chess__read_notes"]}])
    assert len(summary["init_tools"]) == 2 and len(summary["available_tools"]) == 3


@pytest.mark.skipif(os.environ.get("MUSE_GROK_RUNTIME_TESTS") != "1", reason="opt-in installed native CLI canaries")
@pytest.mark.parametrize("kind,attempt", [("muse", n) for n in ("mcp", "mcp_write", "bash", "read_file", "write_file", "write_todos", "subagent_spawn")]
                         + [("grok", n) for n in ("mcp", "mcp_write", "run_terminal_command", "write", "read_file", "workflow")])
def test_real_native_cli_restrictions(kind, attempt):
    result = subprocess.run(["docker", "run", "--rm", "-v", f"{ROOT}:/repo:ro",
        os.environ.get(f"RH_{kind.upper()}_TEST_IMAGE", f"python311-chess-{kind}:runtime"), "python", "/repo/tests/fixtures/muse_grok_native_probe.py", kind, attempt],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "NATIVE POLICY PASS" in result.stderr, result.stderr[-6000:]


def test_muse_meta_go_uses_response_gate_without_changing_upstream():
    agent = make_agent("muse-code", model="meta/muse-spark-1.3",
                       api_base_url="https://api.meta.ai/v1", api_key_env="META_API_KEY")
    from core import trial
    trial.load_task("go").check_agent(agent)
    assert agent.apply_tool_policy(POLICY) == []
    assert agent.setup() == []
    payload = json.loads(agent.blob(row={"id": "test", "tool_surface": "go"},
        prompt="unchanged", files=[], skip_dirs=(), stage_src="", grader_src="test",
        timeout_s=3600, env={}))
    cfg = payload["task"]["muse"]
    assert cfg["route"] == "gateway"
    assert cfg["model_slug"] == "muse-spark-1.3"
    assert cfg["base_url"] == "https://api.meta.ai/v1"
    assert cfg["api_key_env"] == "META_API_KEY"
    assert cfg["mcp_only"]
    assert "forwarder" in payload["modules"]
    assert "--disable-shell" in cfg["cli_flags"]
