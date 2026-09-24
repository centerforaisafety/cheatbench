"""Restricted Kimi: actual game MCP schemas, native policy, and inference guard."""
import json
import os
from pathlib import Path
import subprocess
import tomllib
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest

from core.agents import make_agent, kimi_code_runner as runner
from core.sandbox import native_inference as bridge
from core.trial import load_task
from tasks.chess import build as chess_build, grader as chess_grader
from tests.test_chess_api_tool import rpc

ROOT = Path(__file__).resolve().parents[1]
POLICY = {"web_search": False, "web_fetch": False, "mcp_only": True}


def payload(game):
    agent = make_agent("kimi-code", model="kimi-k3", api_base_url="https://gateway.example/v1",
                       generation_config={"thinking_effort": "high"}, max_turns=8)
    assert agent.apply_tool_policy(POLICY) == []
    return json.loads(agent.blob(row={"id": "probe", "tool_surface": game}, prompt="Use the available tools.",
                                files=[], skip_dirs=(), stage_src="", grader_src="grader", timeout_s=60, env={}))


def checkout(game):
    from tests.test_chess_tools import FIXTURE
    return chess_build.checkout(FIXTURE, "chess")


@pytest.mark.parametrize("game", ["chess"])
def test_allowlist_and_routing_preserve_the_native_loop(game):
    task = payload(game)["task"]
    cfg = tomllib.loads(task["kimi"]["config_toml"])
    expected = {f"mcp__{game}__{name}" for name in runner.MCP_NAMES}
    assert set(cfg["tools"]["enabled"]) == expected and len(cfg["tools"]["enabled"]) == 9
    assert set(cfg["tools"]["disabled"]) == {"FetchURL", "WebSearch"}
    assert cfg["loop_control"]["max_steps_per_turn"] == 8
    assert task["row"]["episode_timeout_s"] == 60 and task["row"]["restricted_tools"]
    assert task["tools"] == []
    load_task(game).check_agent(make_agent("kimi-code", model="kimi-k3"))
    routed = json.loads(bridge.reroute_blob(json.dumps({"task": task}), "http://127.0.0.1:123/v1"))["task"]
    assert routed["kimi"]["api_base_url"] == routed["routing"]["api_base_url"] == "http://127.0.0.1:123/v1"
    assert routed["kimi"]["cli_env"] == task["kimi"]["cli_env"]
    assert routed["content"] == task["content"] and routed["kimi"]["cli_flags"] == []
    task["kimi"]["mcp_only"] = False
    with pytest.raises(ValueError):
        bridge.reroute_blob(json.dumps({"task": task}), "http://127.0.0.1:123/v1")


@pytest.mark.parametrize("game,module", [("chess", chess_grader)])
def test_actual_game_schemas_and_notes_roundtrip(tmp_path, game, module):
    for rel, content in checkout(game).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    row = {"id": "test", "tool_surface": game}
    server = runner.serve_grader(vars(module), row, {}, str(tmp_path))
    try:
        state, _ = module.make_tool(row, {}, str(tmp_path))
        specs = [module.tool_spec(row), *module.make_extra_tools(row, {}, str(tmp_path), state)]
        assert rpc(server, "tools/list")["tools"] == [{k: s[k] for k in ("name", "description", "inputSchema")} for s in specs]
        assert len(specs) == 9
        assert not rpc(server, "tools/call", {"name": "write_notes", "arguments": {"text": "proof"}})["isError"]
        assert "proof" in rpc(server, "tools/call", {"name": "read_notes"})["content"][0]["text"]
        for name in ("Bash", [], None):
            assert rpc(server, "tools/call", {"name": name})["isError"]
        cfg = json.loads(runner.mcp_json(server.url, game, mcp_only=True))["mcpServers"][game]
        assert set(cfg["enabledTools"]) == runner.MCP_NAMES and cfg["deferred"] is False
        assert cfg["toolTimeoutMs"] >= 210_000
    finally:
        server.close()


def test_complete_roster_and_compaction_unchanged_but_bad_declarations_rejected():
    allowed = {"mcp__go__" + n for n in runner.MCP_NAMES}
    declared = [{"type": "function", "function": {"name": n}} for n in sorted(allowed)]
    doc = {"tools": declared, "model": "same", "messages": [], "thinking": {"effort": "high"}}
    assert runner.restricted_request(doc, allowed) is doc
    assert bridge.restrict_tools(doc, mcp_server="go") == doc
    assert runner.restricted_request({"messages": []}, allowed) == {"messages": []}
    for tools in (declared[:-1], declared + declared[:1], [None], ["Bash"], [{"type": "web_search"}],
                  [{"type": "function", "function": {"name": "Bash"}}],
                  [{"type": "function", "function": {"name": "mcp__chess__read_notes"}}], None):
        with pytest.raises(ValueError, match="policy breach"):
            runner.restricted_request({"tools": tools}, allowed)


def test_bad_roster_never_contacts_upstream():
    upstream = Mock(path="/v1")
    server = runner.KimiForwarder(upstream=upstream, allowed_tools={"mcp__go__" + n for n in runner.MCP_NAMES}).start()
    try:
        request = urllib.request.Request(server.base_url + "/chat/completions", data=b'{"tools":[{"type":"function","function":{"name":"Bash"}}]}',
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 403
        upstream.request.assert_not_called()
        assert "policy breach" in server.policy_error
    finally:
        server.close()


@pytest.mark.skipif(os.environ.get("RH_KIMI_POLICY_TEST") != "1", reason="opt-in Enroot Kimi 0.43.0 native canaries")
def test_real_native_cli_restrictions(tmp_path):
    from core import sandbox
    cases = [{"game": game, "attempt": attempt, "task": payload(game)["task"], "files": checkout(game)}
             for game in ("chess",) for attempt in ("mcp", "Bash", "Write", "Edit", "Read", "Agent", "AgentSwarm", "Skill", "mcp__other__read_notes")]
    blob = {"task": {"cases": cases}, "code": (ROOT / "tests/fixtures/kimi_native_policy_probe.py").read_text(),
            "modules": {"runner": (ROOT / "core/agents/kimi_code_runner.py").read_text(),
                        "forwarder": (ROOT / "core/agents/forwarder.py").read_text(),
                        "chess": (ROOT / "tasks/chess/grader.py").read_text()}}
    bootstrap = 'import json,sys;d=json.load(sys.stdin);exec(d["code"],{"TASK":d["task"],"MODULES":d["modules"]})'
    sandbox.ensure_path()
    argv = sandbox.container_argv(os.environ["RH_KIMI_TEST_IMAGE"], bootstrap, private_net=True, pass_key=False)
    # Only loopback is brought up in this fresh network namespace; no egress.
    split = argv.index("--") + 1
    argv[split:] = ["sh", "-c", 'ip link set lo up && exec "$@"', "probe", *argv[split:]]
    result = subprocess.run(argv, input=json.dumps(blob), text=True, capture_output=True, env=sandbox.spawn_env(""), timeout=900)
    (tmp_path / "native.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-12000:]
    # Enroot may merge the payload's stdout into stderr; use a unique record key.
    records = [json.loads(line) for line in (result.stdout + "\n" + result.stderr).splitlines()
               if line.startswith('{"native_policy_cases":')]
    assert len(records) == 1, result.stderr[-3000:]
    report = records[0]
    (tmp_path / "native-policy.json").write_text(json.dumps(report, indent=2))
    assert len(report["native_policy_cases"]) == len(cases)
    assert all(c["passed"] for c in report["native_policy_cases"])
