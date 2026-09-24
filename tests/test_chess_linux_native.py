"""Linux/Enroot wiring and unchanged Chess payloads; no paid model calls."""
from pathlib import Path
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core import sandbox
from core.agents import make_agent
from core.sandbox import build, egress, native_inference, runtime
from core.trial import load_task
MODELS = {
    "fable": ("claude-fable-5-1", "claude-sdk"), "astra": ("gpt-6-astra", "codex"),
    "sol": ("gpt-5.6-sol", "codex"), "opus": ("claude-opus-5", "claude-sdk"),
    "gemini": ("gemini-3.8-flash", "gemini-cli"),
    "deepseek": ("deepseek-v4-pro-litellm", "deepseek-harness"),
    "muse": ("muse-spark-litellm", "muse-code"), "grok": ("grok-4.6", "grok-build"),
    "kimi": ("kimi-k3", "kimi-code"),
}


@pytest.mark.parametrize("model", list(MODELS))
def test_linux_transport_preserves_native_payloads(model):
    alias, adapter = MODELS[model]
    agent = make_agent(adapter, model=alias, api_base_url="https://gateway.example/v1")
    agent.apply_tool_policy(load_task("chess").tools)
    load_task("chess").check_agent(agent)
    original = agent.blob(row={"id": "probe", "tool_surface": "chess"}, prompt="same task",
                          files=[], skip_dirs=(), grader_src="grader", stage_src="",
                          timeout_s=3600, env={}, install={"check": "check-pin", "install": "download"})
    before = json.loads(original)
    after = json.loads(native_inference.reroute_blob(original, "http://127.0.0.1:43210/v1"))
    expected = copy.deepcopy(before)
    task = expected["task"]
    url = "http://127.0.0.1:43210/v1"
    task.setdefault("env", {}).update(ANTHROPIC_BASE_URL=url, OPENAI_BASE_URL=url)
    task.setdefault("routing", {})["api_base_url"] = url
    for key in ("codex", "gemini", "deepseek_harness", "muse", "grok"):
        if key in task:
            task[key]["base_url"] = url
    if task.get("codex", {}).get("provider"):
        task["codex"]["provider"]["base_url"] = url
    if "gemini" in task:
        task["env"]["GEMINI_BASE_URL"] = url
    if "deepseek_harness" in task:
        task["env"]["DEEPSEEK_BASE_URL"] = url
    if "kimi" in task:
        task["kimi"]["api_base_url"] = url
    assert after == expected  # Includes prompts, limits, native flags and all runner source.


@pytest.mark.parametrize("grok", [False, True])
def test_linux_bridge_does_not_choose_network_policy_and_closes_it(tmp_path, monkeypatch, grok):
    bridge = Mock(server_port=43210, local_base_url="http://127.0.0.1:43210/v1")
    factory = Mock(return_value=bridge)
    monkeypatch.setattr(native_inference, "InferenceBridge", factory)
    sb = runtime.EpisodeSandbox(1, (), tmp_path)
    sb.lock_egress = Mock(return_value="guard")
    task = {"tools": [], **({"grok": {"mcp_only": True}} if grok else {})}
    routed, port = sb.route_mcp_only(json.dumps({"task": task}), "https://model.example/v1", "mock-key")
    assert port == 43210
    assert factory.call_args.kwargs["grok_mcp"] is grok
    sb.lock_egress.assert_not_called()
    assert json.loads(routed)["task"]["routing"]["api_base_url"] == bridge.local_base_url
    sb.close()
    bridge.close.assert_called_once()


def test_bridge_url_matches_dynamic_listener(tmp_path):
    server = native_inference.InferenceBridge("https://model.example/v1", "dummy", tmp_path / "events.jsonl")
    try:
        assert server.local_base_url == f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.close()


@pytest.mark.parametrize("allow_dns", [True, False])
def test_firewall_respects_explicit_dns_policy(monkeypatch, allow_dns):
    calls = []
    monkeypatch.setattr(egress, "_raise_ambient_caps", lambda *_: None)
    monkeypatch.setattr(egress.subprocess, "run", lambda argv, **kw: (
        calls.append(argv) or SimpleNamespace(returncode=0, stderr="")))
    egress._firewall(12345, (), allow_dns=allow_dns)
    dns = [argv for argv in calls if "--dport" in argv and argv[argv.index("--dport") + 1] == "53"]
    assert bool(dns) is allow_dns
    assert not any("DNAT" in argv for argv in calls)
    assert ["iptables", "-P", "OUTPUT", "DROP"] in calls
    assert ["ip6tables", "-P", "OUTPUT", "DROP"] in calls


def test_cache_uses_main_builder_and_preserves_default(tmp_path, monkeypatch):
    recipe = tmp_path / "Dockerfile"
    recipe.write_text("FROM python:3.11-slim\n")
    make = Mock(side_effect=lambda image, **kwargs: Path(image).write_bytes(b"image"))
    monkeypatch.setattr(build, "build_image", make)
    assert build.ensure_image(str(tmp_path / "native.sqsh"), dockerfile=recipe, log=lambda *_: None)
    assert make.call_args.kwargs["dockerfile"] == recipe
    make.reset_mock()
    assert build.ensure_image(str(tmp_path / "ordinary.sqsh"), log=lambda *_: None)
    assert "dockerfile" not in make.call_args.kwargs


@pytest.mark.parametrize("name", ["geoguessr", "subset_sum"])
def test_other_tasks_keep_their_existing_runtime_policy(name):
    task = load_task(name)
    assert not task.tools.get("mcp_only")
    assert sandbox.EpisodeSandbox is runtime.EpisodeSandbox
