"""Version precedence, adapter handoff, and isolation without model/network calls."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import trial
from core.agents import config as agent_config
from core.agents import AgentFactory, make_agent


def config(tmp_path, versions=None):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(versions if versions is not None else {"codex": {"version": "0.154.0"}}))
    return path


@pytest.mark.parametrize("task,cli,version,source", [
    ({}, "", "0.154.0", "agents.yaml"),
    ({"codex": {"version": "0.153.0"}}, "", "0.153.0", "task.yaml:agent_config"),
    ({"codex": {"version": "0.153.0"}}, "0.152.0", "0.152.0", "--agent-version"),
    ({"codex": {"version": "latest"}}, "", "latest", "task.yaml:agent_config"),
    ({"codex": {"version": "0.153.0"}}, "latest", "latest", "--agent-version"),
])
def test_precedence(tmp_path, task, cli, version, source):
    path = config(tmp_path)
    choice = agent_config.resolve_version("codex", task_config=task, cli_version=cli, config_path=path)
    assert (choice.version, choice.source) == (version, source)
    assert choice.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_task_overrides_do_not_leak(tmp_path):
    path = config(tmp_path)
    for name, overrides in [("one", {"codex": {"version": "latest"}}), ("two", {})]:
        task_path = tmp_path / name
        task_path.mkdir()
        (task_path / "task.yaml").write_text(yaml.safe_dump({"name": name, "agent_config": overrides}))
    first = trial.load_task(str(tmp_path / "one"))
    second = trial.load_task(str(tmp_path / "two"))
    assert agent_config.resolve_version("codex", task_config=first.agent_config, config_path=path).version == "latest"
    assert agent_config.resolve_version("codex", task_config=second.agent_config, config_path=path).version == "0.154.0"
    assert yaml.safe_load(path.read_text()) == {"codex": {"version": "0.154.0"}}


@pytest.mark.parametrize("raw", [None, [], "latest", {"typo": "1.0.0"},
                                     {"codex": 1.2}, {"codex": True}, {"codex": "^0.154.0"},
                                     {"codex": "0.154.0; echo bad"}, {"codex": {"version": "latest", "typo": 1}},
                                     {"codex": {"version": False}}, {"codex": {"version": "^0.154.0"}}])
def test_bad_configuration_fails_before_install(tmp_path, raw):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit):
        agent_config.resolve_version("codex", config_path=path)


def test_missing_file_and_missing_adapter_fail(tmp_path):
    with pytest.raises(SystemExit, match="cannot load"):
        agent_config.resolve_version("codex", config_path=tmp_path / "missing.yaml")
    with pytest.raises(SystemExit, match="no version"):
        agent_config.resolve_version("gemini-cli", config_path=config(tmp_path))


def test_task_rejects_invalid_override(tmp_path):
    (tmp_path / "task.yaml").write_text('agent_config: {codex: {version: false}}\n')
    with pytest.raises(SystemExit, match="exact version"):
        trial.load_task(str(tmp_path))


@pytest.mark.parametrize("old_request,actual,requested,allowed", [
    ("0.154.0", "0.154.0", "0.154.0", True),
    ("latest", "0.154.0", "0.154.0", True),
    ("0.153.0", "0.153.0", "0.154.0", False),
    ("0.154.0", "0.153.0", "0.154.0", False),
    ("0.154.0", "0.154.0", "latest", False),
    ("latest", "0.154.0", "latest", True),
])
def test_resume_preserves_actual_version_and_prevents_mixed_pins(
        tmp_path, old_request, actual, requested, allowed):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"agent": "codex", "agent_version_requested": old_request,
                            "agent_version": actual}))
    if allowed:
        assert agent_config.resume_version(p, "codex", requested) == actual
    else:
        with pytest.raises(SystemExit, match="new --run-id"):
            agent_config.resume_version(p, "codex", requested)
    with pytest.raises(SystemExit, match="cannot resume"):
        agent_config.resume_version(p, "claude-sdk", requested)


@pytest.mark.parametrize("name", AgentFactory.names())
def test_latest_forces_install_and_is_not_an_exact_pin(name):
    if AgentFactory.get_agent_class(name).FIXED_VERSION:
        with pytest.raises(ValueError, match="pinned"):
            make_agent(name, model="test", version="latest")
        return
    agent = make_agent(name, model="test", version="latest")
    spec = agent.install()
    assert spec["check"] == "false"
    assert not spec["pinned_version"]
    assert agent.pinned_version is None


@pytest.mark.parametrize("name,binary,output", [
    ("codex", "codex", "codex-cli 0.154.0"),
    ("claude-sdk", "claude", "0.154.0 (Claude Code)"),
    ("gemini-cli", "gemini", "0.154.0"),
    ("grok-build", "grok", "grok 0.154.0 (build)"),
    ("kimi-code", "kimi", "kimi, version 0.154.0"),
])
def test_exact_pin_rejects_wrong_cached_binary(tmp_path, name, binary, output):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / binary
    executable.write_text("#!/bin/sh\nprintf '%s\\n' '" + output + "'\n")
    executable.chmod(0o755)
    # The Claude check also imports its Python SDK; model it as available.
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "HOME": str(tmp_path)}
    right = make_agent(name, model="test", version="0.154.0").install()
    wrong = make_agent(name, model="test", version="0.153.0").install()
    assert subprocess.run(["bash", "-c", right["check"]], env=env).returncode == 0
    assert subprocess.run(["bash", "-c", wrong["check"]], env=env).returncode != 0


def test_shipped_defaults_cover_every_adapter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert agent_config.DEFAULT_CONFIG == ROOT / "configs" / "agents.yaml"
    defaults = yaml.safe_load(agent_config.DEFAULT_CONFIG.read_text())
    assert set(defaults) == set(AgentFactory.names())
    for name in AgentFactory.names():
        choice = agent_config.resolve_version(name)
        adapter = make_agent(name, model="test", version=choice.version)
        assert adapter.install()
        assert adapter.pinned_version == choice.version


@pytest.mark.parametrize("override,expected,source", [
    ("", "0.153.0", "task.yaml:agent_config"),
    ("latest", "latest", "--agent-version"),
])
def test_run_hands_selected_version_to_adapter_and_logs_it(tmp_path, monkeypatch, capsys,
                                                          override, expected, source):
    spec = importlib.util.spec_from_file_location("harness_test_run", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    task = SimpleNamespace(agent_config={"codex": {"version": "0.153.0"}}, name="test", variant="", rows=lambda: [])
    monkeypatch.setattr(run.trial, "load_task", lambda *a: task)
    monkeypatch.setattr(run.task_build, "ensure_task_built", lambda *a, **kw: ([], {}))
    cfg = {"model": "test", "generation_config": {}, "max_turns": 0,
           "permission_mode": "", "base_url": None, "api_key_env": None,
           "api_base_url": None, "extra_body": None}
    monkeypatch.setattr(run, "load_config", lambda *a: cfg)
    monkeypatch.setattr(run.judging, "load_judge_config", lambda *a: {})
    class StopBeforeInstall(Exception):
        pass
    def make(name, **kwargs):
        assert name == "codex" and kwargs["version"] == expected
        assert make_agent(name, **kwargs).requested_version == expected
        raise StopBeforeInstall
    monkeypatch.setattr(run, "make_agent", make)
    args = SimpleNamespace(task="test", variant="", models_config="unused", model="test",
                           judge="test", agent="codex", agent_version=override,
                           agents_config=str(config(tmp_path)), run_id="test",
                           output_dir=str(tmp_path / "out"), redo=False, limit=0, repeat=1,
                           no_auto_build=False, rebuild_task=False)
    with pytest.raises(StopBeforeInstall):
        asyncio.run(run.main_async(args))
    output = capsys.readouterr().out
    assert f"adapter=codex requested_version={expected} source={source}" in output


def test_terminus_fixed_source_default_and_overrides():
    from core.agents.terminus_2 import HARBOR_COMMIT
    assert agent_config.resolve_version("terminus-2").version == HARBOR_COMMIT
    assert agent_config.resolve_version("terminus-2", cli_version=HARBOR_COMMIT).version == HARBOR_COMMIT
    for invalid in ("latest", "0.22.0", "main", "0" * 40):
        with pytest.raises(SystemExit, match="always uses pinned source"):
            agent_config.resolve_version("terminus-2", cli_version=invalid)
        with pytest.raises(SystemExit, match="always uses pinned source"):
            agent_config.resolve_version("terminus-2", task_config={"terminus-2": {"version": invalid}})


def test_python_sdk_prerelease_pin():
    assert agent_config.resolve_version("deepseek-harness").version == "0.1.5rc1"
    for invalid in ("0.1.5rc", "0.1.5rc1;echo", ">=0.1.5rc1"):
        with pytest.raises(SystemExit):
            agent_config.validate_version(invalid, "test", agent="deepseek-harness")


def test_installer_cannot_succeed_with_a_different_binary_version(monkeypatch):
    adapter = make_agent("codex", model="test", version="0.154.0")
    monkeypatch.setattr(adapter, "install_check", lambda: "true")
    monkeypatch.setattr(adapter, "install_script", lambda: "true")
    monkeypatch.setattr(adapter, "VERSION_CMD", "printf 0.156.1")
    spec = adapter.install()
    assert subprocess.run(["bash", "-c", spec["check"]], capture_output=True).returncode != 0
    result = subprocess.run(["bash", "-c", spec["install"]], capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not match pin" in result.stderr
