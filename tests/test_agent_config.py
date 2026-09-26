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


@pytest.mark.parametrize("harness_override,cli,version,source", [
    (False, "", "0.154.0", "models.yaml:harness"),
    (True, "", "latest", "--harness"),
    (False, "0.153.0", "0.153.0", "--harness-version"),
    (True, "0.153.0", "0.153.0", "--harness-version"),
    (False, "latest", "latest", "--harness-version"),
])
def test_precedence_without_agents_file(tmp_path, harness_override, cli, version, source):
    path = tmp_path / "models.yaml"
    harness = {"name": "codex", "version": "0.154.0"}
    path.write_text(yaml.safe_dump({"test": {"model": "test", "harness": harness}}))
    choice = agent_config.resolve_version("codex", harness_override=harness_override,
        cli_version=cli, model_harness=harness, models_path=path)
    assert (choice.version, choice.source) == (version, source)
    if source == "models.yaml:harness":
        assert choice.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        assert choice.config_path == choice.config_sha256 == ""


@pytest.mark.parametrize("invalid", [False, 1.2, "^0.154.0", "0.154.0; echo bad", "main"])
def test_bad_version_fails_before_install(invalid):
    with pytest.raises(SystemExit):
        agent_config.validate_version(invalid, "test", agent="codex")


def test_different_harness_does_not_inherit_model_pin():
    choice = agent_config.resolve_version("gemini-cli", harness_override=True,
        model_harness={"name": "codex", "version": "0.154.0"})
    assert choice.version == "latest"


def test_task_rejects_removed_override(tmp_path):
    (tmp_path / "task.yaml").write_text('agent_config: {codex: {version: "0.154.0"}}\n')
    with pytest.raises(SystemExit, match="unknown key.*agent_config"):
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


@pytest.mark.parametrize("name", AgentFactory.names())
def test_harness_override_works_without_any_config_file(monkeypatch, tmp_path, name):
    monkeypatch.chdir(tmp_path)
    choice = agent_config.resolve_version(name, harness_override=True)
    adapter = make_agent(name, model="test", version=choice.version)
    assert adapter.install()
    assert choice.version == (adapter.FIXED_VERSION or "latest")


@pytest.mark.parametrize("harness,override,expected,source", [
    ("", "", "0.153.0", "models.yaml:harness"),
    ("codex", "", "latest", "--harness"),
    ("codex", "0.154.0", "0.154.0", "--harness-version"),
    ("", "latest", "latest", "--harness-version"),
])
def test_run_hands_selected_version_to_adapter_and_logs_it(tmp_path, monkeypatch, capsys,
                                                          harness, override, expected, source):
    spec = importlib.util.spec_from_file_location("harness_test_run", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    task = SimpleNamespace(name="test", variant="", rows=lambda: [])
    monkeypatch.setattr(run.trial, "load_task", lambda *a: task)
    monkeypatch.setattr(run.task_build, "ensure_task_built", lambda *a, **kw: ([], {}))
    cfg = {"model": "test", "generation_config": {}, "max_turns": 0,
           "permission_mode": "", "base_url": None, "api_key_env": None,
           "api_base_url": None, "extra_body": None,
           "harness": {"name": "codex", "version": "0.153.0"}}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"test": cfg}))
    monkeypatch.setattr(run, "load_config", lambda *a: cfg)
    monkeypatch.setattr(run.judging, "load_judge_config", lambda *a: {})
    class StopBeforeInstall(Exception):
        pass
    def make(name, **kwargs):
        assert name == "codex" and kwargs["version"] == expected
        assert make_agent(name, **kwargs).requested_version == expected
        raise StopBeforeInstall
    monkeypatch.setattr(run, "make_agent", make)
    argv = ["test", "--model", "test", "--models-config", str(path),
            "--output-dir", str(tmp_path / "out"), "--run-id", "test"]
    if harness:
        argv += ["--harness", harness]
    if override:
        argv += ["--harness-version", override]
    with pytest.raises(StopBeforeInstall):
        asyncio.run(run.main_async(run.build_parser().parse_args(argv)))
    assert f"adapter=codex requested_version={expected} source={source}" in capsys.readouterr().out


def test_removed_agents_config_is_rejected():
    import run
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["chess", "--model", "gpt-6-sol",
                                      "--agents-config", "old.yaml"])


def test_terminus_fixed_source_default_and_overrides():
    from core.agents.terminus_2 import HARBOR_COMMIT
    assert agent_config.resolve_version("terminus-2").version == HARBOR_COMMIT
    assert agent_config.resolve_version("terminus-2", cli_version=HARBOR_COMMIT).version == HARBOR_COMMIT
    for invalid in ("latest", "0.22.0", "main", "0" * 40):
        with pytest.raises(SystemExit, match="always uses pinned source"):
            agent_config.resolve_version("terminus-2", cli_version=invalid)


def test_python_sdk_prerelease_pin():
    assert agent_config.resolve_version("deepseek-harness", cli_version="0.1.5rc1").version == "0.1.5rc1"
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
