"""Task dependency installation must work across every adapter."""
from __future__ import annotations

import ast
import json
import os
import shlex
import sys
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import trial
from core.agents import make_agent


ADAPTERS = [
    ("claude-sdk", "claude_sdk", "anthropic/claude-fable-5-1"),
    ("codex", "codex", "openai/gpt-6-astra"),
    ("gemini-cli", "gemini_cli", "gemini/gemini-3.1-pro-preview"),
    ("grok-build", "grok_build", "openrouter/x-ai/grok-4.6"),
    ("kimi-code", "kimi_code", "openrouter/moonshotai/kimi-k3"),
    ("muse-code", "muse_code", "meta/muse-spark-1.3"),
]


@pytest.mark.parametrize("name,module,model", ADAPTERS)
@pytest.mark.parametrize("install", [None, {"name": "example", "check": "true", "install": "false"}])
def test_actual_trial_arguments_reach_every_adapter(name, module, model, install):
    agent = make_agent(name, model=model)
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    # Derive the call from run_trial: a new unconditional keyword must be
    # supported by every adapter, including tasks without dependencies.
    tree = ast.parse((ROOT / "core/trial.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "blob")
    class Task:
        def grader_src(self):
            return ""
    task = Task()
    task.install = install
    ns = dict(agent=agent, task=task, row={"id": "test"}, text="Draw an SVG",
              files=[], skip_dirs=(), timeout_s=60, env={}, install=None, ROOT=ROOT)
    blob = eval(compile(ast.Expression(call), "<run_trial blob call>", "eval"), ns)
    assert json.loads(blob)["task"]["task_install"] == install


def runner_helpers(module):
    # Loading a runner normally redirects stdout. Extract its actual install
    # helpers so this test can exercise subprocess execution without that setup.
    tree = ast.parse((ROOT / f"core/agents/{module}_runner.py").read_text())
    names = {"INSTALL_TIMEOUT_S", "_SECRET_ENV_SUBSTRINGS", "_SECRET_ENV_PREFIXES"}
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in {"task_install_env", "install_agent", "_sh"})
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    ns = dict(os=os, time=time, subprocess=subprocess, _log=lambda message: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<runner install helpers>", "exec"), ns)
    return ns


@pytest.mark.parametrize("name,module,model", ADAPTERS)
def test_install_shell_filters_credentials_and_records_outcomes(name, module, model, monkeypatch, tmp_path):
    helpers = runner_helpers(module)
    monkeypatch.setattr(os, "environ", {"PATH": os.defpath, "OPENAI_API_KEY": "test-only",
        "CUSTOM_BASE_URL": "https://example.invalid", "PIP_INDEX_URL": "https://packages.invalid"})
    env = helpers["task_install_env"]()
    assert "OPENAI_API_KEY" not in env and "CUSTOM_BASE_URL" not in env
    assert env["PIP_INDEX_URL"] == "https://packages.invalid"
    assert os.environ["OPENAI_API_KEY"] == "test-only"
    marker = shlex.quote(str(tmp_path / "installed"))
    spec = {"name": "test", "check": f"test -f {marker}",
            "install": f'test -z "$OPENAI_API_KEY" && printf installed > {marker}'}
    install = helpers["install_agent"]
    assert install(spec, env=env)["status"] == "installed"
    assert install({**spec, "install": "exit 1"}, env=env)["status"] == "present"
    assert install({"name": "broken", "check": "false", "install": "exit 7"}, env=env)["status"] == "failed"
    assert install(None, env=env) is None


def test_task_dependency_configuration(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    config = root / "task.yaml"
    config.write_text("name: example\ninstall:\n  check: 'true'\n  install: 'false'\n")
    assert trial.load_task(str(root)).install == {"name": "task", "check": "true", "install": "false"}
    config.write_text("name: example\ninstall:\n  check: 'true'\n")
    with pytest.raises(SystemExit, match="non-empty"):
        trial.load_task(str(root))
