"""Run, judge, and task preflight share duplicate-key rejection."""
from types import SimpleNamespace

import pytest

import run
from core import judge, llm_agents, task_build, trial
from tasks.sycophancy import run_chat


@pytest.mark.parametrize("consumer", [
    lambda path: run.load_config(path, "m"),
    lambda path: judge.load_judge_config(path, "m"),
    lambda path: llm_agents.get_agent_config("m", str(path)),
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_model_consumers_reject_duplicates_before_selection(tmp_path, consumer, wrapped):
    path = tmp_path / "models.yaml"
    body = "m: {model: openai/first}\nm: {model: openai/second}\n"
    if wrapped:
        body = "models:\n" + "".join("  " + line + "\n" for line in body.splitlines())
    path.write_text(body)
    with pytest.raises(SystemExit, match="duplicate key 'm'"):
        consumer(path)


def test_chat_runner_also_rejects_duplicate_models(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  m: {}\n  m: {}\n")
    with pytest.raises(SystemExit, match="duplicate key 'm'"):
        run_chat.load_config(path, "m")


@pytest.mark.parametrize("consumer", [
    lambda root: trial.load_task(str(root)),
    lambda root: task_build._settings(SimpleNamespace(root=root, variant="")),
])
def test_task_preflight_rejects_duplicate_settings(tmp_path, consumer):
    (tmp_path / "task.yaml").write_text("timeout_s: 10\ntimeout_s: 20\n")
    with pytest.raises(SystemExit, match="duplicate key 'timeout_s'"):
        consumer(tmp_path)
