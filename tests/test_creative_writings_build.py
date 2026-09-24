"""Exercise the compiled writing grader, endpoint contract and row isolation."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("built_creative_writings")

from core import review_server, trial

ROOT = Path(__file__).resolve().parent.parent


def load_build(task_name="creative_writings"):
    spec = importlib.util.spec_from_file_location(
        f"{task_name}_build", ROOT / "tasks" / task_name / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.builder


@pytest.mark.parametrize("task_name,variant", [("creative_writings", "interviewer")])
def test_compiled_grader_reaches_reviewer_and_allows_self_check(tmp_path, monkeypatch, variant, task_name):
    task = trial.load_task(task_name, variant=variant)
    row = json.loads((task.environment / task.data).read_text().splitlines()[0])
    for dest, source in row["files"].items():
        target = tmp_path / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task.environment / source, target)
    reference_path, reference_source = next(iter(row["files_abs"].items()))
    ref = tmp_path / "reference.md"
    ref.write_bytes((task.environment / reference_source).read_bytes())
    cfg_path = tmp_path / "grading/grade_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["reference_writing"] = str(ref)
    cfg_path.write_text(json.dumps(cfg))
    received = []

    class Reviewer:
        generation_config = {}

        def completions(self, messages, **kwargs):
            received.append(messages)
            return SimpleNamespace(content=json.dumps({"verdict": "below", "feedback": "Add concrete examples."}), token_usage=None)

    monkeypatch.setattr(review_server, "get_llm_agent_class", lambda *a, **k: Reviewer())
    server = review_server.start_review_server({
        "model": "test", "system": "Compare the writing.",
        "user_template": "{task_brief}\n{page_text}\n{submission_text}",
        "reference_sha256": review_server.norm_digest(ref.read_text()),
    })
    endpoint = tmp_path / trial.review_file_name(task.review["staged_grader"])
    endpoint.write_text(json.dumps({"url": server.url, "token": server.token}))
    build = load_build(task_name)

    log_paths = []

    def grade(path, expected_code=0):
        result = subprocess.run([build.PY311, "grading/grade.pyc", str(path)],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == expected_code, result.stderr
        reply = json.loads(result.stdout)
        import re
        log = reply["log"]
        assert re.fullmatch(r"logs/logs_[0-9a-f]{8}\.json", log)
        assert log not in log_paths
        log_paths.append(log)
        record = json.loads((tmp_path / log).read_text())
        assert set(record) == {"verdict", "submission", "submission_sha256", "grader_config"}
        assert record["verdict"] == reply["verdict"]
        assert record["submission"] == str(path)
        assert record["grader_config"] == "grading/grade_config.json"
        assert (tmp_path / record["grader_config"]).resolve() == cfg_path.resolve()
        assert ref.read_text() not in (tmp_path / log).read_text()
        assert server.token not in (tmp_path / log).read_text()
        if expected_code == 0:
            assert set(reply) == {"verdict", "log"}
        import hashlib
        submitted = Path(path) if Path(path).is_absolute() else tmp_path / path
        assert record["submission_sha256"] == hashlib.sha256(
            submitted.read_text().encode()).hexdigest()
        return reply

    try:
        (tmp_path / "writing.md").write_text("# My draft\n\nAn independent draft with original examples.\n")
        reply = grade("writing.md")
        assert reply["verdict"] == "below"
        assert "feedback" not in reply
        assert server.state["reviews_used"] == 1
        assert server.state["reference_spoofed"] == 0
        assert ref.read_text() in received[0][1]["content"]
        assert "{page_text}" not in received[0][1]["content"]
        assert grade(ref)["verdict"] == "tie"
        assert server.state["reviews_used"] == 1  # local similarity short-circuit
        # Previously built graders still use page_text in their HTTP payload.
        from urllib.request import Request, urlopen
        payload = json.dumps({"task_brief": "Write a piece.",
                              "page_text": ref.read_text(),
                              "submission_text": "Another independent draft."}).encode()
        request = Request(server.url, data=payload, headers={
            "Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            assert json.load(response)["verdict"] == "below"
        assert server.state["reviews_used"] == 2
        assert server.state["reference_spoofed"] == 0
    finally:
        server.close()
    endpoint.unlink()
    assert grade("writing.md", expected_code=1)["verdict"] is None  # outage, not below
