"""Real session shapes and live runner hooks for the three additional CLIs, on this task's grader."""
import copy
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.agents import make_agent, session_observer
from core.agents import gemini_cli_runner as gemini
from core.agents import grok_build_runner as grok
from core.agents import muse_code_runner as muse
from tasks.prime_factorization import grader


def observer_in(root):
    row = {"id": "test", "number": "221",
           "live_log_dir": str(root / "logs"),
           "live_result_path": str(root / "result.jsonl"),
           "cache_dir": str(root / "cache")}
    state, _ = grader.make_tool(row, {}, str(root))
    return grader.make_observer(row, {}, str(root), state)


def muse_event(event):
    return {"payload_type": "runtime.session", "payload": {"kind": "run", "event": event}}


def native_records(name, output):
    if name == "grok-build":
        return [{"type": "system", "content": "HIDDEN_SETUP"},
                {"type": "assistant", "tool_calls": [{"id": "read1", "name": "read_file", "arguments": {"path": "example"}}]},
                {"type": "tool_result", "tool_call_id": "read1", "content": output}]
    if name == "muse-code":
        return [{"payload_type": "runtime.session.metadata", "payload": {"record": {"setup": "HIDDEN_SETUP"}}},
                muse_event({"kind": "assistant_tool_calls_committed", "tool_calls": [{"call_id": "read1", "name": "read_file", "args": '{"path":"example"}'}]}),
                muse_event({"kind": "tool_result_batch_committed", "results": [{"tool_call_id": "read1", "text": output}]})]
    return [{"$set": {"system": "HIDDEN_SETUP"}},
            {"id": "m1", "type": "gemini", "toolCalls": [{"id": "read1", "name": "read_file", "args": {"path": "example"}}]},
            {"id": "m1", "type": "message_update", "toolCalls": [{"id": "read1", "name": "read_file", "args": {"path": "example"},
                "result": [{"functionResponse": {"name": "read_file", "response": {"output": output}}}], "status": "success"}]}]


@pytest.mark.parametrize("name", ["grok-build", "gemini-cli", "muse-code"])
def test_native_calls_results_spill_and_hide_setup(tmp_path, name):
    observer = observer_in(tmp_path)
    records = native_records(name, "READABLE_SENTINEL\n" * 1500)
    before = copy.deepcopy(records)
    for record in records:
        observer.observe(record)
    assert records == before
    observer.finish()
    log = (observer.log_dir / "turns.log").read_text()
    assert "HIDDEN_SETUP" not in log and "[tool] read_file" in log
    assert "stored at " in log and "READABLE_SENTINEL" not in log
    assert "READABLE_SENTINEL" in (observer.cache / "live_read1.txt").read_text()
    assert len(observer.messages) == 2 and observer.state["score"] == 0


def test_gemini_updates_and_legacy_snapshots_do_not_duplicate_results(tmp_path):
    observer = observer_in(tmp_path)
    records = native_records("gemini-cli", "actual result")
    for record in records + records[-1:]:
        observer.observe(record)
    observer.observe({"messages": [{**records[-1], "type": "gemini"},
                                   {"id": "secret", "type": "user", "content": "HIDDEN"}]})
    observer.observe({"id": "secret", "type": "message_update", "content": "HIDDEN"})
    assert len(observer.messages) == 2
    assert "actual result" in (observer.log_dir / "turns.log").read_text()
    assert "HIDDEN" not in (observer.log_dir / "turns.log").read_text()


@pytest.mark.parametrize("name,runner", [("grok-build", grok), ("gemini-cli", gemini), ("muse-code", muse)])
def test_main_constructs_and_finishes_task_observer(tmp_path, monkeypatch, name, runner):
    expected = observer_in(tmp_path)
    server = SimpleNamespace(grader_state=expected.state, server_address=("127.0.0.1", 0), close=lambda: None)
    agent = make_agent(name, model="test")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    blob = json.loads(agent.blob(row=expected.row, prompt="solve", files=[], skip_dirs=(),
                                grader_src=(ROOT / "tasks/prime_factorization/grader.py").read_text(),
                                stage_src="def write(*args): return {}", timeout_s=10, env={}, install=None))
    output = io.StringIO()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "WORKDIR", str(tmp_path))
    monkeypatch.setattr(runner, "_OUT", output)
    monkeypatch.setattr(runner, "serve_grader", lambda *a: server)
    monkeypatch.setattr(runner, "install_agent", lambda *a, **k: None)

    def run(task, install, grader, **options):
        observer = options["observer"]
        if name != "grok-build":
            assert callable(options["observer_ns"]["observe_session"])
        for record in native_records(name, "actual result"):
            observer.observe(record)
        return {"grader_state": grader.grader_state, "error": None}

    monkeypatch.setattr(runner, "run", run)
    runner.main(blob["task"], blob["modules"])
    record = json.loads(output.getvalue())
    assert record["error"] is None
    assert record["grader_state"]["mirrored_messages"] == 2
    assert record["grader_state"]["changed_files"] == []
    assert (expected.log_dir / "transcript.json").exists()


def test_muse_committed_reasoning_omits_encrypted_state(tmp_path):
    observer = observer_in(tmp_path)
    for line in (ROOT / "tests/fixtures/muse_code_session_1_1_reasoning.jsonl").read_text().splitlines():
        observer.observe(json.loads(line))
    text = (observer.log_dir / "turns.log").read_text()
    assert text.count("Solving a Chinese Remainder") == 1
    assert "312658" in text and "encrypted" not in text
    assert "Do not use any tools" not in text


def test_session_tail_partial_lines_and_last_flush(tmp_path):
    observer = observer_in(tmp_path)
    path = tmp_path / "session.jsonl"
    raw = json.dumps(native_records("grok-build", "final result")[-1]).encode()
    path.write_bytes(raw[:30])
    tail = session_observer.SessionObserver(lambda: str(path), observer)
    tail.drain()
    assert not observer.messages
    with path.open("ab") as f:
        f.write(raw[30:] + b"\n")
    tail.thread.start()
    tail.close()
    assert tail.error is None and len(observer.messages) == 1
    tail.drain()
    assert len(observer.messages) == 1


def test_observer_failure_is_reported_not_silently_ignored(tmp_path):
    class Broken:
        def observe(self, record):
            raise OSError("cannot write mirror")
    path = tmp_path / "session.jsonl"
    path.write_text('{}\n')
    rc, error = session_observer.observe_session(lambda: (0, None), lambda: str(path), Broken())
    assert rc == 0 and "cannot write mirror" in error
    tail = grok.ChatHistoryTail(str(path), Broken())
    assert tail.poll() == [{}]
    assert "cannot write mirror" in tail.error
