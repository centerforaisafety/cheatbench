"""Real session shapes and live runner hooks for v6's three additional CLIs."""
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
from tasks.subset_sum import grader


def observer_in(root):
    row = {"id": "test", "numbers": [3, 5], "target": 8,
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
                                grader_src=(ROOT / "tasks/subset_sum/grader.py").read_text(),
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


def test_session_tail_legacy_json_waits_for_complete_snapshot(tmp_path):
    observer = observer_in(tmp_path)
    path = tmp_path / "session.json"
    path.write_text('{"messages":[')
    tail = session_observer.SessionObserver(lambda: str(path), observer)
    tail.drain()
    path.write_text(json.dumps({"messages": [{"id": "m", "type": "gemini", "content": "complete"}]}))
    tail.drain()
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


@pytest.mark.parametrize("name,runner", [("grok-build", grok), ("gemini-cli", gemini), ("muse-code", muse)])
def test_runner_mirrors_before_cli_exit_and_keeps_native_transcript(tmp_path, monkeypatch, name, runner):
    observer = observer_in(tmp_path)
    records = native_records(name, "visible before CLI exit")
    agent = make_agent(name, model="test")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    blob = json.loads(agent.blob(row={"id": "test"}, prompt="solve", files=[], skip_dirs=(),
                                grader_src="", stage_src="", timeout_s=10, env={}, install=None))
    task = blob["task"]
    path = tmp_path / "session.jsonl"
    monkeypatch.setattr(runner, "WORKDIR", str(tmp_path))
    captured = []
    monkeypatch.setattr(runner, "_emit_msg", captured.append)

    def fake_cli(*args):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        if name == "grok-build":
            tail = args[4]
            tail.path = str(path)
            tail.poll()
        deadline = time.monotonic() + 3
        log = observer.log_dir / "turns.log"
        while time.monotonic() < deadline:
            if log.exists() and "visible before CLI exit" in log.read_text():
                break
            time.sleep(0.01)
        assert "visible before CLI exit" in log.read_text()
        return 0, None

    options = {"observer": observer}
    if name == "grok-build":
        task["grok"]["grok_home"] = str(tmp_path / "grok")
        monkeypatch.setenv(task["grok"]["api_key_env"], "test-placeholder")
        monkeypatch.setattr(runner, "prepare_home", lambda *a, **k: None)
        monkeypatch.setattr(runner, "gateway_probe", lambda *a: None)
        monkeypatch.setattr(runner, "find_session_dir", lambda *a: str(tmp_path))
        monkeypatch.setattr(runner, "run_grok", fake_cli)
    else:
        ns = {}
        exec(blob["modules"]["session_observer"], ns)
        options["observer_ns"] = ns
        monkeypatch.setattr(runner, "find_session", lambda *a: str(path) if path.exists() else None)
        if name == "gemini-cli":
            original_expanduser = runner.os.path.expanduser
            monkeypatch.setattr(runner.os.path, "expanduser",
                                lambda path: str(tmp_path) if path == "~" else original_expanduser(path))
            task["gemini"]["route"] = "native"
            task["gemini"]["base_url"] = "https://generativelanguage.googleapis.com"
            monkeypatch.setenv(task["gemini"]["api_key_env"], "test-placeholder")
            monkeypatch.setattr(runner, "write_settings", lambda *a, **k: None)
            monkeypatch.setattr(runner, "compose_command", lambda *a: "fake cli")
            monkeypatch.setattr(runner, "run_cli", fake_cli)
        else:
            cfg = task["muse"]
            for field in ["config_home", "data_home", "prompt_dir"]:
                cfg[field] = str(tmp_path / field)
            cfg["route"], cfg["extra_body"] = "native", {}
            monkeypatch.setenv(cfg["api_key_env"], "test-placeholder")
            monkeypatch.setattr(runner, "prepare_homes", lambda *a, **k: "fake prompt")
            monkeypatch.setattr(runner, "run_muse", fake_cli)
    result = runner.run(task, None, **options)
    assert result["messages"] == records == captured
    assert "observer" not in (result["error"] or "")
    assert len(observer.messages) == 2


def test_gemini_api_failure_with_saved_setup_is_not_success():
    error = {"status": 400, "error": "subset_indices.items: missing field"}
    for returncode in [0, 1]:
        assert "HTTP 400" in gemini.episode_error(returncode, {"n_turns": 0}, [error])
    assert gemini.episode_error(1, {"n_turns": 2}, []) == "gemini exited 1"
    assert gemini.episode_error(0, {"n_turns": 1}, [error, {"status": 200}]) is None
