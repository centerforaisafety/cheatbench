import json
from core.agents.kimi_code_runner import KimiJournalObserver
from tasks.prime_factorization.grader import MessageObserver


def test_partial_journal_and_real_task_spill(tmp_path):
    home = tmp_path / 'home'
    folder = home / 'sessions/ws/session/agents/main'
    folder.mkdir(parents=True)
    wire = folder / 'wire.jsonl'
    row = {'id': 'test', 'live_log_dir': str(tmp_path/'logs'), 'cache_dir': str(tmp_path/'cache'), 'live_result_path': str(tmp_path/'result.json')}
    observer = MessageObserver(row, {}, str(tmp_path), {'attempts': [], 'is_correct': False})
    tail = KimiJournalObserver(str(home), observer)
    call = {'type': 'context.append_loop_event', 'event': {'type': 'tool.call', 'toolCallId': 'Read_1', 'name': 'Read', 'args': {'path': '/some/log'}}}
    result = {'type': 'context.append_loop_event', 'event': {'type': 'tool.result', 'toolCallId': 'Read_1', 'result': {'output': 'answer content ' * 2000}}}
    wire.write_text(json.dumps(call)+'\n'+json.dumps(result)[:40])
    tail.drain()
    assert len(observer.messages) == 1
    with wire.open('a') as f: f.write(json.dumps(result)[40:]+'\n')
    tail.drain()
    tail.drain()
    assert len(observer.messages) == 2
    assert (tmp_path/'cache/live_Read_1.txt').read_text() == result['event']['result']['output']
    assert 'Read' in (tmp_path/'logs/turns.log').read_text()
    observer.finish()
    assert observer.state['mirrored_messages'] == 2
