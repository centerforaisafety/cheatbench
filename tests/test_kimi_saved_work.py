"""The Kimi runner must retain a task's saved files before closing its checker."""
import io
import json
from pathlib import Path


def test_observer_finishes_before_checker_close(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'core/agents/kimi_code_runner.py'
    ns = {'__name__': 'observer_test'}
    exec(compile(path.read_text(), str(path), 'exec'), ns)
    events = []
    state = {}
    class Grader:
        grader_state = state
        server_address = ('127.0.0.1', 1234)
        def close(self):
            events.append('close')
    class Observer:
        def finish(self):
            events.append('finish')
            state['saved_work'] = [{'name': 'binder.fasta'}]
    output = io.StringIO()
    ns.update(_OUT=output, _log=lambda *a: None, install_agent=lambda *a, **k: None,
              task_install_env=lambda: {}, serve_grader=lambda *a: Grader())
    def run(*a, **k):
        events.append('run')
        return {'grader_state': state, 'error': None}
    ns['run'] = run
    # This hook has to be constructed before run, then finished afterward.
    import builtins
    monkeypatch.setattr(builtins, '_protein_test_observer', lambda *a: Observer(), raising=False)
    ns['main']({'id': 'test', 'model': 'test', 'row': {}, 'files': []}, {
        'stage': 'def write(*a): return {}',
        'grader': 'import builtins\nmake_observer = builtins._protein_test_observer',
    })
    assert events == ['run', 'finish', 'close']
    assert json.loads(output.getvalue())['grader_state']['saved_work'] == [{'name': 'binder.fasta'}]
