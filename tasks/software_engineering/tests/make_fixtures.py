"""Generate synthetic protocol examples; no model calls or evaluation records.

Run from the repository root:
    python tasks/software_engineering/tests/make_fixtures.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent / 'fixtures'
COMMANDS = [
    'git status',
    'git log --oneline -3',
    'rg get_action_args_with_defaults src/',
    'ls -la /app',
]


def write(name, filename, value, *, lines=False):
    path = ROOT / name / 'agent' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        text = ''.join(json.dumps(row) + '\n' for row in value)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2) + '\n'
    path.write_text(text)


def main():
    claude = []
    for i, command in enumerate(COMMANDS):
        cid = f'fixture-{i}'
        claude += [
            {'type': 'assistant', 'message': {'content': [
                {'type': 'tool_use', 'id': cid, 'name': 'Bash', 'input': {'command': command}}]}},
            {'type': 'user', 'message': {'content': [
                {'type': 'tool_result', 'tool_use_id': cid, 'content': f'Synthetic output {i}'}]}},
        ]
    write('claude-code', 'claude-code.txt', claude, lines=True)
    write('claude-session', 'sessions/projects/-app/fixture-session.jsonl', claude, lines=True)
    write('codex', 'codex.txt', [
        {'type': 'item.completed', 'item': {'type': 'command_execution', 'command': c,
            'aggregated_output': f'Synthetic output {i}', 'exit_code': 0}}
        for i, c in enumerate(COMMANDS)
    ], lines=True)
    for name in ('acp-gemini', 'acp-deepseek', 'acp-permission-only'):
        events = []
        commands = list(COMMANDS)
        if name == 'acp-permission-only':
            commands[0] = 'wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protoc-3.20.3-linux-x86_64.zip'
        for i, command in enumerate(commands):
            call = {'toolCallId': f'bash__call_fixture{i}', 'kind': 'execute',
                    'title': command if name == 'acp-gemini' else 'bash'}
            if name != 'acp-gemini':
                call['rawInput'] = {'command': command}
            if name == 'acp-permission-only':
                events.append({'event_type': 'request_permission', 'payload': {'tool_call': call}})
            else:
                events.append({'event_type': 'session_update', 'payload': {'update': {
                    **call, 'sessionUpdate': 'tool_call', 'status': 'in_progress'}}})
            events.append({'event_type': 'session_update', 'payload': {'update': {
                **call, 'sessionUpdate': 'tool_call_update', 'status': 'completed',
                'content': [{'type': 'text', 'text': f'Synthetic output {i}'}]}}})
        write(name, 'acp-events.jsonl', events, lines=True)
    kimi = []
    for i, command in enumerate(['git -C /app log --oneline -3', *COMMANDS]):
        kimi += [
            {'role': 'assistant', 'tool_calls': [{'id': f'fixture-{i}', 'type': 'function',
                'function': {'name': 'Shell', 'arguments': json.dumps({'command': command})}}]},
            {'role': 'tool', 'tool_call_id': f'fixture-{i}', 'content': f'Synthetic output {i}'},
        ]
    write('kimi-code', 'kimi-code.txt', kimi, lines=True)
    write('muse-code', 'muse-code.jsonl', [
        {'payload_type': 'tool.result', 'payload': {'correlation_facts': {'tool_name': 'bash'},
            'text': json.dumps({'command': c, 'output': f'Synthetic output {i}'})}}
        for i, c in enumerate(COMMANDS)
    ], lines=True)
    grok = []
    for i, command in enumerate(COMMANDS):
        grok += [
            {'type': 'tool_call', 'toolCallId': f'fixture-{i}', 'toolName': 'bash',
             'rawInput': {'command': command}},
            {'type': 'tool_call_update', 'toolCallId': f'fixture-{i}', 'status': 'completed',
             'rawOutput': {'stdout': list(f'Synthetic output {i}'.encode())}},
        ]
    write('grok-build', 'grok-build.txt', grok, lines=True)
    for i in range(3):
        write('terminus-2', f'episode-{i}/prompt.txt', f'Synthetic terminal screen {i}\n')
        write('terminus-2', f'episode-{i}/response.txt', {
            'analysis': 'Inspect the toy workspace.', 'plan': 'Read local files.',
            'commands': [{'keystrokes': 'ls -la internal/\n'}, {'keystrokes': 'git status\n'}],
            'task_complete': i == 2,
        })
    steps = [{'source': 'agent', 'message': 'Inspect the toy workspace.',
              'tool_calls': [{'tool_call_id': f'fixture-{i}', 'function_name': 'list_files',
                              'arguments': {'target_directory': f'/app/example{i}'}}],
              'observation': {'results': [{'source_call_id': f'fixture-{i}',
                                          'content': f'Synthetic output {i}'}]}}
             for i in range(4)]
    write('atif', 'trajectory.json', {'steps': steps})
    degraded = {'steps': [{'source': 'agent', 'tool_calls': [
        {'tool_call_id': 'fixture-empty', 'function_name': 'bash', 'arguments': {}}]}]}
    write('atif-degraded', 'trajectory.json', degraded)
    write('acp-deepseek', 'trajectory.json', degraded)
    write('gemini-cli', 'gemini-cli.txt', 'Synthetic narration without any tool-call records.\n')
    write('unknown-harness', 'mystery-harness.log', 'Synthetic unsupported format.\n')
    (ROOT / 'unknown-harness/result.json').write_text('{"task_name": "fixture"}\n')


if __name__ == '__main__':
    main()
