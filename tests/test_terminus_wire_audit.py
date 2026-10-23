import ast
import asyncio
import json
from pathlib import Path

import httpx


def test_wire_audit_preserves_request_and_keeps_secrets_out(monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'core/agents/terminus_2_images.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install_wire_audit']
    ns = {'json':json,'WIRE_REQUESTS':[]}
    exec(compile(tree, str(source), 'exec'), ns)
    monkeypatch.setattr(httpx.AsyncClient, 'send', httpx.AsyncClient.send)
    ns['install_wire_audit']()
    body = {'model':'gemini-4-argon','reasoning_effort':'high',
            'messages':[{'role':'user','content':[{'type':'text','text':'never-log-this'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,secret-image'}}]}]}
    def handler(request):
        assert json.loads(request.content) == body
        assert request.headers['Authorization'] == 'Bearer secret-token'
        return httpx.Response(200, json={'model':'gemini-4-argon','choices':[]})
    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await client.post('https://model.example/v1/chat/completions', json=body,
                headers={'Authorization':'Bearer secret-token'})
            assert result.json()['model'] == 'gemini-4-argon'
    asyncio.run(call())
    audit = ns['WIRE_REQUESTS']
    assert len(audit) == 1
    assert audit[0]['model'] == audit[0]['returned_model'] == 'gemini-4-argon'
    assert audit[0]['reasoning_effort'] == 'high'
    assert audit[0]['n_images'] == 1 and audit[0]['status'] == 200
    assert 'secret' not in json.dumps(audit) and 'never-log' not in json.dumps(audit)


def test_anthropic_audit_records_effort_and_image_without_changing_request(monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'core/agents/terminus_2_images.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install_wire_audit']
    ns = {'json': json, 'WIRE_REQUESTS': []}
    exec(compile(tree, str(source), 'exec'), ns)
    monkeypatch.setattr(httpx.AsyncClient, 'send', httpx.AsyncClient.send)
    ns['install_wire_audit']()
    body = {'model': 'claude-opus-5-5', 'output_config': {'effort': 'high'},
            'thinking': {'type': 'adaptive', 'display': 'summarized'},
            'messages': [{'role': 'user', 'content': [{'type': 'image',
                'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'secret-pixels'}}]}]}
    def handler(request):
        assert json.loads(request.content) == body
        return httpx.Response(200, json={'model': 'claude-opus-5-5', 'stop_reason': 'end_turn'})
    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await client.post('https://model.example/v1/messages', json=body,
                              headers={'x-api-key': 'secret-key'})
    asyncio.run(call())
    audit = ns['WIRE_REQUESTS'][0]
    assert audit['output_config'] == {'effort': 'high'}
    assert audit['thinking'] == body['thinking']
    assert audit['returned_model'] == 'claude-opus-5-5'
    assert audit['n_images'] == 1 and audit['stop_reason'] == 'end_turn'
    assert 'secret' not in json.dumps(audit)
