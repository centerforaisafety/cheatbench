import gzip
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('promotion_archive',
    Path(__file__).resolve().parents[1] / 'tasks/protein_design/archive_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archive_redacts_credentials_and_excludes_failed_installs(tmp_path, monkeypatch):
    secret = 'test-only-credential-value-12345'
    monkeypatch.setenv('TEST_API_KEY', secret)
    source = tmp_path / 'run'
    source.mkdir()
    transcript = source / 'transcript.json'
    transcript.write_text(json.dumps({'prompt': 'design', 'messages': [secret]}))
    episodes = [{'id': 'x', 'episode': 'x#1', 'model': 'm', 'failure': None,
                 'ok': True, 'transcript_path': str(transcript)},
                {'id': 'x', 'episode': 'x#2', 'model': 'm', 'failure': 'install_failed'}]
    (source / 'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    verdicts = source / 'judge.jsonl'
    verdicts.write_text(''.join(json.dumps({'model': 'm', 'episode': e['episode'],
        'parse_ok': True, 'cheated': True, 'schema_version': 3})+'\n' for e in episodes))
    dest = tmp_path / 'archive'
    result = module.archive(source, dest, verdicts, 'test')
    assert result['episodes'] == 2 and result['judged_observed_episodes'] == 1
    assert result['installation_failures'] == 1
    assert result['schema_version'] == 3
    assert json.loads((dest / 'judge.jsonl').read_text())['schema_version'] == 3
    text = gzip.decompress((dest / 'trajectories.jsonl.gz').read_bytes()).decode()
    assert secret not in text and '[REDACTED_CREDENTIAL]' in text
    assert 'source_sha256' in json.loads(text)
