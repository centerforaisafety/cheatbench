"""Archive a saved run with portable evidence, preserving failures and schema versions."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def archive(source: Path, destination: Path, judges: Path, cohort: str) -> dict:
    redactions = [value for key, value in os.environ.items()
                  if len(value) >= 16 and any(tag in key for tag in ("KEY", "TOKEN", "SECRET"))]
    def serialise(value):
        text = json.dumps(value)
        for secret in redactions:
            text = text.replace(secret, "[REDACTED_CREDENTIAL]")
        return text

    episodes = [json.loads(x) for x in (source / 'episodes.jsonl').read_text().splitlines()]
    verdicts = [json.loads(x) for x in judges.read_text().splitlines()]
    versions = {v.get('schema_version', 2) for v in verdicts if v.get('parse_ok')}
    if len(versions) > 1:
        raise ValueError('Cannot archive mixed judge definitions together')
    schema_version = next(iter(versions), 2)
    by_key = {(v['model'], v['episode']): v for v in verdicts if v.get('parse_ok')}
    destination.mkdir(parents=True, exist_ok=True)
    kept, judged, evidence = [], [], []
    prompts = {}
    for e in episodes:
        key = (e['model'], e['episode'])
        record = {k: e.get(k) for k in (
            'id', 'episode', 'replicate', 'model', 'agent', 'agent_version',
            'ok', 'failure', 'error', 'wall_time', 'n_tool_calls', 'n_turns',
            'routing', 'egress', 'tool_policy', 'usage', 'init_mcp_servers',
            'cost_usd', 'cost_source', 'install', 'grader_state', 'final_text')}
        record['cohort'] = cohort
        record['evidence'] = f"trajectories.jsonl.gz#{e['episode']}"
        kept.append(record)
        if e.get('transcript_path'):
            path = Path(e['transcript_path'])
            raw = path.read_bytes()
            tr = json.loads(raw)
            prompts[e['episode']] = tr.get('prompt', '')
            evidence.append({'episode': e['episode'], 'source_sha256': hashlib.sha256(raw).hexdigest(),
                             'transcript': tr})
        if key in by_key and e.get('failure') != 'install_failed':
            v = dict(by_key[key])
            for k in ('transcript_path', 'sha256'):
                v.pop(k, None)
            v.update(schema_version=schema_version, evidence=record['evidence'], cohort=cohort)
            judged.append(v)
    for name, rows in [('episodes.jsonl', kept), ('judge.jsonl', judged)]:
        (destination / name).write_text(''.join(serialise(r) + '\n' for r in rows))
    data = ''.join(serialise(r) + '\n' for r in evidence).encode()
    (destination / 'trajectories.jsonl.gz').write_bytes(gzip.compress(data, mtime=0))
    (destination / 'prompts.json').write_text(serialise(prompts) + '\n')
    if (source / 'run.json').exists():
        run = json.loads((source / 'run.json').read_text())
        for key in ('task_root', 'image', 'judge_prompt', 'judge_schema'):
            run.pop(key, None)
        run['archive_judge_schema_version'] = schema_version
        run['note'] = 'Run configuration is original; archived judgments may use a corrected judge definition.'
        (destination / 'run.json').write_text(serialise(run) + '\n')
    result = {'credential_values_redacted' : True, 'cohort': cohort, 'schema_version': schema_version, 'episodes': len(episodes),
              'model': episodes[0]['model'] if episodes else None,
              'installation_failures': sum(e.get('failure') == 'install_failed' for e in episodes),
              'other_interruptions': sum(bool(e.get('failure')) and e['failure'] != 'install_failed' for e in episodes),
              'judged_observed_episodes': len(judged),
              **{k: sum(v.get(k) is True for v in judged) for k in
                 ('honey_pots_seen', 'cheated', 'obtained_designs', 'copied_design')}}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--judges', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.source, args.destination, args.judges, args.cohort), indent=2))
