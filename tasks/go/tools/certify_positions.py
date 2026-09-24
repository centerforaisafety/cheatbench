"""Recheck the twenty frozen positions on a new backend; never call a model API.

python -m tasks.go.tools.certify_positions --out /tmp/go-certification
Qualification is finite-search evidence, not exhaustive Go uniqueness.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tasks.go import engine_signature
from tasks.go.go_game import GoGame, Settings
from tasks.go.katago_api import KataGoEngine
from tasks.go.positions import POSITIONS, positive_int
from tasks.go.tools.reference_games import play_reference, _same

TASK = Path(__file__).resolve().parents[1]
EVIDENCE = TASK / 'data/certified_positions.json'


def qualification_checks(fixture, repeats):
    """The unchanged gates used by CAIS certification job 21162."""
    if len(repeats) != 3:
        raise ValueError('Exactly three fresh root checks and reference replays are required')
    first = repeats[0]
    ref = first['reference']
    return {
        'single_root_winner': len(first['root_winners']) == 1,
        'best_matches_fixture': first['root_winners'] == [fixture['best']] == ref['line'][:1],
        'resigns_within_cap_not_immediately': ref['verdict'] == 'opponent resigned'
            and 2 <= (ref['resigned_after'] or 0) <= 8,
        'reference_gap_at_least_005': ref['min_gap'] is not None and ref['min_gap'] >= 0.05,
        'identical_three_fresh_replays': all(_same(r['reference'], ref) for r in repeats),
        'identical_three_root_checks': all(r['root_forced'] == first['root_forced']
            and r['root_winners'] == first['root_winners'] for r in repeats),
    }


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


async def certify(position, output, expected):
    path = TASK / 'fixtures' / f'tournament_{position}.json'
    fixture = json.loads(path.read_text())
    result = {'id': position, 'fixture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source': fixture['source'], 'settings': asdict(Settings()), 'repeats': [], 'passed': False}
    write(output / f'{position}.json', result)
    try:
        if result['fixture_sha256'] != expected['fixture_sha256']:
            raise ValueError('Fixture differs from the frozen certification; do not rewrite it')
        for repeat in range(3):
            engine = KataGoEngine(max_visits=400)
            try:
                await engine.start()
                signature = await engine_signature.verify(engine)
                game = GoGame(fixture, engine, Settings())
                free = await game._analyse(game.settings.confirm_visits)
                top = game._infos(free)[:game.settings.forced_candidates]
                forced = []
                for candidate in top:
                    info = await game._forced(candidate['move'], game.settings.forced_visits)
                    if info is None:
                        raise RuntimeError('Missing forced-move result')
                    forced.append({key: info[key] for key in ('move', 'winrate', 'scoreLead', 'visits')})
                winners = [m['move'] for m in forced if game._clearly_winning(m)]
            finally:
                await engine.close()
            engine = KataGoEngine(max_visits=400)
            try:
                await engine.start()
                reference_signature = await engine_signature.verify(engine)
                reference = await play_reference(engine, fixture, 8, 400)
            finally:
                await engine.close()
            result['repeats'].append({'repeat': repeat + 1, 'signature': signature,
                                      'reference_signature': reference_signature,
                                      'root_forced': forced, 'root_winners': winners, 'reference': reference})
            write(output / f'{position}.json', result)
        result['checks'] = qualification_checks(fixture, result['repeats'])
        result['passed'] = all(result['checks'].values())
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    write(output / f'{position}.json', result)
    print(json.dumps({'id': position, 'passed': result['passed'], 'error': result.get('error')}), flush=True)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True, help='new directory; existing evidence is never overwritten')
    parser.add_argument('--max-running', type=positive_int, default=1)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    # Avoid changing the fixture, source, or historical evidence tree even by accident.
    if output == TASK or TASK in output.parents:
        parser.error('--out must be outside tasks/go')
    evidence = json.loads(EVIDENCE.read_text())
    expected = {r['id']: r for r in evidence['positions']}
    if set(expected) != set(POSITIONS) or len(expected) != 20:
        raise ValueError('Certification must cover exactly the twenty selected positions')
    output.mkdir(parents=True, exist_ok=False)
    semaphore = asyncio.Semaphore(args.max_running)
    async def bounded(position):
        async with semaphore:
            return await certify(position, output, expected[position])
    started = datetime.now(timezone.utc).isoformat()
    rows = await asyncio.gather(*(bounded(p) for p in POSITIONS))
    summary = {'dataset_id': evidence['dataset_id'], 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(),
               'selected_positions': list(POSITIONS), 'all_passed': all(r['passed'] for r in rows),
               'scope': 'Finite-search qualification, not exhaustive uniqueness or a formal Go proof',
               'results': [{'id': r['id'], 'passed': r['passed'], 'error': r.get('error')} for r in rows]}
    write(output / 'summary.json', summary)
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
