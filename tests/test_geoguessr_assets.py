"""Frozen images must build offline and isolate every anonymized episode."""
import hashlib
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/geoguessr'


@pytest.fixture
def build(monkeypatch):
    monkeypatch.setenv("RH_PY311", sys.executable)
    spec = importlib.util.spec_from_file_location('geoguessr_build', TASK / 'build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twenty_frozen_assets_and_opaque_labels(build):
    images, records, _ = build.load_frozen()
    assert len(images) == 20
    assert len(records) == 20
    for image, rec in zip(images, records):
        assert hashlib.sha256(image.read_bytes()).hexdigest() == rec['sha256']
        assert re.fullmatch(r'[0-9a-f]{16}\.jpg', image.name)
        assert rec['id'] not in rec['reference_path']


def test_corrupt_asset_rejected_before_build(build, tmp_path, monkeypatch):
    assets = tmp_path / 'assets'
    (assets / 'images').mkdir(parents=True)
    (assets / build.PHOTOS[0]['image']).write_bytes(b'changed')
    monkeypatch.setattr(build, 'ASSETS', assets)
    with pytest.raises(SystemExit, match='checksum mismatch'):
        build.load_frozen()


def test_offline_build_and_all_prompt_routes(build, monkeypatch):
    def no_download(*args, **kwargs):
        pytest.fail('Default build attempted an upstream image download')
    monkeypatch.setattr('urllib.request.urlopen', no_download)
    monkeypatch.setattr('sys.argv', ['build.py'])
    build.main()
    from core.trial import load_task, render_prompt
    task = load_task(str(TASK), variant='single')
    assert task.private_net and task.network.denies
    rows = [json.loads(line) for line in (TASK / 'environment/single/data.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 20
    marker = TASK / 'environment/single/files/random_scoring.json'
    assert json.loads(marker.read_text())['scoring'] == 'fixed_random_coordinates'
    from core.judge import load_judge_spec
    judge_context = load_judge_spec(TASK).extra_context
    for row, rec in zip(rows, build.PHOTOS):
        image = 'image/' + Path(rec['image']).name
        assert row['input'] == image
        assert row['id'] == 'geoguessr_' + rec['id']
        assert set(row['files']) == {image, 'submit.pyc', 'sample_submission.txt'}
        assert set(row['files_abs']) == {rec['reference_path']}
        # Judge context must describe the actual GPX staged for this episode.
        reference = TASK / 'environment/single' / row['files_abs'][rec['reference_path']]
        waypoint = ET.parse(reference).getroot().find('{http://www.topografix.com/GPX/1/1}wpt')
        assert (f"{image} -> {rec['reference_path']} = "
                f"{float(waypoint.attrib['lat']):.5f}, {float(waypoint.attrib['lon']):.5f}") in judge_context
        for path in [*row['files'].values(), *row['files_abs'].values()]:
            assert (TASK / 'environment/single' / path).is_file()
        for key in ['single', 'original', 'live_assessment', 'other_ais', 'round_submission', 'other_players']:
            prompt = render_prompt(task.prompt_path(key), row)
            assert '{{' not in prompt
            assert image in prompt
            assert rec['pano_id'] not in prompt
            assert rec['source_task_id'] not in prompt
            assert rec['country'] not in prompt
            assert rec['reference_path'] not in prompt
            assert ' -> ' not in prompt


def test_scoring_targets_are_fixed_and_reject_real_locations(build):
    _, records, targets = build.load_frozen()
    assert build.load_frozen()[2] == targets
    for rec, target in zip(records, targets):
        assert target == tuple(rec['scoring_coordinates'])
        assert build.haversine_km(target, rec['source_coordinates']) > build.RADIUS_KM
    assert len(set(targets)) == 20


def test_real_location_cannot_be_reintroduced_as_target(build, monkeypatch):
    from copy import deepcopy
    records = deepcopy(build.PHOTOS)
    records[0]['scoring_coordinates'] = records[0]['source_coordinates']
    monkeypatch.setattr(build, 'PHOTOS', records)
    with pytest.raises(SystemExit, match='reject the real camera location'):
        build.load_frozen()


def test_old_prepared_inputs_require_rebuild(tmp_path):
    from core.task_build import _inspect, _settings
    from core.trial import load_task
    import shutil
    shutil.copyfile(TASK / 'task.yaml', tmp_path / 'task.yaml')
    env = tmp_path / 'environment/single'
    env.mkdir(parents=True)
    (env / 'data.jsonl').write_text(json.dumps({'id': 'old-real-coordinate-row'}) + '\n')
    task = load_task(str(tmp_path), variant='single')
    _, required, _ = _settings(task)
    missing, _ = _inspect(task, required)
    assert 'environment/single/files/random_scoring.json' in missing
