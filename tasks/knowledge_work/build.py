"""Reuse the selected GDPval assignments and cache their original files locally."""
import argparse
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
# Destination inside the episode container, never a host cache dependency.
REVIEW_ROOT = '/data/cache/artifacts'


SELECTION = json.loads((HERE / 'samples20.json').read_text())
TASK_IDS = tuple(r['task_id'] for r in SELECTION['tasks'])


def download_source(output):
    """Download the selected assignments; dataset contents stay untracked."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    dataset = {row['task_id']: row for row in load_dataset('openai/gdpval', revision=SELECTION['revision'])['train']}
    rows = []
    for task_id in TASK_IDS:
        item = dataset[task_id]
        if len(item['deliverable_files']) != 1:
            raise ValueError(f'{task_id}: expected one deliverable')
        files = {}
        for field, category in [('reference_files', 'reference_files'),
                                ('deliverable_files', 'gold_deliverable_files')]:
            for remote in item[field]:
                cached = Path(hf_hub_download('openai/gdpval', remote, repo_type='dataset', revision=SELECTION['revision']))
                name = Path(remote).name
                key = category + '/' + name
                if key in files:
                    raise ValueError(f'{task_id}: duplicate input basename {name}')
                dest = output / 'environment/files/downloads' / task_id / category / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, dest)
                files[key] = str(dest.relative_to(output / 'environment'))
        rows.append({'id': task_id[:8], 'input': item['prompt'], 'files': files,
                     'meta': {'task_id': task_id, 'sector': item['sector'],
                              'occupation': item['occupation'],
                              'gold_name': Path(item['deliverable_files'][0]).name}})
    return rows


def build(source=None, output=HERE):
    output = Path(output)
    if source is None:
        source = output
        originals = download_source(output)
    else:
        source = Path(source)
        originals = [json.loads(line) for line in (source / 'data.jsonl').read_text().splitlines()
                     if line.strip()]
    rows = []
    for old in originals:
        dest = output / 'environment/files' / old['id']
        dest.mkdir(parents=True, exist_ok=True)
        files, absolute = {}, {}
        for name, relative in old['files'].items():
            category, basename = name.split('/', 1)
            target = dest / category / basename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'environment' / relative, target)
            rel = str(target.relative_to(output / 'environment'))
            if category == 'reference_files':
                files['inputs/' + basename] = rel
            elif category == 'gold_deliverable_files':
                absolute[f'{REVIEW_ROOT}/{basename}'] = rel
            else:
                raise ValueError(f'Unexpected GDPval directory: {category}')
        if len(absolute) != 1:
            raise ValueError('Each row needs one competing deliverable')
        assignment = old['input'].replace('reference_files/', 'inputs/')
        (dest / 'brief.md').write_text(assignment + '\n')
        (dest / 'README.md').write_text((HERE / 'templates/README.md').read_text())
        for name in ('brief.md', 'README.md'):
            files[name] = str((dest / name).relative_to(output / 'environment'))
        rows.append({'id': old['id'], 'files': files, 'files_abs': absolute,
                     'reference_destination': next(iter(absolute)),
                     'meta': old.get('meta', {})})
    (output / 'data.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Optional cache from the previous GDPval build')
    args = parser.parse_args()
    print(f'Built {len(build(args.source))} client evaluation assignments')
