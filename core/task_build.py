"""Prepare generated task inputs before scheduling episodes (never from a judge)."""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .config import load_yaml


class TaskBuildError(RuntimeError):
    pass


def _settings(task):
    raw = load_yaml(task.root / 'task.yaml').get('build', {})
    if not isinstance(raw, dict) or set(raw) - {'args', 'required', 'preflight'}:
        raise TaskBuildError('task.yaml build must be a map containing only args, required, and preflight')
    for key in ('args', 'required', 'preflight'):
        value = raw.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise TaskBuildError(f'task.yaml build.{key} must be a list of strings')
    args = [s.replace('{variant}', task.variant) for s in raw.get('args', [])]
    required = []
    for value in raw.get('required', []):
        rel = Path(value.replace('{variant}', task.variant))
        if rel.is_absolute() or '..' in rel.parts:
            raise TaskBuildError(f'build.required must stay inside the task: {value}')
        required.append(task.root / rel)
    preflight = [s.replace('{variant}', task.variant) for s in raw.get('preflight', [])]
    return args, required, preflight


def _inspect(task, required):
    """Missing generated inputs can be built; malformed inputs must fail loudly."""
    missing = [str(p.relative_to(task.root)) for p in required if not p.exists()]
    if not task.data_path.is_file():
        return [str(task.data_path.relative_to(task.root)), *missing], []
    try:
        rows = task.rows()
    except (ValueError, TypeError) as exc:
        raise TaskBuildError(f'{task.data_path}: invalid task data: {exc}') from exc
    if not rows:
        return [str(task.data_path.relative_to(task.root)) + ' (empty)', *missing], []
    env = task.environment.resolve()
    for row in rows:
        if not isinstance(row, dict) or not row.get('id'):
            raise TaskBuildError(f'{task.data_path}: each row needs an id')
        for key in ('files', 'files_abs'):
            mapping = row.get(key) or {}
            if not isinstance(mapping, dict):
                raise TaskBuildError(f'{row["id"]}: {key} must be a mapping')
            for value in mapping.values():
                if not isinstance(value, str):
                    raise TaskBuildError(f'{row["id"]}: invalid source path {value!r}')
                path = (env / value).resolve()
                if not path.is_relative_to(env):
                    raise TaskBuildError(f'{row["id"]}: source escapes environment: {value}')
                if not path.is_file():
                    missing.append(str(path.relative_to(task.root)))
        if task.review:
            name = (row.get('meta') or {}).get('gold_name')
            if not name:
                raise TaskBuildError(f'{row["id"]}: review task requires meta.gold_name')
            path = env / 'files/golds' / name
            if not path.resolve().is_relative_to(env):
                raise TaskBuildError(f'{row["id"]}: gold path escapes environment')
            if not path.is_file():
                missing.append(str(path.relative_to(task.root)))
    return sorted(set(missing)), rows


def ensure_task_built(task, *, auto_build=True, rebuild=False):
    """One build per task at a time; recheck under the lock after another launch."""
    args, required, preflight = _settings(task)
    builder = task.root / 'build.py'
    command = [sys.executable, '-u', str(builder), *args]
    label = task.name + (f'/{task.variant}' if task.variant else '')
    if preflight:
        check = [sys.executable, '-u', str(builder), *preflight]
        try:
            result = subprocess.run(check, cwd=task.root, check=False, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TaskBuildError(f'{label}: runtime preflight failed: {exc}. No episodes started.') from exc
        if result.returncode:
            raise TaskBuildError(f'{label}: runtime preflight failed; see output above. No episodes started.')
    lock = task.root / '.build.lock'
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'task build: {label}: waiting for another build', flush=True)
            fcntl.flock(stream, fcntl.LOCK_EX)
        missing, rows = ([], []) if rebuild else _inspect(task, required)
        built = False
        if missing or rebuild:
            reason = '--rebuild-task' if rebuild else 'missing ' + ', '.join(missing[:5])
            if not auto_build and not rebuild:
                raise TaskBuildError(f'{label}: {reason}; automatic builds disabled. Run {shlex.join(command)}')
            if not builder.is_file():
                raise TaskBuildError(f'{label}: {reason}; no build.py is available. Prepare this task using its README.')
            print(f'task build: {label}: {reason}', flush=True)
            print(f'task build: running {shlex.join(command)}', flush=True)
            env = os.environ.copy()
            if sys.version_info[:2] == (3, 11):
                env.setdefault('RH_PY311', sys.executable)
            try:
                result = subprocess.run(command, cwd=task.root, env=env, check=False)
            except OSError as exc:
                raise TaskBuildError(f'{label}: could not start builder: {exc}') from exc
            if result.returncode:
                raise TaskBuildError(f'{label}: build failed (exit {result.returncode}); see builder output above. No episodes started.')
            missing, rows = _inspect(task, required)
            if missing:
                raise TaskBuildError(f'{label}: builder finished but artifacts are still missing: {", ".join(missing[:5])}')
            built = True
        print(f'task build: {label}: {"built" if built else "reusing"} validated artifacts ({len(rows)} rows)', flush=True)
        record = {'status': 'built' if built else 'reused', 'command': command if built else None,
                  'data_sha256': hashlib.sha256(task.data_path.read_bytes()).hexdigest(),
                  'variant': task.variant}
        return rows, record
