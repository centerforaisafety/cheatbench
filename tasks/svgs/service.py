"""Host-owned deterministic SVG leaderboard."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from tasks.svgs.leaderboard_server import Round, start as start_server
from core.agents.errors import ReviewerInfrastructureError


@contextmanager
def start(*, task, rows, out_dir, config, load_model):
    if config:
        raise ValueError('SVG leaderboard has no service.config options')
    rounds = [Round(row) for row in rows]
    # A separate file for each invocation preserves submission records when resuming.
    log_dir = Path(out_dir) / 'leaderboard'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{uuid4().hex}.json'
    originals = [dict(row['files']) for row in rows]
    server, endpoints = start_server(rounds)
    def check_health():
        if server.infrastructure_errors:
            raise ReviewerInfrastructureError(server.infrastructure_errors[-1])
    server.check_health = check_health
    try:
        with tempfile.TemporaryDirectory(prefix='service_', dir=task.environment) as tmp:
            for index, (rnd, endpoint) in enumerate(endpoints):
                path = Path(tmp) / f'{index}.json'
                path.write_text(json.dumps({'round': 'final',
                                            'rubric': next(iter(rnd.row['files_abs'])), **endpoint}))
                rnd.row['files']['leaderboard/leaderboard_config.json'] = str(path.relative_to(task.environment))
            print(f'SVG leaderboard ready; submission records -> {log_path}', flush=True)
            yield server
    finally:
        # Join outstanding request threads before writing their final records.
        try:
            server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                for row, original in zip(rows, originals):
                    row['files'] = original
                log_path.write_text(json.dumps({r.row['id']: r.calls for r in rounds}, indent=2) + '\n')
