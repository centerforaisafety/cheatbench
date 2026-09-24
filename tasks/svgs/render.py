"""Render untrusted SVGs in bounded child processes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

_RENDER_SLOTS = threading.BoundedSemaphore(4)
_WALL_TIMEOUT_S = 10
_WORKER = Path(__file__).with_name('render_worker.py')


def render_svg(svg: str) -> bytes:
    if not _RENDER_SLOTS.acquire(blocking=False):
        raise RuntimeError('Renderer busy; try again')
    try:
        with tempfile.TemporaryDirectory(prefix='svg-render-') as directory:
            try:
                result = subprocess.run(
                    [sys.executable, '-I', str(_WORKER)], input=svg.encode(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=directory, env={'PATH': os.defpath, 'LANG': 'C.UTF-8'},
                    timeout=_WALL_TIMEOUT_S, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError('SVG rendering exceeded its time limit') from exc
        if result.returncode:
            raise ValueError('SVG could not be rendered within resource limits')
        if not result.stdout.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('SVG renderer returned an invalid image')
        return result.stdout
    finally:
        _RENDER_SLOTS.release()
