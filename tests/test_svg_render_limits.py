"""Real render-worker limits and recovery, without model calls."""
import subprocess

import pytest

from tasks.svgs import render

NORMAL = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="20" height="20"/></svg>'


def test_expanding_svg_is_terminated_and_next_render_succeeds():
    defs = ['<g id="g0"><rect width="1" height="1"/></g>']
    for i in range(1, 25):
        defs.append(f'<g id="g{i}"><use href="#g{i-1}"/><use href="#g{i-1}"/></g>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><defs>' + ''.join(defs)
           + '</defs><use href="#g24"/></svg>')
    with pytest.raises(ValueError, match='resource limits|time limit'):
        render.render_svg(svg)
    assert render.render_svg(NORMAL).startswith(b'\x89PNG\r\n\x1a\n')


def test_worker_rejects_external_files(tmp_path):
    reference = tmp_path / 'secret.svg'
    reference.write_text(NORMAL)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg">'
           f'<image href="{reference.as_uri()}" width="20" height="20"/></svg>')
    with pytest.raises(ValueError, match='resource limits'):
        render.render_svg(svg)


def test_wall_timeout_and_credential_isolation(monkeypatch):
    monkeypatch.setenv('REVIEWER_API_KEY', 'must-not-reach-renderer')
    def timeout(*args, **kwargs):
        assert 'REVIEWER_API_KEY' not in kwargs['env']
        assert kwargs['timeout'] == 10
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    with monkeypatch.context() as patch:
        patch.setattr(render.subprocess, 'run', timeout)
        with pytest.raises(ValueError, match='time limit'):
            render.render_svg(NORMAL)
    assert render.render_svg(NORMAL).startswith(b'\x89PNG')


def test_renderer_capacity_is_bounded():
    acquired = []
    try:
        for _ in range(4):
            assert render._RENDER_SLOTS.acquire(blocking=False)
            acquired.append(True)
        with pytest.raises(RuntimeError, match='busy'):
            render.render_svg(NORMAL)
    finally:
        for _ in acquired:
            render._RENDER_SLOTS.release()


@pytest.mark.parametrize('svg', [
    NORMAL,
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<defs><linearGradient id="g"><stop stop-color="red"/>'
    '<stop offset="1" stop-color="blue"/></linearGradient>'
    '<g id="shape"><circle cx="50" cy="50" r="40" fill="url(#g)"/></g></defs>'
    '<use href="#shape"/></svg>',
])
def test_normal_artwork_matches_previous_renderer(svg):
    import cairosvg
    expected = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=800, output_height=600, unsafe=False,
    )
    assert render.render_svg(svg) == expected
