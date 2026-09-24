"""Serve the one background reading inside the episode's private network."""
from pathlib import Path

def setup(sandbox, row, env_dir):
    root = Path(env_dir)
    sandbox.start_mirror(site=root / 'site', server=root / 'server.py',
                         hosts=('openai.com',),
                         fallback_ips={'openai.com': ['192.0.2.41']})
