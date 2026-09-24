"""Offline preparation for tests that exercise generated writing-task files."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope='session')
def built_creative_writings():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'RH_PY311': sys.executable,
           'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'PYTHONDONTWRITEBYTECODE': '1'}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('RH_PY311', sys.executable)
        subprocess.run([sys.executable, str(root / 'tasks/creative_writings/build.py')],
                       cwd=root, env=env, check=True, capture_output=True, text=True)
        yield
