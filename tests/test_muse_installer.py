"""Exercise the install fallback with vendor responses, without network access."""
import os
import shlex
import subprocess

import pytest

from core.agents import muse_code


def test_exact_pin_avoids_stable_channel_and_auto_update():
    agent = muse_code.MuseCodeAgent(model="meta/muse-spark-1.3", version="1.3.0-R3401.1")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    assert agent.setup() == []
    spec = agent.install()
    assert '1.3.0-R3401.1' in spec['check']
    assert 'MUSE_NO_AUTO_UPDATE=1' in spec['check']
    assert 'muse-launcher.sh' not in spec['install']
    assert 'checksum/size mismatch' in spec['install']


@pytest.mark.parametrize('corrupt', [False, True])
def test_versioned_artifact_is_verified_before_replacing_binary(tmp_path, monkeypatch, corrupt):
    import hashlib
    import io
    import json
    from core.agents import muse_install
    binary = b'fake binary'
    manifest = {'version': '1.3.0-R3401.1', 'checksum_algorithm': 'sha256',
                'artifacts': {'x86_linux': {'url': 'https://lookaside.facebook.com/binary',
                    'checksum': hashlib.sha256(binary).hexdigest(), 'size': len(binary)}}}
    requests = []
    def fetch(url, **kwargs):
        requests.append(url)
        return io.BytesIO(json.dumps(manifest).encode() if 'manifest.json' in url
                          else (b'wrong binary' if corrupt else binary))
    monkeypatch.setattr(muse_install.urllib.request, 'urlopen', fetch)
    monkeypatch.setattr(muse_install.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(muse_install.platform, 'system', lambda: 'Linux')
    target = tmp_path / 'muse'
    target.write_bytes(b'previous release')
    if corrupt:
        with pytest.raises(ValueError, match='checksum/size'):
            muse_install.install('1.3.0-R3401.1', target)
        assert target.read_bytes() == b'previous release'
    else:
        receipt = muse_install.install('1.3.0-R3401.1', target)
        assert target.read_bytes() == binary
        assert receipt['sha256'] == hashlib.sha256(binary).hexdigest()
    assert 'version=1.3.0-R3401.1' in requests[0]
    assert not list(tmp_path.glob('.muse-*'))


@pytest.mark.parametrize('valid_launcher', [True, False])
def test_html_installer_uses_validated_official_launcher(tmp_path, monkeypatch, valid_launcher):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    curl = bin_dir / 'curl'
    curl.write_text('''#!/bin/sh
url=""; dest=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    https:*) url="$1" ;;
    -o) shift; dest="$1" ;;
  esac
  shift
done
case "$url" in
  https://dev.meta.ai/install.sh) printf '%s' '<h1>Not Logged In</h1>' > "$dest" ;;
  https://api.meta.ai/muse-launcher.sh) printf '%s' "$TEST_LAUNCHER" > "$dest" ;;
  *) exit 9 ;;
esac
''')
    curl.chmod(0o755)
    monkeypatch.setattr(muse_code, 'link_bins', lambda *args: '')
    agent = muse_code.MuseCodeAgent(model='meta/test')
    agent.PATH_EXPORT = 'export PATH=' + shlex.quote(str(bin_dir)) + ':"$PATH"'
    script = agent.install_script().replace('"$HOME/.local/bin', '"' + str(bin_dir))
    env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'],
               TEST_LAUNCHER='#!/bin/sh\necho "Muse Code test-release"\n'
               if valid_launcher else '<h1>Not Logged In</h1>')
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
    if valid_launcher:
        assert result.returncode == 0, result.stderr
        assert 'Muse Code test-release' in result.stdout
        assert (bin_dir / 'muse').is_file()
    else:
        assert result.returncode != 0
        assert not (bin_dir / 'muse').exists()
    assert 'using official Meta launcher' in result.stderr
