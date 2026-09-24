"""Pinned host-only KataGo and network, prepared before any model starts."""
from __future__ import annotations
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

CACHE = Path(__file__).resolve().parent / 'environment/assets'
VERSION = 'v1.18.2'
ARCHIVE = 'katago-v1.18.2-cuda12.1-cudnn8.9.7-linux-x64.zip'
ENGINE_URL = f'https://github.com/lightvector/KataGo/releases/download/{VERSION}/{ARCHIVE}'
ENGINE_SHA256 = '16c69f42291fe8c6d196d722d92299a5b04de852611d6c022a0dd9e0e83b5688'
NETWORK = 'kata1-b18c384nbt-s9996604416-d4316597426.bin.gz'
NETWORK_URL = f'https://media.katagotraining.org/uploaded/networks/models/kata1/{NETWORK}'
NETWORK_SHA256 = '9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path, expected):
    if path.is_file() and sha256(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + '.download')
    try:
        print(f'Go build: downloading {url}', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; CheatBench asset downloader)'})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if sha256(temporary) != expected:
            raise RuntimeError(f'Checksum mismatch for {path.name}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def engine_paths():
    """Resolve already built assets; never download inside an episode."""
    manifest = CACHE / 'install.json'
    if not manifest.is_file():
        raise RuntimeError('Go assets not built. Run run.py go or tasks/go/build.py first.')
    record = json.loads(manifest.read_text())
    return record['binary'], record['network']


def check_runtime():
    """Report the host requirement before downloads or model calls."""
    manifest = CACHE / 'install.json'
    custom = bool(os.environ.get('KATAGO_BINARY'))
    if manifest.is_file() and not custom:
        custom = json.loads(manifest.read_text()).get('archive_sha256') is None
    if custom:
        print('Go runtime: custom KataGo backend; engine compatibility is checked at startup.', flush=True)
        return
    error = ('The default Go build requires an NVIDIA GPU for its CUDA KataGo backend. '
             'Run on a GPU machine or request a Slurm GPU with --gres=gpu:1. '
             'KataGo and its weights are built automatically; no manual download is needed.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        raise RuntimeError(error)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(error) from exc
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(error)
    print('Go runtime: NVIDIA GPU detected; host KataGo uses the GPU, model inference uses APIs.', flush=True)


def ensure_assets():
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / '.install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binary_override = os.environ.get('KATAGO_BINARY')
        network = Path(os.environ.get('KATAGO_NETWORK') or CACHE / NETWORK).resolve()
        if not os.environ.get('KATAGO_NETWORK'):
            download(NETWORK_URL, network, NETWORK_SHA256)
        if not network.is_file() or sha256(network) != NETWORK_SHA256:
            raise RuntimeError('Go requires the pinned certified network; KATAGO_NETWORK has a different hash.')
        if binary_override:
            binary = Path(binary_override).resolve()
        else:
            if platform.system() != 'Linux' or platform.machine() != 'x86_64':
                raise RuntimeError('Automatic KataGo setup requires Linux x86_64 with an NVIDIA GPU. '
                                   'Set KATAGO_BINARY to a compatible v1.18.2 build on other hosts.')
            archive = CACHE / ARCHIVE
            download(ENGINE_URL, archive, ENGINE_SHA256)
            binary = CACHE / 'engine/squashfs-root/AppRun'
            if not binary.is_file():
                with tempfile.TemporaryDirectory(dir=CACHE, prefix='install-') as staging:
                    stage = Path(staging)
                    with zipfile.ZipFile(archive) as bundle:
                        for member in bundle.infolist():
                            target = (stage / member.filename).resolve()
                            if not target.is_relative_to(stage.resolve()):
                                raise RuntimeError('Engine archive path escapes staging directory')
                        bundle.extractall(stage)
                    appimage = stage / 'katago'
                    appimage.chmod(0o755)
                    subprocess.run([str(appimage), '--appimage-extract'], cwd=stage, check=True,
                                   stdout=subprocess.DEVNULL, timeout=180)
                    target = CACHE / 'engine'
                    target.mkdir(exist_ok=True)
                    shutil.move(str(stage / 'squashfs-root'), str(target / 'squashfs-root'))
            binary = binary.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f'KataGo executable is missing or not executable: {binary}')
        # The two position probes verify backend compatibility, not merely the version string.
        from .katago_api import KataGoEngine
        from .engine_signature import verify
        async def probe():
            engine = KataGoEngine(str(binary), str(network), max_visits=400)
            try:
                await engine.start()
                return await verify(engine)
            finally:
                await engine.close()
        signature = asyncio.run(probe())
        result = {'version': VERSION, 'binary': str(binary), 'network': str(network),
                  'binary_sha256': sha256(binary), 'network_sha256': NETWORK_SHA256,
                  'archive_sha256': None if binary_override else ENGINE_SHA256, 'signature': signature}
        temporary = CACHE / 'install.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(CACHE / 'install.json')
        print(f'Go build: engine signature {signature["status"]}', flush=True)
        return result
