"""Stockfish setup: pinned downloads, offline reuse, and no partial installs."""
import hashlib
import io
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from tasks.chess import stockfish


@pytest.fixture
def release(tmp_path, monkeypatch):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, body in [("test-engine", b"engine"), ("Copying.txt", b"license")]:
            member = tarfile.TarInfo(f"stockfish/{name}")
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    payload = data.getvalue()
    monkeypatch.setattr(stockfish, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(stockfish.platform, "system", lambda: "Linux")
    monkeypatch.setattr(stockfish.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stockfish, "ASSETS", {"x86_64": ("test-engine", hashlib.sha256(payload).hexdigest())})
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(stockfish.urllib.request, "urlopen", download)
    monkeypatch.setattr(stockfish, "verify", lambda path: None)
    return calls


def test_install_and_offline_reuse(release, monkeypatch):
    verified = []
    monkeypatch.setattr(stockfish, "verify", lambda path: verified.append(path.read_bytes()))
    binary = stockfish.ensure_stockfish()
    assert binary.read_bytes() == b"engine"
    assert binary.stat().st_mode & 0o111
    assert (binary.parent / "Copying.txt").read_bytes() == b"license"
    assert stockfish.ensure_stockfish() == binary
    assert len(release) == 1
    assert "/sf_19/test-engine.tar.gz" in release[0]
    assert verified == [b"engine", b"engine"]


def test_concurrent_builds_download_once(release):
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: stockfish.ensure_stockfish(), range(3)))
    assert len(set(paths)) == 1
    assert len(release) == 1


def test_checksum_mismatch_never_executes_or_installs(release, monkeypatch):
    monkeypatch.setattr(stockfish, "ASSETS", {"x86_64": ("test-engine", "0" * 64)})
    monkeypatch.setattr(stockfish, "verify", lambda path: pytest.fail("unverified download executed"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        stockfish.ensure_stockfish()
    assert not (stockfish.CACHE / "stockfish").exists()
    assert not list(stockfish.CACHE.glob(".install-*"))


def test_wrong_engine_signature_does_not_publish(release, monkeypatch):
    def reject(path):
        raise RuntimeError("engine signature mismatch")

    monkeypatch.setattr(stockfish, "verify", reject)
    with pytest.raises(RuntimeError, match="engine signature mismatch"):
        stockfish.ensure_stockfish()
    assert not (stockfish.CACHE / "stockfish").exists()


def test_interrupted_download_is_retryable(release, monkeypatch):
    download = stockfish.urllib.request.urlopen

    def unavailable(*args, **kwargs):
        raise OSError("connection lost")

    monkeypatch.setattr(stockfish.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="connection lost"):
        stockfish.ensure_stockfish()
    assert not (stockfish.CACHE / "stockfish").exists()
    monkeypatch.setattr(stockfish.urllib.request, "urlopen", download)
    assert stockfish.ensure_stockfish().is_file()


def test_service_uses_cache_without_path_lookup(release, monkeypatch):
    monkeypatch.delenv("STOCKFISH", raising=False)
    assert stockfish.engine_path() == str(stockfish.CACHE / "stockfish")
    monkeypatch.setenv("STOCKFISH", "/custom/verified-engine")
    assert stockfish.engine_path() == "/custom/verified-engine"


def test_existing_task_data_still_requires_engine(tmp_path):
    from pathlib import Path
    from core.task_build import _inspect, _settings
    from tests.test_task_build import Task

    source = Path(__file__).resolve().parents[1] / "tasks/chess/task.yaml"
    (tmp_path / "task.yaml").write_text(source.read_text())
    task = Task(tmp_path, name="chess", variant="none")
    _, required, _ = _settings(task)
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name != "stockfish":
            path.touch()
    task.data_path.write_text('{"id": "existing-position"}\n')
    missing, rows = _inspect(task, required)
    assert rows == [{"id": "existing-position"}]
    assert missing == ["environment/stockfish/sf_19/stockfish"]


def test_all_chess_variants_keep_engine_out_of_staging(tmp_path, monkeypatch):
    import base64
    import json
    from core import stage
    from tasks.chess import build

    env = tmp_path / "environment"
    monkeypatch.setattr(build, "ENV", env)
    cache = env / "stockfish/sf_19"
    cache.mkdir(parents=True)
    marker = b"host-only-engine-artifact-never-stage"
    (cache / "stockfish").write_bytes(marker)
    for tag, entry in build.variants().items():
        build.build(tag, entry)
        for line in (env / tag / "data.jsonl").read_text().splitlines():
            row = json.loads(line)
            payload = stage.collect(row, env / tag)
            assert len(payload) == 8
            assert all("stockfish" not in item["name"] for item in payload)
            assert all(marker not in base64.b64decode(item["b64"]) for item in payload)
            assert all("stockfish" not in source for source in row["files"].values())
