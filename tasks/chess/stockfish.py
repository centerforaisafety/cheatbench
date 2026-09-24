"""Pinned, host-only Stockfish installation, prepared before episodes start."""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

VERSION = "sf_19"
CACHE = Path(__file__).resolve().parent / "environment" / "stockfish" / VERSION
# Digests published with the official immutable sf_19 GitHub release.
ASSETS = {
    "x86_64": ("stockfish-linux-x86-64-universal", "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"),
    "aarch64": ("stockfish-linux-arm64-universal", "fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01"),
}


def engine_path() -> str:
    """Resolve the host engine without downloading inside an episode."""
    return os.environ.get("STOCKFISH") or str(CACHE / "stockfish")


def verify(binary: Path) -> None:
    import chess.engine
    from tasks.chess.game import engine_signature

    with chess.engine.SimpleEngine.popen_uci(str(binary), timeout=30) as engine:
        engine.configure(engine_signature.OPTIONS)
        engine_signature.forget(str(binary))
        engine_signature.verify(engine, str(binary))


def ensure_stockfish() -> Path:
    """Download once, verify before publishing, and reuse offline thereafter."""
    if platform.system() != "Linux" or platform.machine() not in ASSETS:
        raise RuntimeError("Automatic Stockfish setup supports Linux x86_64 and aarch64 hosts.")
    name, digest = ASSETS[platform.machine()]
    CACHE.mkdir(parents=True, exist_ok=True)
    binary = CACHE / "stockfish"
    # Direct build.py invocations can overlap too, outside the common build lock.
    with (CACHE / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binary.is_file():
            verify(binary)
            return binary
        url = f"https://github.com/official-stockfish/Stockfish/releases/download/{VERSION}/{name}.tar.gz"
        print(f"Stockfish: downloading pinned {VERSION} from {url}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".install-", dir=CACHE) as temporary:
                stage = Path(temporary)
                archive = stage / "download.tar.gz"
                checksum = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        checksum.update(chunk)
                        out.write(chunk)
                if checksum.hexdigest() != digest:
                    raise RuntimeError("Stockfish archive SHA-256 does not match the pinned release")
                # Read only named regular members; never extract archive paths or links.
                with tarfile.open(archive, "r:gz") as bundle:
                    for member_name, target in [(name, "stockfish"), ("Copying.txt", "Copying.txt")]:
                        member = bundle.getmember(f"stockfish/{member_name}")
                        if not member.isfile():
                            raise RuntimeError(f"Stockfish archive member is not a regular file: {member.name}")
                        with bundle.extractfile(member) as source, (stage / target).open("wb") as out:
                            shutil.copyfileobj(source, out)
                candidate = stage / "stockfish"
                candidate.chmod(0o755)
                verify(candidate)
                (stage / "Copying.txt").replace(CACHE / "Copying.txt")
                candidate.replace(binary)
        except Exception as exc:
            raise RuntimeError(f"Stockfish {VERSION} setup failed: {exc}. No episodes started; rerun the build to retry.") from exc
    return binary
