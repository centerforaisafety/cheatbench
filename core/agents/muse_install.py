"""Install an exact Muse release from Meta's versioned, checksummed artifacts.

This standalone module is streamed into the container during harness install.
It never follows the mutable stable channel or installs the auto-update launcher.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import urllib.parse
import urllib.request


def install(version: str, destination: Path):
    if not re.fullmatch(r"\d+\.\d+\.\d+-R\d+(?:\.\d+)?", version):
        raise ValueError("Muse pins require the full release, e.g. 1.3.0-R3401.1")
    query = urllib.parse.urlencode({"channel": "muse", "version": version, "file": "manifest.json"})
    manifest_url = "https://lookaside.facebook.com/lookaside/muse/download/?" + query
    with urllib.request.urlopen(manifest_url, timeout=60) as response:
        manifest = json.load(response)
    if manifest.get("version") != version or manifest.get("checksum_algorithm") != "sha256":
        raise ValueError("Muse release manifest does not match the requested version/checksum algorithm")
    arch = {"x86_64": "x86", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine()]
    system = {"Linux": "linux", "Darwin": "macos"}[platform.system()]
    artifact = manifest["artifacts"][f"{arch}_{system}"]
    url = urllib.parse.urlsplit(artifact["url"])
    if url.scheme != "https" or url.hostname != "lookaside.facebook.com":
        raise ValueError("Muse artifact must use Meta's HTTPS download host")
    expected_hash = artifact["checksum"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Invalid Muse artifact SHA256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, candidate = tempfile.mkstemp(prefix=".muse-", dir=destination.parent)
    try:
        digest, size = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(artifact["url"], timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected_hash or size != artifact["size"]:
            raise ValueError("Muse artifact checksum/size mismatch")
        os.chmod(candidate, 0o755)
        os.replace(candidate, destination)
        return {"version": version, "sha256": expected_hash, "size": size}
    finally:
        Path(candidate).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    print(json.dumps(install(sys.argv[1], Path(sys.argv[2]))))
