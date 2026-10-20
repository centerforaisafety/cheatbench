"""A host-loopback stand-in for a self-hosted model server (vLLM, SGLang, ...).

A model entry may point at a plain-HTTP server on another machine, e.g.

    api_base_url: ${VLLM_BASE_URL}        # http://10.0.0.5:8000/v1

That works for tasks with open networking, but two task policies cannot reach
such an origin:

    egress-enforced   the sandbox relays only a LOOPBACK model endpoint into
                      the episode (egress.api_loopback_ports); a remote
                      host:port is not on the :80/:443 allowlist path.
    mcp_only          the host inference bridge (native_inference.py) accepts
                      only an HTTPS origin.

For those, `run.py` starts an EndpointBridge: a byte-level TCP forwarder from
127.0.0.1 to the remote server, TLS-terminating for mcp_only tasks with a
throwaway self-signed certificate for 127.0.0.1. The run then uses the local
URL; the remote server never sees anything but the agent's own requests.

Only plain `http://` to a non-loopback host is bridged. Provider endpoints
(https) and a server already on loopback keep the existing routing.
"""
from __future__ import annotations

import asyncio
import datetime
import ipaddress
import os
import ssl
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def bridge_mode(api_base_url: str | None, *, mcp_only: bool, egress_enforced: bool) -> str | None:
    """'tls', 'tcp' or None (no bridge) for this endpoint under this task's policy."""
    if not api_base_url:
        return None
    parsed = urlsplit(api_base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.hostname in LOOPBACK:
        return None
    if mcp_only:
        return "tls"
    if egress_enforced:
        return "tcp"
    return None


def _self_signed(directory: Path) -> tuple[Path, Path]:
    """A one-day certificate for 127.0.0.1 (SAN IP + localhost), key unencrypted."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cheatbench endpoint bridge")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = directory / "bridge.crt", directory / "bridge.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    key_path.chmod(0o600)
    return cert_path, key_path


def _trust(cert_path: Path, directory: Path) -> Path:
    """SSL_CERT_FILE bundle: the system/certifi roots plus the bridge certificate.

    Every TLS client in this process (the inference bridge's Upstream, the judge,
    the grader) reads SSL_CERT_FILE, so the bundle keeps the public roots.
    """
    roots = os.environ.get("SSL_CERT_FILE") or ssl.get_default_verify_paths().cafile
    if not roots or not Path(roots).exists():
        import certifi
        roots = certifi.where()
    bundle = directory / "ca-bundle.crt"
    bundle.write_bytes(Path(roots).read_bytes().rstrip(b"\n") + b"\n" + cert_path.read_bytes())
    return bundle


class EndpointBridge:
    """127.0.0.1:<ephemeral> -> the remote server, on a daemon thread's event loop."""

    def __init__(self, api_base_url: str, mode: str):
        if mode not in ("tcp", "tls"):
            raise ValueError(f"unknown bridge mode {mode!r}")
        self.mode = mode
        parsed = urlsplit(api_base_url)
        self._parsed = parsed
        self._rhost, self._rport = parsed.hostname, parsed.port or 80
        # Host-only, as resolved_routing records URLs (no userinfo, no query).
        self.upstream_url = urlunsplit(("http", f"{self._rhost}:{self._rport}", parsed.path, "", ""))
        self._tmp = tempfile.TemporaryDirectory(prefix="cheatbench-bridge-")
        self._ssl = None
        self._old_cert_file = os.environ.get("SSL_CERT_FILE")
        if mode == "tls":
            cert, key = _self_signed(Path(self._tmp.name))
            self._ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl.load_cert_chain(cert, key)
            os.environ["SSL_CERT_FILE"] = str(_trust(cert, Path(self._tmp.name)))
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._server = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="endpoint-bridge", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error or not self._server:
            self.close()
            raise RuntimeError(f"endpoint bridge failed to start: {self._error}")

    def _run(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._server = self._loop.run_until_complete(asyncio.start_server(
                self._handle, "127.0.0.1", 0, backlog=512, ssl=self._ssl))
        except BaseException as exc:  # reported by __init__
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _handle(self, creader, cwriter):
        try:
            rreader, rwriter = await asyncio.open_connection(self._rhost, self._rport)
        except OSError:
            cwriter.close()
            return
        await asyncio.gather(_pipe(creader, rwriter), _pipe(rreader, cwriter))

    @property
    def port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    @property
    def local_base_url(self) -> str:
        scheme = "https" if self.mode == "tls" else "http"
        return urlunsplit((scheme, f"127.0.0.1:{self.port}", self._parsed.path, "", ""))

    async def _shutdown(self):
        if self._server:
            self._server.close()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._loop.stop()

    def close(self):
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        self._thread.join(5)
        if self.mode == "tls":
            if self._old_cert_file is None:
                os.environ.pop("SSL_CERT_FILE", None)
            else:
                os.environ["SSL_CERT_FILE"] = self._old_cert_file
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


async def _pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError, ssl.SSLError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass
