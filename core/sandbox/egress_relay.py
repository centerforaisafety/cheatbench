"""The two halves of an episode's only way out, as one process.

Started by `core/sandbox/egress.py` with its configuration on stdin. It forks
once and the two halves sit on opposite sides of the episode's network
namespace boundary:

    PARENT   stays in the HOST network namespace. It is the only thing in this
             design that can open an outbound socket, and it opens one only for
             a (host, port) that is on the allowlist it was given at startup.
             It passes the connected socket back over a socketpair as an fd.

    CHILD    joins the container's user+net namespaces and binds the listeners
             the container talks to. It has NO route off the box itself -- the
             firewall `egress.py` installs leaves the namespace with nothing but
             loopback and DNS -- so every byte that leaves the episode leaves
             through a socket the parent chose to open.

An fd is the whole point. A relay that merely proxied would put the decision in
the half the container can reach; passing the descriptor puts it in the half it
cannot, and the child never learns how to make a connection the parent would
have refused.

Two kinds of listener:

    egress      one socket, the destination of the firewall's DNAT for every
                allowlisted address on :80 and :443. It is TRANSPARENT: the
                container believes it is talking to the real host. The child
                reads the TLS ClientHello's SNI (443) or the HTTP Host header
                (80) to learn which host was meant, checks it against the
                allowlist, and asks the parent for that host by NAME.

                Checking the NAME and not the address is what makes this worth
                doing. The addresses of an allowlisted host are shared -- most
                of the allowlist is behind a CDN -- so an address allowlist on
                its own would let `curl --resolve some-other-site:443:<allowed>`
                reach any other site on the same edge. The name comes off the
                wire, from the field the TLS handshake will be validated
                against, so a forged one gets a certificate error rather than a
                page.

    loopback    one socket per port the episode is meant to reach on the HOST's
                loopback -- the `writings` review endpoint is the only one
                today. It listens on the SAME port inside the namespace, so the
                URL the container is handed is the one the host bound and
                nothing about the address the agent can see changes.

Every decision is appended to the log as one JSON object per line: what was
asked for, whether it was allowed, and why not when it was not. A run's record
of what its agent TRIED to reach is a finding in its own right.

STDLIB ONLY, and nothing here imports from `core/`: it runs as its own process
so that nothing forks the orchestrator, and its command line carries a pid and
a port and no word about what is being measured.
"""
from __future__ import annotations

import array
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time

# setns(2), for joining the episode's namespaces. Same mechanism as
# core/sandbox/runtime.py; duplicated rather than imported because this file is
# deliberately free of any dependency on the package it ships in.
import ctypes

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

CAP_NET_BIND_SERVICE = 10
CAP_NET_ADMIN = 12
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_RAISE = 2

# How much of the first packet to look at before deciding. A ClientHello with a
# long extension list can run past a single MTU; 16 KiB is the TLS record
# ceiling and covers every one we will see.
_PEEK = 16384

# Read granularity once a connection is approved. `recv` on whatever is ready,
# relayed immediately, so a streamed response is not held in a buffer.
_CHUNK = 65536


def _setns(pid: int, ns: str) -> None:
    fd = os.open(f"/proc/{pid}/ns/{ns}", os.O_RDONLY)
    try:
        if _libc.setns(fd, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"setns({ns}) on pid {pid}: {os.strerror(err)}")
    finally:
        os.close(fd)


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]


def _raise_ambient_caps(caps: tuple) -> None:
    """Make our namespace capabilities survive into a child `ip` or `iptables`.

    setns() into a user namespace we own hands this process a full capability
    set, but execve() drops what is not inheritable+ambient. Same routine as
    runtime.py's, and for the same reason.
    """
    header = _CapHeader(0x20080522, 0)
    data = (_CapData * 2)()
    if _libc.capget(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "capget")
    for cap in caps:
        idx, bit = cap // 32, 1 << (cap % 32)
        data[idx].effective |= bit
        data[idx].permitted |= bit
        data[idx].inheritable |= bit
    if _libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "capset")
    for cap in caps:
        if _libc.prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_RAISE, cap, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), f"PR_CAP_AMBIENT_RAISE({cap})")


# --------------------------------------------------------------------------
# which host the container meant
# --------------------------------------------------------------------------
def sni_of(data: bytes) -> str | None:
    """The server_name of a TLS ClientHello, or None if this is not one.

    Hand-parsed rather than fed to `ssl`: we must not terminate the handshake,
    only read the name off the front of it and then relay the very same bytes
    to the real host, so the certificate the container validates is the real
    one and no trust store is touched.
    """
    try:
        if len(data) < 45 or data[0] != 0x16:      # not a TLS handshake record
            return None
        if data[5] != 0x01:                        # not a ClientHello
            return None
        pos = 5 + 4 + 2 + 32                       # record, header, version, random
        pos += 1 + data[pos]                       # legacy_session_id
        pos += 2 + int.from_bytes(data[pos:pos + 2], "big")     # cipher_suites
        pos += 1 + data[pos]                       # compression_methods
        end = pos + 2 + int.from_bytes(data[pos:pos + 2], "big")
        pos += 2
        while pos + 4 <= min(end, len(data)):
            kind = int.from_bytes(data[pos:pos + 2], "big")
            size = int.from_bytes(data[pos + 2:pos + 4], "big")
            pos += 4
            if kind != 0x0000:                     # not server_name
                pos += size
                continue
            cur = pos + 2                          # server_name_list length
            while cur + 3 <= pos + size:
                name_len = int.from_bytes(data[cur + 1:cur + 3], "big")
                if data[cur] == 0:                 # host_name
                    return data[cur + 3:cur + 3 + name_len].decode("idna")
                cur += 3 + name_len
            return None
    except (IndexError, ValueError, UnicodeError):
        return None
    return None


def http_host_of(data: bytes) -> str | None:
    """The Host header of a plain HTTP request, without its port."""
    try:
        head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1")
        lines = head.split("\r\n")
        if not lines or " " not in lines[0]:
            return None
        for line in lines[1:]:
            if line.lower().startswith("host:"):
                host = line.split(":", 1)[1].strip()
                # An IPv6 literal is bracketed; anything else splits on the port.
                if host.startswith("["):
                    return host.split("]", 1)[0].lstrip("[")
                return host.split(":", 1)[0]
    except (UnicodeError, ValueError):
        return None
    return None


def host_allowed(host: str, allow: tuple) -> bool:
    """Exact name, or a `*.example.com` entry matching a subdomain of it.

    A bare `*` is not a pattern here and never matches: an allowlist that can be
    written as "everything" is one a task can set by accident.
    """
    host = (host or "").rstrip(".").lower()
    if not host:
        return False
    for entry in allow:
        entry = entry.rstrip(".").lower()
        if entry.startswith("*."):
            suffix = entry[1:]                     # ".example.com"
            if host.endswith(suffix) and len(host) > len(suffix):
                return True
        elif host == entry:
            return True
    return False


# --------------------------------------------------------------------------
# the parent: the only half that can open a socket off the box
# --------------------------------------------------------------------------
def _serve_requests(sock: socket.socket, allow: tuple, loopback: set,
                    log) -> None:
    """Answer the child's connect requests with a descriptor, or a refusal.

    The allowlist is checked HERE, in the half the container has no path to,
    and again by the child before it asks. Two checks because this one is the
    one that matters: the child binds sockets the container connects to, and a
    check that only happened there would be a check on the wrong side.
    """
    while True:
        try:
            raw = sock.recv(4096)
        except OSError:
            return
        if not raw:
            return
        try:
            req = json.loads(raw.decode())
            host, port = str(req["host"]), int(req["port"])
        except (ValueError, KeyError, TypeError):
            _refuse(sock, log, "?", 0, "unparseable request")
            continue

        if req.get("kind") == "loopback":
            # A host-loopback forward, and only to a port declared at startup.
            if port not in loopback or host != "127.0.0.1":
                _refuse(sock, log, host, port, "loopback port not declared")
                continue
        elif not host_allowed(host, allow):
            _refuse(sock, log, host, port, "not on the allowlist")
            continue

        try:
            up = socket.create_connection((host, port), 30)
        except OSError as e:
            _refuse(sock, log, host, port, f"upstream unreachable: {e}")
            continue
        try:
            sock.sendmsg([b'{"ok":true}'],
                         [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                           array.array("i", [up.fileno()]))])
            _log(log, {"host": host, "port": port, "allowed": True})
        except OSError:
            return
        finally:
            up.close()


def _refuse(sock: socket.socket, log, host: str, port: int, why: str) -> None:
    _log(log, {"host": host, "port": port, "allowed": False, "reason": why})
    try:
        sock.sendall(json.dumps({"ok": False, "reason": why}).encode())
    except OSError:
        pass


def _log(log, entry: dict) -> None:
    if log is None:
        return
    entry = {"t": round(time.time(), 3), **entry}
    try:
        log.write(json.dumps(entry) + "\n")
        log.flush()
    except OSError:
        pass


# --------------------------------------------------------------------------
# the child: the listeners inside the episode's network namespace
# --------------------------------------------------------------------------
class _Child:
    """Everything that runs after setns. One lock, because fd passing is a
    request/response exchange on a single socketpair and two connections
    arriving together must not interleave their answers."""

    def __init__(self, sock: socket.socket, allow: tuple):
        self.sock = sock
        self.allow = allow
        self.lock = threading.Lock()

    def borrow(self, host: str, port: int, kind: str) -> socket.socket | None:
        """Ask the parent to connect, and take the descriptor it passes back."""
        req = json.dumps({"host": host, "port": port, "kind": kind}).encode()
        with self.lock:
            try:
                self.sock.sendall(req)
                msg, anc, _, _ = self.sock.recvmsg(4096, socket.CMSG_SPACE(4))
            except OSError:
                return None
            for level, kind_, payload in anc:
                if level == socket.SOL_SOCKET and kind_ == socket.SCM_RIGHTS:
                    fds = array.array("i")
                    fds.frombytes(payload[:len(payload) - (len(payload) % 4)])
                    return socket.socket(fileno=fds[0])
        return None

    def pump(self, near: socket.socket, far: socket.socket) -> None:
        """Relay until either side is done. Bytes only; nothing is inspected."""
        near.settimeout(None)
        far.settimeout(None)
        pair = [near, far]
        try:
            while True:
                readable, _, bad = select.select(pair, [], pair, 300)
                if bad:
                    return
                # A quiet model request is not a closed connection. The
                # client/upstream and episode lifecycle own the deadlines;
                # this poll must not impose a hidden five-minute idle cap.
                if not readable:
                    continue
                for src in readable:
                    data = src.recv(_CHUNK)
                    if not data:
                        return
                    (far if src is near else near).sendall(data)
        except OSError:
            return

    def handle_egress(self, conn: socket.socket) -> None:
        """A DNAT'd :80/:443 connection: learn the host, then ask for it.

        The name is read and NOTHING is decided here. The parent is the only
        side that can open a socket, so the parent is the only side whose
        verdict means anything, and putting the check anywhere else would put
        it on the half the container has a path to.
        """
        try:
            conn.settimeout(30)
            peek = conn.recv(_PEEK, socket.MSG_PEEK)
        except OSError:
            conn.close()
            return
        host = sni_of(peek)
        port = 443
        if host is None:
            host = http_host_of(peek)
            port = 80
        # Asked for even when it is plainly not on the list, and refused on the
        # far side. The refusal is the same either way; the difference is that
        # the parent holds the log, so a forged name -- an allowlisted address
        # carrying somebody else's SNI -- is RECORDED as the attempt it is
        # rather than dropped silently by the half that cannot write it down.
        far = self.borrow(host or "", port, "egress")
        if far is None:
            # Refused: closed with nothing sent. To the container this is a
            # connection that opened and went away, which is what a blocked
            # host looks like from behind any middlebox.
            conn.close()
            return
        try:
            self.pump(conn, far)
        finally:
            far.close()
            conn.close()

    def handle_loopback(self, conn: socket.socket, port: int) -> None:
        far = self.borrow("127.0.0.1", port, "loopback")
        if far is None:
            conn.close()
            return
        try:
            self.pump(conn, far)
        finally:
            far.close()
            conn.close()

    def listen(self, port: int, handler) -> None:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(128)
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handler, args=(conn,), daemon=True).start()


def _run_child(sock: socket.socket, cfg: dict) -> None:
    _setns(cfg["netns_pid"], "user")
    _setns(cfg["netns_pid"], "net")
    _raise_ambient_caps((CAP_NET_ADMIN, CAP_NET_BIND_SERVICE))
    # A namespace slirp4netns has configured has tap0 up but says nothing about
    # loopback, and every listener here is on it.
    subprocess.run(["ip", "link", "set", "lo", "up"], capture_output=True)

    child = _Child(sock, tuple(cfg["allow"]))
    for port in cfg.get("loopback") or ():
        threading.Thread(
            target=child.listen,
            args=(int(port), lambda c, p=int(port): child.handle_loopback(c, p)),
            daemon=True).start()
    child.listen(int(cfg["egress_port"]), child.handle_egress)


def main() -> int:
    cfg = json.load(sys.stdin)
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX,
                                                socket.SOCK_SEQPACKET)
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        try:
            _run_child(child_sock, cfg)
        except BaseException as e:                          # noqa: BLE001
            sys.stderr.write(f"relay child: {type(e).__name__}: {e}\n")
            os._exit(1)
        os._exit(0)

    child_sock.close()
    log = None
    if cfg.get("log"):
        log = open(cfg["log"], "a")

    # Ready only once the child's listeners are bound: the caller installs the
    # firewall next, and a DNAT rule pointing at a socket nobody is listening on
    # would turn an allowed host into a connection refused.
    ready = cfg.get("ready")
    deadline = time.time() + 60
    while time.time() < deadline:
        if _child_listening(cfg["netns_pid"], int(cfg["egress_port"])):
            break
        if os.waitpid(pid, os.WNOHANG)[0]:
            sys.stderr.write("relay child exited during startup\n")
            return 1
        time.sleep(0.05)
    else:
        sys.stderr.write("relay child never bound its listener\n")
        return 1
    if ready:
        with open(ready, "w") as f:
            f.write(str(pid))

    try:
        _serve_requests(parent_sock, tuple(cfg["allow"]),
                        {int(p) for p in (cfg.get("loopback") or ())}, log)
    finally:
        if log is not None:
            log.close()
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    return 0


def _child_listening(netns_pid: int, port: int) -> bool:
    """Is the child's listener bound yet?

    Read out of the namespace's own /proc/net/tcp rather than by connecting:
    the parent is in the host's network namespace and cannot reach the socket
    it is waiting for.
    """
    want = f"{port:04X}"
    try:
        with open(f"/proc/{netns_pid}/net/tcp") as f:
            for line in f.readlines()[1:]:
                cols = line.split()
                if len(cols) > 3 and cols[1].endswith(":" + want) \
                        and cols[3] == "0A":               # TCP_LISTEN
                    return True
    except OSError:
        return False
    return False


if __name__ == "__main__":
    sys.exit(main())
