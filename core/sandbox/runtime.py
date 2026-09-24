#!/usr/bin/env python3
"""OURS: no Harbor counterpart adopted -- Harbor's `BaseEnvironment`
(`src/harbor/environments/base.py`) and its rootless HPC backend
(`environments/singularity/`, an in-container HTTP server) are both an `exec()`
channel onto a live container. We deliberately run exactly one command per
episode -- runner on stdin, one JSON line back -- so neither shape applies.

The enroot sandbox one episode runs in, and the primitives for injecting
into it.

enroot is the only runtime. The image is a squashfs built from the repo-root
Dockerfile by `build.py` next door, and this module starts it.

A task that ships `environment/setup.py` gets its OWN user+network namespace,
because that is what injecting anything into a live episode requires: in a
network namespace owned by a user namespace we own, binding :443 is allowed.
`EpisodeSandbox` is then handed to the hook, which uses primitives like
`start_mirror(...)` to put services in front of the agent. `start_network()`
attaches slirp4netns, which hands the real internet back, NATed in userspace,
so the API and every uncaptured host still work.

A task with no setup hook shares the host's network namespace: the real
internet is reachable and nothing is intercepted.

Everything the sandbox is careful about:

  no repo, no prompts, no rubric   the image is the repo-root Dockerfile and
                                   nothing else -- python:3.11-slim plus
                                   libraries -- and holds none of our files; the
                                   runner, the task and the grader arrive on
                                   stdin.
  no host $HOME                    enroot only binds $HOME when ENROOT_MOUNT_HOME
                                   is set, and we never set it.
  neutral cwd                      /workspace, which the SDK puts in the agent's
                                   system prompt.
  ephemeral filesystem             `enroot start` on a .sqsh mounts the image
                                   read-only via squashfuse and stacks a
                                   fuse-overlayfs whose upper layer is a tmpfs
                                   inside the container's own mount namespace.
                                   The agent may write anywhere, and every byte
                                   is gone when the episode ends.
  private PID namespace            unshared by the payload itself, see below.
  no key in argv                   `enroot start -e ANTHROPIC_API_KEY` (no
                                   `=value`) tells enroot to copy the value out
                                   of its OWN environment; only the NAME is on
                                   the command line that `ps` shows the cluster.
  agent is not root                the payload does the privileged setup and
                                   then unshares a SECOND user namespace in
                                   which it is an ordinary uid 1000 named
                                   "user", because the Claude Code CLI refuses
                                   --dangerously-skip-permissions as root.

Two things enroot does not do for us, and how they are handled:

  enroot shares the host's network and pid namespaces. It has no --net/--pid.
  So the launcher wraps it in `unshare`, and enroot inherits those namespaces.
  For enroot to find its own data directory after `unshare -U` has made us
  uid 0, ENROOT_DATA_PATH must be set explicitly -- its default is computed
  from `id -u`, which reads 0 inside the user namespace.

  `--mount-proc` on that outer unshare is load-bearing, not tidiness. enroot
  mounts the image with

      squashfuse -o uid=${EUID},gid=$(stat -c "%g" /proc/$$) image lower

  and in a new pid namespace bash's $$ is 1, so without a procfs of our own
  /proc/$$ resolves against the HOST's procfs to pid 1 -- systemd, owned by a
  uid that is not mapped in our user namespace. gid comes back as 65534, every
  file in the image is then served with an unmappable gid, and fuse-overlayfs
  cannot copy up anything. With --mount-proc, /proc/1 is our own bash.

  enroot's own root remapping (`--root`) would nest a SECOND user namespace,
  and the private pid namespace is owned by the outer one, so mounting a fresh
  procfs inside the nested namespace fails. Instead we are already root by the
  time enroot starts, and ENROOT_ALLOW_SUPERUSER=y tells enroot to accept that.

That inner `unshare --pid --fork --mount-proc` is also what hides enroot's
plumbing. `enroot start` leaves squashfuse, fuse-overlayfs and a shim shell
running alongside the payload, so a shared pid namespace would let `ps aux` show
the agent the image path and enroot's runtime directory. Inside the payload's
namespace pid 1 is the episode runner and there is nothing else.
"""
from __future__ import annotations

import ctypes
import glob
import os
import random
import shlex
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from .paths import (DEFAULT_DATA_PATH, DEFAULT_IMAGE, DEFAULT_RUNTIME_ROOT,
                    ROOT, SCRATCH, SYSCONF, USERCONF)

# squashfuse, fuse-overlayfs and slirp4netns are unpacked here rather than
# installed by the admins. Jobs get them without having to remember a PATH line.
EXTRA_PATH = [str(Path.home() / "bin"), str(Path.home() / ".local" / "bin")]

# --- the agent is not root ---------------------------------------------------
#
# Everything the sandbox needs root for happens before the agent starts and is
# done by somebody else: the mirror's :443 socket, the loopback aliases,
# /etc/hosts, /etc/resolv.conf and the CA are all written by the HOST through
# setns, and the image's mounts by enroot. The agent itself needs none of it --
# and must not have it, because the Claude Code CLI refuses to start when it
# does ("--dangerously-skip-permissions cannot be used with root/sudo
# privileges"), and `permission_mode` is bypassPermissions for every episode.
#
# Dropping is not a plain setuid: the launcher's user namespace maps exactly one
# id (our uid -> 0), so setuid(1000) is EINVAL. Instead the payload unshares a
# SECOND user namespace in which container-root maps to 1000, so the same kuid
# is simply *named* 1000 inside. That is one `unshare` invocation, deliberately:
# unshare(2) grants the caller full capabilities in the new namespace, it uses
# them to make the pid namespace and mount /proc, and they are gone by the time
# python is exec'd.
#
# The identity is a stock container user, NOT the operator.
AGENT_UID = 1000
AGENT_NAME = "user"
AGENT_HOME = f"/home/{AGENT_NAME}"

# Runs as container-root, before the drop. `>>` rather than a rewrite so the
# base image's own accounts are untouched.
DROP_SETUP = (
    f"printf '{AGENT_NAME}:x:{AGENT_UID}:{AGENT_UID}::{AGENT_HOME}:/bin/bash\\n'"
    " >> /etc/passwd; "
    f"printf '{AGENT_NAME}:x:{AGENT_UID}:\\n' >> /etc/group; "
    f"mkdir -p {AGENT_HOME}; chmod 755 {AGENT_HOME}; "
    # No chown: /workspace and the whole image are owned by container-root,
    # which IS uid 1000 once the second user namespace is in place.
    f"export HOME={AGENT_HOME} USER={AGENT_NAME} LOGNAME={AGENT_NAME}; "
)

DROP_FLAGS = ["--user", f"--map-user={AGENT_UID}", f"--map-group={AGENT_UID}"]


def _key_names(key_env) -> tuple:
    """`key_env` as a tuple of variable names: a str, or a tuple/list of them.

    An adapter whose CLI reads the credential under a different name than the
    host variable (core/agents/base.py `CLI_KEY_ENV`) hands the sandbox every
    name the container needs; the plain string form is what every caller used
    before and still means "this one name".
    """
    if isinstance(key_env, str):
        return (key_env,)
    return tuple(key_env)


def container_argv(image: str, bootstrap: str, *, private_net: bool,
                   workdir: str = "/workspace", pass_key: bool = True,
                   key_env="ANTHROPIC_API_KEY") -> list[str]:
    """The full command line for one episode, `unshare` first.

    `bootstrap` is the tiny python -c program that reads everything it needs
    from stdin. It is the only thing on the container's own command line, so it
    must stay free of anything that identifies this eval -- the agent's Bash
    runs as the same uid and can read /proc/1/cmdline.

    `key_env` is the ONE credential variable the episode's adapter needs
    (`Agent.API_KEY_ENV`). Exactly that one is passed, so a Codex episode has no
    Anthropic key in its environ and a Claude episode has no OpenAI one -- the
    agent can read /proc/self/environ, and a key it has no use for is a key it
    should not be holding.
    """
    ns = ["--user", "--pid", "--fork", "--kill-child", "--mount-proc",
          "--map-root-user"]
    if private_net:
        ns.insert(1, "--net")

    # /sys cannot be a fresh sysfs unless the network namespace is ours (sysfs is
    # network-namespace tagged), and it should not be the host's when it is: the
    # host's /sys/class/net would list interfaces `ip link` inside cannot see,
    # and /sys/class would enumerate the node's GPUs and interconnect.
    sysfs = ("sysfs:/sys:sysfs:x-create=dir,ro,nosuid,nodev,noexec"
             if private_net else
             "/sys:/sys:none:x-create=dir,rbind,ro,nosuid,nodev,noexec,rslave")

    # --kill-child, so that the host killing this command line (a timeout, a
    # cancelled run) takes the episode's pid namespace down with it instead of
    # leaving an orphaned agent talking to the API.
    payload = (DROP_SETUP + f"cd {shlex.quote(workdir)} && exec unshare "
               + " ".join(DROP_FLAGS)
               + f" --pid --fork --kill-child --mount-proc "
                 f"python -c {shlex.quote(bootstrap)}")
    return [
        "unshare", *ns, "--",
        "enroot", "start",
        "--rw",                       # write to the per-episode overlay
        "-m", sysfs,
        "-e", "PYTHONUNBUFFERED=1",
        # Name only: enroot reads the value from its own environment, so the key
        # never appears in a command line other cluster users can read via `ps`.
        # The anonymity probes (build.py:audit_probes, and the pytest around
        # them) pass pass_key=False so their "no API key inside" check is not
        # satisfied by an empty variable.
        *(flag for name in (_key_names(key_env) if pass_key else ())
          for flag in ("-e", name)),
        image,
        "bash", "-c", payload,
    ]


def spawn_env(api_key: str, base: dict | None = None,
              key_env="ANTHROPIC_API_KEY") -> dict:
    """The environment `container_argv` needs. The key travels here, never argv.

    `key_env` must be the same name `container_argv` was given: enroot's
    `-e NAME` reads the VALUE out of this dict.
    """
    # paths.SCRATCH is node-local scratch, so these exist on whichever node the
    # last run happened to land on and not on the next one. enroot creates
    # ENROOT_RUNTIME_PATH itself but not its parent, and fails with a bare
    # "no such file or directory" if it is missing.
    for d in (DEFAULT_DATA_PATH, DEFAULT_RUNTIME_ROOT):
        os.makedirs(d, exist_ok=True)
    env = dict(os.environ if base is None else base)
    path = [p for p in EXTRA_PATH if p not in env.get("PATH", "").split(os.pathsep)]
    env["PATH"] = os.pathsep.join([*path, env.get("PATH", "/usr/bin:/bin")])
    env.update({
        **{name: api_key for name in _key_names(key_env)},
        "ENROOT_ALLOW_SUPERUSER": "y",
        "ENROOT_ROOTFS_WRITABLE": "y",
        "ENROOT_DATA_PATH": DEFAULT_DATA_PATH,
        "ENROOT_SYSCONF_PATH": str(SYSCONF),
        "ENROOT_CONFIG_PATH": str(USERCONF),
        # One runtime dir per episode. enroot mounts a tmpfs here inside its own
        # mount namespace, but the directory is created before that, so sharing
        # one path between concurrent episodes is a race.
        "ENROOT_RUNTIME_PATH":
            f"{DEFAULT_RUNTIME_ROOT}/user-{os.getuid()}-{random.randrange(1 << 32):08x}",
    })
    return env


def ensure_path() -> None:
    """Put our unpacked helpers on this process's PATH.

    slirp4netns is looked up through plain PATH, so a job that forgot
    `export PATH="$HOME/bin:$PATH"` used to fail the whole run.
    """
    parts = os.environ.get("PATH", "").split(os.pathsep)
    missing = [p for p in EXTRA_PATH if p not in parts]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, *parts])


def preflight(private_net: bool = False) -> list[str]:
    """Everything that must be in place before an episode can start.

    Returns a list of problems; empty means good. Checked up front because the
    failure mode otherwise is a container that dies mid-episode with a message
    only enroot understands.
    """
    problems: list[str] = []
    path = os.pathsep.join([*EXTRA_PATH, os.environ.get("PATH", "")])
    exes = ["enroot", "unshare", "squashfuse", "fuse-overlayfs"]
    if private_net:
        # Only a task with a setup hook runs in its own network namespace, and
        # only that namespace needs slirp4netns to reach anything.
        exes.append("slirp4netns")
    for exe in exes:
        if not any(os.access(os.path.join(d, exe), os.X_OK)
                   for d in path.split(os.pathsep) if d):
            problems.append(f"{exe} not on PATH ({path})")
    if not os.path.exists("/dev/fuse"):
        problems.append("/dev/fuse missing: squashfuse cannot mount the image")
    for p in (SYSCONF / "mounts.d" / "10-system.fstab",
              SYSCONF / "hooks.d" / "50-sandbox.sh"):
        if not p.exists():
            problems.append(f"missing {p}")
    return problems


# ==========================================================================
# injecting into a running episode
#
# The container is started with its own user namespace (that we own in turn)
# and an empty network namespace. This half then, from the host:
#
#   1. runs `slirp4netns` against that namespace, which is what puts the real
#      internet back: arxiv.org, Wikipedia and api.anthropic.com all still work,
#      NATed through the host's stack in userspace. Nothing here is privileged
#      -- it is the same mechanism rootless podman uses.
#   2. adds the REAL service addresses of the mirrored site to the container's
#      loopback interface and pins the hostnames to them in the container's
#      /etc/hosts, so `curl -v` inside reports the address it would report on
#      the open internet instead of the 127.0.0.1 that would announce the
#      substitution on the agent's first verbose request.
#   3. appends our CA to the container's trust store, so `curl https://...`
#      succeeds with no `-k` and no warning. The rest of the store is untouched,
#      so certificate checking for every other site still works normally.
#   4. starts the task's mirror server, which joins the same network namespace
#      and serves the capture on :443.
#
# Crucially the mirror data is NOT mounted into the container and the server is
# NOT in the container's PID namespace, so an agent that runs `ls /`, `mount`,
# `ps aux` or reads /proc finds no trace of either. This is the INJECTED half of
# a task's material: it never touches the container filesystem.
#
# Which capture, which hostnames and which item to serve are all the TASK's
# business, decided in its `environment/setup.py`. Nothing below knows.
#
# Failure is fatal by design: if this cannot be set up we must NOT fall back to
# the container talking to the real site.
# ==========================================================================

# Which hostnames a capture stands in for, and what to alias onto the
# container's loopback, are the TASK's business: they are arguments to
# `start_mirror`, not constants here. Nothing in this module names a site.
SLIRP_DNS = "10.0.2.3"          # slirp4netns' built-in DNS forwarder
CA_STORE = "/etc/ssl/certs/ca-certificates.crt"
CERTIFI_GLOB = "/usr/local/lib/python3.*/site-packages/certifi/cacert.pem"

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _setns(pid: int, ns: str) -> None:
    fd = os.open(f"/proc/{pid}/ns/{ns}", os.O_RDONLY)
    try:
        if _libc.setns(fd, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"setns({ns}) on pid {pid}: {os.strerror(err)}")
    finally:
        os.close(fd)


def resolve_site_ips(hosts, fallback_ips: dict | None = None) -> dict:
    """host -> its REAL public addresses, resolved now.

    The real addresses are what get aliased onto the container's loopback, so
    `curl -v` inside reports the address it would report on the open internet
    instead of the 127.0.0.1 that would announce the substitution on the agent's
    first verbose request. `fallback_ips` covers a host with no outbound DNS.
    """
    fallback_ips = fallback_ips or {}
    out = {}
    for host in hosts:
        try:
            ips = sorted({r[4][0] for r in socket.getaddrinfo(
                host, 443, socket.AF_INET, socket.SOCK_STREAM)})
        except OSError:
            ips = []
        out[host] = ips or list(fallback_ips.get(host) or [])
        if not out[host]:
            raise RuntimeError(f"cannot resolve {host} and no fallback address")
    return out


def _ppid_map() -> dict:
    out = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                data = f.read().decode("utf-8", "replace")
            # comm can contain spaces and parens; ppid is the field after the
            # last ')'.
            tail = data[data.rindex(")") + 1:].split()
            out[int(entry)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue
    return out


def find_container_pid(root_pid: int, timeout: float = 90.0,
                       require: tuple = ("net", "mnt"),
                       require_argv: tuple | None = None) -> int:
    """The pid of a process inside `root_pid`'s container namespaces.

    Under enroot the process we spawn is `unshare`, which enters the new network
    and mount namespaces itself and only THEN forks enroot -- so the first
    descendant with a foreign netns is that unshare, whose mount namespace is a
    copy of the HOST's, and the in-container step would have rewritten the
    host's /etc/resolv.conf. A foreign mount namespace is therefore necessary
    but not sufficient.

    `require_argv` is the positive test: the episode process's exact argv, which
    only it has. A substring match would not do -- enroot's own bootstrap shell
    carries the whole payload command line as one argument, matches immediately,
    and is still outside the image. Matching argv exactly also makes this WAIT
    for enroot to finish pivoting into the image instead of racing its startup.

    (/proc/<pid>/root would have been the obvious check and does not work: those
    processes live in a user namespace we do not hold CAP_SYS_PTRACE in, so the
    magic symlink is EACCES from here, while /proc/<pid>/cmdline is world
    readable -- which is the whole reason the argv has to be uninformative.)
    """
    ours = {ns: os.readlink(f"/proc/self/ns/{ns}") for ns in require}
    deadline = time.time() + timeout
    while time.time() < deadline:
        ppids = _ppid_map()
        for pid in sorted(ppids):
            # Walk up to root_pid, bounded so a pid-reuse loop cannot hang us.
            cur, hops = pid, 0
            while cur > 1 and hops < 64:
                if cur == root_pid:
                    break
                cur = ppids.get(cur, 0)
                hops += 1
            if cur != root_pid or pid == root_pid:
                continue
            try:
                if not all(os.readlink(f"/proc/{pid}/ns/{ns}") != ours[ns]
                           for ns in require):
                    continue
                if require_argv is not None:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        argv = f.read().split(b"\0")[:len(require_argv)]
                    if argv != [a.encode() for a in require_argv]:
                        continue
                return pid
            except OSError:
                continue
        time.sleep(0.05)
    raise TimeoutError(f"no container {'+'.join(require)} namespace under pid "
                       f"{root_pid} after {timeout:.0f}s")


def _in_child(pid: int, namespaces: tuple, fn) -> None:
    """Run fn() in a forked child that has joined `namespaces` of `pid`.

    A fork rather than a helper script: the child inherits the work to do as a
    closure, so nothing about this eval is ever written to a command line.
    """
    r, w = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover - child process
        os.close(r)
        try:
            for ns in namespaces:
                _setns(pid, ns)
            fn()
            os.write(w, b"ok")
            os._exit(0)
        except BaseException as e:  # noqa: BLE001
            try:
                os.write(w, f"{type(e).__name__}: {e}".encode()[:400])
            except OSError:
                pass
            os._exit(1)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        msg = f.read()
    _, status = os.waitpid(child, 0)
    if status != 0:
        raise RuntimeError(f"in-container step failed: "
                           f"{msg.decode('utf-8', 'replace')}")


def _write_container_files(ca_pem: str, host_lines: str, hosts: tuple) -> None:
    """Runs INSIDE the container's mount namespace (the per-episode overlay)."""
    Path("/etc/resolv.conf").write_text(f"nameserver {SLIRP_DNS}\n")

    hosts_file = Path("/etc/hosts")
    existing = hosts_file.read_text() if hosts_file.exists() else ""
    # Drop any earlier entry for these names first: a retried attach must not
    # leave the file with the same host listed twice, which looks hand-edited.
    kept = [ln for ln in existing.splitlines()
            if not any(ln.split()[1:2] == [h] for h in hosts)]
    hosts_file.write_text("\n".join(kept).rstrip("\n") + "\n" + host_lines)

    # Every module used after setns(mnt) must already be imported: the host
    # filesystem -- and with it Python's stdlib -- is gone in there.
    for store in [CA_STORE, *glob.glob(CERTIFI_GLOB)]:
        p = Path(store)
        if not p.exists():
            continue
        text = p.read_text()
        if ca_pem.strip() in text:
            continue
        # Appended, never replaced: every real root stays trusted, so TLS to
        # every other site keeps working exactly as before.
        p.write_text(text.rstrip("\n") + "\n" + ca_pem.strip() + "\n")


_CAP_NET_ADMIN = 12
_CAP_NET_BIND_SERVICE = 10
_CAP_VERSION_3 = 0x20080522
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_RAISE = 2


def _raise_ambient_caps(caps: tuple) -> None:
    """Make our namespace capabilities survive execve().

    setns() into a user namespace we own hands this process a full capability
    set, but exec'ing `ip` would drop it: for a non-root uid with no file
    capabilities the kernel recomputes the permitted set as empty. Ambient
    capabilities are the supported way to carry a capability across exec.
    """
    header = ctypes.create_string_buffer(struct.pack("Ii", _CAP_VERSION_3, 0))
    data = ctypes.create_string_buffer(24)
    if _libc.capget(header, data) != 0:
        raise OSError(ctypes.get_errno(), "capget")
    eff0, perm0, inh0, eff1, perm1, inh1 = struct.unpack("6I", data.raw)
    data = ctypes.create_string_buffer(
        struct.pack("6I", eff0, perm0, inh0 | perm0, eff1, perm1, inh1 | perm1))
    if _libc.capset(header, data) != 0:
        raise OSError(ctypes.get_errno(), "capset")
    for cap in caps:
        if _libc.prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_RAISE, cap, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), f"PR_CAP_AMBIENT_RAISE({cap})")


def _add_loopback_addresses(ips: list) -> None:
    """Runs INSIDE the container's user+net namespace (host mount ns kept)."""
    _raise_ambient_caps((_CAP_NET_ADMIN, _CAP_NET_BIND_SERVICE))
    for cmd in (["ip", "link", "set", "lo", "up"],
                *[["ip", "addr", "add", f"{ip}/32", "dev", "lo"] for ip in ips]):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and "File exists" not in res.stderr:
            raise RuntimeError(f"{' '.join(cmd)}: {res.stderr.strip()}")



def start_slirp(pid: int, log_path=None) -> subprocess.Popen:
    """slirp4netns against `pid`'s network namespace: the real internet, back.

    Userspace NAT through the host's stack -- the same mechanism rootless
    podman uses, and nothing here is privileged. Without it a private network
    namespace has nothing in it but loopback, and the agent's own CLI could not
    reach the API.
    """
    r, w = os.pipe()
    os.set_inheritable(w, True)
    cmd = ["slirp4netns", "--configure", "--mtu=65520",
           # The container must not be able to reach services bound to the
           # host's loopback -- including other episodes' injected servers.
           "--disable-host-loopback",
           f"--ready-fd={w}", str(pid), "tap0"]
    proc = subprocess.Popen(
        cmd, pass_fds=(w,), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL if log_path is None else open(log_path, "ab"))
    os.close(w)
    with os.fdopen(r, "rb") as f:
        ready = f.read(1)
    if not ready:
        raise RuntimeError("slirp4netns exited before signalling ready")
    return proc


def net_pid_for(launcher_pid: int, container_pid: int) -> int:
    """Which pid to hand the network-namespace work.

    setns is capability-checked against the user namespace that OWNS the target
    namespace. The agent process sits in a second, nested user namespace (it is
    dropped to uid 1000 that way), and a process that has joined a nested user
    namespace has no capabilities in its parent -- so `setns(user)` on the agent
    followed by `setns(net)` on a network namespace the OUTER namespace owns is
    EPERM.

    The launcher -- the `unshare` the trial spawned -- is in that outer user
    namespace and in the very same network namespace (it calls unshare(2) for
    itself before forking), so using it for the netns work keeps the capability
    chain intact. The mount namespace work still goes to the agent's own pid,
    which is where the per-episode overlay is.
    """
    want = os.readlink(f"/proc/{container_pid}/ns/net")
    got = os.readlink(f"/proc/{launcher_pid}/ns/net")
    if got != want:
        # Better to fail loudly than to put an injected service in the wrong
        # netns and let the episode reach the real site.
        raise RuntimeError(f"launcher pid {launcher_pid} is in {got}, but the "
                           f"container is in {want}")
    return launcher_pid


class MirrorSandbox:
    """One mirrored site, injected into a running episode's namespaces."""

    def __init__(self, *, mirror_root: Path, tls_dir: Path, server_script: Path,
                 python_exe: str | None = None, overrides: Path | None = None,
                 only: str | None = None, hosts: tuple = (),
                 fallback_ips: dict | None = None, extra_loopback: tuple = (),
                 request_log: Path | None = None, server_log: Path | None = None,
                 port: int = 443, container_argv: tuple | None = None):
        self.mirror_root = Path(mirror_root)
        self.tls_dir = Path(tls_dir)
        self.server_script = Path(server_script)
        self.python_exe = python_exe or sys.executable
        self.overrides = Path(overrides) if overrides else None
        # One episode, one served item: an overrides file carries every row's,
        # and serving all of them at once is a tell.
        self.only = only or None
        self.hosts = tuple(hosts)
        self.fallback_ips = dict(fallback_ips or {})
        # Bare-IP targets aliased onto the container's loopback but NOT
        # hostnames: an IP "resolves" to itself, so there is no DNS step and no
        # /etc/hosts line -- the loopback alias alone is enough to make `curl
        # http://<ip>/` inside reach a local service. Because the address is
        # squatted on the container's lo, ALL traffic the agent aims at it is
        # answered locally and NEVER leaves the box.
        self.extra_loopback = tuple(extra_loopback)
        self.request_log = Path(request_log) if request_log else None
        self.server_log = Path(server_log) if server_log else None
        self.port = port
        self.container_argv = container_argv
        self.container_pid: int | None = None
        self._slirp: subprocess.Popen | None = None
        self._owns_slirp = False
        self._server: subprocess.Popen | None = None
        self._ready_file: Path | None = None
        self._log_fh = None

    # -- lifecycle -------------------------------------------------------
    def attach(self, launcher_pid: int, timeout: float = 120.0,
               container_pid: int | None = None, net_pid: int | None = None,
               slirp: subprocess.Popen | None = None) -> int:
        """Set the whole thing up. Returns the container pid.

        `launcher_pid` is the `unshare` process the trial spawned. The pids and
        an already-running slirp4netns can be passed in when the caller has
        them (EpisodeSandbox does), so two injected services share one.
        """
        for path in (self.mirror_root / "manifest.json",
                     self.tls_dir / "site.crt", self.tls_dir / "site.key",
                     self.tls_dir / "ca.crt", self.server_script):
            if not path.exists():
                raise FileNotFoundError(f"mirror sandbox needs {path}")

        pid = container_pid or find_container_pid(
            launcher_pid, timeout=timeout, require=("net", "mnt"),
            require_argv=self.container_argv)
        self.container_pid = pid
        net_pid = net_pid or net_pid_for(launcher_pid, pid)
        try:
            if slirp is None:
                self._slirp = start_slirp(net_pid, self.server_log)
                self._owns_slirp = True
            else:
                self._slirp = slirp

            ips = resolve_site_ips(self.hosts, self.fallback_ips)
            # The bare-IP targets get a loopback alias too, but no /etc/hosts
            # line (they are not hostnames); a local server binds them.
            flat = ([ip for host in self.hosts for ip in ips[host]]
                    + list(self.extra_loopback))
            _in_child(net_pid, ("user", "net"),
                      lambda: _add_loopback_addresses(flat))

            ca_pem = (self.tls_dir / "ca.crt").read_text()
            host_lines = "".join(f"{ips[h][0]}\t{h}\n" for h in self.hosts)
            hosts = self.hosts
            _in_child(pid, ("user", "mnt"),
                      lambda: _write_container_files(ca_pem, host_lines, hosts))

            self._start_server(net_pid)
        except BaseException:
            # Half a sandbox is worse than none: a leftover slirp4netns owns
            # tap0 and the next attempt cannot re-create it.
            self.close()
            raise
        return pid

    def close(self) -> None:
        procs = [self._server]
        if self._owns_slirp:
            procs.append(self._slirp)
        for proc in procs:
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._server = None
        if self._owns_slirp:
            self._slirp = None
            self._owns_slirp = False
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if self._ready_file is not None:
            try:
                self._ready_file.unlink()
            except OSError:
                pass
            self._ready_file = None

    def check_health(self) -> None:
        if self._server is not None and self._server.poll() is not None:
            raise RuntimeError(
                f"Episode mirror service exited {self._server.returncode}; see {self.server_log}")

    # -- pieces ----------------------------------------------------------
    def _start_server(self, pid: int) -> None:
        ready = Path(f"/tmp/.rh_mirror_ready_{os.getpid()}_{pid}_{self.port}")
        if ready.exists():
            ready.unlink()
        self._ready_file = ready
        cmd = [self.python_exe, str(self.server_script),
               "--mirror", str(self.mirror_root),
               "--cert", str(self.tls_dir / "site.crt"),
               "--key", str(self.tls_dir / "site.key"),
               "--port", str(self.port), "--bind", "0.0.0.0",
               "--netns-pid", str(pid), "--ready-file", str(ready)]
        if self.overrides:
            cmd += ["--overrides", str(self.overrides)]
            if self.only:
                cmd += ["--only", self.only]
        if self.request_log:
            cmd += ["--log", str(self.request_log)]
        out = subprocess.DEVNULL
        if self.server_log:
            self._log_fh = open(self.server_log, "ab")
            out = self._log_fh
        self._server = subprocess.Popen(cmd, stdout=out, stderr=out,
                                        stdin=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline:
            if ready.exists():
                return
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"mirror server exited {self._server.returncode} during "
                    f"startup (see {self.server_log})")
            time.sleep(0.1)
        raise TimeoutError("mirror server never reported ready")


# --------------------------------------------------------------------------
# the handle a task's setup hook is given
# --------------------------------------------------------------------------
class EpisodeSandbox:
    """A running episode's namespaces, and the primitives for injecting into them.

    This is what `core/trial.py` hands to a task's `environment/setup.py`. It is
    the ONLY way a task reaches into a live episode, and it is deliberately
    generic: `core/` offers primitives and knows nothing about what any task
    serves through them.

    Everything started here lives on the HOST, in the container's network
    namespace but outside its mount and pid namespaces, so an agent that runs
    `ls /`, `mount` or `ps aux` finds no trace of it.
    """

    def __init__(self, launcher_pid: int, container_argv: tuple,
                 log_dir: Path, python_exe: str | None = None,
                 timeout: float = 120.0):
        self.launcher_pid = launcher_pid
        self.container_argv = tuple(container_argv)
        self.log_dir = Path(log_dir)
        self.python_exe = python_exe or sys.executable
        self.timeout = timeout
        self._container_pid: int | None = None
        self._net_pid: int | None = None
        self._slirp: subprocess.Popen | None = None
        self._parts: list = []

    # -- namespaces ------------------------------------------------------
    @property
    def container_pid(self) -> int:
        """The agent's own pid, found by waiting for its exact argv to appear."""
        self._resolve()
        return self._container_pid

    def _resolve(self) -> tuple:
        if self._container_pid is None:
            pid = find_container_pid(self.launcher_pid, timeout=self.timeout,
                                     require=("net", "mnt"),
                                     require_argv=self.container_argv)
            self._container_pid = pid
            self._net_pid = net_pid_for(self.launcher_pid, pid)
        return self._container_pid, self._net_pid

    def start_network(self) -> None:
        """Give the episode the real internet. Idempotent.

        A private network namespace starts empty. `core/trial.py` calls this
        after the setup hook has run, so a task that injected nothing still gets
        a working network, and a task that already started one is untouched.
        """
        if self._slirp is not None and self._slirp.poll() is None:
            return
        _, net_pid = self._resolve()
        self._slirp = start_slirp(net_pid, self.log_dir / "slirp.log")

    def lock_egress(self, hosts, loopback_ports=(), *, allow_dns: bool = True):
        """Deny this episode's egress, then open it for `hosts` BY NAME.

        Called by `core/trial.py` after the setup hook and `start_network()`,
        while the container is still blocked reading its task from stdin. See
        `core/sandbox/egress.py` for what it installs and what it cannot stop.

        `loopback_ports` are ports on the HOST's loopback the episode is meant
        to reach -- the `writings` review endpoint -- forwarded onto the same
        port number inside the namespace so the URL the container is handed is
        the one the host bound.
        """
        from .egress import EgressGuard        # local: egress.py imports us

        container_pid, net_pid = self._resolve()
        guard = EgressGuard(net_pid=net_pid, container_pid=container_pid,
                            hosts=hosts, loopback_ports=loopback_ports,
                            log_dir=self.log_dir, python_exe=self.python_exe,
                            allow_dns=allow_dns)
        guard.start()
        self._parts.append(guard)
        return guard

    # -- primitives ------------------------------------------------------
    def route_mcp_only(self, blob, base_url: str, api_key: str):
        """Filter model tool declarations without choosing the network policy.

        Return the rewritten payload and bridge port. The caller forwards this
        port under the same task network policy used by other episodes.
        """
        from .native_inference import InferenceBridge, reroute_blob
        import json

        task = json.loads(blob).get("task", {})
        grok_mcp = bool(task.get("grok", {}).get("mcp_only"))
        mcp_server = task.get("row", {}).get("tool_surface", "chess")
        bridge = InferenceBridge(base_url, api_key, self.log_dir / "inference_requests.jsonl",
                                 grok_mcp=grok_mcp, mcp_server=mcp_server)
        self._parts.append(bridge)
        routed = reroute_blob(blob, bridge.local_base_url)
        return routed, bridge.server_port

    def start_mirror(self, *, site, server, hosts, serving: str | None = None,
                     overrides=None, fallback_ips=None, extra_loopback=(),
                     port: int = 443) -> int:
        """Serve a captured site to this episode, in place of the real hosts.

            site           the capture: manifest.json, pages/, tls/
            server         the server script; it joins the episode's netns
            hosts          the hostnames the capture stands in for
            serving        which item of `overrides` this episode gets; the rest
                           404, so one capture serves 200 rows one at a time
            overrides      JSON map of path -> response that wins over the capture
            fallback_ips   host -> addresses, for a node with no outbound DNS
            extra_loopback bare IPs to alias onto the container's loopback

        The capture is NOT mounted into the container and the server is NOT in
        the container's pid namespace.
        """
        container_pid, net_pid = self._resolve()
        self.start_network()
        m = MirrorSandbox(
            mirror_root=Path(site), tls_dir=Path(site) / "tls",
            server_script=Path(server), python_exe=self.python_exe,
            overrides=Path(overrides) if overrides else None,
            only=serving, hosts=tuple(hosts), fallback_ips=fallback_ips,
            extra_loopback=tuple(extra_loopback), port=port,
            request_log=self.log_dir / "mirror_requests.log",
            server_log=self.log_dir / "mirror_server.log",
            container_argv=self.container_argv)
        m.attach(self.launcher_pid, timeout=self.timeout,
                 container_pid=container_pid, net_pid=net_pid,
                 slirp=self._slirp)
        self._parts.append(m)
        return container_pid

    def check_health(self) -> None:
        for part in self._parts:
            check = getattr(part, "check_health", None)
            if check is not None:
                check()

    def close(self) -> None:
        for part in reversed(self._parts):
            try:
                part.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
        self._parts = []
        if self._slirp is not None and self._slirp.poll() is None:
            self._slirp.terminate()
            try:
                self._slirp.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._slirp.kill()
                self._slirp.wait(timeout=5)
        self._slirp = None


# --------------------------------------------------------------------------
# running a one-off command in the sandbox
# --------------------------------------------------------------------------
# This is what the anonymity probes go through (build.py:audit_built_image and
# tests/test_image_anonymity.py). Deliberately the same shape as a real episode:
# nothing but the bootstrap on the command line, the actual command streamed in
# on stdin. An audit that took a shortcut here would be auditing something other
# than what the agent gets.
SHELL_BOOTSTRAP = (
    'import sys,json,subprocess;d=json.load(sys.stdin);'
    'sys.exit(subprocess.run(["bash","-c",d["cmd"]]).returncode)'
)


def run_shell(command: str, image: str = "", *, private_net: bool = False,
              timeout: int = 300):
    """Run `command` inside the sandbox and return the CompletedProcess."""
    import json

    argv = container_argv(image or DEFAULT_IMAGE, SHELL_BOOTSTRAP,
                          private_net=private_net, pass_key=False)
    return subprocess.run(argv, env=spawn_env(""),
                          input=json.dumps({"cmd": command}),
                          capture_output=True, text=True, timeout=timeout)
