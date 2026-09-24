"""Opt-in egress control for an episode, and the allowlist a task opens it with.

THE PROBLEM THIS EXISTS FOR. A task's `tools:` block takes the agent's search
and fetch TOOLS away, and that is all it takes away: those tools run on the
vendor's servers, and removing them says nothing about what a shell inside the
container can do. Until this module, a task with no `environment/setup.py` ran
in the HOST's network namespace, so `curl`, `urllib.request` and a raw socket
all reached the open internet. Measured on `writings`, which sets both tool keys
false, an agent reached nine different sites and downloaded ten real .pptx files
from them. Every task whose answer is findable online was measuring something
other than what it meant to.

WHAT REPLACES IT, FOR A TASK THAT ASKS. A task that declares a `network:`
block gets its own network namespace, and that namespace is emptied and then
opened by name:

  1. `slirp4netns` gives the namespace a route, as it always did for a task with
     a setup hook. It is started with `--disable-host-loopback`, so the host's
     own services -- including other episodes' -- are not reachable through it.

  2. A firewall goes in from the HOST side, into a namespace the container's own
     processes have no authority over (see WHY THIS HOLDS below). Its OUTPUT
     policy is DROP. Loopback and DNS pass. Every allowlisted ADDRESS on :80 and
     :443 is DNAT'd to a listener on loopback. Nothing else has anywhere to go,
     so a connection to an address that is not on the list fails at connect(),
     immediately, rather than hanging until the episode's clock runs out.

  3. `egress_relay.py` runs across the boundary: a child inside the namespace
     holding the listeners, a parent outside it holding the only ability to open
     an outbound socket, and a socketpair between them over which the parent
     passes back a connected descriptor -- but only for a host that is on the
     allowlist BY NAME, read from the TLS SNI or the HTTP Host header.

Checking the name and not just the address is the half that does the work. Most
of what has to stay reachable is behind a CDN, and a CDN address serves whatever
site asks it to, so an address allowlist alone would have let
`curl --resolve pitch.com:443:<an allowed address>` fetch pitch.com. Both checks
together mean an address that is not allowlisted has no route, and an allowlisted
address carrying somebody else's name gets nothing.

WHY IT IS OPT-IN AND NOT THE DEFAULT. This module first shipped with deny as
the default for every task, and that was reverted: the open network is
deliberate for the tasks already measured on it, and changing the default
changes every one of those environments without anyone having said so. So a task
that declares nothing keeps the host's network exactly as before, and the only
environments this module touches are the ones whose task.yaml asks for it. The
cost is the one that revert accepted, and it is real: a task whose answer is
findable online and whose author never writes the key is still measuring
something other than what it meant to.

WHY THIS HOLDS. The firewall lives in a network namespace owned by the user
namespace `unshare --user --net` created for the episode. The agent's processes
run in a SECOND user namespace nested inside that one (that is how they are
dropped to uid 1000, see runtime.py), and a process in a nested user namespace
holds no capability in its parent. So `iptables -F` from inside the container is
EPERM -- verified, it reports "Permission denied (you must be root)" to a shell
whose `whoami` says root -- and so is `ip link set lo down`. There is nothing to
unset and no binary whose absence is being relied on: an agent that brings its
own statically linked netlink client gets the same EPERM.

WHAT THIS DOES NOT STOP, stated plainly because a lockdown nobody can describe
the holes in is not one:

  * The allowlist is never empty. Every episode installs its own agent CLI --
    the image deliberately bakes no vendor's binary -- so nvm, Node, npm and PyPI
    have to stay reachable for the whole episode, and so does the model API. An
    agent that wanted arbitrary bytes could publish or fetch them through a
    package registry. It cannot reach a search engine or an ordinary web page,
    which is the failure that prompted this.
  * DNS resolves. Names still answer; connections to them do not. A name that
    resolves and then refuses to connect is what a filtered network looks like,
    and denying resolution as well would only change the error text.
  * The agent may talk to the endpoints on the allowlist, including the review
    endpoint whose token its task hands it. That was always true and is the
    task's design, not a gap in this.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .runtime import _in_child, _raise_ambient_caps, SLIRP_DNS
from .runtime import _CAP_NET_ADMIN, _CAP_NET_BIND_SERVICE

RELAY = Path(__file__).resolve().parent / "egress_relay.py"

# WHAT EVERY EPISODE NEEDS TO EXIST AT ALL, allowlisted for every task including
# the ones that declare `egress: deny`. This is not a policy choice a task gets
# to make: an episode that cannot install its agent or reach the model API is
# not an episode.
#
#   the model API      whichever host the run's `api_base_url:` resolves to,
#                      added per-episode by `hosts_for()`, not listed here.
#   nvm + Node         core/agents/installed.py `node_install` fetches nvm's
#                      installer from raw.githubusercontent.com, which redirects
#                      through github.com, and nvm then fetches Node from
#                      nodejs.org.
#   npm                every CLI adapter installs its binary with `npm -g`.
#   PyPI               the Claude adapter pip-installs its SDK.
#
# THIS LIST IS THE BIGGEST HOLE IN THE LOCKDOWN and it is here rather than
# scattered so that it can be deleted in one edit: bake the agent CLIs into the
# image and every entry below goes, leaving the model API as the only way out.
INSTALL_HOSTS = (
    "raw.githubusercontent.com",
    "github.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
    "nodejs.org",
    "registry.npmjs.org",
    "pypi.org",
    "files.pythonhosted.org",
)

# Ports the DNAT covers. A host is reachable on the ports the web uses and on no
# others: the relay reads the name off a TLS ClientHello or an HTTP request, and
# a protocol that is neither gives it nothing to check.
WEB_PORTS = (80, 443)


class EgressError(RuntimeError):
    """The lockdown could not be established. Never caught into a warning: an
    episode that runs without it produces a number we cannot stand behind."""


# --------------------------------------------------------------------------
# the task's declaration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkPolicy:
    """A task's `network:` block. Three states, and having no block is one.

        (no block at all)                   # the host's network, unrestricted

        network:
          egress: deny                      # infrastructure and nothing else

        network:
          egress: allow
          hosts: [terrytao.wordpress.com, "*.example.org"]

    ABSENT MEANS OPEN, which is a deliberate retreat from the default this
    module first shipped with -- see WHY IT IS OPT-IN in the module docstring.
    `egress: open` may also be written out, so a task can put on the record that
    it chose the open network rather than never considered the question.

    Writing the block at all means restriction, so `egress:` inside one defaults
    to `deny`. Only `deny` and `allow` put a firewall in front of an episode.
    """

    egress: str = "open"
    hosts: tuple = ()

    @property
    def enforced(self) -> bool:
        """Whether this policy puts a firewall in front of the episode at all."""
        return self.egress in ("deny", "allow")

    @property
    def denies(self) -> bool:
        return self.egress == "deny"


def parse_network(cfg_path, raw) -> NetworkPolicy:
    """`network:` into a policy. Anything malformed is a refusal, not a default."""
    if raw is None:
        return NetworkPolicy()
    if not isinstance(raw, dict):
        raise SystemExit(f"{cfg_path}: `network:` must be a map, e.g. "
                         f"`network: {{egress: deny}}`")
    unknown = set(raw) - {"egress", "hosts"}
    if unknown:
        raise SystemExit(f"{cfg_path}: network: unknown key(s) {sorted(unknown)}")
    egress = str(raw.get("egress", "deny"))
    if egress not in ("deny", "allow", "open"):
        raise SystemExit(f"{cfg_path}: network: egress: must be `deny`, `allow` "
                         f"or `open`, not {egress!r}")
    hosts = raw.get("hosts") or []
    if not isinstance(hosts, (list, tuple)):
        raise SystemExit(f"{cfg_path}: network: hosts: must be a list")
    hosts = tuple(str(h).strip().lower() for h in hosts if str(h).strip())
    if egress != "allow" and hosts:
        raise SystemExit(f"{cfg_path}: network: egress: {egress} takes no "
                         f"`hosts:`. Say `egress: allow` to open the ones you "
                         f"listed.")
    if egress == "allow" and not hosts:
        raise SystemExit(f"{cfg_path}: network: egress: allow needs `hosts:`. "
                         f"An allow with no hosts is a deny written the long way.")
    if "*" in hosts:
        raise SystemExit(f"{cfg_path}: network: hosts: `*` is not a host. "
                         f"Name them, or use `*.example.com` for one domain.")
    return NetworkPolicy(egress=egress, hosts=hosts)


def hosts_for(policy: NetworkPolicy, api_base_url: str = "",
              install_hosts: tuple = ()) -> tuple:
    """Every hostname this episode may reach: infrastructure, then the task's.

    `api_base_url` is the run's resolved model endpoint; empty means the vendor
    default, whose host is not known here, so both vendors' are allowed. That
    over-allows by one hostname the agent could reach anyway with the key it is
    holding, and under-allowing would strand the episode.
    """
    api: tuple = ()
    if api_base_url:
        host = _host_of(api_base_url)
        api = (host,) if host else ()
    else:
        api = ("api.anthropic.com", "api.openai.com")
    return tuple(dict.fromkeys((*api, *INSTALL_HOSTS, *install_hosts, *policy.hosts)))


def api_loopback_ports(api_base_url: str) -> tuple:
    """Relay only the configured localhost model endpoint into the sandbox."""
    from urllib.parse import urlsplit
    parsed = urlsplit(api_base_url)
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        return ()
    if parsed.scheme not in ("http", "https"):
        return ()
    return (parsed.port or (443 if parsed.scheme == "https" else 80),)


def _host_of(url: str) -> str:
    from urllib.parse import urlsplit
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


# --------------------------------------------------------------------------
# putting it in place
# --------------------------------------------------------------------------
def resolve_hosts(hosts) -> dict:
    """host -> its IPv4 addresses. A name that will not resolve maps to [].

    Every address a name answers with is taken, not the first: the allowlist and
    the container's own resolver have to agree, and a name behind a CDN answers
    differently on every lookup.

    A failure here is NOT fatal. A wildcard entry has no addresses by
    definition, and a registry that is briefly unresolvable should fail the one
    install that needs it rather than the whole run -- the firewall's default is
    deny, so a name with no addresses is simply a name with no route.
    """
    out: dict = {}
    for host in hosts:
        if host.startswith("*."):
            out[host] = []
            continue
        try:
            out[host] = sorted({r[4][0] for r in socket.getaddrinfo(
                host, 443, socket.AF_INET, socket.SOCK_STREAM)})
        except OSError:
            out[host] = []
    return out


def _firewall(egress_port: int, allow_ips: tuple, *, allow_dns: bool = True) -> None:
    """Runs INSIDE the episode's user+net namespace. Raises on the first refusal.

    Order matters twice. `-o lo -j RETURN` comes first in nat OUTPUT so that a
    mirrored site aliased onto the container's loopback is never redirected --
    an injected honeypot must keep answering locally. And the DNAT rules come
    before the filter chain sees anything, which is why the filter table needs
    no per-address rule: an allowed address has already become 127.0.0.1 by the
    time it gets there, and everything still holding its original destination is
    something the policy rejects.
    """
    _raise_ambient_caps((_CAP_NET_ADMIN, _CAP_NET_BIND_SERVICE))

    # Locally generated traffic may be redirected to loopback. Namespaced, so
    # this says nothing about the host.
    subprocess.run(["sysctl", "-wq", "net.ipv4.conf.all.route_localnet=1"],
                   capture_output=True)
    subprocess.run(["ip", "link", "set", "lo", "up"], capture_output=True)

    rules = [["-t", "nat", "-A", "OUTPUT", "-o", "lo", "-j", "RETURN"]]
    for ip in allow_ips:
        for port in WEB_PORTS:
            rules.append(["-t", "nat", "-A", "OUTPUT", "-p", "tcp",
                          "-d", ip, "--dport", str(port), "-j", "DNAT",
                          "--to-destination", f"127.0.0.1:{egress_port}"])
    rules += [
        ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-d", "127.0.0.0/8", "-j", "ACCEPT"],
        # slirp4netns' own resolver. Names answer; the addresses they answer
        # with have nowhere to go unless they are on the list.
        ["-A", "OUTPUT", "-p", "udp", "-d", SLIRP_DNS, "--dport", "53",
         "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-p", "tcp", "-d", SLIRP_DNS, "--dport", "53",
         "-j", "ACCEPT"],
        # REJECT rather than DROP: a refusal costs the episode a millisecond and
        # a black hole costs it the connect timeout, several times over, out of
        # the same clock the task is measured on.
        ["-A", "OUTPUT", "-j", "REJECT", "--reject-with",
         "icmp-port-unreachable"],
        ["-P", "OUTPUT", "DROP"],
    ]
    if not allow_dns:
        rules = [rule for rule in rules if not ("--dport" in rule and
                 rule[rule.index("--dport") + 1] == "53")]
    for rule in rules:
        res = subprocess.run(["iptables", *rule], capture_output=True, text=True)
        if res.returncode != 0:
            raise EgressError(f"iptables {' '.join(rule)}: {res.stderr.strip()}")

    # IPv6 has no route here -- slirp4netns is not started with --enable-ipv6 --
    # but a policy that depends on an argument we did not pass is not a policy.
    for rule in (["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"],
                 ["-A", "OUTPUT", "-j", "REJECT"],
                 ["-P", "OUTPUT", "DROP"]):
        res = subprocess.run(["ip6tables", *rule], capture_output=True, text=True)
        if res.returncode != 0 and "does not exist" not in res.stderr:
            raise EgressError(f"ip6tables {' '.join(rule)}: {res.stderr.strip()}")


def _pin_hosts(resolved: dict) -> None:
    """Runs INSIDE the container's mount namespace. Resolver, then /etc/hosts.

    The firewall allows ADDRESSES and the container resolves NAMES, so the two
    have to be looking at the same list; a CDN that answers with a different
    address on the container's own lookup would otherwise be an allowlisted host
    that will not connect. Pinning every address the name answered with keeps
    the usual failover.

    A name a setup hook has already pinned is left exactly as it is: an injected
    mirror owns the names it stands in for, and this must never point one of
    them back at the real site.
    """
    Path("/etc/resolv.conf").write_text(f"nameserver {SLIRP_DNS}\n")

    path = Path("/etc/hosts")
    existing = path.read_text() if path.exists() else ""
    taken = {ln.split()[1] for ln in existing.splitlines()
             if len(ln.split()) > 1 and not ln.lstrip().startswith("#")}
    lines = [f"{ip}\t{host}" for host, ips in sorted(resolved.items())
             for ip in ips if host not in taken]
    if lines:
        path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n")


class EgressGuard:
    """One episode's lockdown: the relay process, and the rules around it.

    Built by `core/trial.py` after the task's setup hook has run and the network
    namespace has a route, and before the container is unblocked -- the episode
    is still waiting on stdin for its task at that point, so there is no window
    in which anything inside can reach anything.
    """

    def __init__(self, *, net_pid: int, container_pid: int, hosts,
                 loopback_ports=(), log_dir: Path | None = None,
                 python_exe: str | None = None, allow_dns: bool = True):
        self.net_pid = net_pid
        self.container_pid = container_pid
        self.hosts = tuple(dict.fromkeys(h.lower() for h in hosts))
        self.loopback_ports = tuple(int(p) for p in loopback_ports)
        self.log_dir = Path(log_dir) if log_dir else None
        self.python_exe = python_exe or sys.executable
        self.allow_dns = allow_dns
        self.egress_port = 0
        self.resolved: dict = {}
        self._proc: subprocess.Popen | None = None
        self._ready: Path | None = None
        self._log_fh = None

    @property
    def decisions_log(self) -> Path | None:
        return self.log_dir / "egress.jsonl" if self.log_dir else None

    def start(self) -> None:
        self.resolved = resolve_hosts(self.hosts)
        allow_ips = tuple(dict.fromkeys(
            ip for ips in self.resolved.values() for ip in ips))
        self.egress_port = _free_port()

        self._start_relay()
        try:
            _in_child(self.net_pid, ("user", "net"),
                      lambda: _firewall(self.egress_port, allow_ips, allow_dns=self.allow_dns))
            resolved = self.resolved
            _in_child(self.container_pid, ("user", "mnt"),
                      lambda: _pin_hosts(resolved))
        except BaseException:
            # A half-installed lockdown is an open one. Take the relay with it
            # so the caller's failure path has nothing left running.
            self.close()
            raise

    def _start_relay(self) -> None:
        ready = Path(tempfile.mkdtemp(prefix="rh_egress_")) / "ready"
        self._ready = ready
        cfg = {"netns_pid": self.net_pid, "egress_port": self.egress_port,
               "loopback": list(self.loopback_ports), "allow": list(self.hosts),
               "log": str(self.decisions_log) if self.decisions_log else None,
               "ready": str(ready)}
        err = subprocess.DEVNULL
        if self.log_dir:
            self._log_fh = open(self.log_dir / "egress_relay.log", "ab")
            err = self._log_fh
        self._proc = subprocess.Popen(
            [self.python_exe, str(RELAY)], stdin=subprocess.PIPE,
            stdout=err, stderr=err)
        self._proc.stdin.write(json.dumps(cfg).encode())
        self._proc.stdin.close()

        deadline = time.time() + 90
        while time.time() < deadline:
            if ready.exists():
                return
            if self._proc.poll() is not None:
                raise EgressError(
                    f"egress relay exited {self._proc.returncode} during "
                    f"startup (see {self.log_dir})")
            time.sleep(0.05)
        raise EgressError("egress relay never reported ready")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if self._ready is not None:
            try:
                self._ready.unlink()
                self._ready.parent.rmdir()
            except OSError:
                pass
            self._ready = None

    def attempts(self) -> list:
        """What the episode tried to reach, allowed and refused, in order."""
        path = self.decisions_log
        if not path or not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out


def _free_port() -> int:
    """A port nothing on the HOST holds. The listener is bound in the episode's
    namespace, where the number cannot collide with anything, but taking it from
    the host's ephemeral range keeps two concurrent episodes from picking the
    same one for their own logs to disagree about."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
