"""The egress policy: what a task declares, and what the relay reads off the wire.

The enforcement itself -- the namespace, the firewall, the fd passing -- needs a
real container and lives in `tests/manual_egress_container.py`. What is here is
everything that can be decided without one, and the two that matter most are the
DEFAULT (a task that says nothing keeps the open network, and only a task that
asks is locked down) and the NAME check (an allowlisted address carrying somebody
else's name gets nothing).
"""
from __future__ import annotations

import socket
import ssl
import threading
from pathlib import Path

import pytest
import yaml

from core import trial
from core.sandbox import egress, egress_relay

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------
def test_a_task_that_says_nothing_keeps_the_open_network() -> None:
    """The retreat from default-deny, pinned. This module first shipped with
    deny as the default and it was reverted, because flipping it changes every
    environment already measured on the open network without anyone saying so.
    A task that declares nothing is left exactly where it was."""
    policy = egress.parse_network("t.yaml", None)
    assert policy.egress == "open"
    assert policy.hosts == ()
    assert policy.enforced is False


def test_deny_is_the_default_inside_the_block_too() -> None:
    """Writing the block at all means restriction: the open network is what you
    get by writing nothing, so `network: {}` is nobody's way of asking for it."""
    assert egress.parse_network("t.yaml", {}).denies


def test_open_can_be_written_out() -> None:
    """For a task that wants the record to show it chose the open network rather
    than never considered the question."""
    policy = egress.parse_network("t.yaml", {"egress": "open"})
    assert policy.egress == "open"
    assert policy.enforced is False


def test_allow_names_its_hosts() -> None:
    policy = egress.parse_network("t.yaml", {"egress": "allow",
                                             "hosts": ["Example.COM"]})
    assert policy.egress == "allow"
    assert policy.hosts == ("example.com",)


@pytest.mark.parametrize("raw, says", [
    ("deny", "must be a map"),
    ({"egress": "maybe"}, "must be `deny`, `allow`"),
    ({"egress": "deny", "hosts": ["a.com"]}, "takes no `hosts:`"),
    ({"egress": "open", "hosts": ["a.com"]}, "takes no `hosts:`"),
    ({"egress": "allow"}, "needs `hosts:`"),
    ({"egress": "allow", "hosts": ["*"]}, "is not a host"),
    ({"egress": "allow", "hosts": "a.com"}, "must be a list"),
    ({"egres": "deny"}, "unknown key"),
])
def test_a_malformed_block_is_a_refusal_and_not_a_default(raw, says) -> None:
    """Anything ambiguous stops the run. A policy nobody meant to write is the
    failure mode this module exists to remove, not one to reintroduce."""
    with pytest.raises(SystemExit) as e:
        egress.parse_network("t.yaml", raw)
    assert says in str(e.value)


def test_every_shipped_block_parses_and_explicit_tasks_lock_down() -> None:
    """Keep the explicit restricted-egress task list under review."""
    tasks = sorted(p.parent.name for p in TASKS.glob("*/task.yaml"))
    assert tasks, "no tasks found"
    locked = []
    for name in tasks:
        cfg = yaml.safe_load((TASKS / name / "task.yaml").read_text())
        if egress.parse_network(f"{name}/task.yaml", cfg.get("network")).enforced:
            locked.append(name)
    assert locked == ["chess", "creative_writings", "geoguessr", "go", "knowledge_work", "openmath_agent", "protein_design"]


def test_load_task_carries_the_policy_through() -> None:
    assert trial.load_task("protein_design").network.denies
    for name in sorted(p.parent.name for p in TASKS.glob("*/task.yaml")):
        assert trial.load_task(name).network.egress in ("deny", "allow", "open")


def test_a_namespace_goes_to_the_task_that_needs_one() -> None:
    """Two things need one: a setup hook, which binds ports, and an enforced
    policy, which is installed from the host INTO the namespace. A task with
    neither shares the host's, which is where all of them used to be."""
    for name in sorted(p.parent.name for p in TASKS.glob("*/task.yaml")):
        task = trial.load_task(name)
        want = task.setup_hook.exists() or task.network.enforced
        assert task.private_net is want, name
    assert trial.load_task("protein_design").private_net is True


# --------------------------------------------------------------------------
# what is reachable whatever the task says
# --------------------------------------------------------------------------
def test_the_model_api_is_always_reachable() -> None:
    hosts = egress.hosts_for(egress.NetworkPolicy("deny"),
                             "https://litellm.safe.ai/v1")
    assert "litellm.safe.ai" in hosts


def test_an_unrouted_run_allows_both_vendors_defaults() -> None:
    hosts = egress.hosts_for(egress.NetworkPolicy("deny"), "")
    assert "api.anthropic.com" in hosts and "api.openai.com" in hosts


def test_the_install_registries_are_reachable_under_deny() -> None:
    """Every episode installs its own agent CLI: the image bakes none."""
    hosts = egress.hosts_for(egress.NetworkPolicy("deny"), "https://x.example")
    for host in egress.INSTALL_HOSTS:
        assert host in hosts


def test_a_folding_models_weights_are_not_reachable_under_deny() -> None:
    """protein_design's premise, and the exact shape of what deny buys it. PyPI is
    infrastructure, so `pip install boltz` still succeeds; the checkpoints are
    not, so the fold does not happen and the agent cannot check its own design.
    tasks/protein_design/task.yaml states this and the hole it leaves."""
    hosts = egress.hosts_for(egress.NetworkPolicy("deny"), "https://x.example")
    assert "pypi.org" in hosts
    for host in ("huggingface.co", "model-gateway.boltz.bio",
                 "api.colabfold.com"):
        assert host not in hosts


def test_a_tasks_hosts_are_added_and_nothing_is_duplicated() -> None:
    policy = egress.NetworkPolicy("allow", ("pypi.org", "extra.example"))
    hosts = egress.hosts_for(policy, "https://x.example")
    assert "extra.example" in hosts
    assert len(hosts) == len(set(hosts))


# --------------------------------------------------------------------------
# the name check
# --------------------------------------------------------------------------
@pytest.mark.parametrize("host, allow, ok", [
    ("example.com", ("example.com",), True),
    ("Example.COM", ("example.com",), True),
    ("example.com.", ("example.com",), True),
    ("evil.com", ("example.com",), False),
    ("notexample.com", ("example.com",), False),
    ("example.com.evil.com", ("example.com",), False),
    ("a.example.com", ("*.example.com",), True),
    ("deep.a.example.com", ("*.example.com",), True),
    ("example.com", ("*.example.com",), False),
    ("", ("example.com",), False),
    ("anything.com", ("*",), False),
])
def test_host_allowed(host, allow, ok) -> None:
    assert egress_relay.host_allowed(host, allow) is ok


def _client_hello(server_name: str) -> bytes:
    """Real OpenSSL ClientHello without a socket or timing-dependent thread."""
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client = ctx.wrap_bio(incoming, outgoing, server_hostname=server_name)
    try:
        client.do_handshake()
    except ssl.SSLWantReadError:
        pass
    return outgoing.read()


def test_the_sni_is_read_off_a_real_client_hello() -> None:
    assert egress_relay.sni_of(_client_hello("registry.npmjs.org")) \
        == "registry.npmjs.org"


def test_a_long_name_survives_the_parse() -> None:
    name = "a-rather-long-subdomain.of.some.longer.example.org"
    assert egress_relay.sni_of(_client_hello(name)) == name


def test_something_that_is_not_a_client_hello_yields_no_name() -> None:
    assert egress_relay.sni_of(b"GET / HTTP/1.1\r\nHost: x.com\r\n\r\n") is None
    assert egress_relay.sni_of(b"") is None
    assert egress_relay.sni_of(b"\x16\x03\x01") is None


def test_a_truncated_client_hello_yields_no_name_rather_than_raising() -> None:
    hello = _client_hello("example.com")
    for cut in (46, 60, len(hello) // 2, len(hello) - 1):
        egress_relay.sni_of(hello[:cut])       # must not raise


@pytest.mark.parametrize("raw, want", [
    (b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n", "example.com"),
    (b"GET / HTTP/1.1\r\nhost: example.com:8080\r\n\r\n", "example.com"),
    (b"POST /x HTTP/1.1\r\nA: b\r\nHost: a.example.com\r\n\r\n", "a.example.com"),
    (b"GET / HTTP/1.1\r\n\r\n", None),
    (b"garbage", None),
])
def test_the_http_host_header_is_read(raw, want) -> None:
    assert egress_relay.http_host_of(raw) == want


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------
def test_a_wildcard_entry_has_no_addresses_to_allow() -> None:
    """It is a name check only: there is nothing to put in the firewall, so a
    wildcard host is reachable only if one of its addresses arrived another way."""
    assert egress.resolve_hosts(("*.example.com",)) == {"*.example.com": []}


def test_a_name_that_will_not_resolve_is_not_fatal() -> None:
    """Default-deny means a name with no addresses is a name with no route, and
    failing the whole run over one unresolvable registry helps nobody."""
    assert egress.resolve_hosts(("no-such-host.invalid",)) \
        == {"no-such-host.invalid": []}


@pytest.mark.parametrize("adapter,expected", [
    ("muse_code", {"dev.meta.ai", "api.meta.ai", "lookaside.facebook.com"}),
    ("grok_build", {"x.ai", "storage.googleapis.com"}),
])
def test_adapter_install_hosts_allowed_with_gateway(adapter, expected):
    from core.agents.muse_code import MuseCodeAgent
    from core.agents.grok_build import GrokBuildAgent
    cls = {"muse_code": MuseCodeAgent, "grok_build": GrokBuildAgent}[adapter]
    hosts = egress.hosts_for(egress.NetworkPolicy("deny"),
                             "https://gateway.example", cls.INSTALL_HOSTS)
    assert expected.issubset(set(hosts))
    assert not expected.intersection(egress.hosts_for(egress.NetworkPolicy("deny"),
                                                    "https://gateway.example"))


def test_gemini_gateway_is_allowed_by_resolved_route(monkeypatch):
    from core.agents.gemini_cli import GeminiCLIAgent
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    agent = object.__new__(GeminiCLIAgent)
    agent.api_base_url = ""
    agent.route = "gateway"
    assert "gateway.example" in egress.hosts_for(
        egress.NetworkPolicy("deny"), agent.resolved_base_url())
    agent.route = "native"
    hosts = egress.hosts_for(egress.NetworkPolicy("deny"), agent.resolved_base_url())
    assert "generativelanguage.googleapis.com" in hosts
    assert "gateway.example" not in hosts


def test_gemini_runtime_and_reporting_agree_with_allowlist(monkeypatch):
    import json
    from core.agents.gemini_cli import GeminiCLIAgent
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    agent = GeminiCLIAgent(model="gemini/gemini-3.8-flash")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="test", files=[],
        stage_src="", timeout_s=60, skip_dirs=[], env={}))
    runtime_url = payload["task"]["gemini"]["base_url"]
    assert runtime_url == agent.resolved_base_url() == "https://gateway.example/v1"
    assert agent.resolved_routing()["api_base_url_source"] == "env:OPENAI_BASE_URL"
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert agent.resolved_base_url() == ""
    assert any("OPENAI_BASE_URL" in p for p in agent.setup())


@pytest.mark.parametrize("route", ["gateway", "native"])
def test_gemini_explicit_base_url_reaches_runtime_and_policy(monkeypatch, route):
    import json
    from core.agents.gemini_cli import GeminiCLIAgent
    monkeypatch.setenv("GEMINI_CLI_ROUTE", route)
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gemini-gateway.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    agent = GeminiCLIAgent(model="gemini/gemini-3.8-flash",
                           api_base_url="https://configured.example/v1")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="test", files=[],
        stage_src="", timeout_s=60, skip_dirs=[], env={}))
    assert payload["task"]["gemini"]["base_url"] == agent.resolved_base_url()
    assert agent.base_url_source() == "entry"
    assert "configured.example" in egress.hosts_for(
        egress.NetworkPolicy("deny"), agent.resolved_base_url())
    agent.api_base_url = ""
    if route == "gateway":
        assert agent.resolved_base_url() == "https://gemini-gateway.example/v1"
        assert agent.base_url_source() == "env:GEMINI_BASE_URL"
    else:
        assert agent.resolved_base_url() == "https://generativelanguage.googleapis.com"


@pytest.mark.parametrize("url, ports", [
    ("http://127.0.0.1:43211/v1", (43211,)),
    ("http://localhost:8000", (8000,)),
    ("https://localhost/v1", (443,)),
    ("https://api.openai.com/v1", ()),
    ("http://127.0.0.1.example.org:8000", ()),
    ("", ()),
])
def test_only_configured_local_api_port_is_relayed(url, ports):
    assert egress.api_loopback_ports(url) == ports


def test_relay_preserves_quiet_connection_until_peer_closes(monkeypatch):
    from unittest.mock import Mock
    near, far = Mock(), Mock()
    near.recv.side_effect = [b"request", b""]
    far.recv.return_value = b"delayed response"
    poll = Mock(side_effect=[([], [], []), ([near], [], []),
                            ([], [], []), ([far], [], []), ([near], [], [])])
    monkeypatch.setattr(egress_relay.select, "select", poll)
    # No constructor/real sockets needed: exercise the byte relay itself.
    egress_relay._Child.pump(None, near, far)
    far.sendall.assert_called_once_with(b"request")
    near.sendall.assert_called_once_with(b"delayed response")
    assert poll.call_count == 5


def test_relay_delivers_bytes_after_idle_polls_on_real_sockets(monkeypatch):
    import socket
    import threading
    import time
    client, near = socket.socketpair()
    far, upstream = socket.socketpair()
    original_select = egress_relay.select.select
    monkeypatch.setattr(egress_relay.select, "select",
                        lambda r, w, x, timeout: original_select(r, w, x, 0.01))
    thread = threading.Thread(target=egress_relay._Child.pump,
                              args=(None, near, far), daemon=True)
    try:
        client.settimeout(1)
        upstream.settimeout(1)
        thread.start()
        client.sendall(b"request")
        assert upstream.recv(7) == b"request"
        time.sleep(0.05)
        upstream.sendall(b"response")
        assert client.recv(8) == b"response"
        upstream.shutdown(socket.SHUT_WR)
        thread.join(1)
        assert not thread.is_alive()
    finally:
        for sock in (client, near, far, upstream):
            sock.close()
        thread.join(1)
