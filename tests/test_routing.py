"""Per-model API routing: `api_key_env:`, `api_base_url:`, `extra_body:`.

What these tests pin:

  * the resolver (core/routing.py): `${VAR}` interpolation at load, an unset
    VAR is fatal, a named-but-unset credential variable is fatal and names the
    entry, an absent `api_key_env` falls back to the adapter default ONCE and
    loudly;
  * the adapter side (core/agents/base.py): the key is exported inside the
    container under the CLI's own names, `api_base_url` reaches the CLI where
    that CLI wants it (codex: config.toml `openai_base_url`; claude-sdk:
    ANTHROPIC_BASE_URL), the PASSTHROUGH fallback still applies when the entry
    routes nothing, and `extra_body` ships the forwarder -- or is refused by an
    adapter that has not implemented it;
  * the forwarder (core/agents/forwarder.py): merges extra_body into JSON
    request bodies only, replaces the credential, relays a streamed response
    byte-for-byte, adds no sampling parameter;
  * the judge path (core/judge.py): the same three keys reach the judge
    client as `api_key_env`/`api_base_url` kwargs and `extra_body`;
  * the shipped configs/models.yaml: every claude-*/gpt-* entry states its
    route.
"""
from __future__ import annotations

import http.client
import http.server
import json
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run as run_py                                          # noqa: E402
from core import judge as judging                             # noqa: E402
from core import routing                                      # noqa: E402
from core import sandbox                                      # noqa: E402
from core.agents import make_agent                            # noqa: E402
from core.agents import forwarder as fwd                      # noqa: E402
from core.agents import codex_runner                          # noqa: E402

MODELS = ROOT / "configs" / "models.yaml"
POLICY = {"web_search": False, "web_fetch": False}


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _blob(agent, **kw) -> dict:
    return json.loads(agent.blob(
        row={"id": "r"}, prompt="p", files=[], skip_dirs=(), grader_src="",
        stage_src="", timeout_s=1, env=kw.pop("env", {}), install=None))


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------
def test_dollar_brace_is_a_plain_yaml_string() -> None:
    doc = yaml.safe_load("m:\n  api_base_url: ${OPENAI_BASE_URL}\n")
    assert doc["m"]["api_base_url"] == "${OPENAI_BASE_URL}"


def test_api_base_url_is_interpolated_at_load(monkeypatch) -> None:
    monkeypatch.setenv("RH_TEST_GATEWAY", "https://gw.example.invalid/v1")
    out = routing.resolve("m", {"api_key_env": "K",
                                "api_base_url": "${RH_TEST_GATEWAY}"},
                          where="t")
    assert out == {"api_key_env": "K",
                   "api_base_url": "https://gw.example.invalid/v1",
                   "extra_body": {}}


def test_an_unset_variable_in_api_base_url_is_fatal(monkeypatch) -> None:
    monkeypatch.delenv("RH_TEST_UNSET", raising=False)
    with pytest.raises(SystemExit, match="RH_TEST_UNSET"):
        routing.resolve("m", {"api_base_url": "${RH_TEST_UNSET}"}, where="t")


def test_a_literal_url_passes_through_and_a_bad_extra_body_is_refused() -> None:
    out = routing.resolve("m", {"api_base_url": "http://h/v1",
                                "extra_body": {"provider": {"order": ["x"]}}},
                          where="t")
    assert out["api_base_url"] == "http://h/v1"
    assert out["extra_body"] == {"provider": {"order": ["x"]}}
    with pytest.raises(SystemExit, match="extra_body"):
        routing.resolve("m", {"extra_body": ["not", "a", "map"]}, where="t")
    with pytest.raises(SystemExit, match="scheme"):
        routing.resolve("m", {"api_base_url": "gw.example.invalid"}, where="t")


def test_a_named_but_unset_credential_is_fatal_and_names_the_entry(monkeypatch) -> None:
    monkeypatch.delenv("RH_TEST_KEY", raising=False)
    with pytest.raises(SystemExit, match="model 'm'.*RH_TEST_KEY"):
        routing.require_key("RH_TEST_KEY", name="m", consumer="test")
    monkeypatch.setenv("RH_TEST_KEY", "sk-test")
    assert routing.require_key("RH_TEST_KEY", name="m", consumer="test") == "sk-test"


def test_an_absent_api_key_env_defaults_once_and_loudly(capsys) -> None:
    routing._WARNED.clear()
    assert routing.default_key_env("m", None, "c", "X_KEY") == ("X_KEY", "default")
    assert routing.default_key_env("m", None, "c", "X_KEY") == ("X_KEY", "default")
    assert routing.default_key_env("m", "Y", "c", "X_KEY") == ("Y", "entry")
    err = capsys.readouterr().err
    assert err.count("is assumed") == 1 and "X_KEY" in err   # once, not twice


def test_recorded_urls_carry_no_userinfo_or_query() -> None:
    assert routing.sanitise_url("https://u:p@h.example:8443/v1?k=v#f") == \
        "https://h.example:8443/v1"
    assert routing.url_host("https://u:p@h.example:8443/v1") == "h.example"
    assert routing.sanitise_url("") is None


# --------------------------------------------------------------------------
# the shipped file
# --------------------------------------------------------------------------
def test_every_shipped_claude_and_gpt_entry_states_its_route() -> None:
    doc = yaml.safe_load(MODELS.read_text())
    for name, entry in doc.items():
        if name == "default":
            continue
        if name.startswith("claude-"):
            assert entry.get("api_key_env") == "ANTHROPIC_API_KEY", name
            assert entry.get("api_base_url") == "${ANTHROPIC_BASE_URL}", name
        elif name.startswith("gpt-"):
            assert entry.get("api_key_env") == "OPENAI_API_KEY", name
            assert entry.get("api_base_url") == "${OPENAI_BASE_URL}", name


def test_load_config_resolves_the_shipped_gpt_entry(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example.invalid")
    cfg = run_py.load_config(MODELS, "gpt-5.6-sol")
    assert cfg["api_key_env"] == "OPENAI_API_KEY"
    assert cfg["api_base_url"] == "https://gw.example.invalid"
    assert cfg["extra_body"] == {}
    monkeypatch.delenv("OPENAI_BASE_URL")
    with pytest.raises(SystemExit, match="OPENAI_BASE_URL"):
        run_py.load_config(MODELS, "gpt-5.6-sol")


def test_the_judge_resolves_the_same_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example.invalid")
    cfg = judging.load_judge_config(MODELS, "gpt-5.6-sol")
    assert cfg["api_key_env"] == "OPENAI_API_KEY"
    assert cfg["api_base_url"] == "https://gw.example.invalid"
    rec = judging.judge_routing_record("gpt-5.6-sol", cfg)
    assert rec["api_base_host"] == "gw.example.invalid"
    assert rec["api_key_env_source"] == "entry"


def test_the_judge_client_receives_the_route_as_kwargs(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, {
        "models": {"j": {"model": "openai/x", "api_key_env": "RH_TEST_KEY",
                         "api_base_url": "http://h/v1",
                         "extra_body": {"provider": {"order": ["p"]}},
                         "generation_config": {"reasoning_effort": "low"}}}})
    monkeypatch.setenv("RH_TEST_KEY", "sk-test")
    seen: dict = {}

    def fake(model, generation_config, **kwargs):
        seen.update(model=model, generation_config=generation_config, **kwargs)
        return object()

    monkeypatch.setattr(judging, "get_llm_agent_class", fake)
    judging.make_judge_agent("j", path)
    assert seen == {"model": "openai/x", "api_key_env": "RH_TEST_KEY",
                    "api_base_url": "http://h/v1",
                    "generation_config": {"reasoning_effort": "low",
                                          "extra_body": {"provider": {"order": ["p"]}}}}
    monkeypatch.delenv("RH_TEST_KEY")
    with pytest.raises(SystemExit, match="RH_TEST_KEY"):
        judging.make_judge_agent("j", path)


def test_the_anthropic_judge_client_takes_the_same_kwargs(monkeypatch) -> None:
    from core import llm_agents

    monkeypatch.setenv("RH_TEST_KEY", "sk-test")
    agent = llm_agents.AnthropicAgent("claude-x", api_key_env="RH_TEST_KEY",
                                      api_base_url="https://gw.example.invalid")
    assert str(agent.client.base_url).startswith("https://gw.example.invalid")


# --------------------------------------------------------------------------
# the adapters
# --------------------------------------------------------------------------
def test_the_key_is_exported_under_the_clis_own_name() -> None:
    codex = make_agent("codex", model="openai/x", api_key_env="LITELLM_KEY")
    assert codex.api_key_env == "LITELLM_KEY"           # read on the host
    assert codex.container_key_envs() == ("OPENAI_API_KEY",)   # read in the container
    codex.apply_tool_policy(POLICY)
    assert _blob(codex)["task"]["codex"]["api_key_env"] == "OPENAI_API_KEY"

    claude = make_agent("claude-sdk", model="claude-x", api_key_env="LITELLM_KEY")
    assert claude.container_key_envs() == ("ANTHROPIC_API_KEY",)
    assert claude.api_key_env_source == "entry"
    assert make_agent("claude-sdk", model="claude-x").api_key_env_source == "default"


def test_the_sandbox_exports_every_container_name() -> None:
    argv = sandbox.container_argv("img", "b", private_net=False,
                                  key_env=("A_KEY", "B_KEY"))
    assert argv[argv.index("-e") + 1:] and "A_KEY" in argv and "B_KEY" in argv
    env = sandbox.spawn_env("sk-test", base={}, key_env=("A_KEY", "B_KEY"))
    assert env["A_KEY"] == env["B_KEY"] == "sk-test"
    # The plain-string form every older caller used still means one name.
    env = sandbox.spawn_env("sk-test", base={}, key_env="ONE")
    assert env["ONE"] == "sk-test"


def test_codex_writes_api_base_url_to_config_toml_as_the_gateway(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = make_agent("codex", model="openai/gpt-5.6-sol",
                       api_base_url="https://gw.example.invalid")
    agent.apply_tool_policy(POLICY)
    codex = _blob(agent)["task"]["codex"]
    assert codex["base_url"] == "https://gw.example.invalid"
    assert "provider" not in codex                      # the gateway shape, not the override
    assert codex["model_slug"] == "gpt-5.6-sol"          # prefix stripped, as before
    assert "--ignore-user-config" not in codex["exec_flags"]
    toml = codex_runner.config_toml(codex["base_url"])
    assert 'openai_base_url = "https://gw.example.invalid"' in toml
    assert agent.resolved_routing()["api_base_url_source"] == "entry"


def test_the_passthrough_fallback_still_applies_when_the_entry_routes_nothing(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.invalid")
    agent = make_agent("codex", model="openai/x")
    agent.apply_tool_policy(POLICY)
    assert _blob(agent)["task"]["codex"]["base_url"] == "https://env.example.invalid"
    rec = agent.resolved_routing()
    assert rec["api_base_url_source"] == "env:OPENAI_BASE_URL"
    assert rec["api_base_host"] == "env.example.invalid"
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert make_agent("codex", model="openai/x").resolved_base_url() == ""


def test_the_entry_wins_over_the_passthrough_for_claude(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example.invalid")
    agent = make_agent("claude-sdk", model="claude-x",
                       api_base_url="https://gw.example.invalid")
    env = agent.routed_env({"ANTHROPIC_BASE_URL": "https://env.example.invalid",
                            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1"})
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example.invalid"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1"
    # ...and an entry that routes nothing leaves the copy alone.
    plain = make_agent("claude-sdk", model="claude-x")
    assert plain.routed_env({"ANTHROPIC_BASE_URL": "x"}) == {"ANTHROPIC_BASE_URL": "x"}


def test_extra_body_ships_the_forwarder_and_only_then() -> None:
    for name in ("codex", "claude-sdk"):
        plain = make_agent(name, model="openai/x")
        plain.apply_tool_policy(POLICY)
        payload = _blob(plain)
        assert "forwarder" not in payload["modules"]
        assert payload["task"]["routing"]["extra_body"] == {}
        assert plain.resolved_routing()["forwarder"] is False

        routed = make_agent(name, model="openai/x", api_base_url="http://h/v1",
                            extra_body={"provider": {"order": ["p"]}})
        routed.apply_tool_policy(POLICY)
        payload = _blob(routed)
        assert "class Forwarder" in payload["modules"]["forwarder"]
        assert payload["task"]["routing"] == {
            "cli_key_env": list(routed.container_key_envs()),
            "api_base_url": "http://h/v1",
            "extra_body": {"provider": {"order": ["p"]}}}
        assert routed.setup() == []
        assert routed.resolved_routing()["forwarder"] is True


def test_an_adapter_without_the_forwarder_refuses_extra_body() -> None:
    from core.agents.claude_sdk import ClaudeSDKAgent

    class NoForwarder(ClaudeSDKAgent):
        SUPPORTS_FORWARDER = False

        @staticmethod
        def name() -> str:
            return "claude-sdk"

    agent = NoForwarder(model="x", extra_body={"a": 1})
    agent.apply_tool_policy(POLICY)
    assert any("extra_body" in p for p in agent.setup())


def test_codex_refuses_both_base_url_shapes_at_once() -> None:
    agent = make_agent("codex", model="openai/x", base_url="https://or.example/v1",
                       api_base_url="https://gw.example/v1")
    agent.apply_tool_policy(POLICY)
    assert any("base_url" in p and "api_base_url" in p for p in agent.setup())


# --------------------------------------------------------------------------
# the forwarder
# --------------------------------------------------------------------------
class _Stub(http.server.BaseHTTPRequestHandler):
    """The upstream: records what arrived, answers JSON or an SSE stream."""

    protocol_version = "HTTP/1.1"
    seen: list = []

    def log_message(self, *a) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        type(self).seen.append({"path": self.path, "body": body,
                                "headers": dict(self.headers)})
        if self.path.endswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for piece in (b"data: {\"a\":1}\n\n", b"data: {\"b\":2}\n\n",
                          b"data: [DONE]\n\n"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        try:
            echo = json.loads(body) if body else None
        except ValueError:
            echo = body.decode("utf-8", "replace")
        out = json.dumps({"echo": echo}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def upstream():
    _Stub.seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def _post(url: str, path: str, body: bytes | None, headers: dict):
    parts = url.split("/", 3)
    conn = http.client.HTTPConnection(parts[2], timeout=10)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def test_forwarder_merges_extra_body_and_replaces_the_credential(upstream) -> None:
    f = fwd.Forwarder(base_url=upstream, api_key="sk-real",
                      extra_body={"provider": {"order": ["p"]}, "x": 1},
                      log=lambda m: None).start()
    try:
        assert f.url.endswith("/v1")
        body = json.dumps({"model": "m", "messages": [], "provider": {"allow_fallbacks": False}}).encode()
        resp, data = _post(f.url, "/v1/chat/completions", body,
                           {"Content-Type": "application/json",
                            "Authorization": "Bearer sk-cli", "x-api-key": "sk-cli",
                            "Content-Length": str(len(body))})
        assert resp.status == 200
        sent = json.loads(_Stub.seen[-1]["body"])
        assert sent["provider"] == {"allow_fallbacks": False, "order": ["p"]}
        assert sent["x"] == 1 and sent["model"] == "m"
        assert not {"temperature", "top_p", "seed", "max_tokens"} & set(sent)
        assert _Stub.seen[-1]["headers"]["Authorization"] == "Bearer sk-real"
        assert _Stub.seen[-1]["headers"]["x-api-key"] == "sk-real"
        assert _Stub.seen[-1]["path"] == "/v1/chat/completions"
        assert json.loads(data)["echo"] == sent
        # The per-call summary is appended by the handler thread after the last
        # byte is flushed, so give it a moment.
        for _ in range(100):
            if f.calls:
                break
            time.sleep(0.01)
        assert f.calls[-1]["status"] == 200 and "body" not in f.calls[-1]
    finally:
        f.close()


def test_forwarder_leaves_a_non_json_body_alone_and_streams_sse_byte_for_byte(upstream) -> None:
    f = fwd.Forwarder(base_url=upstream, api_key="sk-real",
                      extra_body={"x": 1}, log=lambda m: None).start()
    try:
        raw = b"not json at all"
        resp, _ = _post(f.url, "/v1/other", raw,
                        {"Content-Type": "text/plain", "Content-Length": str(len(raw))})
        assert resp.status == 200 and _Stub.seen[-1]["body"] == raw

        body = b"{}"
        resp, data = _post(f.url, "/v1/stream", body,
                           {"Content-Type": "application/json",
                            "Content-Length": str(len(body))})
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        assert data == b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n"
    finally:
        f.close()


def test_forwarder_answers_502_when_the_upstream_is_unreachable() -> None:
    f = fwd.Forwarder(base_url="http://127.0.0.1:1/v1", api_key="k",
                      log=lambda m: None).start()
    try:
        body = b"{}"
        resp, data = _post(f.url, "/v1/chat/completions", body,
                           {"Content-Type": "application/json",
                            "Content-Length": str(len(body))})
        assert resp.status == 502 and json.loads(data)["error"]["type"] == "forwarder_error"
    finally:
        f.close()


def test_deep_merge_is_recursive_and_extra_wins() -> None:
    assert fwd.deep_merge({"a": {"b": 1, "c": 2}, "l": [1]},
                          {"a": {"c": 3}, "l": [2], "n": 4}) == \
        {"a": {"b": 1, "c": 3}, "l": [2], "n": 4}


@pytest.mark.parametrize("base,expected", [
    ("https://api.anthropic.com", "claude-opus-5"),
    ("https://litellm.safe.ai", "anthropic/claude-opus-5"),
])
def test_claude_model_prefix_matches_upstream(base, expected):
    agent = make_agent("claude-sdk", model="anthropic/claude-opus-5", api_base_url=base)
    agent.apply_tool_policy(POLICY)
    assert _blob(agent)["task"]["model"] == expected
