"""The codex adapter's grader: it exists, it is the SAME grader, and it leaks nothing.

This file exists because the adapter used to ship no grader at all, on a
justification that was written into three docstrings and was wrong. The
reasoning was: codex takes an MCP server only from `$CODEX_HOME/config.toml`,
the agent can read that file, therefore hosting the grader there hands the agent
the grader. That is true of ONE of the two transports codex accepts:

    stdio   [mcp_servers.x] command = "..." / args = [...]
            A path to a program. The agent can read the file and run the program
            itself, which for a grader means reading its source and the answer
            key. Never used here, and `test_config_toml_never_names_a_command`
            is the guard.
    url     [mcp_servers.x] url = "http://..."
            An endpoint. The agent can call it -- which it is supposed to be
            able to do -- and there is nothing behind it to read.

Harbor makes the same split (`harbor/src/harbor/agents/installed/codex.py`,
`_build_effective_config`: `command`/`args` when `transport == "stdio"`, else
`{"url": ...}`), and codex-cli 0.152.0's own `codex mcp add --url` writes
exactly the url shape.

The consequence of the old mistake was concrete: gdpval ships a grader, so the
Claude adapter offered `grade_deliverable` and this one did not, and the two
adapters were not running the same task.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

try:                                    # 3.11+; the container has it, 3.10 does not
    import tomllib
except ModuleNotFoundError:             # pragma: no cover - host interpreter
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import trial  # noqa: E402
from core.agents import make_agent  # noqa: E402
from core.agents import codex_runner  # noqa: E402

# Freeze the original lightweight grader for transport regression tests.
# Copied unchanged from c2c0785c3:tasks/knowledge_work/results/legacy/task/grader.py.
# The current document reviewer has its own integration test in test_knowledge_work.py.
GRADER_SRC = (ROOT / "tests/fixtures/gdpval_grader.py").read_text()

CLOSED = {"web_search": False, "web_fetch": False}

# A row shaped like gdpval's, with a marker no other string in the repo has.
ANSWER_KEY = "ANSWER-KEY-CANARY-8f3d21"
ROW = {"id": "probe_row", "answer": ANSWER_KEY}

# Identifiers that appear ONLY in the grader's source -- never in a verdict,
# never in the tool's description, never in anything that legitimately crosses
# to the agent.
#
# `reference_path` is deliberately NOT in this list although it is such an
# identifier: Debian's stock `/usr/lib/git-core/git-submodule` contains it four
# times, so a filesystem sweep for it finds a base-image shell script and says
# nothing about us. A needle that fires on the environment cannot answer a
# question about a leak.
SOURCE_ONLY = ("submitted_reference", "_is_reference", "GOLD_SUBDIR")

# The container sweep looks for THIS instead, appended to the grader source that
# actually ships in the blob. A canary cannot collide with the base image, so a
# hit is a leak and an empty result is a real answer rather than a lucky one.
SOURCE_CANARY = "GRADER-SOURCE-CANARY-51c7ae"


def grader_module() -> dict:
    """The task's grader, exec'd the way the runner execs it."""
    ns: dict = {"__name__": "rh_grader"}
    exec(compile(GRADER_SRC, "<grader>", "exec"), ns)
    return ns


@pytest.fixture()
def server():
    ns = grader_module()
    srv = codex_runner.serve_grader(ns, ROW, {}, "/workspace")
    try:
        yield srv
    finally:
        srv.close()


def rpc(url: str, payload: dict, accept: str = "application/json") -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": accept},
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode()
    if "text/event-stream" in (r.headers.get("Content-Type") or ""):
        # SSE framing: `event: message\ndata: {...}\n\n`
        body = "".join(ln[len("data: "):] for ln in body.splitlines()
                       if ln.startswith("data: "))
    return json.loads(body)


# ==========================================================================
# the config file -- the url shape, and never the stdio one
# ==========================================================================
def test_config_toml_never_names_a_command() -> None:
    """THE GUARD. A `command` here is the grader's source, one `cat` away."""
    text = codex_runner.config_toml(base_url="https://example.invalid/v1",
                                    grader_url="http://127.0.0.1:9/tok/mcp",
                                    grader_name="grader")
    assert "url = " in text
    assert "command" not in text
    assert "args" not in text


def test_config_toml_parses_to_harbors_shape() -> None:
    """`[mcp_servers.<name>] url = ...`, byte for byte what `codex mcp add
    --url` writes and what Harbor's `_build_effective_config` emits for a
    non-stdio transport."""
    if tomllib is None:
        pytest.skip("no TOML parser available (tomllib/tomli)")
    url = "http://127.0.0.1:41234/deadbeef/mcp"
    cfg = tomllib.loads(codex_runner.config_toml(
        base_url="https://example.invalid/v1", grader_url=url))
    assert cfg["mcp_servers"]["grader"] == {"url": url}
    # The top-level key must NOT have been swallowed by the table header: in
    # TOML everything after `[table]` belongs to it, so writing the base URL
    # second would silently make it `mcp_servers.grader.openai_base_url`.
    assert cfg["openai_base_url"] == "https://example.invalid/v1"


def test_config_toml_is_empty_when_there_is_nothing_to_say() -> None:
    """An empty config file is not the same thing as no config file."""
    assert codex_runner.config_toml() == ""


def test_ignore_user_config_yields_to_a_grader() -> None:
    """`--ignore-user-config` makes the CLI skip config.toml entirely, so it
    cannot be passed when that file is the only channel the grader has."""
    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    agent.apply_tool_policy(CLOSED)
    assert "--ignore-user-config" in agent.exec_flags("", grader=False)
    assert "--ignore-user-config" not in agent.exec_flags("", grader=True)
    assert "--ignore-user-config" not in agent.exec_flags("https://x/v1",
                                                          grader=True)


def test_a_real_knowledge_work_blob_disables_ignore_user_config(monkeypatch) -> None:
    """The shipped task, end to end: knowledge_work HAS a grader, so its episodes must
    reach config.toml."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    task = trial.load_task("knowledge_work")
    assert task.grader_src(), "knowledge_work stopped shipping a grader"
    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    agent.apply_tool_policy(task.tools)
    payload = json.loads(agent.blob(
        row=ROW, prompt="p", files=[], skip_dirs=(),
        grader_src=task.grader_src(), stage_src="", timeout_s=60, env={},
        install=None))
    assert "--ignore-user-config" not in payload["task"]["codex"]["exec_flags"]
    assert payload["modules"]["grader"] == task.grader_src()


def test_openmath_still_ships_no_grader(monkeypatch) -> None:
    """The other shipped task must be untouched by all of this: no grader, so
    no row, no MCP server, and `grader_state` stays an explicit None.

    `OPENAI_BASE_URL` is cleared for the same reason the sibling test in
    test_codex_adapter.py clears it: importing `core.llm_agents` anywhere in the
    session load_dotenv()s the repo's .env, and a run pointed at a proxy drops
    `--ignore-user-config` for a reason that has nothing to do with graders.
    """
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    task = trial.load_task("openmath")
    assert task.grader_src() == ""
    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    agent.apply_tool_policy(task.tools)
    payload = json.loads(agent.blob(
        row=ROW, prompt="p", files=[], skip_dirs=(), grader_src="",
        stage_src="", timeout_s=60, env={}, install=None))
    assert "row" not in payload["task"]
    assert ANSWER_KEY not in json.dumps(payload)
    assert "--ignore-user-config" in payload["task"]["codex"]["exec_flags"]
    assert codex_runner.config_toml(grader_url="") == ""


# ==========================================================================
# the tool -- same name, same description, same schema, same result shape
# ==========================================================================
def test_the_tool_name_is_the_claude_path_s_tool_name() -> None:
    """Both CLIs prefix an MCP tool `mcp__<server>__<tool>`, so registering the
    same pair produces the same name on both adapters.

    Measured on a real gdpval codex episode, not assumed: the model's tool
    registry showed `{"name":"mcp__grader__grade_deliverable", ...}` and it
    called `tools.mcp__grader__grade_deliverable({deliverable:"..."})`.

    The composition is asserted rather than the literal alone, so a rename of
    either half cannot leave the adapters offering differently-named tools.
    """
    ns = grader_module()
    assert ns["SERVER_NAME"] == "grader"
    assert ns["TOOL_BASENAME"] == "grade_deliverable"
    assert ns["TOOL_NAME"] == f"mcp__{ns['SERVER_NAME']}__{ns['TOOL_BASENAME']}"
    assert ns["TOOL_NAME"] == "mcp__grader__grade_deliverable"


def test_the_server_advertises_exactly_that_tool(server) -> None:
    listed = rpc(server.url, {"jsonrpc": "2.0", "id": 1,
                              "method": "tools/list", "params": {}})
    tools = listed["result"]["tools"]
    assert [t["name"] for t in tools] == ["grade_deliverable"]
    ns = grader_module()
    assert tools[0]["description"] == ns["DESCRIPTION"]
    assert tools[0]["inputSchema"] == ns["INPUT_SCHEMA"]


def _claude_tool_definitions() -> list[dict]:
    """What the Claude SDK's in-process server actually advertises.

    Driven through the SDK's own `tools/list` handler rather than reconstructed:
    the point of the comparison is that the two adapters describe one tool, and
    a re-derivation here would only prove this file agrees with itself.
    """
    import mcp.types as mcp_types

    ns = grader_module()
    servers, _allowed, _state = ns["make_grader"](ROW, {}, "/workspace")
    handlers = servers[ns["SERVER_NAME"]]["instance"].request_handlers
    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        handlers[mcp_types.ListToolsRequest](
            mcp_types.ListToolsRequest(method="tools/list")))
    return [{"name": t.name, "description": t.description,
             "inputSchema": t.inputSchema} for t in result.root.tools]


def test_the_two_adapters_advertise_one_identical_tool(server) -> None:
    """THE PARITY CHECK. Same name, same description, same JSON Schema.

    Not "both have a grader" -- the same tool DEFINITION, so the two models are
    told the same thing about it. `@tool(name, description, {"deliverable":
    str})` on the Claude side and `INPUT_SCHEMA` on this one are two spellings
    of one fact, and this is what stops them drifting.
    """
    pytest.importorskip("claude_agent_sdk")
    claude = _claude_tool_definitions()
    over_http = rpc(server.url, {"jsonrpc": "2.0", "id": 1,
                                 "method": "tools/list",
                                 "params": {}})["result"]["tools"]
    assert claude == over_http


def test_both_adapters_call_one_implementation() -> None:
    """`make_grader` must be a wrapper, not a second copy of the grader.

    If the Claude path ever grows its own verdict logic again, the two adapters
    stop grading the same way and no test downstream would notice.
    """
    ns = grader_module()
    body = GRADER_SRC.split("def make_grader")[1]
    assert "make_tool(" in body, "make_grader no longer delegates to make_tool"
    for marker in ("_verdict(", "state[\"calls\"]"):
        assert marker not in body, (
            f"make_grader reimplements {marker!r}; it must only wrap make_tool")


def test_the_result_shape_is_the_claude_path_s_result_shape(tmp_path,
                                                            server) -> None:
    """Same three answers, same MCP envelope, so a trajectory reads the same."""
    ns = grader_module()
    _state, call = ns["make_tool"](ROW, {}, str(tmp_path))

    deliverable = tmp_path / "work.xlsx"
    deliverable.write_text("some work")

    direct = call({"deliverable": str(deliverable)})
    assert set(direct) == {"content"}
    assert json.loads(direct["content"][0]["text"])["verdict"] == "below_reference"

    missing = call({"deliverable": "/nope/does-not-exist"})
    assert missing["isError"] is True

    # ...and over the wire it is the same envelope, not a re-wrapped one.
    over_http = rpc(server.url, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "grade_deliverable",
                   "arguments": {"deliverable": str(deliverable)}}})
    assert over_http["result"] == direct


def test_the_error_flag_is_the_one_place_the_two_paths_differ(server) -> None:
    """A KNOWN, PRE-EXISTING asymmetry, pinned so nobody rediscovers it.

    The grader returns MCP's own spelling, `isError`. This adapter puts it on
    the wire unchanged. `claude_agent_sdk.create_sdk_mcp_server` does not: its
    `call_tool` wrapper ends

        return CallToolResult(content=content, isError=result.get("is_error", False))

    -- `is_error`, snake_case, which the grader has never set and neither does
    anything else in the SDK's own `@tool` documentation. So on the Claude path
    a missing deliverable and an exhausted attempt budget reach the model as
    ORDINARY tool results carrying the error text, and on this one they reach it
    flagged.

    The TEXT is identical either way, which is the parity that actually matters
    for what the model learns and for what the judge reads. The flag is not, and
    this test fails the day the SDK starts reading `isError` -- at which point
    the two paths agree and this test should simply assert that.

    Not fixed here on purpose: adding `"is_error": True` to the grader's return
    would change what every Claude gdpval episode has ever shown its model, and
    that is an experiment-design decision, not a cleanup.
    """
    import mcp.types as mcp_types

    ns = grader_module()
    servers, _allowed, _state = ns["make_grader"](ROW, {}, "/workspace")
    handler = servers[ns["SERVER_NAME"]]["instance"].request_handlers[
        mcp_types.CallToolRequest]
    loop = asyncio.get_event_loop_policy().new_event_loop()
    claude = loop.run_until_complete(handler(mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name=ns["TOOL_BASENAME"],
            arguments={"deliverable": "/nope/missing"})))).root

    codex = rpc(server.url, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": ns["TOOL_BASENAME"],
                                        "arguments": {"deliverable":
                                                      "/nope/missing"}}})["result"]

    # The text the model reads: IDENTICAL.
    assert claude.content[0].text == codex["content"][0]["text"]
    # The flag: not identical, and this is the whole of the difference.
    assert codex["isError"] is True
    assert claude.isError is False, (
        "the SDK now honours `isError`; the two paths agree and this test "
        "should assert claude.isError is True")


def test_state_survives_the_transport_and_grading_is_uncapped(server) -> None:
    """`state` is authoritative, it is the dict the record reports, and there is
    no attempt cap: every well-formed call grades."""
    assert server.grader_state["calls"] == 0
    assert server.grader_state["id"] == "probe_row"
    n = 5
    for _ in range(n):
        rpc(server.url, {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                         "params": {"name": "grade_deliverable",
                                    "arguments": {"deliverable": __file__}}})
    assert server.grader_state["calls"] == n
    assert "rejected" not in server.grader_state
    assert server.grader_state["verdicts"] == ["below_reference"] * n


def test_grader_max_caps_when_a_row_opts_in(tmp_path) -> None:
    """`grader_max` is opt-in: absent it there is no ceiling, and the cap keys
    appear in `state` only when a row asked for one."""
    g = grader_module()
    f = tmp_path / "d.txt"
    f.write_text("x")

    state, call = g["make_tool"]({"id": "uncapped"}, {}, str(tmp_path))
    for _ in range(4):
        call({"deliverable": str(f)})
    assert state["calls"] == 4
    assert "rejected" not in state and "max_calls" not in state

    state, call = g["make_tool"]({"id": "capped", "grader_max": 2}, {}, str(tmp_path))
    for _ in range(4):
        call({"deliverable": str(f)})
    assert state["calls"] == 2
    assert state["rejected"] == 2
    assert state["max_calls"] == 2


# ==========================================================================
# the transport -- exactly what codex-cli 0.152.0 sends
# ==========================================================================
# Recorded off the wire from a real `codex exec` against a logging server, not
# copied out of a spec: one POST per JSON-RPC message, `Accept: text/event-
# stream, application/json`, in this order, with no session id and no GET
# stream.
CODEX_ACCEPT = "text/event-stream, application/json"


def test_the_server_completes_the_handshake_codex_actually_sends(server) -> None:
    init = rpc(server.url, {
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18",
                   "capabilities": {"elicitation": {"form": {}, "url": {}}},
                   "clientInfo": {"name": "codex-mcp-client", "title": "Codex",
                                  "version": "0.152.0"}}}, CODEX_ACCEPT)
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "grader"
    assert "tools" in init["result"]["capabilities"]
    assert server.connected is True

    # `notifications/initialized` has no id and must get a body-less 202.
    req = urllib.request.Request(
        server.url, data=b'{"jsonrpc":"2.0","method":"notifications/initialized"}',
        headers={"Content-Type": "application/json", "Accept": CODEX_ACCEPT},
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 202
        assert r.read() == b""

    listed = rpc(server.url, {"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                              "params": {"_meta": {"progressToken": 0}}},
                 CODEX_ACCEPT)
    assert listed["result"]["tools"][0]["name"] == "grade_deliverable"
    assert server.listed is True


def test_an_unknown_method_is_a_jsonrpc_error_not_a_crash(server) -> None:
    out = rpc(server.url, {"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
    assert out["error"]["code"] == -32601


def test_a_call_to_the_wrong_tool_is_an_error_result(server) -> None:
    out = rpc(server.url, {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                           "params": {"name": "read_answer_key",
                                      "arguments": {}}})
    assert out["result"]["isError"] is True


# ==========================================================================
# reachability -- narrow bind, unguessable path, no collisions
# ==========================================================================
def test_it_binds_loopback_only(server) -> None:
    """gdpval has no setup hook, so its episode shares the HOST's network
    namespace: a 0.0.0.0 bind here would put the grader on every interface of
    the node."""
    assert server.server_address[0] == "127.0.0.1"


def test_the_path_is_an_unguessable_per_episode_token(server) -> None:
    assert server.grader_path.startswith("/")
    assert server.grader_path.endswith("/mcp")
    token = server.grader_path.strip("/").split("/")[0]
    assert len(token) == 32                      # 16 random bytes, hex
    int(token, 16)


@pytest.mark.parametrize("path", [
    "/mcp", "/wrong/mcp", "/", "",
    # The four codex itself probes before deciding a server needs no auth.
    "/.well-known/oauth-protected-resource",
    "/.well-known/oauth-authorization-server",
    "/.well-known/openid-configuration",
])
def test_every_other_path_reaches_nothing(server, path: str) -> None:
    base = f"http://127.0.0.1:{server.server_address[1]}"
    req = urllib.request.Request(base + path, data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=15)
    assert excinfo.value.code == 404
    assert excinfo.value.read() == b""


def test_the_token_path_answers_get_and_delete_with_405(server) -> None:
    """Not 404: codex reads 405 as "this server has no OAuth and needs none",
    which is what stops it opening an auth flow against the grader."""
    for method in ("GET", "DELETE"):
        req = urllib.request.Request(server.url, method=method)
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=15)
        assert excinfo.value.code == 405


def test_concurrent_episodes_do_not_collide() -> None:
    """`--max-concurrent 5` on a task with no network namespace of its own puts
    five graders on the same loopback. Ports come from the kernel and tokens
    from `secrets`, so neither can repeat."""
    ns = grader_module()
    servers = [codex_runner.serve_grader(ns, ROW, {}, "/workspace")
               for _ in range(5)]
    try:
        assert len({s.server_address[1] for s in servers}) == 5
        assert len({s.grader_path for s in servers}) == 5
        # Each one answers on its own URL and 404s on its neighbour's path.
        for a, b in zip(servers, servers[1:]):
            assert rpc(a.url, {"jsonrpc": "2.0", "id": 1,
                               "method": "ping"})["result"] == {}
            wrong = f"http://127.0.0.1:{a.server_address[1]}{b.grader_path}"
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(urllib.request.Request(
                    wrong, data=b"{}", method="POST"), timeout=15)
            assert excinfo.value.code == 404
    finally:
        for s in servers:
            s.close()


def test_close_is_idempotent_and_stops_serving() -> None:
    """`run()` closes it when codex exits and `main()` closes it again; a second
    call must not deadlock waiting on a server that is already down."""
    srv = codex_runner.serve_grader(grader_module(), ROW, {}, "/workspace")
    url = srv.url
    done = threading.Event()
    threading.Thread(target=lambda: (srv.close(), srv.close(), done.set()),
                     daemon=True).start()
    assert done.wait(timeout=30), "close() hung on a second call"
    with pytest.raises(Exception):
        urllib.request.urlopen(urllib.request.Request(
            url, data=b"{}", method="POST"), timeout=10)


def test_no_late_call_can_change_the_state_after_close(server) -> None:
    """The record is taken from `state`; nothing may write to it afterwards."""
    server.close()
    assert server.grader_state["calls"] == 0
    with pytest.raises(Exception):
        rpc(server.url, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "grade_deliverable",
                                    "arguments": {"deliverable": __file__}}})
    assert server.grader_state["calls"] == 0


# ==========================================================================
# what the endpoint will hand out
# ==========================================================================
def test_the_endpoint_never_returns_the_source_the_row_or_the_key(server) -> None:
    """Sweep every reply the endpoint can be made to produce."""
    replies = [
        json.dumps(rpc(server.url, {"jsonrpc": "2.0", "id": 0,
                                    "method": "initialize", "params": {}})),
        json.dumps(rpc(server.url, {"jsonrpc": "2.0", "id": 1,
                                    "method": "tools/list", "params": {}})),
        json.dumps(rpc(server.url, {"jsonrpc": "2.0", "id": 2,
                                    "method": "tools/call",
                                    "params": {"name": "grade_deliverable",
                                               "arguments": {"deliverable":
                                                             __file__}}})),
        json.dumps(rpc(server.url, {"jsonrpc": "2.0", "id": 3,
                                    "method": "tools/call",
                                    "params": {"name": "grade_deliverable",
                                               "arguments": {"deliverable":
                                                             "/nope"}}})),
        json.dumps(rpc(server.url, {"jsonrpc": "2.0", "id": 4,
                                    "method": "resources/list"})),
    ]
    blob = "\n".join(replies)
    assert ANSWER_KEY not in blob
    for marker in SOURCE_ONLY:
        assert marker not in blob, f"the endpoint returned {marker!r}"
    # The row's id is the one row field the grader legitimately echoes, and it
    # goes into `state` on the HOST side -- never over the wire.
    assert "probe_row" not in blob


def test_the_config_file_carries_the_url_and_nothing_else(tmp_path) -> None:
    """What an agent gets from `cat $CODEX_HOME/config.toml`."""
    ns = grader_module()
    srv = codex_runner.serve_grader(ns, ROW, {}, "/workspace")
    try:
        cfg = {"codex_home": str(tmp_path / "home"),
               "secrets_dir": str(tmp_path / "secrets"),
               "base_url": "https://example.invalid/v1"}
        codex_runner.prepare_home(cfg, "sk-test", grader_url=srv.url,
                                  grader_name=srv.grader_server_name)
        text = (tmp_path / "home" / "config.toml").read_text()
    finally:
        srv.close()
    assert srv.url in text
    assert "command" not in text
    assert ANSWER_KEY not in text
    for marker in SOURCE_ONLY:
        assert marker not in text


# ==========================================================================
# THE EMPIRICAL CHECK: a real episode container
#
# Everything above runs on the host. This runs the production blob, the
# production bootstrap and the production `serve_grader`/`prepare_home` inside a
# real enroot episode, and then asks the questions an AGENT would ask from
# inside it. Modelled on tests/test_image_anonymity.py, which is the same idea
# for the image.
# ==========================================================================
PROBE = (Path(__file__).resolve().parent / "codex_grader_probe.py").read_text()


@pytest.fixture(scope="module")
def image() -> str:
    from core import sandbox

    sandbox.ensure_path()
    problems = sandbox.preflight()
    if problems:
        pytest.skip("sandbox runtime unavailable: " + "; ".join(problems))
    path = os.environ.get("RH_TEST_IMAGE") or sandbox.DEFAULT_IMAGE
    if not os.path.exists(path):
        pytest.skip(f"{path} not built (python -m core.sandbox --build)")
    return path


@pytest.fixture(scope="module")
def in_container(image: str) -> dict:
    """One real episode container, running the probe instead of `codex exec`."""
    from core import sandbox

    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    agent.apply_tool_policy(CLOSED)
    # PINNED, not inherited. Every real run here goes through a proxy, so
    # config.toml carries `openai_base_url` next to the grader's `url` -- and
    # that is the harder shape: a probe that greps the file for `url = ` finds
    # the proxy first. Fixing the ambient value makes this test exercise the
    # production shape whether or not the session happened to load .env.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENAI_BASE_URL", "https://proxy.invalid/v1")
        payload = json.loads(agent.blob(
            row=ROW, prompt="do the thing", files=[], skip_dirs=(),
            # The real grader, plus a canary the base image cannot contain.
            grader_src=f"{GRADER_SRC}\n# {SOURCE_CANARY}\n",
            stage_src=(ROOT / "core" / "stage.py").read_text(),
            timeout_s=600, env={}, install=None))

    # The production blob, with `codex exec` swapped for the probe. The runner
    # rides along as a module so the probe calls the REAL serve_grader and the
    # REAL prepare_home rather than a copy of them.
    payload["task"]["install"] = None          # no npm install: codex never runs
    payload["task"]["probe_needles"] = [ANSWER_KEY, SOURCE_CANARY, *SOURCE_ONLY]
    payload["modules"]["runner"] = payload["code"]
    payload["code"] = PROBE

    argv = sandbox.container_argv(image, agent.bootstrap, private_net=False,
                                  key_env=agent.API_KEY_ENV, pass_key=False)
    res = subprocess.run(argv, env=sandbox.spawn_env(""),
                         input=json.dumps(payload), capture_output=True,
                         text=True, timeout=1800)
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    if not lines:
        pytest.fail(f"probe produced no record.\nSTDERR:\n{res.stderr}")
    return json.loads(lines[-1])


def test_in_container_the_agent_can_call_the_tool(in_container: dict) -> None:
    """The whole point of the url transport: the tool WORKS from inside."""
    report = in_container
    assert report.get("call_error") is None, report.get("call_error")
    assert report["url_recovered_from_config"] is True
    # The grader's own endpoint, not the proxy that sits beside it in the file.
    assert report["url_is_loopback"] is True
    listed = json.loads(report["tools_list"]["body"])
    assert [t["name"] for t in listed["result"]["tools"]] == ["grade_deliverable"]
    called = json.loads(report["tools_call"]["body"])
    assert called["result"]["isError"] is True      # no such deliverable staged
    assert report["wrong_path_status"] == 404


def test_in_container_the_config_file_names_a_url_and_no_command(
        in_container: dict) -> None:
    text = in_container["config_toml"]
    assert "[mcp_servers.grader]" in text
    assert "url = " in text
    assert "command" not in text
    assert ANSWER_KEY not in text
    for marker in SOURCE_ONLY:
        assert marker not in text, f"config.toml leaked {marker!r}"


def test_in_container_nothing_on_the_filesystem_holds_the_grader_or_the_row(
        in_container: dict) -> None:
    """A full sweep of the episode's own filesystem, every regular file.

    Run WHILE the grader is serving and after config.toml has been written, so
    it is the state the agent would actually find.

    NO TRUNCATION on failure: whatever leaked, in full, because the point of the
    message is to say precisely what an agent would have found.
    """
    assert in_container["filesystem_error"] is None, \
        in_container["filesystem_error"]
    hits = in_container["filesystem_hits"]
    assert hits == [], (
        "reachable on the container filesystem:\n"
        + json.dumps(hits, indent=1))


def test_in_container_nothing_in_proc_holds_the_grader_or_the_row(
        in_container: dict) -> None:
    """argv and environ of every process, which the agent runs as the same uid
    as and can therefore read.

    This is the check that catches the mistake the design exists to avoid: a row
    or a grader path delivered on a command line or in the environment is
    readable for the life of the process, and `os.environ.pop` does not change
    what /proc reports.
    """
    assert in_container["proc_error"] is None, in_container["proc_error"]
    hits = in_container["proc_hits"]
    assert hits == [], "reachable through /proc:\n" + json.dumps(hits, indent=1)


def test_in_container_the_codex_home_holds_only_auth_and_config(
        in_container: dict) -> None:
    """`ls -laR $CODEX_HOME`: the grader is an endpoint, not a file."""
    listing = in_container["codex_home_listing"]
    assert "config.toml" in listing
    for name in ("grader.py", "grader", "row.json", "data.jsonl"):
        assert f" {name}" not in listing, f"$CODEX_HOME holds {name}:\n{listing}"
