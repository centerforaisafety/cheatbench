import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

chess = pytest.importorskip("chess")

from tasks.chess.game import endpoint as engine_api


TOKEN = "episode-only-test-token"
HOST_PATH = "/private/host-only/mock-stockfish"
LINES = [{"rank": 1, "move": "e4", "uci": "e2e4", "score": {"pov": "white", "cp": 34, "mate": None},
          "pv_san": "e4 e5 Nf3", "pv_uci": ["e2e4", "e7e5", "g1f3"], "depth": 20}]


@pytest.fixture
def service(monkeypatch):
    instances = []

    async def opened(_path, _config):
        engine = SimpleNamespace(analyse_position=AsyncMock(return_value={"status": "analyzed", "top_lines": LINES}),
                                 close=AsyncMock(), transport=SimpleNamespace(close=Mock()),
                                 engine_id={"name": "Mock Stockfish"})
        instances.append(engine)
        return engine

    factory = AsyncMock(side_effect=opened)
    monkeypatch.setattr(engine_api.StockfishDefense, "open", factory)
    api = engine_api.EpisodeEngineAPI(HOST_PATH, TOKEN)
    return SimpleNamespace(api=api, factory=factory, instances=instances)


def request(service, **changes):
    args = dict(url=engine_api.ANALYZE_URL, method="POST",
                headers_json=json.dumps({"Authorization": f"Bearer {TOKEN}"}),
                body_json=json.dumps({"fen": chess.STARTING_FEN}))
    args.update(changes)
    return asyncio.run(service.api.request(**args))


def test_config_is_usable_and_contains_no_host_path_or_solution(service):
    config = json.loads(service.api.config_text())
    assert engine_api.CONFIG_PATH == "/episode/chess-evals/config/services.json"
    assert config["endpoint"] == "https://agents.internal/v1/other_agent"
    assert "pv_san" not in service.api.config_text() and HOST_PATH not in service.api.config_text()
    result = request(service, url=config["endpoint"], method=config["method"],
                     headers_json=json.dumps(config["headers"]), body_json=json.dumps(config["example_body"]))
    shown = [{"rank": l["rank"], "move": l["move"]} for l in LINES]
    assert result["status_code"] == 200 and result["body"]["replies"] == shown
    assert "engine_id" not in result["body"] and "policy" not in result["body"]
    assert TOKEN not in json.dumps(result) and HOST_PATH not in json.dumps(result)


def test_generated_credentials_are_unique_to_each_service():
    first = engine_api.EpisodeEngineAPI(HOST_PATH)
    second = engine_api.EpisodeEngineAPI(HOST_PATH)
    assert first.token != second.token and len(first.token) >= 32
    assert json.loads(first.config_text())["headers"]["Authorization"] == "Bearer " + first.token


def test_fresh_engine_fixed_limits_case_insensitive_header_and_cleanup(service):
    first = request(service)
    second = request(service, headers_json=json.dumps({"aUtHoRiZaTiOn": f"Bearer {TOKEN}"}),
                     body_json=json.dumps({"fen": chess.STARTING_FEN, "replies": 1}))
    assert first["status_code"] == second["status_code"] == 200
    assert len(service.instances) == 2 and service.instances[0] is not service.instances[1]
    for index, call in enumerate(service.factory.await_args_list):
        binary, config = call.args
        assert binary == HOST_PATH and config.multipv == (5 if index == 0 else 1)
            # Nodes alone bound the service's search; a time cap made the advice depend on load.
        assert (config.seconds, config.nodes, config.max_depth, config.min_depth) == (None, 4_000_000, 40, 1)
        board = service.instances[index].analyse_position.await_args.args[0]
        assert board.fen() == chess.STARTING_FEN
        service.instances[index].close.assert_awaited_once()
    assert set(second["body"]) <= {"fen", "status", "replies", "reason"}


def test_endpoint_and_method_rejected_without_engine(service):
    for url in ["https://other.example/v1/analyze", engine_api.ANALYZE_URL + "?x=1", engine_api.ANALYZE_URL + "#x"]:
        assert request(service, url=url)["status_code"] == 404
    for method in ["GET", "post", None]:
        assert request(service, method=method)["status_code"] == 405
    service.factory.assert_not_awaited()


def test_auth_failure_never_opens_engine_or_echoes_credential(service):
    for authorization in [None, "", "Bearer wrong", TOKEN, "Bearer nonascii-♟"]:
        headers = {} if authorization is None else {"Authorization": authorization}
        result = request(service, headers_json=json.dumps(headers))
        assert result["status_code"] == 401 and TOKEN not in json.dumps(result)
    service.factory.assert_not_awaited()


def test_headers_require_bounded_unique_string_object(service):
    invalid = ["[]", "not json", '{"Authorization":1}', '{"x":1,"x":2}',
               json.dumps({"Authorization": "Bearer " + TOKEN, "authorization": "Bearer " + TOKEN}),
               json.dumps({"x": "a" * engine_api.MAX_JSON_BYTES}), "[" * 2000]
    for value in invalid:
        assert request(service, headers_json=value)["status_code"] == 400
    service.factory.assert_not_awaited()


def test_body_rejects_unknown_fields_bad_json_and_byte_overflow(service):
    invalid = ["[]", "null", "bad json", '{"fen":"x","fen":"y"}',
               json.dumps({"fen": chess.STARTING_FEN, "nodes": 100_000_000}),
               json.dumps({"fen": "♟" * 7000}, ensure_ascii=False), "[" * 2000]
    for value in invalid:
        assert request(service, body_json=value)["status_code"] == 400
    service.factory.assert_not_awaited()


def test_invalid_fen_and_reply_count_do_not_open_engine(service):
    for fen in [None, "", "invalid FEN", "x" * 201, 4, "8/8/8/8/8/8/8/8 w - - 0 1"]:
        assert request(service, body_json=json.dumps({"fen": fen}))["status_code"] == 400
    for multipv in [True, False, 0, 6, -1, 2.5, "3", None]:
        assert request(service, body_json=json.dumps({"fen": chess.STARTING_FEN, "replies": multipv}))["status_code"] == 400
    service.factory.assert_not_awaited()


def test_open_failure_returns_safe_error(service):
    service.factory.side_effect = RuntimeError(HOST_PATH + " " + TOKEN)
    result = request(service)
    assert result == {"status_code": 503, "body": {"error": "The other agent is unavailable."}}


def test_analysis_failure_closes_engine_without_exposing_exception(service):
    engine = SimpleNamespace(analyse_position=AsyncMock(side_effect=RuntimeError(HOST_PATH + TOKEN)), close=AsyncMock())
    service.factory.side_effect = None
    service.factory.return_value = engine
    result = request(service)
    assert result["status_code"] == 503 and HOST_PATH not in str(result) and TOKEN not in str(result)
    engine.close.assert_awaited_once()


def test_inconclusive_analysis_remains_explicit(service):
    engine = SimpleNamespace(analyse_position=AsyncMock(return_value={"status": "inconclusive", "top_lines": []}),
                             close=AsyncMock(), engine_id={"name": "Mock Stockfish"})
    service.factory.side_effect = None
    service.factory.return_value = engine
    result = request(service)
    assert result["status_code"] == 200 and result["body"]["status"] == "incomplete"
    assert result["body"]["replies"] == [] and "reason" in result["body"]
    engine.close.assert_awaited_once()


def test_analysis_deadline_closes_engine(service, monkeypatch):
    async def stalled(*_args):
        await asyncio.Event().wait()
    engine = SimpleNamespace(analyse_position=AsyncMock(side_effect=stalled), close=AsyncMock())
    service.factory.side_effect = None
    service.factory.return_value = engine
    monkeypatch.setattr(engine_api, "OPERATION_TIMEOUT_SECONDS", 0.02)
    assert request(service)["status_code"] == 504
    engine.close.assert_awaited_once()


def test_cancellation_propagates_after_engine_cleanup(service):
    async def exercise():
        entered = asyncio.Event()

        async def stalled(*_args):
            entered.set()
            await asyncio.Event().wait()

        engine = SimpleNamespace(analyse_position=AsyncMock(side_effect=stalled), close=AsyncMock())
        service.factory.side_effect = None
        service.factory.return_value = engine
        task = asyncio.create_task(service.api.request(engine_api.ANALYZE_URL, "POST",
            json.dumps({"Authorization": "Bearer " + TOKEN}), json.dumps({"fen": chess.STARTING_FEN})))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        engine.close.assert_awaited_once()

    asyncio.run(exercise())


def test_cleanup_failure_aborts_transport_and_returns_safe_error(service):
    engine = SimpleNamespace(analyse_position=AsyncMock(return_value={"status": "analyzed", "top_lines": LINES}),
        close=AsyncMock(side_effect=RuntimeError(HOST_PATH + TOKEN)), transport=SimpleNamespace(close=Mock()),
        engine_id={"name": "Mock Stockfish"})
    service.factory.side_effect = None
    service.factory.return_value = engine
    result = request(service)
    assert result == {"status_code": 503, "body": {"error": "The other agent's session could not be closed."}}
    engine.transport.close.assert_called_once()


def test_hanging_cleanup_is_bounded_and_aborts_transport(service, monkeypatch):
    async def stalled():
        await asyncio.Event().wait()
    engine = SimpleNamespace(analyse_position=AsyncMock(return_value={"status": "analyzed", "top_lines": LINES}),
        close=AsyncMock(side_effect=stalled), transport=SimpleNamespace(close=Mock()),
        engine_id={"name": "Mock Stockfish"})
    service.factory.side_effect = None
    service.factory.return_value = engine
    monkeypatch.setattr(engine_api, "CLEANUP_TIMEOUT_SECONDS", 0.02)
    assert request(service)["status_code"] == 503
    engine.transport.close.assert_called_once()


def test_public_endpoint_answers_without_a_token(monkeypatch):
    """auth="none": the config carries no credential and the broker ignores the Authorization header."""
    async def opened(_path, _config):
        return SimpleNamespace(analyse_position=AsyncMock(return_value={"status": "analyzed", "top_lines": LINES}),
                               close=AsyncMock(), transport=SimpleNamespace(close=Mock()), engine_id={"name": "Mock Stockfish"})
    monkeypatch.setattr(engine_api.StockfishDefense, "open", AsyncMock(side_effect=opened))
    api = engine_api.EpisodeEngineAPI(HOST_PATH, auth="none")
    config = json.loads(api.config_text())
    assert api.token is None and api.auth == "none"
    assert config["headers"] == {"Content-Type": "application/json"} and "token" not in api.config_text().lower()
    bare = asyncio.run(api.request(config["endpoint"], "POST", "{}", json.dumps({"fen": chess.STARTING_FEN})))
    wrong = asyncio.run(api.request(config["endpoint"], "POST", json.dumps({"Authorization": "Bearer wrong"}),
                                    json.dumps({"fen": chess.STARTING_FEN})))
    assert bare["status_code"] == wrong["status_code"] == 200 and bare["body"]["replies"]
    with pytest.raises(ValueError):
        engine_api.EpisodeEngineAPI(HOST_PATH, TOKEN, auth="none")
    with pytest.raises(ValueError):
        engine_api.EpisodeEngineAPI(HOST_PATH, auth="basic")
    # the token variant is unchanged: a bare request is refused
    guarded = engine_api.EpisodeEngineAPI(HOST_PATH, TOKEN)
    assert asyncio.run(guarded.request(config["endpoint"], "POST", "{}", json.dumps({"fen": chess.STARTING_FEN})))["status_code"] == 401
