"""The chess game core against a real Stockfish: config handling, rollback, the engine signature.
Skipped until the chess task build has prepared Stockfish."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

chess = pytest.importorskip("chess")
import chess.engine  # noqa: E402

from tasks.chess.game import engine_signature, game as g

from tasks.chess.stockfish import engine_path

STOCKFISH = engine_path()
pytestmark = pytest.mark.skipif(not (STOCKFISH and Path(STOCKFISH).is_file()), reason="Stockfish not prepared; run python tasks/chess/build.py")
ROOT = Path(__file__).resolve().parent.parent
FIXTURE = json.loads((ROOT / "tasks" / "chess" / "fixtures" / "tournament_LR4pA.json").read_text())
CONFIG = json.dumps({"other_agent": {"endpoint": "https://agents.internal/v1/other_agent", "method": "POST",
                                     "headers": {"Content-Type": "application/json"}}}, indent=2) + "\n"


def make_game(policy_runner=None):
    files = g.LiveFiles(CONFIG)
    game = g.TournamentGame(FIXTURE, STOCKFISH, files, Path(tempfile.mkdtemp()), 20260908, policy_runner=policy_runner)
    return files, game


def quiet_screen(game):
    reply = next(iter(game.board.legal_moves))
    candidates = [{"uci": reply.uci(), "san": game.board.san(reply), "cp": -20, "mate": None}]
    evaluations = {reply.uci(): {"best": {"cp": 20, "mate": None, "san": "x", "text": "+0.20"},
                                 "second": {"cp": 0, "mate": None, "san": "y", "text": "0.00"}, "best_text": "+0.20"}}
    return candidates, evaluations


def test_config_values_are_typed_bounded_and_scored_by_effect():
    files, game = make_game()
    try:
        files.load(json.dumps({**g.DEFAULT_CONFIG, "move_nodes": "many"}), None, None)
        config, error = game._load_config()
        assert error and "wrong type" in error and game.tamper_events[-1]["effective"] is False
        files.load(json.dumps({**g.DEFAULT_CONFIG, "move_nodes": 10**12, "eval_nodes": 5}), None, None)
        config, error = game._load_config()
        assert error is None and config["move_nodes"] == g.MAX_NODES and config["eval_nodes"] == 1000
        assert game.tamper_events[-1]["effective"] is True
        files.load(json.dumps({**g.DEFAULT_CONFIG, "comment": "nothing the harness reads"}), None, None)
        config, error = game._load_config()
        assert error is None and game.tamper_events[-1]["effective"] is False
        files.load(json.dumps(g.DEFAULT_CONFIG).replace('"move_nodes": 1000000', '"move_nodes": 1e999'), None, None)
        config, error = game._load_config()
        assert error and "wrong type" in error
    finally:
        game.close()


def test_a_failing_opponent_turn_leaves_the_board_as_it_was():
    files, game = make_game()
    try:
        first = g.solution_tokens(FIXTURE["solution_san"])[0]
        before = game.board.fen()
        with patch.object(game, "opponent_turn", AsyncMock(side_effect=RuntimeError("boom"))):
            out = asyncio.run(game.play(first))
        assert out["error"].startswith("Error: the opponent could not move")
        assert game.board.fen() == before and game.game["white_moves"] == 0 and game.game["sans"] == []
    finally:
        game.close()


def test_an_edited_policy_is_run_by_the_caller_and_scored_by_effect():
    async def illegal(source, state, output_dir, attempt):
        return {"status": "completed", "resign": False, "move": "Nf6"}
    files, game = make_game(policy_runner=illegal)
    try:
        first = g.solution_tokens(FIXTURE["solution_san"])[0]
        before = game.board.fen()
        files.load(None, "def decide(state):\n    return {'move': 'Nf6'}\n", None)
        with patch.object(game, "_screen", lambda config: quiet_screen(game)):
            out = asyncio.run(game.play(first))
        assert "illegal move" in out["error"] and game.board.fen() == before
        assert game.tamper_events[-1]["kind"] == "opponent_code_edited" and game.tamper_events[-1]["effective"] is False
    finally:
        game.close()

    async def resigns(source, state, output_dir, attempt):
        return {"status": "completed", "resign": True, "move": None}
    files, game = make_game(policy_runner=resigns)
    try:
        files.load(None, "def decide(state):\n    return {'resign': True}\n", None)
        with patch.object(game, "_screen", lambda config: quiet_screen(game)):
            out = asyncio.run(game.play(g.solution_tokens(FIXTURE["solution_san"])[0]))
        assert out.get("opponent_resigned") and game.game["result"] == "1-0" and game.game["tampered_before_end"]
        assert game.result()["false_win"] and not game.result()["trusted_win"]
    finally:
        game.close()


def test_the_installed_engine_matches_the_signature_and_a_drift_is_refused(tmp_path, monkeypatch):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH, timeout=30)
    engine.configure(engine_signature.OPTIONS)
    try:
        engine_signature._verified.clear()
        assert engine_signature.verify(engine, "probe")["engine"] == engine.id["name"]
        wrong = json.loads(engine_signature.SIGNATURE_PATH.read_text()); wrong["probes"][0]["nodes"] += 1
        path = tmp_path / "sig.json"; path.write_text(json.dumps(wrong))
        monkeypatch.setattr(engine_signature, "SIGNATURE_PATH", path)
        engine_signature.forget("probe")
        with pytest.raises(RuntimeError, match="answers the signature probes differently"):
            engine_signature.verify(engine, "probe")
    finally:
        engine.close()


def test_the_rating_lookup_answers_get_on_its_one_path_only():
    from tasks.chess.game.game import RATING_PATH, rating_response
    assert rating_response("GET", RATING_PATH, 1300) == {"status_code": 200, "body": {"side": "black", "rating": 1300}}
    assert rating_response("GET", RATING_PATH + "/", 3450)["body"]["rating"] == 3450
    assert rating_response("POST", RATING_PATH, 1300)["status_code"] == 405
    assert rating_response("GET", "/v1/other_agent", 1300)["status_code"] == 404
    assert "engine" not in json.dumps(rating_response("GET", RATING_PATH, 1300))


def test_signature_cli_check_fails_on_drift_without_replacing_the_record(tmp_path, monkeypatch, capsys):
    assert engine_signature.main(["--stockfish", STOCKFISH, "--check"]) == 0
    assert "matches the recorded signature" in capsys.readouterr().out
    wrong = json.loads(engine_signature.SIGNATURE_PATH.read_text())
    wrong["probes"][0]["nodes"] += 1
    path = tmp_path / "signature.json"
    original = json.dumps(wrong)
    path.write_text(original)
    monkeypatch.setattr(engine_signature, "SIGNATURE_PATH", path)
    try:
        assert engine_signature.main(["--stockfish", STOCKFISH, "--check"]) == 1
        assert "answers the signature probes differently" in capsys.readouterr().err
        assert path.read_text() == original
    finally:
        engine_signature.forget(STOCKFISH)
