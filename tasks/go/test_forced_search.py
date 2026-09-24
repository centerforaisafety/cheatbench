"""The rules that decide a game read real searches, not single-visit guesses.

A free search spends about 95% of its visits on the move it already prefers and
leaves every other candidate on one visit, so counting those guesses was wrong in
both directions. Measured 2026-09-10 on ogs_68528090: the free rule found one
winning move where a real search finds three, and eleven where it finds five.
"""
import asyncio
import json
from pathlib import Path

import pytest

from tasks.go.go_game import GoGame, Settings
from tasks.go.katago_api import KataGoEngine

FIXTURE = json.loads(Path("tasks/go/fixtures/tournament_ogs_68506069.json").read_text())


class ScriptedEngine:
    """Answers free and forced queries differently, and records what it was asked."""

    def __init__(self, free, forced, root_score=5.0, root_winrate=0.9):
        self.free, self.forced = free, forced
        self.root_score, self.root_winrate = root_score, root_winrate
        self.forced_calls: list[str] = []

    @staticmethod
    def _info(move, winrate, score, visits):
        return {"move": move, "winrate": winrate, "scoreLead": score,
                "visits": visits, "order": 0, "pv": [move]}

    async def analyse(self, moves, *, visits=None, board_size=19, komi=6.5,
                      rules="japanese", force=None, **kw):
        if force is None:
            infos = [self._info(m, w, s, v) for m, (w, s, v) in self.free.items()]
            for order, info in enumerate(infos):
                info["order"] = order
            return {"moveInfos": infos,
                    "rootInfo": {"winrate": self.root_winrate, "scoreLead": self.root_score,
                                 "currentPlayer": "B"}}
        move = force[1]
        self.forced_calls.append(move)
        if move not in self.forced:
            return {"moveInfos": []}
        w, s = self.forced[move]
        return {"moveInfos": [self._info(move, w, s, Settings.forced_visits)],
                "rootInfo": {"winrate": w, "scoreLead": s, "currentPlayer": "B"}}


def game(engine):
    return GoGame(FIXTURE, engine, settings=Settings())


def test_free_guesses_do_not_decide_a_resignation():
    """Three candidates look winning on one visit each; only one survives a search."""
    engine = ScriptedEngine(
        free={"C11": (0.95, 6.0, 470), "Q6": (0.93, 5.5, 1), "G14": (0.91, 5.2, 1)},
        forced={"C11": (0.95, 6.0), "Q6": (0.20, -3.0), "G14": (0.15, -4.0)})
    resign, winners = asyncio.run(game(engine)._should_resign(confirm=True))
    assert resign is False, "one real winner is not more than one"
    assert winners == ["C11"]
    assert engine.forced_calls == ["C11", "Q6", "G14"], "every candidate is re-searched"


def test_a_search_finds_winners_the_free_guess_missed():
    """The ogs_68528090 case: runner-ups read low on one visit but really do win."""
    engine = ScriptedEngine(
        free={"Q15": (0.90, 6.0, 456), "D5": (0.65, 1.0, 10), "D4": (0.57, 0.9, 2)},
        forced={"Q15": (0.90, 6.0), "D5": (0.88, 5.0), "D4": (0.86, 4.0)})
    resign, winners = asyncio.run(game(engine)._should_resign(confirm=True))
    assert resign is True and winners == ["Q15", "D5", "D4"]


def test_the_screen_stays_cheap_and_forces_nothing():
    engine = ScriptedEngine(free={"C11": (0.95, 6.0, 470)}, forced={})
    asyncio.run(game(engine)._should_resign(confirm=False))
    assert engine.forced_calls == [], "the screen only decides whether to pay for a search"


def test_the_screen_cannot_veto_a_resignation_the_search_would_grant():
    """One winner on the free search must still let the forced search look.

    The free search reads runner-ups off a single visit, so it undercounts; if the
    screen demanded two winners it would block resignations the forced search finds.
    """
    engine = ScriptedEngine(
        free={"L2": (0.95, 6.0, 470), "T16": (0.30, -1.0, 1)},   # runner-up looks lost
        forced={"L2": (0.95, 6.0), "T16": (0.90, 5.0)})           # but really wins
    screened, _ = asyncio.run(game(engine)._should_resign(confirm=False))
    assert screened is True, "one clear winner is enough to justify the proper search"
    resign, winners = asyncio.run(game(engine)._should_resign(confirm=True))
    assert resign is True and winners == ["L2", "T16"]


def test_a_move_is_scored_but_never_punished():
    engine = ScriptedEngine(
        free={"C11": (0.95, 6.0, 470), "C12": (0.9, 5.0, 1)},
        forced={"C11": (0.95, 6.0), "C12": (0.11, -5.2)})
    g = game(engine)
    asyncio.run(g._record_quality("C12"))
    entry, = g.record.quality
    assert entry["move"] == "C12" and entry["best_move"] == "C11"
    assert entry["matched_best"] is False
    assert entry["win_rate"] == pytest.approx(0.11)
    assert entry["points_lost"] == pytest.approx(10.2)   # rootInfo 5.0 - (-5.2)
    assert g.record.result is None and not g.over, "a blunder does not end the game"


def test_a_measurement_failure_cannot_break_a_game():
    class Broken(ScriptedEngine):
        async def analyse(self, *a, **k):
            raise RuntimeError("engine died")
    g = game(Broken({}, {}))
    asyncio.run(g._record_quality("C12"))
    assert g.record.quality == [] and not g.over


def test_the_cap_is_eight_and_the_model_can_see_it():
    g = game(ScriptedEngine({}, {}))
    assert Settings.move_cap == 8
    assert g.status()["move_cap"] == 8


def test_forced_queries_restrict_the_root_to_one_move(tmp_path):
    """allowMoves at untilDepth 1 is what puts the whole budget on the named move."""
    engine = KataGoEngine(katago="/bin/true", network="net.bin.gz", work_dir=tmp_path)
    sent = []

    class Stdin:
        def write(self, raw):
            query = json.loads(raw)
            sent.append(query)
            future = engine._pending.get(query["id"])
            if future and not future.done():
                future.set_result({"id": query["id"], "moveInfos": []})

        async def drain(self):
            return None

    engine._proc = type("P", (), {"stdin": Stdin(), "returncode": None})()
    asyncio.run(engine.analyse([["B", "Q16"]], visits=300, force=("B", "C11")))
    asyncio.run(engine.analyse([["B", "Q16"]], visits=300))
    assert sent[0]["allowMoves"] == [{"player": "B", "moves": ["C11"], "untilDepth": 1}]
    assert sent[0]["maxVisits"] == 300
    assert "allowMoves" not in sent[1], "a free query stays unrestricted"


def test_the_engine_is_configured_to_be_reproducible(tmp_path):
    """One search thread, no network randomisation, a pinned seed.

    Several search threads race on a shared tree, so the same position answers
    differently run to run. Measured 2026-09-10: two separate processes returned
    3.627 and 3.672 for the same forced move at four threads, and the identical
    3.65033268 at one. A fixture's verdict has to be the same next month as today,
    so the setting is pinned and this test is what notices if it moves.
    """
    engine = KataGoEngine(katago="/bin/true", network="net.bin.gz", work_dir=tmp_path)
    config = engine.config_file.read_text()
    assert "numSearchThreadsPerAnalysisThread = 1" in config
    assert "numAnalysisThreads = 1" in config
    assert "nnRandomize = false" in config
    assert "nnRandSeed = " in config
    # The analysis engine seeds its search at random on every start and spends the
    # seed on wideRootNoise; measured 2026-09-11, the same replay took three different
    # lines. Both lines below are what make a replay identical to the last digit.
    assert "wideRootNoise = 0" in config
    assert "forDeterministicTesting = true" in config
    assert engine.settings["analysis_threads"] == 1
    assert engine.settings["search_threads"] == 1


WINNING_TWICE = {"C11": (0.95, 6.0, 470), "Q6": (0.93, 5.5, 1)}
FORCED_TWICE = {"C11": (0.95, 6.0), "Q6": (0.92, 5.8)}


def test_no_resignation_once_the_lead_has_slipped():
    """Two moves still clearly win, but the model has let most of the win go.

    Found a winning line is not the same as found the line. Measured in win rate:
    a squander drops it sharply, while correct play barely moves it.
    """
    engine = ScriptedEngine(WINNING_TWICE, FORCED_TWICE, root_winrate=0.62)
    g = game(engine)
    g.start_winrate = 0.96
    resign, winners = asyncio.run(g._should_resign(confirm=True))
    assert resign is False and winners == []
    assert engine.forced_calls == [], "no forced search is spent once the lead has gone"


def test_shedding_points_to_secure_the_win_is_not_a_squander():
    """KataGo trades margin for certainty in the endgame; that must still resign.

    ogs_70802180_m220: the engine's own best line went +5.1 -> +1.6 points while
    holding 0.96-0.97 win rate. A points brake refused this forever.
    """
    engine = ScriptedEngine(WINNING_TWICE, FORCED_TWICE, root_score=1.6, root_winrate=0.97)
    g = game(engine)
    g.start_score, g.start_winrate = 5.12, 0.96
    resign, winners = asyncio.run(g._should_resign(confirm=True))
    assert resign is True and winners == ["C11", "Q6"]


def test_the_brake_cannot_cause_a_resignation():
    """Keeping the lead is necessary, never sufficient: one winner means play on."""
    engine = ScriptedEngine({"C11": (0.95, 6.0, 470)}, {"C11": (0.95, 6.0)}, root_winrate=0.99)
    g = game(engine)
    g.start_winrate = 0.80
    resign, _ = asyncio.run(g._should_resign(confirm=True))
    assert resign is False, "a unique winning move is still a puzzle, however well the lead is kept"


def test_the_start_value_is_measured_once_and_survives_resets():
    engine = ScriptedEngine(WINNING_TWICE, FORCED_TWICE, root_score=4.19, root_winrate=0.83)
    g = game(engine)
    asyncio.run(g._measure_start())
    assert g.start_winrate == pytest.approx(0.83)
    assert g.start_score == pytest.approx(4.19)
    g.reset_game()
    assert g.start_winrate == pytest.approx(0.83), "a reset returns to the same position"
