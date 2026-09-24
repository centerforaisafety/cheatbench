"""Play each fixture's own best line through the real harness and see when the
opponent resigns.

The chess eval learned this the hard way: a position can look winning and
forced on paper and still be useless, because the opponent resigns on move one.
Four of nine chess candidates were rejected for exactly that. A fixture is only
usable if the resignation lands at the END of the line the model has to find,
not at the start.

    python tasks/go/tools/reference_games.py
    python tasks/go/tools/reference_games.py --moves 6 tournament_ogs_puzzle_8037

Each fixture is replayed several times (--repeat), each in a fresh engine, and the
replays must agree to the last digit; every move of a usable line must also beat
the runner-up by --min-gap in win rate, so that no position rides on a coin flip.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.go.go_game import GoGame, Settings  # noqa: E402
from tasks.go.katago_api import KataGoEngine  # noqa: E402


async def best_move(engine, game: GoGame, visits: int) -> tuple[str | None, list[dict]]:
    a = await engine.analyse(game.move_list, initial_stones=game.stones, visits=visits,
                             board_size=game.size, komi=game.komi, rules=game.rules)
    infos = sorted(a.get("moveInfos", []), key=lambda m: m.get("order", 99))
    top = [{"move": m.get("move"), "winrate": m.get("winrate"), "visits": m.get("visits")} for m in infos[:2]]
    return (top[0]["move"] if top else None), top


async def play_reference(engine, fixture: dict, max_moves: int, visits: int) -> dict:
    game = GoGame(fixture, engine, Settings(move_cap=max_moves))
    line, steps, t0 = [], [], time.monotonic()
    verdict, at = "played out without resigning", None
    for i in range(max_moves):
        move, top = await best_move(engine, game, visits)
        if move is None:
            verdict, at = "engine offered no move", i
            break
        line.append(move)
        before = len(game.move_list)
        result = await game.play(move)
        steps.append({"move": move, "top": top, "reply": [m[1] for m in game.move_list[before + 1:]]})
        if result.get("result") == "1-0":
            verdict, at = "opponent resigned", i + 1
            break
        if result.get("result") is not None or result.get("error"):
            verdict, at = result.get("reason") or result.get("error"), i + 1
            break
    # How clearly the line's own moves stand out from the runner-up, at the worst step.
    gaps = [s["top"][0]["winrate"] - s["top"][1]["winrate"] for s in steps if len(s["top"]) > 1]
    return {"id": fixture["id"], "verdict": verdict, "resigned_after": at,
            "line": line, "seconds": round(time.monotonic() - t0),
            "min_gap": round(min(gaps), 4) if gaps else None, "steps": steps}


def _same(a: dict, b: dict) -> bool:
    keys = ("verdict", "resigned_after", "line", "steps")
    return all(json.dumps(a[k]) == json.dumps(b[k]) for k in keys)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--fixtures", default="tasks/go/fixtures")
    ap.add_argument("--moves", type=int, default=8, help="how many of its own best moves to play")
    ap.add_argument("--visits", type=int, default=400)
    ap.add_argument("--repeat", type=int, default=3,
                    help="replay each fixture this many times, each in a fresh engine; they must agree to the digit")
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="a usable line's every move must beat the runner-up by this much win rate")
    ap.add_argument("--out", default="tasks/go/data/reference_games.json")
    a = ap.parse_args()

    paths = sorted(Path(a.fixtures).glob("tournament_*.json"))
    if a.ids:
        paths = [p for p in paths if p.stem in a.ids or p.stem.replace("tournament_", "") in a.ids]
    rows = []
    for path in paths:
        fixture = json.loads(path.read_text())
        # A fresh engine per replay. The engine is deterministic (see katago_api.py), so
        # every replay of a fixture must agree to the last digit; one that does not is a
        # regression in the engine, not a property of the position.
        runs = []
        for _ in range(max(1, a.repeat)):
            engine = KataGoEngine(max_visits=a.visits)
            await engine.start()
            try:
                runs.append(await play_reference(engine, fixture, a.moves, a.visits))
            finally:
                await engine.close()
        r = dict(runs[0], replays=len(runs), identical=all(_same(x, runs[0]) for x in runs),
                 seconds=sum(x["seconds"] for x in runs))
        rows.append(r)
        resigned = r["verdict"] == "opponent resigned"
        deep = (r["resigned_after"] or 0) >= 2
        clear = r["min_gap"] is not None and r["min_gap"] >= a.min_gap
        flag = ("USABLE" if resigned and deep and clear and r["identical"]
                else "NOT REPRODUCIBLE" if not r["identical"]
                else "TOO EARLY" if resigned and not deep
                else "KNIFE-EDGE" if resigned and not clear
                else "no resignation")
        print(f"  {r['id']:26s} {r['verdict']:32s} after {str(r['resigned_after']):>4s} moves "
              f"x{len(runs)} {'identical' if r['identical'] else 'DIFFER':9s} gap {r['min_gap']!s:>7s} "
              f"({r['seconds']:>3d}s)  {flag}", flush=True)
        print(f"      line: {' '.join(r['line'])}   replies: {' '.join(m for s in r['steps'] for m in s['reply'])}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    usable = [r for r in rows if r["verdict"] == "opponent resigned" and (r["resigned_after"] or 0) >= 2
              and r["identical"] and r["min_gap"] is not None and r["min_gap"] >= a.min_gap]
    print(f"\n{len(usable)} of {len(rows)} fixtures resign after 2+ moves of the model's own best line, "
          f"reproducibly, with every move clear of the runner-up by {a.min_gap}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
