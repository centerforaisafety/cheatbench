"""The engine must answer the same everywhere, and say so before an episode starts.

The engine's numbers depend on its build, its backend and the GPU family's arithmetic --
never on the machine's memory or load (every search is visit-bounded, every setting pinned).
So each harness process analyses two fixed positions on start and compares the answers with
the ones recorded here:

- identical to the last digit: the same build on the same kind of hardware ("exact");
- within 0.001 in win rate: a different backend or GPU family, whose rounding differs at
  the seventh digit and whose decisions agree, because every certified line clears its
  runner-up by at least 0.05 ("compatible"; measured 2026-09-12, Metal against the CPU
  build: 1e-7 apart on both probes, identical lines and depths);
- further than that: refuse to run, because the positions were certified on a different
  engine and the results would not be comparable.

    python -m tasks.go.engine_signature            # print the live signature and the verdict
    python -m tasks.go.engine_signature --record   # write data/engine_signature.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from .go_game import GoGame, Settings
from .katago_api import KataGoEngine

SIGNATURE_PATH = Path(__file__).resolve().parent / "data" / "engine_signature.json"
PROBES = ("ogs_68506069", "ogs_63091411")
VISITS = 400
EXACT = 1e-9
COMPATIBLE = 1e-3


def engine_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    version = next((l for l in lines if l.startswith("KataGo")), "unknown")
    backend = next((l.split("backend")[0].replace("Using", "").strip() for l in lines if "backend" in l), "unknown")
    return f"{version} ({backend})"


async def probe(engine: KataGoEngine) -> dict:
    """The engine's version and, for each probe position, its root numbers and top move."""
    answers = []
    fixtures = Path(__file__).resolve().parent / "fixtures"
    for fid in PROBES:
        fixture = json.loads((fixtures / f"tournament_{fid}.json").read_text())
        game = GoGame(fixture, engine, Settings())
        a = await engine.analyse(game.move_list, initial_stones=game.stones, visits=VISITS,
                                 board_size=game.size, komi=game.komi, rules=game.rules)
        root = a.get("rootInfo") or {}
        top = sorted(a.get("moveInfos", []), key=lambda m: m.get("order", 99))
        answers.append({"position": fid, "visits": VISITS, "winrate": root.get("winrate"),
                        "score_lead": root.get("scoreLead"), "top_move": top[0].get("move") if top else None})
    return {"engine": engine_version(engine.katago), "network": Path(engine.network).name,
            "settings": engine.settings, "probes": answers}


async def verify(engine: KataGoEngine) -> dict:
    """Compare the live engine with the recorded signature. Returns what run.json records;
    raises RuntimeError when the engine is not one the positions were certified on."""
    if not SIGNATURE_PATH.is_file():
        raise RuntimeError(f"{SIGNATURE_PATH} is missing; record it with `python -m tasks.go.engine_signature "
                           "--record` on a machine whose engine is known to be right.")
    expected = json.loads(SIGNATURE_PATH.read_text())
    live = await probe(engine)
    deltas = [abs(float(e["winrate"]) - float(l["winrate"])) for e, l in zip(expected["probes"], live["probes"])]
    moves_agree = all(e["top_move"] == l["top_move"] for e, l in zip(expected["probes"], live["probes"]))
    worst = max(deltas)
    if live["network"] != expected["network"]:
        raise RuntimeError(f"The engine loaded {live['network']}; the positions were certified on {expected['network']}.")
    if worst > COMPATIBLE or not moves_agree:
        raise RuntimeError(
            f"The engine ({live['engine']}) answers the signature probes {worst:.2e} in win rate away from the recorded "
            f"{expected['engine']}, or prefers a different move; the positions were certified on that engine and the "
            f"results would not be comparable. See tasks/go/README.md.")
    return {"status": "exact" if worst <= EXACT else "compatible", "max_winrate_delta": worst,
            "live_engine": live["engine"], "recorded_engine": expected["engine"], "probes": live["probes"]}


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write the live signature to data/engine_signature.json")
    a = ap.parse_args()
    engine = KataGoEngine(max_visits=VISITS)
    await engine.start()
    try:
        live = await probe(engine)
        print(json.dumps(live, indent=1))
        if a.record:
            SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SIGNATURE_PATH.write_text(json.dumps(live, indent=1) + "\n")
            print(f"wrote {SIGNATURE_PATH}")
        else:
            print(json.dumps(await verify(engine), indent=1))
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(_main())
