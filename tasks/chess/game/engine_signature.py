"""The engine must answer the same everywhere.

Stockfish with one thread, a fixed hash and a node-bounded search is deterministic, and its
node count and score for a given search are the same on every architecture the same
version is built for (verified 2026-09-12: the arm64 and x86_64 slices of the Stockfish 19
universal binary agree to the node on three positions). What can differ between machines is
the version. So every harness process searches two fixed positions to a fixed node count on
start and refuses to run unless the answers match the ones recorded here.

    python -m tasks.chess.game.engine_signature --stockfish <binary>     # print the live signature
    python -m tasks.chess.game.engine_signature --stockfish <binary> --check    # refuse a different engine
    python -m tasks.chess.game.engine_signature --stockfish <binary> --record   # write data/engine_signature.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.engine

SIGNATURE_PATH = Path(__file__).resolve().parent / "data" / "engine_signature.json"
PROBES = [("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 200_000),
          ("3r1r2/pp1b1pk1/3R2pp/q3p3/2B1P2N/1P2P3/P4QPP/6K1 w - - 1 25", 200_000)]
OPTIONS = {"Threads": 1, "Hash": 128, "SyzygyProbeLimit": 0}
_verified: set[str] = set()


def probe(engine: chess.engine.SimpleEngine) -> dict:
    """The engine's name and, for each probe, its best move, score and exact node count."""
    answers = []
    for fen, nodes in PROBES:
        engine.configure({"Clear Hash": None})
        info = engine.analyse(chess.Board(fen), chess.engine.Limit(nodes=nodes))
        answers.append({"fen": fen, "nodes_limit": nodes, "bestmove": info["pv"][0].uci(),
                        "score": str(info["score"].white()), "nodes": info.get("nodes"), "depth": info.get("depth")})
    return {"engine": engine.id.get("name"), "options": OPTIONS, "probes": answers}


def forget(binary: str) -> None:
    """Make the next verify() probe this binary again; used after an engine restart."""
    _verified.discard(binary)


def verify(engine: chess.engine.SimpleEngine, binary: str) -> dict:
    """Raise unless this engine answers the probes as the recorded signature says. Once per binary per process."""
    if not SIGNATURE_PATH.is_file():
        raise RuntimeError(f"{SIGNATURE_PATH} is missing; record it with `python -m tasks.chess.game.engine_signature "
                           f"--stockfish {binary} --record` on a machine whose engine is known to be right.")
    expected = json.loads(SIGNATURE_PATH.read_text())
    if binary in _verified:
        return expected
    live = probe(engine)
    mismatches = [(e, l) for e, l in zip(expected["probes"], live["probes"])
                  if (e["bestmove"], e["score"], e["nodes"]) != (l["bestmove"], l["score"], l["nodes"])]
    if live["engine"] != expected["engine"] or mismatches:
        raise RuntimeError(
            f"The engine at {binary} is {live['engine']!r} and answers the signature probes differently from the "
            f"recorded {expected['engine']!r} ({SIGNATURE_PATH.name}). Install that version (see tasks/chess/README.md); "
            f"results from a different engine are not comparable.")
    _verified.add(binary)
    return expected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="verify the recorded signature; exit nonzero on mismatch")
    action.add_argument("--record", action="store_true", help="write the live signature to data/engine_signature.json")
    a = ap.parse_args(argv)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(a.stockfish, timeout=30)
        try:
            engine.configure(OPTIONS)
            if a.check:
                forget(a.stockfish)
                verify(engine, a.stockfish)
                print(f"PASS: {engine.id.get('name')} matches the recorded signature")
                return 0
            live = probe(engine)
        finally:
            engine.close()
    except (OSError, RuntimeError, chess.engine.EngineError) as exc:
        print(f"Stockfish check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(live, indent=1))
    if a.record:
        SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGNATURE_PATH.write_text(json.dumps(live, indent=1) + "\n")
        print(f"wrote {SIGNATURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
