"""Stockfish evidence for competitive defending moves, not an exhaustive mate proof."""

from __future__ import annotations

import argparse

import asyncio

from dataclasses import asdict, dataclass

import json

import math

from pathlib import Path

import time

import chess

import chess.engine

from .board import parse_move, san_tokens, validated_board

@dataclass(frozen=True)
class DefenseConfig:
    multipv: int = 5
    cp_tolerance: int = 150
    mate_tolerance: int = 0
    min_depth: int = 16
    max_depth: int = 40
    nodes: int = 4_000_000
    seconds: float | None = 10.0    # None: nodes alone bound the search, so the answer never depends on load

    def __post_init__(self):
        for name in ("multipv", "min_depth", "max_depth", "nodes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cp_tolerance", "mate_tolerance"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth cannot exceed max_depth")
        if self.seconds is not None and (isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float))
                                         or not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("seconds must be finite and positive")

    def to_dict(self):
        return asdict(self)

def compare_scores(best: chess.engine.Score, candidate: chess.engine.Score,
                   config: DefenseConfig) -> tuple[bool, str]:
    """Compare scores at the same root/depth, from the defender's point of view."""
    for score in (best, candidate):
        if score.is_mate() and score.mate() == 0:
            raise ValueError("A nonterminal root cannot have a mate-zero score")
    if candidate >= best:
        return True, "The submitted reply matches or improves the best comparison score."
    best_mate, candidate_mate = best.mate(), candidate.mate()
    if best_mate is None and candidate_mate is None:
        allowed = best.score() - candidate.score() <= config.cp_tolerance
        return allowed, ("The submitted reply is within the score tolerance." if allowed else
                         "The submitted reply loses more than the allowed centipawn tolerance.")
    if best_mate is not None and candidate_mate is not None:
        if best_mate > 0 and candidate_mate > 0:
            allowed = candidate_mate - best_mate <= config.mate_tolerance
            return allowed, ("The submitted reply preserves the mating distance." if allowed else
                             "The submitted reply delays a forced win beyond the mate tolerance.")
        if best_mate < 0 and candidate_mate < 0:
            allowed = abs(best_mate) - abs(candidate_mate) <= config.mate_tolerance
            return allowed, ("The submitted reply preserves survival time." if allowed else
                             "The submitted reply shortens survival beyond the mate tolerance.")
        return False, "The submitted reply changes a forced win into a forced loss."
    if best_mate is not None:
        return False, "The submitted reply gives up an engine-found forced win."
    return False, "The submitted reply allows forced mate when a stronger defense is available."

def complete_iteration(updates: list[dict], expected_count: int, min_depth: int) -> list[dict]:
    """Use a complete common-depth MultiPV cohort, excluding search bounds."""
    depths: dict[int, dict[int, dict]] = {}
    for info in updates:
        depth, rank = info.get("depth"), info.get("multipv", 1)
        if not isinstance(depth, int) or depth < min_depth or not 1 <= rank <= expected_count:
            continue
        depths.setdefault(depth, {})[rank] = info
    for depth in sorted(depths, reverse=True):
        rows = depths[depth]
        if set(rows) != set(range(1, expected_count + 1)):
            continue
        cohort = [rows[i] for i in range(1, expected_count + 1)]
        if any(not row.get("pv") or row.get("score") is None or row.get("lowerbound")
               or row.get("upperbound") for row in cohort):
            continue
        if any(row["score"].relative.is_mate() and row["score"].relative.mate() == 0 for row in cohort):
            continue
        if len({row["pv"][0] for row in cohort}) != expected_count:
            continue
        return cohort
    return []

def serialize_line(board: chess.Board, info: dict) -> dict:
    replay = board.copy(stack=True)
    sans = []
    for move in info["pv"]:
        if move not in replay.legal_moves:
            raise RuntimeError("Stockfish returned an illegal principal variation")
        sans.append(replay.san(move))
        replay.push(move)
    score = info["score"].pov(board.turn)
    wdl = info.get("wdl")
    if wdl is not None:
        wdl = wdl.pov(board.turn)
        wdl = {"win": wdl.wins, "draw": wdl.draws, "loss": wdl.losses}
    return {"rank": info.get("multipv", 1), "move": sans[0], "uci": info["pv"][0].uci(),
            "score": {"pov": "white" if board.turn else "black", "cp": score.score(), "mate": score.mate()},
            "pv_san": " ".join(sans), "pv_uci": [m.uci() for m in info["pv"]],
            "depth": info["depth"], "seldepth": info.get("seldepth"), "nodes": info.get("nodes"),
            "time_s": info.get("time"), "wdl": wdl,
            "lowerbound": False, "upperbound": False}

def _score(row: dict) -> chess.engine.Score:
    value = row["score"]
    return chess.engine.Cp(value["cp"]) if value["cp"] is not None else chess.engine.Mate(value["mate"])

class StockfishDefense:
    def __init__(self, transport, engine, config: DefenseConfig):
        self.transport, self.engine, self.config = transport, engine, config
        self.engine_id = dict(engine.id)
        self.engine_options = {}

    @classmethod
    async def open(cls, binary: str, config: DefenseConfig):
        transport, engine = await chess.engine.popen_uci(binary)
        checker = cls(transport, engine, config)
        try:
            options = {"Threads": 1, "Hash": 128, "SyzygyProbeLimit": 0}
            if "UCI_ShowWDL" in engine.options:
                options["UCI_ShowWDL"] = True
            await engine.configure(options)
            checker.engine_options = options
        except BaseException:
            transport.close()
            raise
        return checker

    async def close(self):
        try:
            await asyncio.wait_for(self.engine.quit(), timeout=5)
        except (TimeoutError, chess.engine.EngineError):
            pass
        finally:
            self.transport.close()

    async def _search(self, board: chess.Board, count: int, roots=None) -> tuple[list[dict], dict]:
        # Reset per search so earlier submissions and practice games cannot warm this grader.
        await self.engine.configure({"Clear Hash": None})
        limit = chess.engine.Limit(time=self.config.seconds, nodes=self.config.nodes,
                                   depth=self.config.max_depth)
        started = time.monotonic()
        analysis = await self.engine.analysis(board.copy(stack=True), limit, multipv=count,
                                             root_moves=roots, game=object())
        updates = []
        try:
            async for info in analysis:
                updates.append(info)
        finally:
            analysis.stop()
            try:
                await asyncio.wait_for(analysis.wait(), timeout=5)
            except TimeoutError as exc:
                self.transport.close()
                raise RuntimeError("Stockfish failed to stop its search") from exc
        rows = complete_iteration(updates, count, self.config.min_depth)
        return [serialize_line(board, info) for info in rows], {
            "elapsed_s": round(time.monotonic() - started, 3), "requested_lines": count,
            "common_depth": rows[0]["depth"] if rows else None,
            "maximum_reported_depth": max((i.get("depth", 0) for i in updates), default=0),
            "incomplete_or_bounded_updates": sum(bool(i.get("lowerbound") or i.get("upperbound")) for i in updates),
        }

    async def analyse_position(self, board: chess.Board, candidate: chess.Move | None = None) -> dict:
        legal = list(board.legal_moves)
        if not legal:
            return {"status": "inconclusive", "reason": "No legal moves exist at this analysis root.",
                    "top_lines": [], "comparison_lines": [], "searches": []}
        if candidate is not None and candidate not in legal:
            raise ValueError("Candidate move is illegal")
        top, search = await self._search(board, min(self.config.multipv, len(legal)))
        report = {"status": "analyzed" if top else "inconclusive", "reason": None,
                  "top_lines": top, "comparison_lines": top, "searches": [search],
                  "comparison_scope": "global_multipv"}
        if not top:
            report["reason"] = "Stockfish did not return complete comparable scores at the required depth."
            return report
        if candidate is not None and candidate.uci() not in {line["uci"] for line in top}:
            # Include the submitted move even when outside N: rank alone is not the acceptance rule.
            roots = [chess.Move.from_uci(line["uci"]) for line in top] + [candidate]
            compared, extra = await self._search(board, len(roots), roots)
            report.update(comparison_lines=compared, comparison_scope="screened_candidates_plus_submission")
            report["searches"].append(extra)
            if not compared:
                report.update(status="inconclusive",
                              reason="Stockfish could not compare the submitted reply at a common reliable depth.")
        return report


async def inspect(args):
    config = DefenseConfig(multipv=args.multipv, cp_tolerance=args.cp_tolerance,
                           seconds=args.seconds, nodes=args.nodes, min_depth=args.min_depth,
                           max_depth=args.max_depth, mate_tolerance=args.mate_tolerance)
    checker = await StockfishDefense.open(args.stockfish, config)
    try:
        board = validated_board(args.fen)
        if args.san_sequence is not None:
            if args.after_san or args.candidate:
                raise ValueError("--san-sequence cannot be combined with --after-san or --candidate")
            data = await checker.check(args.fen, args.san_sequence)
        else:
            for san in san_tokens(args.after_san):
                board.push(parse_move(board, san))
            candidate = parse_move(board, args.candidate) if args.candidate else None
            data = await checker.analyse_position(board, candidate)
        report = {"fen": board.fen(), "engine_id": checker.engine_id,
                  "engine_options": checker.engine_options, "policy": config.to_dict(),
                  "tablebases_used": False, "hash_reset_policy": "before_every_search", **data}
        encoded = json.dumps(report, indent=2) + "\n"
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as f:
                f.write(encoded)
        print(encoded, end="")
    finally:
        await checker.close()
