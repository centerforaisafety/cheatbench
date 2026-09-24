#!/usr/bin/env python3
"""Recompute `cost_usd` / `cost_source` on runs that predate the cost helper.

Nothing has to be re-run to price a finished episode: every record in
`episodes.jsonl` already carries the token counts the estimate is computed from,
so this reads a run in place and rewrites those two fields with exactly what
`Agent.episode_cost()` would have produced at the time. It is the same code path
a live run uses -- the adapter is rebuilt from the record's own `agent` and
`model` -- so a backfilled number and a fresh one are the same measurement.

    python -m core.cost_backfill outputs/openmath_*_probe_*        # dry run
    python -m core.cost_backfill outputs/... --write               # rewrite

It lives in `core/` and not in `scripts/` because `scripts/` is gitignored (it
holds machine-specific launchers), and a tool that rewrites run records is part
of the harness rather than a local convenience.

DRY RUN BY DEFAULT: it prints what it would change and touches nothing until
`--write`. A run whose SLURM job is still going must not be passed at all; there
is no lock on `episodes.jsonl` and the writer appends to it as episodes finish.

Idempotent. A record already labelled `cost_source: "estimated"` has its
`cost_usd` cleared before recomputation, so a second pass re-estimates rather
than mistaking this script's own earlier output for a figure the vendor
reported.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agents import AgentFactory
from .agents.factory import LEGACY_AGENT

_AGENTS: dict[tuple, object] = {}


def _agent(record: dict):
    """The adapter that produced this record, built to convert and nothing else."""
    key = (record.get("agent") or LEGACY_AGENT, record.get("model") or "")
    if key not in _AGENTS:
        _AGENTS[key] = AgentFactory.create_agent_from_name(key[0], model=key[1])
    return _AGENTS[key]


def recost(record: dict) -> tuple[float | None, str | None]:
    """`(cost_usd, cost_source)` for one saved episode record."""
    probe = dict(record)
    if record.get("cost_source") == "estimated":
        # Ours from a previous pass, not the vendor's. See the module docstring.
        probe["cost_usd"] = None
    return _agent(record).episode_cost(probe)


def backfill(run_dir: Path, write: bool) -> dict:
    """Recost every episode of one run. Returns a summary for the caller."""
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        return {"run": run_dir.name, "error": "no episodes.jsonl"}

    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed, out, total, sources = 0, [], 0.0, {}
    for record in lines:
        before = record.get("cost_usd")
        cost, source = recost(record)
        if cost != before or record.get("cost_source") != source:
            changed += 1
        record["cost_usd"], record["cost_source"] = cost, source
        total += float(cost or 0.0)
        sources[source] = sources.get(source, 0) + 1
        out.append(record)

    if write and changed:
        # Atomic: a half-written episodes.jsonl is a destroyed run record.
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in out))
        os.replace(tmp, path)

    return {"run": run_dir.name, "episodes": len(lines), "changed": changed,
            "total_usd": total, "sources": sources}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path,
                    help="run directories under outputs/")
    ap.add_argument("--write", action="store_true",
                    help="rewrite episodes.jsonl (default: dry run)")
    args = ap.parse_args()

    print(f"{'run':56} {'eps':>4} {'chg':>4} {'total $':>10}  sources")
    for run_dir in args.runs:
        s = backfill(run_dir, args.write)
        if s.get("error"):
            print(f"{s['run']:56} -- {s['error']}")
            continue
        sources = ", ".join(f"{k or 'none'}={v}" for k, v in sorted(
            s["sources"].items(), key=lambda kv: str(kv[0])))
        print(f"{s['run']:56} {s['episodes']:4} {s['changed']:4} "
              f"{s['total_usd']:10.4f}  {sources}")
    if not args.write:
        print("\n(dry run -- pass --write to rewrite episodes.jsonl)")


if __name__ == "__main__":
    main()
