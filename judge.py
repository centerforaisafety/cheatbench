#!/usr/bin/env python3
"""Re-judge a finished run, without re-running any episode.

    python judge.py outputs/gdpval_claude-opus-5_339446
    python judge.py outputs/<run> --judge gpt-5.6-terra --redo   # non-default judge

Reads the run's saved transcripts, the repo-level `prompts/judge.md` and the
task's `judge_schema.py`, and writes verdicts to `<run>/judge.jsonl`. Episode
records are never touched, so a bad rubric cannot corrupt run data: fix the
prompt or the schema, run this again with --redo.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import judge as judging   # noqa: E402
from core.trial import load_task    # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MODELS_CONFIG = ROOT / "configs" / "models.yaml"


def load_records(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python run.py <task>` first")
    recs: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = rec.get("episode") or rec.get("id")
        if key:
            recs[key] = rec           # last write wins, matching run.py's resume
    return list(recs.values())


def load_judged(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = rec.get("episode") or rec.get("id")
        if key and rec.get("parse_ok"):
            done.add(key)
    return done


async def main_async(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    models_config = Path(args.models_config)
    # --judge selects an ordinary entry from the model configuration.
    args.judge = args.judge or judging.DEFAULT_JUDGE
    judge_cfg = judging.load_judge_config(models_config, args.judge)
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found -- is {run_dir} a run directory?")
    meta = json.loads(meta_path.read_text())

    task = load_task(args.task or meta.get("task_root") or meta["task"],
                     meta.get("variant") or "")
    spec = judging.load_judge_spec(task.root, args.judge_prompt or None,
                                   task_name=task.name)

    out_path = run_dir / args.out
    if args.redo and out_path.exists():
        out_path.unlink()             # --redo replaces the verdicts, never appends

    records = load_records(run_dir / "episodes.jsonl")
    done = load_judged(out_path)
    todo = [r for r in records
            if (r.get("episode") or r.get("id")) not in done
            and judging.is_judgeable(r)]
    # Split so the note below says WHICH kind was skipped: an episode with no
    # transcript never produced one, while a LOST episode has a partial
    # transcript that `python run.py ...` will replace by re-running it.
    skipped = [r for r in records if not judging.is_judgeable(r)]
    no_transcript = [r for r in skipped if not r.get("transcript_path")]
    partial = [r for r in skipped if r.get("transcript_path")]
    if args.limit:
        todo = todo[: args.limit]

    fields = spec.fields
    judge = judging.make_judge_agent(args.judge, models_config)

    print(f"run={run_dir}  task={task.name}  prompt={spec.prompt_path}  "
          f"schema={spec.schema_path.name}  judge={args.judge}  "
          f"episodes={len(records)}  judged={len(done)}  judging={len(todo)}  "
          f"fields={list(fields)}", flush=True)
    if no_transcript:
        print(f"note: {len(no_transcript)} record(s) have no transcript "
              f"(failed episodes) -- skipped", flush=True)
    if partial:
        reasons = sorted({r.get("failure") for r in partial})
        print(f"note: {len(partial)} record(s) are partial transcripts from a "
              f"lost episode {reasons} -- skipped; re-run the episode with "
              f"`python run.py ...` to replace them", flush=True)
    if not todo:
        return 0

    sem = asyncio.Semaphore(args.max_concurrent)
    lock = asyncio.Lock()
    tally = {"n": 0, "ok": 0, "null": 0, "cost": 0.0}

    async def one(rec: dict) -> None:
        async with sem:
            try:
                transcript = json.loads(Path(rec["transcript_path"]).read_text())
            except Exception as e:  # noqa: BLE001
                out = {"verdict": None, "raw": "", "parse_ok": False,
                       "output_mode": None, "judge_cost_usd": 0.0,
                       "judge_usage": None,
                       "judge_error": f"transcript unreadable: {type(e).__name__}: {e}",
                       "judge_attempts": 0}
            else:
                try:
                    out = await judging.judge_episode(transcript, judge, spec)
                except Exception as e:  # noqa: BLE001 - isolate one bad episode
                    # A provider/transport error or a reply that will not
                    # validate must not discard every in-flight verdict. Record
                    # the failure on THIS episode as parse_ok=False and let the
                    # rest of the pass complete. The episode is NOT marked done
                    # (load_judged keys on parse_ok), so re-running -- including
                    # judge.py <run> --redo -- re-judges exactly the failed ones.
                    out = {"verdict": None, "raw": "", "parse_ok": False,
                           "output_mode": None, "judge_cost_usd": 0.0,
                           "judge_usage": None,
                           "judge_error": f"{type(e).__name__}: {e}",
                           "judge_attempts": 1}

            row = {"id": rec["id"],
                   "episode": rec.get("episode") or rec["id"],
                   "replicate": rec.get("replicate"),
                   "task": task.name, "model": rec.get("model"),
                   "judge": args.judge,
                   "judge_generation_config": judge_cfg["generation_config"],
                   "judge_generation_config_unsupported": judge_cfg["unsupported"],
                   "judge_prompt": str(spec.prompt_path),
                   "judge_schema": spec.schema_path.name,
                   "parse_ok": out["parse_ok"],
                   **{k: (out["verdict"] or {}).get(k) for k in fields},
                   # How this verdict was produced. There is only one way --
                   # API-enforced structured output -- and the row says so, so
                   # it can never be confused with a pre-refactor row that was
                   # scraped out of prose.
                   "output_mode": out["output_mode"],
                   "judge_cost_usd": out["judge_cost_usd"],
                   "judge_usage": out["judge_usage"],
                   "judge_attempts": out["judge_attempts"],
                   "judge_error": out["judge_error"],
                   # The raw judge text duplicates every parsed field, so it is
                   # noise on a successful verdict. Kept ONLY when parsing
                   # failed, where it is the only way to see what was said.
                   **({} if out["parse_ok"] else {"raw": out["raw"]}),
                   "transcript_path": rec["transcript_path"]}

            async with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                tally["n"] += 1
                tally["cost"] += float(out["judge_cost_usd"] or 0.0)
                tally["ok" if out["parse_ok"] else "null"] += 1
                head = {k: row[k] for k in fields[:3]}
                print(f"[{tally['n']}/{len(todo)}] {row['episode']} {head} "
                      f"${out['judge_cost_usd']:.4f}"
                      f"{' ERR: ' + str(out['judge_error']) if out['judge_error'] else ''}",
                      flush=True)

    # return_exceptions so an unexpected failure in ONE episode's write path
    # cannot cancel the others mid-flight; per-episode judge failures are
    # already caught above and recorded as parse_ok=False rows.
    results = await asyncio.gather(*(one(r) for r in todo),
                                   return_exceptions=True)
    for rec, res in zip(todo, results):
        if isinstance(res, Exception):
            print(f"WARNING: episode {rec.get('episode') or rec.get('id')} "
                  f"raised {type(res).__name__}: {res}", flush=True)

    print("\n=== judge summary ===")
    print(f"episodes judged:      {tally['n']}")
    print(f"unparseable verdicts: {tally['null']}")
    print(f"judge cost:           ${tally['cost']:.4f}")
    print(f"verdicts ->           {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="an outputs/<run> directory")
    ap.add_argument("--judge", default="",
                    help="model name from models.yaml (same entries as --model); "
                         f"default: {judging.DEFAULT_JUDGE}")
    ap.add_argument("--task", default="",
                    help="override the task folder recorded in run.json")
    ap.add_argument("--judge-prompt", default="",
                    help="override the general judge prompt "
                         f"(default: {judging.JUDGE_PROMPT})")
    ap.add_argument("--out", default="judge.jsonl",
                    help="output file inside the run directory")
    ap.add_argument("--redo", action="store_true",
                    help="re-judge ids that already have a parsed verdict")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--models-config", default=str(MODELS_CONFIG))
    ap.add_argument("--max-concurrent", type=int, default=8)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
