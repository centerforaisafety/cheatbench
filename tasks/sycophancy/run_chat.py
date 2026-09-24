#!/usr/bin/env python3
"""run.py for a task that is one message and one reply: no agent, no tools, no sandbox.

    python tasks/sycophancy/run_chat.py --model gpt-6-astra --repeat 100 --run-id n100
    python tasks/sycophancy/run_chat.py --model claude-fable-5-1 --variant dontcheat --repeat 100 --run-id n100
    python tasks/sycophancy/run_chat.py sycophancy --model gpt-5.6-sol --limit 1 --repeat 1

Everything here is run.py's own machinery with the container removed:

  core/trial.py     load_task, render_prompt, episode_id   -- the task, its rows, its prompt
  core/routing.py   resolve, require_key                    -- the entry's endpoint and credential
  core/llm_agents   get_llm_agent_class                     -- the provider client, litellm cost
  core/judge.py     load_judge_spec, make_judge_agent, judge_episode -- the shared judge, the
                                                               task's judge_schema.py
  configs/models.yaml                                       -- the model entry, read as written

The model settings and API parameters are resolved by core/chat_config.py and
core/model_settings.py, shared with judging and review servers. Harness options
are not sent to the API. Archived display options are recorded as unsupported.
Anthropic calls preserve thinking/display and translate shared reasoning effort
into output_config.effort, with the existing 16000-token default output cap.

The model id goes where the adapters send it (factory_model). A vendor's own endpoint
takes the bare name, which the factory's vendor client strips for it; a gateway in
front of the vendor (LiteLLM at `${LITELLM_BASE_URL}`) takes the entry's id verbatim
-- `openrouter/x-ai/grok-4.6`, `gemini/gemini-3.8-flash` on its OpenAI-compatible
route, `anthropic/claude-opus-5` on the Anthropic client -- as grok-build, kimi-code,
gemini-cli and claude-sdk send it, decided by the base URL's host as muse-code decides
its route. A vendor core/llm_agents.py has no client for --
`meta/` (Muse at api.meta.ai) or a bare name (DeepSeek at api.deepseek.com) -- is
reached with the OpenAI client at the entry's own base URL under the name the vendor
expects. Nothing else is added: no temperature, one call per episode; a key the
provider still rejects fails that episode. run.json records the entry's settings as
written (`entry_generation_config`, `entry_extra_body`) and as sent
(`generation_config`), and the client's id (`routing.factory_model`).

Records use run.py's layout and field names (core/trial.py): outputs/<task>_<variant>_<model>_<run-id>/
with run.json, episodes.jsonl, judge.jsonl and trajectories/<episode>/transcript.json, so
`python judge.py outputs/...` re-judges a run and the task's scorers read it unchanged.
transcript.json carries the reply in the message shape the shared judge reads for a
transcript with no `agent` key (core/agents/factory.py: trajectory_from_transcript).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from core import judge as judging      # noqa: E402
from core import routing, trial, task_build        # noqa: E402
from core.config import load_yaml      # noqa: E402
from core.llm_agents import get_llm_agent_class  # noqa: E402

MODELS_CONFIG = ROOT / "configs" / "models.yaml"
OUTPUTS = ROOT / "outputs"
RUNNER = "chat-completion"

# Compatibility imports for external callers; configuration belongs to core.
from core.model_settings import factory_model, ANTHROPIC_MAX_TOKENS


def chat_generation_config(name: str, model: str, gen: dict, extra_body: dict) -> tuple:
    """(sent, extra_body, unsupported): the entry's settings in the chat API's terms.

    `gen` and `extra_body` are the entry's, as written. The translation table is in
    the module docstring; every key is either sent, translated, or recorded as
    unsupported, never dropped in silence.
    """
    from core.model_settings import api_generation
    return api_generation(model, gen, extra_body, legacy_display=True)


def load_config(config_path: Path, name: str) -> dict:
    """The `models:` entry NAME, resolved the way core/judge.py resolves a judge entry.

    Same checks, same routing; `generation_config` and `extra_body` come back in the
    chat API's terms (chat_generation_config) beside the entry's own
    (`entry_generation_config`, `entry_extra_body`), and `factory_model` names the
    client core/llm_agents.py builds (factory_model).
    """
    from core.chat_config import load_chat_config
    cfg = load_chat_config(config_path, name)
    cfg['factory_model'] = factory_model(name, cfg['model'], cfg, where=str(config_path))
    return cfg


def make_model(name: str, cfg: dict):
    """The provider client, built as core/judge.py:make_judge_agent builds the judge's."""
    kwargs: dict = {}
    if cfg.get("api_key_env"):
        routing.require_key(cfg["api_key_env"], name=name, consumer="chat runner")
        kwargs["api_key_env"] = cfg["api_key_env"]
    if cfg.get("api_base_url"):
        kwargs["api_base_url"] = cfg["api_base_url"]
    gen = dict(cfg["generation_config"])
    if cfg.get("extra_body"):
        gen["extra_body"] = dict(cfg["extra_body"])
    return get_llm_agent_class(cfg["factory_model"], gen, **kwargs), gen


def routing_record(cfg: dict) -> dict:
    base = cfg.get("api_base_url")
    return {"model_id": cfg["model"], "factory_model": cfg["factory_model"],
            "api_key_env": cfg.get("api_key_env"),
            "api_base_url": routing.sanitise_url(base), "api_base_host": routing.url_host(base),
            "extra_body": dict(cfg.get("extra_body") or {}) or None}


def sha256(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def row_digest(prompt: str, row: dict) -> str:
    """What an episode of ROW measures: its rendered prompt and the `vars` its record carries."""
    return sha256(json.dumps({"prompt": prompt, "vars": row.get("vars") or {}}, sort_keys=True))


def check_resume(out_dir: Path, done_rows: set, settings: dict, row_sha256: dict) -> None:
    """Refuse to resume a run whose finished episodes were not made from the current inputs.

    Comparing the prompt file's name is not enough: the template, a row's text or
    `vars`, or the judge's prompt or schema can change under the same names, and
    resuming would then append new answers to old ones as if they were one run.
    `settings` holds the run-level values (model, generation config, routing, judge
    and the content hashes of the prompt template and judge inputs); `row_sha256`
    maps each selected row id to the digest of what its episode measures. Every
    finished row still selected must match; a run.json written before these hashes
    existed cannot be verified and is refused as well. --redo starts over.
    """
    path = out_dir / "run.json"
    if not done_rows or not path.exists():
        return
    previous = json.loads(path.read_text())
    if any(k not in previous for k in ("prompt_sha256", "judge_prompt_sha256",
                                      "judge_schema_sha256", "row_sha256")):
        raise SystemExit(f"{out_dir} holds finished episodes but its run.json predates content "
                         f"hashing, so their prompts cannot be verified. Use another --run-id, "
                         f"or --redo to discard them.")
    changed = {k: (previous.get(k), v) for k, v in settings.items() if previous.get(k) != v}
    stale = sorted(r for r in done_rows if r in row_sha256
                   and previous["row_sha256"].get(r) != row_sha256[r])
    if changed or stale:
        why = []
        if changed:
            why.append(f"settings differ: {json.dumps(changed, default=str)}")
        if stale:
            shown = ", ".join(stale[:5]) + (f" and {len(stale) - 5} more" if len(stale) > 5 else "")
            why.append(f"the prompt or vars of finished rows changed: {shown}")
        raise SystemExit(f"{out_dir} holds finished episodes made under different inputs; "
                         f"{'; '.join(why)}. Use another --run-id, or --redo to discard them.")


def write_transcript(trial_dir: Path, task_name: str, row: dict, replicate: int,
                     model_id: str, prompt: str, prompt_file: str, text: str) -> Path:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "prompt.txt").write_text(prompt)
    transcript = {"id": row["id"], "episode": trial.episode_id(row["id"], replicate),
                  "replicate": replicate, "task": task_name, "model": model_id,
                  "prompt": prompt, "prompt_file": prompt_file, "grader_state": None,
                  "messages": [{"_type": "AssistantMessage",
                                "content": [{"_type": "TextBlock", "text": text or ""}]}]}
    path = trial_dir / "transcript.json"
    path.write_text(json.dumps(transcript, indent=1))
    return path


def completed_status(ep_path: Path, judge_path: Path, no_judge: bool) -> int:
    """Provider and judge failures must never look like a successful batch."""
    def latest(path):
        return {r["episode"]: r for r in
                (json.loads(line) for line in path.read_text().splitlines() if line.strip())} if path.exists() else {}
    episodes = latest(ep_path)
    if not episodes or any(not r.get("ok") or r.get("failure") for r in episodes.values()):
        return 1
    if not no_judge:
        judges = latest(judge_path)
        missing = [eid for eid in episodes if not judges.get(eid, {}).get("parse_ok")
                   or judges.get(eid, {}).get("judge_error")]
        if missing:
            print(f"{len(missing)} episodes lack valid judgments; run judge.py on {ep_path.parent}", flush=True)
            return 1
    return 0


async def main_async(args: argparse.Namespace) -> int:
    from tqdm.asyncio import tqdm

    task = trial.load_task(args.task, args.variant)
    models_config = Path(args.models_config)
    cfg = load_config(models_config, args.model)
    args.judge = args.judge or judging.DEFAULT_JUDGE
    judge_cfg = ({"generation_config": {}, "unsupported": {}} if args.no_judge
                 else judging.load_judge_config(models_config, args.judge))
    rows, build_record = task_build.ensure_task_built(task)
    if args.limit:
        rows = rows[:args.limit]

    model, gen = make_model(args.model, cfg)
    if cfg["unsupported"]:
        print(f"model {args.model}: Chat Completions does not support {cfg['unsupported']}; "
              f"reasoning-display settings are not sent.", flush=True)
    if gen != cfg["entry_generation_config"] or (cfg["extra_body"] or {}) != (cfg["entry_extra_body"] or {}):
        print(f"model {args.model}: entry settings {cfg['entry_generation_config']} "
              f"extra_body={cfg['entry_extra_body'] or {}} are sent to the chat API as "
              f"{gen}", flush=True)
    model_routing = routing_record(cfg)
    judge_routing = ({"api_base_url": None, "api_key_env": None, "api_key_env_source": "disabled"}
                     if args.no_judge else judging.judge_routing_record(args.judge, judge_cfg))

    args.prompt = args.prompt or task.default_prompt
    prompt_path = task.prompt_path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"prompt {prompt_path} not found")
    prompt_file = str(prompt_path.relative_to(task.root))

    judge_spec = judging.load_judge_spec(task.root, task_name=task.name)
    judge_agent = None
    if not args.no_judge:
        try:
            judge_agent = judging.make_judge_agent(args.judge, models_config)
        except Exception as e:  # noqa: BLE001
            print(f"judge initialization failed: {type(e).__name__}: {e}")
            return 1

    run_id = args.run_id or os.environ.get("SLURM_JOB_ID") or \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Named as run.py names a variant's run directory.
    run_name = f"{task.name}_{task.variant}" if task.variant else task.name
    out_dir = Path(args.output_dir) / f"{run_name}_{args.model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_path, judge_path = out_dir / "episodes.jsonl", out_dir / "judge.jsonl"

    # Resume: an episode already recorded as ok is kept, as run.py keeps it. A resumed
    # run must be the same run: the settings and the content of everything an episode
    # measures are compared with the existing run.json (check_resume), and a difference
    # is refused rather than written over the record of the episodes already done.
    # --redo starts over.
    done, done_rows = set(), set()
    if args.redo:
        for old in (ep_path, judge_path):
            old.unlink(missing_ok=True)
    elif ep_path.exists():
        for line in ep_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["episode"])
                    done_rows.add(rec["id"])
    prompts = {row["id"]: trial.render_prompt(prompt_path, row) for row in rows}
    row_sha256 = {rid: row_digest(prompts[rid], row) for rid, row in zip(prompts, rows)}
    settings = {"model_id": cfg["model"], "generation_config": gen, "routing": model_routing,
                "prompt_file": prompt_file, "prompt_sha256": sha256(prompt_path.read_bytes()),
                "judge": args.judge, "judge_generation_config": judge_cfg["generation_config"],
                "judge_prompt_sha256": sha256(judge_spec.prompt_path.read_bytes()),
                "judge_schema_sha256": sha256(judge_spec.schema_path.read_bytes())}
    check_resume(out_dir, done_rows, settings, row_sha256)
    todo = [(row, k) for row in rows for k in range(1, args.repeat + 1)
            if trial.episode_id(row["id"], k) not in done]

    (out_dir / "run.json").write_text(json.dumps({
        "task": task.name, "task_root": str(task.root), "variant": task.variant,
        "run_id": run_id, "model": args.model, "model_id": cfg["model"],
        "agent": RUNNER, "runner": "tasks/sycophancy/run_chat.py",
        # The entry's generation_config and extra_body as written, and as sent to
        # the provider SDK (chat_generation_config; extra_body rides under that key,
        # as the judge client sends it).
        "entry_generation_config": cfg["entry_generation_config"],
        "entry_extra_body": cfg["entry_extra_body"] or None,
        "generation_config": gen,
        "generation_config_unsupported": cfg["unsupported"],
        "routing": model_routing, "judge_routing": judge_routing,
        "judge_generation_config": judge_cfg["generation_config"],
        "judge_generation_config_unsupported": judge_cfg["unsupported"],
        "tools": [], "tool_policy": {}, "tool_policy_enforced_by": RUNNER,
        "prompt_file": prompt_file, "prompt_sha256": settings["prompt_sha256"],
        "judge_prompt": str(judge_spec.prompt_path),
        "judge_prompt_sha256": settings["judge_prompt_sha256"],
        "judge_schema": str(judge_spec.schema_path.relative_to(task.root)),
        "judge_schema_sha256": settings["judge_schema_sha256"],
        "judge_fields": list(judge_spec.fields),
        # Per row: the rendered prompt and vars an episode measures, so a resumed run
        # can prove its finished episodes came from the same inputs (check_resume).
        "row_sha256": row_sha256,
        "judge": args.judge, "image": None, "private_net": False,
        "n_rows": len(rows), "repeat": args.repeat, "task_build": build_record,
        "started": datetime.datetime.now().isoformat(),
    }, indent=1))

    print(f"task={task.name}  model={args.model} ({cfg['model']})  agent={RUNNER}\n"
          f"generation_config={gen}\n"
          f"routing: base_url={model_routing['api_base_url'] or 'provider default'}  "
          f"key={model_routing['api_key_env']}\n"
          f"judge routing: base_url={judge_routing['api_base_url'] or 'client default'}  "
          f"key={judge_routing['api_key_env']} ({judge_routing['api_key_env_source']})\n"
          f"tools=[]  sandbox=none\n"
          f"rows={len(rows)}  repeat={args.repeat}  done={len(done)}  "
          f"running={len(todo)}  prompt={prompt_path.name}\n"
          f"out={out_dir}", flush=True)
    if not todo:
        return completed_status(ep_path, judge_path, args.no_judge)

    sem = asyncio.Semaphore(args.max_concurrent)
    lock = asyncio.Lock()
    counters = {"n": 0, "ok": 0, "cost": 0.0, "judged": 0}
    bar = tqdm(total=len(todo), desc=args.model, unit="ep", file=sys.stdout,
               mininterval=2, ascii=True, disable=not args.progress)

    async def one(row: dict, replicate: int) -> None:
        async with sem:
            episode = trial.episode_id(row["id"], replicate)
            trial_dir = out_dir / "trajectories" / episode
            prompt = prompts[row["id"]]
            t0 = time.monotonic()
            rec = {"id": row["id"], "episode": episode, "replicate": replicate,
                   "task": task.name, "model": cfg["model"], "run_id": run_id,
                   "agent": RUNNER, "vars": row.get("vars") or {},
                   "tools": [], "tool_policy": {}, "tool_policy_enforced_by": RUNNER,
                   "prompt_file": prompt_file, "routing": model_routing,
                   "n_turns": 1, "n_tool_calls": 0}
            try:
                # One call, the client as built from the entry, no arguments of ours.
                r = await model.async_completions([{"role": "user", "content": prompt}])
                text = r.content or ""
                u = r.token_usage
                raw = r.raw or {}
                stop = raw.get("stop_reason") or ((raw.get("choices") or [{}])[0] or {}).get("finish_reason")
                path = write_transcript(trial_dir, task.name, row, replicate,
                                        cfg["model"], prompt, prompt_file, text)
                rec.update({"ok": bool(text.strip()), "error": None, "failure": None,
                            "final_text": text, "stop_reason": stop,
                            "wall_time": time.monotonic() - t0,
                            "cost_usd": (u.cost if u else None),
                            "cost_source": "estimated" if u else None,
                            "usage": ({"input_tokens": u.input_tokens,
                                       "output_tokens": u.output_tokens,
                                       "cached_tokens": u.cached_tokens} if u else None),
                            "transcript_path": str(path)})
                if not rec["ok"]:
                    rec["error"] = "empty reply"
            except Exception as e:  # noqa: BLE001
                rec.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}",
                            "failure": "provider_error", "final_text": None, "stop_reason": None,
                            "wall_time": time.monotonic() - t0, "cost_usd": None,
                            "cost_source": None, "usage": None, "transcript_path": None})

            if judge_agent is not None and rec["ok"] and judging.is_judgeable(rec):
                try:
                    tr = json.loads(Path(rec["transcript_path"]).read_text())
                    out = await judging.judge_episode(tr, judge_agent, judge_spec)
                    row_out = {"id": rec["id"], "episode": rec["episode"],
                               "replicate": rec["replicate"], "task": task.name,
                               "model": rec["model"], "judge": args.judge,
                               "parse_ok": out["parse_ok"], **(out["verdict"] or {}),
                               "output_mode": out["output_mode"],
                               "judge_cost_usd": out["judge_cost_usd"],
                               "judge_attempts": out["judge_attempts"],
                               "judge_error": out["judge_error"],
                               "transcript_path": rec["transcript_path"],
                               **({} if out["parse_ok"] else {"raw": out["raw"]})}
                    async with lock:
                        with open(judge_path, "a") as f:
                            f.write(json.dumps(row_out) + "\n")
                    counters["cost"] += float(out["judge_cost_usd"] or 0.0)
                    counters["judged"] += int(bool(out["parse_ok"]))
                except Exception as e:  # noqa: BLE001
                    print(f"JUDGE FAILED on {episode}: {type(e).__name__}: {e}", flush=True)

            async with lock:
                with open(ep_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                counters["n"] += 1
                counters["ok"] += int(bool(rec["ok"]))
                counters["cost"] += float(rec.get("cost_usd") or 0.0)
                bar.update(1)
                bar.set_postfix_str(f"ok {counters['ok']}/{counters['n']} "
                                    f"${counters['cost']:.2f}", refresh=False)

    results = await asyncio.gather(*(one(r, k) for r, k in todo), return_exceptions=True)
    bar.close()
    for (row, k), res in zip(todo, results):
        if isinstance(res, Exception):
            print(f"WARNING: {row['id']}#{k} raised {type(res).__name__}: {res}", flush=True)
    print(f"\n=== summary ===\nepisodes:   {counters['n']}\nno error:   {counters['ok']}\n"
          f"judged:     {counters['judged']}\ncost:       ${counters['cost']:.2f}\n"
          f"episodes -> {ep_path}\nverdicts -> {judge_path}")
    if counters["n"] != len(todo):
        return 1
    return completed_status(ep_path, judge_path, args.no_judge)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", default="sycophancy",
                    help="a folder under tasks/, or a path to one (default: sycophancy)")
    ap.add_argument("--model", required=True,
                    help="a model entry in configs/models.yaml, e.g. claude-fable-5-1 or gpt-6-astra; its settings "
                         "are delivered in the chat API's terms (see the module docstring)")
    ap.add_argument("--variant", default="",
                    help="a tag under the task's `variants:` (all, bubble, forecast, dontcheat, diy); "
                         "unset uses the default")
    ap.add_argument("--prompt", default="", help="a `prompts:` key or a path; unset uses the default")
    ap.add_argument("--judge", default="",
                    help="the judge, a model entry in configs/models.yaml; unset uses the shared default")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="episodes per row")
    ap.add_argument("--no-judge", action="store_true",
                    help="run episodes only; judge later with judge.py")
    ap.add_argument("--redo", action="store_true",
                    help="discard the run directory's finished episodes and start over")
    ap.add_argument("--models-config", default=str(MODELS_CONFIG),
                    help="(default: configs/models.yaml)")
    ap.add_argument("--max-concurrent", type=int, default=12, help="concurrent provider calls")
    ap.add_argument("--output-dir", default=str(OUTPUTS),
                    help="where run directories are written (default: outputs/)")
    ap.add_argument("--run-id", default="",
                    help="defaults to $SLURM_JOB_ID, else a timestamp; repeat one to resume")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
