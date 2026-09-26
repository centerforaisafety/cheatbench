#!/usr/bin/env python3
"""Run a task's episodes and optionally judge their trajectories.

    python run.py knowledge_work --model gpt-6-sol
    python run.py knowledge_work --model gpt-6-sol --harness terminus-2
    python run.py openmath --model claude-opus-5 --limit 3

Each dataset row runs in its own container. --repeat N creates independent
episodes keyed <row_id>#<k>. Records append to episodes.jsonl. Reusing a run ID
skips episode keys with a successful record, unless --redo is set; other keys
can run again. Transient failures can also receive one automatic retry.

--max-concurrent limits active episodes, each of which runs a CLI subprocess.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import judge as judging          # noqa: E402
from core import review_server as review_server_mod  # noqa: E402
from core import routing                   # noqa: E402
from core.config import load_yaml, load_models          # noqa: E402
from core import task_service              # noqa: E402
from core import task_build                # noqa: E402
from core import sandbox, trial            # noqa: E402
from core.agents import AgentFactory, make_agent  # noqa: E402
from core.agents import config as agent_config  # noqa: E402
from core.agents import errors as agent_errors    # noqa: E402

try:  # .env before argv defaults are computed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MODELS_CONFIG = ROOT / "configs" / "models.yaml"
OUTPUTS = ROOT / "outputs"

# Compatibility fallback for archived configs without a model harness.
DEFAULT_AGENT = "claude-sdk"


def apply_tool_policy(task, agent) -> list:
    """Validate and apply the task's tools block before launching episodes.

    The adapter translates policy keys into vendor controls. Unsupported keys stop
    the run because provider-hosted tools can bypass container network isolation.
    """
    try:
        return agent.apply_tool_policy(task.tools)
    except Exception as e:  # noqa: BLE001 - turn it into a clear refusal
        raise SystemExit(
            f"\ntask '{task.name}' sets `tools:` in "
            f"{task.root / 'task.yaml'} that the '{agent.name()}' adapter "
            f"cannot honour:\n  {e}\n\nA tool the adapter cannot hold shut "
            f"runs outside the container's network namespace,\nso this task's "
            f"measurement could not be trusted. Use an adapter that can "
            f"enforce\nit, or change the task.\n")


def load_config(path: Path, model: str) -> dict:
    """Load shared model settings and its default harness; CLI can override it."""
    cfg = load_yaml(path)
    models = load_models(path)
    if model not in models:
        raise SystemExit(f"model '{model}' not in {path}; have: {sorted(models)}")
    from core.model_settings import model_entry
    entry = model_entry(path, model)
    runtime_keys = {"max_turns", "permission_mode"}
    if runtime_keys & (set(cfg.get("defaults") or {}) | set(entry)):
        raise SystemExit("max_turns and permission_mode are runner arguments; remove them from models.yaml")
    return {
        "model": entry.get("model") or model,
        "generation_config": dict(entry.get("generation_config") or {}),
        "harness": entry.get("harness", {}),
        # Optional per-model routing. Both default to None: an entry that omits
        # them runs on the adapter's class defaults. They let one model reach a
        # different provider (base_url) with a different credential
        # (api_key_env) without touching the judge or grader -- see
        # core/agents/base.py.
        "base_url": entry.get("base_url"),
        # Per-model API routing (core/routing.py): `api_key_env` is the HOST
        # variable NAME (None = the adapter's default, announced once);
        # `api_base_url` is interpolated here, so `${OPENAI_BASE_URL}` naming
        # an unset variable fails at load, not after an hour of containers;
        # `extra_body` is a dict, {} when absent.
        **routing.resolve(model, entry, where=str(path)),
    }


def load_done(path: Path) -> set:
    """Episode keys with any successful record; these are skipped on resume."""
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
        if rec.get("ok"):
            done.add(rec.get("episode") or rec["id"])
    return done


def resolve_review(task: trial.Task, models_config: Path) -> dict:
    """The review-endpoint config for a `writings`-style grader.

    Resolves the task's `grader: judge:` through the SAME `configs/models.yaml`
    resolver `core/judge.py` uses, and reads the pairwise prompt. The credential
    is NOT resolved here: `core/review_server.py` builds the judge through
    `core/llm_agents.py`, the one LLM client, which reads the proxy key from the
    host environment -- so the key lives only in the host orchestrator and the
    container never holds it. The judge entry's `generation_config` is passed
    through as written, with the task's `grader: generation_config:` merged on
    top; nothing is added or filtered here.
    """
    review = task.review
    jc = judging.load_judge_config(models_config, review["judge"])
    gen = dict(jc.get("generation_config") or {})
    gen.update(review.get("generation_config") or {})

    prompt_path = trial.ROOT / review["prompt"]
    parts = review_server_mod.parse_prompt(prompt_path.read_text())
    for section in ("system", "user_template"):
        if not parts.get(section):
            raise SystemExit(f"{prompt_path} has no `# {section}` section")

    return {"judge": review["judge"], "model": jc.get("factory_model", jc["model"]),
            "generation_config": gen,
            # Keep the declared path when resolving reviewer routing.
            "staged_grader": review.get("staged_grader", "qa/grade.pyc"),
            # The judge entry's routing, for core/review_server.py's client.
            "api_key_env": jc.get("api_key_env"),
            "api_base_url": jc.get("api_base_url"),
            "extra_body": jc.get("extra_body"),
            "system": parts["system"], "user_template": parts["user_template"]}


def load_all_records(path: Path) -> list[dict]:
    """Read every valid JSON record in order, retaining all attempt costs."""
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def select_transient_retries(records: list[dict],
                             handled: set | None = None) -> list[dict]:
    """Select episodes whose latest record is a transient failure.

    When supplied, handled restricts selection to this invocation's episode keys.
    Successful records, refusals, context/turn limits, and timeouts are excluded.
    API budget exhaustion is lost but not transient: it needs a limit change before
    retrying. Manual resume is separate and may rerun any key not in load_done().
    """
    final: dict = {}
    for rec in records:
        key = rec.get("episode") or rec.get("id")
        if key is None:
            continue
        final[key] = rec
    out: list[dict] = []
    for key, rec in final.items():
        if handled is not None and key not in handled:
            continue
        if rec.get("ok"):
            continue
        if agent_errors.is_transient(rec.get("failure")):
            out.append(rec)
    return out


async def main_async(args: argparse.Namespace) -> int:
    from tqdm.asyncio import tqdm

    task = trial.load_task(args.task, args.variant)
    models_config = Path(args.models_config)
    if not args.model:
        raise SystemExit("--model is required")
    cfg = load_config(models_config, args.model)
    # Resolve the judge from the same model entries before starting episodes.
    args.judge = args.judge or judging.DEFAULT_JUDGE
    judge_cfg = judging.load_judge_config(models_config, args.judge)
    # Explicit harness overrides only the default harness, never the model route.
    agent_source = "--harness" if args.agent else ("models.yaml:harness" if cfg.get("harness") else "default")
    agent_name = args.agent or cfg.get("harness", {}).get("name") or DEFAULT_AGENT
    version_selection = agent_config.resolve_version(
        agent_name, cli_version=args.agent_version, harness_override=bool(args.agent),
        model_harness=cfg.get("harness"), models_path=models_config)
    print(f"agent: adapter={agent_name} requested_version={version_selection.version} "
          f"source={version_selection.source} config={version_selection.config_path}",
          flush=True)
    try:
        rows, build_record = task_build.ensure_task_built(
            task,
            auto_build=not args.no_auto_build, rebuild=args.rebuild_task)
    except task_build.TaskBuildError as exc:
        raise SystemExit(str(exc)) from exc

    run_id = args.run_id or os.environ.get("SLURM_JOB_ID") or \
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # A variant task's run dir carries the tag, so two variants of one task
    # never share a directory.
    run_name = f"{task.name}_{task.variant}" if task.variant else task.name
    out_dir = Path(args.output_dir) / f"{run_name}_{args.model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "episodes.jsonl"
    judge_path = out_dir / "judge.jsonl"

    done = set() if args.redo else load_done(results_path)
    previous_agent_version = (agent_config.resume_version(
        out_dir / "run.json", agent_name, version_selection.version) if done else None)
    if args.limit:
        rows = rows[: args.limit]
    # Replicates are just more work items: they queue through --max-concurrent
    # like anything else.
    todo = [(r, k) for r in rows for k in range(1, args.repeat + 1)
            if trial.episode_id(r["id"], k) not in done]

    from core.model_settings import harness_settings
    generation, extra_body = harness_settings(cfg, agent_name)
    max_turns = getattr(args, "max_turns", 150)
    agent = make_agent(agent_name, model=cfg["model"],
                       generation_config=generation,
                       max_turns=max_turns or None,
                       permission_mode=getattr(args, "permission_mode", None),
                       version=version_selection.version,
                       base_url=cfg["base_url"],
                       api_key_env=cfg["api_key_env"],
                       api_base_url=cfg["api_base_url"],
                       extra_body=extra_body)
    if max_turns and not agent.TURN_LIMIT:
        print(f"WARNING: {agent_name} cannot enforce --max-turns={max_turns}; "
              "the episode timeout still applies. run.json records this limit as unenforced.", flush=True)
    # Before anything else: an adapter that cannot hold the task's tool policy
    # must not run a single episode.
    task.check_agent(agent)
    tools = apply_tool_policy(task, agent)

    sandbox.ensure_path()
    problems = agent.setup() + sandbox.preflight(private_net=task.private_net)
    if problems:
        raise SystemExit("cannot run:\n  " + "\n  ".join(problems))

    # Which image, in order: --image, then the task's `image:` key, then the
    # default. A missing image is not an error -- the sandbox builds it from the
    # task Dockerfile (or the repo default). Doing it here rather than only in run_trial means one
    # build before the progress bar instead of a first episode that looks hung.
    image = args.image or sandbox.resolve_image(task.image)
    await sandbox.ensure_image_async(image, rebuild=args.rebuild_image,
                                     **({"dockerfile": task.image_dockerfile}
                                        if task.image_dockerfile else {}))

    # Which credential: the HOST variable the model entry's `api_key_env:`
    # names, else the adapter's own default -- announced once, so an old entry
    # keeps working and the run log says which variable was assumed. A named
    # variable that is unset is a refusal naming the entry, here, before any
    # container. Which extra variables are the ADAPTER's business: the Claude
    # adapter needs ANTHROPIC_API_KEY inside the container and the Codex
    # adapter needs OPENAI_API_KEY, and neither should be carrying the other
    # vendor's key where the agent can read it out of /proc.
    routing.default_key_env(args.model, cfg["api_key_env"],
                            consumer=f"'{agent.name()}' adapter",
                            fallback=agent.api_key_env)
    key = routing.require_key(agent.api_key_env, name=args.model,
                              consumer=f"'{agent.name()}' adapter")
    child_env = {k: os.environ[k] for k in agent.PASSTHROUGH_ENV
                 if os.environ.get(k)}
    agent_routing = agent.resolved_routing()
    judge_routing = judging.judge_routing_record(args.judge, judge_cfg)

    args.prompt = args.prompt or task.default_prompt
    prompt_path = task.prompt_path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"prompt {prompt_path} not found")

    # A review-endpoint grader (writings): resolve its judge, prompt and proxy
    # credential up front, so a misconfigured grader fails before any episode.
    review_config = resolve_review(task, models_config) if task.review else None

    # The rubric's two halves: the repo-level prompt and the task's schema.
    judge_spec = judging.load_judge_spec(task.root, task_name=task.name)
    judge_agent = None
    if not args.no_judge:
        try:
            judge_agent = judging.make_judge_agent(args.judge, models_config)
        except Exception as e:  # noqa: BLE001
            print(f"judge disabled: {type(e).__name__}: {e}")

    (out_dir / "run.json").write_text(json.dumps({
        "task": task.name, "task_root": str(task.root), "variant": task.variant,
        "run_id": run_id,
        "model": args.model, "model_id": cfg["model"], "agent": agent.name(),
        "agent_selected_by": agent_source,
        # HOW the model was asked to generate, as RESOLVED by the adapter that
        # ran it -- adapter defaults included, not a re-read of the yaml. There
        # was previously no artefact of this anywhere: whether extended thinking
        # was even switched on could only be inferred from the transcript, and
        # codex's own rollout echoes `summary: auto` whatever was asked for. A
        # finished run must be able to state its own generation settings.
        "generation_config": agent.resolved_generation_config(),
        # WHERE the model was reached and under WHICH credential variable, as
        # resolved (core/routing.py): the model id, the base URL host-only,
        # the key variable's NAME on the host and inside the container, where
        # each came from, and the extra_body verbatim. Names, never values. A
        # finished run must say which gateway produced it.
        "routing": agent_routing,
        "judge_routing": judge_routing,
        "max_turns_requested": max_turns,
        "max_turns": agent.max_turns if agent.TURN_LIMIT else None,
        "max_turns_unit": agent.TURN_LIMIT,
        "max_turns_enforced": bool(max_turns and agent.TURN_LIMIT),
        "permission_mode": agent.permission_mode,
        "entry_generation_config": cfg["generation_config"],
        "harness_config": cfg.get("harness", {}),
        # Record parameters sent and any unsupported reasoning-display setting.
        "judge_generation_config": judge_cfg["generation_config"],
        "judge_generation_config_unsupported":
            judge_cfg["unsupported"],
        # The adapter's resolved toolset, the requirement it is holding, and
        # who holds it. A result can never be misread as closed-book when it
        # was not.
        "tools": tools,
        "tool_policy": agent.policy,
        "tool_policy_enforced_by": agent.name(),
        "prompt_file": str(prompt_path.relative_to(task.root)),
        "judge_prompt": str(judge_spec.prompt_path),
        "judge_schema": str(judge_spec.schema_path.relative_to(task.root)),
        "judge_fields": list(judge_spec.fields),
        "task_build": build_record,
        "service": task.service,
        "judge": args.judge, "image": image,
        # The agent CLI version. `requested` is the pin the operator asked for
        # after CLI/model selection; `agent_version` is what ran, filled in
        # from the first episode that reports one, because it is not knowable
        # until a container has been inside the image. A finished run always
        # states which agent build produced it.
        "agent_version_requested": version_selection.version,
        "agent_version_source": version_selection.source,
        "agent_version_config": version_selection.config_path,
        "agent_version_config_sha256": version_selection.config_sha256,
        "agent_version": previous_agent_version,
        "private_net": task.private_net,
        "setup_hook": (str(task.setup_hook.relative_to(task.root))
                       if task.private_net else None),
        "n_rows": len(rows), "repeat": args.repeat,
        "timeout_s": args.timeout or task.timeout_s,
        "started": datetime.datetime.now().isoformat(),
    }, indent=1))

    print(f"task={task.name}  model={args.model} ({cfg['model']})  "
          f"agent={agent.name()} (from {agent_source})\n"
          f"generation_config={agent.resolved_generation_config()}\n"
          f"routing: base_url={agent_routing['api_base_url'] or 'provider default'} "
          f"({agent_routing['api_base_url_source']})  "
          f"key={agent_routing['api_key_env']} ({agent_routing['api_key_env_source']})"
          f" -> container {agent_routing['container_key_env']}  "
          f"forwarder={'on' if agent_routing['forwarder'] else 'off'}\n"
          f"judge routing: base_url={judge_routing['api_base_url'] or 'client default'}  "
          f"key={judge_routing['api_key_env']} ({judge_routing['api_key_env_source']})\n"
          f"tool_policy={agent.policy} (enforced by {agent.name()})\n"
          f"tools={tools}\n"
          f"sandbox={'private netns + ' + task.setup_hook.name if task.private_net else 'host netns, no setup hook'}  "
          f"image={image}\n"
          f"rows={len(rows)}  repeat={args.repeat}  done={len(done)}  "
          f"running={len(todo)}  prompt={prompt_path.name}\n"
          f"out={out_dir}")
    if not todo:
        return 0

    run_json = out_dir / "run.json"

    def _record_agent_version(version: str | None) -> None:
        """Patch the resolved agent version into run.json, once.

        It cannot be written up front: the version comes from `claude --version`
        inside the container, so the first finished episode is the earliest
        anything on the host knows it. Idempotent, and never fatal -- a run.json
        that could not be updated must not lose an episode.
        """
        if not version:
            return
        try:
            data = json.loads(run_json.read_text())
            if data.get("agent_version") == version:
                return
            data["agent_version"] = version
            run_json.write_text(json.dumps(data, indent=1))
            print(f"agent: adapter={agent.name()} installed_version={version} "
                  f"requested_version={version_selection.version}", flush=True)
        except Exception:  # noqa: BLE001
            pass

    sem = asyncio.Semaphore(args.max_concurrent)
    stop_batch = asyncio.Event()
    lock = asyncio.Lock()
    counters = {"n": 0, "ok": 0, "cost": 0.0, "judged": 0}
    # Failed episodes by typed reason, so a run that lost ten episodes says
    # whether it lost them to the context window or to a model that refused.
    failures: dict[str, int] = {}
    bar = None

    async def one(row: dict, replicate: int) -> None:
        async with sem:
            if stop_batch.is_set():
                return
            try:
                episode_row = copy.deepcopy(row)
                with task_service.open_service(
                        task, rows=[episode_row], out_dir=out_dir,
                        load_model=lambda name: load_config(models_config, name)) as active_service:
                    rec = await trial.run_trial(
                        task, episode_row, agent, image=image, api_key=key, out_dir=out_dir,
                        prompt=args.prompt, env=child_env,
                        timeout_s=args.timeout or None, replicate=replicate,
                        review=review_config,
                        service_health=getattr(active_service, "check_health", None))
            except Exception as e:  # noqa: BLE001
                rec = {"id": row["id"],
                       "episode": trial.episode_id(row["id"], replicate),
                       "replicate": replicate,
                       "task": task.name, "model": cfg["model"],
                       "tool_policy": agent.policy,
                       "ok": False, "error": f"{type(e).__name__}: {e}",
                       # An exception THIS side of run_trial: the episode never
                       # got far enough to have a reason of its own.
                       "failure": agent_errors.reason(
                           agent_errors.ReviewerInfrastructureError
                           if isinstance(e, agent_errors.ReviewerInfrastructureError)
                           else agent_errors.HarnessError)}
            rec["run_id"] = run_id
            # A requested pin is not proof that installation succeeded. Use
            # the version command's actual output for the installed-version log.
            _record_agent_version((rec.get("install") or {}).get("version"))

            # `is_judgeable` and not just `transcript_path`: an episode the
            # harness LOST -- a transient fault, or a key at its budget cap --
            # still leaves a PARTIAL transcript, and the auto-retry below is
            # about to re-run it and append a second record. Judging both writes
            # two verdict rows for one episode key. See core/judge.py.
            if judge_agent is not None and judging.is_judgeable(rec):
                try:
                    tr = json.loads(Path(rec["transcript_path"]).read_text())
                    out = await judging.judge_episode(tr, judge_agent, judge_spec)
                    row_out = {"id": rec["id"], "episode": rec["episode"],
                               "replicate": rec["replicate"], "task": task.name,
                               "model": rec.get("model"), "judge": args.judge,
                               "parse_ok": out["parse_ok"],
                               **(out["verdict"] or {}),
                               # How this verdict was produced. There is only
                               # one way -- API-enforced structured output --
                               # and the row says so, so it can never be
                               # confused with a pre-refactor scraped row.
                               "output_mode": out["output_mode"],
                               "judge_cost_usd": out["judge_cost_usd"],
                               # How many times the structured-output call was
                               # made: a transient gateway fault is retried
                               # inside judge_episode, so >1 means it was flaky
                               # but recovered.
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
                    # The episode itself is expensive and already finished, so a
                    # judge failure must not throw it away. The judge already
                    # retried a transient gateway fault up to JUDGE_MAX_ATTEMPTS
                    # inside judge_episode; only now, having exhausted that, do
                    # we record the failure. It is never silent: printed here,
                    # noted on the episode record, AND written as a parse_ok=False
                    # verdict row (so no episode is silently skipped) that carries
                    # the real attempt count. `load_judged` keys on parse_ok, so
                    # `python judge.py <run>` still re-judges exactly these.
                    rec["judge_error"] = f"{type(e).__name__}: {e}"
                    attempts = getattr(e, "attempts", 1)
                    fail_row = {"id": rec["id"], "episode": rec["episode"],
                                "replicate": rec["replicate"], "task": task.name,
                                "model": rec.get("model"), "judge": args.judge,
                                "parse_ok": False, "output_mode": None,
                                "judge_cost_usd": 0.0,
                                "judge_attempts": attempts,
                                "judge_error": rec["judge_error"],
                                "transcript_path": rec["transcript_path"]}
                    async with lock:
                        with open(judge_path, "a") as f:
                            f.write(json.dumps(fail_row) + "\n")
                    print(f"JUDGE FAILED on {rec['episode']} after "
                          f"{attempts} attempt(s): {rec['judge_error']}",
                          flush=True)

            async with lock:
                with open(results_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                counters["n"] += 1
                counters["ok"] += int(bool(rec.get("ok")))
                if not rec.get("ok"):
                    tag = rec.get("failure") or "unclassified"
                    failures[tag] = failures.get(tag, 0) + 1
                counters["cost"] += float(rec.get("cost_usd") or 0.0)
                if bar is not None:
                    bar.set_description(
                        f"ok {counters['ok']}/{counters['n']} | "
                        f"judged {counters['judged']} | ${counters['cost']:.2f}")
                    bar.update(1)

            if rec.get("failure") in {"reviewer_infrastructure", "grader_unreachable"}:
                stop_batch.set()
                raise agent_errors.ReviewerInfrastructureError(
                    f"{rec['episode']}: {rec.get('error')}. "
                    f"Attempt saved to {results_path}; batch stopped. Fix the reviewer before resuming.")

    async def run_batch(items):
        jobs = [asyncio.create_task(one(r, k)) for r, k in items]
        try:
            await asyncio.gather(*jobs)
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            bar.close()

    handled = {trial.episode_id(r["id"], k) for r, k in todo}
    row_by_id = {r["id"]: r for r in rows}

    retried = 0
    bar = tqdm(total=len(todo), desc="starting", unit="ep",
               disable=not args.progress, file=sys.stdout, mininterval=5,
               ascii=True)
    await run_batch(todo)

    # Retry each transient failure once after the initial batch. The retry adds
    # another episode record, retaining the earlier attempt's cost. Refusals,
    # context/turn limits, timeouts, and API budget exhaustion are excluded from
    # this automatic pass. Manual resume uses load_done() instead.
    if not args.no_auto_retry:
        retry_recs = select_transient_retries(
            load_all_records(results_path), handled)
        retry_items = [(row_by_id[rc["id"]], rc["replicate"])
                       for rc in retry_recs if rc["id"] in row_by_id]
        retried = len(retry_items)
        if retry_items:
            reasons = sorted({rc.get("failure") for rc in retry_recs})
            print(f"\nauto-retry: re-running {retried} transient failure(s) "
                  f"{reasons} once (--no-auto-retry to disable)", flush=True)
            bar = tqdm(total=retried, desc="retry", unit="ep",
                       disable=not args.progress, file=sys.stdout,
                       mininterval=5, ascii=True)
            await run_batch(retry_items)

    # Summarize the latest record for each episode handled in this invocation.
    # Costs include every attempt made here, including automatic retries.
    final_recs: dict = {}
    for rec in load_all_records(results_path):
        key = rec.get("episode") or rec.get("id")
        if key in handled:
            final_recs[key] = rec
    ok_final = sum(1 for r in final_recs.values() if r.get("ok"))
    failures_final: dict[str, int] = {}
    for r in final_recs.values():
        if not r.get("ok"):
            tag = r.get("failure") or "unclassified"
            failures_final[tag] = failures_final.get(tag, 0) + 1
    judged_final: set = set()
    if judge_path.exists():
        for line in judge_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                jr = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = jr.get("episode") or jr.get("id")
            if key in handled and jr.get("parse_ok"):
                judged_final.add(key)

    print("\n=== summary ===")
    print(f"episodes:   {len(final_recs)}")
    print(f"no error:   {ok_final}")
    for tag, k in sorted(failures_final.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  failed - {tag}: {k}")
    if retried:
        print(f"retried {retried} transient failure(s)")
    print(f"judged:     {len(judged_final)}")
    print(f"cost:       ${counters['cost']:.2f}")
    print(f"episodes -> {results_path}")
    if judged_final:
        print(f"verdicts -> {judge_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="a folder under tasks/, or a path to one")
    ap.add_argument("--model", required=True,
                    help="a top-level model name in configs/models.yaml")
    ap.add_argument("--variant", default="",
                    help="a tag under the task's `variants:`; unset uses "
                         "`variants: default:` (or the first tag). A task "
                         "without `variants:` takes none")
    ap.add_argument("--prompt", default="",
                    help="a key under the task's `prompts:` or a task-relative "
                         "path, e.g. prompts/instruction_norm.md; unset uses "
                         "the variant's prompt, or `instruction`")
    ap.add_argument("--harness", "--agent", dest="agent", default="",
                    help="override the model's harness and use its latest release (Terminus-2 stays pinned); choices: " + ", ".join(AgentFactory.names()))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="run every selected row N times, as N independent "
                         "episodes keyed <row_id>#<k>. --limit 6 --repeat 5 is "
                         "30 episodes. For run-to-run variance.")
    ap.add_argument("--redo", action="store_true",
                    help="rerun rows that already have a good record")
    ap.add_argument("--no-judge", action="store_true",
                    help="run episodes only; judge later with judge.py")
    ap.add_argument("--no-auto-retry", action="store_true",
                    help="do not re-run episodes that failed with a transient "
                         "gateway/transport fault (proxy 5xx, dropped stream, "
                         "rate limit) at the end of the run. By default those "
                         "are retried once; the original failed record and its "
                         "cost are kept.")
    ap.add_argument("--judge", default="",
                    help="model name from models.yaml (same entries as --model); "
                         f"default: {judging.DEFAULT_JUDGE}")
    ap.add_argument("--harness-version", "--agent-version", dest="agent_version", default="",
                    help="override the agent CLI version with an exact version "
                         "or latest; otherwise use the model pin, or latest with --harness")
    ap.add_argument("--image", default="",
                    help="sandbox image: a name resolved under $RH_IMAGES_DIR, "
                         "or a path. Default is the task's `image:` key, else "
                         f"{sandbox.DEFAULT_IMAGE}. Built from ./Dockerfile if "
                         "it is not there.")
    build_options = ap.add_mutually_exclusive_group()
    build_options.add_argument("--rebuild-task", action="store_true",
                               help="rebuild task inputs before running, even when present")
    build_options.add_argument("--no-auto-build", action="store_true",
                               help="fail if task inputs are missing instead of running build.py")
    ap.add_argument("--rebuild-image", action="store_true",
                    help="rebuild the image from ./Dockerfile before running, "
                         "even if it is already there")
    ap.add_argument("--models-config", default=str(MODELS_CONFIG))
    ap.add_argument("--max-concurrent", type=int, default=4,
                    help="each episode is a CLI subprocess -- keep this low")
    ap.add_argument("--max-turns", type=int, default=150,
                    help="native harness turn/step limit (default: 150; 0 = harness default); Codex cannot enforce this")
    ap.add_argument("--permission-mode", default=None,
                    help="optional harness-specific permission mode; default is unattended execution")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-episode wall clock; 0 = task.yaml timeout_s")
    ap.add_argument("--output-dir", default=str(OUTPUTS))
    ap.add_argument("--run-id", default="",
                    help="defaults to $SLURM_JOB_ID, else a timestamp")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction,
                    default=True)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.max_turns < 0:
        raise SystemExit("--max-turns must be >= 0")
    try:
        return asyncio.run(main_async(args))
    except agent_errors.ReviewerInfrastructureError as exc:
        print(f"FATAL: reviewer infrastructure failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
