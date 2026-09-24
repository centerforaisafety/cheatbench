"""Load task configuration and execute one episode in a sandbox.

The host renders the prompt, collects staged files, configures services and
networking, launches the runner, and saves artifacts. The container stages the
files, installs dependencies, and runs the agent. Grading tools are available
during execution; the separate trajectory judge labels behavior afterwards.

Task configuration and data normally supply task-specific behavior. The
MCP-only mode also has explicit task and adapter checks in Task.check_agent().
Setup hooks and enforced egress policies require a private network namespace.

A row's files map selects the material deliberately exposed to the agent.
Optional environment/setup.py hooks run on the host to inject services into
the sandbox. Prompt templates and judge schemas remain on the host. The selected
row, rendered prompt, and file payload travel to the runner over stdin; only
the selected files are staged into the agent's workspace.

The episode lifecycle follows Harbor's trial design; implementation is local.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .task_service import parse_service

from . import review_server as review_server_mod
from . import sandbox, stage
from .config import load_yaml
from .agents import config as agent_config
from .agents import Agent, errors

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"

# Marker for turn records the container streams on stderr, so the host can write
# messages.jsonl / turns.log live instead of only after the episode ends.
MSG_PREFIX = "\x1eRHMSG "

# Grace on top of the episode's own asyncio timeout: the container enforces
# timeout_s itself and still owes us a JSON line afterwards, so the host kill
# switch must fire strictly later.
CONTAINER_GRACE_S = 180

# Big transcript lines (thinking blocks, whole-file tool results) are normal;
# the default 64 KiB StreamReader limit would raise on them.
PIPE_LIMIT = 64 * 1024 * 1024

# How much of a FAILED episode's container stderr is scanned for the reason it
# failed. A tail, and only ever read when the episode already failed.
STDERR_SCAN_BYTES = 256 * 1024


# ==========================================================================
# the task folder
# ==========================================================================
SETUP_HOOK = "setup.py"

# The judge PROMPT is repo-level (`prompts/judge.md`); the only per-task half is
# this module, which exports `Verdict` and `EXTRA_CONTEXT`. `prompts: judge:`
# used to name a per-task rubric and is now an error, not a no-op.
JUDGE_SCHEMA = "judge_schema.py"


@dataclass
class Task:
    root: Path
    name: str
    timeout_s: int = 3600
    prompts: dict = field(default_factory=dict)
    data: str = "data.jsonl"
    # Optional grader module offered as an in-episode tool. An empty path means
    # no module; a configured path must exist. The trajectory judge is separate.
    grader: str = ""
    # A mapping-valued grader config selects a host review endpoint instead of
    # a grader module. It names a judge, prompt, and optional generation settings;
    # a staged client calls the endpoint. grader and review are mutually exclusive.
    review: dict | None = None
    # Optional host service scoped to the run, including automatic retries.
    service: dict | None = None
    # Optional task dependencies, installed after staging and before the agent.
    # Uses the runner's check/install helper with credentials removed from the
    # child environment. Failures are recorded in task_install; the agent still runs.
    install: dict | None = None
    # Image name or path; empty selects the default. --image overrides this.
    # Bare names resolve to $RH_IMAGES_DIR/<name>.sqsh; missing images are built.
    image: str = ""
    # Optional task-owned image recipe; otherwise use the repository Dockerfile.
    image_dockerfile: Path | None = None
    # Adapter-enforced tool capabilities; omitted supported keys default false.
    # Network egress is a separate policy below. MCP-only mode also constrains
    # sandbox setup and is checked by check_agent().
    tools: dict = field(default_factory=dict)
    # Opt-in egress restrictions; tasks without a policy retain open networking.
    network: sandbox.NetworkPolicy = field(default_factory=sandbox.NetworkPolicy)
    # Each variant selects environment/<tag>/ (files and data.jsonl) and a
    # prompt key. Other variant settings belong to the task's build script.
    # --variant overrides variants.default, which otherwise falls back to the
    # first declared tag. Tasks without variants use environment/ directly.
    variants: dict = field(default_factory=dict)
    variant: str = ""
    # Optional restriction for tasks that need adapter-specific runtime hooks.
    agents: list[str] = field(default_factory=list)
    # Version overrides for this task only; model settings remain independent.
    agent_config: dict[str, dict] = field(default_factory=dict)

    def check_agent(self, agent: Agent) -> None:
        if self.tools.get("mcp_only"):
            if self.name not in {"chess", "go"} or not self.network.denies:
                raise SystemExit("MCP-only execution requires network-denied chess or Go")
            supported = agent.name() in {"claude-sdk", "codex", "deepseek-harness", "grok-build", "kimi-code", "terminus-2", "muse-code"}
            supported |= agent.name() == "gemini-cli" and agent.route == "gateway"
            if not supported:
                raise SystemExit("This adapter cannot enforce the native game MCP-only condition")
        if self.agents and agent.name() not in self.agents:
            raise SystemExit(f"{self.name}: supported agents are {self.agents}; "
                             f"{agent.name()!r} cannot provide this task's runtime hooks")

    @property
    def environment(self) -> Path:
        """The task's `environment/` dir, or the selected variant's subdir."""
        base = self.root / "environment"
        return base / self.variant if self.variant else base

    @property
    def setup_hook(self) -> Path:
        # The hook is code, shared by every variant; it receives the selected
        # variant's environment dir as its `env_dir` argument.
        return self.root / "environment" / SETUP_HOOK

    @property
    def default_prompt(self) -> str:
        """The `prompts:` key --prompt falls back to: the variant's, or `instruction`."""
        if self.variant:
            return str(self.variants[self.variant]["prompt"])
        return "instruction"

    @property
    def private_net(self) -> bool:
        """Setup hooks and enforced egress policies need a private namespace."""
        return self.setup_hook.exists() or self.network.enforced

    @property
    def data_path(self) -> Path:
        """`data:` under the task root, or the variant's built data.jsonl."""
        if self.variant:
            return self.environment / "data.jsonl"
        return self.root / self.data

    @property
    def grader_path(self) -> Path | None:
        """Where this task's grader lives, or None when it ships none."""
        return (self.root / self.grader) if self.grader else None

    def grader_src(self) -> str:
        """Read the grader module, or return an empty string if none is configured.

        The source is sent in the runner payload. An empty string consistently means
        that this task provides no grader module.
        """
        path = self.grader_path
        if path is None:
            return ""
        guard = (ROOT / "core/grader_guard.py").read_text()
        try:
            source = path.read_text()
        except OSError as exc:
            raise errors.ReviewerInfrastructureError(f"Cannot load reviewer {path}: {exc}") from exc
        return guard + "\ninstall_grader_guard(" + repr(source) + ", globals())\n"

    @property
    def judge_schema_path(self) -> Path:
        """The task's half of the rubric: `Verdict` + `EXTRA_CONTEXT`."""
        return self.root / JUDGE_SCHEMA

    def prompt_path(self, which: str = "") -> Path:
        """`which` is either a key under `prompts:` or a task-relative path;
        empty means `default_prompt`."""
        which = which or self.default_prompt
        rel = self.prompts.get(which, which)
        return self.root / rel

    def rows(self) -> list:
        """Read dataset rows, which may contain private answers and gold paths.

        Rows are passed to the runner over stdin, not staged as workspace files.
        """
        p = self.data_path
        if not p.exists():
            raise SystemExit(f"{p} not found -- run `python {self.root}/build.py`"
                             + (f" (variant {self.variant!r})" if self.variant else ""))
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        return rows


def load_task(name: str, variant: str = "") -> Task:
    """The task folder, and for a task with `variants:` the selected variant.

    `variant` is a tag under `variants:`; empty selects `variants: default:`
    or the first tag. Naming a variant on a task that declares none, or a tag
    the task does not declare, is a refusal: an unknown variant would run some
    other environment under the requested name.
    """
    root = Path(name) if os.sep in name else TASKS / name
    cfg_path = root / "task.yaml"
    if not cfg_path.exists():
        have = sorted(p.name for p in TASKS.iterdir() if (p / "task.yaml").exists())
        raise SystemExit(f"no task.yaml in {root}; tasks: {have}")
    cfg = load_yaml(cfg_path)
    # `score_rules:` and `match_threshold:` are read by a task's build.py -- the
    # sentences it renders into a variant's rows, and the number it compiles into
    # the program it stages -- and NEVER by the harness. They are allowed here so
    # a task can keep its build-time declarations beside the variants they apply
    # to, rather than in the build script where they would be code.
    unknown = set(cfg) - {"name", "timeout_s", "prompts", "data", "grader",
                          "tools", "image", "image_dockerfile", "variants", "score_rules",
                          "match_threshold", "install", "agents", "network", "agent_config", "build", "service"}
    if unknown:
        raise SystemExit(f"{cfg_path}: unknown key(s) {sorted(unknown)}")
    agents = cfg.get("agents", [])
    if not isinstance(agents, list) or any(not isinstance(a, str) or not a for a in agents):
        raise SystemExit(f"{cfg_path}: agents must be a list of adapter names")
    prompts = dict(cfg.get("prompts") or {})
    variants, variant = _parse_variants(cfg_path, cfg.get("variants"), prompts, variant)
    # The judge prompt is global now. A task that still names one would be
    # silently ignored, and a silently ignored rubric is a wrong measurement, so
    # say so instead.
    if "judge" in prompts:
        raise SystemExit(
            f"{cfg_path}: `prompts: judge:` is gone. The judge prompt is "
            f"repo-level (prompts/judge.md); the per-task half is "
            f"{root / JUDGE_SCHEMA} (Verdict + EXTRA_CONTEXT). Delete the key.")
    # `grader:` is EITHER a module path (the MCP-grader shape, `grader: grader.py`)
    # OR a dict declaring a review endpoint (the `writings` shape). An OMITTED
    # `grader:` means the task ships neither and the episode runs with no grader.
    raw_grader = cfg.get("grader")
    grader = ""
    review = None
    if isinstance(raw_grader, dict):
        review = _parse_review(cfg_path, root, raw_grader)
    elif raw_grader:
        # A task that names a grader MODULE must own the file: a named grader
        # that does not resolve is a refusal here rather than a
        # FileNotFoundError raised per-episode deep inside a run, and it can
        # never degrade into a silently ungraded run.
        grader = str(raw_grader)
        if not (root / grader).exists():
            raise SystemExit(
                f"{cfg_path}: grader: {grader} names {root / grader}, which "
                f"does not exist. Ship the file, or drop the key -- an omitted "
                f"`grader:` means this task offers the agent no grader tool.")
    image_dockerfile = None
    if cfg.get("image_dockerfile"):
        image_dockerfile = (root / str(cfg["image_dockerfile"])).resolve()
        if not image_dockerfile.is_relative_to(root.resolve()) or not image_dockerfile.is_file():
            raise SystemExit(f"{cfg_path}: image_dockerfile must name an existing file inside the task")
    return Task(root=root.resolve(), name=cfg.get("name") or root.name,
                timeout_s=int(cfg.get("timeout_s") or 3600),
                agent_config=agent_config.parse_agents(
                    cfg.get("agent_config", {}), f"{cfg_path}:agent_config"),
                prompts=prompts,
                data=cfg.get("data") or "data.jsonl",
                grader=grader,
                review=review,
                service=parse_service(root, cfg.get("service")),
                install=_parse_install(cfg_path, cfg.get("install")),
                tools=dict(cfg.get("tools") or {}),
                network=sandbox.parse_network(cfg_path, cfg.get("network")),
                image=str(cfg.get("image") or ""),
                image_dockerfile=image_dockerfile,
                variants=variants, variant=variant, agents=agents)


def _parse_install(cfg_path: Path, raw) -> dict | None:
    """`install:` as the `check || install` spec the container helper takes.

    Refused here rather than in the container: a malformed block would
    otherwise be a shell command that silently did nothing, once per episode,
    for a whole run.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise SystemExit(f"{cfg_path}: `install:` must be a block with `check:` "
                         f"and `install:` (and an optional `name:`); got "
                         f"{type(raw).__name__}")
    unknown = set(raw) - {"name", "check", "install"}
    if unknown:
        raise SystemExit(f"{cfg_path}: `install:` has unknown key(s) "
                         f"{sorted(unknown)}; it takes `name`, `check` and "
                         f"`install`")
    for key in ("check", "install"):
        if not str(raw.get(key) or "").strip():
            raise SystemExit(f"{cfg_path}: `install:` needs a non-empty "
                             f"`{key}:` -- the pair is run as `check || install`")
    return {"name": str(raw.get("name") or "task"),
            "check": str(raw["check"]),
            "install": str(raw["install"])}


def _parse_variants(cfg_path: Path, raw, prompts: dict, requested: str) -> tuple:
    """`variants:` into (tag -> entry, selected tag). No block: ({}, "")."""
    if not raw:
        if requested:
            raise SystemExit(f"{cfg_path} declares no `variants:`; "
                             f"--variant {requested!r} names nothing")
        return {}, ""
    if not isinstance(raw, dict):
        raise SystemExit(f"{cfg_path}: `variants:` must be a map of tag -> entry")
    default = raw.get("default")
    variants = {}
    for tag, entry in raw.items():
        if tag == "default":
            continue
        if not isinstance(entry, dict) or not entry.get("prompt"):
            raise SystemExit(f"{cfg_path}: variants: {tag}: needs a `prompt:` key "
                             f"naming an entry under `prompts:`")
        if entry["prompt"] not in prompts:
            raise SystemExit(f"{cfg_path}: variants: {tag}: prompt "
                             f"{entry['prompt']!r} is not under `prompts:` "
                             f"(have {sorted(prompts)})")
        variants[str(tag)] = dict(entry)
    if not variants:
        raise SystemExit(f"{cfg_path}: `variants:` names no variant")
    selected = requested or (str(default) if default else next(iter(variants)))
    if selected not in variants:
        raise SystemExit(f"{cfg_path}: variant {selected!r} is not declared; "
                         f"have: {sorted(variants)}")
    return variants, selected


def _parse_review(cfg_path: Path, root: Path, raw: dict) -> dict:
    """Validate a dict `grader:` into a review-endpoint config.

    `judge` is a name resolved later through `configs/models.yaml`; `prompt` is
    a repository-relative rubric that MUST exist now, so a typo is a refusal here and
    not a per-episode failure. `generation_config` is an optional override
    merged on top of the judge entry's own `generation_config`; it is passed
    through as written.
    """
    # `return_feedback`, `expose_config` and `tie_threshold` are the task
    # BUILD's, not the endpoint's: a task's build.py reads them and bakes them
    # into the grader it compiles, so they are named here only to be allowed
    # beside the endpoint's own keys and are not carried into the config the
    # review server is started with.
    unknown = set(raw) - {"judge", "prompt", "generation_config",
                          "return_feedback", "expose_config", "tie_threshold",
                          "staged_grader"}
    if unknown:
        raise SystemExit(f"{cfg_path}: grader: unknown key(s) {sorted(unknown)}; "
                         f"a review grader takes judge, prompt, "
                         f"generation_config, return_feedback, expose_config, "
                         f"tie_threshold and staged_grader")
    judge = str(raw.get("judge") or "").strip()
    if not judge:
        raise SystemExit(f"{cfg_path}: grader: needs a `judge:` name")
    prompt_rel = str(raw.get("prompt") or "").strip()
    if not prompt_rel:
        raise SystemExit(f"{cfg_path}: grader: needs a `prompt:` path")
    # Resolve the configured rubric relative to the repository root.
    prompt_path = ROOT / prompt_rel
    if not prompt_path.exists():
        raise SystemExit(f"{cfg_path}: grader: prompt {prompt_path} does not "
                         f"exist")
    gen = dict(raw.get("generation_config") or {})
    # Default preserves older tasks whose grader lives under qa/.
    staged = str(raw.get("staged_grader", "qa/grade.pyc") or "").strip()
    if staged and (Path(staged).is_absolute() or ".." in Path(staged).parts):
        raise SystemExit(f"{cfg_path}: grader: staged_grader {staged!r} must be "
                         f"a path inside the workspace")
    return {"judge": judge, "prompt": prompt_rel,
            "generation_config": gen, "staged_grader": staged}


def _price_review(agent: Agent, review: dict, state: dict) -> float | None:
    """Dollars for the review endpoint's model calls, or None.

    Prices every call the reviewer logged in `grader_state` through the same
    `Agent.cost_from_usage` an episode uses, against the judge model. None --
    never 0.0 -- when nothing was priceable, so an unpriced reviewer is never
    summed as if it were free.
    """
    model = review.get("model")
    total = None
    for call in state.get("calls") or []:
        usage = call.get("usage")
        if not usage:
            continue
        one = agent.cost_from_usage(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(usage.get("cached_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
            model_name=model)
        if one is not None:
            total = (total or 0.0) + one
    return total


# ==========================================================================
# prompt rendering
# ==========================================================================
def render_prompt(template: Path, row: dict) -> str:
    """Substitute {{token}} placeholders from the row.

    Fields are the row's `input` plus everything under its `vars` map, which is
    what a task's build.py uses to hand the template per-row strings (where the
    input file landed, which URL the row's post sits at). A line whose
    placeholders ALL render empty is dropped, so an optional block disappears
    without leaving a hole.

    Prompt variants are files, not config: `--prompt prompts/instruction_norm.md`
    selects the norm arm (that file adds the norm sentence). There is no
    {{norm}} tag.
    """
    fields = {"input": row.get("input") or "", **(row.get("vars") or {})}
    out = []
    for line in template.read_text().split("\n"):
        tokens = [k for k in fields if "{{" + k + "}}" in line]
        if tokens and all(not str(fields[k]).strip() for k in tokens):
            continue
        for k, v in fields.items():
            line = line.replace("{{" + k + "}}", str(v))
        out.append(line)
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


# ==========================================================================
# one episode
# ==========================================================================
async def run_trial(task: Task, row: dict, agent: Agent, *, image: str,
                    api_key: str, out_dir: Path, prompt: str = "instruction",
                    env: dict | None = None, timeout_s: int | None = None,
                    replicate: int = 1, review: dict | None = None,
                    service_health=None) -> dict:
    """Run one episode of `task` on `row` and return its record.

    `replicate` is the repeat index. A row run N times produces N independent
    episodes keyed `<row_id>#<k>`, each with its own trajectory directory, so
    replicates never overwrite each other and per-row variance is recoverable.
    """
    task.check_agent(agent)
    tid = row["id"]
    episode = episode_id(tid, replicate)
    trial_dir = out_dir / "trajectories" / episode
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Build from the task-owned recipe when configured. Its cache receipt
    # prevents an older image at the same path from bypassing the build.
    await sandbox.ensure_image_async(image, **({"dockerfile": task.image_dockerfile}
                                             if task.image_dockerfile else {}))

    timeout_s = int(timeout_s or task.timeout_s)
    prompt_path = task.prompt_path(prompt)
    text = render_prompt(prompt_path, row)
    (trial_dir / "prompt.txt").write_text(text)

    # STAGED: the row's files map, the only task material the agent may see.
    files = stage.collect(row, task.environment)

    # REVIEW ENDPOINT: a `writings`-style grader. The host orchestrator binds a
    # loopback endpoint (holding the credential and the judge's prompt) and
    # writes its URL and a per-episode bearer token into `.review` BESIDE THE
    # STAGED GRADER, which the compiled grader reads and POSTs to. Started here
    # so it is up before the container is, torn down once the agent has exited.
    # See core/review_server.py.
    review_server = None
    if review is not None:
        gold_name = (row.get("meta") or {}).get("gold_name") or ""
        gold_path = task.environment / "files" / "golds" / gold_name
        if not gold_path.is_file():
            raise SystemExit(
                f"{task.name}: row {tid!r} names gold {gold_name!r} but "
                f"{gold_path} is missing -- run `python {task.root}/build.py`")
        config = {**review,
                  "reference_sha256": review_server_mod.norm_digest(
                      gold_path.read_text())}
        staged_grader = str(review.get("staged_grader", "qa/grade.pyc") or "")
        if staged_grader:
            staged_names = {str(f.get("name") or "") for f in files}
            if staged_grader not in staged_names:
                raise SystemExit(
                    f"{task.name}: task.yaml declares `grader: staged_grader: "
                    f"{staged_grader}` but row {tid!r} stages no such file "
                    f"(it stages {sorted(staged_names)}). `.review` would go "
                    f"somewhere the grader never looks and every call would "
                    f"report the reviewer unavailable -- fix the declaration or "
                    f"re-run `python {task.root}/build.py`")
        review_name = review_file_name(staged_grader)
        try:
            review_server = review_server_mod.start_review_server(config)
        except Exception as exc:
            raise errors.ReviewerInfrastructureError(f"Reviewer startup failed: {exc}") from exc
        files = list(files) + [{
            "name": review_name,
            "b64": base64.b64encode(json.dumps(
                {"url": review_server.url,
                 "token": review_server.token}).encode()).decode()}]

    skip_dirs = tuple({d["name"].split("/")[0] for d in files
                       if "/" in d["name"] and not d.get("abs")})

    # The image is the TASK ENVIRONMENT and carries no agent binary: baking one
    # vendor's CLI into it would mean a second image the day we run Codex. The
    # adapter says how to get its own runtime instead, and the blob carries that
    # in -- `check || install`, run by the container after staging and before
    # the agent starts, so a pre-baked image is a cache and never a requirement.
    #
    # It necessarily runs AFTER the task's setup hook too, and that is not
    # luck: the container is blocked reading its blob from stdin for as long as
    # the hook is running, so nothing inside can install anything until the hook
    # has finished and start_network() has given the episode a network to
    # install over.
    install = agent.install()

    # The container environment: the adapter's PASSTHROUGH_ENV copy from
    # run.py, with the model entry's `api_base_url:` applied on top under the
    # names the adapter's CLI reads (core/agents/base.py `routed_env`). An entry
    # that routes nothing leaves the copy exactly as it was.
    env = agent.routed_env(env or {})
    blob = agent.blob(row=row, prompt=text, files=files, skip_dirs=skip_dirs,
                      grader_src=task.grader_src(),
                      stage_src=(ROOT / "core" / "stage.py").read_text(),
                      timeout_s=timeout_s, env=env or {}, install=install,
                      task_install=task.install)

    msgs_path = trial_dir / "messages.jsonl"
    log_path = trial_dir / "turns.log"
    err_path = trial_dir / "container.stderr.log"

    t0 = time.time()
    error: str | None = None
    # WHY it failed, as a type. `error` keeps the raw sentence; this is the
    # thing analysis groups by, so an episode the harness lost (context
    # exhausted, wall clock, a dead container) is never averaged in with one
    # where the model REFUSED. See core/agents/errors.py.
    failure: type[errors.AgentError] | None = None
    payload: dict | None = None
    streamed: list = []

    private_net = task.private_net
    home_dir = tempfile.mkdtemp(prefix="rh_home_")

    try:
        # The credential's VARIABLE NAME comes from the adapter: Claude's
        # episodes need ANTHROPIC_API_KEY inside and Codex's need
        # OPENAI_API_KEY, and neither should have the other vendor's key in its
        # environ for the agent to read out of /proc.
        # ... and the NAME(S) it is exported under inside the container are
        # the adapter's `container_key_envs()`: the host variable the model
        # entry's `api_key_env:` named may differ from what the CLI reads.
        argv = sandbox.container_argv(image, agent.bootstrap,
                                      private_net=private_net,
                                      key_env=agent.container_key_envs())
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=PIPE_LIMIT,
            cwd=str(trial_dir),
            env=sandbox.spawn_env(api_key, key_env=agent.container_key_envs()),
        )
    except Exception as e:  # noqa: BLE001
        proc = None
        error = f"container launch failed: {type(e).__name__}: {e}"
        failure = errors.ContainerLaunchError

    # INJECTED: the task's setup hook runs BEFORE the episode does. The
    # container is already running but blocked reading its task from stdin, so
    # this window is free: nothing inside can touch the network until we feed it.
    sb = None
    guard = None
    if proc is not None and private_net:
        sb = sandbox.EpisodeSandbox(proc.pid, ("python", "-c", agent.bootstrap),
                                    trial_dir, python_exe=sys.executable)
        try:
            await asyncio.to_thread(run_setup_hook, task, row, sb)
            # A hook that injected nothing still needs a working network: a
            # private namespace starts empty. Idempotent if the hook attached
            # one already.
            await asyncio.to_thread(sb.start_network)
            bridge_ports = ()
            if task.tools.get("mcp_only"):
                if not task.network.enforced:
                    raise ValueError("MCP-only tasks must declare an explicit deny or allow network policy")
                blob, bridge_port = await asyncio.to_thread(
                    sb.route_mcp_only, blob, agent.resolved_base_url(), api_key)
                bridge_ports = (bridge_port,)
            if task.network.enforced:
                allow = sandbox.hosts_for(task.network, agent.resolved_base_url(),
                                          agent.INSTALL_HOSTS)
                loopback = ((review_server.server_address[1],)
                            if review_server is not None else ())
                loopback = tuple(dict.fromkeys((*loopback, *bridge_ports,
                    *sandbox.api_loopback_ports(agent.resolved_base_url()))))
                guard = await asyncio.to_thread(sb.lock_egress, allow, loopback)
            with open(err_path, "a") as ef:
                if task.setup_hook.exists():
                    ef.write(f"[host] {task.setup_hook.name} ran against container "
                             f"pid {sb.container_pid}\n")
                if guard is not None:
                    ef.write(f"[host] egress {task.network.egress}: "
                             f"{len(allow)} host(s) allowed\n")
        except (Exception, asyncio.CancelledError) as e:  # noqa: BLE001
            # Never fall through. A task with a setup hook is a task whose
            # measurement depends on what the hook put there; without it the
            # episode would reach the real internet and record a result we
            # cannot reproduce.
            error = f"sandbox setup failed: {type(e).__name__}: {e}"
            failure = (errors.BatchCancelledError if isinstance(e, asyncio.CancelledError)
                       else errors.SetupHookError)
            sb.close()
            sb = None
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            proc = None

    out = b""
    if proc is not None:
        async def _feed() -> None:
            try:
                proc.stdin.write(blob)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass

        async def _read_stdout() -> bytes:
            return await proc.stdout.read()

        async def _pump_stderr() -> None:
            """Tee the container's stderr: turn records to the run's artefacts,
            everything else to container.stderr.log."""
            while True:
                try:
                    line = await proc.stderr.readline()
                except Exception as e:  # noqa: BLE001
                    with open(err_path, "a") as ef:
                        ef.write(f"[host] stderr read failed: {e}\n")
                    return
                if not line:
                    return
                txt = line.decode("utf-8", "replace").rstrip("\n")
                if txt.startswith("\x1eRHGRADER_FATAL "):
                    detail = json.loads(txt[len("\x1eRHGRADER_FATAL "):])["error"]
                    with open(err_path, "a") as ef:
                        ef.write(f"[host] FATAL reviewer infrastructure: {detail}\n")
                    raise errors.ReviewerInfrastructureError(detail)
                if txt.startswith(MSG_PREFIX):
                    try:
                        rec = json.loads(txt[len(MSG_PREFIX):])
                    except json.JSONDecodeError:
                        continue
                    streamed.append(rec)
                    with open(msgs_path, "a") as mf:
                        mf.write(json.dumps(rec) + "\n")
                    block = agent.readable(rec)
                    if block.strip():
                        with open(log_path, "a") as lf:
                            lf.write(block + "\n\n")
                elif txt.strip():
                    with open(err_path, "a") as ef:
                        ef.write(txt + "\n")

        async def _watch_reviewer():
            while proc.returncode is None:
                if sb is not None:
                    try:
                        sb.check_health()
                    except Exception as exc:
                        raise errors.ReviewerInfrastructureError(str(exc)) from exc
                if service_health is not None:
                    service_health()
                if review_server is not None and review_server.state.get("errors"):
                    raise errors.ReviewerInfrastructureError(
                        str(review_server.state["errors"][-1]))
                await asyncio.sleep(0.25)

        io_tasks = [asyncio.create_task(coro) for coro in
                    (_feed(), _read_stdout(), _pump_stderr(), _watch_reviewer())]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*io_tasks),
                timeout=timeout_s + CONTAINER_GRACE_S)
            out = results[1] or b""
            rc = await proc.wait()
            if rc != 0 and not out.strip():
                error = f"container exited {rc}"
                failure = errors.ContainerExitError
        except (errors.ReviewerInfrastructureError, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                error, failure = "batch aborted; episode interrupted", errors.BatchCancelledError
            else:
                error, failure = str(exc), errors.ReviewerInfrastructureError
            for io_task in io_tasks:
                io_task.cancel()
            await asyncio.gather(*io_tasks, return_exceptions=True)
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.transport.abort()
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.communicate()
        except (asyncio.TimeoutError, TimeoutError):
            error = f"timeout after {timeout_s + CONTAINER_GRACE_S}s (container killed)"
            failure = errors.EpisodeTimeoutError
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        except Exception as e:  # noqa: BLE001 - one bad episode must not kill the run
            error = f"{type(e).__name__}: {e}"
            failure = errors.HarnessError
            try:
                proc.kill()
                await proc.wait()
            except Exception:  # noqa: BLE001
                pass

    if sb is not None:
        # Everything the hook started dies with the episode: the namespace it
        # lives in is gone anyway once the container exits.
        sb.close()
    shutil.rmtree(home_dir, ignore_errors=True)

    # The reviewer has no more work once the agent has exited: capture its state
    # and price its calls before tearing it down. `grader_state` for a review
    # task is this, not the runner's (the runner hosts no grader module here).
    grader_state_override = None
    grader_cost_usd = None
    if review_server is not None:
        grader_state_override = dict(review_server.state)
        grader_cost_usd = _price_review(agent, review, grader_state_override)
        review_server.close()

    if error is None or out.strip():
        # The record is the last line on stdout; the runner guarantees stdout
        # carries nothing else, but be forgiving about trailing newlines.
        lines = [ln for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
        if not lines:
            if error is None:
                error, failure = ("container produced no record on stdout",
                                  errors.NoRecordError)
        else:
            try:
                payload = json.loads(lines[-1])
            except json.JSONDecodeError as e:
                if error is None:
                    error, failure = (f"unparseable container stdout: {e}",
                                      errors.UnparseableRecordError)

    traj = agent.record(payload or {})
    # A killed container still leaves whatever it streamed on stderr.
    if not traj["messages"]:
        traj["messages"] = streamed
    if traj.get("error") and failure not in (errors.ReviewerInfrastructureError,
                                             errors.BatchCancelledError):
        # The container's own account of the failure supersedes whatever the
        # host inferred from the outside: it saw the thing that actually
        # happened, and the host only saw the process end.
        error, failure = traj["error"], None
    if error is None:
        # The CLI exited 0 and reported nothing, but that is not always the
        # same as an episode. See `lost_on_a_clean_exit`.
        lost = lost_on_a_clean_exit(agent, traj, err_path)
        if lost is not None:
            error, failure = lost
    failure = classify_failure(agent, error, failure, traj, err_path)
    if service_health is not None:
        try:
            service_health()
        except errors.ReviewerInfrastructureError as exc:
            error, failure = str(exc), errors.ReviewerInfrastructureError
    reviewer_errors = [str(e) for server in traj.get("init_mcp_servers") or []
                       for e in server.get("errors") or []]
    reviewer_errors.extend((grader_state_override or {}).get("errors") or [])
    if reviewer_errors:
        error, failure = str(reviewer_errors[-1]), errors.ReviewerInfrastructureError
    if error is None:
        nothing = grader_served_nothing(review, grader_state_override, traj)
        if nothing is not None:
            error, failure = nothing
    wall = traj.get("wall_time") or round(time.time() - t0, 1)

    # The agent's produced office files, shipped back on stdout by the container.
    # The container overlay is gone by now, so this is the only surviving copy.
    deliverables_dir = None
    if traj["deliverables"]:
        import base64 as _b64

        deliverables_dir = trial_dir / "deliverables"
        for d in traj["deliverables"]:
            try:
                dst = deliverables_dir / d["name"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(_b64.b64decode(d["b64"]))
            except Exception:  # noqa: BLE001 - export is best effort
                pass

    transcript = {"id": tid, "episode": episode, "replicate": replicate,
                  "task": task.name, "variant": task.variant, "model": agent.model,
                  "agent": agent.name(), "tools": agent.tools,
                  "tool_policy": agent.policy, "prompt": text,
                  "prompt_file": str(prompt_path.relative_to(task.root)),
                  "grader_state": (grader_state_override if review is not None
                                   else traj["grader_state"]),
                  # The adapter's NATIVE record, kept verbatim. It is the raw
                  # evidence, and it is what makes a historical run re-judgeable
                  # after the converter changes: `trajectory.json` beside it is
                  # derived, and can always be rebuilt from this.
                  "messages": traj["messages"]}
    (trial_dir / "transcript.json").write_text(json.dumps(transcript, indent=1))

    # The same episode as ATIF -- the interchange format, and the thing the
    # judge is actually measured on. Written next to the native record rather
    # than instead of it: a conversion bug should be visible and fixable, which
    # it would not be if the only surviving copy were the converted one.
    try:
        (trial_dir / "trajectory.json").write_text(json.dumps(
            agent.to_trajectory(transcript).to_json_dict(), indent=1))
    except Exception as e:  # noqa: BLE001 - never lose an episode to conversion
        (trial_dir / "trajectory.error.txt").write_text(
            f"{type(e).__name__}: {e}\n")

    return {
        # The row this episode came from, and which replicate of it. Group by
        # `id` to get the per-row spread across replicates.
        "id": tid,
        "episode": episode,
        "replicate": replicate,
        "task": task.name,
        "variant": task.variant,
        "model": agent.model,
        "agent": agent.name(),
        # The agent build that actually produced this episode, reported by the
        # adapter's own VERSION_CMD inside the container after
        # `check || install`. A pinned run and an unpinned one both record it:
        # the reproducibility property is not that the version is fixed, it is
        # that the finished run says what it was.
        "agent_version": agent.version(),
        # WHERE the model was reached and under WHICH credential variable, as
        # resolved from the model entry (host-only URL, names not values). An
        # episode record must be able to say which gateway produced it.
        "routing": agent.resolved_routing(),
        # The resolved policy, the tool list it produced, and who held it.
        # Recorded on every episode so a number can never be read as
        # closed-book when it was not.
        "tools": agent.tools,
        "tool_policy": agent.policy,
        "tool_policy_enforced_by": agent.name(),
        "prompt_file": str(prompt_path.relative_to(task.root)),
        # How this episode was sandboxed, and with what. Recorded so a result
        # can never be misread later.
        "private_net": private_net,
        "setup_hook": task.setup_hook.name if task.setup_hook.exists() else None,
        "egress": _egress_record(task, guard),
        # What the adapter's runtime install did inside the container:
        # {"status": "present"|"installed"|"failed", "seconds": ...}, or None
        # for an adapter that needs nothing. Recorded so a cold-start cost is
        # measurable and a failed install is never read as a model failure.
        "install": traj.get("install"),
        # And what the TASK's own `install:` block did, on the same terms and
        # kept apart from the adapter's: a task's missing library is a note on
        # the episode, never the reason no agent ran.
        "task_install": traj.get("task_install"),
        "grader_state": (grader_state_override if review is not None
                         else traj["grader_state"]),
        # What the review endpoint's model calls priced out at, through the same
        # `cost_from_usage` every episode uses. None for a task with no review
        # grader, or when the judge model is absent from the pricing table.
        "grader_cost_usd": grader_cost_usd,
        "final_text": traj["final_text"],
        "n_turns": traj["n_turns"],
        "n_tool_calls": traj["n_tool_calls"],
        "wall_time": wall,
        "cost_usd": traj["cost_usd"],
        # Where `cost_usd` came from: "reported" (the vendor priced the episode
        # itself -- Claude's ResultMessage), "estimated" (we priced the recorded
        # tokens through litellm -- every Codex episode, and any Claude episode
        # the host killed before its ResultMessage arrived), or None when
        # neither was possible. Recorded so the two are never summed or compared
        # as if they were the same measurement.
        "cost_source": traj.get("cost_source"),
        "stream_usage": traj.get("stream_usage"),
        "usage": traj.get("usage"),
        "result_subtype": traj.get("result_subtype"),
        "terminal_reason": traj.get("terminal_reason"),
        "session_id": traj.get("session_id"),
        "init_tools": traj.get("init_tools"),
        "init_mcp_servers": traj.get("init_mcp_servers"),
        "deliverables_dir": str(deliverables_dir) if deliverables_dir else None,
        "transcript_path": str(trial_dir / "transcript.json"),
        # Whatever else the adapter says belongs on the record. The fields above
        # are the ones every adapter has; this is the vendor-specific tail (for
        # Codex, the exact `codex exec` line the episode ran and the rollout it
        # was read back from), named by the adapter so `core/` still knows
        # nothing about any particular one.
        **{k: traj.get(k) for k in agent.EXTRA_RECORD_KEYS},
        "error": error,
        # The same failure as a stable tag: "context_window_exceeded",
        # "safety_refusal", "timeout", ... None on a successful episode, which
        # is never classified at all. See core/agents/errors.py.
        "failure": errors.reason(failure),
        "ok": error is None,
    }


def classify_failure(agent: Agent, error: str | None,
                     failure: type[errors.AgentError] | None,
                     traj: dict, err_path) -> type[errors.AgentError] | None:
    """The typed reason this episode failed, or None if it did not.

    Four inputs, in order of authority:

      * a failed INSTALL, which is structural rather than textual and is always
        the root cause -- no agent ever ran, so everything after it is fallout;
      * the harness's own WALL CLOCK: an error that opens with the wording a
        runner uses when it SIGKILLed the CLI at `timeout_s` means the episode
        was still going when we stopped it. That is a deterministic finding
        about the agent (it ran out of time), so it is not up for revision by
        text -- least of all by a gateway 5xx the CLI survived earlier in the
        hour, which the rule below would otherwise pick because it lands later
        in stderr than the timeout line does in the record error. This is what
        keeps a timed-out episode out of `TRANSIENT_REASONS` and therefore
        judgeable rather than silently re-run;
      * whatever else the harness already knows (`failure`), which is definitive
        for the points where the episode died outside the container;
      * what the agent PRINTED: the record's error, plus the container's stderr,
        which is where a CLI puts the thing that actually killed it while the
        record often carries only the exception that followed. Consulted only
        when the process ended on its own, since the two rules above return
        first.

    The guard is the first line: a successful episode is never looked at, so no
    pattern here can ever change what a successful run reports.

    One last, narrow rule after those three: a proxy 502 that arrives MID-episode
    crashes the Claude SDK's message reader, which surfaces only as a generic
    `Command failed with exit code 1` -- the reader died before a ResultMessage
    could carry the 502 onto the record error, so the classifier above sees
    nothing to go on and files it as unclassified. But the 502 body IS in the
    last assistant text (`final_text`). So when, and ONLY when, the failure has
    that reader-crash shape AND the last assistant text is itself a gateway 5xx,
    call it what it is: `proxy_5xx`, transient, and the auto-resume re-runs it.
    Scoped to the reader-crash shape so a legitimate answer that merely quotes a
    5xx is never swept in.
    """
    if error is None:
        return None
    install = traj.get("install")
    if isinstance(install, dict) and install.get("status") == "failed":
        return errors.AgentInstallError
    if str(error).startswith("max_turns reached before confirmed task completion"):
        return errors.MaxTurnsError
    if errors.is_harness_timeout(error):
        # We killed a running CLI. Nothing it printed on the way down, and
        # nothing an adapter reads off its terminal event, describes the
        # episode better than that.
        return errors.EpisodeTimeoutError
    stderr = ""
    try:
        # The TAIL: a CLI's boilerplate comes first and its actual failure last,
        # which is the same reason Harbor's own truncation keeps the end.
        stderr = err_path.read_text(errors="replace")[-STDERR_SCAN_BYTES:]
    except OSError:
        pass
    result = agent.classify_failure(failure, error, stderr)
    # Only when the classifier landed on nothing definitive -- None, the
    # unclassified base, or the generic `unknown_api_error` -- do we consult the
    # last assistant text, and only for the reader-crash shape. A real reason
    # (a refusal, a context overflow) is never second-guessed by the tail.
    if (result in (None, errors.AgentError, errors.UnknownApiError)
            and _is_reader_crash(error, stderr)):
        final_text = (traj or {}).get("final_text") or ""
        if errors.classify(final_text) is errors.ProxyGatewayError:
            return errors.ProxyGatewayError
    return result


# How much of the container's log the clean-exit scan below looks at: the LAST
# few lines, which is where a CLI puts the thing it stopped on. Deliberately far
# smaller than STDERR_SCAN_BYTES -- that scan reads the tail of an episode we
# already know failed, and can afford to search it; this one is looking at an
# episode that reported success, so it may only consider the CLI's final word.
CLEAN_EXIT_SCAN_LINES = 20


def review_file_name(staged_grader: str) -> str:
    """Place the endpoint configuration beside the task's staged grader."""
    return str(Path(staged_grader or ".").parent / ".review")


def grader_served_nothing(review: dict | None, state: dict | None,
                          traj: dict) -> tuple[str, type[errors.AgentError]] | None:
    """Reject an invoked reviewer that served nothing; preserve unused graders.

    Match the wrapper's JSON outage, not the feedback string alone: inspecting
    bytecode can reveal that string without an actual grader outage.
    """
    if review is None or not isinstance(state, dict):
        return None
    if int(state.get("reviews_used") or 0) > 0:
        return None
    if state.get("calls") or state.get("errors"):
        return ("the review endpoint was invoked but served no verdict",
                errors.GraderUnreachableError)
    # Tool outputs can contain JSON nested inside other JSON strings.
    blob = json.dumps(traj.get("messages") or [], default=str)
    blob += str(traj.get("final_text") or "")
    marks = (review_server_mod.UNAVAILABLE_NULL_VERDICT,
             review_server_mod.UNAVAILABLE_FEEDBACK)
    for _ in range(4):
        if all(mark in blob for mark in marks):
            return ("the grader reported the reviewer unavailable and no review "
                    "was served; check endpoint staging and connectivity",
                    errors.GraderUnreachableError)
        peeled = blob.replace('\\"', '"')
        if peeled == blob:
            break
        blob = peeled
    return None


def lost_on_a_clean_exit(agent: Agent, traj: dict,
                         err_path) -> tuple[str, type[errors.AgentError]] | None:
    """An episode that reported success but never got one, or None.

    THE case this exists for. `codex exec` can hit a fatal API error mid-episode
    -- our gateway answering `400 ... Budget has been exceeded` once the key is
    at its dollar cap -- print it, and then exit 0. The runner sets `error` only
    on a launch failure, its own timeout, or a non-zero exit with no rollout, so
    none of those fire: the record comes back `ok: true` with the failure
    visible nowhere but the log. Three pelican episodes were filed as clean runs
    that way. An `ok: true` record is retired by `load_done`, judged by
    `is_judgeable` and counted in the denominator, so a lost episode wearing a
    clean record is the most expensive shape a failure can take.

    The rule, and it is deliberately narrow, because this is the ONE place that
    may contradict an episode that reported success:

      * the episode produced NO final text. An agent that delivered an answer
        finished its turn; whatever the log says afterwards did not stop it from
        arriving, and this function must not touch it. This is the condition
        that keeps a survived 502 -- retried, recovered, twenty turns before the
        end -- from relabelling a good episode.
      * the LAST few lines of the log classify, through the same taxonomy every
        other failure goes through, to a LOST class. Not a finding: a refusal or
        a context overflow reached through this path would be a guess about an
        episode nobody reported as failed, and those arrive on the record
        properly when they happen. Only "the harness lost this" is inferable
        from a log tail.

    Returns the matched text and its class, for the caller to put on the record
    exactly as a runner-reported error, so the tag, the resume and the judge
    gate all behave as if the CLI had said so itself.
    """
    if (traj.get("final_text") or "").strip():
        return None
    try:
        tail = err_path.read_text(errors="replace")[-STDERR_SCAN_BYTES:]
    except OSError:
        return None
    lines = [ln for ln in tail.splitlines() if ln.strip()][-CLEAN_EXIT_SCAN_LINES:]
    if not lines:
        return None
    last = "\n".join(lines)
    found = errors.classify(last, patterns=agent.ERROR_PATTERNS)
    if found is None or not errors.is_lost(found):
        return None
    return (f"{agent.name()} exited 0 after: {last.splitlines()[-1].strip()}",
            found)


# The Claude SDK's message-reader subprocess dying: a generic wrapper with no
# provider detail of its own, so it is a signal to look at the last assistant
# text rather than a classification in itself.
_READER_CRASH = re.compile(
    r"Command failed with exit code 1|Fatal error in message reader",
    re.IGNORECASE)


def _is_reader_crash(*texts: str | None) -> bool:
    return any(t and _READER_CRASH.search(t) for t in texts)


def episode_id(row_id, replicate: int) -> str:
    """The key one episode is stored under: `<row_id>#<k>`, k from 1."""
    return f"{row_id}#{int(replicate)}"


def _egress_record(task: Task, guard) -> dict:
    """What the episode was allowed to reach, and what it tried to.

    `guard` is None only when the container never started, in which case the
    policy is still worth recording and there is nothing to report against it.
    """
    out = {"policy": task.network.egress,
           "declared_hosts": list(task.network.hosts),
           "hosts": [], "reached": [], "refused": []}
    if guard is None:
        return out
    out["hosts"] = list(guard.hosts)
    for entry in guard.attempts():
        host = entry.get("host") or "?"
        bucket = "reached" if entry.get("allowed") else "refused"
        if host not in out[bucket]:
            out[bucket].append(host)
    return out


def run_setup_hook(task: Task, row: dict, sb) -> None:
    """Call `environment/setup.py:setup(sandbox, row, env_dir)` if it exists.

    The hook is host-side task code. It is given a handle on the RUNNING
    episode's namespaces and the task's own `environment/` directory, and it
    decides for itself what to inject and what the row selects -- the row's own
    `id` is the natural selector, so no row needs a config key for it.

    `core/` does not know or care what the hook serves.
    """
    hook = task.setup_hook
    if not hook.exists():
        return
    spec = importlib.util.spec_from_file_location(
        f"rh_setup_{task.name}", hook)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {hook}")
    mod = importlib.util.module_from_spec(spec)
    # The hook may import siblings out of its own directory.
    sys.path.insert(0, str(hook.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(str(hook.parent))
        except ValueError:
            pass
    fn = getattr(mod, "setup", None)
    if not callable(fn):
        raise RuntimeError(f"{hook} defines no setup(sandbox, row, env_dir)")
    fn(sb, row, task.environment)


# ==========================================================================
# live turn log
#
# `Agent.readable` converts ONE streamed record to ATIF and renders it with the
# same renderer the judge's feed uses, so what a human watches during a run
# cannot drift from what the measurement is taken on. See core/render.py.
# ==========================================================================
