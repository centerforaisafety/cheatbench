"""The Moonshot Kimi Code CLI adapter.

One adapter, one module -- Harbor's layout. MODELLED ON HARBOR:
`harbor/src/harbor/agents/installed/kimi_code.py` (the install, the
`KIMI_MODEL_*` environment configuration, the `mcp.json` registration and the
`kimi --prompt ... --output-format stream-json` invocation are Harbor's), with
the trajectory conversion verified against kimi-code 0.43.0 journals and
streamed responses. Its in-container runner is
the sibling `kimi_code_runner.py`, read as source and streamed into the
container on stdin; never imported here, never written to the container
filesystem.
"""
from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

from ..trajectory import (
    SCHEMA_VERSION,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from ..trajectory import Agent as ATIFAgent
from . import errors
from .errors import ErrorPattern
from .installed import NODE_MAJOR, NVM_PRELUDE, InstalledAgent, link_bins, node_install


class KimiCodeAgent(InstalledAgent):
    """Moonshot's Kimi Code CLI (`@moonshot-ai/kimi-code`), driven by
    `kimi_code_runner.py`.

    Everything about how the CLI is installed, configured and invoked is
    Harbor's, reshaped to what this sandbox does differently (see `CodexAgent`
    for the install and no-exec-channel story, which is identical here). Three
    things are this adapter's own:

      * EVERY model request is routed through a loopback forwarder in the
        runner -- `core/agents/forwarder.py`'s `Upstream` for the trip to the
        gateway -- whether or not the model entry carries `extra_body`. With
        `extra_body` the forwarder is what delivers the routing pin; without
        it, it is still the only place the model's REASONING can be read:
        kimi-code's stream-json writer discards thinking deltas and its
        `wire.jsonl` journal records no `think` parts (measured on 0.42.0, on
        calls that billed hundreds of reasoning tokens). So the runner taps
        every response and `to_trajectory` joins the reasoning to the CLI's
        own steps by response id.
      * The CLI never holds the credential. `KIMI_MODEL_API_KEY` is a
        placeholder and the real key is removed from the CLI's environment;
        the forwarder in the runner process injects it.
      * The file of record is the CLI's `wire.jsonl` journal, not stdout. The
        runner returns every journal's records (main agent first, subagents
        after) inside the one JSON line it owes, plus one record per
        forwarded call.
    """
    TURN_LIMIT = 'model steps per user turn'

    @staticmethod
    def name() -> str:
        return "kimi-code"

    # ---- what this adapter can hold shut ----------------------------------
    #
    # Both keys, and they are INDEPENDENT here: Kimi Code has two separate
    # tools, `WebSearch` and `FetchURL`, and `$KIMI_CODE_HOME/config.toml`
    # takes
    #
    #     [tools]
    #     disabled = ["FetchURL", "WebSearch"]
    #
    # Established empirically against kimi-code 0.42.0 by asking the model to
    # enumerate its own tools (the same check the Codex row of the table in
    # core/agents/base.py was passed with):
    #
    #   no [tools] block          Agent, AgentSwarm, AskUserQuestion, Bash,
    #                             CreateGoal, Cron*, Edit, EnterPlanMode,
    #                             ExitPlanMode, FetchURL, GetGoal, Glob, Grep,
    #                             Read, ReadMediaFile, SetGoalBudget, Skill,
    #                             Task*, TodoList, UpdateGoal, WaitFor, Write
    #   disabled = [FetchURL,     the same list with FetchURL and
    #     WebSearch,              AskUserQuestion GONE; WebSearch was never
    #     AskUserQuestion]        offered (see below)
    #
    # Two facts about the tools themselves, from the CLI's source:
    #
    #   WebSearch  is registered only `when hasWebSearchProvider()`, and a
    #              provider exists only with a `services.moonshot_search`
    #              config or a Moonshot OAuth login. Env-only model config
    #              (what this adapter uses) has neither, so the tool is absent
    #              structurally; the `disabled` entry is belt and braces. It
    #              also means `web_search: true` CANNOT be honoured -- there is
    #              no provider to search with -- and `_tools_for` refuses it
    #              rather than recording a tool the episode never had.
    #   FetchURL   resolves `services.moonshot_fetch` -> managed OAuth ->
    #              `LocalFetchURLProvider`, an in-process HTTP fetch. Under
    #              env-only config it is the LOCAL one: the request leaves
    #              from inside the container's network namespace, the same
    #              place the agent's own `curl` leaves from. It is still
    #              disabled under `web_fetch: false`, because the policy names
    #              the tool and not its transport.
    #
    # What has NOT been done: the mirror-log canary for FetchURL itself (the
    # tool is disabled on every shipped task, so no episode has exercised it
    # in-namespace). The shell path is what the openmath smoke exercises.
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}
    MCP_TOOLS = ("play_move", "game_status", "reset_game", "write_notes", "read_notes",
                 "api_request", "list_files", "read_file", "write_file")

    # The gateway pair, as for codex: all traffic goes through the org's
    # litellm proxy, whose key is OPENAI_API_KEY and whose origin is
    # OPENAI_BASE_URL. The model entry may name a different host variable
    # (`api_key_env:`); it arrives in the container under that same name
    # (CLI_KEY_ENV is empty on purpose: the CLI must NOT read it -- the runner
    # hands the CLI a placeholder and strips the real one from its environment).
    API_KEY_ENV = "OPENAI_API_KEY"
    CLI_KEY_ENV = ()
    # The task's grader, running beside the CLI in the runner process, reads
    # the proxy origin from OPENAI_BASE_URL; the CLI itself is pointed at the
    # forwarder, which forwards to `resolved_base_url()`.
    CLI_BASE_URL_ENV = ("OPENAI_BASE_URL",)
    PASSTHROUGH_ENV = ("OPENAI_BASE_URL",)
    SUPPORTS_FORWARDER = True

    # The exact `kimi` line, the journals it was read back from, the forwarder's
    # per-call summaries, the routing probe and the config.toml the policy
    # became. A result that could not say which of these produced it would not
    # be reproducible.
    EXTRA_RECORD_KEYS = ("kimi_command", "wire_paths", "forwarder_calls",
                         "forwarder_probe", "config_toml",
                         "thinking_effort_bound", "terminal_error")

    # Kimi Code's own words for how a turn dies, ahead of the shared list.
    # All of these were read out of `turn.ended.error` in the wire journal and
    # off the CLI's stderr on real episodes.
    ERROR_PATTERNS = [
        # The CLI's fatal stderr line. Anything more specific below or in the
        # shared list ends later in the text and therefore wins; this only
        # keeps a provider fault from being left `unclassified` when nothing
        # else matched.
        ErrorPattern(r"failed to run prompt: provider\.\w+", errors.UnknownApiError),
        # `turn.ended.error.code`, the CLI's structured verdict.
        ErrorPattern(r"provider\.rate_limit|APIProviderRateLimitError",
                     errors.ApiRateLimitError),
        ErrorPattern(r"provider\.context_length_exceeded",
                     errors.ContextWindowExceededError),
        ErrorPattern(r"provider\.authentication|APIAuthenticationError",
                     errors.AgentAuthenticationError),
        # The CLI stopping itself at `[loop_control] max_steps_per_turn`.
        ErrorPattern(r"max_steps_per_turn|step limit reached", errors.MaxTurnsError),
        *errors.ERROR_PATTERNS,
    ]

    # The runner's own verdict on the episode, which it now derives from the
    # CLI's `turn.ended.error` rather than from the stdout tail.
    _RUNNER_VERDICT = re.compile(r"^kimi (?:exited -?\d+|ended the turn )")

    def classify_failure(self, current, *texts):
        """The CLI's terminal event outranks the stderr tail.

        The base implementation scans the record's `error` and the container's
        stderr TOGETHER and lets the match furthest toward the end win. That
        rule assumes the end of stderr is the end of the episode, and for this
        adapter it is not:

          * the CLI's `--output-format stream-json` stdout is teed into the
            same stream, so every `turn.step.retrying` event for a 429 the CLI
            then RETRIED SUCCESSFULLY is in the text being classified;
          * the runner's own `[runner] forwarder: ...` summaries are written
            after the CLI is already dead;
          * an episode the run-level auto-retry re-ran shares one trajectory
            directory with its retry, so the tail of the file can belong to a
            LATER, successful attempt entirely.

        On the four kimi episodes killed by a gateway 524 that is exactly what
        happened: `rate_limit`, from a 429 that had already recovered. So when
        the runner's `error` is its own verdict -- `kimi exited N: ...`, built
        from `turn.ended.error` -- and that verdict names something specific,
        it wins. Everything else falls through to the base rule unchanged, and
        a definitive harness reason still outranks both.
        """
        result = super().classify_failure(current, *texts)
        if result is current and current is not None:
            return result
        record_error = texts[0] if texts else None
        if record_error and self._RUNNER_VERDICT.match(str(record_error)):
            specific = errors.classify(record_error,
                                       patterns=self.ERROR_PATTERNS)
            if specific not in (None, errors.AgentError, errors.UnknownApiError):
                return specific
        return result

    # ---- what an episode cost ---------------------------------------------
    #
    # Nothing in the CLI's output prices an episode, and the gateway's streamed
    # usage carries no `cost` field (the non-streaming probe's does, but that
    # is the probe's own spend). `reported_cost_usd` is therefore None and the
    # base class's litellm estimate is the primary source, keyed on the model
    # id (`openrouter/moonshotai/kimi-k3` is in litellm's table at $3/M in,
    # $15/M out, $0.30/M cache read -- the rates OpenRouter billed on the
    # probe: 88 in + 149 out = $0.002499).
    #
    # `usage` is the forwarder's sum over every call the CLI made, in the
    # gateway's OpenAI shape, where `prompt_tokens` INCLUDES the cache legs
    # (measured: prompt_tokens 19747 = the CLI's own inputOther 19491 +
    # inputCacheRead 256). `stream_usage` is the CLI's own `step.end` tally
    # folded into the same four names, as a fallback when the forwarder saw
    # nothing.
    USAGE_FIELDS = ("usage", "stream_usage")
    USAGE_KEYS = ("prompt_tokens", "completion_tokens",
                  "cached_tokens", "cache_write_tokens")
    PROMPT_TOKENS_INCLUDE_CACHE = True

    def reported_cost_usd(self, raw: dict) -> float | None:
        """None, always: neither the CLI nor the streamed gateway response
        prices the episode."""
        return None

    # ---- this adapter's runtime -------------------------------------------
    #
    # Node 22 via nvm and the `@moonshot-ai/kimi-code` npm package, exactly as
    # Harbor installs them (Harbor uses `--prefix "$HOME/.local"`; we install
    # under nvm's global prefix and link the binaries, as the Codex adapter
    # does, so the same helper serves both).
    PACKAGE = "@moonshot-ai/kimi-code"
    NODE_MAJOR = NODE_MAJOR

    # `kimi --version` prints the bare version on its last line (`0.42.0`).
    VERSION_CMD = (NVM_PRELUDE + "kimi --version 2>/dev/null | tail -1 | "
                   "awk '{print $NF}'")
    INSTALL_CHECK = NVM_PRELUDE + "command -v kimi >/dev/null 2>&1"

    # Where the CLI keeps its state inside the container: config.toml,
    # mcp.json and the session journals all live under this one directory,
    # which the runner removes when the episode ends.
    KIMI_HOME = "/tmp/kimi-home"

    # `KIMI_MODEL_PROVIDER_TYPE`. The CLI accepts kimi|anthropic|openai for an
    # env-configured model; `kimi` is Moonshot's own dialect of the OpenAI chat
    # API (it sends `thinking: {type: enabled, ...}` and reads
    # `reasoning_content`), and is what the vendor's own model expects. Both
    # `kimi` and `openai` were measured working through the gateway.
    PROVIDER_TYPE = "kimi"

    # The placeholder the CLI is given: it refuses to start without a
    # KIMI_MODEL_API_KEY, and never sees the real one.
    PLACEHOLDER_KEY = "rh-forwarder-placeholder"

    # Environment the CLI runs with, every episode. Telemetry and auto-update
    # both talk to Moonshot's servers from outside anything this sandbox
    # observes; NO_COLOR keeps its stderr parseable.
    CLI_ENV = {"KIMI_DISABLE_TELEMETRY": "1",
               "KIMI_CODE_NO_AUTO_UPDATE": "1",
               "NO_COLOR": "1",
               "KIMI_MODEL_PROVIDER_TYPE": PROVIDER_TYPE}

    # THIS adapter's delivery of the model's `generation_config:`.
    # `thinking_effort` -> KIMI_MODEL_THINKING_EFFORT, which the CLI puts on
    # every request as `thinking: {type: "enabled", effort: <value>}` (measured
    # with `high`: the request body carried `"thinking": {"type": "enabled",
    # "effort": "high", "keep": "all"}` and the gateway accepted it). Unset,
    # the CLI's own default is `on` (thinking enabled, no effort field). No
    # default is supplied here: run.json records what WE asked for, and the
    # journal's `profile.bind.thinkingEffort` (surfaced on the record as
    # `thinking_effort_bound`) records what the CLI ran with.
    GENERATION_KEYS = ("thinking_effort",)

    # Kimi Code's own toolset under env-only configuration, as the CLI's
    # `profile.bind.activeToolNames` lists it and the model enumerated it,
    # minus the two web tools the policy switches.
    BASE_TOOLS = ("Agent", "AgentSwarm", "AskUserQuestion", "Bash", "CreateGoal",
                  "CronCreate", "CronDelete", "CronList", "Edit", "EnterPlanMode",
                  "ExitPlanMode", "GetGoal", "Glob", "Grep", "Read",
                  "ReadMediaFile", "SetGoalBudget", "Skill", "TaskList",
                  "TaskOutput", "TaskStop", "TodoList", "UpdateGoal", "WaitFor",
                  "Write")
    WEB_SEARCH_TOOL = "WebSearch"
    WEB_FETCH_TOOL = "FetchURL"

    RUNNER = Path(__file__).resolve().parent / "kimi_code_runner.py"

    # ---- install ----------------------------------------------------------
    def install_check(self) -> str:
        parts = [self.INSTALL_CHECK]
        if self.pinned_version:
            parts.append(f'[ "$({self.VERSION_CMD})" = '
                         f"{shlex.quote(self.pinned_version)} ]")
        return " && ".join(parts)

    def install_script(self) -> str:
        pkg = self.PACKAGE + (f"@{self.pinned_version}" if self.pinned_version
                              else "@latest")
        return "\n".join([
            *node_install(self.NODE_MAJOR),
            'npm_config_cache="$(mktemp -d)"; export npm_config_cache',
            f"npm install -g --no-fund --no-audit {pkg}",
            'rc=$?; rm -rf "$npm_config_cache" "$HOME/.npm"',
            link_bins("node", "npm", "npx", "kimi"),
            "exit $rc",
        ])

    # ---- the task's tool policy -------------------------------------------
    def _tools_for(self, policy: dict) -> list[str]:
        if policy.get("mcp_only"):
            if policy.get("web_search") or policy.get("web_fetch"):
                raise ValueError("MCP-only Kimi cannot enable native web tools")
            return []
        if policy.get("web_search"):
            raise ValueError(
                f"adapter {self.name()!r} cannot offer web search: Kimi Code "
                f"registers WebSearch only with a Moonshot search provider "
                f"(services.moonshot_search or an OAuth login), and the "
                f"env-configured model this adapter runs has neither. "
                f"web_search: true cannot be honoured; this task asks for it")
        tools = list(self.BASE_TOOLS)
        if policy.get("web_fetch"):
            tools.append(self.WEB_FETCH_TOOL)
        return tools

    def disabled_tools(self) -> list[str]:
        """The `[tools] disabled` list for this run. ALWAYS names both web
        tools under the closed-book policy: WebSearch is absent anyway under
        env-only config, and naming it costs nothing and holds if a future CLI
        grows a default search provider."""
        if self.policy is None:
            raise RuntimeError("apply_tool_policy has not been called")
        out = []
        if not self.policy["web_fetch"]:
            out.append(self.WEB_FETCH_TOOL)
        if not self.policy["web_search"]:
            out.append(self.WEB_SEARCH_TOOL)
        return out

    def config_toml(self, *, mcp_server="chess") -> str:
        """`$KIMI_CODE_HOME/config.toml` for this run.

        Two sections and nothing else: the tool policy, and the step cap that
        `max_turns` becomes (`loop_control.max_steps_per_turn`; one step is one
        model call, which is what `max_turns` counts on the Claude adapter).
        The model itself is configured through the environment, so this file
        names no provider, no key and no URL.
        """
        parts = []
        disabled = self.disabled_tools()
        if disabled:
            parts.append("[tools]\ndisabled = "
                         + json.dumps(disabled) + "\n")
        if self.policy.get("mcp_only"):
            if mcp_server not in {"chess", "go"}:
                raise ValueError("Restricted Kimi requires a chess or Go namespace")
            # Kimi 0.43.0 applies this allowlist both to discovery and dispatch.
            # A nonempty exact list also excludes future native tools by default.
            parts.append("enabled = " + json.dumps([
                f"mcp__{mcp_server}__{name}" for name in self.MCP_TOOLS]) + "\n")
        if self.max_turns:
            parts.append(f"\n[loop_control]\nmax_steps_per_turn = "
                         f"{int(self.max_turns)}\n")
        return "".join(parts)

    def cli_env(self) -> dict:
        """The KIMI_* variables the CLI is started with, policy and generation
        settings included. The base URL, the home and the placeholder key are
        added by the runner, which is the only thing that knows the port."""
        env = dict(self.CLI_ENV)
        effort = self.generation_config.get("thinking_effort")
        if effort:
            env["KIMI_MODEL_THINKING_EFFORT"] = str(effort)
        return env

    def cli_flags(self) -> list[str]:
        """Flags before `--prompt`. None today: print mode refuses the
        permission flags and takes the model from KIMI_MODEL_NAME."""
        return []

    def setup(self) -> list[str]:
        problems = [] if self.RUNNER.exists() else [f"missing runner {self.RUNNER}"]
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
            return problems
        problems += self.config_problems()
        effort = self.generation_config.get("thinking_effort")
        if effort is not None and effort not in ("low", "medium", "high", "xhigh", "max"):
            problems.append(f"kimi-code does not support thinking effort {effort!r}")
        if not self.resolved_base_url():
            problems.append(
                f"model {self.model!r}: the {self.name()!r} adapter forwards "
                f"every request to a gateway and has none to forward to; set "
                f"`api_base_url:` on the model entry or OPENAI_BASE_URL in .env")
        # The closed-book policy must have produced the disabled list, or the
        # CLI would run with its defaults (FetchURL on).
        disabled = self.disabled_tools()
        if not self.policy["web_fetch"] and self.WEB_FETCH_TOOL not in disabled:
            problems.append("closed-book policy did not disable FetchURL")
        if self.WEB_FETCH_TOOL in self.tools and not self.policy["web_fetch"]:
            problems.append("web_fetch is closed but FetchURL is in the tool list")
        return problems

    def resolved_routing(self) -> dict:
        """As the base class records it, except that THIS adapter's forwarder
        is always on (it is the reasoning tap as well as the pin), so the
        record says so whatever `extra_body` holds."""
        out = super().resolved_routing()
        out["forwarder"] = True
        out["forwarder_always"] = True
        return out

    # ---- the episode ------------------------------------------------------
    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
        """The stdin payload for one episode. Same shape as the Codex
        adapter's: the row travels only when the task ships a grader, and the
        forwarder module ALWAYS travels (see the class docstring)."""
        task = {
            "id": row["id"],
            "model": self.model,
            "content": prompt,
            "files": files,
            "skip_dirs": list(skip_dirs),
            "deliverable_files": list(row.get("deliverable_files") or []),
            "tools": self.tools,
            "timeout_s": timeout_s,
            "env": dict(env or {}),
            "install": install,
            # The TASK's own `check || install`, from its `install:`
            # block, run by the same helper right after the agent's: a
            # library this task's work needs that the shared image does
            # not carry. None for a task that declares none.
            "task_install": task_install,
            # Where the credential sits in the container, the upstream and the
            # body pin: core/agents/base.py `routing_payload`.
            "routing": self.routing_payload(),
            "kimi": {
                # The FULL gateway id, no prefix stripping: the gateway routes
                # on `openrouter/moonshotai/kimi-k3` as written.
                "model_name": self.model,
                "kimi_home": self.KIMI_HOME,
                "api_key_env": self.api_key_env,
                "api_base_url": self.resolved_base_url(),
                "placeholder_key": self.PLACEHOLDER_KEY,
                "cli_env": self.cli_env(),
                "cli_flags": self.cli_flags(),
                "config_toml": self.config_toml(mcp_server=row.get("tool_surface", "chess")),
                "mcp_only": bool(self.policy.get("mcp_only")),
                "probe": True,
            },
        }
        if grader_src:
            task["row"] = ({**row, "restricted_tools": True, "episode_timeout_s": timeout_s}
                           if self.policy.get("mcp_only") else row)
        return json.dumps({
            "task": task,
            "code": self.RUNNER.read_text(),
            "modules": {"stage": stage_src, "grader": grader_src,
                        "forwarder": self.forwarder_source()},
        }).encode()

    # ---- the journal as ATIF ----------------------------------------------
    #
    # The records the runner returns under `messages`, in order:
    #
    #   wire.jsonl records (main agent first, then subagents), each carrying
    #   `type` and `agentId`:
    #     profile.bind                  system prompt, active tools, effort
    #     turn.prompt                   the instruction
    #     context.append_message        user messages incl. system-reminder
    #                                   injections (`origin.kind: injection`)
    #     context.append_loop_event     one `event` of type
    #                                     step.begin / content.part /
    #                                     tool.call / tool.result / step.end
    #     turn.ended, usage.record, llm.request, llm.tools_snapshot, ...
    #   forwarder.call records          one per model request: id, reasoning,
    #                                   content, tool_calls, usage, provider
    #
    # One ATIF step is one `step.begin`..`step.end` of the main agent: the
    # text parts, the tool calls and their results, the usage, and the
    # reasoning of the forwarder call whose `id` equals the step's
    # `messageId`. A step whose `step.end` never arrived (a killed episode) is
    # still emitted with what it had.
    @staticmethod
    def _iso(ms) -> str | None:
        if not isinstance(ms, (int, float)):
            return None
        try:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None

    @staticmethod
    def _text_of(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for block in content or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    @staticmethod
    def _result_text(result) -> str | None:
        """A `tool.result.result` as text: its `output`, whatever shape."""
        if result is None:
            return None
        if isinstance(result, dict):
            out = result.get("output")
            if out is None:
                out = {k: v for k, v in result.items() if k != "display"} or None
                return json.dumps(out, ensure_ascii=False) if out else None
            if isinstance(out, str):
                return out
            if isinstance(out, list):
                return "\n".join(
                    (p.get("text", "") if isinstance(p, dict) else str(p))
                    for p in out)
            return json.dumps(out, ensure_ascii=False)
        return str(result)

    @staticmethod
    def _metrics(usage: dict | None) -> dict | None:
        if not isinstance(usage, dict):
            return None
        other = int(usage.get("inputOther") or 0)
        read = int(usage.get("inputCacheRead") or 0)
        write = int(usage.get("inputCacheCreation") or 0)
        prompt = other + read + write
        return {"prompt_tokens": prompt or None,
                "completion_tokens": int(usage.get("output") or 0) or None,
                "cached_tokens": read or None,
                "extra": {"input_other": other, "input_cache_creation": write}}

    def to_trajectory(self, raw: dict) -> Trajectory:
        records = [r for r in ((raw or {}).get("messages") or [])
                   if isinstance(r, dict)]
        model = self.model
        calls_by_id: dict = {}
        agents_seen: set = set()
        profile: dict = {}
        tools_sent: list | None = None
        for rec in records:
            if rec.get("type") == "forwarder.call":
                if rec.get("id"):
                    calls_by_id[rec["id"]] = rec
                if isinstance(rec.get("model"), str) and rec["model"]:
                    model = rec["model"]
                continue
            agents_seen.add(rec.get("agentId", "main"))
            if rec.get("agentId", "main") != "main":
                continue
            if rec.get("type") == "profile.bind":
                profile = rec
            elif rec.get("type") == "llm.tools_snapshot" and tools_sent is None:
                # What the model was ACTUALLY offered on its first request:
                # the profile's `activeToolNames` is the allow-list before the
                # `[tools] disabled` filter and still names the web tools.
                tools_sent = [t.get("name") for t in rec.get("tools") or []
                              if isinstance(t, dict) and t.get("name")]

        steps: list[Step] = []
        dropped: list[str] = []
        cur: dict | None = None      # the open agent step

        def flush() -> None:
            nonlocal cur
            if cur is None:
                return
            calls: list[ToolCall] = []
            results: list[ObservationResult] = []
            for tc in cur["tool_calls"]:
                args = tc.get("args")
                if not isinstance(args, dict):
                    args = {"value": args} if args is not None else {}
                calls.append(ToolCall(tool_call_id=tc["id"],
                                      function_name=tc.get("name") or "",
                                      arguments=args))
                if "output" in tc:
                    results.append(ObservationResult(
                        source_call_id=tc["id"], content=tc["output"]))
            reasoning = cur.get("reasoning")
            fw = calls_by_id.get(cur.get("message_id") or "")
            if fw and fw.get("reasoning"):
                reasoning = fw["reasoning"] if not reasoning else \
                    f"{reasoning}\n{fw['reasoning']}"
            extra: dict = {}
            for k in ("message_id", "finish_reason", "turn_id", "step"):
                if cur.get(k) is not None:
                    extra[k] = cur[k]
            if fw and fw.get("provider"):
                extra["provider"] = fw["provider"]
            metrics = cur.get("metrics")
            try:
                steps.append(Step(
                    step_id=len(steps) + 1, timestamp=cur.get("timestamp"),
                    source="agent", message="\n\n".join(cur["text"]),
                    model_name=model,
                    reasoning_content=reasoning or None,
                    tool_calls=calls or None,
                    observation=Observation(results=results) if results else None,
                    metrics=Metrics(**metrics) if metrics else None,
                    llm_call_count=1, extra=extra or None))
            except ValueError as e:
                dropped.append(f"step: {type(e).__name__}: {e}")
            cur = None

        for rec in records:
            rtype = rec.get("type")
            if rtype == "forwarder.call" or rec.get("agentId", "main") != "main":
                continue
            ts = self._iso(rec.get("time"))

            if rtype == "turn.prompt":
                flush()
                text = self._text_of(rec.get("input"))
                steps.append(Step(step_id=len(steps) + 1, timestamp=ts,
                                  source="user", message=text))
                continue
            if rtype == "context.append_message":
                msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
                origin = msg.get("origin") if isinstance(msg.get("origin"), dict) else {}
                if msg.get("role") == "user" and origin.get("kind") == "user":
                    # Already emitted from turn.prompt, which precedes it.
                    continue
                if msg.get("role") == "user":
                    flush()
                    extra = {k: origin[k] for k in ("kind", "variant") if origin.get(k)}
                    steps.append(Step(step_id=len(steps) + 1, timestamp=ts,
                                      source="system",
                                      message=self._text_of(msg.get("content")),
                                      extra=extra or None))
                continue
            if rtype == "stream_json":
                # The no-journal fallback: stream-json lines, one step each.
                line = rec.get("line") or {}
                role = line.get("role")
                if role == "assistant":
                    flush()
                    calls = []
                    for tc in line.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            args = {"input": fn.get("arguments")}
                        calls.append({"id": tc.get("id") or "",
                                      "name": fn.get("name"), "args": args})
                    cur = {"text": [line["content"]] if line.get("content") else [],
                           "tool_calls": calls, "timestamp": ts}
                    if not calls:
                        flush()
                elif role == "tool" and cur is not None:
                    for tc in cur["tool_calls"]:
                        if tc["id"] == line.get("tool_call_id"):
                            tc["output"] = line.get("content")
                continue
            if rtype != "context.append_loop_event":
                continue

            event = rec.get("event") if isinstance(rec.get("event"), dict) else {}
            kind = event.get("type")
            if kind == "step.begin":
                flush()
                cur = {"text": [], "tool_calls": [], "timestamp": ts,
                       "turn_id": event.get("turnId"), "step": event.get("step")}
            elif kind == "content.part":
                if cur is None:
                    cur = {"text": [], "tool_calls": [], "timestamp": ts}
                part = event.get("part") or {}
                if part.get("type") == "think" and isinstance(part.get("think"), str):
                    cur["reasoning"] = (cur.get("reasoning") or "") + part["think"]
                elif isinstance(part.get("text"), str):
                    cur["text"].append(part["text"])
            elif kind == "tool.call":
                if cur is None:
                    cur = {"text": [], "tool_calls": [], "timestamp": ts}
                cur["tool_calls"].append({"id": event.get("toolCallId") or "",
                                          "name": event.get("name"),
                                          "args": event.get("args")})
            elif kind == "tool.result":
                cid = event.get("toolCallId")
                target = None
                if cur is not None:
                    target = next((tc for tc in cur["tool_calls"]
                                   if tc["id"] == cid), None)
                if target is not None:
                    target["output"] = self._result_text(event.get("result"))
                else:
                    # The live tail, or an orphan: a result-only step.
                    flush()
                    steps.append(Step(
                        step_id=len(steps) + 1, timestamp=ts, source="agent",
                        message="",
                        observation=Observation(results=[ObservationResult(
                            source_call_id=None,
                            content=self._result_text(event.get("result")))])))
            elif kind == "step.end":
                if cur is None:
                    cur = {"text": [], "tool_calls": [], "timestamp": ts}
                cur["message_id"] = event.get("messageId")
                cur["finish_reason"] = event.get("finishReason")
                cur["metrics"] = self._metrics(event.get("usage"))
                flush()
        flush()

        if not steps:
            steps = [self.empty_step(
                "(no journal records were captured for this episode)")]

        # -- totals: the forwarder's tally first, the CLI's own second --------
        final_metrics = None
        usage = (raw or {}).get("usage") or (raw or {}).get("stream_usage")
        if isinstance(usage, dict):
            extra = {k: usage[k] for k in ("reasoning_tokens", "total_tokens",
                                           "calls", "calls_with_usage")
                     if usage.get(k) is not None}
            if usage.get("cache_write_tokens"):
                extra["total_cache_write_tokens"] = usage["cache_write_tokens"]
            final_metrics = FinalMetrics(
                total_prompt_tokens=usage.get("prompt_tokens") or None,
                total_completion_tokens=usage.get("completion_tokens") or None,
                total_cached_tokens=usage.get("cached_tokens") or None,
                total_cost_usd=(raw or {}).get("cost_usd"),
                total_steps=len(steps),
                extra=extra or None)

        final_text = (raw or {}).get("final_text") or ""
        if not final_text:
            for step in reversed(steps):
                text = step.message if isinstance(step.message, str) else ""
                if step.source == "agent" and text.strip():
                    final_text = text
                    break

        subagents = sorted(a for a in agents_seen if a != "main")
        notes = []
        if dropped:
            notes.append(f"{len(dropped)} journal step(s) could not be converted "
                         f"to ATIF steps and are missing from this trajectory: "
                         + "; ".join(dropped))
        if subagents:
            notes.append(f"{len(subagents)} subagent journal(s) "
                         f"({', '.join(subagents)}) are in the native record "
                         f"and not rendered as steps")
        agent_extra = {k: profile[k] for k in
                       ("thinkingEffort", "disallowedTools", "subagents")
                       if profile.get(k) is not None}
        if tools_sent is not None:
            agent_extra["tools"] = tools_sent
        agent_extra = agent_extra or None
        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=(raw or {}).get("session_id"),
            agent=self.atif_agent(model_name=model, extra=agent_extra),
            steps=steps,
            notes="; ".join(notes) if notes else None,
            final_metrics=final_metrics,
            extra={"final_text": final_text} if final_text else None,
        )
