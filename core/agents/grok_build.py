"""The xAI Grok Build CLI adapter.

MODELLED ON HARBOR: `harbor/src/harbor/agents/installed/grok_build.py`, which
this is ported from, in the layout of `core/agents/codex.py`. Its in-container
runner is the sibling `grok_build_runner.py`, read as SOURCE and streamed into
the container on stdin; it is never imported here and never written to the
container filesystem.
"""
from __future__ import annotations

import json
import shlex
from urllib.parse import urlsplit
import uuid
from pathlib import Path

from ..trajectory import (
    SCHEMA_VERSION,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from ..trajectory import Agent as ATIFAgent
from . import errors
from .errors import ErrorPattern
from .installed import InstalledAgent


class GrokBuildAgent(InstalledAgent):
    """xAI's Grok Build CLI (`grok`), driven by `grok_build_runner.py`.

    Ported from Harbor's `harbor/agents/installed/grok_build.py`. How grok is
    installed, invoked and read back is Harbor's, reshaped to what this
    sandbox does differently (see `CodexAgent` for the long form):

      * `install()` RETURNS a check/install pair; the runner runs it.
      * the trajectory is not copied out. The runner reads the session's
        `chat_history.jsonl` inside the container and returns its lines as
        `messages`; the CLI's streaming-json stdout supplies the usage.
      * the task's grader is hosted BY THE RUNNER over loopback HTTP and only
        its URL goes into `$GROK_HOME/config.toml`.

    ROUTING, which is the point of this adapter existing here at all. Harbor's
    native path sends `--model grok-4.6` to api.x.ai with XAI_API_KEY. Ours
    never does: every request goes to the LiteLLM gateway's OpenRouter route,
    which the CLI reaches through a `[model."<id>"]` block in its config.toml
    (`base_url`, `env_key`, `api_backend = "chat_completions"` -- the CLI's
    documented BYOK shape). The model id in that block is sent VERBATIM, so
    `openrouter/x-ai/grok-4.6` arrives at the gateway unchanged and the gateway
    routes it to xAI through OpenRouter (the response's `provider` says
    "xAI"). No forwarder is needed: the CLI takes a base URL and a key
    variable directly, and grok needs no provider pin.

    Verified against grok 1.0.24 through a logging proxy, not inferred: each
    inference request carried exactly `model`, `messages`, `tools`,
    `reasoning_effort`, `stream` and `stream_options` -- no temperature, top_p,
    seed, max_tokens or any token cap -- and `--reasoning-effort` reached the
    wire only once the block declared `supports_reasoning_effort = true`.

    ONE SIDE CALL the CLI makes on its own account: at the start of every
    session it asks the `session_summary` model for a title, with its own
    sampling settings (`temperature: 1.0, max_tokens: 100`, a forced
    `session_title` tool). That is the CLI's bookkeeping, not the agent's
    generation; it is not in the trajectory, and the CLI EXCLUDES it from the
    `end` usage totals, so it is not in `cost_usd` either. There is no
    documented switch for it. The auxiliary models are pinned to the same
    entry, as Harbor does, so that call goes through the same gateway route
    rather than falling back to the CLI's native catalog (a 400 through the
    gateway).
    """
    TURN_LIMIT = 'model turns'
    PERMISSION_MODES = ("bypassPermissions", "default", "acceptEdits", "auto", "plan", "dontAsk")

    @staticmethod
    def name() -> str:
        return "grok-build"

    # ---- what this adapter can hold shut ----------------------------------
    #
    # Both keys, through SEPARATE knobs -- the first adapter here where the two
    # can genuinely differ:
    #
    #   web_search   `disable_web_search = true` (config; Harbor's default, "for
    #                closed-book eval integrity") drops the client `web_search`
    #                tool. `[features] backend_tools = false` additionally pins
    #                off the xAI-hosted server-side search, which a custom
    #                `[model]` block never enables anyway.
    #   web_fetch    `[features] web_fetch = <policy>`; the CLI's own default is
    #                false ("Enable or disable web_fetch. Default false").
    #   both closed  `--disable-web-search` on the command line as well, which
    #                the CLI documents as "Disable web search and web fetch
    #                tools".
    #
    # EVIDENCE, on grok 1.0.24 through a logging proxy: under the closed-book
    # configuration the `tools` array in every inference request was
    #     run_terminal_command, read_file, search_replace, list_dir, grep,
    #     kill_command_or_subagent, todo_write, get_command_or_subagent_output,
    #     scheduler_create, scheduler_delete, scheduler_list, monitor,
    #     search_tool, use_tool, workflow, enter_plan_mode, exit_plan_mode,
    #     write
    # with neither `web_search` nor `web_fetch` present, i.e. the model was not
    # OFFERED either tool and cannot call what it is not offered. That is the
    # same "the model enumerates its own tools" check the codex row of
    # base.py's table was admitted on, taken one level lower (the wire rather
    # than the model's report). The mirror-log check on a real openmath episode
    # is in the smoke-test record for this adapter. The open-web side (`true`)
    # is composed but has not been exercised: the client web tools use xAI's
    # own search backend, which this routing never reaches.
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}

    # Harbor's default is XAI_API_KEY for the native path. Ours is the gateway
    # key, and a model entry's `api_key_env:` shadows this per instance. The
    # container receives the key under whatever name this resolves to, and the
    # runner writes exactly that name as `env_key` in config.toml, so the CLI
    # has no fixed key variable of its own here -- hence CLI_KEY_ENV is empty.
    API_KEY_ENV = "OPENAI_API_KEY"
    CLI_KEY_ENV: tuple = ()
    # The base URL the CLI uses travels in config.toml, never in the
    # environment. OPENAI_BASE_URL is still named here for two other reasons,
    # exactly as on the codex adapter: it is the PASSTHROUGH fallback for an
    # entry that names no `api_base_url:` (`resolved_base_url()`), and it is
    # what the TASK's grader reads -- gdpval's runs in the runner process
    # beside the CLI and has to reach the same gateway. A URL is not a secret;
    # the key travels by name.
    CLI_BASE_URL_ENV: tuple = ("OPENAI_BASE_URL",)
    PASSTHROUGH_ENV = ("OPENAI_BASE_URL",)
    # grok needs no `extra_body`: OpenRouter already routes x-ai to xAI, and
    # the CLI takes the base URL and key variable directly.
    SUPPORTS_FORWARDER = False

    # The exact `grok` argv the episode ran, where its session was read back
    # from, the CLI's own cost figure when it stamped one, and the runner's
    # one-request probe of the base URL (`gateway_probe`: which `provider`
    # answered for the model id -- the routing evidence the CLI itself never
    # surfaces).
    EXTRA_RECORD_KEYS = ("grok_command", "session_dir", "reported_cost_usd",
                         "usage_calls", "usage_is_incomplete", "gateway_probe")

    # xAI capacity errors, Harbor's pattern, ahead of the shared list.
    ERROR_PATTERNS = [
        ErrorPattern(r"resource-exhausted|currently at capacity",
                     errors.ApiOverloadedError),
        *errors.ERROR_PATTERNS,
    ]

    # ---- what an episode cost ---------------------------------------------
    #
    # The CLI's `end` event prices the session ONLY when its server stamped a
    # complete cost ("Cost is stamped for API-key traffic today; ... Absence
    # means unreported or incomplete, never free"). Through the gateway that
    # field is absent, so the litellm estimate is this adapter's usual source,
    # keyed on the full id `openrouter/x-ai/grok-4.6`, which litellm's table
    # carries at $2/M in, $6/M out, $0.5/M cache read.
    #
    # The four names are the CLI's own, and its `input_tokens` is EXCLUSIVE of
    # the cache legs (documented "Token field policy": "usage.input_tokens ...
    # are uncached only"; measured: 25152 fresh + 12928 cache reads, total
    # 38350 = 25152 + 12928 + 0 + 270). `output_tokens` already contains
    # `reasoning_tokens` (270 output, of which 188 reasoning), so reasoning is
    # never added on top. `stream_usage` is the sum of the per-response usage
    # events, the fallback for an episode killed before `end`.
    USAGE_FIELDS = ("usage", "stream_usage")
    USAGE_KEYS = ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens")
    PROMPT_TOKENS_INCLUDE_CACHE = False

    def reported_cost_usd(self, raw: dict) -> float | None:
        """The CLI's own `total_cost_usd`, when it stamped a complete one.

        Read from `reported_cost_usd`, which the runner sets ONLY from the
        `end` event -- never from `cost_usd`, which the host may already have
        filled with its own estimate and which must not be relabelled as the
        vendor's figure on a re-cost.
        """
        value = (raw or {}).get("reported_cost_usd")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    # ---- this adapter's runtime -------------------------------------------
    #
    # A single static binary from xAI's installer, Harbor's URL. The installer
    # puts the binary under `$HOME/.grok/downloads/` and symlinks it into
    # GROK_BIN_DIR; /usr/local/bin is where every process looks and is owned by
    # the container's uid (see installed.link_bins). No Node, no npm.
    INSTALL_HOSTS = ('x.ai', 'storage.googleapis.com')
    INSTALL_SCRIPT_URL = "https://x.ai/cli/install.sh"
    BIN_DIR = "/usr/local/bin"

    # `grok --version` prints `grok 1.0.24 (68e414c661e3)`; the second word.
    VERSION_CMD = "grok --version 2>/dev/null | head -1 | awk '{print $2}'"
    INSTALL_CHECK = "command -v grok >/dev/null 2>&1"

    # Where grok keeps its state inside the container: config.toml and the
    # session tree the runner reads back. Removed when the episode ends.
    GROK_HOME = "/tmp/grok-home"

    # Harbor's CLI_FLAGS, as a translation table: `reasoning_effort` is the
    # one generation setting the CLI exposes, delivered as
    # `--reasoning-effort`, with Harbor's default of `high` ("the serving-side
    # default is deploy-dependent").
    DEFAULT_REASONING_EFFORT = "high"
    REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high",
                                "xhigh", "max")
    GENERATION_KEYS = ("reasoning_effort",)

    # grok's own toolset under this adapter's config, as the CLI advertised it
    # in `available_commands` on a real 1.0.24 session (see ENFORCEABLE). All
    # of these run INSIDE the container. Absent by construction: web_search,
    # web_fetch (policy), spawn_subagent (`[subagents] enabled = false` -- a
    # subagent's session is a separate transcript the judge would never see),
    # ask_user_question (`[features]`, nobody is there to answer), image_gen /
    # image_edit / video tools (xAI-side generation), send_feedback.
    BASE_TOOLS = ("run_terminal_command", "read_file", "search_replace",
                  "list_dir", "grep", "kill_command_or_subagent",
                  "todo_write", "get_command_or_subagent_output",
                  "scheduler_create", "scheduler_delete", "scheduler_list",
                  "monitor", "search_tool", "use_tool", "workflow",
                  "enter_plan_mode", "exit_plan_mode", "write")
    WEB_TOOLS = {"web_search": "web_search", "web_fetch": "web_fetch"}

    # Tools the `[features]` switches do not remove and `--disallowed-tools`
    # must (measured: both survive `feedback = false` / `image_gen = false`).
    # Only names the CLI actually carries go here -- an entry that matches
    # nothing is a WARN on stderr, not an error, but it is noise.
    DISALLOWED_TOOLS = ("send_feedback", "image_edit")

    # The runner is a sibling module, read as SOURCE and never imported here.
    RUNNER = Path(__file__).resolve().parent / "grok_build_runner.py"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pre-generated per adapter instance and passed on the command line as
        # `--session-id`, so the runner knows which session directory to read
        # back (Harbor does the same). A fresh one per episode is drawn in
        # `blob()`; this is the seed for a `to_trajectory` of a record that
        # carries none.
        self._session_seed = str(uuid.uuid4())

    # ---- install ----------------------------------------------------------
    def install_check(self) -> str:
        parts = [self.INSTALL_CHECK]
        if self.pinned_version:
            parts.append(f'[ "$({self.VERSION_CMD})" = '
                         f"{shlex.quote(self.pinned_version)} ]")
        return " && ".join(parts)

    def install_script(self) -> str:
        """Harbor's install: `curl -fsSL https://x.ai/cli/install.sh | bash`,
        with `-s <version>` for a pin. Fetched to a temp file first so a
        truncated download is a curl failure rather than a half-run script."""
        version = (f" {shlex.quote(self.pinned_version)}"
                   if self.pinned_version else "")
        return "\n".join([
            "export GROK_DISABLE_AUTOUPDATER=1",
            't="$(mktemp -d)" || exit $?',
            f'curl -fsSL {self.INSTALL_SCRIPT_URL} -o "$t/install.sh" || exit $?',
            f'GROK_BIN_DIR={self.BIN_DIR} bash "$t/install.sh"{version} || exit $?',
            'rm -rf "$t"',
        ])

    # ---- the task's tool policy -------------------------------------------
    def _tools_for(self, policy: dict) -> list[str]:
        if policy.get("mcp_only"):
            return []
        return list(self.BASE_TOOLS) + [tool for key, tool in self.WEB_TOOLS.items()
                                        if policy.get(key)]

    # ---- generation -------------------------------------------------------
    @property
    def reasoning_effort(self) -> str:
        return str(self.generation_config.get("reasoning_effort")
                   or self.DEFAULT_REASONING_EFFORT)

    def resolved_generation_config(self) -> dict:
        """Harbor's `high` unless the entry overrides it; run.json says which."""
        return {"reasoning_effort": self.DEFAULT_REASONING_EFFORT,
                **self.generation_config}

    def permission_flags(self) -> list[str]:
        """`bypassPermissions` is grok's `--always-approve` (Harbor's flag, the
        documented alias of `--permission-mode bypassPermissions`). Any other
        mode is passed through by name; the CLI's enum is the same as ours."""
        if self.permission_mode == "bypassPermissions":
            return ["--always-approve"]
        return ["--permission-mode", str(self.permission_mode)]

    # ---- routing ----------------------------------------------------------
    def api_base(self) -> str:
        """Where the CLI sends its requests, or "" for xAI's own endpoint.

        The shared order (`Agent.resolved_base_url`): the entry's
        `api_base_url:` (core/routing.py), then the OPENAI_BASE_URL
        passthrough -- so an entry that names nothing runs through the same
        gateway origin the codex adapter reads from the environment; the
        legacy per-entry `base_url:` is honoured after those.

        The CLI composes `<base_url>/chat/completions`, so the OpenAI-style
        `/v1` must be ON the base. The .env origin is shared with the codex
        adapter, which wants it bare, so it is appended here when absent.
        """
        base = (self.resolved_base_url() or self.base_url or "").strip().rstrip("/")
        if base and not base.endswith("/v1"):
            base += "/v1"
        return base

    @property
    def model_slug(self) -> str:
        """What goes after `--model`: the entry's id, VERBATIM, when routed
        (it names the `[model."<id>"]` block, whose `model` is sent as-is);
        Harbor's single-prefix strip (`xai/grok-4.6` -> `grok-4.6`) only on
        the native path, where the CLI's own catalog is what it must match."""
        if self.api_base() and urlsplit(self.api_base()).hostname != "api.x.ai":
            return self.model
        provider, _, rest = self.model.partition("/")
        return rest or provider

    def grok_config(self, session_id: str) -> dict:
        """Everything the runner needs to compose config.toml and the argv."""
        policy = self.policy or {}
        return {
            "model_slug": self.model_slug,
            "model_id": self.model_slug,
            "base_url": self.api_base(),
            # The name the key sits under INSIDE the container, which is what
            # the CLI's `env_key` must name; with no CLI_KEY_ENV of its own
            # that is the entry's host variable (`container_key_envs`).
            "api_key_env": self.container_key_envs()[0],
            "reasoning_effort": self.reasoning_effort,
            "max_turns": self.max_turns,
            "permission_flags": self.permission_flags(),
            "web_search": bool(policy.get("web_search")),
            "web_fetch": bool(policy.get("web_fetch")),
            "mcp_only": bool(policy.get("mcp_only")),
            "disallowed_tools": list(self.DISALLOWED_TOOLS),
            "grok_home": self.GROK_HOME,
            "session_id": session_id,
        }

    def setup(self) -> list[str]:
        problems = [] if self.RUNNER.exists() else [f"missing runner {self.RUNNER}"]
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
            return problems
        problems += self.config_problems()
        # `extra_body` on an entry is refused here: SUPPORTS_FORWARDER is
        # False, the CLI talks to its base URL directly, and a body field the
        # provider never sees is a config file that lies.
        problems += self.routing_problems()
        effort = self.generation_config.get("reasoning_effort")
        if effort and str(effort) not in self.REASONING_EFFORT_CHOICES:
            problems.append(f"reasoning_effort={effort!r}; grok takes "
                            f"{list(self.REASONING_EFFORT_CHOICES)}")
        if not self.api_base() and self.container_key_envs()[0] != "XAI_API_KEY":
            problems.append(
                f"model {self.model!r}: no base URL (set `api_base_url:` on "
                f"the entry or OPENAI_BASE_URL in .env) and the credential is "
                f"{self.API_KEY_ENV}, which grok's native endpoint does not "
                f"read; the native path needs api_key_env: XAI_API_KEY")
        # Closed-book composition, asserted rather than assumed.
        cfg = self.grok_config("preflight")
        for key, tool in self.WEB_TOOLS.items():
            if not self.policy.get(key) and tool in self.tools:
                problems.append(f"{key} is closed but {tool} is still in the "
                                f"tool list")
        if not cfg["web_search"] and not cfg["web_fetch"] and \
                "--disable-web-search" not in self._preview_argv(cfg):
            problems.append("closed-book policy did not produce "
                            "`--disable-web-search`")
        return problems

    @staticmethod
    def _preview_argv(cfg: dict) -> list[str]:
        """The argv the runner will compose, for preflight checks and tests.
        Same rules as `grok_build_runner.compose_argv`, which is not imported
        here (the runner is streamed source, never a module)."""
        argv = ["grok", "--no-auto-update", "-p", "<instruction>",
                *cfg["permission_flags"], "--output-format", "streaming-json",
                "--session-id", cfg["session_id"], "--model", cfg["model_slug"]]
        if cfg.get("max_turns"):
            argv += ["--max-turns", str(int(cfg["max_turns"]))]
        if cfg.get("reasoning_effort"):
            argv += ["--reasoning-effort", str(cfg["reasoning_effort"])]
        if not cfg.get("web_search") and not cfg.get("web_fetch"):
            argv.append("--disable-web-search")
        if cfg.get("disallowed_tools"):
            argv += ["--disallowed-tools", ",".join(cfg["disallowed_tools"])]
        if cfg.get("mcp_only"):
            argv += ["--tools", "search_tool,use_tool", "--no-subagents", "--no-plan"]
        argv += ["--cwd", "/workspace"]
        return argv

    # ---- the episode ------------------------------------------------------
    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
        """The stdin payload for one episode. Same shape as the codex blob:
        the row travels only when the task ships a grader, and nothing of the
        harness touches the container filesystem, argv or environment."""
        task = {
            "id": row["id"],
            "model": self.model,
            "content": prompt,
            "files": files,
            "skip_dirs": list(skip_dirs),
            "deliverable_files": list(row.get("deliverable_files") or ()),
            "tools": self.tools,
            "timeout_s": timeout_s,
            "env": dict(env or {}),
            "install": install,
            # The TASK's own `check || install`, from its `install:`
            # block, run by the same helper right after the agent's: a
            # library this task's work needs that the shared image does
            # not carry. None for a task that declares none.
            "task_install": task_install,
            "grok": self.grok_config(str(uuid.uuid4())),
        }
        if grader_src:
            task["row"] = ({**row, "restricted_tools": True, "episode_timeout_s": timeout_s}
                           if (self.policy or {}).get("mcp_only") else row)
        return json.dumps({
            "task": task,
            "code": self.RUNNER.read_text(),
            "modules": {"stage": stage_src, "grader": grader_src,
                        "support": self.RUNNER.with_name("runner_support.py").read_text()},
        }).encode()

    # ---- chat_history.jsonl as ATIF ----------------------------------------
    #
    # PORTED FROM HARBOR, Apache License 2.0: `_convert_messages_to_trajectory`,
    # `_content_to_text`, `_parse_tool_call_arguments` and `_drain_reasoning`
    # from `harbor/src/harbor/agents/installed/grok_build.py`. Harbor validates
    # each line through pydantic models and skips what does not validate; the
    # same five shapes are read here by hand, and a line of any other type is
    # counted in the trajectory's `notes` rather than dropped in silence.
    #
    # chat_history lines are `{"type": ..., ...}`:
    #
    #   system       {"content": str | [{"type","text"}]}     the CLI's prompt
    #   user         {"content": ...}                         the instruction,
    #                                                         and the CLI's own
    #                                                         <user_info>/<rules>
    #                                                         blocks
    #   reasoning    {"summary": [{"type","text"}], "content": [...]?}
    #   assistant    {"content": ..., "tool_calls": [{"id","name","arguments"}],
    #                 "model_id": str}
    #   tool_result  {"tool_call_id": str, "content": ...}
    #
    # Reasoning accumulates until the next assistant line, which takes it as
    # its `reasoning_content`; trailing reasoning with no assistant after it
    # (the model was cut off) is kept as its own step, as Harbor keeps it.
    @staticmethod
    def _content_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                (block.get("text") or "") if isinstance(block, dict)
                else (block if isinstance(block, str) else "")
                for block in content)
        return str(content)

    @staticmethod
    def _reasoning_text(message: dict) -> str:
        """Harbor: `summary` parts, then `content` parts, joined."""
        parts = []
        for key in ("summary", "content"):
            items = message.get(key)
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and isinstance(item.get("text"), str) \
                        and item["text"]:
                    parts.append(item["text"])
                elif isinstance(item, str) and item:
                    parts.append(item)
        return "\n".join(parts).strip()

    @staticmethod
    def _arguments(raw) -> dict:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"raw_arguments": raw}
        return parsed if isinstance(parsed, dict) else {"raw_arguments": raw}

    def to_trajectory(self, raw: dict) -> Trajectory:
        """Harbor's `_convert_messages_to_trajectory`, plus the usage totals."""
        raw = raw or {}
        messages = [m for m in (raw.get("messages") or []) if isinstance(m, dict)]

        steps: list[Step] = []
        by_call: dict[str, Step] = {}
        reasoning: list[str] = []
        dropped: list[str] = []
        model_seen: str | None = None

        def drain() -> str | None:
            text = "\n\n".join(p for p in reasoning if p).strip()
            reasoning.clear()
            return text or None

        for message in messages:
            mtype = message.get("type")
            if mtype == "system":
                steps.append(Step(step_id=len(steps) + 1, source="system",
                                  message=self._content_text(
                                      message.get("content")).strip()))
            elif mtype == "user":
                steps.append(Step(step_id=len(steps) + 1, source="user",
                                  message=self._content_text(
                                      message.get("content")).strip()))
            elif mtype == "reasoning":
                text = self._reasoning_text(message)
                if text:
                    reasoning.append(text)
            elif mtype == "assistant":
                model = message.get("model_id") if isinstance(
                    message.get("model_id"), str) else None
                model_seen = model_seen or model
                step = Step(step_id=len(steps) + 1, source="agent",
                            model_name=model or self.model,
                            message=self._content_text(
                                message.get("content")).strip(),
                            reasoning_content=drain(), llm_call_count=1)
                calls = message.get("tool_calls")
                if isinstance(calls, list) and calls:
                    step.tool_calls = []
                    step.observation = Observation(results=[])
                    for call in calls:
                        if not isinstance(call, dict) or not call.get("id"):
                            continue
                        step.tool_calls.append(ToolCall(
                            tool_call_id=str(call["id"]),
                            function_name=str(call.get("name") or ""),
                            arguments=self._arguments(call.get("arguments"))))
                        by_call[str(call["id"])] = step
                    if not step.tool_calls:
                        step.tool_calls = None
                        step.observation = None
                steps.append(step)
            elif mtype == "tool_result":
                call_id = message.get("tool_call_id")
                step = by_call.get(str(call_id)) if call_id else None
                if step is None or step.observation is None:
                    dropped.append(f"tool_result for unknown call {call_id!r}")
                    continue
                step.observation.results.append(ObservationResult(
                    source_call_id=str(call_id),
                    content=self._content_text(message.get("content"))))
            else:
                dropped.append(f"unsupported message type {mtype!r}")

        trailing = drain()
        if trailing:
            steps.append(Step(step_id=len(steps) + 1, source="agent",
                              model_name=model_seen or self.model, message="",
                              reasoning_content=trailing, llm_call_count=1))

        if not steps:
            steps = [self.empty_step(
                "(no chat history was captured for this episode)")]

        # -- totals, from the CLI's `end` event (or the summed per-response
        # usage on a killed episode), which the runner carries on the record.
        # chat_history itself has no token counts; Harbor merges them in from
        # the streaming output the same way.
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else \
            raw.get("stream_usage") if isinstance(raw.get("stream_usage"), dict) \
            else None
        final_metrics = FinalMetrics(total_steps=len(steps))
        if usage:
            fresh = int(usage.get("input_tokens") or 0)
            cached = int(usage.get("cache_read_input_tokens") or 0)
            written = int(usage.get("cache_creation_input_tokens") or 0)
            final_metrics = FinalMetrics(
                # ATIF's prompt total is inclusive; the CLI's `input_tokens`
                # is the uncached remainder, so the cache legs are added back.
                total_prompt_tokens=(fresh + cached + written) or None,
                total_completion_tokens=int(usage.get("output_tokens") or 0)
                or None,
                total_cached_tokens=cached or None,
                total_cost_usd=self.reported_cost_usd(raw)
                if self.reported_cost_usd(raw) is not None
                else raw.get("cost_usd"),
                total_steps=len(steps),
                extra={"total_cache_creation_input_tokens": written,
                       "total_reasoning_tokens": usage.get("reasoning_tokens"),
                       "uncached_input_tokens": fresh},
            )

        final_text = raw.get("final_text") or ""
        if not final_text:
            for step in reversed(steps):
                text = step.message if isinstance(step.message, str) else ""
                if step.source == "agent" and text.strip():
                    final_text = text
                    break

        agent_extra = {}
        if raw.get("terminal_reason"):
            agent_extra["stop_reason"] = raw["terminal_reason"]
        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=raw.get("session_id") or self._session_seed,
            agent=ATIFAgent(name=self.name(), version=self.version() or "unknown",
                            model_name=model_seen or self.model,
                            extra=agent_extra or None),
            steps=steps,
            notes=(f"{len(dropped)} chat_history line(s) could not be placed "
                   f"in this trajectory: " + "; ".join(dropped))
            if dropped else None,
            final_metrics=final_metrics,
            extra={"final_text": final_text} if final_text else None,
        )
