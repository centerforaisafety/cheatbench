"""The Meta Muse Code CLI adapter.

One adapter, one module, in Harbor's layout: Harbor added its own
`harbor/agents/installed/muse_code.py` on 2026-09-01 and this is that adapter
reshaped to this sandbox, exactly as `codex.py` is Harbor's codex adapter
reshaped. Its in-container runner is the sibling `muse_code_runner.py`, which
is read as SOURCE and streamed into the container on stdin; it is never
imported here and never written to the container filesystem.

Everything below that is stated as a fact about the CLI was MEASURED against
Muse Code 1.0.3 (1.0.3-R2198.1, build 238bb03ff3) on 2026-09-09, by installing
it, running `muse exec` against a loopback HTTP listener standing in for the
Meta API, and reading the requests it made, the JSONL it printed and the session
log it wrote. Nothing here is inferred from Harbor's adapter alone, because
Harbor's adapter does not host a grader, does not touch the tool policy and
does not parse the session log; all three are ours.
"""
from __future__ import annotations

import json
import os
import shlex
import re
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
from .installed import InstalledAgent, link_bins


class MuseCodeAgent(InstalledAgent):
    """Meta's Muse Code CLI (`muse exec`), driven by `muse_code_runner.py`.

    What Harbor's adapter does and we keep:

      * install = `curl -fsSL https://dev.meta.ai/install.sh | MUSE_NO_MODIFY_PATH=1 bash`,
        then `muse --version`, which is what makes the launcher download the
        real binary at install time rather than on the first episode;
      * run = `muse exec --json --yolo --user-input-auto-resolve
        --prompt-file <file> --model <slug> [--reasoning-effort E]
        [--max-model-steps N] [--base-url URL]`, stdin closed;
      * `MUSE_NO_AUTO_UPDATE=1` for the run, so the launcher cannot swap the
        binary mid-job;
      * exact release pins use versioned Meta artifacts with checksum verification;
        explicit latest retains the legacy launcher installation path;
      * the durable session log under `$XDG_DATA_HOME/muse/sessions` is the
        trajectory of record, because the `--json` stream omits tool arguments,
        reasoning and token usage (measured: it carries `tool.result` text,
        `run.output.delta` and the terminal event, and nothing else of use).

    What this sandbox does differently, for the same reasons as `codex.py`:

      * `install()` RETURNS a check/install pair; the runner runs it;
      * the session log is READ inside the container and its records returned
        in the one JSON line the runner owes the host -- there is no bind mount;
      * the task's grader is hosted BY THE RUNNER over loopback streamable HTTP
        and registered through the CLI's `settings.json` `mcp_servers` table by
        URL only. Muse has no `mcp` subcommand and `muse plugins` says "plugins
        are not available in this build"; the settings table is the one route,
        and it was verified end to end against this repo's own `GraderServer`:
        the model's tool spec then carries a `mcp__grader` namespace with
        `grade_deliverable` inside it.
    """
    TURN_LIMIT = 'model steps'

    @staticmethod
    def name() -> str:
        return "muse-code"

    # ---- what this adapter can hold shut ----------------------------------
    #
    # BOTH keys, through ONE knob, `--disable-web-tools`. Verified ON THE WIRE
    # rather than by asking the model: the tool spec Muse 1.0.3 sends to the
    # provider (`tools[0]` is a `namespace` named `muse` listing every function)
    # was captured with and without the flag on the same prompt, and the diff
    # is exactly one function, `web_search`:
    #
    #   flag omitted          workflow, read_file, search, write_file, edit_file,
    #                         read_memory, add_memory, edit_memory, web_search,
    #                         bash, bash_input, cron_*, get_goal, create_goal,
    #                         update_goal, report_progress, request_user_input,
    #                         subagent_*, read_skill, snooze_reminder, write_todos
    #   --disable-web-tools   the same list with web_search ABSENT
    #
    # There is no separate fetch function in this build's spec, so the one
    # search function is the whole vendor-side web surface and the one flag
    # closes it. As with Codex, one knob means the two keys cannot disagree.
    #
    # The session metadata records `web_search_mode: client`, which means the
    # CLI itself calls Meta's search endpoint (`/muse-code/search/`) -- the
    # request leaves the container's namespace for Meta's servers, so the tool
    # is NOT contained by the sandbox and must be off for a closed-book task,
    # which is exactly why the flag is always emitted.
    #
    # STILL PENDING, and stated so it is not mistaken for done: the mirror-log
    # canary on a real openmath episode (a request for the planted URL arriving
    # in-namespace). It needs a working META_API_KEY; the tool-spec evidence
    # above is the same enumeration check Codex passed first.
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}

    # ---- where the model is reached, and with which key --------------------
    #
    # TWO ROUTES, decided by the model entry's `api_base_url:` (core/routing.py)
    # or, failing that, the MUSE_BASE_URL passthrough:
    #
    #   native    no base URL, or one whose host is api.meta.ai. The CLI talks
    #             to Meta's front door itself with `Authorization: Bearer
    #             $META_API_KEY`. The model id is handed over bare (one leading
    #             `meta/` stripped, as Harbor does).
    #   gateway   any other origin. The runner serves the CLI's required
    #             `/muse-code/models` catalog on loopback and relays native
    #             Responses requests to the configured base's `/v1/responses`.
    #             The full model id (e.g. `meta/muse-spark-1.3`) is retained.
    #             Shared forwarder.Upstream supplies the gateway bearer and
    #             merges extra_body; its response relay preserves reasoning,
    #             tool namespaces, images, streaming events and HTTP errors.
    #             The gateway must support the configured model through the
    #             Responses API. There is no Chat Completions translation.
    #             The CLI trusts loopback with a per-episode META_API_KEY;
    #             the real gateway key stays in the runner process.
    #
    # A native Meta entry with extra_body also uses the loopback proxy so those
    # settings reach every request. It keeps the bare Meta model id.
    #
    # The credential the CLI reads is always META_API_KEY inside the container
    # (CLI_KEY_ENV); which HOST variable fills it is the entry's `api_key_env:`
    # (OPENAI_API_KEY for the gateway) or this default for native Meta.
    API_KEY_ENV = "META_API_KEY"
    CLI_KEY_ENV = ("META_API_KEY",)
    CLI_BASE_URL_ENV = ("MUSE_BASE_URL",)
    SUPPORTS_FORWARDER = True
    NATIVE_HOSTS = ("api.meta.ai",)

    # Host variables copied into the container when set.
    #
    #   MUSE_BASE_URL    the pre-routing fallback for an entry that sets no
    #                    `api_base_url:` (CLI_BASE_URL_ENV, see above).
    #   OPENAI_API_KEY / OPENAI_BASE_URL
    #                    for the TASK's grader (gdpval's asks a vision judge for
    #                    feedback through litellm), which runs inside the runner
    #                    process. The runner strips both from the environment it
    #                    hands `muse`, so the agent's shell does not inherit a
    #                    key it has no use for.
    PASSTHROUGH_ENV = ("MUSE_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL")
    GRADER_ONLY_ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL")

    # The exact `muse exec` line, the session log it was read back from, the
    # route taken, the shim's per-request summary (method, path, upstream
    # status, sizes -- never bodies), and the token split between the main
    # session and the CLI's own reminder subagents (see USAGE_KEYS).
    EXTRA_RECORD_KEYS = ("muse_command", "session_path", "route", "shim_calls",
                         "usage_main", "usage_subagents", "n_subagent_sessions")

    # ---- what an episode cost ---------------------------------------------
    #
    # Muse prices nothing. Each model call is closed by a `model_completed`
    # event whose `usage` is the provider's own Responses-API usage, as the
    # runtime spells it:
    #
    #   {"input_tokens", "output_tokens", "cached_tokens", "cache_read_tokens",
    #    "cache_write_tokens", "reasoning_tokens"}
    #
    # Measured against a loopback provider returning Responses-API usage
    # (`input_tokens: 1200, input_tokens_details.cached_tokens: 1000`): the log
    # carries input_tokens 1200 and cache_read_tokens 1000, i.e. the input count
    # is passed through INCLUSIVE of its cache legs, which is the OpenAI
    # Responses convention Meta's API follows. `reasoning_tokens` is a subset of
    # `output_tokens` for the same reason and is never added on top.
    #
    # The runner sums those events over the MAIN session and over the CLI's
    # own reminder/subagent sessions (Muse spawns skill/goal/verify reminder
    # children that make their own model calls on the same key -- three per
    # episode in every run measured) and records the split. `usage` is the
    # total, because that is what the key was billed for.
    USAGE_FIELDS = ("usage",)
    USAGE_KEYS = ("input_tokens", "output_tokens",
                  "cache_read_tokens", "cache_write_tokens")
    PROMPT_TOKENS_INCLUDE_CACHE = True

    def reported_cost_usd(self, raw: dict) -> float | None:
        """None, always: Muse never prices its own episodes.

        litellm's bundled table carries `meta/muse-spark-1.3` at $1.25/M in,
        $4.25/M out, $0.15/M cached (verified against litellm 1.96.0), and
        `cost_from_usage` resolves the model id exactly as written in
        configs/models.yaml, so the shared estimate needs no local price entry.
        """
        return None

    # ---- this adapter's runtime -------------------------------------------
    INSTALL_HOSTS = ('dev.meta.ai', 'api.meta.ai', 'lookaside.facebook.com')
    INSTALL_URL = "https://dev.meta.ai/install.sh"
    LAUNCHER_URL = "https://api.meta.ai/muse-launcher.sh"
    # Where the installer puts the launcher; the launcher then downloads the
    # real binary next to itself. Hosts contacted, measured from the scripts:
    #   dev.meta.ai              install.sh
    #   api.meta.ai              muse-launcher.sh, the muse-stable channel
    #                            manifest, and the Model API itself
    #   lookaside.facebook.com   the ~264 MB static binary
    PATH_EXPORT = 'export PATH="$HOME/.local/bin:$PATH"'

    # `muse --version` prints `Muse Code 1.0.3 (1.0.3-R2198.1)`; the
    # parenthesised release string is the thing a run should record.
    VERSION_CMD = (PATH_EXPORT + '; muse --version 2>/dev/null | head -1 | '
                   "sed -n 's/.*(\\(.*\\)).*/\\1/p'")
    INSTALL_CHECK = PATH_EXPORT + "; command -v muse >/dev/null 2>&1"

    # Where the episode keeps Muse's state inside the container. Both are
    # per-episode and removed afterwards. XDG_CONFIG_HOME holds settings.json
    # (the grader URL and the base-URL pin); XDG_DATA_HOME holds the session
    # store the runner reads back. No credential is ever written to either --
    # auth is META_API_KEY in the environment only.
    CONFIG_HOME = "/tmp/muse-config"
    DATA_HOME = "/tmp/muse-data"
    PROMPT_DIR = "/tmp/muse-prompt"

    # `muse exec` flags that are not negotiable. Harbor's, in Harbor's order.
    #   --json                     JSONL events on stdout
    #   --yolo                     no approval prompts, no OS sandbox, workspace
    #                              trusted: the container is the sandbox
    #   --user-input-auto-resolve  a clarification request cannot hang a
    #                              headless run
    EXEC_FLAGS = ("--json", "--yolo", "--user-input-auto-resolve")

    # Environment fixed for every episode.
    #   MUSE_NO_AUTO_UPDATE   Harbor's: the launcher must not replace the
    #                         binary mid-job
    #   TBH_DISABLE_TELEMETRY the runtime's own telemetry switch (its settings
    #                         schema also has a `telemetry` block); off for the
    #                         same reason codex runs with analytics disabled
    FIXED_ENV = {"MUSE_NO_AUTO_UPDATE": "1", "TBH_DISABLE_TELEMETRY": "1"}

    # THIS adapter's delivery of the model's `generation_config:`.
    # `reasoning_effort` -> `--reasoning-effort`; the CLI's own default is
    # `high` and it is applied explicitly so run.json states it.
    DEFAULT_REASONING_EFFORT = "high"
    REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high",
                                "xhigh", "max", "ultra")
    GENERATION_KEYS = ("reasoning_effort",)

    # Muse's own toolset, as the spec on the wire names it (namespace `muse`).
    # The first group runs INSIDE the container; `web_search` is the one that
    # does not, and the one the policy switches.
    BASE_TOOLS = ("bash", "bash_input", "read_file", "search", "write_file",
                  "edit_file", "workflow", "subagent_spawn", "read_memory",
                  "add_memory", "edit_memory", "write_todos")
    WEB_TOOL = "web_search"

    # Muse's own words for the failures a run can end in, ahead of the shared
    # list. All three were observed on stderr / in the terminal event.
    ERROR_PATTERNS = [
        ErrorPattern(r"your API key from META_API_KEY was rejected",
                     errors.AgentAuthenticationError),
        ErrorPattern(r"access has been restricted due to repeated policy "
                     r"violations|user_blocked",
                     errors.AgentAuthenticationError),
        ErrorPattern(r"failed to fetch model catalog", errors.UnknownApiError),
        ErrorPattern(r"Provider returned malformed response data",
                     errors.UnknownApiError),
        # The runner's own verdict on a "completed" run with zero model output
        # (muse_code_runner.empty_completion_error).
        ErrorPattern(r"provider returned empty completions",
                     errors.UnknownApiError),
        ErrorPattern(r"step limit|stepLimit", errors.MaxTurnsError),
        *errors.ERROR_PATTERNS,
    ]

    RUNNER = Path(__file__).resolve().parent / "muse_code_runner.py"
    # The shared forwarder, whose `Upstream` is the shim's outbound leg. Read
    # as source into the blob for a gateway or extra_body overrides.
    FORWARDER = Path(__file__).resolve().parent / "forwarder.py"

    # ---- install ----------------------------------------------------------
    def install_check(self) -> str:
        if self.pinned_version:
            return (self.PATH_EXPORT + '; export MUSE_NO_AUTO_UPDATE=1; '
                    f'[ "$({self.VERSION_CMD})" = {shlex.quote(self.pinned_version)} ]')
        return self.INSTALL_CHECK

    def install_script(self) -> str:
        if self.pinned_version:
            source = Path(__file__).with_name("muse_install.py").read_text()
            return "\n".join([
                "set -e",
                "python3 -c " + shlex.quote(source) + " " + shlex.quote(self.pinned_version)
                + ' "$HOME/.local/bin/muse"',
                self.PATH_EXPORT,
                "export MUSE_NO_AUTO_UPDATE=1",
                "muse --version",
                link_bins("muse"),
            ])
        return "\n".join([
            "set -o pipefail",
            'muse_installer=$(mktemp) || exit $?',
            f'if curl -fsSL {self.INSTALL_URL} -o "$muse_installer" '
            '&& bash -n "$muse_installer" '
            '&& MUSE_NO_MODIFY_PATH=1 bash "$muse_installer"; then',
            '  rm -f "$muse_installer"',
            'else',
            '  rm -f "$muse_installer"',
            '  echo "Muse installer unavailable; using official Meta launcher" >&2',
            '  mkdir -p "$HOME/.local/bin" || exit $?',
            '  muse_launcher=$(mktemp "$HOME/.local/bin/muse.XXXXXX") || exit $?',
            f'  if curl -fsSL {self.LAUNCHER_URL} -o "$muse_launcher" '
            '&& bash -n "$muse_launcher" && chmod 755 "$muse_launcher"; then',
            '    mv "$muse_launcher" "$HOME/.local/bin/muse" || exit $?',
            '  else',
            '    rm -f "$muse_launcher"',
            '    echo "Official Muse launcher download failed" >&2',
            '    exit 1',
            '  fi',
            'fi',
            self.PATH_EXPORT,
            # Makes the launcher download the real binary now, at install time.
            "muse --version || exit $?",
            link_bins("muse"),
        ])

    # ---- the task's tool policy -------------------------------------------
    def web_enabled(self) -> bool:
        """Whether `web_search` is offered. Both policy keys, or neither."""
        if self.policy is None:
            raise RuntimeError("apply_tool_policy has not been called")
        search, fetch = self.policy["web_search"], self.policy["web_fetch"]
        if search != fetch:
            raise ValueError(
                f"adapter {self.name()!r} has ONE web knob "
                f"(`--disable-web-tools`) and Muse Code 1.0.3 has ONE web "
                f"function (`web_search`), so web_search and web_fetch cannot "
                f"differ; this task asks for web_search={search}, "
                f"web_fetch={fetch}")
        return search

    def _tools_for(self, policy: dict) -> list[str]:
        if policy.get("mcp_only"):
            return []
        return list(self.BASE_TOOLS) + ([self.WEB_TOOL] if self.web_enabled()
                                        else [])

    @property
    def reasoning_effort(self) -> str:
        return str(self.generation_config.get("reasoning_effort")
                   or self.DEFAULT_REASONING_EFFORT)

    def resolved_generation_config(self) -> dict:
        return {"reasoning_effort": self.DEFAULT_REASONING_EFFORT,
                **self.generation_config}

    @property
    def native_origin(self) -> bool:
        """Whether the configured upstream is Meta, independent of the local gate."""
        from urllib.parse import urlsplit
        base = self.resolved_base_url()
        return not base or (urlsplit(base).hostname or "").lower() in self.NATIVE_HOSTS

    @property
    def route(self) -> str:
        """MCP-only runs always use the local response gate, including Meta runs."""
        if (self.policy or {}).get("mcp_only"):
            return "gateway"
        return "native" if self.native_origin else "gateway"

    @property
    def model_slug(self) -> str:
        """What goes after `--model`, which depends on who answers.

        native   Harbor's `_model_slug`: one leading `meta/` stripped, nothing
                 else -- Meta's API takes the bare id.
        gateway  the id IN FULL (`meta/muse-spark-1.3`): it is the
                 gateway's model-group name and the shim's catalog lists it
                 verbatim, so the CLI must ask for exactly that string.
        """
        if not self.native_origin:
            return self.model
        provider, _, rest = self.model.partition("/")
        return rest if provider == "meta" and rest else self.model

    def cli_flags(self) -> list[str]:
        """The variable part of the `muse exec` line.

        `--disable-web-tools` is emitted whenever the policy is closed, and
        the closed policy is what every task gets by omission -- there is no
        CLI default to fall back on and a test pins the line.
        """
        flags = ["--model", self.model_slug,
                 "--reasoning-effort", self.reasoning_effort]
        if not self.web_enabled():
            flags.append("--disable-web-tools")
        if (self.policy or {}).get("mcp_only"):
            # --disable-write also tells the model to avoid the task's allowed
            # MCP write_file. Native editors are removed by settings and the
            # response gate, independently of the chess file capability.
            flags += ["--disable-shell", "--no-foreign-personal-context"]
        if self.max_turns:
            # The same cap the Claude adapter applies through the SDK, spelled
            # the way this CLI spells it. Recorded on the command line.
            flags += ["--max-model-steps", str(int(self.max_turns))]
        base = self.resolved_base_url()
        if base and self.route == "native" and not self.extra_body:
            # An explicit api.meta.ai origin: the CLI's own sanctioned host, so
            # the flag keeps its bearer. When proxied, the runner appends the
            # loopback origin itself, leaving exactly one --base-url flag.
            flags += ["--base-url", base]
        return flags

    def setup(self) -> list[str]:
        problems = [] if self.RUNNER.exists() else [f"missing runner {self.RUNNER}"]
        if (self.route == "gateway" or self.extra_body) and not self.FORWARDER.exists():
            problems.append(f"proxied route needs {self.FORWARDER} (the shim's "
                            f"outbound leg) and it is missing")
        if self.pinned_version and not re.fullmatch(r"\d+\.\d+\.\d+-R\d+(?:\.\d+)?", self.pinned_version):
            problems.append("Muse version pinning requires the full release, e.g. 1.3.0-R3401.1")
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
            return problems
        problems += self.config_problems()
        if (self.policy or {}).get("mcp_only") and self.route != "gateway":
            problems.append("Muse MCP-only requires the gateway response gate")
        if self.reasoning_effort not in self.REASONING_EFFORT_CHOICES:
            problems.append(f"reasoning_effort={self.reasoning_effort!r}; "
                            f"muse takes {list(self.REASONING_EFFORT_CHOICES)}")
        try:
            web = self.web_enabled()
        except ValueError as e:
            return [*problems, str(e)]
        flags = self.cli_flags()
        if not web:
            if self.WEB_TOOL in self.tools:
                problems.append(f"web is closed but {self.WEB_TOOL} is still "
                                f"in the tool list")
            if "--disable-web-tools" not in flags:
                problems.append("closed-book policy did not produce "
                                "`--disable-web-tools`; the CLI's default tool "
                                "spec carries web_search, which runs against "
                                "Meta's search endpoint outside the container")
        elif "--disable-web-tools" in flags:
            problems.append("open-web policy still produced "
                            "`--disable-web-tools`")
        return problems

    # ---- the episode ------------------------------------------------------
    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
        """The stdin payload for one episode. Same shape as the Codex blob.

        With a gateway or extra_body the blob carries `core/agents/forwarder.py`
        as source under `modules.forwarder`, for the proxy's outbound leg.
        """
        route = self.route
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
            "muse": {
                "mcp_only": bool((self.policy or {}).get("mcp_only")),
                "model_slug": self.model_slug,
                "exec_flags": list(self.EXEC_FLAGS),
                "cli_flags": self.cli_flags(),
                "fixed_env": dict(self.FIXED_ENV),
                "grader_only_env": list(self.GRADER_ONLY_ENV),
                "config_home": self.CONFIG_HOME,
                "data_home": self.DATA_HOME,
                "prompt_dir": self.PROMPT_DIR,
                "path_export": self.PATH_EXPORT,
                # Inside the container the key always sits under META_API_KEY
                # (CLI_KEY_ENV), whichever host variable filled it.
                "api_key_env": self.CLI_KEY_ENV[0],
                "route": route,
                # "" means the CLI's default Meta endpoint. A proxy needs an
                # explicit upstream even when the entry uses that default.
                "base_url": self.resolved_base_url() or (
                    "https://api.meta.ai/v1" if self.extra_body else ""),
                "model_id": self.model,
                "extra_body": dict(self.extra_body),
            },
        }
        if grader_src:
            task["row"] = ({**row, "restricted_tools": True, "episode_timeout_s": timeout_s}
                           if (self.policy or {}).get("mcp_only") else row)
        modules = {"stage": stage_src, "grader": grader_src}
        modules["support"] = self.RUNNER.with_name("runner_support.py").read_text()
        modules["session_observer"] = self.RUNNER.with_name("session_observer.py").read_text()
        if route == "gateway" or self.extra_body:
            modules["forwarder"] = self.forwarder_source()
        return json.dumps({
            "task": task,
            "code": self.RUNNER.read_text(),
            "modules": modules,
        }).encode()

    # ---- the session log as ATIF ------------------------------------------
    #
    # A Muse session log is one JSON envelope per line:
    #
    #   {"schema_version":1, "id":..., "stream":{"kind":"session","id":<sid>},
    #    "sequence":N, "recorded_at":<microseconds>, "record_type":...,
    #    "payload_type":"runtime.session", "payload":{"kind":"run",
    #    "run_id":..., "event":{"kind":..., ...}}}
    #
    # The `payload_type == "runtime.session"`, `payload.kind == "run"` records
    # are the conversation. `event.kind` values this converter reads, all
    # observed on a complete two-turn episode:
    #
    #   started                       {prompt}                   the user turn
    #   model_response_created        {response_id}              one model call
    #   reasoning_summary_committed   {response_id, text}        readable CoT
    #   reasoning_committed           {response_id, text,        raw CoT; text is
    #                                  encrypted_content}         "" when hidden
    #   assistant_tool_calls_committed{response_id, message_id,
    #                                  tool_calls:[{id, call_id, name, args}]}
    #   tool_result_batch_committed   {batch_id (== message_id),
    #                                  results:[{tool_call_id, text}]}
    #   assistant_message_committed   {response_id, text}
    #   model_completed               {usage, finish_reason, model}
    #   terminal                      {terminal, reason}
    #
    # and `payload.kind == "task"` with `event.kind == "failed"` carries the
    # reason an API call failed (`your API key ... was rejected`).
    #
    # One ATIF step per model response, as the Codex converter makes one per
    # api_call_id: reasoning + text + tool calls + their results + usage.
    RUN = "runtime.session"

    @staticmethod
    def _iso(recorded_at) -> str | None:
        """`recorded_at` is microseconds since the epoch."""
        if not isinstance(recorded_at, (int, float)):
            return None
        try:
            return datetime.fromtimestamp(recorded_at / 1e6,
                                          tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    @staticmethod
    def _args(raw) -> dict:
        """A tool call's `args` (a JSON string) as a dict."""
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        if raw is None:
            return {}
        return {"value": raw}

    @staticmethod
    def _metrics(usage: dict | None) -> dict | None:
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("input_tokens")
        cached = usage.get("cache_read_tokens")
        if cached is None:
            cached = usage.get("cached_tokens")
        extra = {k: usage.get(k) for k in
                 ("reasoning_tokens", "cache_write_tokens", "cached_tokens")
                 if usage.get(k) is not None}
        return {"prompt_tokens": prompt if prompt else None,
                "completion_tokens": usage.get("output_tokens") or None,
                "cached_tokens": cached or None,
                "extra": extra or None}

    def to_trajectory(self, raw: dict) -> Trajectory:
        raw = raw or {}
        events = [e for e in (raw.get("messages") or []) if isinstance(e, dict)]

        session_id = raw.get("session_id")
        version = self.version() or "unknown"
        model = None
        agent_extra: dict = {}

        # Every response, in order, with what belongs to it.
        responses: list[dict] = []
        by_id: dict[str, dict] = {}
        current: dict | None = None
        # tool call message_id -> the response that issued the calls
        batch_owner: dict[str, dict] = {}
        user_prompt: str | None = None
        user_ts: str | None = None
        terminal: dict | None = None
        failures: list[str] = []

        def open_response(rid: str | None, ts: str | None) -> dict:
            nonlocal current
            key = rid or f"response_{len(responses) + 1}"
            if key in by_id:
                current = by_id[key]
                return current
            current = {"id": key, "timestamp": ts, "reasoning": [],
                       "encrypted_reasoning": False, "text": "",
                       "tool_calls": [], "results": {}, "usage": None,
                       "finish_reason": None, "model": None}
            responses.append(current)
            by_id[key] = current
            return current

        for rec in events:
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            ptype = rec.get("payload_type")
            ts = self._iso(rec.get("recorded_at"))
            stream = rec.get("stream")
            if session_id is None and isinstance(stream, dict):
                session_id = stream.get("id")

            if ptype == "runtime.session.metadata":
                record = payload.get("record") or {}
                build = record.get("build") or {}
                if isinstance(build.get("semver"), str):
                    version = build["semver"]
                model = record.get("model_id") or model
                for key in ("provider_id", "workspace_root", "web_search_mode",
                            "tool_surface_version"):
                    if record.get(key) is not None:
                        agent_extra[key] = record[key]
                continue

            if ptype != self.RUN:
                continue
            kind = payload.get("kind")
            event = payload.get("event")
            if not isinstance(event, dict):
                continue
            ekind = event.get("kind")

            if kind == "task":
                if ekind == "failed" and event.get("reason"):
                    failures.append(str(event["reason"]))
                continue
            if kind != "run":
                continue

            if ekind == "started" and "prompt" in event:
                user_prompt, user_ts = event.get("prompt"), ts
            elif ekind == "model_response_created":
                open_response(event.get("response_id"), ts)
            elif ekind == "reasoning_summary_committed":
                r = open_response(event.get("response_id"), ts)
                if event.get("text"):
                    r["reasoning"].append(event["text"])
            elif ekind == "reasoning_committed":
                r = open_response(event.get("response_id"), ts)
                if event.get("text"):
                    r["reasoning"].append(event["text"])
                elif event.get("encrypted_content"):
                    r["encrypted_reasoning"] = True
            elif ekind == "assistant_tool_calls_committed":
                r = open_response(event.get("response_id"), ts)
                for call in event.get("tool_calls") or []:
                    if isinstance(call, dict):
                        r["tool_calls"].append(call)
                if event.get("message_id"):
                    batch_owner[event["message_id"]] = r
            elif ekind == "tool_result_batch_committed":
                owner = batch_owner.get(event.get("batch_id")) or current
                if owner is None:
                    owner = open_response(None, ts)
                for res in event.get("results") or []:
                    if isinstance(res, dict):
                        owner["results"][res.get("tool_call_id")] = res
            elif ekind == "assistant_message_committed":
                r = open_response(event.get("response_id"), ts)
                text = event.get("text")
                if isinstance(text, str) and text:
                    r["text"] = f"{r['text']}\n\n{text}" if r["text"] else text
            elif ekind == "model_completed":
                r = current or open_response(None, ts)
                r["usage"] = event.get("usage")
                r["finish_reason"] = event.get("finish_reason")
                r["model"] = event.get("model")
                if model is None and isinstance(event.get("model"), str):
                    model = event["model"]
            elif ekind == "terminal":
                terminal = {"terminal": event.get("terminal"),
                            "reason": event.get("reason")}

        model = model or self.model

        steps: list[Step] = []
        if user_prompt is not None:
            steps.append(Step(step_id=1, timestamp=user_ts, source="user",
                              message=user_prompt))

        for r in responses:
            calls: list[ToolCall] = []
            results: list[ObservationResult] = []
            details: dict = {}
            matched: set = set()
            for spec in r["tool_calls"]:
                call_id = spec.get("call_id") or spec.get("id") or ""
                calls.append(ToolCall(tool_call_id=call_id,
                                      function_name=spec.get("name") or "",
                                      arguments=self._args(spec.get("args"))))
                res = r["results"].get(call_id)
                if isinstance(res, dict):
                    # A call whose result has not arrived (the live tail sees
                    # the call record before the result record) gets no
                    # observation rather than an empty one.
                    matched.add(call_id)
                    results.append(ObservationResult(
                        source_call_id=call_id or None, content=res.get("text")))
                if spec.get("id"):
                    details[call_id] = {"item_id": spec["id"]}
            for call_id, res in r["results"].items():
                # A result whose call is in an earlier record: the live tail's
                # case, rendered as a bare `RESULT[tool]` exactly as Codex's
                # tail is. ATIF requires `source_call_id` to name a call in
                # THIS step, so the id travels in `extra` instead.
                if call_id not in matched and isinstance(res, dict):
                    results.append(ObservationResult(
                        source_call_id=None, content=res.get("text"),
                        extra={"tool_call_id": call_id} if call_id else None))
            extra: dict = {"response_id": r["id"]}
            if r["finish_reason"]:
                extra["finish_reason"] = r["finish_reason"]
            if r["encrypted_reasoning"] and not r["reasoning"]:
                # The provider returned reasoning the CLI cannot show. Saying so
                # keeps "no reasoning delivered" distinct from "none happened".
                extra["reasoning_withheld"] = True
            if details:
                extra["tool_call_details"] = details
            metrics = self._metrics(r["usage"])
            steps.append(Step(
                step_id=len(steps) + 1, timestamp=r["timestamp"], source="agent",
                message=r["text"],
                model_name=r["model"] or model,
                reasoning_content="\n".join(r["reasoning"]) or None,
                tool_calls=calls or None,
                observation=Observation(results=results) if results else None,
                metrics=Metrics(**metrics) if metrics else None,
                llm_call_count=1,
                extra=extra,
            ))

        if not steps:
            steps = [self.empty_step(
                "(no session records were captured for this episode)")]

        # -- totals over the main session's model calls -----------------------
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                  "cache_write_tokens": 0, "reasoning_tokens": 0}
        n_calls = 0
        for r in responses:
            usage = r["usage"]
            if not isinstance(usage, dict):
                continue
            n_calls += 1
            for key in totals:
                value = usage.get(key)
                if key == "cache_read_tokens" and value is None:
                    value = usage.get("cached_tokens")
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
        final_metrics = None
        if n_calls:
            final_metrics = FinalMetrics(
                total_prompt_tokens=totals["input_tokens"] or None,
                total_completion_tokens=totals["output_tokens"] or None,
                total_cached_tokens=totals["cache_read_tokens"] or None,
                total_cost_usd=raw.get("cost_usd"),
                total_steps=len(steps),
                extra={"reasoning_tokens": totals["reasoning_tokens"],
                       "cache_write_tokens": totals["cache_write_tokens"],
                       "model_calls": n_calls,
                       # The CLI's own reminder children, when the record
                       # carries their tally; the main session's totals above
                       # never include them.
                       **({"usage_subagents": raw["usage_subagents"]}
                          if raw.get("usage_subagents") else {})},
            )

        notes: list[str] = []
        if terminal and terminal.get("terminal") not in (None, "completed"):
            notes.append(f"run ended {terminal['terminal']}: "
                         f"{terminal.get('reason') or '(no reason given)'}")
        for reason in failures:
            notes.append(f"task failed: {reason}")

        final_text = raw.get("final_text") or ""
        if not final_text:
            for step in reversed(steps):
                text = step.message if isinstance(step.message, str) else ""
                if step.source == "agent" and text.strip():
                    final_text = text
                    break

        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            agent=ATIFAgent(name=self.name(), version=version,
                            model_name=model, extra=agent_extra or None),
            steps=steps,
            notes="; ".join(notes) if notes else None,
            final_metrics=final_metrics,
            extra={"final_text": final_text} if final_text else None,
        )
