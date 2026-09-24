"""The Claude Agent SDK adapter.

One adapter, one module -- Harbor's layout
(`harbor/src/harbor/agents/installed/claude_code.py`). Its in-container runner
is the sibling `claude_sdk_runner.py`, which is read as source and streamed into
the container on stdin; it is never imported here and never written to the
container filesystem.
"""
from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from urllib.parse import urlsplit

from ..trajectory import (
    SCHEMA_VERSION,
    ContentPart,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from .installed import NODE_MAJOR, InstalledAgent, link_bins, node_install


class ClaudeSDKAgent(InstalledAgent):
    """The Claude Agent SDK, driven by `claude_sdk_runner.py`.

    The runner is streamed into the container on stdin with the harness modules
    it needs. Nothing of ours is ever written to the container filesystem, so an
    agent that runs `ls /`, `find / -name '*.py'` or reads /proc finds no trace
    of the harness, the prompt, the rubric or the answer key.
    """
    TURN_LIMIT = 'assistant turns'
    PERMISSION_MODES = ("bypassPermissions", "default", "acceptEdits", "plan", "dontAsk")

    @staticmethod
    def name() -> str:
        return "claude-sdk"

    # The SDK takes an explicit allowed/disallowed tool list, so both keys are
    # enforced by construction. Verified empirically, not assumed.
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}

    API_KEY_ENV = "ANTHROPIC_API_KEY"
    # Host variables copied into the container. run.py builds the container's
    # ENTIRE environment from this tuple plus the one credential, so a vendor
    # knob that is not named here does not exist inside the container no matter
    # what the host has exported.
    #
    #   CLAUDE_CODE_MAX_OUTPUT_TOKENS  the 128k output ceiling, which lives in
    #                                  .env and applies to every run.
    #   ANTHROPIC_BASE_URL             which endpoint the SDK's `claude` CLI
    #                                  talks to. Unset means the vendor's own
    #                                  API; set, it is how a run is pointed at a
    #                                  proxy (the SDK appends `/v1/messages`, so
    #                                  the value is a bare origin). Without it
    #                                  here an ANTHROPIC_BASE_URL exported on the
    #                                  host is dropped at the container boundary
    #                                  and the episode silently goes direct.
    #
    # THE COPY IS BEST-EFFORT: run.py skips any name that is unset, so a typo'd
    # or misspelled variable is not an error -- it is silently absent, and the
    # episode runs with the vendor default while the host config looks right.
    # A name added here is only as good as the measurement that it arrived.
    PASSTHROUGH_ENV = ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "ANTHROPIC_BASE_URL")

    # Per-model routing (configs/models.yaml `api_key_env:` / `api_base_url:` /
    # `extra_body:`, see core/agents/base.py). The CLI reads its key from
    # ANTHROPIC_API_KEY whatever host variable the entry named, and its base
    # URL from ANTHROPIC_BASE_URL, which `routed_env()` sets from the entry's
    # `api_base_url:` (else the passthrough above stands). With `extra_body`
    # the runner sets ANTHROPIC_BASE_URL to core/agents/forwarder.py's loopback
    # URL instead, and the forwarder merges the body on its way to the real URL.
    CLI_KEY_ENV = ("ANTHROPIC_API_KEY",)
    CLI_BASE_URL_ENV = ("ANTHROPIC_BASE_URL",)
    SUPPORTS_FORWARDER = True
    EXTRA_RECORD_KEYS = ("forwarder_calls",)

    # ---- what an episode cost ---------------------------------------------
    #
    # The SDK prices its own episodes, and that figure -- `ResultMessage.
    # total_cost_usd`, the CLI's own accounting -- is AUTHORITATIVE and passes
    # through untouched. The base class's litellm estimate is a FALLBACK and
    # nothing more; `Agent.episode_cost` only reaches for it when the reported
    # number is absent, and records which of the two it used.
    #
    # It is absent in exactly one situation, and it is not a rare one: the host
    # kills the episode at the timeout cap, so `ResultMessage` never arrives and
    # neither does `usage`. Every capped episode used to report no cost at all
    # while having spent real money. That is why the runner sets
    # `include_partial_messages=True` and tallies `message_delta` usage into
    # `stream_usage` -- it is the only token count that survives, which is why
    # it is the FIRST usage source here and the ResultMessage's `usage` the
    # second. (The two disagree slightly even on a clean episode: the stream
    # tally counts every turn, where ResultMessage.usage reports the last one's.
    # The stream tally is the one that matches the money.)
    #
    # Anthropic's `input_tokens` is EXCLUSIVE -- the fresh remainder only, with
    # `cache_creation_input_tokens` and `cache_read_input_tokens` reported
    # beside it, not inside it. On the saved run that reported $0.4466975 the
    # split was input 20, cache write 30682, cache read 69450, output 8579:
    # pricing 20 tokens as the whole prompt would have missed 99.98% of the
    # input. `PROMPT_TOKENS_INCLUDE_CACHE = False` is what makes the base class
    # add the two cache legs back in before pricing.
    USAGE_FIELDS = ("stream_usage", "usage")
    USAGE_KEYS = ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens")
    PROMPT_TOKENS_INCLUDE_CACHE = False

    def reported_cost_usd(self, raw: dict) -> float | None:
        """`ResultMessage.total_cost_usd`, as the runner recorded it.

        Harbor's `claude_code.py::_parse_total_cost_from_stream_json` reads the
        same field off the CLI's stream-json; ours is already on the record, so
        this is a lookup rather than a parse.
        """
        cost = (raw or {}).get("cost_usd")
        return float(cost) if isinstance(cost, (int, float)) else None

    # ---- this adapter's runtime -------------------------------------------
    #
    # Three things, none of which the image carries:
    #
    #   Node 22    The CLI declares `node >=22`; Debian slim ships 20, which
    #              installs with five lines of EBADENGINE and runs on a runtime
    #              its author does not support. Node is the AGENT's dependency,
    #              not the environment's, so the adapter brings its own via nvm
    #              -- the same thing Harbor does for its nine node agents.
    #   claude     the CLI the SDK drives.
    #   the SDK    `claude_agent_sdk`, which the in-container runner imports.
    #              It is vendor-coupled the same way the CLI is, and an agent
    #              running `pip list` in an image that shipped it would find the
    #              harness's own SDK sitting there. It belongs here.
    #
    # ALL THREE ARE CACHEABLE. The check is a plain presence test, so an image
    # somebody DID pre-bake any of them into is a pure cache: the check
    # succeeds, the install never runs, and the episode starts as before. Adding
    # `nodejs npm` or the pip pins back to the Dockerfile is therefore a
    # performance choice and never a correctness one.
    #
    # Version pinning follows Harbor: unset means "latest at episode time", a
    # pin interpolates `@<version>` onto the package AND makes the check
    # version-aware, so an image or a cache carrying the wrong build is
    # reinstalled rather than accepted. Either way VERSION_CMD reports what
    # actually ran and the episode records it.
    #
    # Hygiene: the npm cache goes to a scratch directory and is deleted, because
    # npm otherwise leaves ~/.npm/_logs with a debug file naming the package it
    # just fetched. `~/.nvm` is left in place -- it IS the node install, and a
    # version manager in a developer's home directory is unremarkable. Neither
    # is an eval tell: the agent may freely know which agent it is.
    NODE_MAJOR = NODE_MAJOR
    PACKAGE = "@anthropic-ai/claude-code"
    # The runner imports claude_agent_sdk; anthropic is pinned alongside it
    # because the SDK's own dependency range would otherwise float.
    SDK_PINS = ("claude-agent-sdk==0.2.135", "anthropic==0.121.0")
    VERSION_CMD = "claude --version 2>/dev/null | awk '{print $1}'"

    def install_check(self) -> str:
        parts = ["command -v claude >/dev/null 2>&1",
                 'python -c "import claude_agent_sdk" >/dev/null 2>&1']
        if self.pinned_version:
            # Pinned means the RIGHT build, not any build: a mismatch has to
            # reinstall, or a stale cache would silently decide the run.
            parts.append(f'[ "$({self.VERSION_CMD})" = '
                         f"{shlex.quote(self.pinned_version)} ]")
        return " && ".join(parts)

    def install_script(self) -> str:
        pkg = self.PACKAGE + (f"@{self.pinned_version}" if self.pinned_version else "")
        return "\n".join([
            *node_install(self.NODE_MAJOR),
            "pip install --no-cache-dir --disable-pip-version-check "
            + " ".join(self.SDK_PINS) + " || exit $?",
            'npm_config_cache="$(mktemp -d)"; export npm_config_cache',
            f"npm install -g --no-fund --no-audit {pkg}",
            'rc=$?; rm -rf "$npm_config_cache" "$HOME/.npm"',
            link_bins("node", "npm", "npx", "claude"),
            "exit $rc",
        ])

    # This adapter's own toolset, not an operator's choice.
    #
    # BASE_TOOLS is everything that runs INSIDE the container: a shell, and file
    # read/write. Every byte of network traffic they generate crosses the
    # container's own network namespace, so it is jailed and observable.
    #
    # OPTIONAL_TOOLS run on Anthropic's servers instead, outside that
    # namespace. This dict IS the translation table: a policy key only ever ADDS
    # its tool, and only when true.
    BASE_TOOLS = ("Bash", "Read", "Write")
    OPTIONAL_TOOLS = {"web_search": "WebSearch", "web_fetch": "WebFetch"}

    # THIS adapter's delivery of the model's `generation_config:`. The same
    # settings the codex adapter delivers as `-c` flags and the judge delivers
    # as litellm kwargs arrive here as ClaudeAgentOptions fields, passed through
    # by name (see claude_sdk_runner.py). A key not listed here is refused at
    # preflight rather than dropped on the floor: `thinking` reaching nothing is
    # exactly the failure that made a whole run's transcripts look as though the
    # model never reasoned.
    GENERATION_KEYS = ("thinking", "effort")

    # The runner is a sibling module, read as SOURCE and never imported here:
    # it runs inside the container, streamed in on stdin.
    RUNNER = Path(__file__).resolve().parent / "claude_sdk_runner.py"

    def _tools_for(self, policy: dict) -> list[str]:
        if policy["mcp_only"]:
            if policy["web_search"] or policy["web_fetch"]:
                raise ValueError("mcp_only cannot enable native web tools")
            return []
        return list(self.BASE_TOOLS) + [tool for key, tool
                                        in self.OPTIONAL_TOOLS.items()
                                        if policy[key]]

    @property
    def disallowed(self) -> list[str]:
        """Held shut on top of the runner's own defaults.

        Belt and braces: a tool whose key is false is already absent from
        `tools`, and naming it here as well means the SDK refuses it even if
        something later widened the allowed list.
        """
        if self.policy is None:
            raise RuntimeError("apply_tool_policy has not been called")
        blocked = [tool for key, tool in self.OPTIONAL_TOOLS.items() if not self.policy[key]]
        if self.policy["mcp_only"]:
            blocked += ["Bash", "Read", "Write", "Edit", "NotebookEdit", "Task", "Agent",
                        "Glob", "Grep", "Skill", "ToolSearch", "TodoWrite"]
        return blocked

    def setup(self) -> list[str]:
        problems = [] if self.RUNNER.exists() else [f"missing runner {self.RUNNER}"]
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
            return problems
        for key, tool in self.OPTIONAL_TOOLS.items():
            if not self.policy[key] and tool in self.tools:
                problems.append(f"{key} is false but {tool} is still allowed")
        # The same guarantee codex has always given: a configured knob this
        # adapter cannot deliver stops the run here, before any container, and
        # is never dropped in silence.
        return problems + self.config_problems()

    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
        """The stdin payload for one episode.

        `grader_src` is "" for a task that ships no grader, and the key is then
        carried as the empty string rather than dropped. Carrying it keeps the
        payload one shape: the runner asks "is modules['grader'] non-empty",
        which is a question with an answer, instead of "is the key there", which
        cannot tell "no grader" apart from "a bug dropped the key".
        """
        upstream = self.resolved_base_url()
        # Provider-qualified IDs are gateway routing syntax. Anthropic's native
        # endpoint expects the bare Claude model ID.
        model = (self.model.removeprefix("anthropic/")
                 if not upstream or urlsplit(upstream).hostname == "api.anthropic.com"
                 else self.model)
        task = {
            "id": row["id"],
            "model": model,
            "content": prompt,
            # The whole row, including anything the grader needs to grade with.
            # It rides on stdin only: never argv, never environ, never a file.
            "row": {**row, "episode_timeout_s": timeout_s} if row.get("tool_surface") in {"chess", "go"} else row,
            "files": files,
            "skip_dirs": list(skip_dirs),
            "deliverable_files": list(row.get("deliverable_files") or ()),
            "tools": self.tools,
            "disallowed_tools": self.disallowed,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
            "timeout_s": timeout_s,
            # The model's generation settings, delivered THIS adapter's way:
            # ClaudeAgentOptions kwargs, by name. Validated in `setup()`, so
            # every key here is one of GENERATION_KEYS.
            "model_options": dict(self.generation_config),
            "env": {**(env or {}), "DISABLE_AUTOUPDATER": "1"},
            # `check || install`, run by the runner before the agent starts.
            # Rides on stdin like everything else, so the vendor's package name
            # never appears in argv, in environ or on the container filesystem.
            "install": install,
            # The TASK's own `check || install`, from its `install:`
            # block, run by the same helper right after the agent's: a
            # library this task's work needs that the shared image does
            # not carry. None for a task that declares none.
            "task_install": task_install,
            # The model's routing: where the key sits in the container, the
            # real upstream, and the extra_body that decides whether the CLI
            # goes through the forwarder (whose SOURCE travels in `modules`
            # only when it is needed, like the grader).
            "routing": self.routing_payload(),
        }
        modules = {"stage": stage_src, "grader": grader_src}
        if self.extra_body:
            modules["forwarder"] = self.forwarder_source()
        return json.dumps({
            "task": task,
            "code": self.RUNNER.read_text(),
            "modules": modules,
        }).encode()

    # ---- the SDK's messages as ATIF ---------------------------------------
    #
    # The Claude Agent SDK hands the runner typed message objects, which
    # `core/agents/claude_sdk_runner.py:_slim` reduces to plain dicts before they cross
    # the container boundary:
    #
    #   SystemMessage/init  the tool list and MCP servers the session started
    #                       with -> the ATIF `agent.extra` block
    #   AssistantMessage    content blocks: TextBlock -> Step.message,
    #                       ThinkingBlock -> Step.reasoning_content (NOT folded
    #                       into the message -- the judge reads what the model
    #                       SAID and what it privately reasoned as different
    #                       evidence), ToolUseBlock -> Step.tool_calls
    #   UserMessage         ToolResultBlock -> the Observation on the step that
    #                       ISSUED the call, paired by tool_use_id
    #   ResultMessage       totals -> final_metrics, and the closing answer ->
    #                       extra["final_text"]
    #
    # This is the equivalent of Harbor's `claude_code.py`
    # `_convert_events_to_trajectory`, but it is a hand mapping rather than a
    # port: Harbor reads Claude Code's on-disk JSONL session transcript, and we
    # drive the Agent SDK, whose message objects are a different shape (no
    # `isSidechain`, no `toolUseResult` sidecar, no per-event envelope). What is
    # kept identical is the ATIF SEMANTICS -- one step per assistant turn,
    # reasoning separate from message, each tool call paired with its result.
    # `thinking` blocks arrive in two envelopes, and NEITHER may be dropped on
    # emptiness.
    #
    # ThinkingBlock: `display: "omitted"` -- the Opus 5 DEFAULT -- delivers the
    # block with an EMPTY `thinking` field. Filtering on truthiness made those
    # episodes indistinguishable from episodes where the model did not reason at
    # all, and that is exactly the wrong diagnosis to make easy: Opus 5 always
    # reasons, so "no ThinkingBlock" means the thinking was not DELIVERED. The
    # block's existence is therefore recorded even when it carries no text (see
    # `extra["thinking_blocks"]` below); the renderer, not the converter, decides
    # that an empty one has nothing to show. Harbor appends unconditionally too
    # (`claude_code.py:782`).
    #
    # redacted_thinking: Anthropic's encrypted-reasoning envelope, meant to be
    # passed back unchanged and undecryptable by us. Harbor
    # (`claude_code.py:794`) notes that OpenRouter REUSES the envelope to carry
    # PLAIN reasoning from non-Anthropic models it proxies: `data` is
    # `openrouter.reasoning:<b64>` whose base64 decodes to
    # `{"text": ..., "type": "reasoning.text"}`. That inner text is real
    # reasoning and is surfaced as such; genuine ciphertext is counted but never
    # dumped into a human-readable field, because an unreadable blob in the
    # judge's log is noise pretending to be evidence.
    REDACTED_TYPES = {"RedactedThinkingBlock", "redacted_thinking"}
    OPENROUTER_REASONING_PREFIX = "openrouter.reasoning:"

    @classmethod
    def _is_redacted_thinking(cls, block: dict) -> bool:
        return bool(cls.REDACTED_TYPES & {block.get("_type"), block.get("type")})

    @classmethod
    def _redacted_text(cls, block: dict) -> str:
        """The plain reasoning inside an OpenRouter-style redacted block, if any.

        Empty string for genuine Anthropic ciphertext, which cannot be read and
        must not be rendered.
        """
        data = block.get("data")
        if not isinstance(data, str) or \
                not data.startswith(cls.OPENROUTER_REASONING_PREFIX):
            return ""
        try:
            payload = data[len(cls.OPENROUTER_REASONING_PREFIX):]
            inner = json.loads(base64.b64decode(payload + "==")
                               .decode("utf-8", "replace"))
        except (ValueError, json.JSONDecodeError):
            return ""
        text = inner.get("text") if isinstance(inner, dict) else None
        return text.strip() if isinstance(text, str) else ""

    @staticmethod
    def _atif_content(raw):
        """A ToolResultBlock's content as an ATIF-valid content value.

        The SDK gives a string or a list of blocks. Media blocks were already
        replaced with a note by the runner (`_elide_images`), so everything that
        arrives here is textual; anything unrecognised is JSON-dumped rather
        than dropped, because a tool result the judge cannot see is a hole in
        the evidence.
        """
        if raw is None or isinstance(raw, str):
            return raw
        if not isinstance(raw, list):
            return json.dumps(raw, default=str)
        parts = []
        for item in raw:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text = item["text"]
                elif item.get("type") == "image":
                    text = "<image returned to the model, elided>"
                else:
                    text = json.dumps(item, default=str)
            else:
                text = str(item)
            parts.append(ContentPart(type="text", text=text))
        return parts or None

    @staticmethod
    def _metrics(usage) -> "Metrics | None":
        """Harbor's `_build_metrics`, unchanged in substance.

        `or 0` rather than a `get` default is load-bearing and is Harbor's own
        note: an interrupted stream (the agent killed on timeout) leaves a usage
        field PRESENT but null, and `get(key, 0)` only falls back when the key is
        absent. Without the guard the arithmetic raises and the whole
        trajectory's token accounting is discarded.
        """
        if not isinstance(usage, dict):
            return None
        cached = usage.get("cache_read_input_tokens") or 0
        creation = usage.get("cache_creation_input_tokens") or 0
        inp = usage.get("input_tokens") or 0
        extra = {k: v for k, v in usage.items()
                 if k not in {"input_tokens", "output_tokens"}}
        return Metrics(
            # Align with Anthropic session totals: input + cache read + creation.
            prompt_tokens=inp + cached + creation,
            completion_tokens=usage.get("output_tokens") or 0,
            cached_tokens=cached,
            cost_usd=None,
            extra=extra or None,
        )

    def to_trajectory(self, raw: dict) -> Trajectory:
        messages = (raw or {}).get("messages") or []
        model = (raw or {}).get("model") or self.model

        agent_extra: dict = {}
        steps: list[Step] = []
        issued: dict[str, int] = {}     # tool_use_id -> index into `steps`
        result_msg: dict | None = None

        def _push(step: Step) -> int:
            step.step_id = len(steps) + 1
            steps.append(step)
            return len(steps) - 1

        def _observe(index: int, result: ObservationResult) -> None:
            step = steps[index]
            if step.observation is None:
                step.observation = Observation(results=[result])
            else:
                step.observation.results.append(result)

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("_type")

            if mtype == "SystemMessage":
                if msg.get("subtype") == "init" and not agent_extra:
                    data = msg.get("data") or {}
                    agent_extra = {"tools": data.get("tools"),
                                   "mcp_servers": data.get("mcp_servers")}
                continue

            if mtype == "ResultMessage":
                result_msg = msg
                continue

            blocks = [b for b in (msg.get("content") or []) if isinstance(b, dict)]

            if mtype == "AssistantMessage":
                texts, thinking, calls = [], [], []
                # Thinking blocks SEEN, empty ones included. See REDACTED_TYPES
                # above: an empty block is evidence that thinking happened and
                # was withheld, which is not the same fact as no block at all,
                # and only the count can still say so once the text is gone.
                n_thinking = n_redacted = 0
                for block in blocks:
                    btype = block.get("_type")
                    if btype == "TextBlock" and (block.get("text") or "").strip():
                        texts.append(block["text"].strip())
                    elif btype == "ThinkingBlock":
                        n_thinking += 1
                        if (block.get("thinking") or "").strip():
                            thinking.append(block["thinking"].strip())
                    elif self._is_redacted_thinking(block):
                        n_redacted += 1
                        plain = self._redacted_text(block)
                        if plain:
                            thinking.append(plain)
                    elif btype == "ToolUseBlock":
                        args = block.get("input")
                        calls.append(ToolCall(
                            tool_call_id=str(block.get("id") or f"call_{len(issued)}"),
                            function_name=block.get("name") or "",
                            arguments=args if isinstance(args, dict)
                            else {"value": args}))
                step_extra: dict = {}
                if n_thinking:
                    step_extra["thinking_blocks"] = n_thinking
                if n_redacted:
                    step_extra["redacted_thinking_blocks"] = n_redacted
                index = _push(Step(
                    step_id=1, source="agent",
                    message="\n\n".join(texts),
                    reasoning_content="\n\n".join(thinking) or None,
                    tool_calls=calls or None,
                    model_name=msg.get("model") or model,
                    # Per-message usage when the SDK gave us any. The stream
                    # only reports usage on `message_delta`, so most saved
                    # transcripts carry none and this is None -- the totals in
                    # final_metrics are the number of record either way.
                    metrics=self._metrics(msg.get("usage")),
                    llm_call_count=1,
                    extra=step_extra or None))
                for call in calls:
                    issued[call.tool_call_id] = index
                continue

            if mtype == "UserMessage":
                for block in blocks:
                    btype = block.get("_type")
                    if btype == "ToolResultBlock":
                        call_id = str(block.get("tool_use_id") or "")
                        extra = ({"is_error": True}
                                 if block.get("is_error") in (True, "True")
                                 else None)
                        index = issued.get(call_id)
                        if index is None:
                            # The call is not in this document -- which happens
                            # only when converting ONE streamed record for the
                            # live tail. ATIF forbids a source_call_id that
                            # names no tool_call in the same step, so the id
                            # moves to `extra` rather than being invented or
                            # dropped.
                            _push(Step(
                                step_id=1, source="user", message="",
                                observation=Observation(results=[
                                    ObservationResult(
                                        source_call_id=None,
                                        content=self._atif_content(
                                            block.get("content")),
                                        extra={**(extra or {}),
                                               "tool_use_id": call_id})])))
                        else:
                            _observe(index, ObservationResult(
                                source_call_id=call_id,
                                content=self._atif_content(block.get("content")),
                                extra=extra))
                    elif btype == "TextBlock" and (block.get("text") or "").strip():
                        _push(Step(step_id=1, source="user",
                                   message=block["text"].strip()))
                continue

        if not steps:
            steps = [self.empty_step(
                "(no messages were captured for this episode)")]

        final_text = (raw or {}).get("final_text") or ""
        if result_msg and result_msg.get("result"):
            final_text = str(result_msg["result"])

        usage = (result_msg or {}).get("usage") or (raw or {}).get("usage")
        totals = self._metrics(usage)
        final_metrics = FinalMetrics(
            total_prompt_tokens=totals.prompt_tokens if totals else None,
            total_completion_tokens=totals.completion_tokens if totals else None,
            total_cached_tokens=totals.cached_tokens if totals else None,
            # The SDK reports a real price, so unlike Codex there is nothing to
            # estimate: this is Anthropic's own number.
            total_cost_usd=(result_msg or {}).get("total_cost_usd")
            or (raw or {}).get("cost_usd"),
            total_steps=len(steps),
            extra={"usage": usage} if usage else None,
        )

        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=(raw or {}).get("session_id"),
            agent=self.atif_agent(model_name=model, extra=agent_extra),
            steps=steps,
            final_metrics=final_metrics,
            extra={"final_text": final_text} if final_text else None,
        )
