"""The Google Gemini CLI adapter.

One adapter, one module -- Harbor's layout
(`harbor/src/harbor/agents/installed/gemini_cli.py`, which this is ported
from). Its in-container runner is the sibling `gemini_cli_runner.py`, which is
read as source and streamed into the container on stdin; it is never imported
here and never written to the container filesystem.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

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
from .installed import NODE_MAJOR, NVM_PRELUDE, InstalledAgent, link_bins, node_install


class GeminiCLIAgent(InstalledAgent):
    """Google's Gemini CLI, driven by `gemini_cli_runner.py`.

    Ported from Harbor's `harbor/agents/installed/gemini_cli.py`: the install
    (nvm Node + `@google/gemini-cli`), the invocation (`gemini --yolo
    --model=... --prompt=...` with `GEMINI_CLI_TRUST_WORKSPACE=true`), the
    `settings.json` the CLI needs in headless mode, the session-file lookup and
    the session -> ATIF conversion are Harbor's, reshaped to the things this
    sandbox does differently (see `CodexAgent` for the same three: `install()`
    returns a check/install pair, the runner reads the record inside the
    container, and the runner hosts the grader over loopback HTTP).

    THE ROUTE TO THE MODEL is the one thing with no Harbor precedent. The CLI
    speaks the Gemini-native API, and the org's LiteLLM gateway serves Gemini
    only through its OpenAI-compatible route, so in the default `gateway`
    route the runner hosts a loopback shim that translates between the two
    (see the runner's docstring for what is verified about that translation).
    The `native` route uses a model-pinning relay with Google-native bodies, so
    the two can be A/B'd for harness fidelity. Which route, and therefore
    which credential variable the episode needs, is decided here and recorded
    on every episode as `gemini_route`.
    """
    TURN_LIMIT = 'session model turns'

    @staticmethod
    def name() -> str:
        return "gemini-cli"

    # ---- what this adapter can hold shut ----------------------------------
    #
    # Gemini CLI has two web tools and both are Google-side: `google_web_search`
    # asks the model with the `googleSearch` grounding tool attached, and
    # `web_fetch` asks it with the `urlContext` tool attached (`web-search` and
    # `web-fetch` model configs in the CLI source), so a fetch is executed on
    # Google's servers and never crosses the container's namespace. The knob is
    # `tools.exclude` in `~/.gemini/settings.json` ("Tool names to exclude from
    # discovery"), which removes a tool from the declarations the model is
    # shown. Two keys, two names, so unlike Codex they may differ.
    #
    # Verified EMPIRICALLY, as base.py demands, not inferred from the settings
    # schema: in the gateway route every request passes through the runner's
    # shim, which records the function declarations the CLI actually sent and
    # FAILS the episode if an excluded name is among them (`shim.breaches`), and
    # the openmath smoke episode's mirror log shows the planted post fetched
    # in-namespace by the agent's own shell. The settings file is identical in
    # the native route, so the same exclusion holds there.
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}

    # Which route to the model, and therefore which credential. Set by
    # `GEMINI_CLI_ROUTE` in the environment (`.env`); `gateway` unless told
    # otherwise, because every other adapter here runs through the gateway.
    ROUTE_ENV = "GEMINI_CLI_ROUTE"
    ROUTES = ("gateway", "native")

    # Per route. `API_KEY_ENV` and `PASSTHROUGH_ENV` are set on the INSTANCE in
    # __init__ from this table; run.py reads them off the instance.
    ROUTE_ENV_TABLE = {
        #             credential          copied into the container
        "gateway": ("OPENAI_API_KEY", ("GEMINI_BASE_URL", "OPENAI_BASE_URL")),
        "native": ("GEMINI_API_KEY", ()),
    }
    API_KEY_ENV = "OPENAI_API_KEY"
    PASSTHROUGH_ENV = ("GEMINI_BASE_URL", "OPENAI_BASE_URL")

    # The exact command line, the session file it was read back from, which
    # route the model was reached by, and (gateway route) one summary per
    # upstream call the shim made. A Gemini result that could not say which of
    # the two routes produced it would not be a comparable result.
    EXTRA_RECORD_KEYS = ("gemini_command", "session_path", "gemini_route",
                         "shim_calls")

    # ---- what an episode cost ---------------------------------------------
    #
    # The session file carries per-message token counts and never a price
    # (Harbor: "Gemini CLI's session file has no cost field; back it out from
    # LiteLLM's pricing"), so `reported_cost_usd` is None and the base class's
    # litellm estimate is this adapter's primary cost source. The runner sums
    # the per-message `tokens` blocks into one `usage` dict:
    #
    #     input      promptTokenCount, INCLUSIVE of the cached part
    #                (`cachedContentTokenCount` is a subset of it)
    #     completion output + thoughts + tool -- Harbor's arithmetic, and what
    #                litellm prices at the output rate (reasoning tokens cost
    #                the same as output tokens on every Gemini 3 entry)
    #     cached     cachedContentTokenCount
    #
    # Gemini has no cache-WRITE category (implicit caching is not billed as a
    # write), so the fourth name is one the usage dict never carries and reads
    # as zero.
    USAGE_FIELDS = ("usage",)
    USAGE_KEYS = ("input", "completion", "cached", "cache_write")
    PROMPT_TOKENS_INCLUDE_CACHE = True

    def reported_cost_usd(self, raw: dict) -> float | None:
        """None, always: the CLI never prices its own episodes."""
        return None

    # ---- this adapter's runtime -------------------------------------------
    PACKAGE = "@google/gemini-cli"
    NODE_MAJOR = NODE_MAJOR
    # `gemini --version` prints the bare version, e.g. `0.59.0`.
    VERSION_CMD = NVM_PRELUDE + "gemini --version 2>/dev/null | head -1 | awk '{print $1}'"
    INSTALL_CHECK = NVM_PRELUDE + "command -v gemini >/dev/null 2>&1"

    # ---- the CLI's toolset -------------------------------------------------
    #
    # The CLI's built-in tools (its `*_TOOL_NAME` constants, 0.59.0). The
    # first group runs INSIDE the container; the two web tools run on Google's
    # side and are the ones the policy switches. The list is nominal -- what
    # the CLI actually declares to the model is observed by the shim and
    # recorded as `init_tools` on every gateway-route episode.
    BASE_TOOLS = ("run_shell_command", "read_file", "write_file", "replace",
                  "list_directory", "glob", "grep_search", "read_many_files",
                  "write_todos")
    OPTIONAL_TOOLS = {"web_search": "google_web_search", "web_fetch": "web_fetch"}

    # ---- generation config --------------------------------------------------
    #
    # `reasoning_effort` is delivered Harbor's way: a `modelConfigs.customAliases`
    # entry in settings.json whose `generateContentConfig.thinkingConfig.
    # thinkingLevel` is the effort, selected with `--model=<alias>`. Unset
    # means the CLI's own default for the model (HIGH with thoughts included,
    # for a Gemini 3 model), which is the harness as its author ships it.
    GENERATION_KEYS = ("reasoning_effort",)
    REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high")

    RUNNER = Path(__file__).resolve().parent / "gemini_cli_runner.py"

    def __init__(self, *args, route: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = (route or os.environ.get(self.ROUTE_ENV) or "gateway").strip().lower()
        if self.route not in self.ROUTES:
            raise ValueError(f"{self.ROUTE_ENV}={self.route!r}; choose one of "
                             f"{list(self.ROUTES)}")
        route_key_env, self.PASSTHROUGH_ENV = self.ROUTE_ENV_TABLE[self.route]
        if self.api_key_env_source != "entry":
            self.API_KEY_ENV = route_key_env

    # ---- install ----------------------------------------------------------
    def install_check(self) -> str:
        parts = [self.INSTALL_CHECK]
        if self.pinned_version:
            parts.append(f'[ "$({self.VERSION_CMD})" = '
                         f"{json.dumps(self.pinned_version)} ]")
        return " && ".join(parts)

    def install_script(self) -> str:
        pkg = self.PACKAGE + (f"@{self.pinned_version}" if self.pinned_version
                              else "@latest")
        return "\n".join([
            *node_install(self.NODE_MAJOR),
            'npm_config_cache="$(mktemp -d)"; export npm_config_cache',
            f"npm install -g --no-fund --no-audit {pkg}",
            'rc=$?; rm -rf "$npm_config_cache" "$HOME/.npm"',
            link_bins("node", "npm", "npx", "gemini"),
            "exit $rc",
        ])

    # ---- the task's tool policy -------------------------------------------
    def _tools_for(self, policy: dict) -> list[str]:
        if policy.get("mcp_only"):
            return []
        return list(self.BASE_TOOLS) + [tool for key, tool
                                        in self.OPTIONAL_TOOLS.items()
                                        if policy[key]]

    @property
    def excluded_tools(self) -> list[str]:
        """Web exclusions; core separately disables native tools in MCP-only.

        Gemini 0.59 also applies basename exclusions to MCP tools, so excluding
        native read_file/write_file would remove the chess tools with those names.
        """
        if self.policy is None:
            raise RuntimeError("apply_tool_policy has not been called")
        if self.policy.get("mcp_only"):
            return list(self.OPTIONAL_TOOLS.values())
        return [tool for key, tool in self.OPTIONAL_TOOLS.items() if not self.policy[key]]

    @property
    def model_slug(self) -> str:
        """What the CLI is given: the id with any provider prefix stripped."""
        return self.model.split("/")[-1]

    @property
    def model_alias(self) -> str | None:
        """The settings alias carrying `reasoning_effort`, or None."""
        effort = self.generation_config.get("reasoning_effort")
        return f"{self.model_slug}-{effort}" if effort else None

    @property
    def run_model(self) -> str:
        return self.model_alias or self.model_slug

    def settings(self) -> dict:
        """`~/.gemini/settings.json`, minus the grader entry the runner adds.

        security.auth.selectedType   headless `gemini --prompt` cannot show the
                                     auth dialog and its env auto-detection
                                     flips to a different type when a base URL
                                     is set, so the method is pinned (Harbor).
                                     `gemini-api-key` in both routes: the
                                     gateway route's key is the shim's token.
        tools.exclude                the policy, as the CLI spells it.
        privacy.usageStatisticsEnabled  no telemetry to Google from an eval.
        general.enableAutoUpdate*    no update check, no nag, no npm traffic.
        advanced.autoConfigureMemory the CLI otherwise re-execs itself with a
                                     heap size derived from the host's memory;
                                     one process is easier to time out.
        modelConfigs.customAliases   Harbor's delivery of reasoning_effort.
        """
        out: dict = {
            "security": {"auth": {"selectedType": "gemini-api-key"}},
            "tools": {"exclude": list(self.excluded_tools)},
            "privacy": {"usageStatisticsEnabled": False},
            "general": {"enableAutoUpdate": False,
                        "enableAutoUpdateNotification": False},
            "advanced": {"autoConfigureMemory": False},
        }
        if self.policy.get("mcp_only"):
            # Fail closed until the runner adds the grader's internal MCP
            # policy names (mcp_server_tool). No built-in matches those names.
            # Empty core alone also denies MCP execution in pinned 0.59.0.
            out["tools"]["core"] = []
        if self.max_turns:
            out["model"] = {"maxSessionTurns": self.max_turns}
        effort = self.generation_config.get("reasoning_effort")
        if effort:
            out["modelConfigs"] = {"customAliases": {self.model_alias: {
                "modelConfig": {
                    "model": self.model_slug,
                    "generateContentConfig": {
                        "thinkingConfig": {"includeThoughts": True,
                                           "thinkingLevel": str(effort).upper()}}}}}}
        return out

    def resolved_base_url(self) -> str:
        """Resolve the actual upstream for both runtime and network policy."""
        if self.api_base_url:
            return self.api_base_url
        if self.route == "gateway":
            return (os.environ.get("GEMINI_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL") or "")
        return "https://generativelanguage.googleapis.com"

    def base_url_source(self) -> str:
        if self.api_base_url:
            return "entry"
        if self.route == "gateway":
            for name in ("GEMINI_BASE_URL", "OPENAI_BASE_URL"):
                if os.environ.get(name):
                    return f"env:{name}"
        return "provider-default"

    def setup(self) -> list[str]:
        problems = [] if self.RUNNER.exists() else [f"missing runner {self.RUNNER}"]
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
            return problems
        problems += self.config_problems()
        if self.policy.get("mcp_only") and self.route != "gateway":
            problems.append("Gemini MCP-only requires the verified gateway route")
        effort = self.generation_config.get("reasoning_effort")
        if effort and effort not in self.REASONING_EFFORT_CHOICES:
            problems.append(f"reasoning_effort={effort!r}; gemini-cli takes "
                            f"{list(self.REASONING_EFFORT_CHOICES)}")
        for key, tool in self.OPTIONAL_TOOLS.items():
            if not self.policy[key] and tool in self.tools:
                problems.append(f"{key} is false but {tool} is still in the tool list")
            if not self.policy[key] and tool not in self.excluded_tools:
                problems.append(f"{key} is false but {tool} is not in tools.exclude")
        if self.route == "gateway" and not self.resolved_base_url():
            problems.append("gateway route needs api_base_url in the model config, "
                            "or GEMINI_BASE_URL / OPENAI_BASE_URL in .env; "
                            f"set {self.ROUTE_ENV}=native to go direct")
        return problems

    # ---- the episode ------------------------------------------------------
    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
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
            "gemini": {
                "route": self.route,
                "api_key_env": self.API_KEY_ENV,
                "model_id": self.model,
                "model_slug": self.model_slug,
                "run_model": self.run_model,
                "exclude_tools": list(self.excluded_tools),
                "mcp_only": bool(self.policy.get("mcp_only")),
                "settings": self.settings(),
                # Upstream in the selected route's API format.
                "base_url": self.resolved_base_url(),
            },
        }
        if grader_src:
            task["row"] = {**row, "episode_timeout_s": timeout_s} if row.get("tool_surface") in {"chess", "go"} else row
        return json.dumps({
            "task": task,
            "code": self.RUNNER.read_text(),
            "modules": {"stage": stage_src, "grader": grader_src,
                        "session_observer": self.RUNNER.with_name("session_observer.py").read_text()},
        }).encode()

    # ---- the session file as ATIF -----------------------------------------
    #
    # PORTED FROM HARBOR, Apache License 2.0: `_load_gemini_session`,
    # `_merge_message_update` and `_convert_gemini_to_atif` from
    # `harbor/src/harbor/agents/installed/gemini_cli.py`. Derived work; see
    # core/trajectory/__init__.py for the full notice.
    #
    # The session file is JSONL (0.40+): `{"$set": {...}}` metadata records,
    # `{"type": "user"|"gemini", "id": ..., ...}` message records,
    # `{"type": "message_update", "id": ...}` patches (tool results and token
    # counts arrive this way, after the message they belong to), and
    # `{"$rewindTo": id}`. Harbor normalises those into the legacy single
    # document shape and converts that; so do we.
    #
    # Deviations, both narrow:
    #   * images a tool returned are NOT written to disk (there is no logs dir
    #     on this side of the conversion); they become an elided text part, as
    #     the Claude adapter does.
    #   * a timestamp ATIF cannot parse is dropped rather than the step (the
    #     same reason as CodexAgent._iso).
    @staticmethod
    def _merge_update(message: dict, update: dict) -> None:
        for key, value in update.items():
            if key in {"type", "id"}:
                continue
            current = message.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            else:
                message[key] = value

    @classmethod
    def load_session(cls, records: list) -> dict:
        """Harbor's `_load_gemini_session`, over in-memory records."""
        metadata: dict = {}
        message_ids: list = []
        by_id: dict = {}
        pending: dict = {}
        anon = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            if "messages" in record and isinstance(record["messages"], list):
                # A legacy single-document session, arriving as one record.
                for k, v in record.items():
                    if k != "messages":
                        metadata[k] = v
                for msg in record["messages"]:
                    if isinstance(msg, dict):
                        anon += 1
                        mid = msg.get("id") or f"_anon_{anon}"
                        message_ids.append(mid)
                        by_id[mid] = msg
                continue
            if "$rewindTo" in record:
                rewind_id = record["$rewindTo"]
                if rewind_id in message_ids:
                    idx = message_ids.index(rewind_id)
                    for removed in message_ids[idx:]:
                        by_id.pop(removed, None)
                        pending.pop(removed, None)
                    del message_ids[idx:]
                else:
                    message_ids.clear()
                    by_id.clear()
                    pending.clear()
            elif "$set" in record and isinstance(record["$set"], dict):
                metadata.update(record["$set"])
            elif record.get("type") in {"user", "gemini"}:
                mid = record.get("id")
                if isinstance(mid, str) and mid in by_id:
                    by_id[mid].update(record)
                    message = by_id[mid]
                else:
                    message = dict(record)
                    if not isinstance(mid, str):
                        anon += 1
                        mid = f"_anon_{anon}"
                    message_ids.append(mid)
                    by_id[mid] = message
                for update in pending.pop(mid, []):
                    cls._merge_update(message, update)
            elif record.get("type") == "message_update":
                mid = record.get("id")
                if isinstance(mid, str) and mid in by_id:
                    cls._merge_update(by_id[mid], record)
                elif isinstance(mid, str):
                    pending.setdefault(mid, []).append(record)
            elif "sessionId" in record:
                for k, v in record.items():
                    if k != "messages":
                        metadata[k] = v
        return {"sessionId": metadata.get("sessionId"),
                "messages": [by_id[m] for m in message_ids],
                "metadata": metadata}

    @staticmethod
    def _iso(timestamp):
        if not isinstance(timestamp, str):
            return None
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return timestamp

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("thought"):
                        continue
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return str(content) if content else ""

    @staticmethod
    def _tool_result(result) -> tuple:
        """(content, extra) for one recorded tool call's `result`.

        The CLI records what it sent back to the model: a list of parts, each a
        `functionResponse` whose `response` is `{"output": ...}` or
        `{"error": ...}`, plus media parts for tools that return images. Text
        is kept in full; an image is named and elided.
        """
        if result is None:
            return None, None
        if isinstance(result, str):
            return result, None
        texts: list = []
        media = 0
        extra: dict = {}
        items = result if isinstance(result, list) else [result]
        for item in items:
            if not isinstance(item, dict):
                texts.append(str(item))
                continue
            fr = item.get("functionResponse")
            if isinstance(fr, dict):
                response = fr.get("response")
                if isinstance(response, dict):
                    if isinstance(response.get("output"), str):
                        texts.append(response["output"])
                    elif response.get("error") is not None:
                        texts.append(f"ERROR: {response['error']}")
                        extra["is_error"] = True
                    elif response:
                        texts.append(json.dumps(response, ensure_ascii=False))
                elif response is not None:
                    texts.append(str(response))
                for part in fr.get("parts") or []:
                    if isinstance(part, dict) and part.get("inlineData"):
                        media += 1
            elif item.get("inlineData"):
                media += 1
            elif isinstance(item.get("text"), str):
                texts.append(item["text"])
        if media:
            parts = [ContentPart(type="text", text="\n".join(texts))] if texts else []
            parts += [ContentPart(type="text",
                                  text="<image returned to the model, elided>")] * media
            return parts, extra or None
        return ("\n".join(texts) if texts else None), extra or None

    def to_trajectory(self, raw: dict) -> Trajectory:
        records = [r for r in ((raw or {}).get("messages") or []) if isinstance(r, dict)]
        session = self.load_session(records)
        session_id = session.get("sessionId") or (raw or {}).get("session_id")

        steps: list = []
        totals = {"prompt": 0, "completion": 0, "cached": 0, "thoughts": 0, "tool": 0}
        saw_tokens = False
        dropped: list = []

        for message in session["messages"]:
            mtype = message.get("type")
            timestamp = self._iso(message.get("timestamp"))
            if mtype == "user":
                text = self._text(message.get("content"))
                if not text.strip():
                    # A synthetic user record (tool responses re-recorded as a
                    # user turn) carries nothing the gemini record does not.
                    continue
                steps.append(Step(step_id=len(steps) + 1, timestamp=timestamp,
                                  source="user", message=text))
                continue
            if mtype != "gemini":
                continue

            reasoning = None
            thoughts = message.get("thoughts") or []
            pieces = []
            for thought in thoughts:
                if not isinstance(thought, dict):
                    continue
                subject = (thought.get("subject") or "").strip()
                description = (thought.get("description") or "").strip()
                if subject and description:
                    pieces.append(f"{subject}: {description}")
                elif description or subject:
                    pieces.append(description or subject)
            if pieces:
                reasoning = "\n".join(pieces)

            calls: list = []
            results: list = []
            details: dict = {}
            for i, tc in enumerate(message.get("toolCalls") or []):
                if not isinstance(tc, dict):
                    continue
                call_id = str(tc.get("id") or f"call_{len(steps) + 1}_{i}")
                args = tc.get("args")
                calls.append(ToolCall(tool_call_id=call_id,
                                      function_name=tc.get("name") or "",
                                      arguments=args if isinstance(args, dict)
                                      else {"value": args}))
                content, extra = self._tool_result(tc.get("result"))
                status = tc.get("status")
                if status and status != "success":
                    extra = {**(extra or {}), "status": status}
                results.append(ObservationResult(source_call_id=call_id,
                                                 content=content, extra=extra))
                detail = {k: tc.get(k) for k in ("status", "displayName")
                          if tc.get(k)}
                if detail:
                    details[call_id] = detail

            metrics = None
            tokens = message.get("tokens")
            if isinstance(tokens, dict):
                saw_tokens = True
                inp = int(tokens.get("input") or 0)
                out = int(tokens.get("output") or 0)
                cached = int(tokens.get("cached") or 0)
                thought_tokens = int(tokens.get("thoughts") or 0)
                tool_tokens = int(tokens.get("tool") or 0)
                completion = out + thought_tokens + tool_tokens
                totals["prompt"] += inp
                totals["completion"] += completion
                totals["cached"] += cached
                totals["thoughts"] += thought_tokens
                totals["tool"] += tool_tokens
                metrics = Metrics(prompt_tokens=inp, completion_tokens=completion,
                                  cached_tokens=cached,
                                  extra={"thoughts_tokens": thought_tokens,
                                         "tool_tokens": tool_tokens,
                                         "total_tokens": tokens.get("total")})

            step_extra: dict = {}
            if details:
                step_extra["tool_call_details"] = details
            try:
                steps.append(Step(
                    step_id=len(steps) + 1, timestamp=timestamp, source="agent",
                    model_name=message.get("model") or self.model,
                    message=self._text(message.get("content")),
                    reasoning_content=reasoning,
                    tool_calls=calls or None,
                    observation=Observation(results=results) if results else None,
                    metrics=metrics,
                    llm_call_count=1,
                    extra=step_extra or None))
            except ValueError as e:
                dropped.append(f"{message.get('id')}: {type(e).__name__}: {e}")

        if not steps:
            steps = [self.empty_step(
                "(no session records were captured for this episode)")]

        final_metrics = None
        if saw_tokens:
            final_metrics = FinalMetrics(
                total_prompt_tokens=totals["prompt"] or None,
                total_completion_tokens=totals["completion"] or None,
                total_cached_tokens=totals["cached"] or None,
                total_cost_usd=(raw or {}).get("cost_usd"),
                total_steps=len(steps),
                extra={"total_thoughts_tokens": totals["thoughts"],
                       "total_tool_tokens": totals["tool"]})

        # The AGENT model is the configured one. The CLI resolves `--model`
        # through its own table and stamps each session message with the name
        # it resolved to (a `*flash` collapses onto its flash-GA default), so
        # the session's self-reported model may differ from the upstream request.
        # The gateway shim or native relay pins the configured request ID
        # (see its `upstream_model`).
        # The CLI's label is kept on `agent.extra` so the discrepancy is
        # visible rather than silently overwritten, but the top line the judge
        # reads names the model requested upstream.
        cli_model = next((s.model_name for s in steps
                          if s.source == "agent" and s.model_name), None)

        final_text = (raw or {}).get("final_text") or ""
        if not final_text:
            for step in reversed(steps):
                text = step.message if isinstance(step.message, str) else ""
                if step.source == "agent" and text.strip():
                    final_text = text
                    break

        agent_extra: dict = {}
        for key, name in (("gemini_route", "route"), ("init_tools", "tools")):
            if (raw or {}).get(key) is not None:
                agent_extra[name] = raw[key]
        if cli_model and cli_model != self.model_slug:
            agent_extra["cli_reported_model"] = cli_model

        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            agent=self.atif_agent(model_name=self.model, extra=agent_extra or None),
            steps=steps,
            notes=(f"{len(dropped)} session message(s) could not be converted "
                   f"to ATIF steps and are missing from this trajectory: "
                   + "; ".join(dropped)) if dropped else None,
            final_metrics=final_metrics,
            extra={"final_text": final_text} if final_text else None,
        )
