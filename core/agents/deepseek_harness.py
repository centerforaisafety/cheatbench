"""Official DeepSeek Harness Python SDK adapter."""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from .installed import InstalledAgent
from ..trajectory import (SCHEMA_VERSION, Trajectory, Step, ToolCall,
                          Observation, ObservationResult, FinalMetrics)


class DeepSeekHarnessAgent(InstalledAgent):
    TURN_LIMIT = 'model API requests (including retries)'
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    CLI_BASE_URL_ENV = ("DEEPSEEK_BASE_URL",)
    PASSTHROUGH_ENV = ("DEEPSEEK_BASE_URL",)
    SUPPORTS_FORWARDER = True
    GENERATION_KEYS = ("reasoning_effort", "max_tokens")
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}
    USAGE_KEYS = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "cache_creation_tokens")
    PROMPT_TOKENS_INCLUDE_CACHE = True
    RUNNER = Path(__file__).with_name("deepseek_harness_runner.py")
    VERSION_CMD = "python -c \"import importlib.metadata as m; print(m.version('deepseek-harness-sdk'))\""
    EXTRA_RECORD_KEYS = ("notifications", "forwarder_calls", "tool_policy", "usage_complete", "runtime_profile")

    @staticmethod
    def name() -> str:
        return "deepseek-harness"

    def install_check(self) -> str:
        code = "import deepseek_harness, deepseek_harness_runtime; import importlib.metadata as m"
        if self.pinned_version:
            code += f"; assert m.version('deepseek-harness-sdk') == {self.pinned_version!r}; assert m.version('deepseek-harness-runtime-bin') == {self.pinned_version!r}"
        return "python -c " + shlex.quote(code)

    def install_script(self) -> str:
        package = "deepseek-harness-sdk" + ("==" + self.pinned_version if self.pinned_version else "")
        return "python -m pip install --disable-pip-version-check " + shlex.quote(package)

    def _tools_for(self, policy: dict) -> list[str]:
        if policy.get("web_fetch") or policy.get("web_search"):
            raise ValueError("deepseek-harness currently supports only disabled web_search/web_fetch")
        if policy.get("mcp_only"):
            return []
        # The installed full SDK profile owns the roster. Actual schemas are
        # recorded from requests, including dynamically discovered MCP tools.
        return ["bash", "create_goal", "edit", "exit_plan_mode", "get_goal", "glob", "grep",
                "interrupt_agent", "job_kill", "job_list", "job_output", "list_agents", "ralph",
                "read", "read_image", "send_message", "skill", "subagent", "subagent_fork",
                "todo_write", "update_goal", "workflow", "write"]

    def resolved_generation_config(self) -> dict:
        return {"reasoning_effort": "high", **self.generation_config}

    def setup(self) -> list[str]:
        problems = super().setup() + self.routing_problems()
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
        if self.permission_mode != "bypassPermissions":
            problems.append("deepseek-harness unattended runs require permission_mode: bypassPermissions")
        cfg = self.resolved_generation_config()
        if cfg["reasoning_effort"] not in ("off", "low", "high", "max"):
            problems.append("deepseek-harness reasoning_effort must be off, low, high or max")
        if "max_tokens" in cfg and (type(cfg["max_tokens"]) is not int or cfg["max_tokens"] <= 0):
            problems.append("deepseek-harness max_tokens must be a positive integer")
        return problems

    def blob(self, *, row, prompt, files, skip_dirs, grader_src="", stage_src,
             timeout_s, env, install=None, task_install=None) -> bytes:
        if not isinstance(prompt, str):
            raise ValueError("DeepSeek Harness takes a text instruction; stage images as workspace files")
        base = (self.resolved_base_url() or self.base_url or "https://api.deepseek.com").rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        task = {"id": row["id"], "model": self.model, "content": prompt,
                "files": files, "skip_dirs": list(skip_dirs),
                "deliverable_files": list(row.get("deliverable_files") or []),
                "timeout_s": timeout_s, "env": dict(env or {}), "install": install,
                "task_install": task_install, "tools": self.tools,
                "deepseek_harness": {"base_url": base, "api_key_env": self.container_key_envs()[0],
                             "mcp_only": bool((self.policy or {}).get("mcp_only")),
                             "extra_body": dict(self.extra_body), "max_turns": self.max_turns,
                             "generation": self.resolved_generation_config()}}
        if grader_src:
            task["row"] = {**row, "episode_timeout_s": timeout_s} if row.get("tool_surface") in {"chess", "go"} else row
        return json.dumps({"task": task, "code": self.RUNNER.read_text(), "modules": {
            "stage": stage_src, "grader": grader_src, "forwarder": self.forwarder_source(),
            "support": self.RUNNER.with_name("runner_support.py").read_text()}}).encode()

    def reported_cost_usd(self, raw: dict) -> float | None:
        calls = [c for c in raw.get("forwarder_calls") or [] if c.get("status") == 200]
        costs = [(c.get("usage") or {}).get("cost") for c in calls]
        if costs and all(isinstance(cost, (int, float)) and cost >= 0 for cost in costs):
            return float(sum(costs))
        return None

    def to_trajectory(self, raw: dict) -> Trajectory:
        steps, calls, notes = [], {}, []
        for rec in raw.get("messages") or []:
            sid = rec.get("session_id", raw.get("session_id", ""))
            kind, data = rec.get("type"), rec.get("data") or {}
            message = data.get("message") or data
            blocks = message.get("content") or []
            if kind in ("assistant/message", "user/message"):
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                reasoning = "".join(b.get("text", "") for b in blocks if b.get("type") == "reasoning")
                if kind == "user/message" and reasoning:
                    # Subagent completion notices can quote the child's reasoning.
                    # Preserve it in the received message, in native block order;
                    # it is not a new reasoning step by the receiving agent.
                    text = "".join(
                        b.get("text", "") if b.get("type") == "text" else
                        "\n[reasoning block in received message]\n" + b.get("text", "") + "\n"
                        for b in blocks if b.get("type") in {"text", "reasoning"})
                    reasoning = ""
                step = Step(step_id=len(steps) + 1,
                    source="agent" if kind == "assistant/message" else "user",
                    message=text, reasoning_content=reasoning or None,
                    model_name=((message.get("source") or {}).get("model") or self.model) if kind == "assistant/message" else None,
                    llm_call_count=1 if kind == "assistant/message" else None,
                    extra={"session_id": sid, "native_message_id": message.get("id")})
                for block in blocks:
                    if block.get("type") != "tool-call":
                        continue
                    args = block.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {"raw_arguments": args}
                    if not isinstance(args, dict):
                        args = {"raw_arguments": args}
                    cid = str(block.get("id") or "")
                    step.tool_calls = (step.tool_calls or []) + [ToolCall(
                        tool_call_id=f"{sid}:{cid}", function_name=block.get("name") or "", arguments=args)]
                    step.observation = Observation(results=[])
                    calls[sid, cid] = step
                steps.append(step)
            elif kind == "tool/result":
                for block in blocks:
                    if block.get("type") != "tool-result":
                        continue
                    cid = str(block.get("toolCallId") or "")
                    step = calls.get((sid, cid))
                    content = block.get("content") or []
                    text = "\n".join(b.get("text", "") if b.get("type") == "text" else json.dumps(b) for b in content)
                    if step:
                        step.observation.results.append(ObservationResult(source_call_id=f"{sid}:{cid}", content=text))
                    else:
                        notes.append(f"unmatched tool result {sid}:{cid}")
            elif kind == "tool/ptc-dispatch":
                cid = f"{sid}:{data.get('subCallId', '')}"
                args = data.get("arguments") or {}
                content = data.get("content") or []
                steps.append(Step(step_id=len(steps)+1, source="agent", message="",
                    tool_calls=[ToolCall(tool_call_id=cid, function_name=data.get("name") or "",
                        arguments=args if isinstance(args, dict) else {"raw_arguments": args})],
                    observation=Observation(results=[ObservationResult(source_call_id=cid,
                        content="\n".join(b.get("text", "") if b.get("type") == "text" else json.dumps(b) for b in content))])))
        if not steps:
            steps = [self.empty_step("No DeepSeek Harness session messages captured")]
        usage = raw.get("usage") or {}
        return Trajectory(schema_version=SCHEMA_VERSION,
            session_id=raw.get("session_id") or "unknown", agent=self.atif_agent(), steps=steps,
            notes="; ".join(notes) or None,
            final_metrics=FinalMetrics(total_steps=len(steps),
                total_prompt_tokens=usage.get("prompt_tokens"),
                total_completion_tokens=usage.get("completion_tokens"),
                total_cached_tokens=usage.get("prompt_cache_hit_tokens"),
                total_cost_usd=raw.get("cost_usd")),
            extra={"final_text": raw.get("final_text") or ""})
