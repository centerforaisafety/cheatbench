"""Environment/measurement bridge, loaded in memory by Python 3.12.

The subclass observes native responses/terminal results and streams completed
ATIF steps to the host. Image observations are supplied by terminus_2_images.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig


# Injected by the stdin bootstrap; no runtime helpers are written to the task.
SUPPORT = globals()["SUPPORT"]
ImageTerminus2 = globals()["MCPImageTerminus2"]
IMAGE_EXTENSION_VERSION = globals()["IMAGE_EXTENSION_VERSION"]


class EpisodeEnvironment:
    """Harbor's environment operations within the already isolated container."""

    default_user = None

    def __init__(self, workdir, logs_dir, session_id, env):
        self.workdir = workdir
        self.trial_paths = SimpleNamespace(agent_dir=logs_dir)
        self.session_id = session_id
        self.env = env

    async def exec(self, command, *, cwd=None, env=None, timeout_sec=60, user=None):
        # enroot has already selected the episode uid. T2 requests root for
        # package installation; it must never cause host privilege escalation.
        if user not in (None, "root", 0, os.getuid()):
            raise ValueError(f"unsupported episode user: {user}")
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd or self.workdir,
            env={**self.env, **(env or {})},
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def is_dir(self, path):
        return Path(path).is_dir()

    async def upload_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(source_path).resolve() != Path(target_path).resolve():
            shutil.copyfile(source_path, target_path)

    async def download_file(self, source_path, target_path):
        await self.upload_file(source_path, target_path)


class ObservedTerminus2(ImageTerminus2):
    """Observe the upstream loop with the local image extension enabled."""

    observer = None
    emitted = 0

    async def _handle_llm_interaction(self, *args, **kwargs):
        result = await super()._handle_llm_interaction(*args, **kwargs)
        commands, _, _, analysis, plan, response = result
        if self.observer:
            blocks = []
            if response.reasoning_content:
                blocks.append(
                    {"_type": "ThinkingBlock", "thinking": response.reasoning_content}
                )
            blocks.append({"_type": "TextBlock", "text": response.content})
            self.observer.observe({"_type": "AssistantMessage", "content": blocks})
        return result

    async def _execute_commands(self, commands, session):
        call_id = f"terminal-batch-{self._n_episodes}"
        if self.observer and commands:
            self.observer.observe(
                {
                    "_type": "AssistantMessage",
                    "content": [
                        {
                            "_type": "ToolUseBlock",
                            "id": call_id,
                            "name": "Bash",
                            "input": {
                                "commands": [
                                    {
                                        "keystrokes": c.keystrokes,
                                        "duration": c.duration_sec,
                                    }
                                    for c in commands
                                ]
                            },
                        }
                    ],
                }
            )
        result = await super()._execute_commands(commands, session)
        if self.observer:
            self.observer.observe(
                {
                    "_type": "UserMessage",
                    "content": [
                        {
                            "_type": "ToolResultBlock",
                            "tool_use_id": call_id,
                            "content": result[1],
                        }
                    ],
                }
            )
        return result

    def _dump_trajectory(self):
        super()._dump_trajectory()
        for step in self._trajectory_steps[self.emitted :]:
            SUPPORT["_emit_msg"](step.model_dump(mode="json", exclude_none=True))
        self.emitted = len(self._trajectory_steps)


async def episode(task, logs_dir, grader, observer):
    print(
        f"[runner] terminus-2: image extension v{IMAGE_EXTENSION_VERSION} enabled",
        file=sys.stderr,
        flush=True,
    )
    context = AgentContext()
    options = dict(task.get("generation_config") or {})
    call_kwargs = {
        k: options.pop(k)
        for k in ("thinking", "max_tokens", "output_config")
        if k in options
    }
    if task.get("extra_body"):
        call_kwargs["extra_body"] = task["extra_body"]
    key = os.environ.get(task["api_key_env"])
    if not key:
        raise RuntimeError(f"{task['api_key_env']} is unset in the episode")
    options.update(
        max_turns=task.get("max_turns"),
        api_base=task.get("api_base") or None,
        llm_call_kwargs=call_kwargs,
    )
    mcp = (
        [
            MCPServerConfig(
                name=grader.grader_server_name,
                transport="streamable-http",
                url=grader.url,
            )
        ]
        if grader
        else []
    )
    agent = ObservedTerminus2(
        logs_dir=logs_dir,
        model_name=task["model"],
        # The host bridge waits up to 900 seconds for upstream headers. Allow
        # it to return its response/error before LiteLLM abandons the socket.
        # The outer episode timeout still enforces the original solve budget.
        llm_kwargs={"api_key": key, "timeout": 930},
        mcp_servers=[] if task.get("terminus_2", {}).get("mcp_only") else mcp,
        mcp_only=task.get("terminus_2", {}).get("mcp_only", False),
        task_grader=grader,
        **options,
    )
    # Harbor copies llm_kwargs into the LLM at construction, but also includes
    # the options copy in trajectory.agent.extra. Remove the secret from the
    # metadata copy before any trajectory can be written.
    agent.options.llm_kwargs = {}
    agent.observer = observer
    from harbor.models.trial.paths import EnvironmentPaths

    Path(EnvironmentPaths.agent_dir).mkdir(parents=True, exist_ok=True)
    environment = EpisodeEnvironment(
        SUPPORT["WORKDIR"], logs_dir, str(task["id"]), SUPPORT["task_install_env"]()
    )
    error = None
    phase = "setup"
    try:
        # Tool setup has its own bounded upstream install commands. It is
        # separate from the model's solve budget, just as in Harbor trials.
        await asyncio.wait_for(agent.setup(environment), 600)
        phase = "run"
        await asyncio.wait_for(
            agent.run(task["content"], environment, context), task["timeout_s"]
        )
    except TimeoutError:
        error = (
            f"timeout after {task['timeout_s']}s"
            if phase == "run"
            else "Terminus-2 setup timeout after 600s"
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if agent._session is not None:
            try:
                await asyncio.wait_for(agent._session.stop(), 15)
            except Exception:
                pass
    return context.model_dump(mode="json"), agent.options.model_dump(mode="json"), error


def run(task, install, grader=None, observer=None, observer_ns=None):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="terminal-session-") as path:
        logs_dir = Path(path)
        context, options, error = asyncio.run(episode(task, logs_dir, grader, observer))
        artifacts = {
            p.name: json.loads(p.read_text()) for p in logs_dir.glob("trajectory*.json")
        }
        trajectory = artifacts.get("trajectory.json")
        steps = (trajectory or {}).get("steps", [])
        usage = dict(
            input=context.get("n_input_tokens") or 0,
            output=context.get("n_output_tokens") or 0,
            cached=context.get("n_cache_tokens") or 0,
        )
        # core.trial saves native `messages` verbatim in transcript.json, but
        # not adapter-specific top-level keys. Keep the entire native document
        # and auxiliary trajectories in that envelope so historical rejudging
        # never loses session/config/metrics or summarization evidence.
        messages = (
            [
                dict(
                    type="harbor_trajectory",
                    trajectory=trajectory,
                    artifacts={
                        k: v for k, v in artifacts.items() if k != "trajectory.json"
                    },
                    harbor_commit=task["harbor_commit"],
                    options=options,
                    context=context,
                )
            ]
            if trajectory
            else []
        )
        last_agent = next((s for s in reversed(steps) if s["source"] == "agent"), {})
        if not trajectory and not error:
            error = "Harbor produced no trajectory"
        if (
            not error
            and task.get("max_turns")
            and (context.get("metadata") or {}).get("n_episodes", 0)
            >= task["max_turns"]
        ):
            # Exhaustion is incomplete, unless the final two turns confirm
            # completion (upstream requires the confirmation turn).
            recent = [s for s in steps if s["source"] == "agent"][-2:]
            if len(recent) != 2 or not all(
                any(
                    c["function_name"] == "mark_task_complete"
                    for c in s.get("tool_calls") or []
                )
                for s in recent
            ):
                error = "max_turns reached before confirmed task completion"
        return dict(
            id=task["id"],
            model=task["model"],
            messages=messages,
            final_text=last_agent.get("message") or "",
            usage=usage,
            n_turns=(context.get("metadata") or {}).get("n_episodes", 0),
            n_tool_calls=sum(len(s.get("tool_calls") or []) for s in steps),
            wall_time=round(time.monotonic() - started, 1),
            error=error,
            install=task.get("install_result") or install,
            grader_state=grader.grader_state if grader else None,
            init_mcp_servers=[grader.report()] if grader else [],
            init_tools=task["tools"],
            deliverables=SUPPORT["collect_deliverables"](
                SUPPORT["WORKDIR"],
                tuple(task["skip_dirs"]),
                tuple(task.get("deliverable_files") or []),
            ),
            harbor_commit=task["harbor_commit"],
            harbor_trajectory=trajectory,
            harbor_artifacts=artifacts,
            harbor_context=context,
            harbor_options=options,
            harbor_cost_usd=context.get("cost_usd"),
        )
