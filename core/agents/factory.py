"""Which adapters exist, and how one is built.

Harbor's `harbor/src/harbor/agents/factory.py`, in miniature: the registry is a
dict from the adapter's NAME to the string `"module.path:ClassName"`, and the
class is imported only when that name is actually asked for. The point of the
indirection is the same as Harbor's -- the registry can name every adapter
without importing any of them, so listing the choices for `--agent` costs
nothing and a vendor module that fails to import breaks only its own adapter.
"""
from __future__ import annotations

import importlib

from .base import Agent
from ..trajectory import Trajectory


class AgentFactory:
    """The one place that knows which adapters exist."""

    _AGENT_MAP: dict[str, str] = {
        "terminus-2": "core.agents.terminus_2:Terminus2Agent",
        "claude-sdk": "core.agents.claude_sdk:ClaudeSDKAgent",
        "deepseek-harness": "core.agents.deepseek_harness:DeepSeekHarnessAgent",
        "codex": "core.agents.codex:CodexAgent",
        "gemini-cli": "core.agents.gemini_cli:GeminiCLIAgent",
        "grok-build": "core.agents.grok_build:GrokBuildAgent",
        "kimi-code": "core.agents.kimi_code:KimiCodeAgent",
        "muse-code": "core.agents.muse_code:MuseCodeAgent",
    }

    @classmethod
    def names(cls) -> list[str]:
        """Every registered adapter name. Importing nothing."""
        return sorted(cls._AGENT_MAP)

    @classmethod
    def get_agent_class(cls, name: str) -> type[Agent]:
        """The adapter class registered under `name`, imported on demand."""
        if name not in cls._AGENT_MAP:
            raise SystemExit(
                f"unknown agent {name!r}; have: {cls.names()}")
        agent_class = _import_agent_class(cls._AGENT_MAP[name])
        # The registry key and the class's own `name()` are two statements of
        # the same fact, and a run records the SECOND one. If they ever drift,
        # `--agent x` would produce records saying `y`, so say so here instead.
        if agent_class.name() != name:
            raise SystemExit(
                f"adapter registered as {name!r} calls itself "
                f"{agent_class.name()!r}")
        return agent_class

    @classmethod
    def create_agent_from_name(cls, name: str, **kwargs) -> Agent:
        """Build the adapter registered under `name`."""
        return cls.get_agent_class(name)(**kwargs)


def _import_agent_class(import_path: str) -> type[Agent]:
    """`"module.path:ClassName"` -> the class. Harbor's `_import_agent_class`."""
    module_path, _, class_name = import_path.partition(":")
    return getattr(importlib.import_module(module_path), class_name)


def make_agent(name: str, **kwargs) -> Agent:
    return AgentFactory.create_agent_from_name(name, **kwargs)


# What a transcript that predates the `agent` key must have been produced by.
# There was only one adapter when those runs were made, so this is a fact about
# the history rather than a guess.
LEGACY_AGENT = "claude-sdk"


def trajectory_from_transcript(transcript: dict) -> Trajectory:
    """A saved `transcript.json` as ATIF. THE back-compat reader.

    Runs recorded before ATIF stored the adapter's native messages under
    `messages`, which is the same key and the same shape the container still
    returns today -- so the adapter's own `to_trajectory` reads a historical
    transcript unchanged, and re-judging a year-old run needs no separate code
    path and no migration of anything on disk. This matters: being able to
    re-judge history is how a rubric change is shown to be a rubric change
    rather than a drift in the harness.

    The adapter is constructed with the transcript's own model and nothing
    else. It is used only to convert, never to run, so it needs no tool policy,
    no version pin and no credential.
    """
    # Imported external runs can carry validated ATIF directly. This preserves
    # their real agent identity instead of pretending a local adapter ran them.
    if "trajectory" in transcript:
        return Trajectory.model_validate(transcript["trajectory"])
    name = transcript.get("agent") or LEGACY_AGENT
    if name not in AgentFactory._AGENT_MAP:
        raise SystemExit(
            f"transcript was produced by agent {name!r}, which is not a "
            f"registered adapter; cannot convert it. have: "
            f"{AgentFactory.names()}")
    cls = AgentFactory.get_agent_class(name)
    return cls(model=transcript.get("model") or "").to_trajectory(transcript)
