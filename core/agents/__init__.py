"""Agent adapters and their shared interfaces.

Each adapter supplies a small bootstrap command, a stdin payload containing
runner source and task data, and a converter from native messages to ATIF.
Keeping the payload off the command line avoids exposing task and harness
source through the agent's process arguments. The runner consumes the payload
before starting the agent.

The shared renderer uses ATIF for both live logs and trajectory judging.
Adapters own vendor tool names, policy translation, API routing, and native
message conversion. InstalledAgent adds runtime installation and version checks;
images may already contain a matching CLI.

Add adapters to AgentFactory's lazy registry in factory.py. Its names() method
lists the available adapters. Each adapter's *_runner.py is read as source and
sent to the container, rather than imported by the host.

The interface layout and ATIF types follow Harbor (Apache-2.0; see NOTICE).
"""
from __future__ import annotations

from . import errors
from .base import Agent
from .errors import AgentError, ErrorPattern
from .factory import (
    LEGACY_AGENT,
    AgentFactory,
    make_agent,
    trajectory_from_transcript,
)
from .installed import InstalledAgent, link_bins, node_install

__all__ = [
    "Agent",
    "AgentError",
    "AgentFactory",
    "ClaudeSDKAgent",
    "CodexAgent",
    "DeepSeekHarnessAgent",
    "ErrorPattern",
    "GeminiCLIAgent",
    "GrokBuildAgent",
    "InstalledAgent",
    "KimiCodeAgent",
    "LEGACY_AGENT",
    "MuseCodeAgent",
    "errors",
    "link_bins",
    "make_agent",
    "node_install",
    "trajectory_from_transcript",
]

# The concrete adapters are reachable as `core.agents.ClaudeSDKAgent` but are
# NOT imported when the package is. That is the same promise `factory.py` makes
# -- naming an adapter must not import it -- and importing them here would break
# it for every consumer that only wanted `Agent` or the list of names.
_LAZY = {"ClaudeSDKAgent": "claude_sdk", "CodexAgent": "codex",
         "DeepSeekHarnessAgent": "deepseek_harness",
         "GeminiCLIAgent": "gemini_cli", "GrokBuildAgent": "grok_build",
         "KimiCodeAgent": "kimi_code", "MuseCodeAgent": "muse_code"}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(
        __import__(f"{__name__}.{module}", fromlist=[name]), name)
