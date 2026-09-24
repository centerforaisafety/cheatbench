"""Agent configuration model for ATIF trajectories.

PORTED FROM HARBOR, Apache License 2.0.
Source: `src/harbor/models/trajectories/agent.py` in the Harbor repository.
This is a derived work; see `core/trajectory/__init__.py` for the full notice.
The only change from Harbor's file is the module docstring. The FIELDS ARE
HARBOR'S and are deliberately not trimmed -- an interchange format that dropped
the fields one consumer happened not to need would stop being interchangeable.
"""

from typing import Any

from pydantic import BaseModel, Field


class Agent(BaseModel):
    """Agent configuration."""

    name: str = Field(
        default=...,
        description="The name of the agent system",
    )
    version: str = Field(
        default=...,
        description="The version identifier of the agent system",
    )
    model_name: str | None = Field(
        default=None,
        description="Default LLM model used for this trajectory",
    )
    tool_definitions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Array of tool/function definitions available to the agent. Each element follows OpenAI's function calling schema.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom agent configuration details",
    )

    model_config = {"extra": "forbid"}
