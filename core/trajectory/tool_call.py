"""Tool call model for ATIF trajectories.

PORTED FROM HARBOR, Apache License 2.0.
Source: `src/harbor/models/trajectories/tool_call.py` in the Harbor repository.
Derived work; see `core/trajectory/__init__.py` for the full notice. Fields are
Harbor's, unmodified.
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call within a step."""

    tool_call_id: str = Field(
        default=...,
        description="Unique identifier for this specific tool call",
    )
    function_name: str = Field(
        default=...,
        description="The name of the function or tool being invoked",
    )
    arguments: dict[str, Any] = Field(
        default=...,
        description="Arguments passed to the function (can be empty dict)",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Custom tool-call-level metadata (e.g., timeout, retry count, tool version). "
            "Added in ATIF-v1.7."
        ),
    )

    model_config = {"extra": "forbid"}
