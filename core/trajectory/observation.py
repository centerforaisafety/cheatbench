"""Observation model for ATIF trajectories.

PORTED FROM HARBOR, Apache License 2.0.
Source: `src/harbor/models/trajectories/observation.py` in the Harbor
repository. Derived work; see `core/trajectory/__init__.py` for the full notice.
Fields are Harbor's, unmodified.
"""

from pydantic import BaseModel, Field

from .observation_result import ObservationResult


class Observation(BaseModel):
    """Environment feedback/result after actions or system events."""

    results: list[ObservationResult] = Field(
        default=...,
        description="Array of result objects from tool calls or actions",
    )

    model_config = {"extra": "forbid"}
