"""ATIF (Agent Trajectory Interchange Format) -- the ONE internal trajectory.

=========================================================================
PORTED FROM HARBOR. Apache License 2.0.
=========================================================================

Every module in this package is a derived work of Harbor's ATIF model package:

    harbor repo, `src/harbor/models/trajectories/`
        __init__.py   agent.py       content.py       final_metrics.py
        metrics.py    observation.py observation_result.py
        step.py       subagent_trajectory_ref.py      tool_call.py
        trajectory.py

Harbor is licensed under the Apache License, Version 2.0. A copy of that
licence ships with the Harbor source. Each file in this package carries a header
naming the exact Harbor file it came from.

WHAT CHANGED IN THE PORT, in full:

  * import paths -- `harbor.models.trajectories.X` became the sibling modules of
    this package;
  * module docstrings gained the attribution headers;
  * `trajectory.py` gained a module constant, `SCHEMA_VERSION`, naming the
    version this repo emits.

Nothing else. In particular NO FIELD WAS TRIMMED, including the several this
repo will never populate (`prompt_token_ids`, `logprobs`, `subagent_*`, the
audio half of `content.py`). The point of an interchange format is that it is
the same format everywhere; a "lite" ATIF that dropped the fields we did not
happen to need would be a private format wearing ATIF's name, and a consumer
handed one could not rely on the spec.

WHY THIS EXISTS HERE

Before this package, each adapter carried its own trajectory shape and its own
renderer, and `core/judge.py` dispatched between them. That put the vendor's
data model on the critical path of the MEASUREMENT: adding a third agent meant a
third renderer and a third chance for the judge to see a subtly different log.
Now every adapter converts its native output to a `Trajectory` and there is
exactly one renderer (`core.render.render_trajectory`), so the judge reads the
same shape whatever produced it.
"""

from .agent import Agent
from .content import AudioSource, ContentPart, ImageSource
from .final_metrics import FinalMetrics
from .metrics import Metrics
from .observation import Observation
from .observation_result import ObservationResult
from .step import Step
from .subagent_trajectory_ref import SubagentTrajectoryRef
from .tool_call import ToolCall
from .trajectory import SCHEMA_VERSION, Trajectory

__all__ = [
    "SCHEMA_VERSION",
    "Agent",
    "AudioSource",
    "ContentPart",
    "FinalMetrics",
    "ImageSource",
    "Metrics",
    "Observation",
    "ObservationResult",
    "Step",
    "SubagentTrajectoryRef",
    "ToolCall",
    "Trajectory",
]
