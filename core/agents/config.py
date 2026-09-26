"""Resolve agent CLI versions without coupling models, tasks, or adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .factory import AgentFactory

_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?(?:[-+][A-Za-z0-9._+-]+)?")


def validate_version(value, source: str, *, agent: str | None = None) -> str:
    fixed = AgentFactory.get_agent_class(agent).FIXED_VERSION if agent else None
    if fixed is not None:
        if value != fixed:
            raise SystemExit(f"{source}: {agent} always uses pinned source {fixed}; "
                             "latest, tags and other revisions are not supported")
        return fixed
    if not isinstance(value, str) or not (value == "latest" or _VERSION.fullmatch(value)):
        raise SystemExit(f"{source}: use a quoted exact version (e.g. \"0.154.0\") "
                         "or latest; version ranges and commands are not supported")
    return value


@dataclass(frozen=True)
class VersionSelection:
    version: str
    source: str
    config_path: str
    config_sha256: str


def resolve_version(agent: str, *, cli_version: str = "",
                    harness_override: bool = False,
                    model_harness: dict | None = None,
                    models_path: Path | None = None) -> VersionSelection:
    """Use the model pin by default; an explicit harness selects its latest release."""
    adapter = AgentFactory.get_agent_class(agent)
    if cli_version:
        version = validate_version(cli_version, "--harness-version", agent=agent)
        return VersionSelection(version, "--harness-version", "", "")
    if not harness_override and (model_harness or {}).get("name") == agent:
        version = validate_version(model_harness.get("version"), "models.yaml:harness", agent=agent)
        path = Path(models_path).resolve()
        return VersionSelection(version, "models.yaml:harness", str(path),
                                hashlib.sha256(path.read_bytes()).hexdigest())
    # Source-based adapters have no package release channel. Their immutable
    # revision remains owned by the adapter, not by a second configuration file.
    if adapter.FIXED_VERSION:
        return VersionSelection(adapter.FIXED_VERSION, "adapter:fixed", "", "")
    return VersionSelection("latest", "--harness" if harness_override else "adapter:default", "", "")


def resume_version(path: Path, agent: str, requested: str) -> str | None:
    """Keep a resumed run's installed-version record and reject pin changes."""
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot verify resumed run's agent version in {path}: {e}") from e
    actual = previous.get("agent_version")
    old_request = previous.get("agent_version_requested") or "latest"
    compatible = (old_request == "latest" if requested == "latest"
                  else (actual or old_request) == requested)
    if previous.get("agent") != agent or not compatible:
        raise SystemExit(f"{path}: cannot resume {previous.get('agent')} "
                         f"version {actual or old_request} with {agent} version {requested}; "
                         "use a new --run-id or --redo")
    return actual
