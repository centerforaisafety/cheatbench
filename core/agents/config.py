"""Resolve agent CLI versions without coupling models, tasks, or adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import yaml

from .factory import AgentFactory

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "agents.yaml"
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


def parse_agents(raw, source: str) -> dict[str, dict]:
    if not isinstance(raw, dict):
        raise SystemExit(f"{source}: expected a map of adapter names to configuration blocks")
    names = AgentFactory.names()
    for name, entry in raw.items():
        if name not in names:
            raise SystemExit(f"{source}: unknown adapter {name!r}; have: {names}")
        if not isinstance(entry, dict):
            raise SystemExit(f"{source}.{name}: expected a block with a version key")
        unknown = set(entry) - {"version"}
        if unknown:
            raise SystemExit(f"{source}.{name}: unknown key(s) {sorted(unknown)}")
        validate_version(entry.get("version"), f"{source}.{name}.version", agent=name)
    return {name: dict(entry) for name, entry in raw.items()}


@dataclass(frozen=True)
class VersionSelection:
    version: str
    source: str
    config_path: str
    config_sha256: str


def resolve_version(agent: str, *, task_config: dict[str, dict] | None = None,
                    cli_version: str = "", config_path: Path = DEFAULT_CONFIG,
                    model_harness: dict | None = None,
                    models_path: Path | None = None) -> VersionSelection:
    """CLI > model pin > task > repository fallback; an explicit config overrides model pins."""
    path = Path(config_path).resolve()
    try:
        raw = path.read_bytes()
        defaults = parse_agents(yaml.safe_load(raw), str(path))
    except (OSError, yaml.YAMLError) as e:
        raise SystemExit(f"cannot load agents config {path}: {e}") from e
    if agent not in AgentFactory.names():
        raise SystemExit(f"unknown agent {agent!r}; have: {AgentFactory.names()}")
    overrides = parse_agents(task_config if task_config is not None else {}, "task.yaml:agent_config")
    if cli_version:
        version = validate_version(cli_version, "--agent-version", agent=agent)
        source = "--agent-version"
    elif (model_harness or {}).get("name") == agent and path == DEFAULT_CONFIG.resolve():
        version = validate_version(model_harness.get("version"), "models.yaml:harness", agent=agent)
        source = "models.yaml:harness"
        path = Path(models_path).resolve()
        raw = path.read_bytes()
    elif agent in overrides:
        version, source = overrides[agent]["version"], "task.yaml:agent_config"
    elif agent in defaults:
        version, source = defaults[agent]["version"], "agents.yaml"
    else:
        raise SystemExit(f"{path}: no version for {agent!r}; add one or set "
                         "task.yaml:agent_config or --agent-version")
    return VersionSelection(version, source, str(path), hashlib.sha256(raw).hexdigest())


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
