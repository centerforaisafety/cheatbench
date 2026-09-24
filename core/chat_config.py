"""Resolve direct-chat settings from canonical models plus optional overrides."""
from copy import deepcopy
import os
import re
from pathlib import Path

from .config import load_models
from . import routing


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str) and re.fullmatch(r'\$\{[A-Za-z_][A-Za-z0-9_]*\}', value):
        name = value[2:-1]
        if not os.environ.get(name):
            raise ValueError(f'{name} is required by the chat configuration')
        raw = os.environ[name]
        return int(raw) if raw.isdecimal() else raw
    return value


def load_chat_config(path: Path, name: str, overrides: Path | None = None) -> dict:
    from .model_settings import model_entry, api_generation, factory_model
    entry = model_entry(path, name)
    if overrides is not None:
        override = load_models(overrides).get(name, {})
        if not isinstance(override, dict):
            raise ValueError(f'{overrides}: {name} must be a mapping')
        unknown = set(override) - {'model', 'generation_config', *routing.ROUTING_KEYS}
        if unknown:
            raise ValueError(f'{overrides}: unknown override keys {sorted(unknown)}')
        entry = _merge(entry, override)
    if 'options' in entry:
        raise ValueError('Use generation_config, not options')
    model = entry.get('model') or name
    route = routing.resolve(name, entry, where=str(path))
    gen = _expand(deepcopy(entry.get('generation_config') or {}))
    sent, extra, unsupported = api_generation(
        model, gen, route.get('extra_body'), legacy_display='harness' not in entry)
    return {'model': model, 'factory_model': factory_model(name, model, route, str(path)),
            'generation_config': sent, 'unsupported': unsupported,
            **route, 'extra_body': extra,
            'entry_generation_config': gen, 'entry_extra_body': route.get('extra_body', {})}
