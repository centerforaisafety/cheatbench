"""Per-model API routing: the optional keys a `models:` entry may carry.

    api_key_env:   NAME of the host environment variable holding the credential
                   (read from .env by run.py / judge.py). Absent: the adapter's
                   own `API_KEY_ENV` class default, announced once on stderr.
    api_base_url:  where the model is reached. A literal URL, or `${VAR}`
                   interpolated from the host environment when the config is
                   LOADED. Absent: the provider's default (for the agent
                   adapters, their PASSTHROUGH_ENV fallback still applies).
    api_base_url_env: NAME of the environment variable holding the base URL.
                   Resolved at construction/config load, overriding api_base_url.
                   A named but missing or blank variable is an error.
    extra_body:    a dict deep-merged into every JSON request body the harness
                   sends for this model (a judge sends it as the SDK's
                   `extra_body`; an agent CLI cannot, so its runner routes the
                   CLI through `core/agents/forwarder.py`, which merges it).

Every consumer -- run.py for the agent path, core/judge.py for judges -- goes
through `resolve()` so the routing keys mean the same thing everywhere. Nothing
here knows a gateway URL, a model list or a provider name: values come from the
yaml and the environment, never from this file.
"""
from __future__ import annotations

import os
import re
import sys
from urllib.parse import urlsplit, urlunsplit

ROUTING_KEYS = ("api_key_env", "api_base_url", "api_base_url_env", "extra_body")

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Announced once per (entry, adapter) per process, not once per episode.
_WARNED: set = set()


def interpolate(value: str, *, where: str) -> str:
    """`${VAR}` -> its value from the environment. An unset VAR is fatal.

    Fatal on purpose: an entry that says `${OPENAI_BASE_URL}` is claiming to
    route through that URL, and quietly falling back to the provider default
    would be a run whose config file lies about where its requests went.
    """
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        got = os.environ.get(name)
        if not got:
            raise SystemExit(
                f"{where}: `${{{name}}}` names environment variable {name}, "
                f"which is not set (add it to .env, or remove the key)")
        return got
    return _VAR.sub(_sub, str(value))


def resolve_base_url(base: str | None, env_name: str | None = None) -> str | None:
    """Read an explicitly named URL variable at call time; never silently fall back."""
    if env_name is None:
        return base
    if not isinstance(env_name, str) or not env_name.strip():
        raise ValueError("api_base_url_env must be a non-empty environment variable name")
    env_name = env_name.strip()
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(f"api_base_url_env names {env_name}, which is not set or blank (add it to .env)")
    if urlsplit(value).scheme not in ("http", "https") or not urlsplit(value).netloc:
        raise ValueError(f"api_base_url_env names {env_name}, which must contain an HTTP(S) URL")
    return value


def resolve(name: str, entry: dict, *, where: str) -> dict:
    """The routing fields of one `models:` entry, interpolated and validated.

    Returns `{"api_key_env": str | None, "api_base_url": str, "extra_body":
    dict}`. `api_key_env` is None when the entry omits it -- the CONSUMER owns
    the default (each adapter's `API_KEY_ENV`), so the fallback is applied
    there, through `default_key_env()`, and announced.
    """
    key_env = entry.get("api_key_env")
    if key_env is not None and (not isinstance(key_env, str) or not key_env.strip()):
        raise SystemExit(f"{where}: model '{name}': api_key_env must be a "
                         f"non-empty variable NAME, got {key_env!r}")
    try:
        base = resolve_base_url(entry.get("api_base_url"), entry.get("api_base_url_env"))
    except ValueError as exc:
        raise SystemExit(f"{where}: model '{name}': {exc}") from exc
    if base is not None and not isinstance(base, str):
        raise SystemExit(f"{where}: model '{name}': api_base_url must be a "
                         f"string URL (or ${{VAR}}), got {base!r}")
    base = interpolate(base, where=f"{where}: model '{name}': api_base_url") \
        if base else ""
    if base and not urlsplit(base).scheme:
        raise SystemExit(f"{where}: model '{name}': api_base_url {base!r} "
                         f"has no scheme (http:// or https://)")
    extra = entry.get("extra_body")
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise SystemExit(f"{where}: model '{name}': extra_body must be a "
                         f"mapping, got {type(extra).__name__}")
    return {"api_key_env": key_env.strip() if key_env else None,
            "api_base_url": base.strip(),
            "extra_body": dict(extra)}


def default_key_env(name: str, entry_key_env: str | None, consumer: str,
                    fallback: str) -> tuple[str, str]:
    """(the variable name to read, where it came from: "entry" | "default").

    Announces the fallback ONCE per (entry, consumer) so old entries keep
    working while every run log says which credential variable was assumed.
    """
    if entry_key_env:
        return entry_key_env, "entry"
    tag = (name, consumer)
    if tag not in _WARNED:
        _WARNED.add(tag)
        print(f"model '{name}' has no `api_key_env:`; the {consumer} default "
              f"{fallback} is assumed. Add `api_key_env: {fallback}` to the "
              f"entry in configs/models.yaml to make it explicit.",
              file=sys.stderr, flush=True)
    return fallback, "default"


def require_key(key_env: str, *, name: str, consumer: str) -> str:
    """The credential's VALUE, or a refusal naming the entry and the variable."""
    got = os.environ.get(key_env)
    if not got:
        raise SystemExit(
            f"model '{name}' routes its credential through {key_env}, which is "
            f"not set (add it to .env) -- the {consumer} cannot run without it")
    return got


def sanitise_url(url: str | None) -> str | None:
    """A URL fit for a run record: scheme, host, port and path only.

    Userinfo, query and fragment are dropped -- a credential has no business
    in a base URL, but a record must not be the place it leaks if one is put
    there anyway.
    """
    if not url:
        return None
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def url_host(url: str | None) -> str | None:
    if not url:
        return None
    return urlsplit(url).hostname or None
