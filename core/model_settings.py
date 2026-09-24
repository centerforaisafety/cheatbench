"""Model settings translated explicitly for a harness or direct API consumer."""
from copy import deepcopy
import inspect
from pathlib import Path

from .config import load_models
from . import routing


def model_entry(path: Path, name: str) -> dict:
    models = load_models(path)
    entry = models.get(name)
    if name == "default" or not isinstance(entry, dict):
        raise SystemExit(f"model {name!r} not in {path}; entries: {sorted(k for k in models if k != 'default')}")
    if "options" in entry:
        raise SystemExit(f"{name}: use generation_config or harness.options, not options")
    entry = deepcopy(entry)
    if not isinstance(entry.get("generation_config", {}), dict):
        raise SystemExit(f"{name}: generation_config must be a mapping")
    if "harness" in entry:
        from .agents.config import validate_version
        from .agents.factory import AgentFactory
        harness = entry["harness"]
        if not isinstance(harness, dict) or set(harness) - {"name", "version", "options"}:
            raise SystemExit(f"{name}: harness accepts name, version and options")
        if harness.get("name") not in AgentFactory.names():
            raise SystemExit(f"{name}: unknown harness {harness.get('name')!r}")
        validate_version(harness.get("version"), f"{name}.harness.version", agent=harness["name"])
        if not isinstance(harness.get("options", {}), dict):
            raise SystemExit(f"{name}: harness.options must be a mapping")
    return entry


def _put(mapping, key, value):
    if key in mapping and mapping[key] != value:
        raise ValueError(f"conflicting {key} settings")
    mapping[key] = value


def _effort(config):
    """Accept archived provider spellings, but reject conflicting declarations."""
    out = deepcopy(config)
    for alias in ("effort", "thinking_effort"):
        if alias in out:
            _put(out, "reasoning_effort", out.pop(alias))
    return out


def harness_generation(agent: str, model: str, config: dict) -> dict:
    out = _effort(config)
    if "reasoning_effort" in out:
        if agent == "claude-sdk":
            out["effort"] = out.pop("reasoning_effort")
        elif agent == "kimi-code":
            out["thinking_effort"] = out.pop("reasoning_effort")
        elif agent == "terminus-2" and model.startswith(("anthropic/", "claude-")):
            _put(out.setdefault("output_config", {}), "effort", out.pop("reasoning_effort"))
    return out


def harness_settings(entry: dict, agent: str) -> tuple[dict, dict]:
    """Only the selected harness receives its options; task policy is separate."""
    gen = deepcopy(entry.get("generation_config") or {})
    extra = deepcopy(entry.get("extra_body") or {})
    harness = entry.get("harness") or {}
    if harness.get("name") == agent:
        options = deepcopy(harness.get("options") or {})
        option_extra = options.pop("extra_body", {})
        if not isinstance(option_extra, dict):
            raise ValueError("harness.options.extra_body must be a mapping")
        for key, value in option_extra.items():
            _put(extra, key, value)
        for key, value in options.items():
            _put(gen, key, value)
    gen = harness_generation(agent, entry.get("model", ""), gen)
    if agent == "terminus-2" and "kimi" in entry.get("model", "").lower() and "reasoning_effort" in gen:
        thinking = extra.setdefault("thinking", {})
        _put(thinking, "type", "enabled")
        _put(thinking, "effort", gen.pop("reasoning_effort"))
    return gen, extra


def api_generation(model: str, config: dict, extra_body: dict | None = None,
                   *, legacy_display: bool = False, validate: bool = True) -> tuple[dict, dict, dict]:
    """Translate shared settings for Messages or Chat Completions.

    Legacy display keys are recorded only when loading archived configurations.
    New configurations keep consumer-specific display options under harness.
    """
    gen = _effort(config)
    extra = deepcopy(extra_body or {})
    for key, value in gen.pop("extra_body", {}).items():
        _put(extra, key, value)
    unsupported = {}
    if "reasoning_summary" in gen:
        if not legacy_display:
            raise ValueError("reasoning_summary belongs in harness.options for Codex; Chat Completions does not accept it")
        unsupported["reasoning_summary"] = gen.pop("reasoning_summary")
    reasoning = extra.get("reasoning")
    if isinstance(reasoning, dict) and "summary" in reasoning:
        if not legacy_display:
            raise ValueError("extra_body.reasoning.summary belongs in harness.options; Chat Completions does not accept it")
        unsupported["extra_body.reasoning.summary"] = reasoning.pop("summary")
        if not reasoning:
            del extra["reasoning"]
    if model.startswith(("anthropic/", "claude-")):
        if "reasoning_effort" in gen:
            _put(gen.setdefault("output_config", {}), "effort", gen.pop("reasoning_effort"))
        gen.setdefault("max_tokens", 16000)
        if type(gen["max_tokens"]) is not int or gen["max_tokens"] <= 0:
            raise ValueError("Anthropic max_tokens must be a positive integer")
        if validate:
            from anthropic.resources.messages import AsyncMessages
            create = AsyncMessages.create
    else:
        if "thinking_effort" in config or "kimi" in model.lower():
            if "reasoning_effort" in gen:
                thinking = extra.setdefault("thinking", {})
                _put(thinking, "type", "enabled")
                _put(thinking, "effort", gen.pop("reasoning_effort"))
        if validate:
            from openai.resources.chat.completions import AsyncCompletions
            create = AsyncCompletions.create
    if validate:
        # Catch invalid SDK arguments before installing a harness or paying for a call.
        inspect.signature(create).bind(None, model=model, messages=[], **gen,
                                       **({"extra_body": extra} if extra else {}))
    return gen, extra, unsupported


# Provider prefixes core/llm_agents.py builds a client for.
FACTORY_PROVIDERS = ("openai", "anthropic", "gemini", "xai", "openrouter", "bedrock")
# The vendors' own hosts, per prefix: there the factory's client sends the bare name.
# Any other host is a gateway, which takes the entry's id verbatim (factory_model).
NATIVE_HOSTS = {"openrouter": ("openrouter.ai",),
                "gemini": ("generativelanguage.googleapis.com",),
                "xai": ("api.x.ai",),
                "anthropic": ("api.anthropic.com",)}
# The Messages API requires a token cap and the entries set none: 16k output tokens,
# the paper's setting (Betley, Treutlein et al.) and the cap of every archived Claude
# record. Thinking counts against it; an episode's `stop_reason` shows a cut.
ANTHROPIC_MAX_TOKENS = 16000


def factory_model(name: str, model: str, route: dict, where: str) -> str:
    """The `provider/name` core/llm_agents.py builds a client from, for entry MODEL.

    The factory's vendor clients strip one leading `vendor/` and send the rest, which
    is what the vendor's own endpoint takes. An entry routed to a gateway instead
    (a host that is not the vendor's, NATIVE_HOSTS) keeps its id verbatim on the
    gateway's OpenAI-compatible route, as the grok-build, kimi-code and gemini-cli
    adapters send it. A vendor the factory has no client for -- `meta/muse-spark-1.3`
    (Meta's own endpoint) or a bare `deepseek-v4-pro` (DeepSeek's) -- is reached with
    the OpenAI client under the name the vendor expects, one leading `vendor/`
    stripped as the adapters and Harbor do; such an entry must route explicitly, the
    OpenAI client's default endpoint being the wrong vendor.
    """
    if model.startswith("claude-") and "/" not in model:
        return "anthropic/" + model
    provider, _, rest = model.partition("/")
    host = routing.url_host(route.get("api_base_url") or "") or ""
    if rest and provider in FACTORY_PROVIDERS:
        native = NATIVE_HOSTS.get(provider)
        if native and host and not host.endswith(native):
            # An `anthropic/` id keeps the Anthropic client (the Messages API takes
            # `thinking` and `output_config`) and the full id, as core/agents/claude_sdk.py
            # sends it on a gateway; the others ride the gateway's OpenAI-compatible route.
            return ("anthropic/" if provider == "anthropic" else "openai/") + model
        return model
    if not route.get("api_base_url"):
        raise SystemExit(f"model '{name}' in {where} has model={model!r}, which core/llm_agents.py "
                         f"has no client for, and no api_base_url to reach it through the "
                         f"OpenAI client")
    return "openai/" + (rest or model)
