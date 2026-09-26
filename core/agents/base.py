"""Agent interface and shared routing, recording, and cost helpers.

Adapters translate task policies and native messages for the harness. CLI
installation belongs in `installed.py`; adapters that need no installation can
subclass `Agent` directly. This split follows Harbor's agent interfaces.
"""
from __future__ import annotations

import abc
import os
import sys
from pathlib import Path

from .. import render
from ..trajectory import Step, Trajectory
from ..trajectory import Agent as ATIFAgent
from . import errors


# Task policies describe capabilities such as web_search and web_fetch, rather
# than vendor tool names. Each adapter declares supported keys in ENFORCEABLE
# and translates them in _tools_for(); unsupported keys stop the run.
#
# Provider-hosted web tools can bypass the container's network restrictions or
# injected sites. Verify enforcement against the actual CLI before declaring a
# capability. Adapter modules and their tests document vendor-specific controls.

class Agent(abc.ABC):
    """One agent runtime with a shared record and trajectory interface.

    Subclasses implement name(), _tools_for(), bootstrap, blob(),
    to_trajectory(), and reported_cost_usd(). They also declare ENFORCEABLE and
    USAGE_KEYS; an empty usage tuple explicitly disables cost estimation.

    This class validates policies, normalizes records, estimates costs, and
    renders live logs. InstalledAgent adds CLI installation and version checks.
    """

    # Supported task-policy keys. The empty default rejects every supplied key.
    ENFORCEABLE: set = set()

    # Some source-based adapters always run one reviewed immutable revision.
    FIXED_VERSION: str | None = None

    # Additional installer hosts beyond shared package infrastructure.
    # Restricted tasks allow these for the whole episode.
    INSTALL_HOSTS: tuple[str, ...] = ()

    # Default host credential variable; model routing may override it.
    # container_key_envs() determines the names exported inside the container.
    API_KEY_ENV: str = "ANTHROPIC_API_KEY"

    # Host environment variables copied into the container when they are set.
    # Vendor-specific knobs, so they belong to the adapter and not to run.py.
    PASSTHROUGH_ENV: tuple = ()

    # -- per-model routing (configs/models.yaml `api_key_env:` /
    #    `api_base_url:` / `extra_body:`; resolved by core/routing.py) -------
    #
    # The names this adapter's CLI reads its credential from INSIDE the
    # container. The host variable is whatever the model entry's `api_key_env:`
    # names (default: API_KEY_ENV); the sandbox exports its VALUE under every
    # name here, so a CLI that only reads e.g. `KIMI_MODEL_API_KEY` can be
    # handed the gateway key under its own name. Empty means "the same name as
    # the host variable", which is what every adapter got before this existed.
    CLI_KEY_ENV: tuple = ()

    # The names this adapter's CLI (or the task's grader, running beside it)
    # reads a base URL from inside the container. When the model entry sets
    # `api_base_url:`, `routed_env()` exports it under each of these; when it
    # does not, whatever PASSTHROUGH_ENV copied from the host stands, which is
    # the pre-existing fallback. An adapter whose CLI takes the URL some other
    # way (codex: config.toml) reads `self.api_base_url` in `blob()` instead.
    CLI_BASE_URL_ENV: tuple = ()

    # Additional native payload fields copied into normalized episode records.
    EXTRA_RECORD_KEYS: tuple = ()

    # -- identity ---------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """Registry name used by AgentFactory, configuration, and run records."""

    def version(self) -> str | None:
        """The agent version that ACTUALLY ran, for the run record.

        Resolution order, and the reason for it:

          1. what the container reported after `check || install` -- the only
             source that describes the build that really executed;
          2. the resolved CLI or models.yaml pin, when the episode has not run yet
             (run.py prints this before any container starts);
          3. None, meaning "latest at episode time, not yet resolved".

        The requested version and its source are recorded separately from the
        version reported by the installed binary.
        """
        return self._resolved_version or self.pinned_version

    def observe_version(self, resolved: str | None) -> None:
        """Record the version the container reported. Called once per episode."""
        if resolved:
            self._resolved_version = str(resolved)

    # Keys of a model's `generation_config:` that THIS adapter translates.
    #
    # A generation setting describes how the MODEL generates; only its DELIVERY
    # is vendor-specific -- a codex `-c model_reasoning_*` flag, a
    # ClaudeAgentOptions field, a litellm kwarg -- so the setting is written
    # once per model in configs/models.yaml and each consumer owns its own
    # translation of it.
    #
    # A key this adapter does not translate is a LOUD refusal at preflight,
    # naming the key and the adapter -- never a silent drop. A silently dropped
    # generation setting is a run whose config file lies about it.
    GENERATION_KEYS: tuple = ()
    PERMISSION_MODES = ("bypassPermissions",)
    TURN_LIMIT = None  # A concrete adapter declares the native counting unit.

    def __init__(self, *, model: str, generation_config: dict | None = None,
                 max_turns: int | None = None,
                 permission_mode: str = "bypassPermissions",
                 version: str | None = None,
                 base_url: str | None = None,
                 api_key_env: str | None = None,
                 api_base_url: str | None = None,
                 extra_body: dict | None = None):
        self.model = model
        from ..model_settings import harness_generation
        self.generation_config = harness_generation(self.name(), model, generation_config or {})
        self.max_turns = max_turns
        self.permission_mode = permission_mode or "bypassPermissions"
        # The model entry's ROUTING (core/routing.py), already interpolated:
        #
        #   api_base_url  the resolved URL this model is reached at, or "" for
        #                 "the provider default / this adapter's PASSTHROUGH
        #                 fallback". `routed_env()` exports it under
        #                 CLI_BASE_URL_ENV; an adapter whose CLI wants it in a
        #                 config file reads it in `blob()`.
        #   extra_body    merged into every JSON request the CLI sends. A CLI
        #                 cannot do that itself, so a non-empty dict is the
        #                 opt-in for the runner to route the CLI through
        #                 core/agents/forwarder.py (see `routing_payload()`).
        #
        # `api_key_env` below is the HOST variable name and is recorded with
        # where it came from; `api_key_env_source` is "entry" when the model
        # entry named it and "default" when the class's API_KEY_ENV was assumed.
        self.api_base_url = (api_base_url or "").strip()
        self.extra_body = dict(extra_body or {})
        self.api_key_env_source = "entry" if api_key_env else "default"
        # Per-model overrides, both optional and both None by default so an
        # entry that sets neither runs exactly as before. They let ONE model in
        # configs/models.yaml reach a different provider than the adapter's
        # class defaults:
        #
        #   base_url      where this adapter's CLI reaches its provider. None
        #                 means "the adapter's usual origin" (for codex, the
        #                 OPENAI_BASE_URL passthrough).
        #   api_key_env   which environment variable holds the credential. When
        #                 set it SHADOWS the class's API_KEY_ENV for this
        #                 instance only, so `agent.API_KEY_ENV` -- which run.py
        #                 and core/trial.py read to pick the one key that enters
        #                 the container -- names the override.
        #
        # The point is to point a single model (gpt-6-astra) at OpenRouter with
        # its own key WITHOUT moving the judge or the grader, which keep the
        # class defaults (OPENAI_BASE_URL / OPENAI_API_KEY through the gateway).
        self.base_url = base_url or None
        if api_key_env:
            self.API_KEY_ENV = api_key_env
        # The agent CLI version to pin, or None for "whatever is latest at
        # episode time".
        self.requested_version = version or None
        self.pinned_version = None if version == "latest" else (version or None)
        # Filled in from the container's record; see observe_version().
        self._resolved_version: str | None = None
        # Set by apply_tool_policy, which core/trial.py calls before any episode.
        self.policy: dict | None = None
        self._tools: list | None = None

    # -- the task's tool policy ------------------------------------------
    def apply_tool_policy(self, policy: dict | None) -> list[str]:
        """Translate a task's whole `tools:` block into this adapter's toolset.

        Called once, before any episode. Every key must be in ENFORCEABLE; a key
        the adapter cannot hold raises, and the caller turns that into a refusal
        naming the task. Missing keys default to False, so the closed-book
        setting is what a task gets by omission.

        Returns the resolved tool list, which is also available afterwards as
        `.tools`.
        """
        policy = dict(policy or {})
        unknown = sorted(set(policy) - set(self.ENFORCEABLE))
        if unknown:
            raise ValueError(
                f"adapter {self.name()!r} cannot enforce {unknown}; it can "
                f"enforce {sorted(self.ENFORCEABLE)}")
        self.policy = {k: bool(policy.get(k, False))
                       for k in sorted(self.ENFORCEABLE)}
        self._tools = self._tools_for(self.policy)
        return list(self._tools)

    @abc.abstractmethod
    def _tools_for(self, policy: dict) -> list[str]:
        """This adapter's toolset under `policy`. All keys present, all bools."""

    @property
    def tools(self) -> list[str]:
        """The tools this adapter will actually run with, as resolved.

        Recorded in run.json and every episode record, so a result always says
        what the agent could reach for.
        """
        if self._tools is None:
            raise RuntimeError("apply_tool_policy has not been called")
        return list(self._tools)

    # -- what the sandbox needs ------------------------------------------
    @property
    @abc.abstractmethod
    def bootstrap(self) -> str:
        """The one-liner that goes in the container's argv."""

    @abc.abstractmethod
    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
        """Serialize the task and runner source for the container's stdin.

        `install` configures the agent runtime; `task_install` configures task
        dependencies. Both use check/install commands, but their outcomes are
        recorded separately. A task-install failure does not stop the agent.

        `grader_src` is grader module source, or an empty string when no module
        is supplied. Each adapter's runner owns its grader transport.
        """

    # -- native record -> ATIF --------------------------------------------
    @abc.abstractmethod
    def to_trajectory(self, raw: dict) -> Trajectory:
        """This adapter's native record, converted to an ATIF `Trajectory`.

        `raw` is either the container's stdout record or a saved
        `transcript.json` -- they carry the same `messages` under the same key,
        which is what makes re-judging a historical run possible at all.

        This is the ONLY adapter-specific thing left in the measurement path.
        Everything downstream (the judge's log, the verdict, the record) is
        driven off the `Trajectory` and cannot tell which agent produced it.
        """

    # -- what one episode cost ---------------------------------------------
    #
    # Cost handling is ported from Harbor (Apache-2.0):
    # agents/installed/codex.py::_compute_cost_from_pricing and
    # claude_code.py::_parse_total_cost_from_stream_json/_estimate_total_cost_from_steps.
    # Prefer the vendor's reported cost, otherwise estimate from token usage.
    # Every concrete adapter must declare both sources, even if unavailable.

    # Which record fields carry this adapter's token usage, best source first.
    # Claude has two (the SDK's per-turn stream tally survives a killed episode;
    # the ResultMessage's `usage` does not), Codex has one.
    USAGE_FIELDS: tuple = ("usage",)

    # What THIS vendor calls the four token categories, in this order:
    #
    #     (prompt, completion, cache read, cache write)
    #
    # None means "not declared" and is a definition-time error for a concrete
    # adapter. The empty tuple is the explicit way to say "this adapter's
    # records carry no token usage, so no estimate is possible" -- which is a
    # statement, where a forgotten attribute is an accident.
    USAGE_KEYS: tuple | None = None

    # Whether this vendor's prompt count ALREADY INCLUDES its cache-read and
    # cache-write tokens. Codex's `input_tokens` does (input == cached + written
    # + fresh, and total == input + output). Anthropic's does NOT: its
    # `input_tokens` is the fresh remainder only, with the cache legs reported
    # beside it. Getting this backwards is the single most expensive mistake
    # available here -- see `cost_from_usage` -- so it is stated per adapter and
    # pinned by a test.
    PROMPT_TOKENS_INCLUDE_CACHE: bool = True

    def __init_subclass__(cls, **kwargs) -> None:
        """Require cost declarations when an adapter becomes concrete.

        Intermediate abstract classes are checked when their subclasses supply
        the remaining runtime methods.
        """
        super().__init_subclass__(**kwargs)
        pending = {n for n in dir(cls)
                   if getattr(getattr(cls, n, None), "__isabstractmethod__",
                              False)}
        if pending - {"reported_cost_usd"}:
            return                      # still abstract for other reasons
        if "reported_cost_usd" in pending:
            raise TypeError(
                f"adapter {cls.__name__} does not implement "
                f"reported_cost_usd(). Every adapter must SAY where its cost "
                f"comes from: return the vendor's own dollar figure from the "
                f"episode record, or return None when the vendor reports none "
                f"(Codex) so the shared litellm estimate is used instead.")
        if cls.USAGE_KEYS is None:
            raise TypeError(
                f"adapter {cls.__name__} does not declare USAGE_KEYS. Name the "
                f"four token categories as this vendor spells them -- (prompt, "
                f"completion, cache read, cache write) -- and set "
                f"PROMPT_TOKENS_INCLUDE_CACHE, or set USAGE_KEYS = () to state "
                f"that this adapter's records carry no usage to price.")
        if cls.USAGE_KEYS and len(cls.USAGE_KEYS) != 4:
            raise TypeError(
                f"adapter {cls.__name__} declares {len(cls.USAGE_KEYS)} "
                f"USAGE_KEYS; it must name exactly four, in the order "
                f"(prompt, completion, cache read, cache write).")

    @abc.abstractmethod
    def reported_cost_usd(self, raw: dict) -> float | None:
        """The VENDOR's own dollar figure for this episode, or None.

        Abstract on purpose (see the block comment above): an adapter must
        state, in code, whether the thing it drives prices its own episodes.

        Returning None is a perfectly good answer -- it is Codex's answer, and
        it means "this vendor never told us, so price the tokens ourselves".
        What is not a good answer is silence, which is what an adapter that
        simply never set `cost_usd` was giving.
        """

    def cost_from_usage(self, *, prompt_tokens: int, completion_tokens: int,
                        cached_tokens: int = 0, cache_write_tokens: int = 0,
                        model_name: str | None = None) -> float | None:
        """Four token counts -> dollars, through litellm's pricing table.

        `prompt_tokens` IS INCLUSIVE: it counts every input token of the call,
        cache reads and cache writes among them. litellm subtracts the two cache
        legs back out and prices each at its own rate, so passing an EXCLUSIVE
        prompt count does not merely round oddly -- on a cache-heavy episode it
        goes NEGATIVE (a real one here: -$4.70 instead of $1.19). The opposite
        slip, pricing every input token at the fresh rate, overstates a
        95%-cached episode by more than 5x while looking entirely plausible.
        Neither is detectable by eye in a results table, so both are pinned by
        tests, and a prompt count smaller than its own cache legs raises here
        rather than returning a number.

        Returns None -- never 0.0 -- when the model is absent from the pricing
        table. They mean different things: "we do not know" is not "it was
        free", and a run whose costs are unknown must not be summed as if it
        were cheap.
        """
        import litellm                            # Harbor imports it here too

        name = model_name or self.model or ""
        # Harbor's key resolution, verbatim: the full name, then the name with
        # any provider prefix stripped. Our records carry `openai/gpt-5.6-sol`
        # and `anthropic/claude-opus-5`; litellm's table keys them bare.
        for key in (name, name.split("/", 1)[-1]):
            if key and key in litellm.model_cost:
                break
        else:
            return None

        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        cached_tokens = int(cached_tokens or 0)
        cache_write_tokens = int(cache_write_tokens or 0)
        if prompt_tokens < cached_tokens + cache_write_tokens:
            raise ValueError(
                f"prompt_tokens={prompt_tokens} is smaller than its own cache "
                f"legs (read={cached_tokens}, write={cache_write_tokens}); "
                f"cost_from_usage takes an INCLUSIVE prompt count. See "
                f"PROMPT_TOKENS_INCLUDE_CACHE on this adapter.")

        try:
            in_cost, out_cost = litellm.cost_per_token(
                model=key,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_write_tokens,
                cache_read_input_tokens=cached_tokens,
            )
        except Exception:  # noqa: BLE001 - Harbor swallows this the same way
            return None
        return in_cost + out_cost

    def usage_categories(self, raw: dict) -> dict | None:
        """This adapter's usage dict as the four INCLUSIVE categories.

        Concrete and shared: the only thing that varies between adapters is
        which record field carries the usage, what the vendor calls each
        category and whether its prompt count already includes the cache legs --
        all three of which are DATA on the adapter (`USAGE_FIELDS`,
        `USAGE_KEYS`, `PROMPT_TOKENS_INCLUDE_CACHE`), in the same way the tool
        set, the install check and the error patterns are.
        """
        if not self.USAGE_KEYS:
            return None
        p_key, c_key, r_key, w_key = self.USAGE_KEYS
        for field in self.USAGE_FIELDS:
            usage = (raw or {}).get(field)
            if not isinstance(usage, dict):
                continue

            def _n(key):
                value = usage.get(key)
                return int(value) if isinstance(value, (int, float)) else 0

            prompt, completion = _n(p_key), _n(c_key)
            cached, written = _n(r_key), _n(w_key)
            if not (prompt or completion or cached or written):
                # An all-zero tally is a usage field that was created but never
                # filled (Claude's stream counters on an episode that died
                # before its first turn). Fall through to the next source.
                continue
            if not self.PROMPT_TOKENS_INCLUDE_CACHE:
                prompt += cached + written
            return {"prompt_tokens": prompt, "completion_tokens": completion,
                    "cached_tokens": cached, "cache_write_tokens": written}
        return None

    def estimated_cost_usd(self, raw: dict) -> float | None:
        """What this episode's recorded tokens price out at, or None."""
        categories = self.usage_categories(raw)
        if not categories:
            return None
        return self.cost_from_usage(**categories)

    def episode_cost(self, raw: dict) -> tuple[float | None, str | None]:
        """(dollars, where the number came from) for one episode record.

        The precedence is the same for every adapter and lives here rather than
        in either of them: a figure the vendor itself reported is authoritative
        and is passed through UNCHANGED, and the token estimate is a fallback
        for when there is none. For Claude that fallback is the timed-out
        episode, whose `ResultMessage` -- and with it `total_cost_usd` -- never
        arrives; for Codex it is every episode, because the rollout carries
        token counts and never a price.

        `cost_source` travels with the number for exactly one reason: an
        estimate and a reported figure are not the same measurement, and a table
        that mixes them without saying so invites a comparison nobody can
        defend.
        """
        reported = self.reported_cost_usd(raw or {})
        if reported is not None:
            return float(reported), "reported"
        try:
            estimated = self.estimated_cost_usd(raw or {})
        except Exception as e:  # noqa: BLE001 - never lose a record over pricing
            print(f"[{self.name()}] cost estimate failed: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return None, None
        return (estimated, "estimated") if estimated is not None else (None, None)

    # -- the shared record envelope ----------------------------------------
    #
    # NOT abstract, and not per-adapter. Both runners are OUR code, and both
    # already return the same envelope around the vendor-specific `messages`:
    # the grader's state, the deliverables, what the install did, the wall
    # clock, the error. Two identical `parse()` implementations is how this
    # started, and the second copy is how they drift.
    def record(self, raw: dict) -> dict:
        """The container's one JSON line, normalised.

        `EXTRA_RECORD_KEYS` is the vendor-specific tail an adapter may add on
        top (for Codex, the exact `codex exec` line and the rollout it was read
        back from). `core/trial.py` copies those across without knowing what
        they are, which is what keeps the vendor's facts out of core.

        `cost_usd` is resolved here, for every adapter, by `episode_cost()`: the
        vendor's own figure when it reported one, else the shared litellm
        estimate off the token counts the record already carries. `cost_source`
        says which of the two it is.
        """
        raw = raw or {}
        cost_usd, cost_source = self.episode_cost(raw)
        out = {
            "messages": raw.get("messages") or [],
            "final_text": raw.get("final_text") or "",
            "n_turns": raw.get("n_turns"),
            "n_tool_calls": raw.get("n_tool_calls"),
            "wall_time": raw.get("wall_time"),
            "cost_usd": cost_usd,
            # "reported" (the vendor priced it), "estimated" (we priced its
            # tokens through litellm) or None (neither was possible).
            "cost_source": cost_source,
            "stream_usage": raw.get("stream_usage"),
            "usage": raw.get("usage"),
            "result_subtype": raw.get("result_subtype"),
            "terminal_reason": raw.get("terminal_reason"),
            "session_id": raw.get("session_id"),
            "init_tools": raw.get("init_tools"),
            "init_mcp_servers": raw.get("init_mcp_servers"),
            "grader_state": raw.get("grader_state"),
            "deliverables": raw.get("deliverables") or [],
            # What the runtime install did: "present" (already in the image),
            # "installed", "failed" or None. Recorded so a cold-start cost is
            # measurable and a silent install failure is not mistaken for a
            # model failure.
            "install": raw.get("install"),
            # The same, for the TASK's own `install:` block: "present",
            # "installed", "failed" or None for a task that declares none. Kept
            # apart from the adapter's so a task's missing library is never read
            # as the agent's runtime having failed to install.
            "task_install": raw.get("task_install"),
            "error": raw.get("error"),
        }
        out.update({k: raw.get(k) for k in self.EXTRA_RECORD_KEYS})
        install = out.get("install")
        if isinstance(install, dict):
            self.observe_version(install.get("version"))
        return out

    # -- the live turn log --------------------------------------------------
    def readable(self, rec: dict) -> str:
        """ONE streamed record as a block for the live turns.log.

        Concrete, and deliberately routed through the SAME converter and the
        SAME renderer as the judge's feed, so the thing a human watches during a
        run cannot drift from the thing the measurement is taken on.

        It is a TAIL: it sees one record at a time with no memory of the records
        before it, so tool-call numbering restarts and a tool result whose call
        arrived in an earlier record renders as `RESULT[tool]`. That is fine
        here and would not be fine in the judge's feed, which is why the two
        callers differ in envelope only (`render_steps` vs `render_trajectory`).
        """
        try:
            traj = self.to_trajectory({"messages": [rec]})
        except Exception:  # noqa: BLE001 - a live log must never kill an episode
            return ""
        # A record that carries no step of its own -- a session header, a usage
        # event -- converts to the empty-episode placeholder. That placeholder
        # is right for a whole trajectory and wrong for a tail, where it would
        # claim nothing was captured on every such record.
        if len(traj.steps) == 1 and (traj.steps[0].extra or {}).get("placeholder"):
            return ""
        return "\n".join(render.render_steps(traj))

    # -- ATIF scaffolding shared by every adapter ---------------------------
    def atif_agent(self, *, model_name: str | None = None,
                   extra: dict | None = None) -> ATIFAgent:
        """The ATIF `agent` block for this adapter.

        `version` is required by ATIF and must be a string, so an unresolved
        version becomes the literal "unknown" -- which is what Harbor's own
        converters do, and is honest in a way that omitting the field is not.
        """
        return ATIFAgent(
            name=self.name(),
            version=self.version() or "unknown",
            model_name=model_name or self.model,
            tool_definitions=None,
            extra=extra or None,
        )

    @staticmethod
    def empty_step(note: str) -> Step:
        """A single placeholder step.

        ATIF requires `steps` to be non-empty (`min_length=1`). An episode that
        died before the agent emitted anything -- a failed install, a container
        killed on timeout -- still has to produce a VALID trajectory, because
        the alternative is that the worst failures are the ones that cannot be
        recorded or judged. The note says plainly that nothing was captured, so
        a reader cannot mistake it for an agent that chose to do nothing.
        """
        # `extra` marks it as scaffolding rather than something the agent did,
        # so the live tail can skip it (see `readable`) and a reader of a saved
        # trajectory can tell a placeholder from a real system message.
        return Step(step_id=1, source="system", message=note,
                    extra={"placeholder": True})

    # -- this adapter's runtime ------------------------------------------
    def install(self, sandbox=None) -> dict | None:
        """The install step this adapter needs run inside the episode, if any.

        None means "nothing to install", which is the whole of the CONTRACT:
        `core/trial.py` calls this while building the blob and carries whatever
        comes back into the container. An agent that is a CLI we must put into
        the container overrides this in `InstalledAgent`, which is where the
        check/install/version machinery lives.
        """
        return None

    # -- why an episode failed ---------------------------------------------
    #
    # The patterns a failed episode's text is classified with. Harbor's list,
    # ported in `core/agents/errors.py`. It is a class attribute so a vendor
    # whose CLI words something its own way can prepend to it:
    #
    #     ERROR_PATTERNS = [ErrorPattern(r"...", ApiRateLimitError),
    #                       *errors.ERROR_PATTERNS]
    #
    # DELIBERATE DIVERGENCE FROM HARBOR, which hangs this off
    # `BaseInstalledAgent` because it classifies a command IT exec'd. We have no
    # exec channel: a failure arrives as text on the one record the episode
    # returns, which every adapter returns whether it installs anything or not,
    # so it belongs to the contract.
    ERROR_PATTERNS = errors.ERROR_PATTERNS

    def classify_failure(self, current, *texts) -> type[errors.AgentError] | None:
        """Sharpen a failure with what the agent actually printed.

        `current` is what the harness already knows -- None, or the class for
        the point the episode died at. It is only overwritten when it is a
        GENERIC bucket: a definitive harness reason (the container never
        launched, the setup hook failed, the wall clock ran out) is the cause,
        and the text that follows it is a consequence.

        Called ONLY for an episode that has already failed. A successful
        episode's output is never scanned, so no amount of pattern drift can
        relabel one.
        """
        if current not in (None, errors.AgentError, errors.HarnessError,
                           errors.ContainerExitError, errors.NoRecordError):
            return current
        return (errors.classify(*texts, patterns=self.ERROR_PATTERNS)
                or current or errors.AgentError)

    # -- what this adapter was actually configured with --------------------
    def resolved_generation_config(self) -> dict:
        """The generation settings this adapter will ACTUALLY apply.

        Not a re-read of the yaml: an adapter that supplies its own default for
        a knob nobody set (codex's `reasoning_effort`) includes it here, because
        run.json records this and a run record that omitted an applied default
        would be describing a run that did not happen.
        """
        return dict(self.generation_config)

    # -- per-model routing, as this adapter will actually apply it ----------
    @property
    def api_key_env(self) -> str:
        """The HOST variable the credential is read from (entry, or default)."""
        return self.API_KEY_ENV

    def container_key_envs(self) -> tuple:
        """The variable NAMES the key is exported under inside the container.

        `CLI_KEY_ENV` when the adapter declares what its CLI reads, else the
        host name itself. `core/trial.py` hands exactly these to the sandbox;
        nothing else about the credential ever enters the container.
        """
        return tuple(self.CLI_KEY_ENV) or (self.api_key_env,)

    def resolved_base_url(self) -> str:
        """The base URL this adapter will actually use, "" for provider default.

        The entry's `api_base_url:` first; failing that, the first set host
        variable in CLI_BASE_URL_ENV, which is the PASSTHROUGH_ENV fallback
        every adapter had before entries could route themselves.
        """
        if self.api_base_url:
            return self.api_base_url
        for name in self.CLI_BASE_URL_ENV:
            if os.environ.get(name):
                return os.environ[name]
        return ""

    def base_url_source(self) -> str:
        if self.api_base_url:
            return "entry"
        for name in self.CLI_BASE_URL_ENV:
            if os.environ.get(name):
                return f"env:{name}"
        return "provider-default"

    def routed_env(self, env: dict | None) -> dict:
        """The container environment with this model's base URL applied.

        Called by `core/trial.py` on the PASSTHROUGH_ENV copy before `blob()`.
        An entry's `api_base_url:` overrides whatever the host exported under
        the same names; an entry without one leaves the copy alone.
        """
        out = dict(env or {})
        if self.api_base_url:
            for name in self.CLI_BASE_URL_ENV:
                out[name] = self.api_base_url
        return out

    def routing_payload(self) -> dict:
        """What the in-container runner needs to route the CLI. Rides on stdin.

        `cli_key_env` is where the key sits in the container's environment,
        `api_base_url` the resolved upstream ("" for the vendor default), and
        `extra_body` the dict the forwarder merges -- an empty dict means "no
        forwarder": the CLI talks to `api_base_url` directly.
        """
        return {"cli_key_env": list(self.container_key_envs()),
                "api_base_url": self.resolved_base_url(),
                "extra_body": dict(self.extra_body)}

    @staticmethod
    def forwarder_source() -> str:
        """`core/agents/forwarder.py` as SOURCE, for the blob's `modules`.

        Streamed in like the runner itself, so the forwarder never touches the
        container filesystem. Adapters ship it only when `extra_body` is set.
        """
        return (Path(__file__).resolve().parent / "forwarder.py").read_text()

    def resolved_routing(self) -> dict:
        """This model's routing as RESOLVED, for run.json and every record.

        Host-only URL (no userinfo, no query), the credential's variable NAME
        on both sides of the container boundary and where each came from,
        and the extra_body verbatim. A finished run must be able to say which
        gateway and which key variable produced it.
        """
        from .. import routing

        base = self.resolved_base_url()
        return {
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_env_source": self.api_key_env_source,
            "container_key_env": list(self.container_key_envs()),
            "api_base_url": routing.sanitise_url(base),
            "api_base_host": routing.url_host(base),
            "api_base_url_source": self.base_url_source(),
            "extra_body": dict(self.extra_body) or None,
            "forwarder": bool(self.extra_body),
        }

    def routing_problems(self) -> list[str]:
        """Routing this adapter cannot deliver. Extended per adapter."""
        problems = []
        if self.extra_body and not self.SUPPORTS_FORWARDER:
            problems.append(
                f"model {self.model!r}: extra_body is set but the "
                f"{self.name()!r} adapter does not route its CLI through the "
                f"forwarder, so the body fields would never reach the "
                f"provider. Remove extra_body, or teach the adapter.")
        return problems

    # Whether this adapter's runner routes the CLI through
    # core/agents/forwarder.py when `extra_body` is set. False is the safe
    # default: an adapter that has not implemented the opt-in refuses an
    # entry with extra_body at preflight rather than dropping it.
    SUPPORTS_FORWARDER: bool = False

    def config_problems(self) -> list[str]:
        """Generation settings this adapter cannot deliver.

        Concrete and shared: every adapter owes the same guarantee, and the
        alternative is each one growing its own half of it -- codex had exactly
        this check and Claude had none, so a Claude knob that reached nothing
        failed at nothing and was only found by measuring an episode.
        """
        unknown = sorted(set(self.generation_config) - set(self.GENERATION_KEYS))
        problems = self.routing_problems()
        if self.permission_mode not in self.PERMISSION_MODES:
            problems.append(f"{self.name()}: permission mode {self.permission_mode!r} is unsupported; "
                            f"supported modes: {self.PERMISSION_MODES}")
        if not unknown:
            return problems
        return [*problems,
                f"model {self.model!r}: generation_config keys {unknown} are "
                f"not translated by the {self.name()!r} adapter, which "
                f"translates {list(self.GENERATION_KEYS)}. A silently dropped "
                f"generation setting is a run whose config file lies about it: "
                f"remove the key, or teach this adapter to deliver it."]

    # -- optional preflight ----------------------------------------------
    def setup(self) -> list[str]:
        """Problems that would stop this adapter running; empty means good."""
        return list(self.config_problems())
