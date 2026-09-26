"""Why an episode failed, as a TYPE rather than a sentence.

Ported from Harbor's `harbor/src/harbor/agents/installed/base.py` (Apache 2.0):
the `ApiError` family and the `ErrorPattern` mechanism that matches one out of
what the agent printed. Derived work; see core/trajectory/__init__.py for the
full notice.

Why this exists at all. Until now every way an episode could fail collapsed into
ONE free-text `error` string on the record, so

    "prompt is too long: 214431 tokens > 200000 maximum"

and

    "Claude Code is unable to respond to this request, which appears to
     violate our Usage Policy"

were the same data point: `ok: false`, some string. They are not the same data
point. The first is an episode the harness lost -- the model never got to decide
anything, and the run should be repeated or the budget raised. The second is the
model DECLINING, which is a finding about the model and is exactly the kind of
thing this eval exists to measure. Averaging them together, or quietly dropping
both as "errors", is how a refusal rate gets read as a context bug and a context
bug gets read as a refusal. We have been burned by precisely that confusion.

So a failed episode records a `failure` alongside `error`: a stable snake_case
tag from the classes below, which is what downstream groups by, while `error`
keeps the raw text nothing should ever have to parse a second time.

Two deliberate divergences from Harbor:

  * Harbor's root class is `NonZeroAgentExitCodeError`, named for the mechanism
    it is raised by -- it holds a live environment and raises when a command it
    exec'd exits non-zero. We have no exec channel: an episode is one
    `enroot start` that hands back one JSON line, so our failures arrive as
    TEXT in a record and there is no exit code to name a base class after.
    `AgentError` is the same taxonomy with the mechanism taken out of the name.
  * Harbor RAISES these, and its retry policy catches them
    (`--retry-include ApiRateLimitError`). We CLASSIFY with them: the class is
    used as a label on a record, not thrown. Both uses want exactly the same
    hierarchy, so the hierarchy is Harbor's; only the verb differs.

The harness-side reasons at the bottom (`HarnessError` and below) are ours and
have no Harbor equivalent, because Harbor's orchestrator does not stream a
runner into a container over stdin. They are in the same tree so that a record's
`failure` has ONE vocabulary and a reader never has to ask which of two schemes
a tag came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class AgentError(RuntimeError):
    """Any reason an episode did not produce a usable result.

    `reason` is the tag that goes on the record. It is snake_case, stable, and
    the thing analysis groups by; subclasses override it and nothing else.

    `transient` is one axis a subclass may set: True means the failure is a
    passing fault of the transport or the gateway, not a decision of the model
    or a cap we set, so re-running the SAME episode is expected to succeed. It
    is what the judge retries on. Deterministic failures -- a refusal, a context
    overflow, an output-token cap, a bad login -- leave it False, because
    retrying them just spends money to fail identically.

    `lost_episode` is the other, and it answers a DIFFERENT question: is this a
    fact about the model, or an episode we did not get? Every transient class is
    lost -- a dropped stream is nobody's finding -- but the two are not the same
    axis, and `ApiUsageLimitError` is the case that proves it. A key at its
    budget cap is NOT transient: the gateway answers `400 ... Budget has been
    exceeded` to every call until a human raises the cap, so an immediate retry
    fails identically. It is still an episode the harness lost. The transcript
    behind it is a truncated attempt that says nothing about the model, and
    judging it puts a verdict on work that was cut off.

    So: `lost_episode` is what `is_judgeable` refuses and what the run-level
    resume re-runs; `transient` is what an immediate retry is expected to fix.
    A FINDING -- a refusal, a context overflow, a `max_turns` cap, a wall-clock
    timeout -- leaves both False: nothing re-runs it, its transcript is the last
    word on the episode, and the judge should see it.
    """

    reason = "unclassified"
    transient = False
    lost_episode = False


# ---------------------------------------------------------------------------
# the model provider said no -- Harbor's `ApiError` family, names unchanged
class ApiError(AgentError):
    """The model provider rejected or dropped the request."""

    reason = "api_error"


class ApiRateLimitError(ApiError):
    """The provider rate-limited the request. Retrying later is the fix."""

    reason = "rate_limit"
    transient = True
    lost_episode = True


class ApiUsageLimitError(ApiError):
    """An account or project usage limit is exhausted. Money, not the model.

    NOT transient and still LOST, which is the whole reason `lost_episode`
    exists as a separate axis. Our litellm gateway keys carry a dollar budget,
    and a key that has spent it answers every request with
    `400 ... Budget has been exceeded ...` -- so an immediate retry fails
    identically (not transient), while the episode itself was cut off mid-run
    by our accounting and not by anything the model did (lost). Judging the
    truncated transcript would file our own spending cap as a fact about the
    model; the fix is to raise the cap and re-run, which is exactly what
    `load_done` in run.py already does with an `ok: false` record.
    """

    reason = "usage_limit"
    lost_episode = True


class ApiInternalServerError(ApiError):
    """The provider returned a 500."""

    reason = "api_internal_server"
    transient = True
    lost_episode = True


class ApiOverloadedError(ApiError):
    """The provider is temporarily overloaded or at capacity."""

    reason = "overloaded"
    transient = True
    lost_episode = True


class ProxyGatewayError(ApiError):
    """A gateway in front of the provider returned a 5xx, not the model.

    THE reason this class exists. The litellm proxy at litellm.safe.ai sits
    behind Cloudflare, and Cloudflare intermittently answers a request with its
    own 502/503/504 page -- a `{"type": ".../cloudflare-5xx-errors/error-502/"}`
    body, or a bare "Bad gateway" -- while the model behind it is fine. The CLI
    surfaces that as `API Error: 502 {...}` on the record, which used to fall
    through to `UnknownApiError`: a passing gateway hiccup filed next to genuine,
    unexplained provider errors, so a run that lost ten episodes to a flaky
    proxy read as ten mysteries instead of one retryable fault.

    It is TRANSIENT: the same episode re-run a second later gets a real answer.
    Kept distinct from `ApiConnectionClosedError` (a stream that dropped
    mid-response) and from `ApiOverloadedError` (the MODEL at capacity, a 529)
    because the fix is the same -- retry -- but the cause named on the record is
    a different fact.
    """

    reason = "proxy_5xx"
    transient = True
    lost_episode = True


class ApiConnectionClosedError(ApiError):
    """The provider closed the connection before the response completed."""

    reason = "connection_closed"
    transient = True
    lost_episode = True


class ApiResponseStalledError(ApiError):
    """The response stalled mid-stream before completing."""

    reason = "response_stalled"
    transient = True
    lost_episode = True


class OutputTokenExceededError(ApiError):
    """The response hit the configured output token maximum."""

    reason = "output_token_exceeded"


class ContextWindowExceededError(ApiError):
    """The request exceeded the model's context window.

    THE reason this module exists. An episode that ran out of context did not
    fail at the task -- it was cut off mid-attempt, and counting it as a failed
    attempt understates the model exactly as much as dropping it overstates it.
    """

    reason = "context_window_exceeded"


class AgentSafetyRefusalError(ApiError):
    """The provider blocked the request on safety grounds.

    A deterministic, request-level decision that will not succeed on retry --
    and, unlike everything else in this file, a FINDING rather than a fault.
    Kept distinct from `UnknownApiError` for that reason: a real refusal read as
    a flaky API error is a measurement thrown away.

    THE BOUNDARY, because it is easy to misread: this is the PROVIDER hard-
    stopping the request, matched out of the error text of an episode that
    already failed. It says nothing about whether the model's ANSWER declined,
    hedged or cheated -- that is behaviour, it is the judge's job, and it is
    never decided by a regex here. A successful episode is never even scanned.
    """

    reason = "safety_refusal"


class RequestTooLargeError(ApiError):
    """Serialized request exceeded the transport body limit."""

    reason = "request_too_large"
    lost_episode = True


class InvalidApiInputError(ApiError):
    """Provider rejects the request schema or unsupported image input."""

    reason = "invalid_api_input"
    lost_episode = True


class UnknownApiError(ApiError):
    """The output says "API error" and nothing more specific matched."""

    reason = "unknown_api_error"


# ---------------------------------------------------------------------------
# the request never reached a model -- Harbor keeps these outside ApiError
class AgentAuthenticationError(AgentError):
    """The agent CLI reports no login: usually a missing or wrong API key."""

    reason = "auth"


class ModelNotFoundError(AgentError):
    """The requested model is unknown or unavailable to this account."""

    reason = "model_not_found"


class NetworkConnectionError(AgentError):
    """DNS, connection refused, TLS handshake, curl: a transport failure."""

    reason = "network"
    transient = True
    lost_episode = True


# ---------------------------------------------------------------------------
# ours: the episode broke on this side of the container boundary
class HarnessError(AgentError):
    """The harness lost the episode. Says nothing about the model."""

    reason = "harness"


class ContainerLaunchError(HarnessError):
    """`enroot start` never started."""

    reason = "container_launch"


class SetupHookError(HarnessError):
    """The task's setup hook failed, so the episode never had its environment."""

    reason = "setup_hook"


class AgentInstallError(HarnessError):
    """`check || install` failed inside the container: no agent ever ran."""

    reason = "install_failed"


class EpisodeTimeoutError(HarnessError):
    """The episode hit its wall clock and the container was killed."""

    reason = "timeout"


class MaxTurnsError(HarnessError):
    """The agent hit `--max-turns` and the CLI
    stopped it. A cap we set, so a harness reason, not an API one: the episode
    was cut off, not broken."""
    reason = "max_turns"


class ContainerExitError(HarnessError):
    """The container exited non-zero without leaving a record."""

    reason = "container_exit"


class NoRecordError(HarnessError):
    """The container produced no record on stdout."""

    reason = "no_record"


class UnparseableRecordError(HarnessError):
    """The container's stdout was not the one JSON line the contract promises."""

    reason = "unparseable_record"


class GraderUnreachableError(HarnessError):
    """The grader was invoked but served no review during the episode."""

    reason = "grader_unreachable"
    lost_episode = True


class ReviewerInfrastructureError(HarnessError):
    """An unexpected reviewer failure invalidated the episode."""

    reason = "reviewer_infrastructure"
    lost_episode = True


class BatchCancelledError(HarnessError):
    """An in-flight episode stopped when its batch was aborted."""

    reason = "batch_cancelled"
    lost_episode = True


# ---------------------------------------------------------------------------
# matching one out of what the agent printed


@dataclass(frozen=True)
class ErrorPattern:
    """Harbor's `ErrorPattern`, unchanged.

    A regex that classifies failed output into a specific error. Searched
    case-insensitively over the whole text; the match furthest toward the END
    wins, because a CLI reports its actual failure last and its boilerplate
    first.
    """

    pattern: str
    exception: type[AgentError]


# The wording every one of our runners opens the record's `error` with when IT
# killed the CLI at the task's `timeout_s`. Named once because it is read two
# ways: as one pattern among many below, and ANCHORED, as the harness's own
# verdict that outranks any text (see `is_harness_timeout`).
TIMEOUT_WORDING = r"timeout after \d+s"

# Harbor's `BaseInstalledAgent.ERROR_PATTERNS`, ported with its ordering and its
# comments, plus the four at the end for our own runners' wording. Adapters may
# extend this; nothing here is vendor-specific enough to need it yet.
ERROR_PATTERNS: list[ErrorPattern] = [
    ErrorPattern(r"No endpoints found that support image input|does not support (?:image|vision)|image input.*not supported|Failed to deserialize the JSON body|unknown variant .?image_url", InvalidApiInputError),
    ErrorPattern(r"request_too_large|Request exceeds the maximum size", RequestTooLargeError),
    ErrorPattern(r"rate.?limit", ApiRateLimitError),
    ErrorPattern(r"too many requests", ApiRateLimitError),
    ErrorPattern(r"specified API usage limits", ApiUsageLimitError),
    ErrorPattern(r"You've hit your usage limit", ApiUsageLimitError),
    ErrorPattern(r"You have an unpaid invoice", ApiUsageLimitError),
    ErrorPattern(r"Quota exceeded.", ApiUsageLimitError),
    # Our litellm gateway's wording when a virtual key has spent its dollar
    # budget: `400 ... ExceededBudget: Budget has been exceeded! Current cost:
    # ..., Max budget: ...`. Ahead of the generic "API Error" catch-all, which
    # used to swallow it as `unknown_api_error` -- an unexplained provider fault
    # rather than our own spending cap, which is the difference between "the
    # model did something strange" and "top the key up and re-run".
    ErrorPattern(r"Budget has been exceeded", ApiUsageLimitError),
    ErrorPattern(r"API Error: 500 Internal server error", ApiInternalServerError),
    ErrorPattern(r"RetriableError: \[internal\] Error", ApiInternalServerError),
    ErrorPattern(r"API Error: Overloaded", ApiOverloadedError),
    ErrorPattern(r"ServiceUnavailableError", ApiOverloadedError),
    ErrorPattern(r"Selected model is at capacity\. Please try a different model\.",
                 ApiOverloadedError),
    ErrorPattern(r"(?:litellm\.)?InternalServerError:.*Server disconnected",
                 ApiConnectionClosedError),
    # The DeepSeek gateway reports http.client truncated responses this way.
    ErrorPattern(r"gateway failed: IncompleteRead:", ApiConnectionClosedError),
    ErrorPattern(r"API Error: Connection closed mid-response",
                 ApiConnectionClosedError),
    # Newer CLI wording of the same disconnect, and its server-side sibling.
    ErrorPattern(r"API Error: Connection lost mid-response",
                 ApiConnectionClosedError),
    ErrorPattern(r"API Error: Server error mid-response", ApiInternalServerError),
    # The CLI's retries-exhausted wording for a 529.
    ErrorPattern(r"Repeated 529 Overloaded errors", ApiOverloadedError),
    # Our own turn cap, reported by the CLI as an error result.
    ErrorPattern(r"Reached maximum number of turns", MaxTurnsError),
    ErrorPattern(r"^max_turns reached before confirmed task completion$", MaxTurnsError),
    ErrorPattern(r"Maximum session turns exceeded|MAX_TURNS_EXCEEDED", MaxTurnsError),
    # OpenRouter-style phrasing of the same mid-stream disconnect.
    ErrorPattern(r"API Error: stream closed before completion",
                 ApiConnectionClosedError),
    ErrorPattern(r"API Error: Response stalled mid-stream", ApiResponseStalledError),
    ErrorPattern(r"response exceeded .+ output token maximum",
                 OutputTokenExceededError),
    ErrorPattern(r"input token count exceeds the maximum number of tokens|"
                 r"prompt is too long: \d+ tokens > \d+ maximum",
                 ContextWindowExceededError),
    ErrorPattern(r"Not logged in", AgentAuthenticationError),
    ErrorPattern(r"Cannot use this model", ModelNotFoundError),
    # A Cloudflare/gateway 5xx in front of the litellm proxy, not the model.
    # Must precede the generic "API Error" catch-all so a retryable gateway
    # hiccup is not filed as an unexplained provider error. The needles are the
    # shapes the CLI actually surfaces: `API Error: 502 {...}`, the Cloudflare
    # body's own URL, its `"cloudflare_error": true` marker, and the bare gateway
    # phrases. 529 is deliberately NOT here -- that is the MODEL overloaded
    # (ApiOverloadedError), a different fact with the same retry.
    ErrorPattern(r"cloudflare-5xx-errors|"
                 r"/error-50[234]/|"
                 r'"cloudflare_error"\s*:\s*true|'
                 r"API Error: 50[234]\b|"
                 r"HTTP 50[234]\b|"
                 r"\b50[234] (?:Bad Gateway|Gateway Time-?out)|"
                 # Cloudflare's OWN 52x family, which it returns when the
                 # origin -- litellm.safe.ai -- does not answer in time or drops
                 # the connection: 520 unknown, 521 down, 522 connect timeout,
                 # 523 unreachable, 524 "a timeout occurred", 525/526 TLS, 527.
                 # Observed as a hard 524 with an HTML body on a streaming chat
                 # completion after ~125s, which Kimi Code surfaces verbatim as
                 # `[provider.api_error] 524 status code (no body)` and treats as
                 # NON-retryable, aborting the episode. Same fact and same fix as
                 # the 50x above: transient, re-run the episode. 529 is
                 # deliberately excluded -- that is the MODEL overloaded
                 # (ApiOverloadedError), not a gateway.
                 r"API Error: 52[0-7]\b|"
                 r"HTTP 52[0-7]\b|"
                 r"\b52[0-7] status code\b|"
                 r"Bad Gateway|"
                 r"Gateway Time-?out",
                 ProxyGatewayError),
    # Must precede the generic "API Error" catch-all below.
    # High-precision safety hard-stop needles only. Do NOT match:
    # - bare "Request blocked" (infra/provider noise)
    # - soft Claude "model_refusal_fallback" retries that continue the run
    #   (skipped in `classify` when there is no hard-stop subtype)
    # - bare provider "400" / stream errors without refusal language
    ErrorPattern(r"safety measures that flagged|Cyber Verification Program|"
                 r"flagged for possible cybersecurity risk|"
                 # Codex Trusted Access / cybersecurity program hard-stop.
                 r"Trusted Access for Cyber|chatgpt\.com/cyber|"
                 r"Output blocked by content filtering policy|"
                 # Anthropic AUP / cyber API refusal hard-stops.
                 r"violate our Usage Policy|"
                 r"https://www\.anthropic\.com/legal/aup|"
                 r"triggered cyber-related safeguards|"
                 r"model_refusal_no_fallback|"
                 r'ContentFilterError|"reason"\s*:\s*"content-filter"',
                 AgentSafetyRefusalError),
    ErrorPattern(r"API Error", UnknownApiError),
    ErrorPattern(r"SSL_ERROR_SYSCALL", NetworkConnectionError),
    ErrorPattern(r"SSL_connect", NetworkConnectionError),
    ErrorPattern(r"Could not resolve host", NetworkConnectionError),
    ErrorPattern(r"Connection refused", NetworkConnectionError),
    ErrorPattern(r"Connection timed out", NetworkConnectionError),
    ErrorPattern(r"Request timed out", NetworkConnectionError),
    ErrorPattern(r"curl: \(\d+\)", NetworkConnectionError),
    # Ours: the wording our own runners use, so a failure they detected is not
    # left `unclassified` just because no vendor phrased it.
    ErrorPattern(TIMEOUT_WORDING, EpisodeTimeoutError),
    ErrorPattern(r"container exited -?\d+", ContainerExitError),
    ErrorPattern(r"container produced no record on stdout", NoRecordError),
    ErrorPattern(r"unparseable container stdout", UnparseableRecordError),
]

_COMPILED = [(re.compile(p.pattern, re.IGNORECASE), p.exception)
             for p in ERROR_PATTERNS]


def classify(*texts: str | None,
             patterns: list[ErrorPattern] | None = None) -> type[AgentError] | None:
    """The error `texts` describe, or None when nothing matched.

    Harbor's `_classify_exec_error`, minus the raising: last match wins, ties go
    to the earlier pattern, and a soft Claude divert never counts as a refusal.
    """
    blob = "\n".join(t for t in texts if t)
    if not blob.strip():
        return None
    compiled = ([(re.compile(p.pattern, re.IGNORECASE), p.exception)
                 for p in patterns] if patterns is not None else _COMPILED)
    # Harbor's note: a soft `model_refusal_fallback` divert CONTINUES the run,
    # and explanatory prose can quote the same policy language a hard stop uses.
    # Never call it a refusal on free-text needles alone.
    skip_refusal = ("model_refusal_fallback" in blob
                    and "model_refusal_no_fallback" not in blob)
    best: tuple[int, type[AgentError]] | None = None
    for pattern, exception in compiled:
        if skip_refusal and exception is AgentSafetyRefusalError:
            continue
        for match in pattern.finditer(blob):
            if best is None or match.end() > best[0]:
                best = (match.end(), exception)
    return best[1] if best else None


# The same wording anchored to the START of the record's error: our runners
# assign it BEFORE anything a dying CLI reports, so an error that opens with it
# is the harness saying it pulled the plug itself.
_HARNESS_TIMEOUT = re.compile(rf"^\s*{TIMEOUT_WORDING}", re.IGNORECASE)


def is_harness_timeout(error: str | None) -> bool:
    """Did the harness itself kill this episode at its wall clock?

    A deterministic FINDING about the episode -- the agent ran out of time --
    and not a fault of the transport, so it is never transient however the
    session ended. Every runner sets this wording first and only falls through
    to the CLI's own terminal error when it did not fire, so an error that
    OPENS with it means the harness killed a still-running CLI; whatever that
    CLI printed on its way down, including a gateway 5xx it survived half an
    hour earlier, is fallout rather than the cause.

    Anchored on purpose: a CLI that merely mentions a timeout somewhere in its
    output, or a runner error that names its own failure first, is not this.
    """
    return bool(error and _HARNESS_TIMEOUT.match(error))


def reason(error: type[AgentError] | None) -> str | None:
    """The tag that goes on the record. None stays None: no failure, no tag."""
    return error.reason if error is not None else None


# ---------------------------------------------------------------------------
# which reasons are worth re-running the same episode for


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


# The tags of every class that set `transient = True`, computed from the tree so
# it can never disagree with the classes: adding a transient subclass adds its
# tag here for free, and there is no second list to forget to update. This is
# the vocabulary the run-level auto-resume and the judge retry both select on.
TRANSIENT_REASONS: frozenset[str] = frozenset(
    c.reason for c in _all_subclasses(AgentError) if c.transient)


def is_transient(value: type[AgentError] | str | None) -> bool:
    """Is this failure a passing transport/gateway fault worth retrying?

    Accepts either an error CLASS or the snake_case `reason` tag a record
    carries, so callers holding a record's `failure` string and callers holding
    a classified class both ask the same question. None -- no failure, or an
    unrecognised tag -- is never transient.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value in TRANSIENT_REASONS
    return bool(getattr(value, "transient", False))


# The tags of every class the harness LOST the episode to: computed from the
# tree the same way, and a superset of TRANSIENT_REASONS by construction, since
# every transient class also sets `lost_episode`. The union is taken rather than
# read off the flag alone so a future transient class that forgets the flag is
# still counted as lost -- a dropped stream is never a finding.
LOST_REASONS: frozenset[str] = TRANSIENT_REASONS | frozenset(
    c.reason for c in _all_subclasses(AgentError) if c.lost_episode)


def is_lost(value: type[AgentError] | str | None) -> bool:
    """Did the harness lose this episode, rather than measure something?

    The question `is_judgeable` and the run-level resume both ask. True means
    the transcript is a truncated attempt that says nothing about the model and
    the episode is going to be re-run; False means the record is the last word
    on the episode, whether it succeeded or failed.

    Strictly weaker than `is_transient`: every transient fault is lost, plus
    `usage_limit`, which no immediate retry can fix but which is still our
    spending cap cutting an episode short rather than a fact about the model.
    Same two accepted shapes as `is_transient`, same None rule.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value in LOST_REASONS
    return bool(getattr(value, "transient", False)
                or getattr(value, "lost_episode", False))
