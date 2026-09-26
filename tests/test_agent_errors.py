"""The failure taxonomy: a lost episode and a refused one are different data.

The one thing these tests exist to protect is the distinction in
`test_context_exhaustion_is_not_a_refusal`. Everything else here guards the
machinery that has to keep working for that distinction to survive: the
classification rule, the vocabulary, and above all the rule that a SUCCESSFUL
episode is never classified at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import trial  # noqa: E402
from core.agents import errors  # noqa: E402
from core.agents.errors import ErrorPattern  # noqa: E402
from core.agents.claude_sdk import ClaudeSDKAgent  # noqa: E402
from core.agents import AgentFactory, make_agent  # noqa: E402
from core import judge  # noqa: E402


def _agent() -> ClaudeSDKAgent:
    agent = ClaudeSDKAgent(model="claude-opus-5")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    return agent


# --------------------------------------------------------------------------
# the distinction this whole file is for
CONTEXT = ("Error: prompt is too long: 214431 tokens > 200000 maximum")
REFUSAL = ("API Error: Claude Code is unable to respond to this request, "
           "which appears to violate our Usage Policy "
           "(https://www.anthropic.com/legal/aup).")


def test_context_exhaustion_is_not_a_refusal() -> None:
    """The two failures this taxonomy exists to keep apart.

    Both used to be `ok: false` plus a string. One is an episode the harness
    lost with the model mid-attempt; the other is the model DECLINING, which is
    a finding. Averaging them together is how a refusal rate gets read as a
    context bug.
    """
    assert errors.classify(CONTEXT) is errors.ContextWindowExceededError
    assert errors.classify(REFUSAL) is errors.AgentSafetyRefusalError
    assert errors.reason(errors.classify(CONTEXT)) == "context_window_exceeded"
    assert errors.reason(errors.classify(REFUSAL)) == "safety_refusal"


def test_a_refusal_is_not_an_unknown_api_error() -> None:
    """The refusal text also contains the `API Error` catch-all needle.

    Harbor orders the patterns so the specific one wins; if that ordering were
    lost, every refusal would be recorded as flaky provider noise -- which is
    the measurement being thrown away, not a cosmetic mislabel.
    """
    assert errors.classify(REFUSAL) is not errors.UnknownApiError


def test_a_soft_divert_is_never_a_refusal() -> None:
    """`model_refusal_fallback` CONTINUES the run; only the no-fallback form
    is a hard stop. Harbor's rule, kept because prose can quote the policy."""
    soft = f'{{"subtype":"model_refusal_fallback"}} {REFUSAL}'
    assert errors.classify(soft) is not errors.AgentSafetyRefusalError
    hard = f'{{"subtype":"model_refusal_no_fallback"}} {REFUSAL}'
    assert errors.classify(hard) is errors.AgentSafetyRefusalError


# --------------------------------------------------------------------------
# the classification rule
@pytest.mark.parametrize("text,reason", [
    ("API Error: 429 rate limit exceeded", "rate_limit"),
    ("You've hit your usage limit", "usage_limit"),
    ("API Error: Overloaded", "overloaded"),
    ("response exceeded 64000 output token maximum", "output_token_exceeded"),
    ("Not logged in", "auth"),
    ("Cannot use this model", "model_not_found"),
    ("curl: (60) SSL certificate problem", "network"),
    # our own runners' wording
    ("timeout after 900s", "timeout"),
    ("container exited 137", "container_exit"),
    ("container produced no record on stdout", "no_record"),
    ("unparseable container stdout: Expecting value", "unparseable_record"),
])
def test_representative_failures(text: str, reason: str) -> None:
    assert errors.reason(errors.classify(text)) == reason


def test_the_last_match_wins() -> None:
    """A CLI prints its boilerplate first and the thing that killed it last."""
    text = f"API Error: 429 rate limit\n...retrying...\n{CONTEXT}"
    assert errors.classify(text) is errors.ContextWindowExceededError


def test_nothing_matched_is_none_not_a_guess() -> None:
    assert errors.classify("") is None
    assert errors.classify(None) is None
    assert errors.classify("the agent wrote a file and stopped") is None
    assert errors.reason(None) is None


def test_every_reason_tag_is_unique() -> None:
    """The tag is what analysis groups by, so two classes sharing one would
    silently merge two populations."""
    seen: dict[str, str] = {}
    stack = [errors.AgentError]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        assert cls.reason not in seen or seen[cls.reason] == cls.__name__, (
            f"{cls.__name__} and {seen[cls.reason]} share the tag "
            f"{cls.reason!r}")
        seen[cls.reason] = cls.__name__


def test_an_adapter_can_extend_the_patterns() -> None:
    """A vendor that words something its own way overrides ERROR_PATTERNS."""

    class Picky(ClaudeSDKAgent):
        ERROR_PATTERNS = [ErrorPattern(r"the wombat is full",
                                       errors.ContextWindowExceededError)]

    agent = Picky(model="m")
    assert agent.classify_failure(None, "the wombat is full") is \
        errors.ContextWindowExceededError
    # ...and the base list is what everyone else still gets.
    assert _agent().classify_failure(None, "the wombat is full") is \
        errors.AgentError


# --------------------------------------------------------------------------
# how a failure gets onto the record
def test_a_definitive_harness_reason_is_not_overwritten() -> None:
    """A setup hook that failed with a connection error is a FAILED HOOK.

    The transport error inside it is the consequence; recording it as a network
    failure would say the episode ran and lost its network, which is a
    different episode.
    """
    agent = _agent()
    assert agent.classify_failure(
        errors.SetupHookError,
        "setup hook failed: OSError: Connection refused") is errors.SetupHookError


def test_a_generic_bucket_is_sharpened() -> None:
    """`container exited 1` says nothing; the text underneath it does."""
    agent = _agent()
    assert agent.classify_failure(errors.ContainerExitError, CONTEXT) is \
        errors.ContextWindowExceededError
    # ...and stays the generic bucket when the text adds nothing.
    assert agent.classify_failure(errors.ContainerExitError, "boom") is \
        errors.ContainerExitError


def test_an_unclassifiable_failure_is_labelled_as_such() -> None:
    agent = _agent()
    assert errors.reason(agent.classify_failure(None, "boom")) == "unclassified"


def test_a_successful_episode_is_never_classified(tmp_path: Path) -> None:
    """THE guard. `classify_failure` returns before it looks at anything, so
    no pattern here can ever change what a successful run reports -- even if
    the episode's stderr happens to contain the word "rate limit"."""
    err = tmp_path / "container.stderr.log"
    err.write_text(f"[runner] chatter\n{CONTEXT}\n")
    assert trial.classify_failure(_agent(), None, None,
                                  {"install": {"status": "installed"}},
                                  err) is None


def test_a_failed_install_is_the_root_cause(tmp_path: Path) -> None:
    """No agent ever ran, so whatever the SDK said afterwards is fallout."""
    err = tmp_path / "container.stderr.log"
    err.write_text("npm ERR! code E404\n")
    got = trial.classify_failure(
        _agent(), "CLINotFoundError: claude", None,
        {"install": {"status": "failed", "error": "npm ERR! code E404"}}, err)
    assert got is errors.AgentInstallError


def test_the_container_stderr_is_read_when_the_record_is_thin(
        tmp_path: Path) -> None:
    """A CLI puts the real cause on stderr and the record often carries only
    the exception that followed it."""
    err = tmp_path / "container.stderr.log"
    err.write_text(f"[runner] starting\n{REFUSAL}\n")
    got = trial.classify_failure(_agent(), "container exited 1",
                                 errors.ContainerExitError, {}, err)
    assert got is errors.AgentSafetyRefusalError


def test_a_missing_stderr_file_is_not_fatal(tmp_path: Path) -> None:
    got = trial.classify_failure(_agent(), CONTEXT, None, {},
                                 tmp_path / "does-not-exist.log")
    assert got is errors.ContextWindowExceededError


# --------------------------------------------------------------------------
# the wall clock outranks whatever the CLI printed on its way down
#
# openmath_grok-4.6_nopassword: `lonely_runner_ce#1` and `navier_stokes_ce#1`
# both ended `timeout after 3600s` and were filed `proxy_5xx`, from a gateway
# 524 that had landed EARLIER in the same hour. `proxy_5xx` is transient, so
# `is_judgeable` skipped them and the auto-resume would have re-run them: two
# real findings about the model dropped because of a fault it survived.
TIMED_OUT = "timeout after 3600s"
GATEWAY_STDERR = (
    "[runner] forwarder: POST /v1/chat/completions -> 524 id=None finish=None\n"
    "[kimi] retrying step 3\n"
    "error: failed to run prompt: provider.api_error: 524 status code (no body)\n"
    "[runner] forwarder: POST /v1/chat/completions -> 200 id=abc finish=stop\n")


@pytest.mark.parametrize("name", AgentFactory.names())
def test_a_wall_clock_kill_outranks_a_5xx_it_survived(
        name: str, tmp_path: Path) -> None:
    """Every adapter, because the wall clock is the harness's, not a vendor's.

    The 524 sits LATER in the blob than the timeout line does, so the
    last-match-wins rule picks it; the anchored check has to return first.
    """
    err = tmp_path / "container.stderr.log"
    err.write_text(GATEWAY_STDERR)
    got = trial.classify_failure(make_agent(name, model="m"), TIMED_OUT, None,
                                 {"final_text": "Error 524 Ray ID: 9c1f"}, err)
    assert got is errors.EpisodeTimeoutError
    assert errors.reason(got) == "timeout"


def test_a_timed_out_episode_is_a_finding_and_is_judged(tmp_path: Path) -> None:
    """Not transient, so nothing re-runs it and the judge is the last word."""
    err = tmp_path / "container.stderr.log"
    err.write_text(GATEWAY_STDERR)
    got = trial.classify_failure(_agent(), TIMED_OUT, None, {}, err)
    assert errors.is_transient(got) is False
    assert errors.reason(got) not in errors.TRANSIENT_REASONS
    assert judge.is_judgeable(
        {"transcript_path": str(tmp_path / "transcript.json"),
         "failure": errors.reason(got)}) is True


def test_a_5xx_that_ended_the_cli_itself_is_still_transient(
        tmp_path: Path) -> None:
    """The case this precedence must not break: the CLI died OF the gateway,
    the episode is a partial one, and the auto-resume re-runs it."""
    err = tmp_path / "container.stderr.log"
    err.write_text(GATEWAY_STDERR)
    got = trial.classify_failure(
        make_agent("kimi-code", model="m"),
        "kimi exited 1: [provider.api_error] 524 status code (no body)",
        None, {}, err)
    assert got is errors.ProxyGatewayError
    assert errors.is_transient(got) is True
    assert judge.is_judgeable(
        {"transcript_path": str(tmp_path / "transcript.json"),
         "failure": errors.reason(got)}) is False


def test_only_the_harness_wording_at_the_START_counts() -> None:
    """A CLI that merely mentions a timeout is not the harness killing it, and
    a runner error that names its own failure first is not either."""
    assert errors.is_harness_timeout("timeout after 3600s") is True
    assert errors.is_harness_timeout(
        "timeout after 3660s (container killed)") is True
    assert errors.is_harness_timeout(
        "kimi exited 1: read timeout after 30s on /v1/chat/completions") is False
    assert errors.is_harness_timeout("Request timed out") is False
    assert errors.is_harness_timeout(None) is False


# --------------------------------------------------------------------------
# the money family: not transient, still lost
#
# A litellm virtual key carries a dollar budget. Once it is spent the gateway
# answers EVERY request `400 ... Budget has been exceeded`, which used to fall
# through the pattern list to the generic "API Error" catch-all and be filed
# `unknown_api_error`: not transient, so nothing re-ran it -- and, worse, not
# transient meant `is_judgeable` said yes, so run.py judged the truncated
# transcript inline. Then the cap was raised, the run resubmitted, `load_done`
# re-ran the episode (it is `ok: false`), and the episode got a SECOND verdict.
# Two verdicts for one episode, both traceable to our own accounting.
BUDGET_ERROR = (
    "API Error: 400 {\"error\":{\"message\":\"litellm.BudgetExceededError: "
    "Budget has been exceeded! Current cost: 2000.0038, Max budget: 2000.0\","
    "\"type\":\"budget_exceeded\",\"code\":\"400\"}}")


def test_a_budget_cap_is_a_usage_limit_and_not_an_unknown_error() -> None:
    got = errors.classify(BUDGET_ERROR)
    assert got is errors.ApiUsageLimitError
    assert errors.reason(got) == "usage_limit"


def test_a_budget_cap_is_lost_but_not_transient() -> None:
    """The two axes, on the one case that separates them.

    Not transient: the very next call hits the same cap, so an immediate retry
    is not expected to succeed. Lost anyway: the episode was cut off by our
    spending, and its transcript says nothing about the model.
    """
    assert errors.is_transient(errors.ApiUsageLimitError) is False
    assert errors.is_transient("usage_limit") is False
    assert errors.is_lost(errors.ApiUsageLimitError) is True
    assert errors.is_lost("usage_limit") is True


def test_every_transient_failure_is_also_a_lost_one() -> None:
    """A dropped stream is never a finding, so `lost` must be the weaker test.

    Guards the flag against drift: a new transient class that forgets
    `lost_episode` would otherwise be re-run by run.py AND judged.
    """
    assert errors.TRANSIENT_REASONS <= errors.LOST_REASONS
    for tag in errors.TRANSIENT_REASONS:
        assert errors.is_lost(tag) is True, tag
    assert errors.LOST_REASONS - errors.TRANSIENT_REASONS == {
        "usage_limit", "grader_unreachable", "reviewer_infrastructure", "batch_cancelled", "request_too_large", "invalid_api_input"}
    assert errors.is_lost(None) is False
    assert errors.is_lost("no_such_tag") is False


def test_a_lost_episode_is_not_judged_and_a_finding_is(tmp_path: Path) -> None:
    """THE gate. Judged and re-run are complements, over every tag we have.

    A record is either the last word on the episode (judge it) or a truncated
    attempt about to be replaced (re-run it, do not judge it). Nothing may be
    both, because `load_judged` keys on the episode and a second verdict for the
    same key makes which one an analysis sees arbitrary.
    """
    import run as run_cli

    tp = str(tmp_path / "transcript.json")
    lost = ["usage_limit", "proxy_5xx", "rate_limit", "connection_closed",
            "network", "overloaded", "api_internal_server", "response_stalled"]
    findings = ["safety_refusal", "context_window_exceeded", "max_turns",
                "timeout", "output_token_exceeded", "unknown_api_error"]

    for tag in lost:
        rec = {"episode": f"{tag}#1", "ok": False, "failure": tag,
               "transcript_path": tp}
        assert judge.is_judgeable(rec) is False, tag
    for tag in findings:
        rec = {"episode": f"{tag}#1", "ok": False, "failure": tag,
               "transcript_path": tp}
        assert judge.is_judgeable(rec) is True, tag

    # No transcript is still no transcript, whatever the tag says.
    assert judge.is_judgeable({"failure": "max_turns"}) is False

    # The in-run auto-retry takes the transport faults and NOT the budget cap:
    # a key at its dollar limit answers the retry with the same 400, because
    # raising it needs a human.
    recs = [{"episode": f"{t}#1", "id": t, "ok": False, "failure": t,
             "transcript_path": tp} for t in lost + findings]
    picked = {r["episode"] for r in run_cli.select_transient_retries(recs)}
    assert picked == {f"{t}#1" for t in lost if t != "usage_limit"}
    assert "usage_limit#1" not in picked

    # Which leaves the capped episode neither judged nor retried in-process --
    # and that is the whole point of `load_done` retiring only `ok` records:
    # the resubmit that follows a raised cap re-runs it. Nothing here may retire
    # it, or it would never be re-run at all.
    results = tmp_path / "episodes.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in recs))
    retired = run_cli.load_done(results)
    assert retired == set()
    for rec in recs:
        judged = judge.is_judgeable(rec)
        assert judged is False or rec["episode"] not in picked, rec
        # Every episode this refuses is one SOME resume will replace.
        assert judged or (rec["episode"] in picked
                          or rec["episode"] not in retired), rec

    # A record already superseded by a good one is neither judged as the
    # failure nor re-run.
    good = {"episode": "usage_limit#1", "id": "usage_limit", "ok": True,
            "failure": None, "transcript_path": tp}
    assert run_cli.select_transient_retries(recs + [good]) == \
        run_cli.select_transient_retries(recs)
    assert judge.is_judgeable(good) is True
    results.write_text("".join(json.dumps(r) + "\n" for r in recs + [good]))
    assert run_cli.load_done(results) == {"usage_limit#1"}


# --------------------------------------------------------------------------
# a CLI that exits 0 after the episode was already lost
#
# `codex exec` printed the budget 400 above and then exited 0. The runner sets
# `error` only on a launch failure, its own timeout, or a non-zero exit with no
# rollout, so the record came back `ok: true`: retired by `load_done`, judged,
# and counted in the denominator. Three pelican episodes were filed that way.
def _traj(final_text: str = "") -> dict:
    return {"final_text": final_text}


def test_a_clean_exit_with_no_answer_and_a_lost_error_is_not_ok(
        tmp_path: Path) -> None:
    log = tmp_path / "container.stderr.log"
    log.write_text("[codex] thinking\n[codex] running command\n" + BUDGET_ERROR)
    got = trial.lost_on_a_clean_exit(_agent(), _traj(), log)
    assert got is not None
    text, cls = got
    assert cls is errors.ApiUsageLimitError
    assert "Budget has been exceeded" in text
    # And it lands on the record as a real failure, with the real tag: the
    # record's `ok` is `error is None`, so an error at all is `ok: false`.
    failure = trial.classify_failure(_agent(), text, cls, _traj(), log)
    assert failure is errors.ApiUsageLimitError
    assert errors.reason(failure) == "usage_limit"
    rec = {"ok": text is None, "failure": errors.reason(failure),
           "transcript_path": str(tmp_path / "transcript.json")}
    assert rec["ok"] is False
    assert judge.is_judgeable(rec) is False        # lost, so never judged


def test_an_episode_that_ANSWERED_is_never_relabelled(tmp_path: Path) -> None:
    """The condition that makes this safe to run on a successful episode.

    A gateway 502 the CLI retried and survived is still in the log at the end
    of a perfectly good episode. The agent delivered an answer, so the episode
    finished, so nothing here may touch it.
    """
    log = tmp_path / "container.stderr.log"
    log.write_text(GATEWAY_STDERR + "\n" + BUDGET_ERROR)
    assert trial.lost_on_a_clean_exit(
        _agent(), _traj("Here is the finished SVG."), log) is None


def test_a_clean_exit_is_never_relabelled_a_FINDING(tmp_path: Path) -> None:
    """Only "the harness lost this" is inferable from a log tail.

    A refusal or a context overflow arrives on the record properly when it
    happens. Reading one out of the log of an episode nobody reported as failed
    would be a guess, and it would be a guess about the measurement itself.
    """
    log = tmp_path / "container.stderr.log"
    log.write_text("Claude Code is unable to respond to this request, which "
                   "appears to violate our Usage Policy")
    assert trial.lost_on_a_clean_exit(_agent(), _traj(), log) is None
    log.write_text("prompt is too long: 214431 tokens > 200000 maximum")
    assert trial.lost_on_a_clean_exit(_agent(), _traj(), log) is None


def test_a_clean_exit_with_nothing_wrong_in_the_log_is_left_alone(
        tmp_path: Path) -> None:
    log = tmp_path / "container.stderr.log"
    log.write_text("[codex] tokens used: 41232\n[codex] done\n")
    assert trial.lost_on_a_clean_exit(_agent(), _traj(), log) is None
    assert trial.lost_on_a_clean_exit(_agent(), _traj(),
                                      tmp_path / "no-such-file.log") is None


def test_only_the_END_of_the_log_can_condemn_a_clean_exit(
        tmp_path: Path) -> None:
    """A fault a hundred lines back is history, not the reason it stopped."""
    log = tmp_path / "container.stderr.log"
    log.write_text(BUDGET_ERROR + "\n"
                   + "\n".join(f"[codex] step {i}" for i in range(100)))
    assert trial.lost_on_a_clean_exit(_agent(), _traj(), log) is None


def test_deepseek_incomplete_read_is_retryable_and_not_judgeable():
    from core.agents import errors
    from core.judge import is_judgeable
    from run import select_transient_retries
    failure = errors.classify("gateway failed: IncompleteRead: IncompleteRead(0 bytes read)")
    assert failure.reason == "connection_closed"
    assert errors.is_transient(failure) and errors.is_lost(failure)
    record = dict(id="rYUkb", episode="rYUkb#1", ok=False,
                  failure=failure.reason, transcript_path="transcript.json")
    assert not is_judgeable(record)
    assert select_transient_retries([record]) == [record]
    assert is_judgeable({**record, "failure": "timeout"})
    assert select_transient_retries([{**record, "failure": "timeout"}]) == []


def test_terminus_turn_limit_outranks_recovered_rate_limit(tmp_path):
    from core import trial
    from core.agents import make_agent
    log = tmp_path / "stderr.log"
    log.write_text("earlier request hit rate limit; recovered")
    assert trial.classify_failure(make_agent("terminus-2", model="openai/probe"),
        "max_turns reached before confirmed task completion", None, {}, log) is errors.MaxTurnsError


@pytest.mark.parametrize("message", [
    "No endpoints found that support image input",
    "This model does not support image input",
    "Failed to deserialize the JSON body into the target type: messages[2].content",
])
def test_unsupported_provider_inputs_are_excluded(message):
    from core import judge
    assert errors.classify(message) is errors.InvalidApiInputError
    assert not judge.is_judgeable({"transcript_path": "partial.json", "failure": "invalid_api_input", "error": message})


def test_terminus_disconnected_response_is_lost_even_in_legacy_record():
    from core import judge
    message = "InternalServerError: litellm.InternalServerError: AnthropicException - Server disconnected. Handle with litellm.InternalServerError."
    assert errors.classify(message) is errors.ApiConnectionClosedError
    assert not judge.is_judgeable({"transcript_path": "partial.json", "failure": "unclassified", "error": message})
    assert judge.is_judgeable({"transcript_path": "complete.json", "failure": None, "error": None})


def test_historical_t2_turn_cap_is_judged_without_rewriting_saved_record():
    from core import judge
    for failure in ("rate_limit", "unclassified", "max_turns"):
        record = {"failure": failure, "transcript_path": "saved.json",
                  "error": "max_turns reached before confirmed task completion"}
        assert judge.is_judgeable(record)
        assert record["failure"] == failure
    assert not judge.is_judgeable({"failure": "rate_limit", "transcript_path": "partial.json",
                                  "error": "rate limit exceeded"})
