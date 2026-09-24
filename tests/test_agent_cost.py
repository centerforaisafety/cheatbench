"""What an episode cost, and the one convention that decides whether it is right.

`litellm.cost_per_token` takes an INCLUSIVE prompt count: cache reads and cache
writes are inside `prompt_tokens`, and litellm subtracts them back out to price
each leg at its own rate. Codex reports its input count that way already;
Anthropic does not. Get it backwards in one direction and a 95%-cached episode
prices every input token as fresh (5x too much, and entirely plausible-looking);
get it backwards in the other and the cost goes NEGATIVE.

Neither is visible in a results table, so both directions are pinned here,
together with the two things that would let a future adapter reintroduce the
bug this file exists because of: a missing model returning 0.0 instead of None,
and an adapter shipping with no statement about cost at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agents import make_agent  # noqa: E402
from core.agents.base import Agent  # noqa: E402
from core.trajectory import Trajectory  # noqa: E402

# A REAL Codex episode: outputs/openmath_gpt-5.6-sol_proxy_codex_e2e. 95% of its
# input was cache reads, which is what makes it the useful fixture.
CODEX_USAGE = {"input_tokens": 1551808, "cached_input_tokens": 1473125,
               "cache_write_input_tokens": 78584, "output_tokens": 10380,
               "reasoning_output_tokens": 3630, "total_tokens": 1562188}

# Recorded rates for this historical episode. The fixture pins both aliases
# so updates to the live price table cannot change this arithmetic regression.
SOL_RATES = {"input_cost_per_token": 4e-06, "output_cost_per_token": 2e-05,
             "cache_creation_input_token_cost": 5e-06,
             "cache_read_input_token_cost": 4e-07}
CODEX_REFERENCE_USD = 1.190166


@pytest.fixture(autouse=True)
def historical_sol_prices(monkeypatch):
    """The archived dollar fixture uses its recorded rates, not a live price table."""
    import litellm
    for key in ("gpt-5.6-sol", "openai/gpt-5.6-sol"):
        monkeypatch.setitem(litellm.model_cost, key,
                            {**litellm.model_cost.get(key, {}), **SOL_RATES})

# A REAL Claude episode: outputs/openmath_claude-opus-5_proxy_claude_e2e, which
# recorded this exact `total_cost_usd`. `input_tokens: 20` is the FRESH
# remainder only -- the cache legs sit beside it, not inside it.
CLAUDE_STREAM_USAGE = {"input_tokens": 20, "output_tokens": 8579,
                       "cache_creation_input_tokens": 30682,
                       "cache_read_input_tokens": 69450}
CLAUDE_REPORTED_USD = 0.4466975


@pytest.fixture(autouse=True)
def reference_sol_pricing(monkeypatch):
    """Test token accounting against the recorded rates, not a changing catalog."""
    import litellm
    for key in ("gpt-5.6-sol", "openai/gpt-5.6-sol"):
        info = dict(litellm.model_cost.get(key) or litellm.model_cost["gpt-5.6-sol"])
        info = {k: v for k, v in info.items() if "_above_" not in k}
        info.update(SOL_RATES)
        monkeypatch.setitem(litellm.model_cost, key, info)


def _codex(model: str = "openai/gpt-5.6-sol"):
    return make_agent("codex", model=model)


def _claude(model: str = "anthropic/claude-opus-5"):
    return make_agent("claude-sdk", model=model)


def _rates(model: str) -> dict:
    import litellm
    return litellm.model_cost[model]


# --------------------------------------------------------------------------
# THE GUARD: all four token categories, priced at their own rates
# --------------------------------------------------------------------------
def test_codex_episode_reproduces_the_reference_dollar_figure() -> None:
    """The four categories go through as four categories, not as one total."""
    cost, source = _codex().episode_cost({"usage": CODEX_USAGE})
    assert cost == pytest.approx(CODEX_REFERENCE_USD, abs=5e-6)
    assert source == "estimated"


def test_pricing_the_whole_total_at_the_input_rate_would_be_5x_wrong() -> None:
    """Why the categories matter: the plausible-looking mistake, measured."""
    naive = CODEX_USAGE["total_tokens"] * _rates("gpt-5.6-sol")["input_cost_per_token"]
    cost, _ = _codex().episode_cost({"usage": CODEX_USAGE})
    assert naive > 5 * cost


def test_cache_reads_are_priced_at_the_cache_read_rate() -> None:
    """Each leg at its own rate, checked against the table rather than a number."""
    r = _rates("gpt-5.6-sol")
    fresh = (CODEX_USAGE["input_tokens"] - CODEX_USAGE["cached_input_tokens"]
             - CODEX_USAGE["cache_write_input_tokens"])
    expected = (fresh * r["input_cost_per_token"]
                + CODEX_USAGE["cache_write_input_tokens"]
                * r["cache_creation_input_token_cost"]
                + CODEX_USAGE["cached_input_tokens"]
                * r["cache_read_input_token_cost"]
                + CODEX_USAGE["output_tokens"] * r["output_cost_per_token"])
    cost, _ = _codex().episode_cost({"usage": CODEX_USAGE})
    assert cost == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# INCLUSIVE vs EXCLUSIVE -- the single most likely silent 10x
# --------------------------------------------------------------------------
def test_codex_prompt_count_is_inclusive_of_its_cache_legs() -> None:
    """Codex's own arithmetic says so: input == cached + written + fresh."""
    assert (CODEX_USAGE["input_tokens"]
            >= CODEX_USAGE["cached_input_tokens"]
            + CODEX_USAGE["cache_write_input_tokens"])
    assert (CODEX_USAGE["input_tokens"] + CODEX_USAGE["output_tokens"]
            == CODEX_USAGE["total_tokens"])
    assert _codex().PROMPT_TOKENS_INCLUDE_CACHE is True
    got = _codex().usage_categories({"usage": CODEX_USAGE})
    assert got == {"prompt_tokens": 1551808, "completion_tokens": 10380,
                   "cached_tokens": 1473125, "cache_write_tokens": 78584}


def test_claude_prompt_count_is_exclusive_and_gets_the_cache_legs_added() -> None:
    """Anthropic reports the fresh remainder only; the base class adds them in."""
    assert _claude().PROMPT_TOKENS_INCLUDE_CACHE is False
    got = _claude().usage_categories({"stream_usage": CLAUDE_STREAM_USAGE})
    assert got["prompt_tokens"] == (CLAUDE_STREAM_USAGE["input_tokens"]
                                    + CLAUDE_STREAM_USAGE["cache_creation_input_tokens"]
                                    + CLAUDE_STREAM_USAGE["cache_read_input_tokens"])
    assert got["cached_tokens"] == CLAUDE_STREAM_USAGE["cache_read_input_tokens"]
    assert got["cache_write_tokens"] == CLAUDE_STREAM_USAGE["cache_creation_input_tokens"]


def test_an_exclusive_prompt_count_raises_rather_than_going_negative() -> None:
    """The tripwire. litellm would happily return -$4.70 for this."""
    agent = _codex()
    with pytest.raises(ValueError, match="INCLUSIVE"):
        agent.cost_from_usage(
            prompt_tokens=(CODEX_USAGE["input_tokens"]
                           - CODEX_USAGE["cached_input_tokens"]),
            completion_tokens=CODEX_USAGE["output_tokens"],
            cached_tokens=CODEX_USAGE["cached_input_tokens"],
            cache_write_tokens=CODEX_USAGE["cache_write_input_tokens"])


def test_claude_estimate_lands_near_the_figure_the_vendor_reported() -> None:
    """The two conventions, cross-checked against real money.

    Not an equality: the SDK's own figure and a litellm reprice of the same
    tokens are two different measurements. But a convention error would be off
    by orders of magnitude, not by 2%.
    """
    cost, source = _claude().episode_cost(
        {"cost_usd": None, "stream_usage": CLAUDE_STREAM_USAGE})
    assert source == "estimated"
    assert cost == pytest.approx(CLAUDE_REPORTED_USD, rel=0.05)


def test_reasoning_tokens_are_not_added_on_top_of_output_tokens() -> None:
    """Codex's `output_tokens` already contains them; adding would double-count."""
    assert "reasoning_output_tokens" not in _codex().USAGE_KEYS
    r = _rates("gpt-5.6-sol")
    cost, _ = _codex().episode_cost({"usage": CODEX_USAGE})
    doubled, _ = _codex().episode_cost(
        {"usage": {**CODEX_USAGE,
                   "output_tokens": (CODEX_USAGE["output_tokens"]
                                     + CODEX_USAGE["reasoning_output_tokens"])}})
    assert doubled - cost == pytest.approx(
        CODEX_USAGE["reasoning_output_tokens"] * r["output_cost_per_token"],
        rel=1e-6)


# --------------------------------------------------------------------------
# "we do not know" is not "it was free"
# --------------------------------------------------------------------------
def test_a_model_absent_from_the_pricing_table_returns_none_not_zero() -> None:
    agent = _codex(model="openai/not-a-real-model-xyz")
    cost, source = agent.episode_cost({"usage": CODEX_USAGE})
    assert cost is None and cost != 0.0
    assert source is None


def test_a_record_with_no_usage_at_all_returns_none() -> None:
    cost, source = _codex().episode_cost({"usage": None})
    assert cost is None and source is None


def test_an_all_zero_usage_field_falls_through_to_the_next_source() -> None:
    """Claude's stream counters exist from the start and can be empty."""
    zeros = dict.fromkeys(CLAUDE_STREAM_USAGE, 0)
    cost, source = _claude().episode_cost(
        {"cost_usd": None, "stream_usage": zeros,
         "usage": {"input_tokens": 18, "cache_creation_input_tokens": 29218,
                   "cache_read_input_tokens": 55655, "output_tokens": 8579}})
    assert source == "estimated" and cost > 0


def test_the_provider_prefix_is_stripped_to_find_the_pricing_key() -> None:
    """Records carry `openai/gpt-5.6-sol`; litellm keys it bare."""
    bare, _ = _codex(model="gpt-5.6-sol").episode_cost({"usage": CODEX_USAGE})
    prefixed, _ = _codex().episode_cost({"usage": CODEX_USAGE})
    assert bare == prefixed


# --------------------------------------------------------------------------
# reported beats estimated, and the record says which it was
# --------------------------------------------------------------------------
def test_a_reported_claude_cost_is_passed_through_unchanged() -> None:
    """The SDK's own figure is authoritative and must not be re-derived."""
    cost, source = _claude().episode_cost(
        {"cost_usd": CLAUDE_REPORTED_USD, "stream_usage": CLAUDE_STREAM_USAGE})
    assert cost == CLAUDE_REPORTED_USD
    assert source == "reported"


def test_the_estimate_is_only_a_fallback_for_claude() -> None:
    """Same usage, with and without the reported figure."""
    usage = {"stream_usage": CLAUDE_STREAM_USAGE}
    reported, _ = _claude().episode_cost({**usage, "cost_usd": CLAUDE_REPORTED_USD})
    estimated, _ = _claude().episode_cost({**usage, "cost_usd": None})
    assert reported == CLAUDE_REPORTED_USD
    assert estimated != reported          # it is a different measurement
    assert estimated is not None          # and it exists, where before there was none


def test_a_timed_out_claude_episode_is_no_longer_free() -> None:
    """The case the whole fallback exists for: no ResultMessage, real money."""
    killed = {"cost_usd": None, "usage": None,
              "stream_usage": CLAUDE_STREAM_USAGE,
              "error": "timeout after 1800s"}
    rec = _claude().record(killed)
    assert rec["cost_usd"] > 0
    assert rec["cost_source"] == "estimated"


def test_the_record_envelope_carries_the_cost_and_its_source() -> None:
    codex_rec = _codex().record({"usage": CODEX_USAGE})
    assert codex_rec["cost_source"] == "estimated"
    assert codex_rec["cost_usd"] == pytest.approx(CODEX_REFERENCE_USD, abs=5e-6)

    claude_rec = _claude().record({"cost_usd": CLAUDE_REPORTED_USD,
                                   "stream_usage": CLAUDE_STREAM_USAGE})
    assert claude_rec["cost_source"] == "reported"
    assert claude_rec["cost_usd"] == CLAUDE_REPORTED_USD

    empty = _codex().record({})
    assert empty["cost_usd"] is None and empty["cost_source"] is None


def test_codex_never_claims_a_reported_cost() -> None:
    """Codex prices nothing, and says so rather than staying silent."""
    assert _codex().reported_cost_usd({"cost_usd": 9.99}) is None


# --------------------------------------------------------------------------
# An adapter cannot ship without deciding about cost
# --------------------------------------------------------------------------
def _adapter_body(**extra):
    """The bits of the adapter contract that are not about cost."""
    body = {
        "name": staticmethod(lambda: "stub"),
        "_tools_for": lambda self, policy: [],
        "bootstrap": property(lambda self: ""),
        "blob": lambda self, **kw: b"",
        "to_trajectory": lambda self, raw: Trajectory(steps=[]),
    }
    body.update(extra)
    return body


def test_an_adapter_that_ignores_cost_fails_when_the_class_is_defined() -> None:
    """Loud at import, not a silent None at the end of an expensive run."""
    with pytest.raises(TypeError, match="reported_cost_usd"):
        type("NoCostAdapter", (Agent,), _adapter_body(USAGE_KEYS=()))


def test_an_adapter_that_does_not_declare_usage_keys_is_refused() -> None:
    with pytest.raises(TypeError, match="USAGE_KEYS"):
        type("NoUsageKeysAdapter", (Agent,),
             _adapter_body(reported_cost_usd=lambda self, raw: None))


def test_usage_keys_must_name_exactly_the_four_categories() -> None:
    with pytest.raises(TypeError, match="four"):
        type("ThreeKeysAdapter", (Agent,),
             _adapter_body(reported_cost_usd=lambda self, raw: None,
                           USAGE_KEYS=("a", "b", "c")))


def test_a_complete_adapter_that_declares_both_is_accepted() -> None:
    cls = type("CompleteAdapter", (Agent,),
               _adapter_body(reported_cost_usd=lambda self, raw: None,
                             USAGE_KEYS=()))
    assert cls(model="x").episode_cost({}) == (None, None)


def test_backfilling_twice_does_not_relabel_an_estimate_as_reported() -> None:
    """The backfill writes into the same field the vendor's figure lives in.

    Claude's `reported_cost_usd` reads `cost_usd` off the record, so a second
    pass over an already-backfilled run would read back THIS code's own estimate
    and call it the vendor's -- which is the one way an estimate could end up
    presented as a reported figure.
    """
    from core.cost_backfill import recost
    record = {"agent": "claude-sdk", "model": "anthropic/claude-opus-5",
              "cost_usd": None, "usage": None,
              "stream_usage": CLAUDE_STREAM_USAGE}
    record["cost_usd"], record["cost_source"] = recost(record)
    assert record["cost_source"] == "estimated"
    again = recost(record)
    assert again == (record["cost_usd"], "estimated")


def test_backfilling_never_moves_a_reported_figure() -> None:
    from core.cost_backfill import recost
    record = {"agent": "claude-sdk", "model": "anthropic/claude-opus-5",
              "cost_usd": CLAUDE_REPORTED_USD,
              "stream_usage": CLAUDE_STREAM_USAGE}
    assert recost(record) == (CLAUDE_REPORTED_USD, "reported")
    assert recost({**record, "cost_source": "reported"}) == (
        CLAUDE_REPORTED_USD, "reported")


def test_a_still_abstract_intermediate_class_is_not_checked() -> None:
    """`InstalledAgent` implements none of the vendor surface; its children do."""
    from core.agents.installed import InstalledAgent
    partial = type("PartialInstalled", (InstalledAgent,), {})
    assert partial.USAGE_KEYS is None
