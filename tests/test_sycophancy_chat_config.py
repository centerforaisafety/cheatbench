"""tasks/sycophancy/run_chat.py loads the same model entry run.py gives an agent and
delivers its settings in the chat API's terms: `reasoning_effort` as written; a Claude
entry's `effort` and `thinking` as the Claude SDK sends them, with the Messages API's
required token cap; Kimi's `thinking_effort` as the body its CLI sends; GLM's `effort`
as the gateway's `reasoning_effort`; and the Responses API display options
(`reasoning_summary`, `extra_body.reasoning.summary`) recorded rather than sent. A vendor
core/llm_agents.py has no client for is reached with the OpenAI client at the entry's
own endpoint. Every entry in configs/models.yaml therefore runs without a `-chat` twin."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks" / "sycophancy"))
import run_chat  # noqa: E402


def _models(tmp_path, body: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text("models:\n" + body)
    return path


def test_agent_entry_drops_reasoning_summary_and_records_it(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  gpt-6-astra:\n    model: openai/gpt-6-astra\n    api_key_env: OPENAI_API_KEY\n"
        "    api_base_url: https://gateway.example/v1\n"
        "    generation_config: {reasoning_effort: high, reasoning_summary: detailed}\n")), "gpt-6-astra")
    assert cfg["generation_config"] == {"reasoning_effort": "high"}
    assert cfg["unsupported"] == {"reasoning_summary": "detailed"}
    assert cfg["model"] == "openai/gpt-6-astra"


def test_chat_entry_is_sent_as_written(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  m-chat:\n    model: openai/m\n    api_key_env: OPENAI_API_KEY\n"
        "    api_base_url: https://gateway.example/v1\n"
        "    generation_config: {reasoning_effort: high}\n")), "m-chat")
    assert cfg["generation_config"] == {"reasoning_effort": "high"}
    assert cfg["unsupported"] == {}


def test_unknown_entry_lists_every_entry(tmp_path):
    with pytest.raises(SystemExit, match=r"entries: \['a', 'b-chat'\]"):
        run_chat.load_config(_models(tmp_path, (
            "  default: a\n  a: {model: openai/a}\n  b-chat: {model: openai/b}\n")), "zzz")


def test_claude_entry_effort_and_thinking_go_as_the_sdk_sends_them(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  claude-fable-5-1:\n    model: anthropic/claude-fable-5-1\n    api_key_env: ANTHROPIC_API_KEY\n"
        "    api_base_url: https://gateway.example\n"
        "    generation_config:\n      effort: high\n      thinking: {type: adaptive, display: summarized}\n")),
        "claude-fable-5-1")
    assert cfg["generation_config"] == {"output_config": {"effort": "high"},
                                        "thinking": {"type": "adaptive", "display": "summarized"},
                                        "max_tokens": run_chat.ANTHROPIC_MAX_TOKENS}
    assert cfg["entry_generation_config"] == {"effort": "high",
                                              "thinking": {"type": "adaptive", "display": "summarized"}}
    assert cfg["unsupported"] == {}
    # routed to a gateway: the Anthropic client sends the entry's full id, as claude-sdk does
    assert cfg["factory_model"] == "anthropic/anthropic/claude-fable-5-1"


def test_claude_entry_without_effort_gets_only_the_required_cap(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  claude-opus-5:\n    model: anthropic/claude-opus-5\n    api_key_env: ANTHROPIC_API_KEY\n"
        "    api_base_url: https://gateway.example\n"
        "    generation_config:\n      thinking: {type: adaptive, display: summarized}\n")), "claude-opus-5")
    assert cfg["generation_config"] == {"thinking": {"type": "adaptive", "display": "summarized"},
                                        "max_tokens": run_chat.ANTHROPIC_MAX_TOKENS}


def test_kimi_thinking_effort_is_the_cli_request_body(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  kimi-k3:\n    model: openrouter/moonshotai/kimi-k3\n    api_key_env: LITELLM_API_KEY\n"
        "    api_base_url: https://gateway.example/v1\n"
        "    extra_body: {provider: {order: [moonshotai], allow_fallbacks: false}}\n"
        "    generation_config: {thinking_effort: high}\n")), "kimi-k3")
    assert cfg["generation_config"] == {}
    assert cfg["extra_body"] == {"provider": {"order": ["moonshotai"], "allow_fallbacks": False},
                                 "thinking": {"type": "enabled", "effort": "high"}}
    assert cfg["entry_extra_body"] == {"provider": {"order": ["moonshotai"], "allow_fallbacks": False}}
    assert cfg["entry_generation_config"] == {"thinking_effort": "high"}


def test_openrouter_effort_is_reasoning_effort(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  glm-5.3:\n    model: openrouter/z-ai/glm-5.3\n    api_key_env: LITELLM_API_KEY\n"
        "    api_base_url: https://gateway.example/v1\n    generation_config: {effort: high}\n")), "glm-5.3")
    assert cfg["generation_config"] == {"reasoning_effort": "high"}
    assert cfg["factory_model"] == "openai/openrouter/z-ai/glm-5.3"


def test_gateway_routed_ids_go_verbatim_and_vendor_routed_ids_go_to_the_vendor_client(tmp_path):
    def load(model, base):
        return run_chat.load_config(_models(tmp_path, (
            f"  m:\n    model: {model}\n    api_key_env: K\n"
            + (f"    api_base_url: {base}\n" if base else ""))), "m")
    for model in ("openrouter/x-ai/grok-4.6", "openrouter/moonshotai/kimi-k3", "gemini/gemini-3.8-flash"):
        assert load(model, "https://gateway.example/v1")["factory_model"] == "openai/" + model
    assert load("openrouter/x-ai/grok-4.6", "https://openrouter.ai/api/v1")["factory_model"] == "openrouter/x-ai/grok-4.6"
    assert load("gemini/gemini-3.8-flash", "")["factory_model"] == "gemini/gemini-3.8-flash"
    assert load("openai/gpt-6-astra", "https://gateway.example/v1")["factory_model"] == "openai/gpt-6-astra"
    # An anthropic/ id keeps the Anthropic client; on a gateway the full id is sent, as claude-sdk does.
    assert load("anthropic/claude-opus-5", "https://gateway.example")["factory_model"] == "anthropic/anthropic/claude-opus-5"
    assert load("anthropic/claude-opus-5", "https://api.anthropic.com")["factory_model"] == "anthropic/claude-opus-5"
    assert load("anthropic/claude-opus-5", "")["factory_model"] == "anthropic/claude-opus-5"


def test_meta_entry_uses_the_openai_client_at_its_own_endpoint(tmp_path):
    cfg = run_chat.load_config(_models(tmp_path, (
        "  muse-spark-1.3:\n    model: meta/muse-spark-1.3\n    api_key_env: META_API_KEY\n"
        "    api_base_url: https://api.meta.example/v1\n    generation_config: {reasoning_effort: high}\n"
        "    extra_body: {reasoning: {summary: detailed}}\n")), "muse-spark-1.3")
    assert cfg["factory_model"] == "openai/muse-spark-1.3"
    assert cfg["model"] == "meta/muse-spark-1.3"
    assert cfg["generation_config"] == {"reasoning_effort": "high"}
    assert cfg["extra_body"] == {}
    assert cfg["unsupported"] == {"extra_body.reasoning.summary": "detailed"}


def test_bare_model_name_needs_an_endpoint(tmp_path):
    body = ("  deepseek-v4-pro:\n    model: deepseek-v4-pro\n    api_key_env: DEEPSEEK_API_KEY\n"
            "    generation_config: {reasoning_effort: high}\n")
    with pytest.raises(SystemExit, match="no api_base_url"):
        run_chat.load_config(_models(tmp_path, body), "deepseek-v4-pro")
    cfg = run_chat.load_config(_models(tmp_path, body + "    api_base_url: https://api.deepseek.example\n"),
                               "deepseek-v4-pro")
    assert cfg["factory_model"] == "openai/deepseek-v4-pro"
    assert cfg["model"] == "deepseek-v4-pro"
    assert cfg["generation_config"] == {"reasoning_effort": "high"}
