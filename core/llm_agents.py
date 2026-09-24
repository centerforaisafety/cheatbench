"""Shared provider clients, structured responses, and token-cost accounting."""
import asyncio
import base64
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from core.routing import resolve_base_url
from core.config import load_yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Must run BEFORE `import litellm`: litellm load_dotenv()s at import time and
# dotenv never overrides an already-set var, so a stale ~/.env would win.
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

import anthropic  # noqa: E402
import openai  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import litellm  # noqa: E402
import requests  # noqa: E402

litellm.suppress_debug_info = True
# register_model warns once per model that has no cache-cost fields; the price
# snapshot has hundreds. We do not track cache cost, so the noise is pure.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# Price table: MODEL_PRICES_JSON if set, else a live fetch. UNSET IS THE
# DEFAULT AND THE RIGHT ONE. `register_model` MERGES this over litellm's own
# bundled table, so a stale snapshot silently WINS for every model both contain:
# the 2026-07-16 file this used to point at predated both gpt-5.6-sol and
# claude-opus-5, which is how episode costs went missing. Re-pin only against a
# file pulled fresh from UPSTREAM_URL, and only to make a run reproducible
# against a price list that is known to contain the models it prices.
#
# `Agent.cost_from_usage` (core/agents/base.py) prices episodes off whatever
# `litellm.model_cost` holds after this runs -- but it does NOT depend on this
# module having been imported. litellm's bundled table already carries both
# models at the same rates as upstream, so an episode costs the same whether the
# fetch above happened, failed, or never ran.
LOCAL_PATH = os.environ.get("MODEL_PRICES_JSON")
UPSTREAM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
try:
    if LOCAL_PATH:
        with open(LOCAL_PATH) as _f:
            _model_cost_data = json.load(_f)
    else:
        _model_cost_data = requests.get(UPSTREAM_URL, timeout=10).json()
    _model_cost_data = {k: v for k, v in _model_cost_data.items() if not k.startswith("github_copilot/")}
    litellm.register_model(model_cost=_model_cost_data)
except Exception as e:
    print(f"Warning: Failed to load model costs ({LOCAL_PATH or UPSTREAM_URL}): {e}")

TIMEOUT=3600


def _coerce_response_format(kwargs: dict) -> dict:
  """Let callers pass `response_format=SomePydanticModel`.

  Structured output: the chat-completions API wants a `json_schema` block, so a
  pydantic class is converted to one here (strict mode -- every field required,
  no additional properties). Anything else, including an already-built dict, is
  passed through untouched. A provider that refuses the parameter raises, and
  the caller is expected to fall back.
  """
  rf = kwargs.get("response_format")
  if isinstance(rf, type) and issubclass(rf, BaseModel):
    from openai.lib._parsing._completions import type_to_response_format_param
    kwargs = dict(kwargs)
    kwargs["response_format"] = type_to_response_format_param(rf)
  return kwargs


def get_llm_agent_class(model: str, generation_config: dict = {}, **kwargs):
  provider, model_name = model.split("/", 1)

  provider_to_class = {
    'openai': OpenAIAgent,
    'anthropic': AnthropicAgent,
    'gemini': GeminiAgent,
    'xai': GrokAgent,
    'openrouter': OpenRouterAgent,
    'bedrock': BedrockAgent,
  }
  assert provider in provider_to_class, f"Provider {provider} not supported"
  if provider != 'bedrock':
    from core.model_settings import api_generation
    generation_config, extra, _ = api_generation(model, generation_config)
    if extra:
      generation_config['extra_body'] = extra
  return provider_to_class[provider](model=model_name, **generation_config, **kwargs)


class TokenUsage(BaseModel):
  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  cached_tokens: int = 0
  cost: float = 0.0

class LLMResponse(BaseModel):
  content: str | None = None
  reasoning_content: str | None = None
  token_usage: TokenUsage | None = None
  raw: dict | None = None


class LLMAgent(ABC):
  def __init__(self, model: str, provider: str = None):
    self.model = model
    self.provider = provider
    self.all_token_usage = TokenUsage()
    self.max_token_usage = TokenUsage()
    self._usage_lock = asyncio.Lock()  # Lock for async usage updates

  def _update_usage(self, token_usage: TokenUsage | None):
    if token_usage is None:
      return
    self.all_token_usage = sum_token_usage([self.all_token_usage, token_usage])
    self.max_token_usage = get_max_token_usage([self.max_token_usage, token_usage])

  async def _update_usage_async(self, token_usage: TokenUsage | None):
    """Update token usage asynchronously with lock (for concurrent async calls)."""
    if token_usage is None:
      return
    async with self._usage_lock:
      self.all_token_usage = sum_token_usage([self.all_token_usage, token_usage])
      self.max_token_usage = get_max_token_usage([self.max_token_usage, token_usage])
    
  def _calculate_cost(self, response: Any) -> float:
    # Add total_tokens field if missing (e.g., Anthropic responses don't have this)
    usage = getattr(response, 'usage', None)
    if usage and not hasattr(usage, 'total_tokens'):
      usage.total_tokens = getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)

    # OpenRouter returns the exact cost in `usage.cost` — trust it directly.
    upstream_cost = getattr(usage, "cost", None)
    if upstream_cost is not None:
      return float(upstream_cost)
    # xAI returns `usage.cost_in_usd_ticks` (1 tick = $1e-10). Some newer models
    # (e.g. grok-4.5) return the field as 0 before billing is wired — treat a
    # zero/falsy tick as "no native cost" and fall through to the price table.
    ticks = getattr(usage, "cost_in_usd_ticks", None)
    if ticks:
      return float(ticks) * 1e-10

    # Normalize the response model name so litellm's pricing lookup hits the base
    # key registered in model_prices_and_context_window.json. xAI returns a
    # `-internal` suffix (e.g. `grok-4.5-internal`); OpenAI-style APIs append a
    # date (e.g. `gpt-5.4-nano-2026-03-17`). Fall back to self.model if absent.
    import re as _re
    base_model = getattr(response, "model", None) or self.model
    if base_model:
      base_model = _re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base_model)
      base_model = _re.sub(r"-internal$", "", base_model)

    try:
      cost = litellm.cost_calculator.completion_cost(
          completion_response=response,
          model=base_model,
          custom_llm_provider=self.provider,
      )
    except Exception as e:
      # The provider-scoped lookup fails for a model reached through an
      # OpenAI-compatible endpoint under another vendor's name -- DeepSeek at
      # api.deepseek.com, Muse at api.meta.ai, or a gateway-routed
      # `gemini/...` -- which litellm keys under the vendor (`deepseek/...`,
      # `meta/...`), not under `openai`. Price it by the table key that names
      # it instead, as Agent.cost_from_usage (core/agents/base.py) does.
      cost = self._cost_by_table_key(base_model, usage)
      if cost is None:
        print(f"Warning: Cost calculation failed for {self.provider}/{self.model}: {e}")
        cost = 0.0

    # Recover hidden thinking tokens (Gemini/xAI via OpenAI-compat hide them in total_tokens).
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    if pt is not None and ct is not None:
      hidden = usage.total_tokens - pt - ct
      if hidden > 0:
        info = (litellm.model_cost.get(base_model)
                or litellm.model_cost.get(f"{self.provider}/{base_model}")
                or litellm.model_cost.get(base_model.split("/", 1)[-1]) or {})
        cost = (cost or 0.0) + hidden * (info.get("output_cost_per_token") or 0)

    return cost

  @staticmethod
  def _price_keys(name: str) -> list:
    """litellm price-table keys that could price a model NAME, best first.

    Harbor's resolution first (core/agents/base.py cost_from_usage): the full
    name, then the name with its provider prefix stripped. Then one more step
    for a name that reaches a vendor through an OpenAI-compatible route: every
    table key whose last path component is that bare name, shortest first, so
    `deepseek-v4-pro` prices as `deepseek/deepseek-v4-pro` (the vendor's own
    listing) before a reseller's `openrouter/deepseek/deepseek-v4-pro`. Keys
    that carry no output price are skipped: the table holds bare placeholder
    entries (`deepseek-v4-pro`) beside the priced vendor entry.
    """
    if not name:
      return []
    bare = name.split("/", 1)[-1]
    last = bare.rsplit("/", 1)[-1]
    ordered = [name, bare]
    # The vendor's own listing next: the table names each entry's provider, so a
    # bare entry (`deepseek-v4-pro`, provider `deepseek`) points at
    # `deepseek/deepseek-v4-pro`. Resellers (`tencent/...`, `azure_ai/...`)
    # price the same name differently and come last, shortest first.
    for key in (name, bare):
      provider = (litellm.model_cost.get(key) or {}).get("litellm_provider")
      if provider:
        ordered.append(f"{provider}/{last}")
    ordered += sorted((k for k in litellm.model_cost if k.rsplit("/", 1)[-1] == last), key=len)
    out = []
    for key in ordered:
      info = litellm.model_cost.get(key)
      if info and info.get("output_cost_per_token") and key not in out:
        out.append(key)
    return out

  def _cost_by_table_key(self, name: str, usage) -> float | None:
    """Price a response's token counts by name, when the provider lookup cannot."""
    if usage is None:
      return None
    pt = getattr(usage, "prompt_tokens", None)
    if pt is None:
      pt = getattr(usage, "input_tokens", 0)
    ct = getattr(usage, "completion_tokens", None)
    if ct is None:
      ct = getattr(usage, "output_tokens", 0)
    cached = int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
    pt, ct = int(pt or 0), int(ct or 0)
    keys = self._price_keys(name)
    if not keys:
      return None
    # Priced from the table entry itself rather than through litellm's pricer, which
    # in some versions rejects a provider prefix it does not route (`meta/...`) even
    # though the table carries the price. Cache reads are priced at the entry's
    # cache-read rate when it has one, as litellm does.
    info = litellm.model_cost[keys[0]]
    in_rate = info.get("input_cost_per_token") or 0.0
    out_rate = info.get("output_cost_per_token") or 0.0
    cache_rate = info.get("cache_read_input_token_cost", in_rate) or 0.0
    cached = min(cached, pt)
    return (pt - cached) * in_rate + cached * cache_rate + ct * out_rate

  @abstractmethod
  def _completions(self, messages) -> LLMResponse:
    raise NotImplementedError

  @abstractmethod
  async def _async_completions(self, messages) -> LLMResponse:
    raise NotImplementedError

  def completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    return self._completions(messages, **kwargs)

  async def async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    return await self._async_completions(messages, **kwargs)

class OpenAIAgent(LLMAgent):
  def __init__(self, 
               model: str,
               api_key_env: str = 'OPENAI_API_KEY',
               api_base_url: str | None = None,
               provider: str = 'openai',
               api_base_url_env: str | None = None,
               **generation_config):
    super().__init__(model=model, provider=provider)

    api_base_url = resolve_base_url(api_base_url, api_base_url_env)
    api_base_url = api_base_url or os.getenv('OPENAI_API_BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv(api_key_env)
    if not api_key:
      raise ValueError(f"API key not found in environment variable {api_key_env}")

    self.client = openai.OpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
    self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
    self.generation_config = generation_config

  def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
    return messages

  def _parse_response(self, response):
    """Parse response and extract content, reasoning_content, token usage, and cost."""
    raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, 'message', None) if choice is not None else None
    content = getattr(message, 'content', None) if message is not None else None
    # Capture reasoning_content if present (e.g., DeepSeek reasoner models)
    reasoning_content = getattr(message, 'reasoning_content', None) if message is not None else None
    usage = getattr(response, 'usage', None)
    if usage is None:
      # Usage is optional on compatible endpoints; preserve the answer and
      # raw response without inventing token counts or a zero-dollar cost.
      return content, reasoning_content, None, raw_response
    cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0

    cost = self._calculate_cost(response)
    token_usage = TokenUsage(
      input_tokens=usage.prompt_tokens,
      output_tokens=usage.completion_tokens,
      total_tokens=usage.total_tokens,
      cached_tokens=cached_tokens,
      cost=cost,
    )

    return content, reasoning_content, token_usage, raw_response

  def _completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    messages = self._preprocess_messages(messages)
    kwargs = _coerce_response_format(kwargs)
    response = self.client.chat.completions.create(
      model=self.model,
      messages=messages,
      **self.generation_config,
      **kwargs,
    )
    content, reasoning_content, token_usage, raw_response = self._parse_response(response)
    self._update_usage(token_usage)

    return LLMResponse(content=content, reasoning_content=reasoning_content, token_usage=token_usage, raw=raw_response)

  async def _async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    messages = self._preprocess_messages(messages)
    kwargs = _coerce_response_format(kwargs)
    response = await self.async_client.chat.completions.create(
      model=self.model,
      messages=messages,
      **self.generation_config,
      **kwargs,
    )
    content, reasoning_content, token_usage, raw_response = self._parse_response(response)
    await self._update_usage_async(token_usage)

    return LLMResponse(content=content, reasoning_content=reasoning_content, token_usage=token_usage, raw=raw_response)


class GrokAgent(OpenAIAgent):
  def __init__(self, model: str,
               api_key_env: str = 'XAI_API_KEY',
               api_base_url: str = 'https://api.x.ai/v1', 
               provider: str = 'xai',
               api_base_url_env: str | None = None,
               **generation_config):
    super().__init__(model=model, 
                     api_key_env=api_key_env, 
                     api_base_url=api_base_url, 
                     provider=provider,
                     api_base_url_env=api_base_url_env,
                     **generation_config)
    # Check if this is a grok-4 model that needs conversation flattening
    self.needs_flattening = 'grok-4' in model.lower() and "grok-4-fast" not in model.lower()
    # self.needs_flattening = False
  
  def _flatten_conversation(self, messages: list[dict]) -> list[dict]:
    """Flatten multi-turn conversation for Grok-4 model. This is because Grok-4 constantly give errors on multi-turn conversations.
    
    Converts user/assistant pairs into single message with USER:/ASSISTANT: markers.
    """
    if not messages:
      return messages
    
    # Separate system from conversation messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs = [m for m in messages if m["role"] != "system"]
    
    # No flattening needed for N message yet
    if len(conv_msgs) <= 4:
      return messages
    
    # Extract text content, handling both string and list formats
    def get_text(content):
      if isinstance(content, str):
        return content
      if isinstance(content, list):
        return " ".join(item.get("text", str(item)) for item in content if isinstance(item, dict))
      return str(content)
    
    # Build flattened conversation with USER:/ASSISTANT: markers
    flattened = "\n\n".join(f"{m['role'].upper()}: {get_text(m['content'])}" for m in conv_msgs)
    
    return system_msgs + [{"role": "user", "content": flattened}]
  
  def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
    """Preprocess messages, applying flattening for Grok-4 models."""
    if self.needs_flattening:
      return self._flatten_conversation(messages)
    return messages
 
class GeminiAgent(OpenAIAgent):
  def __init__(self, model: str, 
               api_key_env: str = 'GEMINI_API_KEY',
               api_base_url: str | None = None,
               provider: str = 'gemini',
               vertexai: bool = False,
               api_base_url_env: str | None = None,
               **generation_config):

    if vertexai and api_base_url_env is not None:
        raise ValueError("api_base_url_env is not supported with vertexai=True")
    if not vertexai:
        api_base_url = (api_base_url or os.environ.get("GEMINI_BASE_URL")
                        or "https://generativelanguage.googleapis.com/v1beta/openai/")
        super().__init__(model=model,
                         api_key_env=api_key_env, 
                         api_base_url=api_base_url, 
                         provider=provider,
                         api_base_url_env=api_base_url_env,
                         **generation_config)
    else:
        # https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/chat-completions/intro_chat_completions_api.ipynb
        from google.auth import default
        from google.auth.transport.requests import Request
        
        if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
            raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set")
        
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        
        api_host = f"{location}-aiplatform.googleapis.com" if location != "global" else "aiplatform.googleapis.com"
        api_base_url = f"https://{api_host}/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
        api_key = credentials.token
        # Only add google/ prefix for Google's own models, not third-party Vertex AI models
        if "/" not in model:
            model = f"google/{model}"

        # Initialize base class attributes (vertexai path doesn't call super().__init__)
        LLMAgent.__init__(self, model=model, provider='vertex_ai')
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
        self.generation_config = generation_config

class AnthropicAgent(LLMAgent):
  def __init__(self, model: str,
               use_cache: bool = False,
               vertexai: bool = False,
               provider: str = 'anthropic',
               api_key_env: str = 'ANTHROPIC_API_KEY',
               api_base_url: str | None = None,
               api_base_url_env: str | None = None,
               **generation_config):
    if vertexai and api_base_url_env is not None:
      raise ValueError("api_base_url_env is not supported with vertexai=True")
    api_base_url = resolve_base_url(api_base_url, api_base_url_env)
    # Determine provider before parent init
    provider = 'vertex_ai' if vertexai else provider
    super().__init__(model=model, provider=provider)

    if vertexai:
      # pip install --upgrade anthropic[vertexai]
      region = os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
      project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
      assert project_id, "GOOGLE_CLOUD_PROJECT environment variable must be set for Vertex AI"
      self.client = anthropic.AnthropicVertex(region=region, project_id=project_id, timeout=TIMEOUT)
      self.async_client = anthropic.AsyncAnthropicVertex(region=region, project_id=project_id, timeout=TIMEOUT)
    else:
      # `api_key_env` / `api_base_url` are the same per-model routing kwargs
      # OpenAIAgent takes (configs/models.yaml `api_key_env:`/`api_base_url:`
      # via core/judge.py). base_url=None keeps the SDK's own default.
      api_key = os.getenv(api_key_env)
      assert api_key, f"{api_key_env} environment variable not set"
      self.client = anthropic.Anthropic(api_key=api_key, base_url=api_base_url or None, timeout=TIMEOUT)
      self.async_client = anthropic.AsyncAnthropic(api_key=api_key, base_url=api_base_url or None, timeout=TIMEOUT)
    
    # Extract cache setting from generation_config
    self.use_cache = use_cache
    self.generation_config = generation_config

  def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
    system = None
    caching_messages = []
    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    # As of June 2025, Anthropic does not support auto-caching but need to define the final block as caching
    for i, message in enumerate(messages):
      if message["role"] == "system":
        system = [dict(type="text", text=message["content"])]
      else:
        new_block = {"role": message["role"]}
        
        # Handle both string content and multimodal content (list)
        if isinstance(message["content"], str):
          # Simple text message
          if self.use_cache and i == len(messages) - 1:
            content = [dict(type="text", text=message['content'], cache_control={"type": "ephemeral"})]
          else:
            content = [dict(type="text", text=message['content'])]
        elif isinstance(message["content"], list):
          # Multimodal message with text and images
          content = []
          for item in message["content"]:
            if item["type"] == "text":
              # Add cache_control to the last text block of the last message (only if caching enabled)
              if self.use_cache and i == len(messages) - 1 and item == message["content"][-1]:
                content.append(dict(type="text", text=item["text"], cache_control={"type": "ephemeral"}))
              else:
                content.append(dict(type="text", text=item["text"]))
            elif item["type"] == "image_url":
              # Convert OpenAI-style image_url to Anthropic format
              image_url = item["image_url"]["url"]
              if image_url.startswith("data:image/"):
                # Extract media type and base64 data
                media_type, base64_data = image_url.split(";base64,")
                media_type = media_type.replace("data:", "")
                content.append({
                  "type": "image",
                  "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data
                  }
                })
        else:
          # Fallback for unexpected content types
          if self.use_cache and i == len(messages) - 1:
            content = [dict(type="text", text=str(message['content']), cache_control={"type": "ephemeral"})]
          else:
            content = [dict(type="text", text=str(message['content']))]
        
        new_block["content"] = content
        caching_messages.append(new_block)
    
    return system, caching_messages

  def _parse_response(self, response):
    """Parse response and extract content, token usage, and cost."""
    text_blocks = [block for block in response.content if hasattr(block, 'text')]
    content = text_blocks[-1].text if text_blocks else None
    usage = response.usage
    cost = self._calculate_cost(response)

    token_usage = TokenUsage(
      input_tokens=usage.input_tokens + (usage.cache_creation_input_tokens or 0),
      output_tokens=usage.output_tokens,
      cached_tokens=usage.cache_read_input_tokens or 0,
      total_tokens=usage.input_tokens + usage.output_tokens,
      cost=cost,
    )
    
    # Convert response to dict for raw logging
    raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
    
    return content, None, token_usage, raw_response

  def _completions(self, messages: list[dict], **call_kwargs) -> LLMResponse:
    system, messages = self._preprocess_messages(messages)
    kwargs = {
      "model": self.model,
      "messages": messages,
      **self.generation_config,
      **call_kwargs,
    }
    if system is not None:
      kwargs["system"] = system

    response = self.client.messages.create(**kwargs)
    content, reasoning_content, token_usage, raw_response = self._parse_response(response)
    self._update_usage(token_usage)

    return LLMResponse(content=content, reasoning_content=reasoning_content, token_usage=token_usage, raw=raw_response)

  async def _async_completions(self, messages: list[dict], **call_kwargs) -> LLMResponse:
    system, messages = self._preprocess_messages(messages)
    kwargs = {
      "model": self.model,
      "messages": messages,
      **self.generation_config,
      **call_kwargs,
    }
    if system is not None:
      kwargs["system"] = system

    response = await self.async_client.messages.create(**kwargs)
    content, reasoning_content, token_usage, raw_response = self._parse_response(response)
    await self._update_usage_async(token_usage)

    return LLMResponse(content=content, reasoning_content=reasoning_content, token_usage=token_usage, raw=raw_response)

class OpenRouterAgent(OpenAIAgent):
  def __init__(self, model: str,
               api_key_env: str = 'OPENROUTER_API_KEY',
               api_base_url: str = 'https://openrouter.ai/api/v1',
               provider: str = 'openrouter',
               api_base_url_env: str | None = None,
               **generation_config):
    super().__init__(model=model,
                     api_key_env=api_key_env,
                     api_base_url=api_base_url,
                     provider=provider,
                     api_base_url_env=api_base_url_env,
                     **generation_config)


class BedrockAgent(LLMAgent):
  """AWS Bedrock provider via boto3 + the Converse API.

  Auth: boto3 reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
  AWS_REGION_NAME from the environment. Cost tracking uses litellm's
  pricing db, keyed by 'bedrock/<modelId>'.
  """
  def __init__(self,
               model: str,
               region_env: str = 'AWS_REGION_NAME',
               provider: str = 'bedrock',
               **generation_config):
    super().__init__(model=model, provider=provider)
    import boto3
    region = os.getenv(region_env, 'us-west-2')
    self.client = boto3.client('bedrock-runtime', region_name=region)
    self.generation_config = generation_config

  def _resize_image_bytes_if_needed(self, raw: bytes, fmt: str,
                                    max_dim: int = 1024,
                                    max_bytes: int = 3_500_000):
    """Downscale image bytes if longest edge > max_dim OR total bytes > max_bytes.
    Returns (new_bytes, new_fmt). Bedrock Converse rejects requests with bodies
    over ~5MB; this clamps well below that. Returns input unchanged on failure.
    """
    if len(raw) <= max_bytes and fmt in ('png', 'jpeg', 'gif', 'webp'):
      # quick check: also probe dimensions to decide
      pass
    try:
      from PIL import Image
      import io
      img = Image.open(io.BytesIO(raw))
      img.load()
      w, h = img.size
      needs_resize = max(w, h) > max_dim or len(raw) > max_bytes
      if not needs_resize:
        return raw, fmt
      if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
      # convert to RGB JPEG for max compression (smallest body)
      if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
      buf = io.BytesIO()
      img.save(buf, format='JPEG', quality=85, optimize=True)
      return buf.getvalue(), 'jpeg'
    except Exception as e:
      print(f"Warning: image resize failed, sending original: {e}")
      return raw, fmt

  def _image_url_to_bedrock(self, url):
    """Convert an OpenAI image_url (data: URI or http(s)) to a Bedrock Converse image block.

    Returns dict {'image': {'format': ..., 'source': {'bytes': ...}}} or None on failure.
    Supports formats: png, jpeg, gif, webp (per Bedrock Converse spec).
    Auto-downsizes large images to stay under Bedrock's request body limit.
    """
    if not url or not isinstance(url, str):
      return None
    try:
      if url.startswith('data:'):
        header, encoded = url.split(',', 1)
        mime = header.split(':', 1)[1].split(';', 1)[0]
        fmt = mime.split('/', 1)[1].lower() if '/' in mime else 'png'
        raw = base64.b64decode(encoded)
      elif url.startswith('http://') or url.startswith('https://'):
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        ctype = r.headers.get('content-type', 'image/png')
        fmt = ctype.split('/', 1)[1].split(';', 1)[0].lower() if '/' in ctype else 'png'
        raw = r.content
      else:
        return None
      if fmt == 'jpg':
        fmt = 'jpeg'
      if fmt not in ('png', 'jpeg', 'gif', 'webp'):
        return None
      raw, fmt = self._resize_image_bytes_if_needed(raw, fmt)
      return {'image': {'format': fmt, 'source': {'bytes': raw}}}
    except Exception as e:
      print(f"Warning: failed to fetch image for Bedrock: {e}")
      return None

  def _preprocess_messages(self, messages: list[dict]):
    """Convert OpenAI-style messages to Bedrock Converse format.
    Returns (converse_messages, system_blocks).
    """
    converse_messages = []
    system_blocks = []
    for msg in messages:
      role = msg['role']
      content = msg['content']
      if isinstance(content, str):
        blocks = [{'text': content}]
      elif isinstance(content, list):
        blocks = []
        for item in content:
          if isinstance(item, dict):
            if item.get('type') == 'text':
              blocks.append({'text': item['text']})
            elif item.get('type') == 'image_url':
              iu = item.get('image_url')
              url = iu.get('url') if isinstance(iu, dict) else iu
              img_block = self._image_url_to_bedrock(url)
              if img_block is not None:
                blocks.append(img_block)
          elif isinstance(item, str):
            blocks.append({'text': item})
        if not blocks:
          blocks = [{'text': ''}]
      else:
        blocks = [{'text': str(content)}]

      if role == 'system':
        system_blocks.extend(blocks)
      else:
        converse_messages.append({'role': role, 'content': blocks})
    return converse_messages, system_blocks

  def _build_inference_config(self) -> dict:
    """Map OpenAI-style generation_config to Bedrock's inferenceConfig."""
    cfg = {}
    if 'max_tokens' in self.generation_config:
      cfg['maxTokens'] = self.generation_config['max_tokens']
    if 'temperature' in self.generation_config:
      cfg['temperature'] = self.generation_config['temperature']
    if 'top_p' in self.generation_config:
      cfg['topP'] = self.generation_config['top_p']
    if 'stop' in self.generation_config:
      cfg['stopSequences'] = self.generation_config['stop']
    return cfg

  def _parse_response(self, response: dict):
    """Parse Bedrock Converse response into our standard format."""
    message = response['output']['message']
    blocks = message.get('content', []) or []
    text_parts = [b['text'] for b in blocks if isinstance(b, dict) and 'text' in b]
    content = ''.join(text_parts) if text_parts else None

    usage = response.get('usage', {}) or {}
    input_tokens = usage.get('inputTokens', 0) or 0
    output_tokens = usage.get('outputTokens', 0) or 0
    total_tokens = usage.get('totalTokens', input_tokens + output_tokens)
    cached_tokens = usage.get('cacheReadInputTokens', 0) or 0

    try:
      in_cost, out_cost = litellm.cost_per_token(
        model=f"bedrock/{self.model}",
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
      )
      cost = in_cost + out_cost
    except Exception as e:
      print(f"Warning: Bedrock cost calc failed for {self.model}: {e}")
      cost = 0.0

    token_usage = TokenUsage(
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      total_tokens=total_tokens,
      cached_tokens=cached_tokens,
      cost=cost,
    )
    return content, None, token_usage, response

  def _completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    converse_messages, system_blocks = self._preprocess_messages(messages)
    inference_config = self._build_inference_config()
    call_kwargs = dict(modelId=self.model, messages=converse_messages, inferenceConfig=inference_config)
    if system_blocks:
      call_kwargs['system'] = system_blocks
    response = self.client.converse(**call_kwargs)
    content, reasoning_content, token_usage, raw_response = self._parse_response(response)
    self._update_usage(token_usage)
    return LLMResponse(content=content, reasoning_content=reasoning_content,
                       token_usage=token_usage, raw=raw_response)

  async def _async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    converse_messages, system_blocks = self._preprocess_messages(messages)
    inference_config = self._build_inference_config()
    call_kwargs = dict(modelId=self.model, messages=converse_messages, inferenceConfig=inference_config)
    if system_blocks:
      call_kwargs['system'] = system_blocks
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: self.client.converse(**call_kwargs))
    content, reasoning_content, token_usage, raw_response = self._parse_response(response)
    await self._update_usage_async(token_usage)
    return LLMResponse(content=content, reasoning_content=reasoning_content,
                       token_usage=token_usage, raw=raw_response)

# =================== Utils ===================
def get_agent_config(model: str, models_config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """
    Load model configuration and format it for llm_agents.
    Expected Config files to be something like this:
    ```
    gpt-5:
      model: openai/gpt-5
      generation_config:
        reasoning_effort: high
    ```
    
    Args:
        model: Model name to load
        models_config_path: Path to models configuration YAML file
        
    Returns:
        Dictionary with agent configuration {model: str, generation_config: dict}
    """
    from core.chat_config import load_chat_config
    cfg = load_chat_config(Path(models_config_path), model)
    gen = dict(cfg['generation_config'])
    if cfg.get('extra_body'):
        gen['extra_body'] = cfg['extra_body']
    return {'model': cfg['factory_model'], 'generation_config': gen,
            **{key: cfg[key] for key in ('api_key_env', 'api_base_url') if cfg.get(key)}}

def sum_token_usage(token_usages: list[TokenUsage]):
  input_tokens = sum(t.input_tokens for t in token_usages)
  output_tokens = sum(t.output_tokens for t in token_usages)
  total_tokens = sum(t.total_tokens for t in token_usages)
  cached_tokens = sum(t.cached_tokens for t in token_usages)
  cost = sum(t.cost for t in token_usages)
  return TokenUsage(input_tokens=input_tokens, 
                    output_tokens=output_tokens, 
                    total_tokens=total_tokens, 
                    cached_tokens=cached_tokens,
                    cost=cost)

def get_max_token_usage(token_usages: list[TokenUsage]):
  return TokenUsage(input_tokens=max(t.input_tokens for t in token_usages), 
                    output_tokens=max(t.output_tokens for t in token_usages), 
                    total_tokens=max(t.total_tokens for t in token_usages), 
                    cached_tokens=max(t.cached_tokens for t in token_usages),
                    cost=max(t.cost for t in token_usages))
