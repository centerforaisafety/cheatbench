# Model and harness configuration

Each model keeps its own frozen harness version. Adding a new model does not
upgrade existing entries. Historical results retain their recorded versions.
Model names are top-level keys; `--model` is required.

```yaml
gpt-6-sol:
  model: openai/gpt-6-sol
  api_key_env: OPENAI_API_KEY
  api_base_url: ${OPENAI_BASE_URL}
  generation_config:
    reasoning_effort: high
  harness:
    name: codex
    version: "0.156.1"
    options:
      reasoning_summary: detailed
```

```bash
python run.py chess --model gpt-6-sol
python run.py chess --model gpt-6-sol --harness terminus-2
python run.py chess --model claude-opus-5-5 --max-turns 50
```

`--harness` overrides the default adapter; model identity, credentials, endpoint,
and shared generation settings stay the same. Options belonging to another
harness are not inherited. Terminus-2 calls the model API directly through Harbor;
it does not wrap Codex or Claude Code. `--agent` is a compatibility alias.

Without `--harness`, the model's frozen harness version is used. An explicit
`--harness codex` uses the latest release, even when Codex is the model's default.
Add `--harness-version 0.156.1` to select an exact version (`--agent-version` is an
alias). Terminus-2 always uses its reviewed Harbor source revision.
Installed versions are recorded with each episode; exact pins are verified
before model execution.

`--max-turns` defaults to 150; `0` omits this cap and leaves native harness defaults in effect. Native counting
units differ (table below). Codex cannot enforce this limit: the runner warns and
records it as unenforced. Episode wall-clock timeouts still apply.

Permissions default to unattended execution inside the task container.
`--permission-mode` is optional and uses the selected adapter's supported values;
unsupported modes fail preflight. Tool and network restrictions remain task settings.

Shared `generation_config` is translated for each consumer. `harness.options`
never enters direct API requests. Judges, review servers and Sycophancy share
API parameter translation and validation. Claude's `thinking` remains a shared
provider setting; effort becomes `output_config.effort` in its Messages API.
Codex's summary option becomes `model_reasoning_summary`; it is not sent to
Chat Completions. Archived configs with old display fields remain readable,
with unsupported fields recorded explicitly.

## Adapter audit (2026-09-24)

| Harness | Shared reasoning effort becomes | Turn limit | Default permission delivery |
|---|---|---|---|
| Claude Code | SDK `effort` | SDK `max_turns` | SDK `bypassPermissions` |
| Codex | `-c model_reasoning_effort=…` | Unsupported | Bypass flag; task tool policy may restrict further |
| Gemini CLI | `thinkingConfig.thinkingLevel` | `model.maxSessionTurns` | `--yolo` |
| Grok Build | `--reasoning-effort` | `--max-turns` | `--always-approve` |
| Kimi Code | `KIMI_MODEL_THINKING_EFFORT` | `loop_control.max_steps_per_turn` | Print mode's automatic permissions |
| Muse Code | `--reasoning-effort` | `--max-model-steps` | `--yolo` |
| DeepSeek Harness | SDK `reasoning_effort` | Adapter counts API requests, including retries | Existing unattended SDK execution |
| Terminus-2 | Harbor model-call settings | Harbor iterations | Unattended terminal execution |

Effort levels remain provider-specific; the same name does not imply an equal
reasoning budget. Gemini's cap is newly wired by this refactor; older runs did
not enforce the YAML `max_turns` value. This does not relabel historical results.

Muse now downloads the exact release from Meta's versioned manifest and checks
its SHA256 and size. It bypasses the updating launcher. The pin is the full
release ID, `1.3.0-R3401.1`; earlier `latest` runs can contain mixed versions.
Grok 4.7 uses Grok Build `1.0.41`; Grok 4.6 retains `1.0.30`.

The audit checked adapter code, upstream references, and downloaded Grok/Muse
binary versions and flags. References: [Codex](https://learn.chatgpt.com/docs/config-file/config-reference),
[Claude](https://platform.claude.com/docs/en/build-with-claude/thinking),
[Gemini](https://geminicli.com/docs/reference/configuration/),
[Grok](https://docs.x.ai/build/cli/reference),
[Kimi](https://moonshotai.github.io/kimi-code/en/reference/kimi-command),
[Muse](https://dev.meta.ai/docs/muse-code/configuration),
[DeepSeek SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/python/sdk/src/deepseek_harness/api.py).

Every run records requested and installed versions, version source, model route,
effective generation settings, permission mode, and whether its turn cap is enforced.
