# Run with Harbor

See the [implicit honesty expectation](../README.md)
for the assessment context and the boundary between permitted work and cheating.

SWE-bench Pro uses Harbor to run its Docker environments and correctness tests.
Requires **Python 3.12+, Docker, Git, and `patch`**. Run commands from the
CheatBench repository root.

## Setup

Install Harbor in a Python virtual environment and apply the included adapter fixes:

```bash
pip install harbor==0.23.0
patch -p1 -d "$(python -c 'import harbor,os;print(os.path.dirname(os.path.dirname(harbor.__file__)))')" \
    < tasks/software_engineering/harbor/harbor-0.23.0.patch
```

## Configuration

Choose a job from [configs/](configs/). These files configure the model, agent,
resources, and timeouts. [acp/](acp/) defines the Gemini and DeepSeek integrations.
Harbor execution uses these settings independently of CheatBench's common runner.

| Agent | Credential |
|---|---|
| Claude Code | `ANTHROPIC_API_KEY` |
| Codex | `OPENAI_API_KEY` |
| Gemini CLI | `--ae GEMINI_API_KEY=$GEMINI_API_KEY` |
| Kimi Code | `--ae KIMI_MODEL_API_KEY=$OPENROUTER_API_KEY` |
| DeepSeek Harness | `--ae DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY` |
| Muse Code | `--ae META_API_KEY=$META_API_KEY` |
| Grok Build | `OPENROUTER_API_KEY` |

Gemini and DeepSeek versions are pinned in `acp/`. For other agents, set a
supported `version` argument when you need a fixed CLI version.

## Run

For example, run all 20 tasks with Kimi:

```bash
harbor run -c tasks/software_engineering/harbor/configs/kimi-k3_kimi-code.json \
    --ae KIMI_MODEL_API_KEY=$OPENROUTER_API_KEY
```

The default is four concurrent trials, each with 8 CPUs and 16 GiB of memory.
Use `--n-concurrent 1` to run sequentially. Agent and verifier phases each have a
6,000-second limit; image builds and installation take additional time.
Output is saved under `jobs/<job_name>/`.

## Judge

Export a completed job, then follow the [behavioral judging instructions](../README.md#judge-behavior):

```bash
python tasks/software_engineering/build_archive.py --job jobs/kimi-k3_kimi-code \
    --out outputs/swe_archives/kimi-k3_kimi-code --model kimi-k3 --harness kimi-code
```

## Task definitions

[tasks/](tasks/) contains the 20 environments pinned in
[instances.json](../instances.json). [overrides/](overrides/) contains their
build and prompt adjustments; [registry.json](registry.json) lists them for
Harbor's remote registry interface.

To verify or regenerate the definitions from the pinned upstream dataset:

```bash
python tasks/software_engineering/harbor/make_dataset.py --check
python tasks/software_engineering/harbor/make_dataset.py
```

## Compatibility

- The Alpine 3.18 `protonmail/webclients` image cannot load the Node 22 builds used
  by Kimi Code and DeepSeek Harness. Record these as infrastructure failures.
- Muse Code requires Meta developer access. Set `MUSE_CODE_INSTALL_SCRIPT` to use
  a local installer when `dev.meta.ai` is unavailable.
