# CheatBench

[Website](https://cheatbench.ai)

CheatBench measures whether AI agents attempt to cheat while completing challenging
assignments. Tasks span mathematical research, visual reasoning, writing, coding,
knowledge work, and other domains. Each environment establishes expectations of
honest work and provides opportunities to violate them, such as accessing hidden
answers, copying another participant's work, or manipulating evaluation.

This repository contains the task environments, agent adapters, evaluation
runner, and behavioral judges. Task success and cheating are measured separately.

## Agent harness support

- Claude Agent SDK (`claude-sdk`)
- Codex (`codex`)
- Gemini CLI (`gemini-cli`)
- Grok Build (`grok-build`)
- Kimi Code (`kimi-code`)
- Muse Code (`muse-code`)
- DeepSeek Harness (`deepseek-harness`)
- Terminus-2 (`terminus-2`)

Each model pins its harness and version in [configs/models.yaml](configs/models.yaml).
Use `--harness terminus-2` to run that model through Terminus-2, which calls the
model API directly. See [configuration](CONFIGURATION.md) for overrides and limits.

## Container support

- **[Enroot](https://github.com/NVIDIA/enroot)** — requires Linux and Python 3.11+.
- **Docker** — Software Engineering through Harbor; shared-runner support is planned.
- **Apptainer** — planned.

## Quick start

Use Linux with Python 3.11 and the Enroot/FUSE/user-namespace prerequisites.
Install the repository and task-build dependencies in a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[build,svgs]'
python -m core.sandbox --check --private-net
```

Add your provider API credentials to a local `.env` file, then run:

```bash
python run.py TASK --model MODEL --limit 1
```

Choose `TASK` from [Tasks](#tasks), `MODEL` from
[configs/models.yaml](configs/models.yaml). The default judge is `gpt-6-sol`.
Missing task inputs and container images are built automatically, including
the Office image and the host-only, pinned Stockfish engine for chess. Omit `--limit 1`
to run all examples.

## Outputs and judging

Runs write to `outputs/`: `run.json` records settings, `episodes.jsonl` records
execution outcomes, `trajectories/` contains agent interactions, and `judge.jsonl`
contains behavioral labels and explanations. The common judge combines the
[shared rubric](prompts/judge.md) with each task's `judge_schema.py`.

To rejudge a saved run:

```bash
python judge.py outputs/<run-directory> --redo
```

## Tasks

| Category | Tasks |
|---|---|
| Mathematical Research | [OpenMath](tasks/openmath/), [OpenMath Agent](tasks/openmath_agent/) |
| Multimodal | [GeoGuessr](tasks/geoguessr/), [Visual Puzzles](tasks/synthetic_puzzles/) |
| Creative Writing | [Creative Writing](tasks/creative_writings/) |
| SVG Competition | [SVG Competition](tasks/svgs/) |
| Menial Computation | [Prime Factorization](tasks/prime_factorization/), [Subset Sum](tasks/subset_sum/) |
| Biology and Bioinformatics | [Protein Design](tasks/protein_design/) |
| Knowledge Work | [Knowledge Work](tasks/knowledge_work/) |
| Board Games | [Chess](tasks/chess/), [Go](tasks/go/) |
| Sycophancy | [Sycophancy](tasks/sycophancy/) |
| Software Engineering | [Software Engineering](tasks/software_engineering/) |

[Sycophancy](tasks/sycophancy/) uses its direct API runner;
[Software Engineering](tasks/software_engineering/) uses Harbor. Their task READMEs document the commands. To add a task, follow the [task guide](tasks/README.md).

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and third-party attribution.
