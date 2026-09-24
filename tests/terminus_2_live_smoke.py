"""Opt-in paid route smoke through the normal task runner; not a benchmark."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from core import sandbox, trial
from core.agents import make_agent
from run import load_config


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-sol-chat")
    parser.add_argument(
        "--models-config", type=Path, default=ROOT / "configs/models.yaml"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    cfg = load_config(args.models_config, args.model)
    agent = make_agent(
        "terminus-2",
        model=cfg["model"],
        api_key_env=cfg["api_key_env"],
        api_base_url=cfg["api_base_url"],
        extra_body=cfg["extra_body"],
        generation_config=cfg["generation_config"],
        max_turns=10,
    )
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    if problems := agent.setup():
        raise RuntimeError(problems)
    task_dir = args.output_dir / "fixture"
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text(
        "Calculate 6 * 7 using the terminal. Call the available MCP verifier's verify tool "
        "with the integer answer. Once it confirms the answer, finish the task."
    )
    (task_dir / "grader.py").write_text(
        (ROOT / "tests/terminus_2_probe_grader.py").read_text()
    )
    task = trial.Task(
        root=task_dir,
        name="terminus2-smoke",
        timeout_s=180,
        prompts={"instruction": "instruction.md"},
        grader="grader.py",
    )
    key = os.environ[agent.api_key_env]
    sandbox.ensure_path()
    record = await trial.run_trial(
        task,
        {"id": "smoke", "input": "", "files": {}},
        agent,
        image=sandbox.DEFAULT_IMAGE,
        api_key=key,
        out_dir=args.output_dir,
        env=agent.routed_env({}),
    )
    serialized = json.dumps(record)
    if key in serialized:
        raise AssertionError("credential present in exported record")
    (args.output_dir / "episode.json").write_text(serialized + "\n")
    state = record.get("grader_state") or {}
    summary = {
        k: record.get(k)
        for k in (
            "ok",
            "error",
            "n_turns",
            "n_tool_calls",
            "cost_usd",
            "cost_source",
            "harbor_commit",
        )
    }
    summary["verified_answer"] = state.get("answer")
    summary["observed_before_call"] = state.get("observed_before_call")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert record["ok"] and state.get("answer") == 42, summary
    assert state.get("observed_before_call"), summary


if __name__ == "__main__":
    asyncio.run(main())
