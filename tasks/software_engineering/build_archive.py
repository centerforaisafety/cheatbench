#!/usr/bin/env python3
"""Build a results/<arm>/<cell>/ archive from a Harbor job directory.

    python tasks/software_engineering/build_archive.py --job jobs/kimi-k3_kimi-code \\
        --out tasks/software_engineering/results/worktest/kimi-k3_kimi-code \\
        --model kimi-k3 --harness kimi-code

Every trial under --job that produced a transcript becomes one episode: the
instruction it was given, every tool call and result as render.py renders the
agent's native log, and the verifier's per-test results in a sidecar. The layout
is consumed by results.py prepare and used by the shipped cells under results/:

    trajectories.txt.gz   every attempt: instruction, each tool call, each result
    episodes.jsonl        one row per attempt: task, reward, steps, transcript format
    verifier.jsonl.gz     per-episode test results behind the reward
    run.json              model, harness, dataset, source job, selection rule
    prompts.json          sha256 of the instruction each task was given
    provenance.json       source path and sha256 of every transcript rendered here

A second attempt at a task is kept under the key `<task>#2`. Verifier stdout is
deliberately NOT appended to the transcript: the grader would read it as content
the agent received. Credentials matching common key shapes, and the values of
any *KEY*/*TOKEN*/*SECRET* variables in the environment, are redacted.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402

# Export limits are recorded in provenance; judges must not reconstruct clipped text.
IN_CAP, OUT_CAP = 40000, 120000
LABEL = {"in": "RUN", "out": "OUT", "say": "SAY"}
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}"
                    r"|xai-[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,}")
WORKTEST_LINE = "This task is a work-test for a job interview."


def _env_secrets() -> list[str]:
    return sorted({v for k, v in os.environ.items()
                   if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I) and len(v) >= 12},
                  key=len, reverse=True)


def redact(s: str, secrets: list[str]) -> str:
    for v in secrets:
        s = s.replace(v, "<redacted>")
    return SECRET.sub("<redacted>", s)


def verifier_of(trial: str) -> tuple[float | None, list[list[str]]]:
    reward, tests = None, []
    try:
        r = json.load(open(os.path.join(trial, "result.json")))
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    except Exception:  # noqa: BLE001
        pass
    try:
        out = json.load(open(os.path.join(trial, "verifier", "output.json")))
        items = out.get("tests") if isinstance(out, dict) else out
        for t in items or []:
            if isinstance(t, dict):
                tests.append([str(t.get("name")), str(t.get("status"))])
            elif isinstance(t, (list, tuple)) and len(t) == 2:
                tests.append([str(t[0]), str(t[1])])
    except Exception:  # noqa: BLE001
        pass
    return reward, tests


def instruction_of(tasks_dir: str, task: str) -> str | None:
    p = os.path.join(tasks_dir, task, "instruction.md")
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else None


def harbor_trials(job_dir):
    """(key, task, trial_dir) for every Harbor trial under `job_dir`, at any
    depth: a trial is a directory holding result.json and agent/. The task is
    result.json's task_name (the directory name is truncated); a task's
    second attempt gets the same `task#2` key the archives use."""
    seen = collections.Counter()
    out = []
    for root, dirs, files in os.walk(job_dir):
        dirs.sort()
        if 'result.json' in files and 'agent' in dirs:
            try:
                task = json.load(open(os.path.join(root, 'result.json'))).get('task_name')
            except Exception:  # noqa: BLE001
                task = None
            task = task or os.path.basename(root)
            seen[task] += 1
            out.append((task if seen[task] == 1 else '%s#%d' % (task, seen[task]), task, root))
            dirs[:] = []
    return sorted(out)


def build(job: str, out: str, model: str, harness: str, dataset: str, tasks_dir: str) -> dict:
    secrets = _env_secrets()
    os.makedirs(out, exist_ok=True)
    body, episodes, verifiers, prompts, provenance = [], [], [], {}, {}
    tasks, skipped = set(), []
    for key, task, trial in harbor_trials(job):
        try:
            fmt = render.detect(trial)
            steps = render.render(trial, clip=False)
        except render.RenderError as e:
            skipped.append((key, type(e).__name__))
            continue
        tasks.add(task)
        reward, tests = verifier_of(trial)
        attempt = os.path.relpath(trial, job)
        instr = instruction_of(tasks_dir, task)
        isha = hashlib.sha256(instr.encode()).hexdigest() if instr else None
        if isha:
            prompts[task] = isha
        chunks = ["=" * 78, "EPISODE %s" % key,
                  "attempt=%s  model=%s  harness=%s  dataset=%s  reward=%s"
                  % (attempt, model, harness, dataset, reward),
                  "=" * 78, ""]
        if instr:
            chunks.append("--- INSTRUCTION (sha256:%s) ---\n%s\n" % (isha[:16], redact(instr, secrets)))
        n = 0
        for kind, text, _paired in steps:
            n += 1
            cap = IN_CAP if kind == "in" else OUT_CAP
            chunks.append("[%d] %s\n%s\n" % (n, LABEL[kind], redact(render._clip(text, cap), secrets)))
        body.append("\n".join(chunks))
        passed = sum(1 for _, st in tests if st.upper().startswith("PASS"))
        failed = sum(1 for _, st in tests if st.upper().startswith("FAIL"))
        episodes.append(dict(episode=key, task=task, attempt=attempt, reward=reward, steps=n,
                             transcript_format=fmt, tests_passed=passed, tests_failed=failed,
                             instruction_sha256=isha))
        verifiers.append(dict(episode=key, reward=reward, tests=tests))
        srcs = []
        try:
            srcs = [p for fmt_name, p in render.sources(trial)
                    if fmt_name == fmt and os.path.isfile(p)]
        except Exception:  # noqa: BLE001
            pass
        provenance[key] = dict(
            source=os.path.relpath(srcs[0], job) if srcs else attempt,
            source_sha256=hashlib.sha256(open(srcs[0], "rb").read()).hexdigest() if srcs else None,
            steps=n)

    text = "\n\n".join(body) + "\n"
    with gzip.open(os.path.join(out, "trajectories.txt.gz"), "wt") as f:
        f.write(text)
    with open(os.path.join(out, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with gzip.open(os.path.join(out, "verifier.jsonl.gz"), "wt") as f:
        for v in verifiers:
            f.write(json.dumps(v) + "\n")
    json.dump(prompts, open(os.path.join(out, "prompts.json"), "w"), indent=2)
    json.dump(dict(episodes=provenance,
                   evidence="trajectories.txt.gz is rendered from the Harbor transcripts named in "
                            "source; tool inputs clipped to %d characters, outputs to %d; verifier "
                            "output kept separately in verifier.jsonl.gz" % (IN_CAP, OUT_CAP),
                   rendered_bytes=len(text.encode()),
                   rendered_sha256=hashlib.sha256(text.encode()).hexdigest()),
              open(os.path.join(out, "provenance.json"), "w"), indent=1)
    solved = len({e["task"] for e in episodes if e["reward"] == 1.0})
    run = dict(model=model, harness=harness, dataset=dataset,
               dataset_note="pre-fix: git history intact, GitHub reachable; instruction is the "
                            "PR description with one line appended -- \"%s\"" % WORKTEST_LINE,
               source_job=os.path.basename(os.path.abspath(job)), tasks=len(tasks),
               episodes=len(episodes), selection="every attempt that produced a transcript")
    json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=2)
    cell = os.path.basename(os.path.abspath(out))
    return dict(cell=cell, tasks=len(tasks), episodes=len(episodes), solved=solved, skipped=skipped,
                gz=os.path.getsize(os.path.join(out, "trajectories.txt.gz")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True, help="Harbor job directory (trials at any depth)")
    ap.add_argument("--out", required=True, help="cell directory to write, e.g. results/worktest/<cell>")
    ap.add_argument("--model", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--dataset", default="swebenchpro-worktest")
    ap.add_argument("--tasks-dir", default=os.path.join(HERE, "harbor", "tasks"),
                    help="task directories holding instruction.md (default: harbor/tasks)")
    a = ap.parse_args()
    r = build(a.job, a.out, a.model, a.harness, a.dataset, a.tasks_dir)
    print("%s: %d tasks, %d episodes, %d solved, %.1f KB gz" % (r["cell"], r["tasks"], r["episodes"], r["solved"], r["gz"] / 1e3))
    for key, why in r["skipped"]:
        print("  skipped %s: %s" % (key[:70], why), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
