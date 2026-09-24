# The sandbox image every episode runs in, and the single source of truth for
# what is in it. `core/sandbox/build.py` reads THIS FILE and builds from it --
# with `docker build` where a daemon is reachable, and otherwise by replaying
# the same steps through enroot and exporting a squashfs. There is no second
# description of the image anywhere.
#
# What this is: the TASK ENVIRONMENT, and nothing else. An interpreter and the
# libraries a task's work needs. From inside it is an ordinary Python container.
#
# What this is NOT:
#
#   no harness source, no prompts, no judge rubric, no answer keys, no dataset.
#   The runner, the staging module, the task's grader and the row are streamed
#   in on stdin at exec time (core/trial.py) and never touch the container
#   filesystem, so an agent that runs `ls /`, `find / -name '*.md'` or reads
#   /proc finds nothing that identifies this eval.
#
#   no agent, and nothing the agent is made of. Baking
#   `npm install -g @anthropic-ai/claude-code` in welded one vendor to the
#   sandbox and would mean a second image the day we run Codex. The same goes
#   for the vendor's Python SDK -- an agent running `pip list` should not find
#   the harness's own `claude-agent-sdk` sitting in its environment -- and for
#   Node, which is the CLI's dependency and not the environment's. All three
#   are installed at episode start by the adapter (core/agents/installed.py:
#   install_check / install_script), which is why only `curl` is here.
#
# Anything an adapter installs may also be pre-baked here as a pure CACHE: the
# adapter's check succeeds, its install never runs, and the episode starts
# exactly as before. So this split is a performance choice, never a correctness
# one. Measured cold start for all three: 33 s per episode, against a 900-3600 s
# episode budget (Node 22 via nvm 12.6 s, the SDK pins ~15 s, the CLI 4.9 s --
# and Node 22 makes the CLI install faster than the Node 20 that used to be
# here, 4.9 s against 10.0 s, so moving Node out costs ~7 s net).
#
# Anonymity is a regression test, not a convention: tests/test_image_anonymity.py.

FROM python:3.11-slim

# curl/git: the network and VCS tools an ordinary Python box has. curl is also
# what the agent adapter fetches nvm with -- a general-purpose tool, unlike a
# Node runtime, which would say something about what runs here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Task-side libraries only: what a row's work plausibly needs. Installed by name
# rather than from a requirements.txt of ours, so no file of ours -- and none of
# its comments about this eval -- lands in the image.
RUN pip install --no-cache-dir \
        numpy \
        sympy \
        mpmath \
        matplotlib \
        pillow==12.3.0 \
        requests==2.34.2 \
        pyyaml==6.0.3

# A neutral, non-identifying working directory for the agent.
RUN mkdir -p /workspace
WORKDIR /workspace

# No ENTRYPOINT and no CMD: the runner is streamed in on stdin at exec time.
# No ENV either -- PYTHONUNBUFFERED is passed per episode by core/sandbox, and
# ANTHROPIC_API_KEY is passed by name so it never appears in a command line.
