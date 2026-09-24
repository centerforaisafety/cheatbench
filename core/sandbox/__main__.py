"""OURS: no Harbor counterpart -- a shell entry point onto our own build/audit/run
primitives. Harbor's nearest CLI, `harbor exec`, drives agents over ad-hoc
tasks; it does not put you inside the container the way `--build`/`--audit` do.

`python -m core.sandbox` -- the sandbox from a shell.

    python -m core.sandbox --build            build the image if it is missing
    python -m core.sandbox --build --rebuild  build it again regardless
    python -m core.sandbox --build --audit    run the full anonymity probe set
    python -m core.sandbox --check            preflight only
    python -m core.sandbox 'ls -la /'         run a command as the agent would

None of it is a required step. A run builds the image itself when it is absent
(core/sandbox/build.py: ensure_image), so this exists for first-time setup and
for looking at what the agent sees.
"""
from __future__ import annotations

import sys

from . import build, paths
from .runtime import preflight, run_shell


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m core.sandbox",
        description="build, audit, or run a command in the eval sandbox")
    ap.add_argument("--image", default="", help=f"default {paths.DEFAULT_IMAGE}")
    ap.add_argument("--build", action="store_true",
                    help="build the image if missing, then exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="with --build: build even if it is already there")
    ap.add_argument("--audit", action="store_true",
                    help="run the full anonymity probe set and exit")
    ap.add_argument("--private-net", action="store_true")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("command", nargs="?", default="")
    args = ap.parse_args(argv)

    image = args.image or paths.DEFAULT_IMAGE
    if args.build or args.rebuild:
        if not build.ensure_image(image, rebuild=args.rebuild):
            print(f"{image} is already built (--rebuild to force)")
        return 0

    problems = preflight(private_net=args.private_net)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    if args.check:
        print("preflight ok")
        return 0

    if args.audit:
        try:
            build.audit_built_image(image, probes=None)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            print("FAIL - image leaks eval information; do NOT run experiments "
                  "with it", file=sys.stderr)
            return 1
        print("PASS - image reveals nothing about the eval")
        return 0

    res = run_shell(args.command, image, private_net=args.private_net)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
