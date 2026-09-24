"""Harbor's pinned Terminus-2, running against our episode container.

Harbor is installed with its frozen dependency lock. The local image extension
adds view_images observations to the JSON command loop; execution and context
management remain upstream. See docs/terminus-2.md for provenance.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from .installed import InstalledAgent
from ..trajectory import SCHEMA_VERSION, Trajectory

HARBOR_COMMIT = "4641fd590f190411c57e1cbd824a9f6786cbe92e"
HARBOR_ARCHIVE_SHA256 = (
    "ca4079f8189f290134c8619c66a131b21a598893e4d454cceb64a8f53b8cfc79"
)
RUNTIME = "/opt/terminus-runtime"


class Terminus2Agent(InstalledAgent):
    TURN_LIMIT = 'Harbor agent iterations'
    FIXED_VERSION = HARBOR_COMMIT
    # Debian terminal packages and uv's managed-Python release assets.
    INSTALL_HOSTS = (
        "deb.debian.org",
        "security.debian.org",
        "release-assets.githubusercontent.com",
    )
    API_KEY_ENV = "OPENAI_API_KEY"
    CLI_BASE_URL_ENV = ("OPENAI_BASE_URL",)
    PASSTHROUGH_ENV = ("OPENAI_BASE_URL",)
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}
    USAGE_FIELDS = ("usage",)
    USAGE_KEYS = ("input", "output", "cached", "cache_write")
    PROMPT_TOKENS_INCLUDE_CACHE = True
    GENERATION_KEYS = (
        "temperature",
        "reasoning_effort",
        "max_thinking_tokens",
        "model_info",
        "use_responses_api",
        "thinking",
        "output_config",
        "max_tokens",
    )
    EXTRA_RECORD_KEYS = (
        "harbor_commit",
        "harbor_trajectory",
        "harbor_artifacts",
        "harbor_context",
        "harbor_options",
        "harbor_cost_usd",
    )
    RUNNER = Path(__file__).with_name("terminus_2_runner.py")
    VERSION_CMD = f"cat {RUNTIME}/commit"

    @staticmethod
    def name():
        return "terminus-2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.requested_version and self.requested_version != HARBOR_COMMIT:
            raise ValueError(
                f"terminus-2 is pinned to Harbor {HARBOR_COMMIT}; "
                "update the reviewed source pin to change versions"
            )
        self.requested_version = self.pinned_version = HARBOR_COMMIT

    def _tools_for(self, policy):
        self.mcp_only = bool(policy.get("mcp_only"))
        enabled = [key for key in ("web_search", "web_fetch") if policy.get(key)]
        if enabled:
            raise ValueError(
                f"terminus-2 has no provider web tools; cannot enable {enabled}"
            )
        if self.mcp_only:
            return []
        # T2 has no provider-side web tools. Shell networking is governed by
        # the task's existing network namespace, exactly as for native agents.
        return ["bash_command", "view_images", "mark_task_complete"]

    def reported_cost_usd(self, raw):
        # Harbor prices usage with LiteLLM; this is NOT a provider invoice.
        return None

    def estimated_cost_usd(self, raw):
        return (
            raw.get("harbor_cost_usd")
            if raw.get("harbor_cost_usd") is not None
            else super().estimated_cost_usd(raw)
        )

    def routing_problems(self):
        # extra_body goes through Harbor's documented llm_call_kwargs, without
        # an HTTP forwarder. Do not advertise SUPPORTS_FORWARDER=True.
        return ["terminus-2 uses api_base_url, not base_url"] if self.base_url else []

    def api_model(self):
        from ..model_settings import factory_model
        return factory_model(self.model, self.model,
                             {"api_base_url": self.resolved_base_url()}, "terminus-2")

    def resolved_routing(self):
        out = super().resolved_routing()
        out["api_model_id"] = self.api_model()
        return out

    def setup(self):
        problems = self.config_problems()
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
        if self.permission_mode != "bypassPermissions":
            problems.append(
                "terminus-2 executes terminal commands without permission prompts; "
                "use permission_mode: bypassPermissions"
            )
        try:
            self.api_model()
        except SystemExit as exc:
            problems.append(str(exc))
        return problems

    def install_check(self):
        return (
            f'test "$(cat {RUNTIME}/commit 2>/dev/null)" = {HARBOR_COMMIT} '
            "&& tmux -V >/dev/null && asciinema --version >/dev/null "
            f"&& {RUNTIME}/.venv/bin/python -c "
            + shlex.quote(
                "from harbor.agents.terminus_2.terminus_2 import Terminus2; import PIL; assert PIL.__version__ == '11.3.0'"
            )
        )

    def install_script(self):
        # The source archive contains uv.lock; no floating Harbor or LiteLLM
        # dependency resolution takes place when an episode installs this.
        return "\n".join(
            [
                "set -eu",
                f"mkdir -p {RUNTIME}",
                't2_archive="$(mktemp)"',
                't2_source="$(mktemp -d)"',
                'trap \'rm -f "$t2_archive"; rm -rf "$t2_source"\' EXIT',
                f'curl -fsSL https://codeload.github.com/harbor-framework/harbor/tar.gz/{HARBOR_COMMIT} -o "$t2_archive"',
                f'echo "{HARBOR_ARCHIVE_SHA256}  $t2_archive" | sha256sum -c -',
                'tar -xzf "$t2_archive" --strip-components=1 -C "$t2_source"',
                # Install tooling separately; never upgrade the task's libraries.
                f"python -m venv {RUNTIME}/installer",
                f"{RUNTIME}/installer/bin/pip install --disable-pip-version-check uv==0.9.9",
                # apt installation checks uid 0, which this sandbox deliberately
                # lacks. Download and unpack only the terminal/runtime packages;
                # the episode owns its disposable root filesystem.
                "apt-get -o APT::Sandbox::User=user update",
                f"mkdir -p {RUNTIME}/terminal-debs",
                f"cd {RUNTIME}/terminal-debs",
                "apt-get -o APT::Sandbox::User=user download tmux $(apt-cache depends tmux | awk '/Depends: lib/ && $2 != \"libc6\" {print $2}')",
                'for t2_deb in ./*.deb; do dpkg-deb -x "$t2_deb" /; done',
                f"{RUNTIME}/installer/bin/pip install --disable-pip-version-check asciinema==2.4.0",
                f"ln -sf {RUNTIME}/installer/bin/asciinema /usr/local/bin/asciinema",
                "tmux -V",
                "asciinema --version",
                'cd "$t2_source"',
                f"UV_PROJECT_ENVIRONMENT={RUNTIME}/.venv {RUNTIME}/installer/bin/uv sync --frozen --no-dev --no-editable --package harbor --python 3.12.11",
                f"{RUNTIME}/installer/bin/uv pip install --python {RUNTIME}/.venv/bin/python --no-deps pillow==11.3.0",
                f"cp uv.lock {RUNTIME}/uv.lock",
                f"echo {HARBOR_COMMIT} > {RUNTIME}/commit",
            ]
        )

    def blob(
        self,
        *,
        row,
        prompt,
        files,
        skip_dirs,
        grader_src="",
        stage_src,
        timeout_s,
        env,
        install=None,
        task_install=None,
    ):
        if isinstance(prompt, list):
            if any(p.get("type") != "text" for p in prompt):
                raise ValueError(
                    "inline image input is unsupported; stage image files for view_images"
                )
            prompt = "\n".join(p.get("text", "") for p in prompt)
        if not isinstance(prompt, str):
            raise ValueError("terminus-2 requires a text prompt")
        task = dict(
            id=row["id"],
            model=self.api_model(),
            content=prompt,
            files=files,
            skip_dirs=list(skip_dirs),
            deliverable_files=list(row.get("deliverable_files") or []),
            tools=self.tools,
            terminus_2={"mcp_only": self.mcp_only},
            timeout_s=timeout_s,
            env=dict(env or {}),
            install=install,
            task_install=task_install,
            row=row if grader_src else {},
            runtime=RUNTIME,
            harbor_commit=HARBOR_COMMIT,
            api_key_env=self.API_KEY_ENV,
            api_base=self.resolved_base_url(),
            extra_body=self.extra_body,
            generation_config=self.generation_config,
            max_turns=self.max_turns,
        )
        return json.dumps(
            {
                "task": task,
                "code": self.RUNNER.read_text(),
                "modules": {
                    "stage": stage_src,
                    "grader": grader_src,
                    "runner": self.RUNNER.read_text(),
                    "support": self.RUNNER.with_name("runner_support.py").read_text(),
                    "bridge": self.RUNNER.with_name("terminus_2_bridge.py").read_text(),
                    "images": self.RUNNER.with_name("terminus_2_images.py").read_text(),
                    "mcp": self.RUNNER.with_name("terminus_2_mcp.py").read_text(),
                },
            }
        ).encode()

    def to_trajectory(self, raw):
        native = raw.get("harbor_trajectory")
        if not native:
            # The common historical transcript stores only native `messages`,
            # so the complete Harbor document travels there in an envelope.
            native = next(
                (
                    m["trajectory"]
                    for m in raw.get("messages") or []
                    if m.get("type") == "harbor_trajectory"
                ),
                None,
            )
        if native:
            return Trajectory.model_validate(native)
        if raw.get("messages"):
            # Live tails carry individual native ATIF steps. Renumber just the
            # envelope; retain their tool IDs, arguments and observations.
            steps = [dict(s, step_id=i) for i, s in enumerate(raw["messages"], 1)]
            return Trajectory(
                schema_version=SCHEMA_VERSION, agent=self.atif_agent(), steps=steps
            )
        return Trajectory(
            schema_version=SCHEMA_VERSION,
            agent=self.atif_agent(),
            steps=[
                self.empty_step(raw.get("error") or "No Harbor trajectory was produced")
            ],
        )
