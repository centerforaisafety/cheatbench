"""The OpenAI Codex CLI adapter.

One adapter, one module -- Harbor's layout
(`harbor/src/harbor/agents/installed/codex.py`, which this is ported from). Its
in-container runner is the sibling `codex_runner.py`, which is read as source
and streamed into the container on stdin; it is never imported here and never
written to the container filesystem.
"""
from __future__ import annotations

import json
import os
import shlex
from datetime import datetime
from pathlib import Path

from ..trajectory import (
    SCHEMA_VERSION,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from ..trajectory import Agent as ATIFAgent
from .installed import NODE_MAJOR, NVM_PRELUDE, InstalledAgent, link_bins, node_install


class CodexAgent(InstalledAgent):
    """OpenAI's Codex CLI, driven by `codex_runner.py`.

    Ported from Harbor's `harbor/agents/installed/codex.py`. Everything about
    how Codex is installed, invoked and read back is Harbor's, reshaped to the
    two things this sandbox does differently:

      * `install()` RETURNS a check/install pair instead of executing one.
        Harbor can `exec` into a live environment as many times as it likes; an
        episode here is ONE `enroot start` whose overlay dies with it and which
        has no exec channel, so the pair rides in on stdin and the runner runs
        `check || install` itself.
      * the trajectory is not copied out of the container. Harbor copies
        `$CODEX_HOME/sessions` to the host and parses it there. There is no bind
        mount here, so the runner READS the rollout file and returns its lines
        inside the one JSON line it already owes us.
      * the task's grader is hosted BY THE RUNNER, not by the host. Harbor's
        `mcp_servers` are services the task brought with it and it only writes
        their addresses out (`command`/`args` for stdio, `{"url": ...}` for
        everything else -- `codex.py:1269-1283`). We have no such service, so
        `codex_runner.serve_grader` starts one inside the container, on
        loopback, and the config file gets the url shape and never the stdio
        one. See `blob()` for why that distinction is the whole point.

    Codex does not stream its record on stdout. `--json` prints events for a
    human, but the file of record is
    `$CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<ISO>-<uuid>.jsonl`, which is
    what Harbor parses and what `render_trajectory` below renders.
    """

    @staticmethod
    def name() -> str:
        return "codex"

    # ---- what this adapter can hold shut ----------------------------------
    #
    # BOTH keys, because in Codex they are ONE TOOL.
    #
    # Established empirically against codex-cli 0.152.0 by asking the model to
    # enumerate its own tools under `--dangerously-bypass-approvals-and-sandbox`
    # (consistent across four runs), NOT inferred from Harbor:
    #
    #   flag omitted            view_image, image_gen__imagegen, web__run,
    #                           collaboration.*
    #   -c web_search=disabled  functions.wait, request_user_input, exec,
    #                           collaboration.*          <- web__run is GONE
    #
    # `web__run` is a single tool whose operations are `search_query`, `open`
    # ("Open/fetch a specific URL"), `click`, `find` and `screenshot`. Search and
    # fetch are the same tool, so the one knob answers both keys, and a task
    # declaring `web_fetch: false` is genuinely satisfied rather than merely
    # unenforced. This is the canary-style empirical check the table at the top
    # of this file demands before a key may enter ENFORCEABLE.
    #
    # Because it is one knob, the two keys cannot DISAGREE. `_tools_for` refuses
    # a task that asks for one without the other rather than quietly picking a
    # side -- silently disabling search a task asked for is as wrong a
    # measurement as silently allowing a fetch it forbade.
    #
    # The shell is not a policy key at all: `--dangerously-bypass-approvals-and-
    # sandbox` is hardcoded in Harbor's invocation and there is no knob for it.
    # That is fine -- the shell runs INSIDE the container's network namespace,
    # which is exactly the tool this sandbox does contain.
    ENFORCEABLE = {"web_search", "web_fetch", "mcp_only"}

    API_KEY_ENV = "OPENAI_API_KEY"
    # Per-model routing (configs/models.yaml `api_key_env:` / `api_base_url:` /
    # `extra_body:`, see core/agents/base.py). The CLI reads its key from
    # OPENAI_API_KEY (and auth.json, which the runner writes from it) whatever
    # host variable the entry named; the base URL goes into config.toml as
    # `openai_base_url` (read in `blob()`) and is ALSO exported as
    # OPENAI_BASE_URL for the task's grader running beside the CLI. With
    # `extra_body` the runner points config.toml at core/agents/forwarder.py
    # instead, which merges the body and forwards to the real URL.
    CLI_KEY_ENV = ("OPENAI_API_KEY",)
    CLI_BASE_URL_ENV = ("OPENAI_BASE_URL",)
    SUPPORTS_FORWARDER = True
    # The exact `codex exec` line the episode ran, and the rollout it was read
    # back from. A Codex result that could not say which flags produced it --
    # `-c web_search=disabled` above all -- would not be a reproducible result.
    EXTRA_RECORD_KEYS = ("codex_command", "rollout_path", "forwarder_calls")

    # ---- what an episode cost ---------------------------------------------
    #
    # Codex prices NOTHING. The rollout's `token_count` events carry counts and
    # never a dollar figure, so `reported_cost_usd` returning None here is the
    # adapter's actual answer rather than an omission, and the base class's
    # litellm estimate is this adapter's PRIMARY cost source. That is also what
    # Harbor does for Codex (`codex.py:862-880`, summed at `1086-1093`), except
    # that Harbor prices each API call separately and sums; we price the
    # session's `total_token_usage` once, which is the same arithmetic on the
    # same rates because every call in an episode uses one model.
    #
    # The four names below are Codex's own, and its `input_tokens` is INCLUSIVE:
    # on a real episode here, input=1551808 = cached 1473125 + written 78584 +
    # 99 fresh, and total_tokens 1562188 = input + output. `output_tokens`
    # likewise already contains `reasoning_output_tokens` (10380 output, of
    # which 3630 reasoning, and the total still adds up), so reasoning must NOT
    # be added on top -- doing so would double-count the most expensive tokens
    # in the episode.
    USAGE_FIELDS = ("usage",)
    USAGE_KEYS = ("input_tokens", "output_tokens",
                  "cached_input_tokens", "cache_write_input_tokens")
    PROMPT_TOKENS_INCLUDE_CACHE = True

    def reported_cost_usd(self, raw: dict) -> float | None:
        """None, always: Codex never prices its own episodes.

        Deliberately NOT `raw.get("cost_usd")`. The runner hardcodes that field
        to None for this adapter, so reading it back would only ever pick up a
        figure THIS code had already estimated -- and would then relabel an
        estimate as `cost_source="reported"` the second time a record was
        re-costed. If a future `codex` CLI grows a price of its own (there is
        already an `info.total_cost` slot in the rollout schema, always absent
        so far), `codex_runner.summarise` should surface it and this method
        should return it.
        """
        return None

    # ---- this adapter's runtime -------------------------------------------
    #
    # Node 22 via nvm (shared with the Claude adapter -- see node_install) and
    # the `@openai/codex` npm package, exactly as Harbor installs them. Harbor
    # also apt-installs ripgrep alongside; we cannot, because the episode has
    # already dropped to an unprivileged uid by the time the install runs, and
    # Codex does not require it.
    PACKAGE = "@openai/codex"
    NODE_MAJOR = NODE_MAJOR

    # Harbor's version command and Harbor's parse of it: `codex --version`
    # prints `codex-cli 0.5.0`, and Harbor strips a leading `codex-cli`.
    VERSION_CMD = (NVM_PRELUDE + "codex --version 2>/dev/null | head -1 | "
                   "sed 's/^codex-cli//' | awk '{print $1}'")

    # Harbor's own check, verbatim: source nvm if it is there, then look for the
    # binary. It must stay cheap and network-free -- it runs at the top of every
    # episode and is what makes a pre-baked image a cache rather than a
    # requirement.
    INSTALL_CHECK = NVM_PRELUDE + "command -v codex >/dev/null 2>&1"

    # ---- where Codex keeps its state inside the container ------------------
    # Harbor's paths. CODEX_HOME is where the rollout file we read back lands;
    # the auth file is written to its own directory and symlinked in, which is
    # what Harbor does so the credential is not sitting in the same tree as the
    # config.
    CODEX_HOME = "/tmp/codex-home"
    SECRETS_DIR = "/tmp/codex-secrets"

    # `codex exec` flags that are not negotiable. The first two are Harbor's,
    # in Harbor's order; the last two are ours and are load-bearing:
    #
    #   --ignore-user-config  a `~/.codex/config.toml` on the host image (or one
    #                         an agent writes for itself mid-episode) must not be
    #                         able to re-enable anything the policy closed.
    #   --strict-config       every `-c` key below is then verified by the CLI.
    #                         Without it a renamed key would be accepted and
    #                         ignored, which is the exact failure mode that
    #                         produces a run whose config file claims a policy it
    #                         never applied.
    EXEC_FLAGS = ("--dangerously-bypass-approvals-and-sandbox",
                  "--skip-git-repo-check",
                  "--strict-config")
    IGNORE_USER_CONFIG = "--ignore-user-config"
    # Harbor turns the unified exec tool on explicitly.
    ENABLE = ("unified_exec",)

    # Features switched OFF for every episode, whatever the policy says.
    #
    # image_generation is `image_gen__imagegen`, which is present by default and
    # executes on OpenAI's servers -- outside this container's network
    # namespace, and therefore outside everything this sandbox can observe or
    # contain. The other three are desktop-app tools that the CLI does not carry
    # anyway; they cost nothing and mean a future CLI that DOES carry them does
    # not quietly acquire them here.
    # All four are `stable true` in `codex features list` on 0.152.0, i.e. ON
    # unless switched off. `--disable X` is `-c features.X=false`, so under
    # `--strict-config` a name that did not exist would fail the episode loudly
    # rather than being ignored.
    DISABLE = ("image_generation", "browser_use", "computer_use", "apps")

    # Config keys set on every episode regardless of policy. Both default to ON
    # in the CLI and both talk to OpenAI's servers from outside the container's
    # namespace, which is the same objection as the web tool.
    FIXED_CONFIG = ("analytics.enabled=false",
                    "check_for_update_on_startup=false")

    # Harbor's CLI_FLAGS, as a translation table. THIS adapter's delivery of the
    # model's `generation_config:` -- the same settings the judge path delivers
    # to litellm as kwargs arrive here as `-c model_reasoning_*` CLI flags.
    # `reasoning_effort` carries Harbor's default of "high"; the others are
    # unset unless the model's entry asks for them.
    DEFAULT_REASONING_EFFORT = "high"
    REASONING_SUMMARY_CHOICES = ("auto", "concise", "detailed", "none")
    # web_search is deliberately NOT one of them: it comes from the task's
    # `tools:` policy, which configs/models.yaml may not override.
    GENERATION_KEYS = ("reasoning_effort", "reasoning_summary")

    # Codex's own toolset. The first two run INSIDE the container, so their
    # every byte crosses the container's network namespace. `web__run` is the
    # one tool that does not, and it is the one thing the policy switches -- it
    # is a single tool covering BOTH policy keys, which is why they cannot
    # disagree.
    # `exec` is the tool `--enable unified_exec` exposes (the feature name and
    # the tool name differ); both names are observed in a real 0.152.0 rollout.
    BASE_TOOLS = ("exec", "apply_patch")
    WEB_TOOL = "web__run"

    # When a model entry carries its own `base_url:` (configs/models.yaml), the
    # CLI is reaching a provider that is NOT OpenAI's own endpoint, and the
    # built-in `openai` provider's wire protocol -- the Responses API -- is the
    # wrong one for it. Codex reaches an override provider through a
    # `[model_providers.<id>]` block, and `wire_api` on that block is pinned to
    # chat completions: OpenRouter serves `/api/v1/chat/completions` for these
    # models and not the Responses API. The block's `env_key` stays
    # OPENAI_API_KEY on purpose -- `codex_runner` puts the resolved credential
    # there whatever env var `api_key_env:` named, so auth works either way and
    # no second key name has to be plumbed into config.toml.
    OVERRIDE_PROVIDER_ID = "override"
    OVERRIDE_PROVIDER_ENV_KEY = "OPENAI_API_KEY"
    OVERRIDE_WIRE_API = "chat"

    # The runner is a sibling module, read as SOURCE and never imported here:
    # it runs inside the container, streamed in on stdin.
    RUNNER = Path(__file__).resolve().parent / "codex_runner.py"

    # ---- install ----------------------------------------------------------
    def install_check(self) -> str:
        parts = [self.INSTALL_CHECK]
        if self.pinned_version:
            # Pinned means the RIGHT build, not any build: Harbor re-installs on
            # a version mismatch rather than accepting what it found, and so
            # must we, or a stale cache silently decides the run.
            parts.append(f'[ "$({self.VERSION_CMD})" = '
                         f"{shlex.quote(self.pinned_version)} ]")
        return " && ".join(parts)

    def install_script(self) -> str:
        pkg = self.PACKAGE + (f"@{self.pinned_version}" if self.pinned_version
               else "@latest")
        return "\n".join([
            *node_install(self.NODE_MAJOR),
            'npm_config_cache="$(mktemp -d)"; export npm_config_cache',
            f"npm install -g --no-fund --no-audit {pkg}",
            'rc=$?; rm -rf "$npm_config_cache" "$HOME/.npm"',
            link_bins("node", "npm", "npx", "codex"),
            "exit $rc",
        ])

    # ---- the task's tool policy -------------------------------------------
    def web_enabled(self) -> bool:
        """Whether `web__run` is offered. Both policy keys, or neither.

        One tool serves both keys, so a task that sets them differently is
        asking for something no invocation of this CLI can express. Refuse it
        here -- `apply_tool_policy` turns this into the same refusal a
        non-enforceable key gets -- rather than picking a side and recording a
        policy the episode did not run under.
        """
        if self.policy is None:
            raise RuntimeError("apply_tool_policy has not been called")
        search, fetch = self.policy["web_search"], self.policy["web_fetch"]
        if search != fetch:
            raise ValueError(
                f"adapter {self.name()!r} has ONE web knob (`-c web_search=`) and "
                f"Codex has ONE web tool (`web__run`, whose operations are "
                f"search_query/open/click/find/screenshot), so web_search and "
                f"web_fetch cannot differ; this task asks for "
                f"web_search={search}, web_fetch={fetch}")
        return search

    def _tools_for(self, policy: dict) -> list[str]:
        if policy["mcp_only"]:
            if self.web_enabled():
                raise ValueError("mcp_only cannot enable native web tools")
            return []
        return list(self.BASE_TOOLS) + ([self.WEB_TOOL] if self.web_enabled()
                                        else [])

    def cli_flags(self, *, mcp_server="chess") -> list[str]:
        """The `-c key=value` config flags for this run.

        `web_search` is ALWAYS emitted, and that is the whole point of this
        method. Harbor omits the flag when nothing asked for it, and the CLI's
        default is not `disabled` but `cached` -- which still carries the
        `open`/`click` URL-fetch surface, executing on OpenAI's servers outside
        this container's network namespace. A faithful port of Harbor's
        "omit when unset" would therefore ship an episode that can reach the
        REAL terrytao.wordpress.com instead of the mirror injected into the
        episode's namespace, and the task's fiction would break with nothing in
        the record to say so. There is a test on this line.

        `false -> disabled`, `true -> live`. The CLI's other two values
        (`cached`, `indexed`) are not reachable from a boolean policy, and
        `cached` is specifically the trap above. The value is a string enum: the
        CLI rejects `-c web_search=false` outright.
        """
        flags: list[str] = []
        effort = self.generation_config.get("reasoning_effort",
                                            self.DEFAULT_REASONING_EFFORT)
        if effort:
            flags += ["-c", f"model_reasoning_effort={effort}"]
        summary = self.generation_config.get("reasoning_summary")
        if summary:
            flags += ["-c", f"model_reasoning_summary={summary}"]
        for setting in self.FIXED_CONFIG:
            flags += ["-c", setting]
        flags += ["-c", "web_search=" + ("live" if self.web_enabled()
                                         else "disabled")]
        if self.policy["mcp_only"]:
            flags += ["-c", 'features.code_mode.direct_only_tool_namespaces=' + json.dumps(['mcp__' + mcp_server]),
                      "-c", "features.code_mode.enabled=false", "-c", "agents.enabled=false"]
        return flags

    def resolved_generation_config(self) -> dict:
        """What this adapter will actually send, defaults included.

        `reasoning_effort` is Harbor's `high` unless the entry overrides it, so
        a run that set nothing still ran with an effort -- and run.json has to
        say which, or the record would imply the CLI's own default was used.
        """
        return {"reasoning_effort": self.DEFAULT_REASONING_EFFORT,
                **self.generation_config}

    def exec_flags(self, base_url: str = "", *, grader: bool = False) -> list[str]:
        """The non-negotiable `codex exec` flags for this run.

        `--ignore-user-config` is conditional, and the condition is not a
        preference. It makes the CLI ignore `$CODEX_HOME/config.toml` -- which
        is the ONLY place codex >= 0.118 reads `openai_base_url` from (the env
        var being ignored) and the ONLY place it takes MCP servers from at all.
        So it can only be passed when this episode has NOTHING to put in that
        file: no base URL AND no grader. It is not needed for the case it was
        added for either: pointing CODEX_HOME at our own fresh directory already
        means a `~/.codex/config.toml` on the image is never read.
        """
        flags = list(self.EXEC_FLAGS)
        if self.policy and self.policy["mcp_only"]:
            flags.remove("--dangerously-bypass-approvals-and-sandbox")
            flags += ["--sandbox", "read-only", "-c", 'approval_policy="never"']
        if not base_url and not grader:
            flags.append(self.IGNORE_USER_CONFIG)
        return flags

    def disable_features(self) -> list[str]:
        """`--disable` names for this run. The same set for every policy."""
        disabled = list(self.DISABLE)
        if self.policy and self.policy["mcp_only"]:
            disabled += ["shell_tool", "unified_exec", "shell_snapshot", "multi_agent", "code_mode_host",
                         "plugins", "remote_plugin", "view_image", "workspace_dependencies", "goals",
                         "tool_suggest", "enable_request_compression"]
        return disabled

    @property
    def model_slug(self) -> str:
        """What goes after `--model`. Harbor: `model_name.split("/")[-1]`.

        configs/models.yaml writes `openai/gpt-5.6-sol` because the judge path
        needs a provider prefix; the CLI wants the bare id, so the prefix is
        stripped -- for the default openai provider.

        A per-model `base_url:` override (a custom `model_providers` block, see
        `blob`) is the exception: an OpenRouter model id IS `openai/gpt-6-astra`,
        prefix and all, because that is the string OpenRouter routes on.
        Stripping it there would ask OpenRouter for a model called
        `gpt-6-astra`, which is not one. So the model goes through verbatim when
        this instance carries its own base URL.
        """
        if self.base_url:
            return self.model
        return self.model.split("/")[-1]

    def setup(self) -> list[str]:
        problems = [] if self.RUNNER.exists() else [f"missing runner {self.RUNNER}"]
        if self.policy is None:
            problems.append("apply_tool_policy was never called")
            return problems
        # A model's `generation_config:` may only carry knobs this adapter can
        # deliver, and an `agents:` entry only knobs it understands. Anything
        # else would be silently dropped, and a silently dropped reasoning
        # setting is a run whose config file lies about it.
        problems += self.config_problems()
        summary = self.generation_config.get("reasoning_summary")
        if summary and summary not in self.REASONING_SUMMARY_CHOICES:
            problems.append(f"reasoning_summary={summary!r}; codex takes "
                            f"{list(self.REASONING_SUMMARY_CHOICES)}")
        try:
            web = self.web_enabled()
        except ValueError as e:
            return [*problems, str(e)]
        # The cached-default trap, asserted rather than assumed: a closed-book
        # episode that did not carry `-c web_search=disabled` would run with the
        # CLI's default of `cached`, which still fetches URLs server-side.
        flags = self.cli_flags()
        pairs = list(zip(flags, flags[1:]))
        if not web:
            if self.WEB_TOOL in self.tools:
                problems.append(f"web is closed but {self.WEB_TOOL} is still "
                                f"in the tool list")
            if ("-c", "web_search=disabled") not in pairs:
                problems.append("closed-book policy did not produce "
                                "`-c web_search=disabled`; the CLI would "
                                "default to `cached`, which still fetches URLs "
                                "on OpenAI's servers")
        elif ("-c", "web_search=live") not in pairs:
            problems.append("open-web policy did not produce "
                            "`-c web_search=live`")
        return problems

    # ---- the episode ------------------------------------------------------
    def blob(self, *, row: dict, prompt, files: list, skip_dirs: tuple,
             grader_src: str = "", stage_src: str, timeout_s: int,
             env: dict, install: dict | None = None,
             task_install: dict | None = None) -> bytes:
        """The stdin payload for one episode.

        `row` and `grader_src` ARE forwarded, and only when the task ships a
        grader -- `grader_src` is "" otherwise and neither key carries anything,
        so openmath's episodes are unchanged and its `grader_state` stays an
        explicit None.

        This used to forward neither, on the grounds that Codex takes an MCP
        server "as a command line or a URL in `$CODEX_HOME/config.toml`, which
        the agent can read and re-run itself". THAT WAS WRONG, and it was wrong
        in a way worth spelling out because it is easy to re-derive: it is true
        of the STDIO transport and false of the URL one. Codex, like Harbor's own
        adapter (`harbor/src/harbor/agents/installed/codex.py`, which writes
        `command`/`args` for `transport == "stdio"` and `{"url": ...}` for
        everything else), accepts an endpoint:

            [mcp_servers.grader]
            url = "http://127.0.0.1:<port>/<token>/mcp"

        There is no command in that file to re-run and no path to any source.
        `core/agents/codex_runner.py` execs this grader out of the stdin blob
        into its own heap and serves it on loopback, which is the same property
        that makes the Claude adapter's in-process SDK server safe. The
        consequence of getting this wrong was not academic: on gdpval, which
        HAS a grader, the Claude adapter offered `grade_deliverable` and this one
        did not, so the two adapters were not running the same task.

        What the row and the grader still never do is touch the container
        filesystem, appear in argv or appear in the environment. They ride on
        stdin, which this process consumes before Codex exists.
        """
        # Which URL goes into config.toml, in order: the legacy per-model
        # `base_url:` override (a non-OpenAI provider block, below), else the
        # entry's `api_base_url:` (the gateway, written as `openai_base_url`
        # exactly as the passthrough was), else the OPENAI_BASE_URL passthrough
        # -- the pre-existing fallback, so an entry that names nothing runs as
        # before. `resolved_base_url()` is this same order.
        base_url = self.resolved_base_url()
        codex = {
            "model_slug": self.model_slug,
            "mcp_only": self.policy["mcp_only"],
            "exec_flags": self.exec_flags(base_url,
                                          grader=bool(grader_src)),
            "enable": [] if self.policy["mcp_only"] else list(self.ENABLE),
            "disable": self.disable_features(),
            "cli_flags": self.cli_flags(mcp_server=row.get("tool_surface", "chess")),
            "codex_home": self.CODEX_HOME,
            "secrets_dir": self.SECRETS_DIR,
            # The name the key sits under INSIDE the container (CLI_KEY_ENV),
            # which is what the runner reads; the host-side name may differ.
            "api_key_env": self.container_key_envs()[0],
            # codex >= 0.118 ignores OPENAI_BASE_URL and reads
            # `openai_base_url` from config.toml only. Harbor says so in a
            # comment and writes the file; so do we. Empty means "the
            # vendor's default", and no config.toml is written at all.
            "base_url": base_url,
        }
        # A model that carries its OWN base URL is reaching a non-OpenAI
        # provider, so config.toml gets a `[model_providers.<id>]` block with a
        # chat-completions wire_api instead of the bare `openai_base_url` (which
        # would leave the built-in openai provider on the Responses API). When
        # this key is absent the runner writes the old openai_base_url shape, so
        # the gateway path is untouched.
        if self.base_url:
            codex["provider"] = {
                "id": self.OVERRIDE_PROVIDER_ID,
                "base_url": self.base_url,
                "env_key": self.OVERRIDE_PROVIDER_ENV_KEY,
                "wire_api": self.OVERRIDE_WIRE_API,
            }
        task = {
            "id": row["id"],
            "model": self.model,
            "content": prompt,
            "files": files,
            "skip_dirs": list(skip_dirs),
            "deliverable_files": list(row.get("deliverable_files") or ()),
            "tools": self.tools,
            "timeout_s": timeout_s,
            "env": dict(env or {}),
            "install": install,
            # The TASK's own `check || install`, from its `install:`
            # block, run by the same helper right after the agent's: a
            # library this task's work needs that the shared image does
            # not carry. None for a task that declares none.
            "task_install": task_install,
            # Everything the runner needs to compose Harbor's command line.
            "codex": codex,
        }
        if grader_src:
            # The whole row, as the Claude adapter ships it: the grader is the
            # only thing that reads it, and it must read the same row on both
            # adapters or the two are not grading the same episode. A task with
            # no grader has no reader for it, so it is not sent at all.
            task["row"] = {**row, "episode_timeout_s": timeout_s} if row.get("tool_surface") in {"chess", "go"} else row
        # The model's routing, for the runner: where the key sits, the real
        # upstream, and the extra_body that decides whether the CLI goes
        # through the forwarder. The forwarder's SOURCE travels only when it
        # is needed, in `modules` like the grader.
        task["routing"] = self.routing_payload()
        modules = {"stage": stage_src, "grader": grader_src}
        if self.extra_body:
            modules["forwarder"] = self.forwarder_source()
        return json.dumps({
            "task": task,
            "code": self.RUNNER.read_text(),
            # "" IS the convention for "this task ships no grader", threaded
            # from `Task.grader_src()`; the key is carried either way so the
            # runner asks "is it non-empty" rather than "is it there".
            "modules": modules,
        }).encode()

    # ---- per-model routing ------------------------------------------------
    def resolved_base_url(self) -> str:
        """Legacy `base_url:` (provider block) first, then the shared order."""
        return self.base_url or super().resolved_base_url()

    def base_url_source(self) -> str:
        return "entry:base_url" if self.base_url else super().base_url_source()

    def routing_problems(self) -> list[str]:
        problems = super().routing_problems()
        if self.base_url and self.api_base_url:
            problems.append(
                f"model {self.model!r}: sets both `base_url:` (the codex "
                f"non-OpenAI provider override, chat wire API) and "
                f"`api_base_url:` (the shared routing key, Responses API via "
                f"openai_base_url). They name different config.toml shapes; "
                f"keep one.")
        return problems

    # ---- the rollout as ATIF ----------------------------------------------
    #
    # PORTED FROM HARBOR, Apache License 2.0: this is
    # `harbor/src/harbor/agents/installed/codex.py`, methods
    # `_extract_message_text`, `_parse_output_blob`, `_group_events_by_api_call_id`,
    # `_metrics_from_token_count_payload`, `_convert_event_to_step` and
    # `_convert_events_to_trajectory`, ported rather than reimplemented. Derived
    # work; see core/trajectory/__init__.py for the full notice.
    #
    # Two deviations, both forced and both narrow:
    #
    #   * Harbor reads the rollout `.jsonl` off disk after copying
    #     `$CODEX_HOME/sessions` out of the environment. There is no bind mount
    #     here, so `core/agents/codex_runner.py` reads the file INSIDE the container and
    #     returns its lines in the one JSON line it already owes us. The events
    #     therefore arrive as a list rather than a path, and the file-globbing
    #     half of Harbor's method is gone. The event handling is unchanged.
    #   * Harbor prices each API call through litellm host-side. The container
    #     has no pricing table, so `cost_usd` is left None rather than guessed;
    #     the token counts, which ARE in the rollout, are recorded exactly.
    #
    # Codex's rollout lines are `{"timestamp":..., "type":..., "payload":...}`.
    # The types Harbor handles, and so do we:
    #
    #   session_meta                         the session header
    #   turn_context                         model + effort for the turn
    #   event_msg / token_count              per-API-call and total usage
    #   response_item / message              user, developer or assistant text
    #   response_item / reasoning            the summarised chain of thought
    #   response_item / function_call        a tool call, args as a JSON string
    #   response_item / custom_tool_call     a free-form tool call, args raw
    #   response_item / *_call_output        what came back
    #   response_item / web_search_call      a server-side search, no output line
    @staticmethod
    def _iso(timestamp):
        """A rollout timestamp, or None when it is not ISO 8601.

        DELIBERATE DIVERGENCE FROM HARBOR. ATIF validates `Step.timestamp` as
        ISO 8601, pydantic raises `ValidationError` when it is not, and
        `ValidationError` is a subclass of `ValueError` -- so Harbor's
        `except ValueError: continue` around the step conversion silently drops
        every step of a rollout whose timestamps it cannot parse, turning a
        cosmetic metadata problem into a TOTAL loss of the trajectory (and, in
        this repo, of the evidence a verdict is taken on). A timestamp we cannot
        read is worth losing; the step it was attached to is not.
        """
        if not isinstance(timestamp, str):
            return None
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return timestamp

    @staticmethod
    def _message_text(content) -> str:
        """Harbor's `_extract_message_text`."""
        parts = []
        for block in content or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    @staticmethod
    def _output_blob(raw):
        """Harbor's `_parse_output_blob`: (text, metadata) from a tool output."""
        if raw is None:
            return None, None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return raw, None
        else:
            parsed = raw
        if isinstance(parsed, dict):
            out = parsed.get("output")
            if out is None and parsed:
                out = json.dumps(parsed, ensure_ascii=False)
            meta = parsed.get("metadata")
            return out, meta if isinstance(meta, dict) else None
        if isinstance(parsed, list):
            # Codex's `exec` returns a list of content parts. Keep the text they
            # contain, not `str()` of the list, which would put a Python repr --
            # quotes, `{'type': 'input_text', ...}` and escaped newlines -- in
            # front of the judge instead of the command's actual output.
            return "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in parsed), None
        return str(parsed), None

    @staticmethod
    def _metrics_from_token_count(payload: dict) -> dict | None:
        """Harbor's `_metrics_from_token_count_payload`."""
        info = payload.get("info")
        if not isinstance(info, dict):
            return None
        last = info.get("last_token_usage")
        if not isinstance(last, dict):
            return None
        extra = {"reasoning_output_tokens": last.get("reasoning_output_tokens"),
                 "total_tokens": last.get("total_tokens")}
        cache_write = last.get("cache_write_input_tokens")
        if cache_write is not None:
            extra["cache_write_input_tokens"] = cache_write
        prompt = last.get("input_tokens")
        return {"prompt_tokens": prompt if prompt else None,
                "completion_tokens": last.get("output_tokens") or None,
                "cached_tokens": last.get("cached_input_tokens") or None,
                "extra": extra}

    @staticmethod
    def _group_events_by_api_call_id(normalized: list[dict]) -> list[dict]:
        """Harbor's `_group_events_by_api_call_id`, verbatim in behaviour.

        Merges every assistant event that came out of the SAME model request
        into one bundled event, so one ATIF step is one inference rather than
        one rollout line. Without it a turn that issued four tool calls would
        become four steps with the same usage attached to each.
        """
        result: list[dict] = []
        groups: dict[str, dict] = {}
        order: list[str] = []

        def add_reasoning(group: dict, event: dict) -> None:
            """DELIBERATE DIVERGENCE FROM HARBOR, for the same reason as _iso.

            Harbor assigns here -- a message overwrites the group's reasoning, a
            tool call is kept only when the group has none -- so a bundle
            carrying reasoning from two events keeps exactly one of them. Every
            event in a bundle came out of the SAME model request and its
            reasoning is the same step's, so dropping either is losing evidence
            the judge is meant to weigh. Accumulate; the buffer that fills these
            events is drained by each consumer, so nothing is counted twice.
            """
            text = event.get("reasoning")
            if not text:
                return
            group["reasoning"] = (text if not group.get("reasoning")
                                  else f"{group['reasoning']}\n{text}")

        def flush() -> None:
            for gid in order:
                group = groups.pop(gid, None)
                if group is None:
                    continue
                group["tool_calls"].sort(key=lambda tc: tc.get("tool_order", 0))
                group["text"] = "\n\n".join(
                    p for p in group.pop("message_parts")
                    if isinstance(p, str) and p)
                result.append(group)
            order.clear()

        for event in normalized:
            api_call_id = event.get("api_call_id")
            kind = event.get("kind")

            if kind == "message" and event.get("role") != "assistant":
                flush()
                result.append(event)
                continue
            if kind == "observation":
                # A result whose call was in an earlier record (the live tail) or
                # had no call at all. It is its own step, never folded into a
                # bundle, so it cannot pick up a neighbouring call's identity.
                flush()
                result.append(event)
                continue
            if not isinstance(api_call_id, str):
                flush()
                result.append(event)
                continue

            if api_call_id not in groups:
                groups[api_call_id] = {
                    "kind": "bundled",
                    "api_call_id": api_call_id,
                    "codex_turn_id": event.get("codex_turn_id"),
                    "timestamp": event.get("timestamp"),
                    "message_parts": [],
                    "reasoning": None,
                    "tool_calls": [],
                    "metrics": event.get("metrics"),
                }
                order.append(api_call_id)

            group = groups[api_call_id]
            if kind == "message":
                text = event.get("text")
                if isinstance(text, str) and text:
                    group["message_parts"].append(text)
                add_reasoning(group, event)
                if event.get("timestamp"):
                    group["timestamp"] = event["timestamp"]
            elif kind == "tool_call":
                group["tool_calls"].append(event)
                add_reasoning(group, event)
                if not group.get("metrics") and event.get("metrics"):
                    group["metrics"] = event["metrics"]

        flush()
        return result

    def _event_to_step(self, event: dict, step_id: int, model: str | None) -> Step:
        """Harbor's `_convert_event_to_step`, for the kinds this runner produces."""
        kind = event.get("kind")
        timestamp = self._iso(event.get("timestamp"))

        if kind == "message":
            role = event.get("role", "user")
            source = ("agent" if role == "assistant"
                      else "user" if role == "user" else "system")
            return Step(
                step_id=step_id, timestamp=timestamp, source=source,
                message=event.get("text", ""),
                reasoning_content=(event.get("reasoning")
                                   if source == "agent" else None),
                model_name=model if source == "agent" else None,
                llm_call_count=1 if source == "agent" else None,
            )

        if kind == "observation":
            # source_call_id is left None ON PURPOSE: the call it belongs to is
            # in a different step (the live tail) or absent (a truly orphan
            # output), and ATIF requires a source_call_id to name a tool_call in
            # THIS step. It renders as `RESULT[tool]`, which is the documented
            # tail behaviour.
            return Step(
                step_id=step_id, timestamp=timestamp, source="agent",
                message="",
                observation=Observation(results=[ObservationResult(
                    source_call_id=None, content=event.get("output"))]),
            )

        if kind == "bundled":
            calls: list[ToolCall] = []
            results: list[ObservationResult] = []
            details: dict = {}
            for spec in event.get("tool_calls", []):
                call_id = spec.get("call_id") or ""
                args = spec.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                calls.append(ToolCall(tool_call_id=call_id,
                                      function_name=spec.get("tool_name") or "",
                                      arguments=args))
                # A call with no paired output (a killed episode's last call, or
                # any call in the live tail) carries no `output` key. Emit the
                # call without a phantom empty result beside it.
                if "output" in spec:
                    results.append(ObservationResult(
                        source_call_id=call_id or None,
                        content=spec.get("output")))
                detail = {k: spec.get(k) for k in
                          ("metadata", "raw_arguments", "item_type", "status")
                          if spec.get(k)}
                if detail:
                    details[call_id] = detail

            extra: dict = {}
            for key in ("api_call_id", "codex_turn_id"):
                if event.get(key):
                    extra[key] = event[key]
            if details:
                extra["tool_call_details"] = details

            return Step(
                step_id=step_id, timestamp=timestamp, source="agent",
                message=event.get("text", ""),
                model_name=model,
                reasoning_content=event.get("reasoning") or None,
                tool_calls=calls or None,
                observation=Observation(results=results) if results else None,
                metrics=Metrics(**event["metrics"]) if event.get("metrics") else None,
                llm_call_count=1,
                extra=extra or None,
            )

        raise ValueError(f"Unsupported event kind '{kind}'")

    def to_trajectory(self, raw: dict) -> Trajectory:
        """Harbor's `_convert_events_to_trajectory`, over in-memory events."""
        events = [e for e in ((raw or {}).get("messages") or [])
                  if isinstance(e, dict)]

        session_meta = next(
            (e for e in events if e.get("type") == "session_meta"), None)
        meta = (session_meta or {}).get("payload") or {}
        session_id = meta.get("id") or (raw or {}).get("session_id")

        agent_version = meta.get("cli_version") or self.version() or "unknown"
        agent_extra = {k: meta[k] for k in ("originator", "cwd")
                       if meta.get(k) is not None} or None

        model = None
        for event in events:
            if event.get("type") == "turn_context":
                candidate = (event.get("payload") or {}).get("model")
                if isinstance(candidate, str):
                    model = candidate
                    break
        model = model or self.model

        # -- normalise every rollout line into one flat event stream ---------
        normalized: list[dict] = []
        pending: dict[str, dict] = {}
        reasoning: str | None = None
        turn_id: str | None = None
        api_index = 1
        api_call_id = f"api_call_{api_index}"
        api_metrics: dict[str, dict] = {}
        saw_output = False
        tool_order = 0

        def finish_api_call(payload: dict) -> None:
            nonlocal api_index, api_call_id, saw_output, tool_order
            if not saw_output:
                return
            metrics = self._metrics_from_token_count(payload)
            if metrics:
                api_metrics[api_call_id] = metrics
            api_index += 1
            api_call_id = f"api_call_{api_index}"
            saw_output = False
            tool_order = 0

        for event in events:
            etype = event.get("type")
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            timestamp = event.get("timestamp")

            if etype == "event_msg":
                kind = payload.get("type")
                if kind in {"task_started", "turn_started"}:
                    tid = payload.get("turn_id")
                    turn_id = tid if isinstance(tid, str) else None
                elif kind in {"task_complete", "turn_complete", "turn_aborted"}:
                    turn_id = None
                elif kind == "token_count":
                    # A token_count event closes one model API call.
                    finish_api_call(payload)
                continue

            if etype == "turn_context":
                tid = payload.get("turn_id")
                if isinstance(tid, str) and turn_id is None:
                    turn_id = tid
                continue

            if etype != "response_item":
                continue

            ptype = payload.get("type")

            if ptype == "reasoning":
                summary = payload.get("summary")
                parts = []
                for item in summary if isinstance(summary, list) else []:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                # ACCUMULATE, never assign. `reasoning` is a buffer that the next
                # message/tool_call drains (each of those sets it back to None),
                # so two reasoning items with no consumer between them both
                # belong to the SAME upcoming step. Assigning here discarded the
                # earlier one: measured on a real rollout, one `reasoning
                # reasoning` adjacency lost 991 of 4025 summary chars before the
                # judge ever saw them. A rollout without that adjacency lost
                # nothing, which is why this only showed up sometimes.
                #
                # An item with no readable summary is a no-op rather than a
                # reset: `model_reasoning_summary` emits plenty of them (7 of 15
                # on the same rollout, carrying only `encrypted_content`), and
                # letting one clear the buffer would drop text for exactly the
                # reason above.
                if parts:
                    chunk = "\n".join(parts)
                    reasoning = chunk if reasoning is None else f"{reasoning}\n{chunk}"
                continue

            if ptype == "message":
                role = payload.get("role", "user")
                content = payload.get("content")
                normalized.append({
                    "kind": "message", "api_call_id": api_call_id,
                    "codex_turn_id": turn_id, "timestamp": timestamp,
                    "role": role,
                    "text": (self._message_text(content)
                             if isinstance(content, list) else ""),
                    "reasoning": reasoning if role == "assistant" else None,
                })
                if role == "assistant":
                    saw_output = True
                reasoning = None
                continue

            if ptype == "web_search_call":
                action = payload.get("action") or {}
                args = {"action_type": action.get("type", "")}
                for key in ("query", "queries", "url"):
                    if key in action:
                        args[key] = action[key]
                normalized.append({
                    "kind": "tool_call", "api_call_id": api_call_id,
                    "codex_turn_id": turn_id, "tool_order": tool_order,
                    "timestamp": timestamp, "call_id": "",
                    "tool_name": "web_search_call", "arguments": args,
                    "raw_arguments": None, "reasoning": reasoning,
                    "status": payload.get("status"),
                    # The provider returns the hits inside its own response and
                    # never writes them to the rollout, so there is no output
                    # line to pair. Say so, rather than leaving a call whose
                    # result silently reads as empty.
                    "output": "(results are returned to the model inside the "
                              "provider's response and are not recorded in the "
                              "rollout)",
                })
                tool_order += 1
                saw_output = True
                reasoning = None
                continue

            if ptype in {"function_call", "custom_tool_call"}:
                call_id = payload.get("call_id")
                if not call_id:
                    continue
                raw_args = payload.get(
                    "arguments" if ptype == "function_call" else "input")
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = None
                if args is None:
                    # Free-form tool input can be a complete JavaScript program:
                    # batches, mixed tools, variables, loops and output handling.
                    # Keep all of it in the visible arguments. Extracting object
                    # fields hid later calls from the judge, even though their
                    # outputs survived. Metadata alone is not rendered.
                    if isinstance(raw_args, str):
                        args = {"input": raw_args}
                    elif raw_args is None:
                        args = {}
                    else:
                        args = {"value": raw_args}
                pending[call_id] = {
                    "kind": "tool_call", "api_call_id": api_call_id,
                    "codex_turn_id": turn_id, "tool_order": tool_order,
                    "timestamp": timestamp, "call_id": call_id,
                    "tool_name": payload.get("name") or "", "arguments": args,
                    "raw_arguments": raw_args, "item_type": ptype,
                    "reasoning": reasoning, "status": payload.get("status"),
                }
                tool_order += 1
                saw_output = True
                reasoning = None
                continue

            if ptype in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload.get("call_id")
                output, metadata = self._output_blob(payload.get("output"))
                info = pending.pop(call_id, None) if call_id else None
                if info is None:
                    # No call to pair with. This is the norm in the LIVE turn log,
                    # where the tail sees the output record with no memory of the
                    # call record before it. Emit a result-only step, NOT a
                    # fabricated tool call: the old fallback invented a nameless,
                    # argument-less call, which rendered as `TOOL_CALL : {}` and
                    # read as if codex had run an empty command.
                    normalized.append({
                        "kind": "observation", "api_call_id": api_call_id,
                        "codex_turn_id": turn_id, "tool_order": tool_order,
                        "timestamp": timestamp, "call_id": call_id or "",
                        "output": output, "metadata": metadata,
                    })
                    tool_order += 1
                    reasoning = None
                    continue
                info["output"] = output
                info["metadata"] = metadata
                info["timestamp"] = info.get("timestamp") or timestamp
                normalized.append(info)
                reasoning = None
                continue

        # A tool call whose output never arrived is still a step: it is what the
        # agent asked for. In the batch path this is the final call of a killed
        # episode (previously dropped in silence); in the LIVE turn log it is
        # EVERY call, since the tail sees the call record with no output record
        # beside it -- without this flush the command never reached turns.log.
        # No `output` key is set, so the step renders the call and no result.
        for info in pending.values():
            normalized.append(info)
        pending.clear()

        for event in normalized:
            key = event.get("api_call_id")
            if isinstance(key, str) and key in api_metrics:
                event["metrics"] = api_metrics[key]

        steps: list[Step] = []
        dropped: list[str] = []
        for event in self._group_events_by_api_call_id(normalized):
            try:
                steps.append(self._event_to_step(event, len(steps) + 1, model))
            except ValueError as e:
                # Harbor swallows this. We COUNT it and say so in the
                # trajectory's `notes`, because a step that vanished between
                # the rollout and the judge is a hole in the evidence, and a
                # hole nobody is told about is the expensive kind.
                dropped.append(f"{event.get('kind')}: {type(e).__name__}: {e}")

        if not steps:
            steps = [self.empty_step(
                "(no rollout events were captured for this episode)")]

        # -- totals, from the LAST token_count that carried them -------------
        final_metrics = None
        for event in reversed(events):
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            total = info.get("total_token_usage")
            if not isinstance(total, dict):
                continue
            extra = {"reasoning_output_tokens": total.get("reasoning_output_tokens"),
                     "total_tokens": total.get("total_tokens"),
                     "last_token_usage": info.get("last_token_usage")}
            cache_write = total.get("cache_write_input_tokens")
            if cache_write is not None:
                extra["total_cache_write_input_tokens"] = cache_write
            prompt = total.get("input_tokens")
            final_metrics = FinalMetrics(
                total_prompt_tokens=prompt if prompt else None,
                total_completion_tokens=total.get("output_tokens") or None,
                total_cached_tokens=total.get("cached_input_tokens") or None,
                # Codex's rollout carries token counts but never a price, and
                # the container has no pricing table (Harbor computes this
                # host-side through litellm). Left None rather than guessed.
                total_cost_usd=info.get("total_cost")
                if info.get("total_cost") is not None else (raw or {}).get("cost_usd"),
                total_steps=len(steps),
                extra=extra,
            )
            break

        final_text = (raw or {}).get("final_text") or ""
        if not final_text:
            for step in reversed(steps):
                text = step.message if isinstance(step.message, str) else ""
                if step.source == "agent" and text.strip():
                    final_text = text
                    break

        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            agent=ATIFAgent(name=self.name(), version=agent_version,
                            model_name=model, extra=agent_extra),
            steps=steps,
            notes=(f"{len(dropped)} rollout event(s) could not be converted to "
                   f"ATIF steps and are missing from this trajectory: "
                   + "; ".join(dropped)) if dropped else None,
            final_metrics=final_metrics,
            extra={"final_text": final_text} if final_text else None,
        )
