"""Agents that are a CLI we have to put into the container first.

The second level of Harbor's two-level split
(`harbor/src/harbor/agents/installed/base.py`): `BaseAgent` is the contract,
`BaseInstalledAgent` is everything that follows from "this agent is a binary
that is not in the environment yet". Both of our adapters are installed agents,
so both subclass this rather than `Agent` directly.

Shared here, and written once rather than per vendor:

  node_install / link_bins   Node 22 via nvm, which more than one adapter needs
                             and which neither the image nor either vendor ships
  INSTALL_CHECK / INSTALL    the check/install pair for this adapter's binary
  VERSION_CMD                what actually got installed, for the record
  BOOTSTRAP                  the deliberately uninformative one-liner that goes
                             in the container's argv

Harbor's own `installed/node_install.py` is the same helper for the same reason;
ours differs only in shape (a list of shell lines rather than one `&&` chain)
because our runner runs the install as a script and not as a single exec.
"""
from __future__ import annotations

import shlex

from .base import Agent


# ---------------------------------------------------------------------------
# Node, which more than one adapter needs and neither the image nor either
# vendor ships.
#
# Both CLIs we drive declare `node >=22`; Debian slim ships 20, which installs
# with five lines of EBADENGINE and runs on a runtime its author does not
# support. Node is the AGENT's dependency, not the environment's, so an adapter
# brings its own via nvm -- the same thing Harbor does for its nine node agents,
# and the same nvm release Harbor pins.
NVM_VERSION = "v0.40.2"
NODE_MAJOR = 22


def node_install(node_major: int = NODE_MAJOR) -> list[str]:
    """Shell lines that leave nvm's Node `node_major` on the PATH of this shell."""
    return [
        # nvm's installer refuses to run with NODE_VERSION set, which the base
        # image exports.
        'export NVM_DIR="$HOME/.nvm"; mkdir -p "$NVM_DIR"',
        "env -u NODE_VERSION bash -c 'curl -fsSL "
        f"https://raw.githubusercontent.com/nvm-sh/nvm/{NVM_VERSION}"
        "/install.sh | bash' || exit $?",
        '. "$NVM_DIR/nvm.sh" || exit $?',
        f"nvm install {node_major} || exit $?",
        f"nvm alias default {node_major} || exit $?",
    ]


def link_bins(*names: str) -> str:
    """Put nvm's binaries where every process looks.

    nvm puts node and the global bin dir on the PATH of the shell that sourced
    it, and that shell exits with the install script. A runner spawning the CLI
    from python needs them reachable through the ordinary PATH. The whole image
    is owned by container-root, which IS the agent's uid, so this needs no
    privilege.
    """
    return (f'for b in {" ".join(names)}; do p="$(command -v "$b" '
            '2>/dev/null)" && ln -sf "$p" /usr/local/bin/"$b"; done')


# Sourcing nvm before looking for a binary, which is what Harbor's own install
# check and version command do: the binary may only exist on nvm's PATH.
NVM_PRELUDE = "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "


class InstalledAgent(Agent):
    """An agent whose runtime the episode installs before the agent starts."""

    # -- this adapter's runtime, which the image does not carry ---------------
    #
    # INSTALL_CHECK exits 0 when the agent's binary is already present. It runs
    # at the top of EVERY episode, so it must be cheap and must not touch the
    # network -- it is what makes a pre-baked image an optional cache rather
    # than a requirement.
    #
    # INSTALL puts the binary there. Empty means this adapter needs nothing
    # installed, which is the safe default for a subclass that forgets.
    INSTALL_CHECK: str = "true"
    INSTALL: str = ""

    # Shell that prints the version that actually got installed, on one line.
    # Run by the container after `check || install`, whichever branch was taken,
    # so a finished run always states which agent build produced it. A run whose
    # agent version is unknown is a run nobody can reproduce.
    VERSION_CMD: str = ""

    # The one-liner that goes in the container's argv, and the ONLY thing on it.
    #
    # Shared by every adapter here because every adapter here works the same
    # way: the runner is streamed in on stdin and this loads it. It says nothing
    # about the eval, the task or the vendor -- the agent's Bash runs as our uid
    # and can read `/proc/<pid>/cmdline` of its own ancestors, so anything
    # informative on this line is a tell. See `core/agents/__init__.py`.
    BOOTSTRAP = ('import sys,json;d=json.load(sys.stdin);'
                 'g={"TASK":d["task"],"MODULES":d["modules"]};'
                 'exec(compile(d["code"],"<r>","exec"),g)')

    @property
    def bootstrap(self) -> str:
        return self.BOOTSTRAP

    # -- this adapter's runtime ------------------------------------------
    def install(self, sandbox=None) -> dict | None:
        """The install step this adapter needs run inside the episode.

        DELIBERATE DIVERGENCE FROM HARBOR, and the one place our adapter
        interface does not match theirs. Harbor's
        `BaseInstalledAgent.install(environment)` (harbor/src/harbor/agents/
        installed/base.py) EXECUTES the install: it holds a live environment
        handle and can exec into it as often as it likes.

        We cannot. This process is on the HOST and the sandbox has no exec
        channel: an episode is ONE `enroot start` whose overlay dies with it, so
        anything a second container installed would be thrown away, and enroot
        has no `exec` to reach into the running one. So `install()` RETURNS the
        check/install pair instead of running it; `core/trial.py` calls this and
        the blob carries the result in to the runner, which runs
        `check || install` after staging and before the agent starts -- the same
        point in the episode's life that Harbor's `install(environment)`
        occupies, reached the only way this sandbox allows.

        `sandbox` is unused and None at every current call site -- the blob is
        built before the container exists, so there is no `EpisodeSandbox` yet.
        It is in the signature so an adapter whose runtime genuinely needs
        host-side namespace work can be handed one later without changing the
        interface for the adapters that do not.

        None means nothing to install, which is the default for an adapter that
        declares no INSTALL.
        """
        script = self.install_script()
        if not script:
            return None
        # `latest` is an explicit update request, not permission to use an
        # arbitrary preinstalled version. Exact pins keep version-aware checks.
        check = "false" if self.requested_version == "latest" else self.install_check()
        if self.pinned_version and self.VERSION_CMD:
            # Verify the installed binary too: a successful vendor installer
            # must never silently substitute a newer release for an exact pin.
            verify = ('[ "$(' + self.VERSION_CMD + ')" = '
                      + shlex.quote(self.pinned_version) + ' ]')
            check = '(' + check + ') && ' + verify
            script += ('\n' + verify + ' || { echo "Installed harness version does not match pin" >&2; exit 1; }')
        return {"name": self.name(), "check": check,
                "install": script, "version_cmd": self.VERSION_CMD,
                "pinned_version": self.pinned_version or ""}

    def install_check(self) -> str:
        """Shell that exits 0 when this adapter's runtime is already usable.

        A method rather than a bare constant because a PINNED adapter has a
        stricter idea of "already there": the right binary, not any binary. The
        default is the constant, so an adapter with nothing to pin overrides
        nothing.
        """
        return self.INSTALL_CHECK

    def install_script(self) -> str:
        """Shell that puts this adapter's runtime in place."""
        return self.INSTALL
