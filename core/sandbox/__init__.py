"""OURS: no Harbor counterpart adopted -- Harbor's environment layer
(`src/harbor/environments/`) is organised around a live `exec()` channel, and an
episode here is one `enroot start` with no second command, so none of it fits.

The sandbox layer: the container an episode runs in, and the image it runs.

Three files, split by what they are responsible for:

    paths.py     where things live, all of it environment-variable driven
    runtime.py   starting the container, and injecting into a live one
    build.py     realising the repo-root Dockerfile as the image, on demand

`import core.sandbox as sandbox` gets the same names it always did; this package
replaced a single module and re-exports its whole surface, so nothing that used
it had to change.

The enroot runtime configuration -- `enroot/sysconf/{mounts.d,hooks.d}` -- sits
in this package rather than beside the Dockerfile, because it configures the
RUNTIME and not the image: a fresh /proc bound to our pid namespace, no stock
hooks copying the operator into the container, a tmpfs /dev with no GPUs in it.
Every line of those two files is load-bearing and they are commented in place.

enroot is the only supported runtime.
"""
from .paths import (DEFAULT_DATA_PATH, DEFAULT_IMAGE, DEFAULT_IMAGE_NAME,
                    DEFAULT_RUNTIME_ROOT, IMAGES_DIR, ROOT, SCRATCH, SYSCONF,
                    USERCONF, image_path, resolve_image)
from .build import (DOCKERFILE, audit_built_image, audit_probes, build_image,
                    ensure_image, ensure_image_async, parse_dockerfile)
from .runtime import (AGENT_HOME, AGENT_NAME, AGENT_UID, EXTRA_PATH,
                      SHELL_BOOTSTRAP, EpisodeSandbox, MirrorSandbox,
                      container_argv, ensure_path, find_container_pid,
                      net_pid_for, preflight, resolve_site_ips, run_shell,
                      spawn_env, start_slirp)

from .egress import NetworkPolicy, api_loopback_ports, hosts_for, parse_network

__all__ = [
    "NetworkPolicy", "api_loopback_ports", "hosts_for", "parse_network",
    "AGENT_HOME", "AGENT_NAME", "AGENT_UID", "DEFAULT_DATA_PATH",
    "DEFAULT_IMAGE", "DEFAULT_IMAGE_NAME", "DEFAULT_RUNTIME_ROOT", "DOCKERFILE",
    "EXTRA_PATH", "IMAGES_DIR", "ROOT", "SCRATCH", "SHELL_BOOTSTRAP", "SYSCONF",
    "USERCONF", "EpisodeSandbox", "MirrorSandbox", "audit_built_image",
    "audit_probes", "build_image", "container_argv", "ensure_image",
    "ensure_image_async", "ensure_path", "find_container_pid", "image_path",
    "net_pid_for", "parse_dockerfile", "preflight", "resolve_image",
    "resolve_site_ips", "run_shell", "spawn_env", "start_slirp",
]
