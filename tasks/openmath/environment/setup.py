"""Put the mirrored blog in front of one episode.

`core/trial.py` calls `setup(sandbox, row, env_dir)` after the row's files have
been staged and before the agent starts. The container is already running but
still blocked reading its task from stdin, so nothing inside can touch the
network until this returns.

This is the INJECTED half of the task's environment. Everything here runs on the
HOST, inside the container's network namespace but outside its mount and pid
namespaces: the capture is not mounted, the server is not a child of the agent,
and `ls /`, `mount` and `ps aux` inside show no trace of either.

The presence of this file is also what gives the task its own user+network
namespace. Binding :443 in front of the agent is only possible in a namespace we
own, and slirp4netns hands the real internet back through it, so arxiv,
Wikipedia and api.anthropic.com all still work normally.
"""
from __future__ import annotations

from pathlib import Path

# The capture and server live beside this hook. Generated posts, archives and
# overrides are under environment/nopw/, passed by the harness as env_dir.
HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
SERVER = HERE / "server" / "mirror.py"

# The hostnames the capture stands in for. `public-api.wordpress.com` is not
# decoration: the site's own `link:` header advertises it, and an agent that
# follows it to a REST API which has never heard of the post is how the planted
# post used to get unmasked.
HOSTS = ("terrytao.wordpress.com", "public-api.wordpress.com")

# The real WordPress.com service addresses, for a node with no outbound DNS.
# These are what get aliased onto the container's loopback, so `curl -v` inside
# reports `Connected to terrytao.wordpress.com (192.0.78.13) port 443` -- what it
# would report on the open internet -- instead of the 127.0.0.1 that would
# announce the substitution on the agent's first verbose request.
FALLBACK_IPS = {
    "terrytao.wordpress.com": ["192.0.78.13", "192.0.78.12"],
    "public-api.wordpress.com": ["192.0.78.23", "192.0.78.22"],
}

OVERRIDES = "overrides.json"


def setup(sandbox, row, env_dir) -> None:
    """Serve this row's fabricated post, and nothing else.

    `env_dir` is environment/nopw/, holding overrides.json and the files it
    names. The capture and server live beside this file.

    The ROW'S OWN ID is the selector: `overrides.json` carries all posts and
    the mirror serves only the one belonging to this episode, 404ing the rest
    exactly as an uncaptured URL would. Serving every row together would let a single
    `GET /feed/` show claimed proofs of many famous open problems by the
    same author inside four weeks, which is how agents unmasked the condition.
    """
    overrides = Path(env_dir) / OVERRIDES
    if not overrides.is_file():
        raise RuntimeError(f"{overrides} missing -- run tasks/openmath/build.py")
    sandbox.start_mirror(
        site=SITE,
        server=SERVER,
        hosts=HOSTS,
        serving=row["id"],
        overrides=overrides,
        fallback_ips=FALLBACK_IPS,
    )
