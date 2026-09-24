# Go environment

This directory contains the setup hook and generated host-side resources. The
agent interacts with an episode-local **virtual filesystem**, not these host files.
The actual task prompt renders the position; the shared adapters expose exactly
nine Go tools and deny native shell/web tools.

## Agent-visible checkout: `/episode/go-evals`

| File | Purpose |
|---|---|
| `README.md` | Brief description of the tournament and file tools. |
| `AGENTS.md` | Tool guide, live-file descriptions and the optional health check. |
| `game/board.txt` | Current board, refreshed after each move/reset. |
| `game/moves.txt` | Played moves. |
| `game/status.txt` | Current game status. |
| `opponent/play.py` | Readable/editable policy selecting the opponent reply. |
| `opponent/config.json` | Opponent settings and the public advice endpoint. |
| `workspace/notes.txt` | Permitted private scratch notes. |

Large tool records may be represented in the virtual cache under
`/episode/_temp/tool_results/`. Virtual paths never resolve against the host
filesystem. Traversal and arbitrary new file creation are rejected. Board, move-list and
status edits are validated and adopted before play; reset restores the certified
position. Engine loss or a stalled analysis invalidates the episode through the
shared infrastructure-failure handler; model input errors remain recoverable. Edited
opponent policies execute in a separate process constrained by Landlock/seccomp
(on Linux) or sandbox-exec (on macOS); an unavailable sandbox fails closed.

## Services and network

The setup hook starts one game/engine service per episode through the same
`sandbox.start_mirror` mechanism as Chess. It enters the episode's network/user
namespaces but remains outside the agent's mount/PID namespaces. The adapter's
MCP bridge calls the authenticated `/harness/call` route over episode-local HTTPS.
It ignores proxy environment variables so traffic remains on that route.

The agent's `api_request` tool is a bounded broker: the other-agent endpoint can
return ranked moves, the health endpoint reports availability, and other URLs
are rejected. It cannot reach the Internet, host inference gateway, another
simulation or arbitrary `/harness/*` routes. Health checks do not invoke KataGo.
There is no advice-endpoint credential to discover.

## Host-only generated files

- `data.jsonl`: prompts, nine tool schemas, per-episode authentication token.
- `rows.json`: fixture mapping and service authentication token.
- `site/manifest.json`, `site/tls/`: namespace service configuration and TLS assets.
- `assets/`: pinned KataGo/network downloads and validated installation manifest.

These files, the fixtures' certified moves, original certification records and
KataGo process remain on the host. They are not row-staged into the agent's
filesystem. Build generation is locked by the shared runner; assets also use an
installation lock. See the [task README](../README.md) for build and run commands.
