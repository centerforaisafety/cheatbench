"""OURS: no Harbor counterpart -- Harbor stages a task's whole `environment/` dir
with `BaseEnvironment.upload_dir` (docker cp, or a tar stream over `exec`), a
channel into a live container we do not have; and a Harbor task is one directory,
so there is no per-row `files` map there to stage in the first place.

Copy a row's `files` map into the container.

A data.jsonl row carries

    "files": {"reference_files/in.pdf": "files/inputs/9a3f.pdf",
              "gold_deliverable_files/gold.pdf": "files/golds/c71b.pdf"}

destination inside the agent's working directory -> path relative to that
task's `environment/`. That map is the ONLY thing the agent is meant to see;
`data.jsonl` itself (which holds answers and gold paths) is host-only and is
never staged.

A row may also carry `files_abs`, the same map with ABSOLUTE destinations, for
a task that plants something deliberately OUTSIDE the working directory:

    "files_abs": {"/tmp/ursa/X404-133-0167/answer.txt": "files/keys/X404-133-0167.txt"}

It is a separate key so that escaping /workspace is declared in the row rather
than smuggled through a `../` in `files`.

A row may also carry `file_mtimes`, keyed by the SAME destinations, to age what
it stages:

    "file_mtimes": {"/tmp/ursa/X404-133-0167/notes.md": "2026-03-02 09:14"}

Without it every staged file carries the timestamp of the moment the episode
started, seconds apart, which is a tell: a folder whose own log describes work
done over three weeks cannot have been written all at once. Dates are parsed on
the HOST (UTC) and shipped as an epoch int, so the container half stays dumb.
Directories this module CREATES are aged too, to the newest file beneath them --
a directory shows when it last changed, and one dated today under a parent dated
in March is the same tell one level up. Pre-existing directories are never
touched: /home/user belongs to the image, not to a row.

A row may also carry `file_modes`, keyed the same way, for a staged file that has
to be executable:

    "file_modes": {"/home/user/work/promotion_worktest/bin/pull-designs": "755"}

Everything else stays 0644. This exists because a task can stage a small tool a
colleague left behind, and a script sitting in a `bin/` directory without its
executable bit is a tell -- the agent would have to invoke it through `sh`, which
no real user would do.

This module has two halves that must stay together, because the second one runs
somewhere the first one's filesystem does not exist:

  collect()   HOST side. Reads the bytes out of `environment/` and base64s them
              into the stdin blob.
  write()     CONTAINER side. Materialises them before the agent starts, so its
              very first `ls` already sees them.

Nothing is bind-mounted. A bind would put a host path into the container's
/proc/self/mounts, which image/audit.sh forbids; the bytes ride the same stdin
payload as the runner instead, and die with the per-episode overlay.

Stdlib only: the container half is exec'd from the stdin blob inside an image
that holds none of our code.
"""
from __future__ import annotations

import base64
import calendar
import os
import time


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------
def collect(row: dict, env_root) -> list[dict]:
    """The staging payload for one row: [{"name": dest, "b64": ...}, ...].

    `env_root` is the task's `environment/` directory. A source that escapes it
    is refused: a row must not be able to name an arbitrary host file.
    """
    from pathlib import Path

    env_root = Path(env_root).resolve()
    out: list[dict] = []
    mtimes = row.get("file_mtimes") or {}

    modes = row.get("file_modes") or {}

    def _mode(dest):
        """`file_modes[dest]` as an int, or None. Accepts 493, "755" or "0o755"."""
        raw = modes.get(dest)
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw
        text = str(raw)
        try:
            return int(text, 8)
        except ValueError:
            raise ValueError(f"file_modes[{dest!r}]: want an octal mode like "
                             f"\"755\", got {raw!r}") from None

    def _when(dest):
        """`file_mtimes[dest]` as an epoch, or None. UTC, so a build is not
        reproducible only on the machine that ran it."""
        raw = mtimes.get(dest)
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return int(raw)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return int(calendar.timegm(time.strptime(str(raw), fmt)))
            except ValueError:
                continue
        raise ValueError(f"file_mtimes[{dest!r}]: want YYYY-MM-DD[ HH:MM[:SS]], "
                         f"got {raw!r}")

    def _src(src_rel) -> str:
        src = (env_root / str(src_rel)).resolve()
        if env_root not in src.parents and src != env_root:
            raise ValueError(f"staging source escapes {env_root}: {src_rel!r}")
        if not src.is_file():
            raise FileNotFoundError(f"staging source missing: {src}")
        return base64.standard_b64encode(src.read_bytes()).decode()

    # `files`: destinations INSIDE the working directory. A leading "/" is
    # stripped and ".." refused, so a row cannot reach outside /workspace by
    # accident. This is the default and what most tasks use.
    for dest, src_rel in (row.get("files") or {}).items():
        dest = str(dest).lstrip("/")
        if ".." in dest.split("/"):
            raise ValueError(f"staging destination escapes the workdir: {dest!r}")
        rec = {"name": dest, "b64": _src(src_rel)}
        mode = _mode(dest)
        if mode is not None:
            rec["mode"] = mode
        when = _when(dest)
        if when is not None:
            rec["mtime"] = when
        out.append(rec)

    # `files_abs`: destinations OUTSIDE the working directory, by absolute
    # path, for a task that deliberately plants something where a listing of
    # the workspace will not show it. It is a separate key so the escape is
    # declared in the row, not smuggled through `files`. The path must be
    # absolute, ".." is still refused, and the container's uid must be able to
    # write there (so /tmp works; / does not).
    for dest, src_rel in (row.get("files_abs") or {}).items():
        dest = str(dest)
        if not dest.startswith("/") or ".." in dest.split("/"):
            raise ValueError(f"files_abs destination must be absolute and clean: {dest!r}")
        if dest.startswith("/workspace/"):
            raise ValueError(f"files_abs is for OUTSIDE the workdir; use `files`: {dest!r}")
        rec = {"name": dest, "abs": True, "b64": _src(src_rel)}
        mode = _mode(dest)
        if mode is not None:
            rec["mode"] = mode
        when = _when(dest)
        if when is not None:
            rec["mtime"] = when
        out.append(rec)

    # A date for something that is not being staged is a typo in the row -- most
    # likely a destination that was renamed on one side only. Catch it here,
    # where the row is still visible, not as a file that silently stayed new.
    staged_names = {r["name"] for r in out}
    for key, seen in (("file_mtimes", mtimes), ("file_modes", modes)):
        unknown = set(seen) - staged_names
        if unknown:
            raise ValueError(f"{key} names paths that are not staged: "
                             f"{sorted(unknown)[:3]}")
    return out


# --------------------------------------------------------------------------
# container side
# --------------------------------------------------------------------------
def write(files: list, root: str = "/workspace") -> dict:
    """Materialise the payload under `root`. Returns dest -> absolute path.

    Parent directories are created 0755 and files 0644, so everything is
    world-readable: the point of a plant is that a curious agent CAN open it.

    A record carrying `mtime` is aged to it, and so is any directory THIS call
    had to create -- to the newest file staged beneath it. See the module
    docstring for why.
    """
    staged: dict = {}
    created: set = set()      # directories that did not exist before this call
    newest: dict = {}         # created dir -> newest mtime staged beneath it
    for rec in files or []:
        dest = rec["name"]
        # An `abs` record names its own absolute destination and ignores `root`.
        is_abs = bool(rec.get("abs"))
        q = dest if is_abs else os.path.join(root, dest)
        d = os.path.dirname(q)
        base = "/" if is_abs else root
        if d:
            # Which components are missing has to be read BEFORE makedirs, so the
            # ageing pass below can tell a directory this call brought into being
            # from one the image already had. /home/user is not ours to re-date.
            cur = base
            for part in os.path.relpath(d, base).split(os.sep):
                if not part or part == ".":
                    continue
                cur = os.path.join(cur, part)
                if not os.path.isdir(cur):
                    created.add(cur)
            os.makedirs(d, exist_ok=True)
            # makedirs honours umask; be explicit, and walk up so an
            # intermediate directory is listable too.
            cur = base
            for part in os.path.relpath(d, base).split(os.sep):
                if not part or part == ".":
                    continue
                cur = os.path.join(cur, part)
                try:
                    os.chmod(cur, 0o755)
                except OSError:
                    pass
        with open(q, "wb") as fh:
            fh.write(base64.b64decode(rec["b64"]))
        try:
            os.chmod(q, rec.get("mode", 0o644))
        except OSError:
            pass
        when = rec.get("mtime")
        if when is not None:
            try:
                os.utime(q, (when, when))
            except OSError:
                pass
            cur = os.path.dirname(q)
            while cur and cur != "/" and cur != base:
                if cur in created:
                    newest[cur] = max(newest.get(cur, when), when)
                cur = os.path.dirname(cur)
        staged[dest] = q

    # Directories LAST, and only now: writing a file into a directory bumps that
    # directory's mtime, so anything set during the loop above would be undone by
    # the next file to land in it.
    for d, when in sorted(newest.items()):
        try:
            os.utime(d, (when, when))
        except OSError:
            pass
    return staged
