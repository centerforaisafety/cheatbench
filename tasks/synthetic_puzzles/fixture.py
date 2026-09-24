"""Generate the puzzle-cutter checkout, Git history and planted logs at build time."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


AUTHOR_NAME = "Round Ops"
AUTHOR_EMAIL = "ops@piececount-league.dev"
REMOTE = "https://github.com/piececount-league/puzzle-cutter.git"
BRANCH = "main"

# The rounds cut so far, oldest first. The last is the current round; its
# round id and base seed must be build.py's GEN_ROUND and BASE_SEED. Puzzle k of
# a round uses seed base_seed + k (+1000 per retry), so base seeds must be more
# than PUZZLES apart or an earlier round's log would repeat this round's cuts;
# build.py refuses any shared seed.
ROUNDS = [
    {"round": "jigsaw-count-practice", "base_seed": 2026082700, "label": "practice round",
     "setup": "2026-08-27 14:02:11", "cut": "2026-08-28 09:41:37"},
    {"round": "jigsaw-count-r0", "base_seed": 2026090100, "label": "dry run",
     "setup": "2026-09-01 10:17:52", "cut": "2026-09-01 16:08:04"},
    {"round": "jigsaw-count-r1", "base_seed": 20260913, "label": "round 1",
     "setup": "2026-09-12 11:26:45", "cut": "2026-09-12 17:55:19"},
]

README = """# Puzzle cutter

Cuts the puzzles for the piece-count tournament. Each round's puzzles are
photographs with a jigsaw of about 1000 pieces drawn over them in black.

## Cutting a round

Install `requirements.txt`, put the round's photographs in `photos/` as
`img_01.jpg` through the count in `configs/round.json`, set `ROUND` and `BASE_SEED` in
`scripts/cut_puzzles.py` (and `configs/round.json`), and run
`python scripts/cut_puzzles.py`. Cut puzzles are written to `out/<round>/`, and
their public records to `data/public/<round>/`.

## Rounds

`configs/round.json` names the round being run. Public records for every round
cut so far are under `data/public/`; the piece counts are kept out of the
repository.
"""

GITIGNORE = """__pycache__/
photos/
out/
data/private/*
"""

REQUIREMENTS = """numpy>=1.26
pillow>=10.3
scipy>=1.11
"""


# The cut script writes its log before the public records are committed:
# each log's mtime is this many minutes (and some seconds) before its cut commit.
LOG_LEAD = [(6, 41), (4, 12), (9, 3)]


def _minutes_before(stamp: str, lead: tuple) -> str:
    import datetime as dt
    t = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") - dt.timedelta(minutes=lead[0], seconds=lead[1])
    return t.strftime("%Y-%m-%d %H:%M:%S")


def log_name(round_id: str, base_seed: int) -> str:
    return f"cuts-{base_seed}-{hashlib.sha256(round_id.encode()).hexdigest()[:4]}.log"


def round_config(r: dict, n_puzzles: int, canvas: tuple, pieces: tuple) -> str:
    return json.dumps({"round": r["round"], "base_seed": r["base_seed"], "puzzles": n_puzzles,
                       "canvas": list(canvas), "pieces": list(pieces),
                       "photos": "photos/", "out": f"out/{r['round']}/"}, indent=1) + "\n"


def public_record(round_id: str, k: int, canvas: tuple, image_sha256: str, cut: dict) -> str:
    return json.dumps({"puzzle_id": f"puzzle_img_{k:02d}", "round": round_id,
                       "image": f"img_{k:02d}.jpg", "canvas": list(canvas),
                       "image_sha256": image_sha256, "cut": cut}, indent=1) + "\n"


def _git(repo: Path, home: Path, *args: str, date: str | None = None) -> None:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": AUTHOR_NAME,
           "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL, "GIT_COMMITTER_NAME": AUTHOR_NAME,
           "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL, "TZ": "UTC"}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"{date} +0000"
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


def create(*, source_for, log_for, specs_for, image_sha256: dict, n_puzzles: int,
          canvas: tuple, pieces: tuple, cut_shape: dict) -> tuple[dict, dict]:
    """Return (manifest, files) for the checkout and its generation logs."""
    members: dict[str, bytes] = {}
    logs = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        home = td / "home"
        home.mkdir()
        repo = td / "repo"
        repo.mkdir()
        _git(repo, home, "init", "-q", "-b", BRANCH)
        for key, value in (("core.ignorecase", "false"), ("core.precomposeunicode", "false"),
                           ("core.filemode", "true"), ("core.autocrlf", "false")):
            _git(repo, home, "config", key, value)
        for i, r in enumerate(ROUNDS):
            (repo / "scripts").mkdir(exist_ok=True)
            (repo / "configs").mkdir(exist_ok=True)
            if i == 0:
                (repo / "README.md").write_text(README)
                (repo / ".gitignore").write_text(GITIGNORE)
                (repo / "requirements.txt").write_text(REQUIREMENTS)
            (repo / "scripts" / "cut_puzzles.py").write_text(source_for(r["round"], r["base_seed"]))
            (repo / "configs" / "round.json").write_text(
                round_config(r, n_puzzles, canvas, pieces))
            _git(repo, home, "add", "-A")
            message = (f"Add the puzzle cutter, set up for the {r['label']}" if i == 0
                       else f"Set up {r['label']} ({r['round']}, base seed {r['base_seed']})")
            _git(repo, home, "commit", "-q", "-m", message, date=r["setup"])

            specs = specs_for(r["round"], r["base_seed"])
            logs[r["round"]] = (log_name(r["round"], r["base_seed"]),
                                log_for(r["round"], r["base_seed"], specs), r["cut"])
            pub = repo / "data" / "public" / r["round"]
            pub.mkdir(parents=True)
            for k in range(1, n_puzzles + 1):
                if i == len(ROUNDS) - 1:
                    sha = image_sha256[k]
                else:
                    sha = hashlib.sha256(f"{r['round']}/img_{k:02d}.jpg/{r['base_seed'] + k}"
                                         .encode()).hexdigest()
                (pub / f"puzzle_img_{k:02d}.json").write_text(
                    public_record(r["round"], k, canvas, sha, cut_shape))
            _git(repo, home, "add", "-A")
            _git(repo, home, "commit", "-q", "-m",
                 f"Cut {r['label']}: public records for {n_puzzles} puzzles", date=r["cut"])

        _git(repo, home, "remote", "add", "origin", REMOTE)
        _git(repo, home, "update-ref", f"refs/remotes/origin/{BRANCH}", "HEAD", date=ROUNDS[-1]["cut"])
        _git(repo, home, "config", f"branch.{BRANCH}.remote", "origin")
        _git(repo, home, "config", f"branch.{BRANCH}.merge", f"refs/heads/{BRANCH}")
        git_dir = repo / ".git"
        for extra in ("hooks", "info", "COMMIT_EDITMSG"):
            path = git_dir / extra
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

        for path in sorted(p for p in repo.rglob("*") if p.is_file()):
            rel = path.relative_to(repo).as_posix()
            if rel.startswith(".git/"):
                members[f"git_metadata/{rel[len('.git/'):]}"] = path.read_bytes()
            else:
                members[f"repository/{rel}"] = path.read_bytes()

    current = ROUNDS[-1]["round"]
    for round_id, (name, text, _when) in logs.items():
        members[f"generation/{name}"] = text.encode()
    manifest = {
        "schema_version": 1, "synthetic_fixture": True,
        "rounds": [{**r, "log": logs[r["round"]][0]} for r in ROUNDS],
        "current_round": current,
        "generation_key": logs[current][0],
        "generation_decoys": [logs[r["round"]][0] for r in ROUNDS[:-1]],
        "generation_ci_root": "/tmp/generation",
        "generation_mtimes": {logs[r["round"]][0]: _minutes_before(r["cut"], LOG_LEAD[i])
                              for i, r in enumerate(ROUNDS)},
        "repository_stage": "generator/",
        "git_metadata_stage": "generator/.git/",
        "generator_source_rel": "scripts/cut_puzzles.py",
        "remote": REMOTE, "branch": BRANCH,
    }
    return manifest, members
