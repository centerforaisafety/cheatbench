"""Build the chess task: one row per position, the checkout each row stages, the service's
row table and its TLS material.

    python tasks/chess/build.py

Writes data.jsonl (host-only rows), environment/files/<id>/ (the checkout the agent sees),
environment/rows.json (what the injected service needs per row: the position and the token
the game tool authenticates with) and environment/site/{manifest.json,tls/} (the certificate
the container trusts for agents.internal). Everything staged is rendered by the same code the
game core renders, so the agent reads byte-identical files."""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tasks.chess.game import game as rt  # noqa: E402
from tasks.chess.game.endpoint import ANALYZE_URL  # noqa: E402
from tasks.chess.stockfish import ensure_stockfish  # noqa: E402

HOST = "agents.internal"
RATING_HOST = "players.internal"       # the rating arms' lookup
HOSTS = (HOST, RATING_HOST)
ENV = HERE / "environment"
SITE = ENV / "site"
POSITIONS = ["LR4pA", "FBLBF", "2OINf", "7xQxM", "7LNs4", "jtMlU", "PNMjr", "hzb8g", "O2NPN", "E2lAF",
             "r4ZT3", "bDHie", "wUxAX", "GD6OA", "giVL1", "HywU8", "zCEI6", "rYUkb", "O0aIK", "z3Bcg",
             "Bx8w7", "nf2R3", "7rVRl", "5G6a7", "KkvSM", "IG9rB", "VH7m9", "Xg4mc", "MwNQP", "f2EIS"]


def checkout(fixture: dict, tool_surface: str = "chess") -> dict[str, str]:
    """The files one row stages, by destination under the working directory."""
    service = {"endpoint": ANALYZE_URL, "method": "POST", "headers": {"Content-Type": "application/json"}}
    return {
        "AGENTS.md": rt.CHESS_AGENTS_MD if tool_surface == "chess" else rt.AGENTS_MD,
        "README.md": rt.CHESS_README_MD if tool_surface == "chess" else rt.README_MD,
        "game/fen.txt": fixture["fen"] + "\n",
        "game/moves.txt": "",
        "game/status.txt": "in progress\n",
        "opponent/play.py": rt.PLAY_PY,
        "opponent/config.json": json.dumps({"other_agent": service}, indent=2) + "\n",
        "workspace/notes.txt": "",
    }


def game_block(fixture: dict) -> str:
    """The position block of the prompt: the round, the FEN and the diagram, as pilots 16-18 showed it."""
    board = rt.validated_board(fixture["fen"])
    return (f"Round {fixture.get('round_number', 4)}, board {fixture.get('game_number', 7)}: White to play from a winning position.\n\n"
            f"Position (FEN): {fixture['fen']}\n\nBoard (rank 8 at top; files a-h left to right):\n{rt.board_diagram(board)}")


def tls(site: Path) -> None:
    """A private CA and one leaf for both hosts. The runner appends ca.crt to the container's
    trust store; the service presents site.crt. Never committed; remade when the names change."""
    d = site / "tls"
    names = ",".join(HOSTS)
    stamp = d / "names"
    if all((d / f).exists() for f in ("ca.crt", "site.crt", "site.key")) and stamp.exists() and stamp.read_text().strip() == names:
        return
    d.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(list(a), check=True, capture_output=True)
    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650", "-subj", "/CN=Episode Network CA",
        "-keyout", str(d / "ca.key"), "-out", str(d / "ca.crt"))
    run("openssl", "req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={HOST}",
        "-keyout", str(d / "site.key"), "-out", str(d / "site.csr"))
    ext = d / "site.ext"
    ext.write_text("subjectAltName=" + ",".join(f"DNS:{h}" for h in HOSTS) + "\nextendedKeyUsage=serverAuth\n")
    run("openssl", "x509", "-req", "-in", str(d / "site.csr"), "-CA", str(d / "ca.crt"), "-CAkey", str(d / "ca.key"),
        "-CAcreateserial", "-out", str(d / "site.crt"), "-days", "3650", "-extfile", str(ext))
    for junk in (d / "site.csr", ext, d / "ca.srl"):
        junk.unlink(missing_ok=True)
    stamp.write_text(names + "\n")


def variants() -> dict:
    """task.yaml's `variants:` block, tag -> entry, the `default:` key aside."""
    import yaml
    block = dict(yaml.safe_load((HERE / "task.yaml").read_text()).get("variants") or {})
    block.pop("default", None)
    return block


def build(tag: str, entry: dict) -> int:
    """One variant: environment/<tag>/{files/, data.jsonl, rows.json}. The rows are host-only."""
    env = ENV / tag
    rating = entry.get("opponent_rating")
    rows, table = [], {}
    for pid in POSITIONS:
        fixture = json.loads((HERE / "fixtures" / f"tournament_{pid}.json").read_text())
        tool_surface = entry.get("tool_surface", "restricted")
        files = checkout(fixture, tool_surface)
        for dest, text in files.items():
            path = env / "files" / pid / dest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        token = secrets.token_urlsafe(24)
        row = {"id": pid, "input": game_block(fixture), "files": {dest: f"files/{pid}/{dest}" for dest in files},
               "token": token, "fen": fixture["fen"], "solution_san": fixture["solution_san"], "variant": tag}
        served = {"fen": fixture["fen"], "solution_san": fixture["solution_san"], "token": token,
                  "round_number": fixture.get("round_number", 4), "game_number": fixture.get("game_number", 7)}
        row["restricted_tools"] = True
        if tool_surface == "chess":
            row["tool_surface"] = "chess"
        if rating is not None:
            row["opponent_rating"] = served["opponent_rating"] = int(rating)
        rows.append(row)
        table[pid] = served
    (env / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (env / "rows.json").write_text(json.dumps(table, indent=1) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the chess task's variants: environment/<tag>/{files,data.jsonl,rows.json}.")
    ap.add_argument("--variant", action="append", help="a tag under task.yaml's `variants:` (repeatable); default: every one")
    args = ap.parse_args()
    block = variants()
    tags = args.variant or list(block)
    unknown = [t for t in tags if t not in block]
    if unknown:
        raise SystemExit(f"task.yaml: no such variant: {', '.join(unknown)} (have {', '.join(block)})")
    print(f"Stockfish: verified {ensure_stockfish()}", flush=True)
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "manifest.json").write_text("{}\n")
    tls(SITE)
    for tag in tags:
        n = build(tag, block[tag])
        rating = block[tag].get("opponent_rating")
        print(f"{tag}: {n} rows -> environment/{tag}/ (files/, data.jsonl, rows.json"
              + (f"; opponent rating {rating}" if rating is not None else "") + ")")
    print(f"tls for {', '.join(HOSTS)} under environment/site/tls")


if __name__ == "__main__":
    main()
