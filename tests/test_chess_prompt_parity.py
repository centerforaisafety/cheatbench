"""Freeze the task prompt against saved local chess runs, not regenerated old code.

The first ten were captured from both fable_implicit_<id>/prompt.md and
opus_implicit_<id>/prompt.md in the saved tournament reference on 2026-09-15.
Both saved copies matched for every position. Only the checkout root and reset
tool spelling are normalized; no other whitespace/content changes are hidden.
The twenty positions added on 2026-09-16 have no saved run; their hashes are
build.py's render on the day they were added, frozen the same way.
This does not assert equivalence of the native CLI system context or tools.
"""
import hashlib
import json
from pathlib import Path

import pytest

from tasks.chess.build import POSITIONS, game_block

ROOT = Path(__file__).resolve().parents[1]
CHESS_PROMPTS = {
    "LR4pA": "8ff4bb14cf63e80ab3cc4c3bd26278d16e79a71f2b1c2be5a96606457d7ea99a",
    "FBLBF": "dfbf6fd5e77849ce4d64155b6c3fa18301ee83574114e9f616b357f3bd668a1e",
    "2OINf": "e62d51135844a3b476bdd3a0ac75624ae91eb7d8c41068dcc5365b0d7dc044d5",
    "7xQxM": "b856f5375c26f3e526eba74cc05394ff9d922ed5dc47468fb64fca19f2217e62",
    "7LNs4": "719a64c0238e639c6858f59c1381034a4b8455eed2c81f88336ecb1b3e9638f7",
    "jtMlU": "babe5f084c337a1803b9d1757362985f71a3707b2a4d083f2afc243834014f29",
    "PNMjr": "f1351c677e1bd0b8970d526f5ee6784c0efe91b314227c8e0551171b4cfc0d67",
    "hzb8g": "b776c9587d1a26b046b64c554d6dccea301ca2d1a7556fcec8ec5eaa3ede7a62",
    "O2NPN": "2e1a2db5134d669140f8a6db6c17f6b6e9de180748d0b8d342f4ca74ab5dd586",
    "E2lAF": "0ff5010fa3a530f779828c37ced701d051cf4917f1d7269f99272026f5a49dd9",
    "r4ZT3": "fae954a5c0a0a8d4e4d95c510f4246f12949973db14431b0d940d5c6a36cc22b",
    "bDHie": "cf96b390f29e155b4956dc8b1d6b465e54fd9e384b8675d23d1a303a95e05122",
    "wUxAX": "704b04887e28f70ae95d3a3cf025c95a01f6fbfd642cfe182513758e05735184",
    "GD6OA": "37b86d393eb2d1158e4d6288cefb8bf3a145a9a023fbd0bf7a3719e2d83bde88",
    "giVL1": "ef2e773bccb6d91c0ff3979874c1c5a5ee0b087e0e09fd8485d4c22d3ab2af0f",
    "HywU8": "7252f7af020c2a1ddff70f63cc3dd5ca13ff1ffe9e9a3728f339f0f1e1b9afaa",
    "zCEI6": "1fbfe86adccfc93366fd9f28ebab0adf3e2ac3fc8a176b00c3b0203eb399e8c8",
    "rYUkb": "24095bc70c740aabc0ba9e98ed0338fefc4c079f87af96cdf923bdab76d37543",
    "O0aIK": "34fffe27f2d4e7d245bc92d3d80c08054d6c8c951734c7f88404a3b1535765de",
    "z3Bcg": "2a8db5480ef794db8dfc6e876a067cb5c44f01b0e08e9e1d218b3e6bfd8cce23",
    "Bx8w7": "33a3e7ed982b878cf4a67906a431e637ce4a20d740c9998e950f136d636cff01",
    "nf2R3": "2167c2bf95f3b48c47b6d3c5d1758e9e8f48b928134070f2b26ae135a29b761a",
    "7rVRl": "892a02b9ad96f8dcaab9826881db8c049c29d243372087a22e9c1554d6f867ee",
    "5G6a7": "911fe4e842f7b62b12f9c8b539f93eb2e5fc7efe3ae436f302cf9bfb3bab3076",
    "KkvSM": "332119eb9f9bfe657ff891f24c6da02426f6961f29f572f623a87320133002b4",
    "IG9rB": "64ea6f9671714b2963dc9984c3dffbaf4d43400dff5047db8eb516acb452d81f",
    "VH7m9": "70ab754fd445bd6e79f67414596e0f2287820e58f20f41f643710212dcff3b51",
    "Xg4mc": "282b9da0badbc9f7bd597819347d57a8d1284606bbaadd185b807b04b8c7eb0d",
    "MwNQP": "b4d39cbfb321981ed6f9bb8acba1850d9fd3bb33c73d6959feb170d6ea7bb4dd",
    "f2EIS": "3520cb2d7afcc7bcb01953f75e4e84af2600d454ffa2a4c5382c8844048a9e60",
}


@pytest.mark.parametrize("template_name", ["instruction.md", "instruction_chess.md"])
@pytest.mark.parametrize("position", POSITIONS)
def test_task_prompt_matches_saved_chess(position, template_name):
    fixture = json.loads((ROOT / f"tasks/chess/fixtures/tournament_{position}.json").read_text())
    template = (ROOT / "tasks/chess/prompts" / template_name).read_text()
    prompt = template.replace("{{input}}", game_block(fixture)).strip()
    prompt = prompt.replace("`/workspace`", "`/episode/chess-evals`")
    prompt = prompt.replace("`play_move` with `reset`", "`reset_game`")
    assert hashlib.sha256(prompt.encode()).hexdigest() == CHESS_PROMPTS[position]


def test_baseline_covers_every_current_position():
    assert set(POSITIONS) == set(CHESS_PROMPTS)
