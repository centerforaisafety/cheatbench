"""The harness refuses an engine that is not the one the positions were certified on."""
import asyncio, json

import pytest

from tasks.go import engine_signature as sig
from tasks.go.katago_api import KataGoEngine

RECORDED = json.loads(sig.SIGNATURE_PATH.read_text())


def test_engine_pins_full_precision_in_config_and_provenance(tmp_path):
    engine = KataGoEngine(katago="/unused/katago", network="/unused/net.bin.gz",
                          work_dir=tmp_path)
    assert "useFP16 = false" in engine.config_file.read_text().splitlines()
    assert engine.settings["fp16"] is False


class Fake:
    """An engine whose probe answers are the recorded ones shifted by `offset`."""

    def __init__(self, offset=0.0, top=None, network=None):
        self.katago, self.network, self.settings = "/nonexistent/katago", network or RECORDED["network"], {}
        self.offset, self.top = offset, top

    async def analyse(self, moves, **kw):
        n = len(self.queries) if hasattr(self, "queries") else 0
        self.queries = getattr(self, "queries", []) + [moves]
        rec = RECORDED["probes"][n]
        return {"rootInfo": {"winrate": rec["winrate"] + self.offset, "scoreLead": rec["score_lead"]},
                "moveInfos": [{"move": self.top or rec["top_move"], "order": 0}]}


def test_the_recorded_engine_is_exact_and_a_different_backend_is_compatible():
    assert asyncio.run(sig.verify(Fake()))["status"] == "exact"
    out = asyncio.run(sig.verify(Fake(offset=5e-4)))
    assert out["status"] == "compatible" and 4e-4 < out["max_winrate_delta"] < 6e-4


def test_a_drifted_engine_a_different_move_and_a_different_network_are_refused():
    with pytest.raises(RuntimeError, match="not be comparable"):
        asyncio.run(sig.verify(Fake(offset=0.01)))
    with pytest.raises(RuntimeError, match="not be comparable"):
        asyncio.run(sig.verify(Fake(top="A1")))
    with pytest.raises(RuntimeError, match="certified on"):
        asyncio.run(sig.verify(Fake(network="other-net.bin.gz")))


def test_a_missing_signature_file_says_how_to_record_one(tmp_path, monkeypatch):
    monkeypatch.setattr(sig, "SIGNATURE_PATH", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="--record"):
        asyncio.run(sig.verify(Fake()))
