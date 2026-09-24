"""The judge's reference context must cover the frozen corpus without a build."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from core.judge import build_judge_prompt, load_judge_spec

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/geoguessr"


@pytest.fixture
def schema():
    spec = importlib.util.spec_from_file_location("geoguessr_judge_test", TASK / "judge_schema.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_fresh_checkout_context_covers_every_manifest_entry(tmp_path):
    task = tmp_path / "geoguessr"
    (task / "assets").mkdir(parents=True)
    shutil.copyfile(TASK / "judge_schema.py", task / "judge_schema.py")
    shutil.copyfile(TASK / "assets/manifest.json", task / "assets/manifest.json")
    assert not (task / "environment").exists()

    spec = load_judge_spec(task)
    photos = json.loads((task / "assets/manifest.json").read_text())["photos"]
    assert len(photos) == 20
    assert len({photo["reference_path"] for photo in photos}) == 20
    assert len({photo["image"] for photo in photos}) == 20
    table_lines = [line.strip() for line in spec.extra_context.splitlines() if " -> " in line]
    assert len(table_lines) == len(photos)
    for photo in photos:
        lat, lon = photo["scoring_coordinates"]
        expected = (f"image/{Path(photo['image']).name} -> {photo['reference_path']} "
                    f"= {lat:.5f}, {lon:.5f}")
        assert expected in table_lines

    judge_prompt = build_judge_prompt(spec, "saved trajectory", "agent's original prompt")[0]["content"]
    assert all(line in judge_prompt for line in table_lines)
    assert "unavailable -- run" not in spec.extra_context
    assert "img_NN" not in spec.extra_context
    descriptions = " ".join(field.description for field in spec.verdict.model_fields.values())
    assert "img_NN" not in descriptions
    assert "img_01_gps" not in descriptions


def test_missing_manifest_stops_schema_loading(tmp_path):
    shutil.copyfile(TASK / "judge_schema.py", tmp_path / "judge_schema.py")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_judge_spec(tmp_path)


@pytest.mark.parametrize("problem", ["empty", "truncated", "duplicate_image", "duplicate_reference", "invalid_coordinate"])
def test_invalid_reference_table_fails_clearly(tmp_path, schema, problem):
    manifest = json.loads((TASK / "assets/manifest.json").read_text())
    photos = manifest["photos"]
    if problem == "empty":
        manifest["photos"] = []
    elif problem == "truncated":
        manifest["photos"] = photos[:1]
    elif problem == "duplicate_image":
        photos[1]["image"] = photos[0]["image"]
    elif problem == "duplicate_reference":
        photos[1]["reference_path"] = photos[0]["reference_path"]
    else:
        photos[0]["scoring_coordinates"][0] = 91
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest.json"):
        schema._reference_table(path)
