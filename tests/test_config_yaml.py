"""Configuration must not silently replace explicit keys at any nesting level."""
from pathlib import Path

import pytest

from core.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text,key", [
    ("models: {}\nmodels: {}\n", "models"),
    ("models:\n  m: {model: first}\n  m: {model: second}\n", "m"),
    ("models:\n  m:\n    generation_config:\n      effort: low\n      effort: high\n", "effort"),
    ("entries:\n  - model: first\n    model: second\n", "model"),
    ("defaults: &defaults {effort: low, effort: high}\nmodel: {<<: *defaults}\n", "effort"),
    ("a: &a {effort: low}\nb: &b {temperature: 0.2}\nc: {<<: *a, <<: *b}\n", "<<"),
])
def test_duplicates_name_the_key_file_and_both_locations(tmp_path, text, key):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(SystemExit) as error:
        load_yaml(path)
    message = str(error.value)
    assert f"duplicate key '{key}'" in message
    assert str(path) in message
    assert "first definition" in message
    assert message.count("line ") == 2


def test_merge_sequence_explicit_overrides_and_aliases(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""defaults: &defaults {effort: low, temperature: 0.2}
model: &model
  <<: *defaults
  effort: high
copy: *model
merged: {<<: [*model, *defaults]}
""")
    result = load_yaml(path)
    assert result["model"] == {"effort": "high", "temperature": 0.2}
    assert result["copy"] is result["model"]
    assert result["merged"] == result["model"]


def test_recursive_alias_does_not_loop_during_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("loop: &loop {self: *loop}\n")
    result = load_yaml(path)
    assert result["loop"]["self"] is result["loop"]


@pytest.mark.parametrize("text", ["[]", "false", "42", "!!python/object:os.system {}"])
def test_invalid_document_or_unsafe_tag_is_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(SystemExit, match=str(path)):
        load_yaml(path)


def test_empty_config_remains_an_empty_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("# empty\n")
    assert load_yaml(path) == {}


@pytest.mark.parametrize("path", sorted([
    *ROOT.glob("configs/*.yaml"), *ROOT.glob("tasks/*/task.yaml"),
]), ids=lambda path: str(path.relative_to(ROOT)))
def test_shipped_configuration_has_no_duplicate_keys(path):
    load_yaml(path)


def test_deduplicated_models_keep_the_previously_effective_settings():
    models = load_yaml(ROOT / "configs/models.yaml")
    assert models["deepseek-v4-pro"] == {
        "model": "deepseek-v4-pro", "api_key_env": "DEEPSEEK_API_KEY",
        "api_base_url_env": "DEEPSEEK_BASE_URL",
        "generation_config": {"reasoning_effort": "high"},
        "harness": {"name": "deepseek-harness", "version": "0.1.5rc1"},
    }
    assert models["glm-5.3"] == {
        "model": "openrouter/z-ai/glm-5.3", "api_key_env": "LITELLM_API_KEY",
        "api_base_url_env": "LITELLM_BASE_URL", "generation_config": {"reasoning_effort": "high"},
        "extra_body": {"provider": {"only": ["z-ai"], "allow_fallbacks": False}},
        "harness": {"name": "claude-sdk", "version": "2.1.270"},
    }
