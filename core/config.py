"""Read YAML configuration without silently discarding duplicate keys."""
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, SequenceNode


class _UniqueKeyLoader(yaml.SafeLoader):
    def get_single_data(self):
        node = self.get_single_node()
        if node is None:
            return None
        # Inspect the original mappings before SafeLoader expands YAML merges.
        # An explicit override of an inherited key is valid; repeating a key in
        # the same mapping is ambiguous, including repeated << keys (use a merge
        # sequence instead). Track aliases to handle recursive YAML.
        self._check_keys(node, set())
        return self.construct_document(node)

    def _check_keys(self, node, visited):
        if node in visited:
            return
        visited.add(node)
        if isinstance(node, MappingNode):
            keys = {}
            for key_node, value_node in node.value:
                key = ("<<" if key_node.tag == "tag:yaml.org,2002:merge"
                       else self.construct_object(key_node, deep=True))
                try:
                    previous = keys.get(key)
                    if previous is not None:
                        raise ConstructorError(
                            "first definition", previous.start_mark,
                            f"duplicate key {key!r}", key_node.start_mark)
                    keys[key] = key_node
                except TypeError as exc:
                    raise ConstructorError(
                        "while constructing a mapping", node.start_mark,
                        "found unhashable key", key_node.start_mark) from exc
                self._check_keys(value_node, visited)
        elif isinstance(node, SequenceNode):
            for item in node.value:
                self._check_keys(item, visited)


def load_yaml(path: str | Path) -> dict:
    """Load a configuration mapping, naming the file and lines on YAML errors."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise SystemExit(f"cannot load config {path}: {exc}") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: expected a YAML mapping")
    return config


def load_models(path: str | Path) -> dict:
    """Load top-level model entries; archived wrapped configurations remain readable.

    An archived default name is ignored: callers must select a model explicitly.
    """
    config = load_yaml(path)
    if "models" in config:
        unknown = set(config) - {"models", "defaults"}
        if unknown or not isinstance(config["models"], dict):
            raise SystemExit(f"{path}: invalid archived models mapping")
        config = {name: entry for name, entry in config["models"].items()
                  if name != "default"}
    elif "default" in config:
        raise SystemExit(f"{path}: remove default; select a model with --model")
    for name, entry in config.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise SystemExit(f"{path}: model {name!r} must be a mapping")
    return config
