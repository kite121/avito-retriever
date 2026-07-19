from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load YAML, recursively resolve ``extends``, and apply dotted overrides.

    Overrides use YAML values, for example::

        retrieval.knn.top_k=20
        reranker.enabled=true
        fusion.inputs.knn=0.5
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}

    parent_ref = current.pop("extends", None)
    if parent_ref:
        parent = load_config(config_path.parent / parent_ref)
        current = _deep_merge(parent, current)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have KEY=VALUE form: {override!r}")
        key, raw_value = override.split("=", 1)
        _set_dotted(current, key, yaml.safe_load(raw_value))

    return current


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

