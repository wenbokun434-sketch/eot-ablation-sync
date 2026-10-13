"""Configuration loading with portable, validated paths."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"Environment variable {name!r} is required by the configuration."
                )
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment configuration and stable configuration hash."""

    values: dict[str, Any]
    source: Path
    project_root: Path
    config_hash: str

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing configuration section: {name}")
        return value

    def path(self, name: str) -> Path:
        paths = self.section("paths")
        if name not in paths:
            raise KeyError(f"Missing paths.{name}")
        value = Path(str(paths[name])).expanduser()
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML, expand required environment variables and resolve project root."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The top-level YAML value must be a mapping.")
    values = _expand_environment(raw)
    paths = values.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration must contain a paths mapping.")
    root_value = Path(str(paths.get("project_root", "..")))
    project_root = (
        root_value.resolve()
        if root_value.is_absolute()
        else (source.parent / root_value).resolve()
    )
    stable = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    config_hash = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return ExperimentConfig(values, source, project_root, config_hash)
