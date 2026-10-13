"""Read-only optimization configuration; no training or legacy config mutation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

PROTOCOL_ID = "eot_threshold_loss_v1"
PROTOCOL_DIR = "outputs/optimization/v1/protocol"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def project_path(root: Path, relative: str, *, allowed_prefix: str | None = None) -> Path:
    """Reject traversal, Windows drives/ADS/reserved names and redirected ancestors.

    Resolve against the caller's project anchor, never the shell working directory.
    No directory is created. This guard is also applied when run paths are consumed.
    """
    if not isinstance(relative, str) or not relative:
        raise ValueError("Path must be a non-empty project-relative string")
    parts = relative.replace("\\", "/").split("/")
    windows = PureWindowsPath(relative)
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if windows.drive or windows.root or relative.startswith("/") or any(
        part in {"", ".", ".."} or part.endswith((" ", "."))
        or any(c in part for c in '<>:"|?*') or any(ord(c) < 32 for c in part)
        or part.split(".")[0].upper() in reserved for part in parts
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    if allowed_prefix is not None:
        prefix = allowed_prefix.split("/")
        if [p.casefold() for p in parts[:len(prefix)]] != [p.casefold() for p in prefix]:
            raise ValueError(f"Path must stay under {allowed_prefix}: {relative!r}")
    anchored = root.resolve().joinpath(*parts)
    resolved = anchored.resolve()
    # A junction back into legacy outputs is unsafe even if still inside the project.
    if resolved != anchored or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Redirected or escaping path: {relative!r}")
    return resolved


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"YAML keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _keys(value: Any, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} requires exactly these keys: {sorted(expected)}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    return float(value)


@dataclass(frozen=True)
class LossSettings:
    final_loss: str
    alpha: float
    prefix_weight: float
    gamma: float
    final_weight: float
    prefix_loss: str
    prefix_checkpoints: tuple[int, ...]


@dataclass(frozen=True)
class ThresholdSettings:
    grid_min: float
    grid_max: float
    grid_step: float
    grid_count: int
    consecutive: int


@dataclass(frozen=True)
class OptimizationConfig:
    source: Path
    project_root: Path
    candidate_id: str
    loss: LossSettings
    thresholds: ThresholdSettings
    config_hash: str
    source_sha256: str
    protocol_sha256: str
    protocol_lock_sha256: str
    base_config_hash: str
    _effective_json: str
    _base_json: str
    _protocol_json: str

    @property
    def values(self) -> dict:
        """Return a copy so callers cannot silently mutate a hashed config."""
        return json.loads(self._effective_json)

    @property
    def protocol(self) -> dict:
        return json.loads(self._protocol_json)

    @property
    def base_values(self) -> dict:
        """Legacy model/training settings, copied; never an output-writer config."""
        return json.loads(self._base_json)

    def path(self, name: str) -> Path:
        prefixes = {"output_root": "outputs/optimization/v1", "data_root": "data/optimization/v1"}
        if name not in prefixes:
            raise KeyError(f"Unknown optimization path: {name}")
        return project_path(self.project_root, self.values["paths"][name], allowed_prefix=prefixes[name])

    def section(self, name: str) -> dict:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Unknown optimization section: {name}")
        return value


def _locked_inputs(root: Path, lock_sha: str) -> tuple[dict, dict, str]:
    lock_path = project_path(root, f"{PROTOCOL_DIR}/protocol_lock.json")
    if file_sha256(lock_path) != lock_sha:
        raise ValueError("O02 protocol lock checksum mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected protocol lock identity")
    hashes = lock.get("sha256", {})
    protocol_relative = f"{PROTOCOL_DIR}/protocol_v1.json"
    protection_relative = f"{PROTOCOL_DIR}/baseline_protection.json"
    required = {protocol_relative, protection_relative, "docs/optimization/PROTOCOL_v1.md",
                "docs/optimization/audit_o02.py"}
    if set(hashes) != required:
        raise ValueError("Incomplete or unexpected O02 lock entries")
    for name, expected in hashes.items():
        if file_sha256(project_path(root, name)) != require_sha256(expected, name):
            raise ValueError(f"Locked O02 file changed: {name}")
    protocol = json.loads(project_path(root, protocol_relative).read_text(encoding="utf-8"))
    if (protocol.get("protocol_id") != PROTOCOL_ID
            or protocol.get("status") != "locked_before_optimization_training"):
        raise ValueError("Optimization requires the locked v1 protocol")
    protection = json.loads(project_path(root, protection_relative).read_text(encoding="utf-8"))
    protected = {r["path"]: r["sha256"] for r in protection["records"] if r["anchor"] == "project"}
    for name in ("configs/eot_main.yaml", "outputs/reports/raw_data_audit.json"):
        if name not in protected or file_sha256(project_path(root, name)) != protected[name]:
            raise ValueError(f"Protected base input changed: {name}")
    # Restore the original path STRING to reproduce the legacy hash, without reading
    # the raw dataset or temporarily changing process-wide environment variables.
    audit = json.loads(project_path(root, "outputs/reports/raw_data_audit.json").read_text(encoding="utf-8"))
    base = yaml.safe_load(project_path(root, "configs/eot_main.yaml").read_text(encoding="utf-8"))
    if base["paths"]["raw_data_root"] != "${ERISK_TASK2_ROOT}":
        raise ValueError("Unexpected legacy raw-data placeholder")
    base["paths"]["raw_data_root"] = audit["dataset"]["root"]
    base_hash = hashlib.sha256(canonical_json(base).encode("utf-8")).hexdigest()
    if base_hash != protocol["base_config_hash"] or base_hash != audit["config_hash"]:
        raise ValueError("Resolved base config hash differs from the frozen experiment")
    return protocol, base, hashes[protocol_relative]


def load_optimization_config(path: str | Path, *, project_root: str | Path | None = None) -> OptimizationConfig:
    source = Path(path).absolute()
    root = Path(project_root).resolve() if project_root is not None else source.parents[2].resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Config must live in this project's configs/optimization directory") from exc
    source = project_path(root, relative, allowed_prefix="configs/optimization")
    raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    _keys(raw, {"schema_version", "protocol_id", "protocol_lock_sha256", "candidate_id", "loss", "thresholds", "paths"}, "config")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or raw["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Only schema 1 and the locked EOT v1 protocol are supported")
    lock_sha = require_sha256(raw["protocol_lock_sha256"], "protocol_lock_sha256")
    protocol, base, protocol_sha = _locked_inputs(root, lock_sha)
    candidate = raw["candidate_id"]
    if not isinstance(candidate, str) or candidate not in protocol["candidates"]:
        raise ValueError("O03 supports only A-D; optional candidates require the O12 implementation and gate")
    supplied = _keys(raw["loss"], {"final_loss", "alpha", "prefix_weight", "gamma"}, "loss")
    loss_values = {"final_loss": supplied["final_loss"],
                   "alpha": _number(supplied["alpha"], "loss.alpha", strict=True),
                   "prefix_weight": _number(supplied["prefix_weight"], "loss.prefix_weight"),
                   "gamma": _number(supplied["gamma"], "loss.gamma")}
    if loss_values != protocol["candidates"][candidate]:
        raise ValueError(f"Loss parameters do not match locked candidate {candidate}")
    loss = LossSettings(**loss_values, final_weight=protocol["loss"]["final_weight"],
                        prefix_loss=protocol["loss"]["prefix_loss"],
                        prefix_checkpoints=tuple(protocol["loss"]["prefix_checkpoints"]))
    threshold = _keys(raw["thresholds"], {"grid", "consecutive"}, "thresholds")
    grid = _keys(threshold["grid"], {"min", "max", "step", "count"}, "thresholds.grid")
    normalized_grid = {"min": _number(grid["min"], "grid.min"),
                       "max": _number(grid["max"], "grid.max"),
                       "step": _number(grid["step"], "grid.step", strict=True), "count": grid["count"]}
    if (type(grid["count"]) is not int or type(threshold["consecutive"]) is not int
            or normalized_grid != protocol["thresholds"]["grid"]
            or threshold["consecutive"] != protocol["thresholds"]["early"]["consecutive"]):
        raise ValueError("Threshold grid or consecutive count differs from the locked protocol")
    thresholds = ThresholdSettings(normalized_grid["min"], normalized_grid["max"], normalized_grid["step"],
                                   grid["count"], threshold["consecutive"])
    paths = _keys(raw["paths"], {"output_root", "data_root"}, "paths")
    normalized_paths = {}
    for name, prefix in (("output_root", "outputs/optimization/v1"), ("data_root", "data/optimization/v1")):
        resolved = project_path(root, paths[name], allowed_prefix=prefix)
        # v1 uses exactly these roots; custom descendants would split one experiment.
        if resolved != root / prefix:
            raise ValueError(f"paths.{name} must be exactly {prefix}")
        normalized_paths[name] = resolved.relative_to(root).as_posix()
    effective = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "candidate_id": candidate,
                 "base_config_hash": protocol["base_config_hash"], "protocol_sha256": protocol_sha,
                 "protocol_lock_sha256": lock_sha, "loss": asdict(loss), "thresholds": asdict(thresholds),
                 "training": base["training"], "paths": normalized_paths}
    encoded = canonical_json(effective)
    return OptimizationConfig(source, root, candidate, loss, thresholds,
                              hashlib.sha256(encoded.encode("utf-8")).hexdigest(), file_sha256(source),
                              protocol_sha, lock_sha, protocol["base_config_hash"], encoded,
                              canonical_json(base), canonical_json(protocol))
