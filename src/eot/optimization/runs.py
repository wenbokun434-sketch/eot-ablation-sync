"""Deterministic run provenance and bounded paths; no directories or runs created."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .config import OptimizationConfig, PROTOCOL_ID, canonical_json, file_sha256, project_path, require_sha256


def source_fingerprint(project_root: Path, relative_files: Iterable[str]) -> str:
    """Hash explicit runtime dependencies; unrelated reporting edits need not re-key runs."""
    records = {}
    for name in relative_files:
        path = project_path(project_root, name)
        if not path.is_file():
            raise ValueError(f"Code fingerprint input must be a file: {name}")
        relative = path.relative_to(project_root.resolve()).as_posix()
        if relative in records:
            raise ValueError(f"Duplicate source fingerprint input: {relative}")
        records[relative] = file_sha256(path)
    if not records:
        raise ValueError("Code fingerprint requires explicit non-empty dependencies")
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    project_root: Path
    run_id: str
    _metadata_json: str
    _relative_directory: str

    @property
    def metadata(self) -> dict:
        return json.loads(self._metadata_json)

    @property
    def directory(self) -> Path:
        return project_path(self.project_root, self._relative_directory, allowed_prefix="outputs/optimization/v1")

    def artifact_path(self, relative: str) -> Path:
        # Validate relative on its own so absolute/drive-prefixed strings cannot be
        # hidden by concatenating them after the run directory.
        project_path(self.directory, relative)
        return project_path(self.project_root, f"{self._relative_directory}/{relative}",
                            allowed_prefix=self._relative_directory)


def make_run_identity(config: OptimizationConfig, *, stage: str, outer_fold: int,
                      inner_fold: int | None, seed: int, code_sha256: str, split_sha256: str) -> RunIdentity:
    if stage not in {"smoke", "screening", "stability", "final"}:
        raise ValueError("Unknown or not-yet-enabled stage; conditional runs are enabled at O12")
    if type(outer_fold) is not int or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be an integer in 0..4")
    if type(seed) is not int or seed not in config.protocol["seeds"]:
        raise ValueError("seed must be one of the locked seeds 42, 43, 44")
    if stage in {"smoke", "screening"} and seed != 42:
        raise ValueError("smoke/screening use seed 42")
    if stage == "stability" and seed == 42:
        raise ValueError("Reuse the completed screening run for seed 42; do not duplicate it in stability")
    if stage == "final":
        if inner_fold is not None:
            raise ValueError("Final runs require inner_fold=None")
    elif type(inner_fold) is not int or not 0 <= inner_fold < config.protocol["inner_cv"]["folds"]:
        raise ValueError("Inner runs require an integer inner_fold in 0..2")
    metadata = {"protocol_id": PROTOCOL_ID, "stage": stage, "outer_fold": outer_fold,
                "inner_fold": inner_fold, "candidate_id": config.candidate_id, "seed": seed,
                "config_sha256": config.config_hash, "protocol_sha256": config.protocol_sha256,
                "protocol_lock_sha256": config.protocol_lock_sha256,
                "code_sha256": require_sha256(code_sha256, "code_sha256"),
                "split_sha256": require_sha256(split_sha256, "split_sha256")}
    encoded = canonical_json(metadata)
    run_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    inner_name = "final" if inner_fold is None else f"inner_{inner_fold}"
    relative = (f"{config.path('output_root').relative_to(config.project_root).as_posix()}/"
                f"{stage}/outer_{outer_fold}/{inner_name}/{config.candidate_id}/seed_{seed}/{run_id}")
    project_path(config.project_root, relative, allowed_prefix="outputs/optimization/v1")
    return RunIdentity(config.project_root, run_id, encoded, relative)
