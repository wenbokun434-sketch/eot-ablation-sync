"""Protocol-v1 dual thresholds. Selection reads H_j or C, never test labels.

Call load_selection_scope for verified membership. Prediction sequences contain
only valid prefixes (no padding), already averaged across seeds by the caller.
Training/checkpoint and ensemble validation belong to O07/O08; final-stage
lock gating belongs to O13/O14. This module does not promote or replace models.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json

import numpy as np

from eot.metrics import (DEFAULT_THRESHOLD_GRID, RiskSequence, first_alert_position,
                         select_threshold)
from .config import (OptimizationConfig, canonical_json, file_sha256,
                     load_optimization_config, project_path, require_sha256)
from .runs import source_fingerprint
from .splits import checked_ids, digest


@dataclass(frozen=True)
class SelectionScope:
    """Created from verified split files, contains membership but no test labels."""
    _json: str

    @property
    def metadata(self) -> dict:
        return json.loads(self._json)


def load_selection_scope(config: OptimizationConfig, *, role: str, outer_fold: int,
                         inner_fold: int | None = None) -> SelectionScope:
    if role not in {"inner_holdout", "calibration"}:
        raise ValueError("Threshold selection allows only inner_holdout or calibration")
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("Invalid outer fold")
    protection = json.loads(project_path(config.project_root,
        "outputs/optimization/v1/protocol/baseline_protection.json").read_text(encoding="utf-8"))
    outer_relative = f"data/splits/fold_{outer_fold}.json"
    outer_sha = next(r["sha256"] for r in protection["records"]
                     if r["anchor"] == "project" and r["path"] == outer_relative)
    outer_path = project_path(config.project_root, outer_relative)
    if file_sha256(outer_path) != outer_sha:
        raise ValueError("Frozen outer membership changed")
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    if role == "inner_holdout":
        if type(inner_fold) is not int or inner_fold not in range(3):
            raise ValueError("Inner holdout requires inner fold 0..2")
        relative = f"data/optimization/v1/splits/outer_{outer_fold}.json"
        manifest = json.loads(project_path(config.project_root, "data/optimization/v1/manifest.json").read_text(encoding="utf-8"))
        for key, expected in (("protocol_id", config.values["protocol_id"]),
                              ("protocol_sha256", config.protocol_sha256),
                              ("protocol_lock_sha256", config.protocol_lock_sha256),
                              ("base_config_hash", config.base_config_hash)):
            if manifest.get(key) != expected:
                raise ValueError("Stale optimization split manifest")
        expected_sha = manifest["sha256"][relative]
    else:
        if inner_fold is not None:
            raise ValueError("Calibration requires inner_fold=None")
        relative = outer_relative
        expected_sha = outer_sha
    path = project_path(config.project_root, relative)
    if file_sha256(path) != require_sha256(expected_sha, "split hash"):
        raise ValueError("Selection split checksum mismatch")
    split = json.loads(path.read_text(encoding="utf-8"))
    ids = (split["inner"][inner_fold]["roles"]["holdout"] if role == "inner_holdout"
           else split["validation_user_ids"])
    if role == "inner_holdout" and not set(ids).issubset(outer["train_user_ids"]):
        raise ValueError("Inner selection users must belong to original S")
    if set(ids) & set(outer["test_user_ids"]):
        raise ValueError("Outer test users cannot enter threshold selection")
    return SelectionScope(canonical_json({"role": role, "outer_fold": outer_fold,
        "inner_fold": inner_fold, "user_ids": checked_ids(ids, "selection"),
        "split_sha256": expected_sha, "protocol_sha256": config.protocol_sha256}))


@dataclass(frozen=True)
class SelectionPredictions:
    candidate_id: str
    config_sha256: str
    # One entry for a single seed; three for an ensemble. Hash actual checkpoints.
    seed_model_sha256: tuple[tuple[int, str], ...]
    sequences: tuple[RiskSequence, ...]


def _validated_predictions(predictions, scope, expected_config):
    if (predictions.candidate_id != expected_config.candidate_id
            or predictions.config_sha256 != expected_config.config_hash):
        raise ValueError("Prediction candidate/config binding mismatch")
    models = sorted(predictions.seed_model_sha256)
    seeds = [seed for seed, _ in models]
    if (any(type(seed) is not int for seed in seeds)
            or seeds not in ([42], [43], [44], [42, 43, 44])):
        raise ValueError("Expected one protocol seed or all three ensemble seeds")
    for _, sha in models:
        require_sha256(sha, "checkpoint hash")
    ids = checked_ids([s.user_id for s in predictions.sequences], "predictions")
    if ids != scope["user_ids"]:
        raise ValueError("Predictions must cover exactly the selection users")
    # Copy and revalidate: RiskSequence arrays can be mutated after construction.
    sequences = [RiskSequence(s.user_id, s.label, s.probabilities.copy())
                 for s in sorted(predictions.sequences, key=lambda s: s.user_id)]
    if {s.label for s in sequences} != {0, 1}:
        raise ValueError("Selection requires both classes")
    rows = [[s.user_id, int(s.label), s.probabilities.tolist()] for s in sequences]
    source = {"candidate_id": predictions.candidate_id, "config_sha256": predictions.config_sha256,
              "seed_model_sha256": [[seed, sha] for seed, sha in models],
              "model_or_ensemble_sha256": digest(models), "predictions_sha256": digest(rows),
              "aggregation": "single_seed" if len(models) == 1 else "mean_probabilities_before_selection"}
    return sequences, source


def _counts(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "correct": tp + tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0}


def _final_search(labels, probabilities, correct_floor, grid):
    rows = [{"threshold": float(t), **_counts(labels, probabilities, t)} for t in grid]
    for row in rows:
        row["feasible"] = row["correct"] >= correct_floor
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        return {"threshold_final": None, "status": "infeasible", "counts": None, "search": rows}
    # Exact counts and rational precision: no rounding or tolerance on this rule.
    best = max(feasible, key=lambda r: (r["tp"], r["correct"],
               Fraction(r["tp"], r["tp"] + r["fp"]) if r["tp"] + r["fp"] else Fraction(0),
               r["threshold"]))
    return {"threshold_final": best["threshold"], "status": "selected",
            "counts": {k: best[k] for k in ("tp", "tn", "fp", "fn", "correct", "precision")},
            "search": rows}


@dataclass(frozen=True)
class ThresholdSelection:
    _json: str

    @property
    def manifest(self) -> dict:
        return json.loads(self._json)

    def save(self, config: OptimizationConfig, relative_path: str):
        """Persist both thresholds and provenance; never overwrite different bytes."""
        record = self.manifest
        if record["candidate_source"]["config_sha256"] != config.config_hash:
            raise ValueError("Cannot save under a different candidate config")
        path = project_path(config.project_root, relative_path, allowed_prefix="outputs/optimization/v1")
        if path.suffix != ".json":
            raise ValueError("Threshold manifest requires a JSON path")
        content = (canonical_json({"sha256": digest(record), "selection": record}) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("Refusing to overwrite a different threshold selection")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
        return path


def select_dual_thresholds(config: OptimizationConfig, scope: SelectionScope,
                           reference_a: SelectionPredictions,
                           candidate: SelectionPredictions) -> ThresholdSelection:
    """Fit on exactly one H_j or C; no test-label or test-metric parameter.

    Recheck saved scope membership on every call. Returned R1 and R2 use A's
    probabilities; an infeasible candidate retains its own early threshold.
    """
    supplied = scope.metadata
    verified = load_selection_scope(config, role=supplied["role"], outer_fold=supplied["outer_fold"],
                                    inner_fold=supplied["inner_fold"])
    if supplied != verified.metadata:
        raise ValueError("Stale or forged selection scope")
    a_config = load_optimization_config(config.project_root / "configs/optimization/A.yaml")
    if (config.protocol_sha256 != a_config.protocol_sha256
            or config.protocol_lock_sha256 != a_config.protocol_lock_sha256
            or config.base_config_hash != a_config.base_config_hash):
        raise ValueError("Candidate and reference protocols differ")
    a, a_source = _validated_predictions(reference_a, supplied, a_config)
    q, q_source = _validated_predictions(candidate, supplied, config)
    if ([s.label for s in a] != [s.label for s in q]
            or [len(s.probabilities) for s in a] != [len(s.probabilities) for s in q]
            or [s for s, _ in a_source["seed_model_sha256"]] != [s for s, _ in q_source["seed_model_sha256"]]):
        raise ValueError("A and candidate labels, valid prefixes and seeds must align")
    if config.candidate_id == "A" and a_source != q_source:
        raise ValueError("R1 and R2 require identical A predictions and models")
    grid = DEFAULT_THRESHOLD_GRID.copy()
    settings = config.thresholds
    if (settings.consecutive != 2 or settings.grid_min != 0 or settings.grid_max != 1
            or settings.grid_step != .005 or settings.grid_count != 201
            or not np.array_equal(grid, np.linspace(0., 1., 201))):
        raise ValueError("Threshold settings differ from the locked protocol")
    a_early = select_threshold(a, consecutive=settings.consecutive, candidates=grid)
    q_early = a_early if config.candidate_id == "A" else select_threshold(q, consecutive=settings.consecutive, candidates=grid)
    labels = np.array([s.label for s in a])
    a_final = np.array([s.probabilities[-1] for s in a])
    q_final = np.array([s.probabilities[-1] for s in q])
    r1 = {"threshold_final": a_early["threshold"], "threshold_early": a_early["threshold"],
          "counts": _counts(labels, a_final, a_early["threshold"])}
    floor = r1["counts"]["correct"]
    r2 = _final_search(labels, a_final, floor, grid)
    if r2["threshold_final"] is None:
        raise AssertionError("A must be feasible at its early threshold")
    r2["threshold_early"] = a_early["threshold"]
    if np.array_equal(a_final >= r2["threshold_final"], a_final >= r1["threshold_final"]):
        r2["status"] = "unchanged"
    selected = dict(r2) if config.candidate_id == "A" else _final_search(labels, q_final, floor, grid)
    selected["threshold_early"] = q_early["threshold"]
    protocol = config.protocol["thresholds"]
    record = {"schema_version": 1, "protocol_sha256": config.protocol_sha256,
        "protocol_lock_sha256": config.protocol_lock_sha256,
        "scope": supplied, "selection_user_ids_sha256": digest(supplied["user_ids"]),
        "selection_labels_sha256": digest([[s.user_id, int(s.label)] for s in a]),
        "reference_source": a_source, "candidate_source": q_source,
        "code_sha256": source_fingerprint(config.project_root,
            ["src/eot/metrics.py", "src/eot/optimization/decisions.py", "src/eot/optimization/config.py",
             "src/eot/optimization/splits.py", "src/eot/optimization/runs.py"]),
        "grid": grid.tolist(), "positive_comparison": protocol["positive_comparison"],
        "final_objective": "maximize_tp_subject_to_correct_count_at_least_R1",
        "final_tie_break": protocol["final_rank"], "early_policy": protocol["early"],
        "correct_floor": floor, "R1": r1, "R2": r2, "candidate": selected,
        "reference_early_selection": a_early, "candidate_early_selection": q_early}
    return ThresholdSelection(canonical_json(record))


def apply_dual_thresholds(probabilities, *, threshold_final: float, threshold_early: float) -> dict:
    """Label-free application to valid prefixes; final and early decisions differ."""
    values = RiskSequence("inference", 0, np.asarray(probabilities)).probabilities
    for threshold in (threshold_final, threshold_early):
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Applying decisions requires two finite thresholds in [0, 1]")
    position = first_alert_position(values, threshold_early, consecutive=2)
    return {"final_prediction": int(values[-1] >= threshold_final),
            "alert_prediction": int(position is not None), "alert_position": position}
