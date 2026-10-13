"""Protocol-v1 reference comparison: R1/R2/OPT records and paired user bootstrap.

Every object in a comparison is a user-level decision frame over exactly the
same ordered user list, produced by the evaluation module. Thresholds are never
re-selected here: R1/R2 come from an O06 threshold record and OPT reuses the
locked candidate record, so a comparison can only describe decisions that were
already fixed elsewhere.

The bootstrap resamples users (never posts), draws one index array per iteration
and applies it to both objects, and skips an iteration whose resample contains a
single class instead of redrawing. Inner-holdout and outer cohorts are separate
objects: pooling refuses frames whose declared roles differ.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from eot.metrics import classification_metrics, early_detection_metrics
from .config import OptimizationConfig, canonical_json
from .decisions import ThresholdSelection
from .evaluation import DECISION_COLUMNS, PredictionBundle, decision_frame, evaluate_frame
from .splits import digest

FLOAT_TOLERANCE = 1e-12
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_SEED = 42
PRIMARY_ENDPOINTS = ("recall", "accuracy")
PAIRWISE_COMPARISONS = (("R2", "OPT"), ("R1", "R2"), ("R1", "OPT"), ("R0", "OPT"))
CLASSIFICATION_KEYS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "macro_f1",
                       "balanced_accuracy", "mcc", "brier_score")
EARLY_KEYS = ("erde_5", "erde_50", "median_latency", "speed", "f_latency",
              "early_precision", "early_recall", "early_f1")
OBJECT_IDS = ("R0", "R1", "R2", "OPT")


def _source_matches(bundle: PredictionBundle, source: Mapping) -> None:
    if bundle.provenance["config_sha256"] != source["config_sha256"]:
        raise ValueError("Threshold record was selected for a different candidate configuration")
    if bundle.predictions_sha256() != source["predictions_sha256"]:
        raise ValueError("Threshold record was selected from different probabilities")
    if bundle.model_or_ensemble_sha256() != source["model_or_ensemble_sha256"]:
        raise ValueError("Threshold record belongs to a different model or seed ensemble")


def _scope_matches(selection: ThresholdSelection, bundle: PredictionBundle) -> dict:
    record = selection.manifest
    scope = record["scope"]
    if scope["user_ids"] != bundle.user_ids:
        raise ValueError("Threshold record was selected on a different user set; it cannot score this cohort")
    if scope["role"] != bundle.provenance["role"] or scope["outer_fold"] != bundle.provenance["outer_fold"]:
        raise ValueError("Threshold record comes from a different data role or outer fold")
    if scope["inner_fold"] != bundle.provenance["inner_fold"]:
        raise ValueError("Threshold record comes from a different selection fold")
    if scope["protocol_sha256"] != bundle.provenance["protocol_sha256"]:
        raise ValueError("Threshold record and bundle use different protocols")
    return scope


@dataclass(frozen=True)
class SelectionEvaluation:
    """R1/R2/OPT decision frames and metrics for one locked threshold record."""
    frames: dict
    objects: dict
    opt_source: str
    _json: str

    @property
    def record(self) -> dict:
        return json.loads(self._json)


def evaluate_selection(config: OptimizationConfig, selection: ThresholdSelection, *,
                       a_bundle: PredictionBundle, candidate_bundle: PredictionBundle | None = None,
                       consecutive: int = 2) -> SelectionEvaluation:
    """Score one locked threshold record; no threshold or seed is re-chosen."""
    record = selection.manifest
    if (record["protocol_sha256"] != config.protocol_sha256
            or record["protocol_lock_sha256"] != config.protocol_lock_sha256):
        raise ValueError("Threshold record belongs to a different protocol lock")
    if config.config_hash != record["candidate_source"]["config_sha256"]:
        raise ValueError("Threshold record was produced under a different candidate configuration")
    _scope_matches(selection, a_bundle)
    _source_matches(a_bundle, record["reference_source"])

    frames = {
        "R1": decision_frame(a_bundle, threshold_final=record["R1"]["threshold_final"],
                             threshold_early=record["R1"]["threshold_early"], consecutive=consecutive),
        "R2": decision_frame(a_bundle, threshold_final=record["R2"]["threshold_final"],
                             threshold_early=record["R2"]["threshold_early"], consecutive=consecutive),
    }
    if config.candidate_id == "A":
        if candidate_bundle is not None and candidate_bundle.provenance != a_bundle.provenance:
            raise ValueError("A has no separate candidate bundle; R2 is the optimized object")
        frames["OPT"] = frames["R2"].copy()
        source = "A_reference_R2"
    else:
        if candidate_bundle is None:
            raise ValueError("A non-A candidate requires its own prediction bundle")
        _source_matches(candidate_bundle, record["candidate_source"])
        if candidate_bundle.provenance["role"] != a_bundle.provenance["role"]:
            raise ValueError("Reference and candidate bundles must cover the same data role")
        if candidate_bundle.user_ids != a_bundle.user_ids:
            raise ValueError("Reference and candidate bundles must cover the same users")
        if candidate_bundle.provenance["seeds"] != a_bundle.provenance["seeds"]:
            raise ValueError("Reference and candidate bundles must use the same seed set")
        selected = record["candidate"]
        if selected["threshold_final"] is None or selected["status"] == "infeasible":
            raise ValueError("An infeasible candidate has no final threshold and cannot be evaluated")
        frames["OPT"] = decision_frame(candidate_bundle, threshold_final=selected["threshold_final"],
                                       threshold_early=selected["threshold_early"], consecutive=consecutive)
        source = f"candidate_{config.candidate_id}"
    objects = {name: evaluate_frame(frame) for name, frame in frames.items()}
    summary = {"schema_version": 1, "protocol_sha256": record["protocol_sha256"],
               "candidate_id": config.candidate_id, "opt_source": source,
               "role": a_bundle.provenance["role"], "outer_fold": a_bundle.provenance["outer_fold"],
               "inner_fold": a_bundle.provenance["inner_fold"], "seeds": a_bundle.provenance["seeds"],
               "selection_user_ids_sha256": digest(a_bundle.user_ids),
               "selection_labels_sha256": record["selection_labels_sha256"],
               "object_metrics": objects,
               "opt_predictions_sha256": (a_bundle if config.candidate_id == "A"
                                          else candidate_bundle).predictions_sha256()}
    return SelectionEvaluation(frames, objects, source, canonical_json(summary))


def _frame_arrays(frame: pd.DataFrame, name: str):
    if set(frame.columns) != set(DECISION_COLUMNS):
        raise ValueError(f"{name} must be a decision frame with columns {sorted(DECISION_COLUMNS)}")
    labels = frame["label"].to_numpy(dtype=np.int8)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    predictions = frame["final_prediction"].to_numpy(dtype=np.int8)
    positions = frame["alert_position"].to_numpy(dtype=np.float64)
    if not np.array_equal(np.isfinite(positions).astype(np.int8),
                          frame["alert_prediction"].to_numpy(np.int8)):
        raise ValueError(f"{name}: alert_prediction and alert_position disagree")
    if not np.array_equal((probabilities >= frame["threshold_final"].to_numpy(np.float64)).astype(np.int8),
                          predictions):
        raise ValueError(f"{name}: final_prediction does not follow its own threshold")
    return labels, probabilities, predictions, positions


def pool_fold_frames(frames: Sequence[pd.DataFrame], *, roles: Sequence[str], name: str = "pooled") -> pd.DataFrame:
    """Concatenate per-fold frames after proving roles match and users are disjoint.

    Mixing inner-holdout with outer-test (or calibration) results is rejected, as
    are duplicate users and folds that overlap.
    """
    if not frames:
        raise ValueError("Pooling requires at least one frame")
    if len(roles) != len(frames) or any(not isinstance(role, str) or not role for role in roles):
        raise ValueError("Pooling requires one non-empty role per frame")
    if len(set(roles)) != 1:
        raise ValueError(f"Refusing to pool frames from different data roles: {sorted(set(roles))}")
    seen: set[str] = set()
    for index, frame in enumerate(frames):
        _frame_arrays(frame, f"{name}[{index}]")
        ids = [str(value) for value in frame["user_id"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}[{index}] contains duplicate users")
        overlap = seen & set(ids)
        if overlap:
            raise ValueError(f"{name}: folds overlap on {sorted(overlap)[:3]}")
        seen |= set(ids)
    pooled = pd.concat([frame[list(DECISION_COLUMNS)] for frame in frames], ignore_index=True)
    return pooled.sort_values("user_id", kind="stable").reset_index(drop=True)


def _iteration_metrics(labels, probabilities, predictions, positions) -> dict:
    classification = classification_metrics(labels, probabilities, predictions=predictions)
    early = early_detection_metrics(labels, positions)
    values = {"accuracy": float(np.mean(predictions == labels)), **classification, **early}
    return {name: (float(value) if value is not None else float("nan"))
            for name, value in values.items()}


def paired_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, *, iterations: int = BOOTSTRAP_ITERATIONS,
                     level: float = BOOTSTRAP_LEVEL, seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile intervals of candidate-minus-reference over identical user draws.

    One index array per iteration is applied to both objects, so every recorded
    difference is computed on the same resampled users. Iterations whose resample
    contains a single class are skipped and counted, never redrawn. The unit is
    the user; posts are never resampled.
    """
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if list(reference["user_id"]) != list(candidate["user_id"]):
        raise ValueError("Paired comparison requires identical users in identical order")
    if not np.array_equal(reference["label"].to_numpy(), candidate["label"].to_numpy()):
        raise ValueError("Paired comparison requires identical labels")
    left = _frame_arrays(reference, "reference")
    right = _frame_arrays(candidate, "candidate")
    labels = left[0]
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Paired comparison requires both classes")

    generator = np.random.default_rng(seed)
    hasher = hashlib.sha256()
    hasher.update(str(iterations).encode("ascii"))
    differences: dict[str, list[float]] = {}
    skipped = 0
    for _ in range(iterations):
        indices = generator.integers(0, labels.size, size=labels.size)
        hasher.update(np.asarray(indices, dtype=np.int64).tobytes())
        if np.unique(labels[indices]).size < 2:
            skipped += 1
            continue
        base = _iteration_metrics(labels[indices], left[1][indices], left[2][indices], left[3][indices])
        other = _iteration_metrics(labels[indices], right[1][indices], right[2][indices], right[3][indices])
        for name, value in other.items():
            differences.setdefault(name, []).append(value - base[name])
    if not differences:
        raise AssertionError("Every bootstrap resample contained a single class")

    base_point = _iteration_metrics(labels, left[1], left[2], left[3])
    other_point = _iteration_metrics(labels, right[1], right[2], right[3])
    alpha = (1.0 - float(level)) / 2.0
    intervals = {}
    for name, values in differences.items():
        array = np.asarray(values, dtype=np.float64)
        valid = array[np.isfinite(array)]
        intervals[name] = {
            "point_difference": other_point[name] - base_point[name],
            "lower": float(np.quantile(valid, alpha)) if valid.size else float("nan"),
            "upper": float(np.quantile(valid, 1.0 - alpha)) if valid.size else float("nan"),
            "mean_difference": float(np.mean(valid)) if valid.size else float("nan"),
            "valid_iterations": int(valid.size)}
    return {"iterations": iterations, "level": float(level), "seed": seed, "unit": "users",
            "paired": True, "post_level_bootstrap": False, "indices_sha256": hasher.hexdigest(),
            "skipped_single_class_iterations": skipped, "valid_iterations": iterations - skipped,
            "differences": intervals, "reference_point": base_point, "candidate_point": other_point}


def joint_support(comparison: dict, endpoints: Sequence[str] = PRIMARY_ENDPOINTS) -> dict:
    """Two-primary-endpoint check: both bootstrap 2.5% quantiles above zero."""
    record = comparison["differences"]
    result = {}
    for name in endpoints:
        if name not in record:
            raise ValueError(f"Missing primary endpoint in the comparison: {name}")
        lower = float(record[name]["lower"])
        result[name] = {"lower": lower,
                        "point_difference": float(record[name]["point_difference"]),
                        "supported": bool(np.isfinite(lower) and lower > 0.0)}
    supported = all(item["supported"] for item in result.values())
    return {"endpoints": list(endpoints), "per_endpoint": result, "both_above_zero": supported,
            "interpretation": ("both Recall and Accuracy differences strictly positive at the 2.5% quantile"
                               if supported else "joint support not established")}


def _metric(result: dict, key: str) -> float:
    if key == "accuracy":
        return float(result["accuracy"])
    if key in result["classification"]:
        return float(result["classification"][key])
    value = result["early_detection"][key]
    return float("nan") if value is None else float(value)


def compare_frames(reference: pd.DataFrame, candidate: pd.DataFrame, *, reference_id: str, candidate_id: str,
                   iterations: int = BOOTSTRAP_ITERATIONS, level: float = BOOTSTRAP_LEVEL,
                   seed: int = BOOTSTRAP_SEED) -> dict:
    """Descriptive paired comparison of two already-decided objects."""
    for name in (reference_id, candidate_id):
        if not isinstance(name, str) or not name:
            raise ValueError("Comparison objects need non-empty identifiers")
    evaluation = {"reference": evaluate_frame(reference), "candidate": evaluate_frame(candidate)}
    bootstrap = paired_bootstrap(reference, candidate, iterations=iterations, level=level, seed=seed)
    point = {"accuracy": evaluation["candidate"]["accuracy"] - evaluation["reference"]["accuracy"]}
    for key in (*CLASSIFICATION_KEYS, *EARLY_KEYS):
        point[key] = _metric(evaluation["candidate"], key) - _metric(evaluation["reference"], key)
    point["counts"] = {key: evaluation["candidate"]["counts"][key] - evaluation["reference"]["counts"][key]
                       for key in ("tp", "tn", "fp", "fn", "correct")}
    return {"schema_version": 1, "reference_id": reference_id, "candidate_id": candidate_id,
            "users": int(len(reference)), "evaluation": evaluation, "point_differences": point,
            "bootstrap": bootstrap, "joint_support": joint_support(bootstrap),
            "uncertainty_caveat": ("conditional on the fitted models, inner selection and locked thresholds; "
                                   "no retraining or reselection uncertainty")}


def compare_objects(frames: Mapping[str, pd.DataFrame], *, pairs=PAIRWISE_COMPARISONS,
                    iterations: int = BOOTSTRAP_ITERATIONS, level: float = BOOTSTRAP_LEVEL,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Run every configured pair that is present; absent objects are recorded, not invented."""
    results, skipped = {}, {}
    for reference_id, candidate_id in pairs:
        if reference_id not in frames or candidate_id not in frames:
            skipped[f"{candidate_id}_vs_{reference_id}"] = "object not available for this stage"
            continue
        if list(frames[reference_id]["user_id"]) != list(frames[candidate_id]["user_id"]):
            raise ValueError(f"{candidate_id} and {reference_id} do not cover the same users in the same order")
        results[f"{candidate_id}_vs_{reference_id}"] = compare_frames(
            frames[reference_id], frames[candidate_id], reference_id=reference_id, candidate_id=candidate_id,
            iterations=iterations, level=level, seed=seed)
    return {"schema_version": 1, "iterations": iterations, "level": float(level), "seed": seed,
            "primary_comparator": "OPT_vs_R2", "comparisons": results, "skipped": skipped}
