"""Protocol-v1 prediction export and dual-threshold evaluation.

A bundle averages the three (or one) seed probability arrays inside one fold
*before* any threshold exists, so there is deliberately no per-seed or
best-seed evaluation path. Bundles write to a content-addressed directory under
the optimization output root and refuse to be replaced by different content.

Inner-holdout and calibration bundles must match the frozen split membership.
Outer test prediction export stays closed until the O13/O14 post-calibration
lock gate exists, so this module cannot read an outer test cohort at all.

The original ERDE/F-latency formulas, ``alert_positions`` and
``metrics_from_final_predictions`` are reused unchanged; nothing here
reimplements a metric formula.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from eot.evaluation import metrics_from_final_predictions
from eot.metrics import RiskSequence, alert_positions
from .config import (PROTOCOL_DIR, OptimizationConfig, canonical_json, file_sha256,
                     project_path, require_sha256)
from .decisions import load_selection_scope
from .runs import source_fingerprint
from .splits import checked_ids, digest

BUNDLE_SCHEMA = 1
EVALUATION_ROOT = "outputs/optimization/v1"
FINAL_COLUMNS = ("user_id", "label", "length", "probability")
PREFIX_COLUMNS = ("user_id", "label", "length", "position", "probability")
DECISION_COLUMNS = ("user_id", "label", "length", "probability", "threshold_final",
                    "threshold_early", "final_prediction", "alert_prediction", "alert_position")
SINGLE_SEED = (42,)
ENSEMBLE_SEEDS = (42, 43, 44)
# The frozen experiment compared final probabilities with the last prefix at 1e-7.
FINAL_PREFIX_TOLERANCE = 1e-7
METRIC_TOLERANCE = 1e-12

# Every source whose behaviour can change a stored probability or decision.
BUNDLE_SOURCES = (
    "src/eot/dataset.py", "src/eot/evaluation.py", "src/eot/metrics.py", "src/eot/model.py",
    "src/eot/optimization/config.py", "src/eot/optimization/runs.py",
    "src/eot/optimization/splits.py", "src/eot/optimization/data.py",
    "src/eot/optimization/training.py", "src/eot/optimization/decisions.py",
    "src/eot/optimization/evaluation.py",
)

LEGACY_OOF_FILES = (
    "outputs/predictions/oof_final_predictions.parquet",
    "outputs/predictions/oof_prefix_predictions.parquet",
    "outputs/metrics/oof_metrics.json",
    "outputs/reports/evaluation_manifest.json",
)
# Original frozen report, restated here so the replay can be cross-checked
# against a value that does not come from the artifact being verified.
LEGACY_OOF_COUNTS = {"tp": 56, "tn": 798, "fp": 9, "fn": 46}


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Content digest of a frame; independent of parquet bytes and row order."""
    rows = [[_scalar(value) for value in row]
            for row in frame[list(columns)].itertuples(index=False, name=None)]
    return digest(rows)


def confusion_counts(labels, predictions) -> dict:
    """Exact TP/TN/FP/FN, computed without any metric library."""
    actual = np.asarray(labels, dtype=np.int64) == 1
    predicted = np.asarray(predictions, dtype=np.int64) == 1
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("labels and predictions must be non-empty aligned vectors")
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "users": int(actual.size),
            "correct": tp + tn, "accuracy": (tp + tn) / int(actual.size)}


def _require_close(observed: float, reported: float, name: str) -> None:
    if not np.isfinite(observed) or not np.isfinite(reported) or abs(observed - reported) > METRIC_TOLERANCE:
        raise AssertionError(f"{name} is inconsistent with the confusion matrix")


def cohort_user_ids(config: OptimizationConfig, role: str, *, outer_fold: int,
                    inner_fold: int | None = None) -> list[str]:
    """Membership only, re-derived from the frozen splits; never any label."""
    if role == "test":
        raise ValueError("Outer test prediction export is gated to O14: it requires all five "
                         "post-calibration locks, which O08 does not create")
    scope = load_selection_scope(config, role=role, outer_fold=outer_fold, inner_fold=inner_fold)
    return scope.metadata["user_ids"]


def _validated_frame(frame: pd.DataFrame, columns: Sequence[str], name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if set(frame.columns) != set(columns):
        raise ValueError(f"{name} must have exactly the columns {sorted(columns)}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    selected = frame[list(columns)].copy()
    ids = [str(value) for value in selected["user_id"]]
    if any(not value for value in ids):
        raise ValueError(f"{name}: empty user ID")
    if list(columns) == list(FINAL_COLUMNS) and len(ids) != len(set(ids)):
        # Prefix frames legitimately repeat one user per position; final frames must not.
        raise ValueError(f"{name}: duplicate users")
    if not np.all(np.isin(selected["label"].to_numpy(), [0, 1])):
        raise ValueError(f"{name}: labels must be binary 0/1")
    lengths = selected["length"].to_numpy(dtype=np.int64)
    if np.any(lengths < 1):
        raise ValueError(f"{name}: sequence lengths must be positive")
    values = selected["probability"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name}: probabilities must be finite and inside [0, 1]")
    selected["user_id"] = ids
    selected["label"] = selected["label"].to_numpy(dtype=np.int8)
    selected["length"] = lengths.astype(np.int16)
    selected["probability"] = values
    if list(columns) == list(PREFIX_COLUMNS):
        selected["position"] = selected["position"].to_numpy(dtype=np.int16)
        order = np.lexsort((selected["position"].to_numpy(), selected["user_id"].to_numpy()))
        selected = selected.iloc[order].reset_index(drop=True)
        _check_contiguous(selected, name)
    else:
        selected = selected.sort_values("user_id", kind="stable").reset_index(drop=True)
    return selected


def _check_contiguous(prefix: pd.DataFrame, name: str) -> None:
    """Prefix rows must be exactly positions 1..length for every user."""
    for user_id, group in prefix.groupby("user_id", sort=False):
        positions = group["position"].to_numpy(dtype=np.int64)
        expected = np.arange(1, len(group) + 1, dtype=np.int64)
        if not np.array_equal(positions, expected):
            raise ValueError(f"{name}: non-contiguous prefix positions for {user_id}")
        if group["length"].nunique() != 1 or int(group["length"].iloc[0]) != len(group):
            raise ValueError(f"{name}: prefix row count differs from the sequence length of {user_id}")


def _check_pair(final: pd.DataFrame, prefix: pd.DataFrame, name: str) -> None:
    if list(final["user_id"]) != sorted(set(prefix["user_id"])):
        raise ValueError(f"{name}: final and prefix users differ")
    lengths = final.set_index("user_id")["length"]
    labels = final.set_index("user_id")["label"]
    grouped = prefix.groupby("user_id", sort=False)
    for user_id in final["user_id"]:
        group = grouped.get_group(user_id)
        if int(group["length"].iloc[0]) != int(lengths[user_id]):
            raise ValueError(f"{name}: prefix length differs for {user_id}")
        if int(group["label"].iloc[0]) != int(labels[user_id]):
            raise ValueError(f"{name}: prefix label differs for {user_id}")
        last = float(group["probability"].iloc[-1])
        if abs(last - float(final.set_index("user_id")["probability"][user_id])) > FINAL_PREFIX_TOLERANCE:
            raise ValueError(f"{name}: final probability is not the last valid prefix of {user_id}")


@dataclass(frozen=True)
class SeedRun:
    """Checkpoint identity of one seed inside the bundle; hashes only, no paths."""
    seed: int
    run_id: str
    run_sha256: str
    checkpoint_sha256: str

    def as_record(self) -> dict:
        return {"seed": self.seed, "run_id": require_sha256(self.run_id, "run_id"),
                "run_sha256": require_sha256(self.run_sha256, "run_sha256"),
                "checkpoint_sha256": require_sha256(self.checkpoint_sha256, "checkpoint_sha256")}


@dataclass(frozen=True)
class PredictionBundle:
    """Ensembled user-level predictions with verified provenance."""
    _provenance_json: str
    final: pd.DataFrame
    prefix: pd.DataFrame

    @property
    def provenance(self) -> dict:
        return json.loads(self._provenance_json)

    @property
    def user_ids(self) -> list[str]:
        return [str(value) for value in self.final["user_id"]]

    @property
    def labels(self) -> list[int]:
        return [int(value) for value in self.final["label"]]

    def sequences(self) -> list[RiskSequence]:
        """Valid prefixes per user, ordered by user ID (decisions' expected order)."""
        grouped = {str(key): frame for key, frame in self.prefix.groupby("user_id", sort=False)}
        sequences = []
        for row in self.final.itertuples(index=False):
            user_id = str(row.user_id)
            frame = grouped[user_id]
            sequences.append(RiskSequence(user_id, int(row.label),
                                          frame["probability"].to_numpy(dtype=np.float64)))
        return sequences

    def model_or_ensemble_sha256(self) -> str:
        """Same identity string the threshold selection records for this model set."""
        runs = sorted(self.provenance["seed_runs"], key=lambda item: item["seed"])
        return digest([[run["seed"], run["checkpoint_sha256"]] for run in runs])

    def predictions_sha256(self) -> str:
        """Must equal the threshold record's candidate/reference predictions hash."""
        rows = [[sequence.user_id, int(sequence.label), sequence.probabilities.tolist()]
                for sequence in self.sequences()]
        return digest(rows)

    def relative_directory(self) -> str:
        record = self.provenance
        inner = "final" if record["inner_fold"] is None else f"inner_{record['inner_fold']}"
        seeds = "-".join(str(value) for value in record["seeds"])
        return (f"{EVALUATION_ROOT}/predictions/{record['stage']}/outer_{record['outer_fold']}/{inner}/"
                f"{record['candidate_id']}/seeds_{seeds}/{digest(record)[:16]}")


def _seed_list(stage: str, seeds) -> tuple[int, ...]:
    if stage not in {"screening", "stability", "final"}:
        raise ValueError("Bundle stage must be screening, stability or final")
    values = tuple(seeds)
    expected = SINGLE_SEED if stage == "screening" else ENSEMBLE_SEEDS
    if values != expected or any(type(value) is not int for value in values):
        raise ValueError(f"{stage} bundles require exactly the seeds {list(expected)}")
    return values


def _role_fold(stage: str, role: str, inner_fold):
    if stage == "final":
        if role != "calibration" or inner_fold is not None:
            raise ValueError("Final-stage bundles evaluate the calibration role with inner_fold=None")
        return "calibration"
    if role != "inner_holdout" or type(inner_fold) is not int or inner_fold not in range(3):
        raise ValueError("Screening/stability bundles evaluate one inner holdout fold 0..2")
    return f"inner_{inner_fold}"


def build_bundle(config: OptimizationConfig, *, stage: str, outer_fold: int, inner_fold: int | None,
                 role: str, seeds: Sequence[int], runs: Sequence[SeedRun],
                 final_frames: Sequence[pd.DataFrame],
                 prefix_frames: Sequence[pd.DataFrame]) -> PredictionBundle:
    """Average the seed probabilities of one fold, then verify memberships/hashes.

    ``runs`` and the two frame lists are aligned with ``seeds``. The cohort is
    re-derived from the frozen splits, so a bundle can never be built for users
    outside its declared role, and no test cohort is reachable here.
    """
    seeds = _seed_list(stage, seeds)
    _role_fold(stage, role, inner_fold)
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in 0..4")
    if len(final_frames) != len(seeds) or len(prefix_frames) != len(seeds):
        raise ValueError("Exactly one final and one prefix frame are required per seed")
    records = [item if isinstance(item, SeedRun) else SeedRun(*item) for item in runs]
    if len(records) != len(seeds):
        raise ValueError("Exactly one run record is required per seed")
    by_seed = {record.seed: record.as_record() for record in records}
    if sorted(by_seed) != sorted(seeds) or len(by_seed) != len(records):
        raise ValueError("Run records must match the required seeds exactly")
    if stage == "stability" and any(by_seed[seed]["run_id"] == by_seed[seeds[0]]["run_id"] for seed in seeds[1:]):
        raise ValueError("Stability seeds must come from distinct runs")

    finals, prefixes = [], []
    for seed, final_frame, prefix_frame in zip(seeds, final_frames, prefix_frames, strict=True):
        final = _validated_frame(final_frame, FINAL_COLUMNS, f"seed {seed} final")
        prefix = _validated_frame(prefix_frame, PREFIX_COLUMNS, f"seed {seed} prefix")
        _check_pair(final, prefix, f"seed {seed}")
        finals.append(final)
        prefixes.append(prefix)
    reference = finals[0]
    for seed, final, prefix in zip(seeds[1:], finals[1:], prefixes[1:], strict=True):
        if (list(final["user_id"]) != list(reference["user_id"])
                or not np.array_equal(final["label"].to_numpy(), reference["label"].to_numpy())
                or not np.array_equal(final["length"].to_numpy(), reference["length"].to_numpy())):
            raise ValueError(f"seed {seed} users, labels or lengths are not aligned with seed {seeds[0]}")
        if (list(prefix["user_id"]) != list(prefixes[0]["user_id"])
                or not np.array_equal(prefix["position"].to_numpy(), prefixes[0]["position"].to_numpy())):
            raise ValueError(f"seed {seed} prefix positions are not aligned with seed {seeds[0]}")

    expected = checked_ids(cohort_user_ids(config, role, outer_fold=outer_fold, inner_fold=inner_fold),
                           f"{role} cohort")
    if list(reference["user_id"]) != expected:
        raise ValueError(f"Bundle users must equal the frozen {role} membership of outer fold {outer_fold}")
    labels = reference["label"].to_numpy()
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Evaluation requires both classes")

    ensemble_final = reference.copy()
    ensemble_final["probability"] = np.mean(
        np.vstack([frame["probability"].to_numpy(dtype=np.float64) for frame in finals]), axis=0)
    ensemble_prefix = prefixes[0].copy()
    ensemble_prefix["probability"] = np.mean(
        np.vstack([frame["probability"].to_numpy(dtype=np.float64) for frame in prefixes]), axis=0)

    record = {
        "schema_version": BUNDLE_SCHEMA, "protocol_id": config.values["protocol_id"],
        "protocol_sha256": config.protocol_sha256, "protocol_lock_sha256": config.protocol_lock_sha256,
        "base_config_hash": config.base_config_hash, "candidate_id": config.candidate_id,
        "config_sha256": config.config_hash, "stage": stage, "outer_fold": outer_fold,
        "inner_fold": inner_fold, "role": role, "seeds": list(seeds),
        "aggregation": "arithmetic_mean_probabilities_across_seeds_within_fold",
        "selection_rule": "ensemble_before_any_threshold; single_seed_only_when_one_seed",
        "seed_runs": [by_seed[seed] for seed in seeds],
        "cohort_user_ids_sha256": digest(expected), "users": len(expected),
        "class_counts": {"negative": int(np.sum(labels == 0)), "positive": int(np.sum(labels == 1))},
        "labels_sha256": digest([[user_id, int(label)] for user_id, label in zip(expected, labels, strict=True)]),
        "predictions_sha256": "",  # filled below from the ensembled sequences
        "final_sha256": frame_digest(ensemble_final, FINAL_COLUMNS),
        "prefix_sha256": frame_digest(ensemble_prefix, PREFIX_COLUMNS),
        "prefix_rows": int(len(ensemble_prefix)),
        "code_sha256": source_fingerprint(config.project_root, BUNDLE_SOURCES),
    }
    provisional = PredictionBundle(canonical_json(record), ensemble_final, ensemble_prefix)
    record["predictions_sha256"] = provisional.predictions_sha256()
    return PredictionBundle(canonical_json(record), ensemble_final, ensemble_prefix)


def _files(bundle: PredictionBundle) -> dict[str, bytes]:
    output = io.BytesIO()
    bundle.final.to_parquet(output, index=False, compression="zstd")
    final_bytes = output.getvalue()
    output = io.BytesIO()
    bundle.prefix.to_parquet(output, index=False, compression="zstd")
    return {"final.parquet": final_bytes, "prefix.parquet": output.getvalue(),
            "bundle.json": (canonical_json({"sha256": digest(bundle.provenance),
                                            "bundle": bundle.provenance}) + "\n").encode("utf-8")}


def _existing_matches(path: Path, name: str, bundle: PredictionBundle) -> bool:
    """Content check, so an identical re-save is a no-op even if bytes differ."""
    try:
        if name == "bundle.json":
            return path.read_bytes() == _files(bundle)[name]
        columns = FINAL_COLUMNS if name == "final.parquet" else PREFIX_COLUMNS
        expected = bundle.provenance["final_sha256" if name == "final.parquet" else "prefix_sha256"]
        return frame_digest(pd.read_parquet(path), columns) == expected
    except Exception:
        return False


def save_bundle(config: OptimizationConfig, bundle: PredictionBundle) -> Path:
    """Write to the content-addressed bundle directory; never overwrite other content."""
    record = bundle.provenance
    if (record["config_sha256"] != config.config_hash or record["candidate_id"] != config.candidate_id
            or record["base_config_hash"] != config.base_config_hash):
        raise ValueError("Bundle belongs to a different optimization config")
    directory = project_path(config.project_root, bundle.relative_directory(),
                            allowed_prefix=EVALUATION_ROOT)
    for name, content in _files(bundle).items():
        path = directory / name
        if path.exists():
            if not _existing_matches(path, name, bundle):
                raise ValueError(f"Refusing to overwrite differing bundle file: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
    return directory


def load_bundle(config: OptimizationConfig, relative_directory: str) -> PredictionBundle:
    """Reload and re-verify checksum, binding, memberships and content digests."""
    directory = project_path(config.project_root, relative_directory, allowed_prefix=EVALUATION_ROOT)
    wrapper = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    record = wrapper["bundle"]
    if wrapper.get("sha256") != digest(record) or record.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Prediction bundle checksum mismatch")
    if (record["protocol_id"] != config.values["protocol_id"]
            or record["protocol_sha256"] != config.protocol_sha256
            or record["protocol_lock_sha256"] != config.protocol_lock_sha256
            or record["base_config_hash"] != config.base_config_hash):
        raise ValueError("Bundle was produced under a different locked protocol")
    if record["config_sha256"] != config.config_hash or record["candidate_id"] != config.candidate_id:
        raise ValueError("Bundle belongs to a different candidate configuration")
    seeds = _seed_list(record["stage"], record["seeds"])
    _role_fold(record["stage"], record["role"], record["inner_fold"])
    final = _validated_frame(pd.read_parquet(directory / "final.parquet"), FINAL_COLUMNS, "final.parquet")
    prefix = _validated_frame(pd.read_parquet(directory / "prefix.parquet"), PREFIX_COLUMNS, "prefix.parquet")
    _check_pair(final, prefix, "stored bundle")
    if (frame_digest(final, FINAL_COLUMNS) != record["final_sha256"]
            or frame_digest(prefix, PREFIX_COLUMNS) != record["prefix_sha256"]
            or int(len(prefix)) != int(record["prefix_rows"])):
        raise ValueError("Stored prediction content differs from the bundle manifest")
    expected = checked_ids(cohort_user_ids(config, record["role"], outer_fold=record["outer_fold"],
                                           inner_fold=record["inner_fold"]), "cohort")
    if list(final["user_id"]) != expected or digest(expected) != record["cohort_user_ids_sha256"]:
        raise ValueError("Stored bundle users do not match the frozen cohort")
    labels = final["label"].to_numpy()
    if digest([[user_id, int(label)] for user_id, label in zip(expected, labels, strict=True)]) != record["labels_sha256"]:
        raise ValueError("Stored bundle labels do not match the bundle manifest")
    bundle = PredictionBundle(canonical_json(record), final, prefix)
    if bundle.predictions_sha256() != record["predictions_sha256"]:
        raise ValueError("Stored probabilities do not reproduce the recorded prediction hash")
    if bundle.model_or_ensemble_sha256() != digest(sorted(
            [[run["seed"], run["checkpoint_sha256"]] for run in record["seed_runs"]])):
        raise ValueError("Stored seed run records are inconsistent")
    return bundle


def decision_frame(bundle: PredictionBundle, *, threshold_final: float,
                   threshold_early: float, consecutive: int = 2) -> pd.DataFrame:
    """Apply the two thresholds separately. Final uses the last valid prefix."""
    for name, threshold in (("threshold_final", threshold_final), ("threshold_early", threshold_early)):
        if type(threshold) is bool or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
            raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if type(consecutive) is not int or consecutive < 1:
        raise ValueError("consecutive must be a positive integer")
    sequences = bundle.sequences()
    positions = alert_positions(sequences, float(threshold_early), consecutive)
    frame = bundle.final.copy()
    frame["threshold_final"] = float(threshold_final)
    frame["threshold_early"] = float(threshold_early)
    frame["final_prediction"] = (frame["probability"] >= float(threshold_final)).astype(np.int8)
    frame["alert_prediction"] = np.isfinite(positions).astype(np.int8)
    frame["alert_position"] = positions
    return frame[list(DECISION_COLUMNS)]


def evaluate_frame(frame: pd.DataFrame) -> dict:
    """Confusion matrix, Accuracy and the original classification/early metrics."""
    if set(frame.columns) != set(DECISION_COLUMNS):
        raise ValueError(f"Decision frames must have exactly the columns {sorted(DECISION_COLUMNS)}")
    if frame.empty:
        raise ValueError("Decision frame must not be empty")
    positions = frame["alert_position"].to_numpy(dtype=np.float64)
    if not np.array_equal(np.isfinite(positions).astype(np.int8),
                          frame["alert_prediction"].to_numpy(dtype=np.int8)):
        raise ValueError("alert_prediction and alert_position disagree")
    expected = (frame["probability"].to_numpy(dtype=np.float64)
                >= frame["threshold_final"].to_numpy(dtype=np.float64)).astype(np.int8)
    if not np.array_equal(expected, frame["final_prediction"].to_numpy(dtype=np.int8)):
        raise ValueError("final_prediction does not follow probability >= threshold_final")
    if set(np.unique(frame["label"]).tolist()) != {0, 1}:
        raise ValueError("Evaluation requires both classes")
    counts = confusion_counts(frame["label"].to_numpy(), frame["final_prediction"].to_numpy())
    metrics = metrics_from_final_predictions(frame)
    classification = metrics["classification"]
    # The metric library must agree with the exact counts computed above.
    denominators = {"precision": (counts["tp"], counts["tp"] + counts["fp"]),
                    "recall": (counts["tp"], counts["tp"] + counts["fn"]),
                    "f1": (2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"])}
    for name, (numerator, denominator) in denominators.items():
        _require_close(numerator / denominator if denominator else 0.0,
                       float(classification[name]), name)
    return {"thresholds": {"final": float(frame["threshold_final"].iloc[0]),
                           "early": float(frame["threshold_early"].iloc[0])},
            "counts": counts, "accuracy": counts["accuracy"], **metrics}


def _protected_records(config: OptimizationConfig) -> dict:
    protection = json.loads(project_path(config.project_root,
                            f"{PROTOCOL_DIR}/baseline_protection.json").read_text(encoding="utf-8"))
    return {row["path"]: row for row in protection["records"] if row["anchor"] == "project"}


def legacy_oof_recompute(config: OptimizationConfig) -> dict:
    """Replay the frozen original OOF metrics from its saved predictions.

    Read-only: every input hash is rechecked against the O02 protection list
    before it is opened, and nothing is written.
    """
    records = _protected_records(config)
    hashes = {}
    for name in LEGACY_OOF_FILES:
        row = records.get(name)
        if row is None:
            raise ValueError(f"Legacy artifact is not in the protection list: {name}")
        path = project_path(config.project_root, name)
        actual = file_sha256(path)
        if actual != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Protected legacy artifact changed: {name}")
        hashes[name] = actual
    final = pd.read_parquet(project_path(config.project_root, LEGACY_OOF_FILES[0]))
    saved = json.loads(project_path(config.project_root, LEGACY_OOF_FILES[2]).read_text(encoding="utf-8"))
    replayed = metrics_from_final_predictions(final)
    counts = confusion_counts(final["label"].to_numpy(), final["final_prediction"].to_numpy())
    if counts["users"] != 909 or final["user_id"].duplicated().any():
        raise AssertionError("Legacy OOF cohort is not 909 unique users")
    if {key: counts[key] for key in LEGACY_OOF_COUNTS} != LEGACY_OOF_COUNTS:
        raise AssertionError(f"Replayed legacy confusion matrix differs from the frozen report: {counts}")
    differences: dict[str, float] = {}
    for section, values in replayed.items():
        for name, value in values.items():
            stored = saved["point_estimate"][section][name]
            if value is None or stored is None:
                if value is not None or stored is not None:
                    raise AssertionError(f"Legacy metric presence differs: {section}.{name}")
                differences[f"{section}.{name}"] = 0.0
            else:
                differences[f"{section}.{name}"] = abs(float(value) - float(stored))
    maximum = max(differences.values(), default=0.0)
    if maximum > METRIC_TOLERANCE:
        raise AssertionError(f"Saved legacy metrics cannot be replayed from the saved predictions: {maximum}")
    return {"status": "pass", "files": hashes, "users": counts["users"], "counts": counts,
            "accuracy": counts["accuracy"], "maximum_metric_abs_difference": maximum,
            "metrics": replayed, "predictions_sha256": hashes[LEGACY_OOF_FILES[0]]}


def legacy_oof_decision_frame(config: OptimizationConfig) -> pd.DataFrame:
    """R0's frozen OOF decisions as a decision frame, for descriptive comparison only.

    Read-only and hash-checked. The original used one early-selected threshold for
    both decisions, so both threshold columns carry that value and no new
    threshold is selected here.
    """
    records = _protected_records(config)
    for name in (LEGACY_OOF_FILES[0], LEGACY_OOF_FILES[3]):
        row = records.get(name)
        if row is None or file_sha256(project_path(config.project_root, name)) != row["sha256"]:
            raise ValueError(f"Protected legacy artifact changed or unlisted: {name}")
    final = pd.read_parquet(project_path(config.project_root, LEGACY_OOF_FILES[0]))
    frame = pd.DataFrame({
        "user_id": [str(value) for value in final["user_id"]],
        "label": final["label"].to_numpy(dtype=np.int8),
        "length": final["length"].to_numpy(dtype=np.int16),
        "probability": final["probability"].to_numpy(dtype=np.float64),
        "threshold_final": final["threshold"].to_numpy(dtype=np.float64),
        "threshold_early": final["threshold"].to_numpy(dtype=np.float64),
        "final_prediction": final["final_prediction"].to_numpy(dtype=np.int8),
        "alert_prediction": final["alert_prediction"].to_numpy(dtype=np.int8),
        "alert_position": final["alert_position"].to_numpy(dtype=np.float64)})
    if frame["user_id"].duplicated().any():
        raise ValueError("Legacy OOF frame contains duplicate users")
    return frame.sort_values("user_id", kind="stable").reset_index(drop=True)


def validate_bundle_artifacts(config: OptimizationConfig, relative_directory: str) -> dict:
    """Independent reload check of one saved bundle directory."""
    bundle = load_bundle(config, relative_directory)
    record = bundle.provenance
    sequences = bundle.sequences()
    return {"status": "pass", "directory": relative_directory, "stage": record["stage"],
            "outer_fold": record["outer_fold"], "inner_fold": record["inner_fold"],
            "role": record["role"], "candidate_id": record["candidate_id"], "seeds": record["seeds"],
            "users": record["users"], "unique_users": len(set(bundle.user_ids)),
            "prefix_rows": record["prefix_rows"], "expected_prefix_rows": int(sum(len(s.probabilities) for s in sequences)),
            "class_counts": record["class_counts"], "predictions_sha256": bundle.predictions_sha256(),
            "model_or_ensemble_sha256": bundle.model_or_ensemble_sha256(),
            "aggregation": record["aggregation"], "code_sha256": record["code_sha256"]}
