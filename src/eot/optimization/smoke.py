"""O09 smoke pipeline: two-epoch subset training, recovery, threshold selection
and evaluation on the frozen inner-holdout cohort.

The smoke adapter does not copy any loss, metric, threshold or split formula. It
builds a small legitimate G/E subset, calls the same O07 fit/read/load engine,
uses the O06 selector on the real H_j scope, and applies O08 decisions/metrics.
Every write is confined to ``outputs/optimization/v1/smoke/`` and smoke results
must never be reported as screening/stability/final results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from eot.config import ExperimentConfig
from eot.dataset import SequenceStore, UserSequenceDataset
from eot.evaluation import predict_dataset
from eot.metrics import RiskSequence

from . import training as training_engine
from .config import (OptimizationConfig, canonical_json, file_sha256,
                     load_optimization_config, project_path)
from .data import (POST_COLUMNS, InnerDataContext, _inputs,
                   fit_training_statistics, load_inner_context)
from .decisions import (SelectionPredictions, ThresholdSelection, load_selection_scope,
                        select_dual_thresholds)
from .evaluation import PredictionBundle, decision_frame, evaluate_frame
from .runs import make_run_identity, source_fingerprint
from .splits import checked_ids, digest

SMOKE_ROOT = "outputs/optimization/v1/smoke"
SMOKE_CANDIDATES = ("A", "B", "C", "D")
SMOKE_SEED = 42
SMOKE_SUBSET = {"train_negative": 6, "train_positive": 6,
                "early_negative": 4, "early_positive": 4}
SMOKE_SOURCES = tuple(dict.fromkeys((*training_engine.TRAINING_SOURCES,
    "src/eot/evaluation.py", "src/eot/metrics.py",
    "src/eot/optimization/decisions.py", "src/eot/optimization/evaluation.py",
    "src/eot/optimization/comparison.py", "src/eot/optimization/smoke.py")))
REQUIRED_TRAINING = {
    "optimizer": "adamw", "learning_rate": .0008, "weight_decay": .01,
    "batch_users": 128, "mixed_precision": False, "max_epochs": 50,
    "early_stopping_metric": "pr_auc", "early_stopping_patience": 8,
    "gradient_clip_norm": 1.0, "seeds": [42, 43, 44],
}


def _role_counts(store: SequenceStore, ids: Iterable[str]) -> dict:
    values = checked_ids(ids, "smoke role")
    labels = [int(store.users[user_id].label) for user_id in values]
    positive = int(sum(labels))
    return {"users": len(values), "negative": len(values) - positive, "positive": positive}


def _balanced_subset(store: SequenceStore, ids: Iterable[str], *, positive: int,
                     negative: int, role: str) -> tuple[str, ...]:
    if type(positive) is not int or type(negative) is not int or positive < 1 or negative < 1:
        raise ValueError("Smoke subset limits must be positive integers")
    values = checked_ids(ids, role)
    by_class = {0: [user_id for user_id in values if store.users[user_id].label == 0],
                1: [user_id for user_id in values if store.users[user_id].label == 1]}
    if len(by_class[0]) < negative or len(by_class[1]) < positive:
        raise ValueError(f"{role} does not have enough users for a balanced smoke subset")
    selected = by_class[0][:negative] + by_class[1][:positive]
    return tuple(sorted(selected))


def choose_smoke_subset(store: SequenceStore, context: InnerDataContext, *,
                        limits: dict | None = None) -> "SmokeSubset":
    """Deterministic balanced subset of G and E; H remains the full frozen H_j."""
    values = dict(SMOKE_SUBSET if limits is None else limits)
    required = {"train_negative", "train_positive", "early_negative", "early_positive"}
    if set(values) != required:
        raise ValueError(f"Smoke subset requires exactly {sorted(required)}")
    train = _balanced_subset(store, context.user_ids("train"), positive=values["train_positive"],
                             negative=values["train_negative"], role="smoke train")
    early = _balanced_subset(store, context.user_ids("early_stop"), positive=values["early_positive"],
                             negative=values["early_negative"], role="smoke early_stop")
    holdout = tuple(checked_ids(context.user_ids("holdout"), "smoke holdout"))
    roles = {"train": train, "early_stop": early, "holdout": holdout}
    for left, right in (("train", "early_stop"), ("train", "holdout"), ("early_stop", "holdout")):
        if set(roles[left]) & set(roles[right]):
            raise ValueError(f"Smoke roles overlap: {left}, {right}")
    counts = {role: _role_counts(store, ids) for role, ids in roles.items()}
    return SmokeSubset(train, early, holdout, counts)


@dataclass(frozen=True)
class SmokeSubset:
    train: tuple[str, ...]
    early_stop: tuple[str, ...]
    holdout: tuple[str, ...]
    counts: dict

    def as_record(self) -> dict:
        return {"train": list(self.train), "early_stop": list(self.early_stop),
                "holdout": list(self.holdout), "counts": self.counts}


@dataclass(frozen=True)
class SmokeTrainingPlan:
    config: OptimizationConfig
    context: InnerDataContext
    identity: object
    subset: SmokeSubset
    _binding_json: str

    @property
    def binding(self) -> dict:
        return json.loads(self._binding_json)


@dataclass(frozen=True)
class SmokePreparation:
    configs: dict
    store: SequenceStore
    full_context: InnerDataContext
    subset: SmokeSubset
    statistics: dict
    context: InnerDataContext
    input_sha256: dict
    plans: dict
    scope: object


def _legacy_config(config: OptimizationConfig) -> ExperimentConfig:
    return ExperimentConfig(config.base_values, config.project_root / "configs/eot_main.yaml",
                            config.project_root, config.base_config_hash)


def _validate_training_constants(config: OptimizationConfig) -> None:
    training = config.section("training")
    if any(training.get(key) != value for key, value in REQUIRED_TRAINING.items()):
        raise ValueError("Smoke refuses to run because the locked training constants differ")


def _build_plan(config: OptimizationConfig, context: InnerDataContext, subset: SmokeSubset,
                input_sha256: dict) -> SmokeTrainingPlan:
    _validate_training_constants(config)
    fresh = load_optimization_config(config.source, project_root=config.project_root)
    if fresh != config:
        raise ValueError("Config object differs from its locked, current file")
    code_sha = source_fingerprint(config.project_root, SMOKE_SOURCES)
    split_sha = digest({"split": context.split_sha256, "statistics": context.statistics_sha256,
                        "smoke_subset": subset.as_record()})
    identity = make_run_identity(config, stage="smoke", outer_fold=0, inner_fold=0,
                                 seed=SMOKE_SEED, code_sha256=code_sha, split_sha256=split_sha)
    binding = {
        "schema_version": 1, "run_id": identity.run_id, **identity.metadata,
        "split_content_sha256": context.split_sha256,
        "statistics_sha256": context.statistics_sha256,
        "training_statistics": context.statistics,
        "train_user_ids": context.user_ids("train"),
        "early_stop_user_ids": context.user_ids("early_stop"),
        "holdout_user_ids": context.user_ids("holdout"),
        "training": config.section("training"),
        "loss": config.values["loss"],
        "input_sha256": input_sha256,
        "smoke_subset": subset.as_record(),
        "smoke_stage": {"train_epochs": 2, "device_independent": True,
                        "holdout_role": "frozen inner_holdout H_j"},
        "early_stop_role": "early_stop",
        "early_stop_rule": "E AP > best AP + 1e-12; ties retain earlier epoch",
    }
    return SmokeTrainingPlan(config, context, identity, subset, canonical_json(binding))


def prepare_smoke(project_root: str | Path) -> SmokePreparation:
    """Read-only smoke preflight: frozen data, balanced subset, settings, plans."""
    root = Path(project_root).resolve()
    configs = {candidate: load_optimization_config(
        root / f"configs/optimization/{candidate}.yaml", project_root=root)
        for candidate in SMOKE_CANDIDATES}
    reference = configs["A"]
    for candidate, config in configs.items():
        _validate_training_constants(config)
        if (config.protocol_sha256 != reference.protocol_sha256
                or config.protocol_lock_sha256 != reference.protocol_lock_sha256
                or config.base_config_hash != reference.base_config_hash):
            raise ValueError(f"Candidate {candidate} does not share the locked protocol/base")
    full_context = load_inner_context(reference, 0, 0)
    base = _legacy_config(reference)
    store = SequenceStore(base)
    subset = choose_smoke_subset(store, full_context)
    users, posts, _outers, input_sha256 = _inputs(reference)
    missing = set(subset.train) | set(subset.early_stop)
    if missing - set(users["user_id"]):
        raise ValueError("Smoke subset user missing from frozen processed users")
    statistics = fit_training_statistics(
        users, posts, subset.train, quantile=float(reference.base_values["time"]["clipping_quantile"]),
        max_posts=int(reference.base_values["data"]["max_posts"]))
    roles = {"train": list(subset.train), "early_stop": list(subset.early_stop),
             "holdout": list(subset.holdout)}
    split_sha = digest({"source_split_sha256": full_context.split_sha256,
                        "smoke_subset": subset.as_record(), "roles": roles})
    context = InnerDataContext(0, 0, split_sha, digest(statistics),
                               canonical_json(roles), canonical_json(statistics))
    plans = {candidate: _build_plan(config, context, subset, input_sha256)
             for candidate, config in configs.items()}
    scope = load_selection_scope(reference, role="inner_holdout", outer_fold=0, inner_fold=0)
    if scope.metadata["user_ids"] != list(subset.holdout):
        raise ValueError("Smoke holdout must be the complete frozen H_j scope")
    return SmokePreparation(configs, store, full_context, subset, statistics, context,
                            input_sha256, plans, scope)


def _device(name: str | None) -> torch.device:
    value = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if value.type not in {"cpu", "cuda"}:
        raise ValueError("Smoke training supports only CPU/CUDA")
    if value.type == "cuda" and value.index is None:
        value = torch.device("cuda", torch.cuda.current_device())
    return value


def _fit_two_epochs(plan: SmokeTrainingPlan, store: SequenceStore, *, device_name: str | None):
    """Run epoch 1 paused, then resume to epoch 2 through the production engine."""
    existing = plan.identity.artifact_path("run.json").exists()
    if not existing:
        training_engine._fit_inner(plan, store, device_name=device_name,
                                   resume=False, stop_after_epoch=1)
    manifest = training_engine._fit_inner(plan, store, device_name=device_name,
                                          resume=True, stop_after_epoch=2)
    state, _ = training_engine.read_checkpoint(plan)
    epochs = [int(row["epoch"]) for row in manifest["history"]]
    if epochs != [1, 2]:
        raise AssertionError(f"Smoke run did not execute exactly two epochs: {epochs}")
    for row in manifest["history"]:
        for key in ("train_loss", "train_final_loss", "train_prefix_loss",
                    "early_stop_loss", "early_stop_pr_auc"):
            if not math.isfinite(float(row[key])):
                raise FloatingPointError(f"Smoke produced a non-finite {key}")
    return state, manifest


def _check_prediction_pair(final: pd.DataFrame, prefix: pd.DataFrame, expected_ids: Sequence[str]) -> None:
    expected = checked_ids(expected_ids, "smoke holdout")
    if [str(value) for value in final["user_id"]] != expected:
        raise ValueError("Smoke predictions do not cover exactly the frozen H_j in order")
    if final["user_id"].duplicated().any():
        raise ValueError("Smoke predictions contain duplicate users")
    grouped = {str(user_id): frame for user_id, frame in prefix.groupby("user_id", sort=False)}
    if set(grouped) != set(expected):
        raise ValueError("Smoke prefix predictions do not cover exactly the frozen H_j")
    for row in final.itertuples(index=False):
        user_id = str(row.user_id)
        frame = grouped[user_id]
        positions = frame["position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(1, len(frame) + 1)):
            raise ValueError(f"Smoke prefix positions are not contiguous for {user_id}")
        if int(frame["length"].iloc[0]) != int(row.length) or len(frame) != int(row.length):
            raise ValueError(f"Smoke prefix length mismatch for {user_id}")
        if abs(float(frame["probability"].iloc[-1]) - float(row.probability)) > 1e-7:
            raise ValueError(f"Smoke final probability is not the last valid prefix for {user_id}")
    if set(final["label"].tolist()) != {0, 1}:
        raise ValueError("Smoke H_j must contain both classes")


def _predict_holdout(plan: SmokeTrainingPlan, store: SequenceStore, *, device_name: str | None):
    device = _device(device_name)
    model_one, manifest = training_engine.load_trained_model(plan)
    holdout = plan.context.dataset(store, "holdout")
    batch_size = int(plan.config.section("training")["batch_users"])
    final, prefix = predict_dataset(model_one, holdout, device=device, batch_size=batch_size)
    _check_prediction_pair(final, prefix, plan.context.user_ids("holdout"))

    # Independent reload: two freshly loaded best checkpoints must score a small
    # deterministic H_j prefix identically.
    check_ids = list(plan.context.user_ids("holdout"))[:16]
    check_dataset = UserSequenceDataset(store, check_ids, plan.context.time_normalizer)
    first, first_prefix = predict_dataset(model_one, check_dataset, device=device, batch_size=batch_size)
    reload_model, reload_manifest = training_engine.load_trained_model(plan)
    second, second_prefix = predict_dataset(reload_model, check_dataset, device=device, batch_size=batch_size)
    max_final = float(np.max(np.abs(first["probability"].to_numpy(np.float64)
                                    - second["probability"].to_numpy(np.float64))))
    max_prefix = float(np.max(np.abs(first_prefix["probability"].to_numpy(np.float64)
                                     - second_prefix["probability"].to_numpy(np.float64))))
    if max_final > 1e-7 or max_prefix > 1e-7:
        raise AssertionError("Best-checkpoint reload did not reproduce smoke predictions")
    if manifest["checkpoint_sha256"] != reload_manifest["checkpoint_sha256"]:
        raise AssertionError("Reloaded run manifest changed checkpoint identity")
    reload_record = {"checkpoint_sha256": manifest["checkpoint_sha256"],
                     "users_checked": len(check_ids), "max_abs_final_difference": max_final,
                     "max_abs_prefix_difference": max_prefix, "matched": True}
    return final, prefix, manifest, reload_record


def _selection_predictions(config: OptimizationConfig, final: pd.DataFrame, prefix: pd.DataFrame,
                           checkpoint_sha256: str) -> SelectionPredictions:
    grouped = {str(user_id): frame for user_id, frame in prefix.groupby("user_id", sort=False)}
    sequences = []
    for row in final.itertuples(index=False):
        user_id = str(row.user_id)
        values = grouped[user_id]["probability"].to_numpy(dtype=np.float64)
        sequences.append(RiskSequence(user_id, int(row.label), values))
    ordered = tuple(sorted(sequences, key=lambda item: item.user_id))
    return SelectionPredictions(config.candidate_id, config.config_hash,
                                ((SMOKE_SEED, checkpoint_sha256),), ordered)


def _bundle(final: pd.DataFrame, prefix: pd.DataFrame, candidate_id: str) -> PredictionBundle:
    record = {"schema_version": 1, "candidate_id": candidate_id, "stage": "smoke",
              "role": "inner_holdout", "outer_fold": 0, "inner_fold": 0,
              "aggregation": "single_seed", "model": "smoke"}
    return PredictionBundle(canonical_json(record), final, prefix)


def _summary(metrics: dict) -> dict:
    classification = metrics["classification"]
    return {"counts": metrics["counts"], "accuracy": metrics["accuracy"],
            "recall": classification["recall"], "precision": classification["precision"],
            "f1": classification["f1"], "pr_auc": classification["pr_auc"],
            "early_f1": metrics["early_detection"]["early_f1"],
            "f_latency": metrics["early_detection"]["f_latency"]}


def _evaluate_selection(candidate_id: str, selection: ThresholdSelection,
                        a_bundle: PredictionBundle, candidate_bundle: PredictionBundle) -> dict:
    record = selection.manifest
    r1 = decision_frame(a_bundle, threshold_final=record["R1"]["threshold_final"],
                        threshold_early=record["R1"]["threshold_early"])
    r2 = decision_frame(a_bundle, threshold_final=record["R2"]["threshold_final"],
                        threshold_early=record["R2"]["threshold_early"])
    candidate = record["candidate"]
    if candidate_id == "A":
        opt = r2.copy()
        opt_source = "A_reference_R2"
    elif candidate["threshold_final"] is None or candidate["status"] == "infeasible":
        opt = r2.copy()
        opt_source = f"fallback_R2_{candidate['status']}"
    else:
        opt = decision_frame(candidate_bundle, threshold_final=candidate["threshold_final"],
                             threshold_early=candidate["threshold_early"])
        opt_source = f"candidate_{candidate_id}"
    frames = {"R1": r1, "R2": r2, "OPT": opt}
    return {"opt_source": opt_source,
            "frames": frames,
            "metrics": {name: evaluate_frame(frame) for name, frame in frames.items()}}


def _smoke_path(root: Path, relative: str) -> Path:
    return project_path(root, f"{SMOKE_ROOT}/{relative}", allowed_prefix=SMOKE_ROOT)


def _write_json(root: Path, relative: str, value: dict) -> Path:
    path = _smoke_path(root, relative)
    content = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different smoke artifact: {relative}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_parquet(root: Path, relative: str, frame: pd.DataFrame) -> Path:
    path = _smoke_path(root, relative)
    if path.exists():
        stored = pd.read_parquet(path)
        if not stored.equals(frame.reset_index(drop=True)):
            raise ValueError(f"Refusing to overwrite a different smoke prediction: {relative}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_smoke(candidate_ids: Sequence[str] = SMOKE_CANDIDATES, *, project_root: str | Path,
              device_name: str | None = None) -> dict:
    """Train, recover, threshold-select and evaluate A-D on the smoke subset."""
    candidates = tuple(candidate_ids)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Smoke candidates must be a non-empty unique sequence")
    if any(candidate not in SMOKE_CANDIDATES for candidate in candidates):
        raise ValueError(f"Smoke supports only {SMOKE_CANDIDATES}")
    if candidates != tuple(candidate for candidate in SMOKE_CANDIDATES if candidate in candidates):
        raise ValueError("Smoke runs must keep protocol order A, B, C, D")
    if "A" not in candidates:
        raise ValueError("Smoke runs require A as the R1/R2 reference")
    preparation = prepare_smoke(project_root)
    root = Path(project_root).resolve()
    artifacts, candidate_reports = {}, {}

    for candidate in candidates:
        plan = preparation.plans[candidate]
        state, manifest = _fit_two_epochs(plan, preparation.store, device_name=device_name)
        final, prefix, manifest, reload_record = _predict_holdout(
            plan, preparation.store, device_name=device_name)
        run_directory = plan.identity.directory
        final_path = _write_parquet(root, f"predictions/{candidate}_holdout_final.parquet", final)
        prefix_path = _write_parquet(root, f"predictions/{candidate}_holdout_prefix.parquet", prefix)
        artifacts[candidate] = {"plan": plan, "state": state, "manifest": manifest,
                                "final": final, "prefix": prefix,
                                "final_path": final_path, "prefix_path": prefix_path,
                                "reload": reload_record, "run_directory": run_directory}
        candidate_reports[candidate] = {
            "run_id": manifest["binding"]["run_id"],
            "run_directory": _relative(root, run_directory),
            "run_manifest": _relative(root, run_directory / "run.json"),
            "checkpoint": _relative(root, run_directory / manifest["checkpoint"]),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "run_manifest_sha256": file_sha256(run_directory / "run.json"),
            "epochs": [int(row["epoch"]) for row in manifest["history"]],
            "history": manifest["history"],
            "loss_finite": all(math.isfinite(float(row[key]))
                               for row in manifest["history"]
                               for key in ("train_loss", "train_final_loss", "train_prefix_loss",
                                           "early_stop_loss", "early_stop_pr_auc")),
            "best_epoch": int(manifest["best_epoch"]),
            "best_score": float(manifest["best_score"]),
            "positive_weight": float(manifest["positive_weight"]),
            "training_users": len(manifest["binding"]["train_user_ids"]),
            "early_stop_users": len(manifest["binding"]["early_stop_user_ids"]),
            "selection_users": len(manifest["binding"]["holdout_user_ids"]),
            "reload": reload_record,
            "final_predictions": _relative(root, final_path),
            "final_predictions_sha256": file_sha256(final_path),
            "prefix_predictions": _relative(root, prefix_path),
            "prefix_predictions_sha256": file_sha256(prefix_path),
        }
        print(f"smoke {candidate}: epochs={manifest['history'][-1]['epoch']} "
              f"loss={manifest['history'][-1]['train_loss']:.6f} "
              f"best_epoch={manifest['best_epoch']} reload={reload_record['matched']}", flush=True)

    a_reference = _selection_predictions(
        preparation.configs["A"], artifacts["A"]["final"], artifacts["A"]["prefix"],
        artifacts["A"]["manifest"]["checkpoint_sha256"])
    selections = {}
    for candidate in candidates:
        candidate_config = preparation.configs[candidate]
        candidate_reference = (a_reference if candidate == "A" else
            _selection_predictions(candidate_config, artifacts[candidate]["final"],
                                   artifacts[candidate]["prefix"],
                                   artifacts[candidate]["manifest"]["checkpoint_sha256"]))
        selection = select_dual_thresholds(candidate_config, preparation.scope,
                                           a_reference, candidate_reference)
        selection_path = selection.save(candidate_config, f"{SMOKE_ROOT}/thresholds/{candidate}.json")
        selections[candidate] = selection
        candidate_reports[candidate]["threshold_selection"] = _relative(root, selection_path)
        candidate_reports[candidate]["threshold_selection_sha256"] = file_sha256(selection_path)
        candidate_reports[candidate]["thresholds"] = {
            "R1": selection.manifest["R1"], "R2": selection.manifest["R2"],
            "candidate_status": selection.manifest["candidate"]["status"],
            "candidate_threshold_final": selection.manifest["candidate"]["threshold_final"],
            "candidate_threshold_early": selection.manifest["candidate"]["threshold_early"]}

    a_bundle = _bundle(artifacts["A"]["final"], artifacts["A"]["prefix"], "A")
    for candidate in candidates:
        candidate_bundle = (a_bundle if candidate == "A" else
            _bundle(artifacts[candidate]["final"], artifacts[candidate]["prefix"], candidate))
        evaluation = _evaluate_selection(candidate, selections[candidate], a_bundle, candidate_bundle)
        object_summaries = {}
        for name, metrics in evaluation["metrics"].items():
            decision_path = _write_parquet(root, f"decisions/{candidate}_{name}.parquet",
                                           evaluation["frames"][name])
            object_summaries[name] = {"decision_frame": _relative(root, decision_path),
                                      "decision_frame_sha256": file_sha256(decision_path),
                                      **_summary(metrics)}
        metrics_path = _write_json(root, f"reports/{candidate}_metrics.json",
                                   {"candidate_id": candidate,
                                    "opt_source": evaluation["opt_source"],
                                    "objects": {name: metrics for name, metrics in evaluation["metrics"].items()}})
        candidate_reports[candidate].update({
            "opt_source": evaluation["opt_source"],
            "objects": object_summaries,
            "metrics_report": _relative(root, metrics_path),
            "metrics_report_sha256": file_sha256(metrics_path),
        })

    checks = {
        "all_candidates_two_epochs": all(candidate_reports[c]["epochs"] == [1, 2] for c in candidates),
        "all_losses_finite": all(candidate_reports[c]["loss_finite"] for c in candidates),
        "all_reloads_matched": all(candidate_reports[c]["reload"]["matched"] for c in candidates),
        "threshold_selection_completed": all(candidate in selections for candidate in candidates),
        "evaluation_completed": all("objects" in candidate_reports[c] for c in candidates),
        "outer_test_labels_accessed": False,
        "formal_results_not_mixed": True,
        "small_scale_only": (len(preparation.subset.train) <= 16
                             and len(preparation.subset.early_stop) <= 12),
    }
    passed = all(value for key, value in checks.items() if key != "outer_test_labels_accessed")
    passed = passed and checks["outer_test_labels_accessed"] is False
    report = {
        "schema_version": 1, "task": "O09", "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": preparation.configs["A"].values["protocol_id"],
        "protocol_sha256": preparation.configs["A"].protocol_sha256,
        "protocol_lock_sha256": preparation.configs["A"].protocol_lock_sha256,
        "base_config_hash": preparation.configs["A"].base_config_hash,
        "outer_fold": 0, "inner_fold": 0, "seed": SMOKE_SEED,
        "selection_role": "inner_holdout",
        "selection_scope_sha256": digest(preparation.scope.metadata["user_ids"]),
        "selection_users": len(preparation.scope.metadata["user_ids"]),
        "smoke_subset": preparation.subset.as_record(),
        "statistics": {"fit_user_count": preparation.statistics["fit_user_count"],
                       "class_counts": preparation.statistics["class_counts"],
                       "base_positive_weight": preparation.statistics["base_positive_weight"],
                       "fit_labels_sha256": preparation.statistics["fit_labels_sha256"]},
        "input_sha256": preparation.input_sha256,
        "candidates": candidate_reports,
        "checks": checks,
        "caveat": ("Smoke only: no outer test performance was read, and these metrics are not "
                   "screening/stability/final results."),
    }
    _write_json(root, "smoke_acceptance.json", report)
    return report
