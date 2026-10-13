"""O10 inner screening orchestration.

One seed-42 run is completed for each of A-D, outer fold, and inner fold. The
module reuses the O07 training/recovery engine, O06 dual-threshold selector and
O08 decision/metrics code. It never loads an outer test cohort or test label;
selection always uses the frozen H_j scope from O04.

Artifacts are written below ``outputs/optimization/v1/screening/``. A run ID,
prediction manifest, threshold manifest and per-outer report carry the hashes
needed by the later O11/O13 stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from eot.dataset import SequenceStore, UserSequenceDataset
from eot.evaluation import predict_dataset
from eot.metrics import RiskSequence

from . import training as training_engine
from .comparison import pool_fold_frames
from .config import (OptimizationConfig, canonical_json, file_sha256,
                     load_optimization_config, project_path)
from .decisions import SelectionPredictions, load_selection_scope, select_dual_thresholds
from .evaluation import DECISION_COLUMNS, PredictionBundle, decision_frame, evaluate_frame
from .splits import checked_ids, digest

SCREENING_ROOT = "outputs/optimization/v1/screening"
CANDIDATES = ("A", "B", "C", "D")
INNER_FOLDS = (0, 1, 2)
OUTER_FOLDS = (0, 1, 2, 3, 4)
SCREENING_SEED = 42
PREDICTION_TOLERANCE = 1e-6


def _device(name: str | None) -> torch.device:
    value = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if value.type not in {"cpu", "cuda"}:
        raise ValueError("Screening supports only CPU/CUDA")
    if value.type == "cuda" and value.index is None:
        value = torch.device("cuda", torch.cuda.current_device())
    return value


def candidate_configs(root: Path) -> dict[str, OptimizationConfig]:
    root = Path(root).resolve()
    configs = {candidate: load_optimization_config(root / f"configs/optimization/{candidate}.yaml")
               for candidate in CANDIDATES}
    reference = configs["A"]
    for candidate, config in configs.items():
        if (config.protocol_sha256 != reference.protocol_sha256
                or config.protocol_lock_sha256 != reference.protocol_lock_sha256
                or config.base_config_hash != reference.base_config_hash):
            raise ValueError(f"Candidate {candidate} does not share the locked protocol/base")
    return configs


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _root_path(root: Path, relative: str) -> Path:
    return project_path(root, f"{SCREENING_ROOT}/{relative}", allowed_prefix=SCREENING_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(root: Path, relative: str, value: dict) -> Path:
    path = _root_path(root, relative)
    content = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different screening artifact: {relative}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    return path


def _write_parquet(root: Path, relative: str, frame: pd.DataFrame) -> Path:
    path = _root_path(root, relative)
    if path.exists():
        stored = pd.read_parquet(path)
        if not stored.equals(frame.reset_index(drop=True)):
            raise ValueError(f"Refusing to overwrite different screening predictions: {relative}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return path


def _run_dir(plan) -> Path:
    return plan.identity.directory


def _read_manifest(plan) -> dict:
    state, manifest = training_engine.read_checkpoint(plan)
    if manifest.get("status") != "complete":
        raise ValueError(f"Screening run is not complete: {plan.identity.run_id}")
    if [row["epoch"] for row in manifest["history"]] != list(range(1, int(manifest["epoch"]) + 1)):
        raise ValueError("Screening run history is not contiguous")
    for row in manifest["history"]:
        for key in ("train_loss", "train_final_loss", "train_prefix_loss",
                    "early_stop_loss", "early_stop_pr_auc"):
            if not math.isfinite(float(row[key])):
                raise FloatingPointError(f"Screening run has non-finite {key}: {plan.identity.run_id}")
    return manifest


def ensure_screening_run(plan, store: SequenceStore, *, device_name: str | None) -> dict:
    """Run or resume one screening job until the O07 engine reports complete."""
    manifest_path = plan.identity.artifact_path("run.json")
    if manifest_path.exists():
        _state, manifest = training_engine.read_checkpoint(plan)
        if manifest.get("status") != "complete":
            training_engine._fit_inner(plan, store, device_name=device_name, resume=True,
                                           stop_after_epoch=None)
    else:
        training_engine._fit_inner(plan, store, device_name=device_name, resume=False,
                                           stop_after_epoch=None)
    return _read_manifest(plan)


def _check_prediction_pair(final: pd.DataFrame, prefix: pd.DataFrame,
                           expected_ids: Sequence[str]) -> None:
    expected = checked_ids(expected_ids, "screening holdout")
    if [str(value) for value in final["user_id"]] != expected:
        raise ValueError("Screening predictions do not match frozen H_j order")
    if final["user_id"].duplicated().any():
        raise ValueError("Screening predictions contain duplicate users")
    grouped = {str(user_id): frame for user_id, frame in prefix.groupby("user_id", sort=False)}
    if set(grouped) != set(expected):
        raise ValueError("Screening prefix predictions do not match frozen H_j")
    for row in final.itertuples(index=False):
        user_id = str(row.user_id)
        frame = grouped[user_id]
        if not frame["label"].eq(row.label).all() or not frame["length"].eq(row.length).all():
            raise ValueError(f"Screening prefix label/length mismatch for {user_id}")
        values = frame["probability"].to_numpy(dtype=np.float64)
        if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                or not np.isfinite(row.probability) or not 0 <= row.probability <= 1):
            raise ValueError(f"Invalid screening probability for {user_id}")
        if not np.array_equal(frame["position"].to_numpy(dtype=np.int64),
                              np.arange(1, len(frame) + 1, dtype=np.int64)):
            raise ValueError(f"Screening prefix positions are not contiguous for {user_id}")
        if int(frame["length"].iloc[0]) != int(row.length) or len(frame) != int(row.length):
            raise ValueError(f"Screening prefix length mismatch for {user_id}")
        if abs(float(frame["probability"].iloc[-1]) - float(row.probability)) > PREDICTION_TOLERANCE:
            raise ValueError(f"Screening final probability is not the last valid prefix for {user_id}")
    if set(final["label"].tolist()) != {0, 1}:
        raise ValueError("Screening H_j must contain both classes")


def export_predictions(config: OptimizationConfig, plan, store: SequenceStore, *,
                       outer_fold: int, inner_fold: int, device_name: str | None) -> dict:
    """Load the best checkpoint, predict H_j, and persist checksum-bound predictions."""
    manifest = _read_manifest(plan)
    relative = (f"evaluation/outer_{outer_fold}/inner_{inner_fold}/"
                f"{config.candidate_id}/seed_{SCREENING_SEED}")
    metadata_relative = f"{relative}/prediction_manifest.json"
    metadata_path = _root_path(config.project_root, metadata_relative)
    if metadata_path.exists():
        record = _read_json(metadata_path)
        if record.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            raise ValueError("Existing H_j predictions belong to a different checkpoint")
        _selection_predictions(config, record)
        return record

    device = _device(device_name)
    model, manifest = training_engine.load_trained_model(plan, device_name=str(device))
    dataset = plan.context.dataset(store, "holdout")
    batch_size = int(config.section("training")["batch_users"])
    final, prefix = predict_dataset(model, dataset, device=device, batch_size=batch_size)
    expected = plan.context.user_ids("holdout")
    _check_prediction_pair(final, prefix, expected)
    final_path = _write_parquet(config.project_root, f"{relative}/final.parquet", final)
    prefix_path = _write_parquet(config.project_root, f"{relative}/prefix.parquet", prefix)
    record = {
        "schema_version": 1, "candidate_id": config.candidate_id,
        "outer_fold": outer_fold, "inner_fold": inner_fold, "seed": SCREENING_SEED,
        "run_id": manifest["binding"]["run_id"],
        "run_manifest_sha256": file_sha256(_run_dir(plan) / "run.json"),
        "checkpoint": _relative(config.project_root, _run_dir(plan) / manifest["checkpoint"]),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "best_epoch": int(manifest["best_epoch"]),
        "users": len(expected), "class_counts": {
            "negative": int(np.sum(final["label"].to_numpy() == 0)),
            "positive": int(np.sum(final["label"].to_numpy() == 1))},
        "labels_sha256": digest([[str(row.user_id), int(row.label)]
                                 for row in final.itertuples(index=False)]),
        "final": _relative(config.project_root, final_path),
        "final_sha256": file_sha256(final_path),
        "prefix": _relative(config.project_root, prefix_path),
        "prefix_sha256": file_sha256(prefix_path),
    }
    _write_json(config.project_root, metadata_relative, record)
    return record


def _selection_predictions(config: OptimizationConfig, record: dict) -> SelectionPredictions:
    root = config.project_root
    for key in ("final", "prefix"):
        path = project_path(root, record[key], allowed_prefix=SCREENING_ROOT)
        if file_sha256(path) != record[key + "_sha256"]:
            raise ValueError(f"Cached screening {key} checksum mismatch")
    final = pd.read_parquet(root / record["final"])
    prefix = pd.read_parquet(root / record["prefix"])
    _check_prediction_pair(final, prefix, final["user_id"].tolist())
    grouped = {str(user_id): frame for user_id, frame in prefix.groupby("user_id", sort=False)}
    sequences = []
    for row in final.itertuples(index=False):
        user_id = str(row.user_id)
        sequences.append(RiskSequence(user_id, int(row.label),
                                      grouped[user_id]["probability"].to_numpy(dtype=np.float64)))
    if digest([[s.user_id, s.label] for s in sequences]) != record["labels_sha256"]:
        raise ValueError("Cached screening label source mismatch")
    return SelectionPredictions(config.candidate_id, config.config_hash,
                                ((SCREENING_SEED, record["checkpoint_sha256"]),),
                                tuple(sorted(sequences, key=lambda item: item.user_id)))


def select_inner_fold(configs: dict[str, OptimizationConfig], *, outer_fold: int,
                      inner_fold: int) -> dict:
    """Run O06 on H_j for A-D; all candidates share A's R1/R2 reference."""
    records = {candidate: _read_json(_root_path(
        configs[candidate].project_root,
        f"evaluation/outer_{outer_fold}/inner_{inner_fold}/{candidate}/seed_{SCREENING_SEED}/"
        "prediction_manifest.json")) for candidate in CANDIDATES}
    scope = load_selection_scope(configs["A"], role="inner_holdout",
                                 outer_fold=outer_fold, inner_fold=inner_fold)
    a_predictions = _selection_predictions(configs["A"], records["A"])
    summary = {"outer_fold": outer_fold, "inner_fold": inner_fold, "candidates": {}}
    for candidate in CANDIDATES:
        predictions = (a_predictions if candidate == "A"
                       else _selection_predictions(configs[candidate], records[candidate]))
        selection = select_dual_thresholds(configs[candidate], scope, a_predictions, predictions)
        selection_path = selection.save(
            configs[candidate],
            f"{SCREENING_ROOT}/thresholds/outer_{outer_fold}/inner_{inner_fold}/{candidate}.json")
        record = selection.manifest
        summary["candidates"][candidate] = {
            "selection": _relative(configs[candidate].project_root, selection_path),
            "R1_threshold_final": record["R1"]["threshold_final"],
            "R1_threshold_early": record["R1"]["threshold_early"],
            "R1_counts": record["R1"]["counts"],
            "R2_threshold_final": record["R2"]["threshold_final"],
            "R2_threshold_early": record["R2"]["threshold_early"],
            "R2_counts": record["R2"]["counts"],
            "candidate_status": record["candidate"]["status"],
            "candidate_threshold_final": record["candidate"]["threshold_final"],
            "candidate_threshold_early": record["candidate"]["threshold_early"],
            "candidate_counts": record["candidate"]["counts"]}
        print(f"screening select outer={outer_fold} inner={inner_fold} {candidate} "
              f"status={record['candidate']['status']} "
              f"threshold={record['candidate']['threshold_final']}", flush=True)
    _write_json(configs["A"].project_root,
                f"reports/selections/outer_{outer_fold}_inner_{inner_fold}.json", summary)
    return summary


def _load_selection(configs: dict[str, OptimizationConfig], outer_fold: int, inner_fold: int,
                    candidate: str) -> dict:
    path = _root_path(configs[candidate].project_root,
                      f"thresholds/outer_{outer_fold}/inner_{inner_fold}/{candidate}.json")
    wrapper = _read_json(path)
    if wrapper.get("sha256") != digest(wrapper.get("selection", {})):
        raise ValueError("Screening threshold manifest checksum mismatch")
    return wrapper["selection"]


def _load_predictions(config: OptimizationConfig, outer_fold: int, inner_fold: int) -> PredictionBundle:
    base = _root_path(config.project_root,
                      f"evaluation/outer_{outer_fold}/inner_{inner_fold}/"
                      f"{config.candidate_id}/seed_{SCREENING_SEED}")
    final = pd.read_parquet(base / "final.parquet")
    prefix = pd.read_parquet(base / "prefix.parquet")
    record = _read_json(base / "prediction_manifest.json")
    if set(final.columns) != {"user_id", "label", "length", "probability"}:
        raise ValueError("Screening final prediction columns differ")
    if set(prefix.columns) != {"user_id", "label", "length", "position", "probability"}:
        raise ValueError("Screening prefix prediction columns differ")
    return PredictionBundle(canonical_json({"stage": "screening", "candidate_id": config.candidate_id,
                                             "outer_fold": outer_fold, "inner_fold": inner_fold,
                                             "seed": SCREENING_SEED}),
                            final, prefix)


def _metric_summary(metrics: dict) -> dict:
    classification = metrics["classification"]
    return {"counts": metrics["counts"], "accuracy": metrics["accuracy"],
            "recall": classification["recall"], "precision": classification["precision"],
            "f1": classification["f1"], "pr_auc": classification["pr_auc"],
            "roc_auc": classification["roc_auc"],
            "balanced_accuracy": classification["balanced_accuracy"],
            "early_detection": metrics["early_detection"]}


def _fold_analysis(configs: dict[str, OptimizationConfig], outer_fold: int, inner_fold: int) -> dict:
    selections = {candidate: _load_selection(configs, outer_fold, inner_fold, candidate)
                  for candidate in CANDIDATES}
    bundles = {candidate: _load_predictions(configs[candidate], outer_fold, inner_fold)
               for candidate in CANDIDATES}
    a_record = selections["A"]
    a_bundle = bundles["A"]
    r1 = decision_frame(a_bundle, threshold_final=a_record["R1"]["threshold_final"],
                        threshold_early=a_record["R1"]["threshold_early"])
    r2 = decision_frame(a_bundle, threshold_final=a_record["R2"]["threshold_final"],
                        threshold_early=a_record["R2"]["threshold_early"])
    result = {"outer_fold": outer_fold, "inner_fold": inner_fold,
              "R1": _metric_summary(evaluate_frame(r1)),
              "R2": _metric_summary(evaluate_frame(r2)),
              "candidates": {}}
    frames = {"R2": r2}
    for candidate in CANDIDATES:
        selected = selections[candidate]["candidate"]
        if candidate == "A":
            frame = r2.copy()
            feasible = True
            source = "A_reference_R2"
        elif selected.get("threshold_final") is None or selected.get("status") == "infeasible":
            frame = None
            feasible = False
            source = "infeasible"
        else:
            frame = decision_frame(bundles[candidate],
                                   threshold_final=selected["threshold_final"],
                                   threshold_early=selected["threshold_early"])
            feasible = True
            source = f"candidate_{candidate}"
        result["candidates"][candidate] = {
            "feasible": feasible, "source": source,
            "threshold_final": None if not feasible else selected.get("threshold_final"),
            "threshold_early": None if not feasible else selected.get("threshold_early"),
            "metrics": None if frame is None else _metric_summary(evaluate_frame(frame))}
        if frame is not None:
            frames[candidate] = frame
    return {"summary": result, "frames": frames}


def _merge_frames(frames: Sequence[pd.DataFrame], *, outer_fold: int, name: str) -> pd.DataFrame:
    return pool_fold_frames(list(frames), roles=["inner_holdout"] * len(frames),
                            name=f"outer_{outer_fold}_{name}")


def _tradeoff(r2: dict, candidate: dict) -> dict:
    r2_counts, candidate_counts = r2["counts"], candidate["counts"]
    return {
        "delta_accuracy": candidate["accuracy"] - r2["accuracy"],
        "delta_recall": candidate["recall"] - r2["recall"],
        "delta_precision": candidate["precision"] - r2["precision"],
        "delta_f1": candidate["f1"] - r2["f1"],
        "delta_tp": candidate_counts["tp"] - r2_counts["tp"],
        "delta_tn": candidate_counts["tn"] - r2_counts["tn"],
        "delta_fp": candidate_counts["fp"] - r2_counts["fp"],
        "delta_fn": candidate_counts["fn"] - r2_counts["fn"],
        "correct_delta": candidate_counts["correct"] - r2_counts["correct"],
        "candidate_accuracy": candidate["accuracy"], "r2_accuracy": r2["accuracy"],
        "candidate_recall": candidate["recall"], "r2_recall": r2["recall"],
        "candidate_fp": candidate_counts["fp"], "r2_fp": r2_counts["fp"],
        "candidate_fn": candidate_counts["fn"], "r2_fn": r2_counts["fn"],
    }


def build_outer_report(configs: dict[str, OptimizationConfig], outer_fold: int) -> dict:
    folds = {j: _fold_analysis(configs, outer_fold, j) for j in INNER_FOLDS}
    r2_frames = [folds[j]["frames"]["R2"] for j in INNER_FOLDS]
    r2_merged_frame = _merge_frames(r2_frames, outer_fold=outer_fold, name="R2")
    r2_merged = _metric_summary(evaluate_frame(r2_merged_frame))
    report = {
        "schema_version": 1, "task": "O10", "outer_fold": outer_fold,
        "seed": SCREENING_SEED, "inner_folds": list(INNER_FOLDS),
        "R1": {str(j): folds[j]["summary"]["R1"] for j in INNER_FOLDS},
        "R2": {str(j): folds[j]["summary"]["R2"] for j in INNER_FOLDS},
        "R2_merged": r2_merged, "candidates": {}, "advanced": [], "fallback_to_A": False,
    }
    for candidate in CANDIDATES:
        per_fold = folds
        candidate_rows = {str(j): per_fold[j]["summary"]["candidates"][candidate]
                          for j in INNER_FOLDS}
        feasible_all = all(candidate_rows[str(j)]["feasible"] for j in INNER_FOLDS)
        merged_frame = None
        merged = None
        if candidate == "A":
            merged_frame = r2_merged_frame
            merged = r2_merged
            trade = None
            advance = False
        elif feasible_all:
            merged_frame = _merge_frames([per_fold[j]["frames"][candidate] for j in INNER_FOLDS],
                                         outer_fold=outer_fold, name=candidate)
            merged = _metric_summary(evaluate_frame(merged_frame))
            per_fold_ge = sum(
                per_fold[j]["summary"]["candidates"][candidate]["metrics"]["counts"]["correct"]
                >= per_fold[j]["summary"]["R2"]["counts"]["correct"]
                and per_fold[j]["summary"]["candidates"][candidate]["metrics"]["counts"]["tp"]
                >= per_fold[j]["summary"]["R2"]["counts"]["tp"]
                for j in INNER_FOLDS)
            advance = bool(
                merged["counts"]["correct"] >= r2_merged["counts"]["correct"]
                and merged["counts"]["tp"] > r2_merged["counts"]["tp"]
                and per_fold_ge >= 2)
            trade = _tradeoff(r2_merged, merged)
        else:
            trade = None
            advance = False
        report["candidates"][candidate] = {
            "per_fold": candidate_rows,
            "feasible_all_inner_folds": feasible_all,
            "merged": merged,
            "tradeoff_vs_R2": trade,
            "advances_to_O11": advance,
        }
        if advance:
            report["advanced"].append(candidate)
        if merged_frame is not None:
            object_name = "R2" if candidate == "A" else f"{candidate}_OPT"
            _write_parquet(configs["A"].project_root,
                           f"decisions/outer_{outer_fold}/{object_name}.parquet", merged_frame)
    report["advanced"] = sorted(report["advanced"])
    report["fallback_to_A"] = not bool(report["advanced"])
    report["recall_accuracy_tradeoffs"] = {
        candidate: report["candidates"][candidate]["tradeoff_vs_R2"]
        for candidate in ("B", "C", "D")}
    report_path = _write_json(configs["A"].project_root,
                              f"reports/outer_{outer_fold}.json", report)
    print(f"screening outer={outer_fold} advanced={report['advanced']} "
          f"fallback={report['fallback_to_A']}", flush=True)
    return {**report, "report": _relative(configs["A"].project_root, report_path)}


def run_outer(outer_fold: int, *, root: Path, device_name: str | None = None) -> dict:
    """Sequentially run the 12 jobs of one outer fold, select and report."""
    root = Path(root).resolve()
    if outer_fold not in OUTER_FOLDS:
        raise ValueError("outer_fold must be 0..4")
    configs = candidate_configs(root)
    store = SequenceStore(training_engine._legacy_read_config(configs["A"]))
    for inner_fold in INNER_FOLDS:
        for candidate in CANDIDATES:
            plan = training_engine.prepare_inner_run(configs[candidate], stage="screening",
                                                     outer_fold=outer_fold, inner_fold=inner_fold,
                                                     seed=SCREENING_SEED)
            ensure_screening_run(plan, store, device_name=device_name)
            export_predictions(configs[candidate], plan, store, outer_fold=outer_fold,
                               inner_fold=inner_fold, device_name=device_name)
        select_inner_fold(configs, outer_fold=outer_fold, inner_fold=inner_fold)
    return build_outer_report(configs, outer_fold)


def screening_manifest(*, root: Path, write: bool = True) -> dict:
    """Scan expected screening artifacts and report completion without training."""
    root = Path(root).resolve()
    missing_runs, missing_predictions, missing_selections, missing_reports = [], [], [], []
    run_summaries = {}
    for outer_fold in OUTER_FOLDS:
        for inner_fold in INNER_FOLDS:
            for candidate in CANDIDATES:
                config = load_optimization_config(root / f"configs/optimization/{candidate}.yaml")
                plan = training_engine.prepare_inner_run(config, stage="screening",
                                                         outer_fold=outer_fold,
                                                         inner_fold=inner_fold,
                                                         seed=SCREENING_SEED)
                run_path = plan.identity.artifact_path("run.json")
                prediction_rel = (f"evaluation/outer_{outer_fold}/inner_{inner_fold}/"
                                  f"{candidate}/seed_{SCREENING_SEED}/prediction_manifest.json")
                if not _root_path(root, prediction_rel).is_file():
                    missing_predictions.append(prediction_rel)
                selection_rel = f"thresholds/outer_{outer_fold}/inner_{inner_fold}/{candidate}.json"
                if not _root_path(root, selection_rel).is_file():
                    missing_selections.append(selection_rel)
                if not run_path.is_file():
                    missing_runs.append(_relative(root, run_path))
                    continue
                manifest = _read_json(run_path)
                if manifest.get("status") != "complete":
                    missing_runs.append(_relative(root, run_path))
                    continue
                run_summaries[f"outer_{outer_fold}/inner_{inner_fold}/{candidate}"] = {
                    "run_id": manifest["binding"]["run_id"], "epochs": manifest["epoch"],
                    "best_epoch": manifest["best_epoch"], "status": manifest["status"]}
        report_rel = f"reports/outer_{outer_fold}.json"
        if not _root_path(root, report_rel).is_file():
            missing_reports.append(report_rel)
    expected_runs = len(OUTER_FOLDS) * len(INNER_FOLDS) * len(CANDIDATES)
    complete = not (missing_runs or missing_predictions or missing_selections or missing_reports)
    record = {
        "schema_version": 1, "task": "O10", "status": "pass" if complete else "incomplete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": expected_runs,
        "completed_runs": expected_runs - len(missing_runs),
        "completed_predictions": expected_runs - len(missing_predictions),
        "completed_selections": expected_runs - len(missing_selections),
        "completed_outer_reports": len(OUTER_FOLDS) - len(missing_reports),
        "missing_runs": missing_runs, "missing_predictions": missing_predictions,
        "missing_selections": missing_selections, "missing_reports": missing_reports,
        "runs": run_summaries, "outer_test_labels_accessed": False,
    }
    if write:
        path = _root_path(root, "screening_manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes((canonical_json(record) + "\n").encode("utf-8"))
        os.replace(temporary, path)
    return record


def run_screening(*, root: Path, device_name: str | None = None,
                  outer_folds: Sequence[int] = OUTER_FOLDS) -> dict:
    """Sequential fallback driver; production throughput uses one process per outer fold."""
    root = Path(root).resolve()
    folds = tuple(outer_folds)
    if any(fold not in OUTER_FOLDS for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError("outer_folds must be unique values in 0..4")
    for outer_fold in folds:
        run_outer(outer_fold, root=root, device_name=device_name)
    return screening_manifest(root=root)
