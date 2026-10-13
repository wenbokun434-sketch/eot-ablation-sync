"""Formal fold/seed inference, threshold locking, OOF assembly, and evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import SequenceStore, UserSequenceDataset
from .metrics import (
    RiskSequence,
    alert_positions,
    bootstrap_metric_intervals,
    classification_metrics,
    early_detection_metrics,
    select_threshold,
)
from .model import EOTModel
from .reproducibility import set_global_seed, write_json
from .time_features import load_time_normalizer


METRIC_SOURCE_URLS = {
    "erde": "https://tec.citius.usc.es/ir/pdf/eRisk2017.pdf",
    "f_latency": "https://www.ceur-ws.org/Vol-2936/paper-72.pdf",
}


def _id_hash(user_ids: Sequence[str]) -> str:
    stable = "\n".join(sorted(str(value) for value in user_ids))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(config: ExperimentConfig, fold: int) -> dict[str, Any]:
    path = config.path("splits_dir") / f"fold_{fold}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config.config_hash or int(payload.get("fold", -1)) != fold:
        raise ValueError(f"Split metadata does not match current configuration: {path}")
    return payload


@torch.inference_mode()
def predict_dataset(
    model: EOTModel,
    dataset: UserSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return user-level final probabilities and every valid prefix probability."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    final_frames: list[pd.DataFrame] = []
    prefix_frames: list[pd.DataFrame] = []
    for batch in loader:
        user_ids = [str(value) for value in batch["user_id"]]
        embeddings = batch["post_embeddings"].to(
            device, non_blocking=device.type == "cuda"
        )
        delta_t = batch["delta_t"].to(device, non_blocking=device.type == "cuda")
        mask = batch["mask"].to(device, non_blocking=device.type == "cuda")
        output = model(embeddings, delta_t, mask)
        final_probability = torch.sigmoid(output["final_logits"]).cpu().numpy()
        prefix_probability = torch.sigmoid(output["prefix_logits"]).cpu().numpy()
        labels = batch["label"].numpy().astype(np.int8)
        lengths = batch["length"].numpy().astype(np.int16)
        final_frames.append(
            pd.DataFrame(
                {
                    "user_id": user_ids,
                    "label": labels,
                    "length": lengths,
                    "probability": final_probability.astype(np.float32),
                }
            )
        )
        for index, user_id in enumerate(user_ids):
            length = int(lengths[index])
            prefix_frames.append(
                pd.DataFrame(
                    {
                        "user_id": user_id,
                        "label": int(labels[index]),
                        "length": length,
                        "position": np.arange(1, length + 1, dtype=np.int16),
                        "probability": prefix_probability[index, :length].astype(np.float32),
                    }
                )
            )
    final = pd.concat(final_frames, ignore_index=True)
    prefix = pd.concat(prefix_frames, ignore_index=True)
    if final["user_id"].duplicated().any():
        raise AssertionError("inference produced duplicate users")
    counts = prefix.groupby("user_id", sort=False).size()
    expected = final.set_index("user_id")["length"].astype(int)
    if not counts.reindex(expected.index).equals(expected):
        raise AssertionError("prefix row counts do not match sequence lengths")
    return final, prefix


def _ensemble_predictions(
    final_frames: Sequence[pd.DataFrame],
    prefix_frames: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not final_frames or len(final_frames) != len(prefix_frames):
        raise ValueError("aligned seed prediction frames are required")
    seed_count = len(final_frames)
    final_all = pd.concat(final_frames, ignore_index=True)
    final_keys = ["user_id", "label", "length"]
    final = (
        final_all.groupby(final_keys, as_index=False, sort=True)
        .agg(probability=("probability", "mean"), seed_count=("probability", "size"))
    )
    if not (final["seed_count"] == seed_count).all():
        raise AssertionError("seed-level final predictions are not aligned")
    final = final.drop(columns="seed_count")

    prefix_all = pd.concat(prefix_frames, ignore_index=True)
    prefix_keys = ["user_id", "label", "length", "position"]
    prefix = (
        prefix_all.groupby(prefix_keys, as_index=False, sort=True)
        .agg(probability=("probability", "mean"), seed_count=("probability", "size"))
    )
    if not (prefix["seed_count"] == seed_count).all():
        raise AssertionError("seed-level prefix predictions are not aligned")
    return final, prefix.drop(columns="seed_count")


def _risk_sequences(final: pd.DataFrame, prefix: pd.DataFrame) -> list[RiskSequence]:
    grouped = {str(key): frame for key, frame in prefix.groupby("user_id", sort=False)}
    sequences: list[RiskSequence] = []
    for row in final.itertuples(index=False):
        user_id = str(row.user_id)
        if user_id not in grouped:
            raise AssertionError(f"missing prefix predictions for {user_id}")
        frame = grouped[user_id].sort_values("position")
        expected = np.arange(1, int(row.length) + 1)
        if not np.array_equal(frame["position"].to_numpy(), expected):
            raise AssertionError(f"non-contiguous prefix positions for {user_id}")
        sequences.append(
            RiskSequence(
                user_id=user_id,
                label=int(row.label),
                probabilities=frame["probability"].to_numpy(dtype=np.float64),
            )
        )
    return sequences


def _annotate_decisions(
    final: pd.DataFrame,
    prefix: pd.DataFrame,
    *,
    threshold: float,
    consecutive: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = final.copy()
    prefix = prefix.copy()
    sequences = _risk_sequences(final, prefix)
    positions = alert_positions(sequences, threshold, consecutive)
    final["threshold"] = float(threshold)
    final["final_prediction"] = (final["probability"] >= threshold).astype(np.int8)
    final["alert_prediction"] = np.isfinite(positions).astype(np.int8)
    final["alert_position"] = positions
    prefix["threshold"] = float(threshold)
    prefix["above_threshold"] = (prefix["probability"] >= threshold).astype(np.int8)
    return final, prefix


def metrics_from_final_predictions(final: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Recompute all metrics from a saved, threshold-annotated final frame."""

    required = {
        "label",
        "probability",
        "final_prediction",
        "alert_prediction",
        "alert_position",
    }
    missing = sorted(required - set(final.columns))
    if missing:
        raise ValueError(f"final prediction frame is missing columns: {missing}")
    positions = final["alert_position"].to_numpy(dtype=np.float64)
    if not np.array_equal(np.isfinite(positions).astype(np.int8), final["alert_prediction"]):
        raise AssertionError("alert_prediction and alert_position disagree")
    labels = final["label"].to_numpy(dtype=np.int8)
    return {
        "classification": classification_metrics(
            labels,
            final["probability"].to_numpy(dtype=np.float64),
            predictions=final["final_prediction"].to_numpy(dtype=np.int8),
        ),
        "early_detection": early_detection_metrics(labels, positions),
    }


def _metric_row(
    final: pd.DataFrame,
    *,
    fold: int,
    split: str,
    threshold: float,
    seed: int | str,
) -> dict[str, float | int | str | None]:
    metrics = metrics_from_final_predictions(final)
    return {
        "fold": fold,
        "seed": seed,
        "split": split,
        "threshold": threshold,
        "users": len(final),
        **metrics["classification"],
        **metrics["early_detection"],
    }


def _write_predictions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _load_model(
    config: ExperimentConfig,
    fold: int,
    seed: int,
    device: torch.device,
) -> tuple[EOTModel, dict[str, Any]]:
    path = config.path("outputs_dir") / "checkpoints" / f"fold_{fold}_seed_{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing formal checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = (config.config_hash, fold, seed)
    observed = (
        checkpoint.get("config_hash"),
        int(checkpoint.get("fold", -1)),
        int(checkpoint.get("seed", -1)),
    )
    if observed != expected:
        raise ValueError(f"checkpoint identity mismatch for {path}: {observed} != {expected}")
    model = EOTModel.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def _summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "brier_score",
        "erde_5",
        "erde_50",
        "median_latency",
        "speed",
        "f_latency",
        "early_precision",
        "early_recall",
        "early_f1",
    ]
    rows: list[dict[str, float | int | str]] = []
    for split, frame in fold_metrics.groupby("split", sort=False):
        for name in metric_names:
            values = pd.to_numeric(frame[name], errors="coerce")
            rows.append(
                {
                    "split": str(split),
                    "metric": name,
                    "folds": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def run_formal_evaluation(
    config: ExperimentConfig,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate 15 checkpoints and build five-fold seed-ensemble OOF predictions."""

    started_at = datetime.now(timezone.utc).isoformat()
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = config.path("outputs_dir")
    prediction_root = output_root / "predictions"
    metric_root = output_root / "metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    seed_prediction_root = prediction_root / "seed_level"
    fold_prediction_root = prediction_root / "fold_ensembles"
    seeds = [int(value) for value in config.section("training")["seeds"]]
    folds = range(int(config.section("cross_validation")["n_splits"]))
    batch_size = int(config.section("training")["batch_users"])
    consecutive = int(config.section("decision")["consecutive_positive_predictions"])
    if str(config.section("decision")["threshold_source"]) != "validation":
        raise ValueError("formal evaluation requires decision.threshold_source=validation")
    if str(config.section("decision")["threshold_objective"]) != "f_latency":
        raise ValueError("formal evaluation currently implements f_latency threshold selection")

    store = SequenceStore(config)
    seed_metric_rows: list[dict[str, float | int | str | None]] = []
    fold_metric_rows: list[dict[str, float | int | str | None]] = []
    threshold_records: list[dict[str, Any]] = []
    oof_final_frames: list[pd.DataFrame] = []
    oof_prefix_frames: list[pd.DataFrame] = []
    expected_test_ids: set[str] = set()
    checkpoint_records: list[dict[str, Any]] = []

    for fold in folds:
        split = _load_split(config, fold)
        validation_ids = [str(value) for value in split["validation_user_ids"]]
        test_ids = [str(value) for value in split["test_user_ids"]]
        if set(validation_ids) & set(test_ids):
            raise AssertionError(f"validation/test overlap in fold {fold}")
        expected_test_ids.update(test_ids)
        normalizer = load_time_normalizer(config, fold)
        validation_dataset = UserSequenceDataset(store, validation_ids, normalizer)
        test_dataset = UserSequenceDataset(store, test_ids, normalizer)
        seed_predictions: dict[int, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}

        for seed in seeds:
            set_global_seed(seed)
            model, checkpoint = _load_model(config, fold, seed, device)
            validation_final, validation_prefix = predict_dataset(
                model, validation_dataset, device=device, batch_size=batch_size
            )
            test_final, test_prefix = predict_dataset(
                model, test_dataset, device=device, batch_size=batch_size
            )
            seed_predictions[seed] = {
                "validation": (validation_final, validation_prefix),
                "test": (test_final, test_prefix),
            }
            checkpoint_path = (
                output_root / "checkpoints" / f"fold_{fold}_seed_{seed}" / "best.pt"
            )
            checkpoint_records.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": int(checkpoint["epoch"]),
                    "validation_pr_auc_at_training": float(checkpoint["validation_pr_auc"]),
                    "path": str(checkpoint_path.relative_to(config.project_root)),
                    "sha256": _file_sha256(checkpoint_path),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Every seed receives its own validation-only threshold for stability reporting.
        for seed in seeds:
            validation_final, validation_prefix = seed_predictions[seed]["validation"]
            test_final, test_prefix = seed_predictions[seed]["test"]
            selected = select_threshold(
                _risk_sequences(validation_final, validation_prefix), consecutive
            )
            threshold = float(selected["threshold"])
            validation_final, validation_prefix = _annotate_decisions(
                validation_final,
                validation_prefix,
                threshold=threshold,
                consecutive=consecutive,
            )
            test_final, test_prefix = _annotate_decisions(
                test_final,
                test_prefix,
                threshold=threshold,
                consecutive=consecutive,
            )
            for frame in (validation_final, validation_prefix, test_final, test_prefix):
                frame["fold"] = fold
                frame["seed"] = seed
            _write_predictions(
                validation_final,
                seed_prediction_root / f"fold_{fold}_seed_{seed}_validation_final.parquet",
            )
            _write_predictions(
                validation_prefix,
                seed_prediction_root / f"fold_{fold}_seed_{seed}_validation_prefix.parquet",
            )
            _write_predictions(
                test_final,
                seed_prediction_root / f"fold_{fold}_seed_{seed}_test_final.parquet",
            )
            _write_predictions(
                test_prefix,
                seed_prediction_root / f"fold_{fold}_seed_{seed}_test_prefix.parquet",
            )
            seed_metric_rows.extend(
                [
                    _metric_row(
                        validation_final,
                        fold=fold,
                        seed=seed,
                        split="validation",
                        threshold=threshold,
                    ),
                    _metric_row(
                        test_final,
                        fold=fold,
                        seed=seed,
                        split="test",
                        threshold=threshold,
                    ),
                ]
            )

        # Primary result: mean probabilities across the three seeds within each fold.
        validation_final, validation_prefix = _ensemble_predictions(
            [seed_predictions[seed]["validation"][0] for seed in seeds],
            [seed_predictions[seed]["validation"][1] for seed in seeds],
        )
        test_final, test_prefix = _ensemble_predictions(
            [seed_predictions[seed]["test"][0] for seed in seeds],
            [seed_predictions[seed]["test"][1] for seed in seeds],
        )
        selected = select_threshold(_risk_sequences(validation_final, validation_prefix), consecutive)
        threshold = float(selected["threshold"])
        validation_final, validation_prefix = _annotate_decisions(
            validation_final,
            validation_prefix,
            threshold=threshold,
            consecutive=consecutive,
        )
        test_final, test_prefix = _annotate_decisions(
            test_final,
            test_prefix,
            threshold=threshold,
            consecutive=consecutive,
        )
        for frame in (validation_final, validation_prefix, test_final, test_prefix):
            frame["fold"] = fold
            frame["seed_aggregation"] = "mean_probability"
        _write_predictions(
            validation_final,
            fold_prediction_root / f"fold_{fold}_validation_final.parquet",
        )
        _write_predictions(
            validation_prefix,
            fold_prediction_root / f"fold_{fold}_validation_prefix.parquet",
        )
        _write_predictions(
            test_final,
            fold_prediction_root / f"fold_{fold}_test_final.parquet",
        )
        _write_predictions(
            test_prefix,
            fold_prediction_root / f"fold_{fold}_test_prefix.parquet",
        )
        fold_metric_rows.extend(
            [
                _metric_row(
                    validation_final,
                    fold=fold,
                    seed="ensemble",
                    split="validation",
                    threshold=threshold,
                ),
                _metric_row(
                    test_final,
                    fold=fold,
                    seed="ensemble",
                    split="test",
                    threshold=threshold,
                ),
            ]
        )
        threshold_records.append(
            {
                "fold": fold,
                "threshold": threshold,
                "consecutive_positive_predictions": consecutive,
                "objective": "f_latency",
                "source": "validation_seed_ensemble_only",
                "validation_user_count": len(validation_ids),
                "validation_user_ids_sha256": _id_hash(validation_ids),
                "test_user_count": len(test_ids),
                "test_user_ids_sha256_for_audit_only": _id_hash(test_ids),
                "validation_objective_metrics": selected,
            }
        )
        oof_final_frames.append(test_final)
        oof_prefix_frames.append(test_prefix)
        print(
            f"fold={fold} threshold={threshold:.3f} "
            f"validation_f_latency={float(selected['f_latency']):.6f} test_users={len(test_final)}",
            flush=True,
        )

    seed_metrics = pd.DataFrame(seed_metric_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    seed_metrics.to_csv(metric_root / "seed_run_metrics.csv", index=False)
    fold_metrics.to_csv(metric_root / "fold_metrics.csv", index=False)
    fold_summary = _summarize_fold_metrics(fold_metrics)
    fold_summary.to_csv(metric_root / "fold_metric_summary.csv", index=False)
    write_json(
        {
            "config_hash": config.config_hash,
            "policy": "thresholds selected exclusively on each fold's validation seed ensemble",
            "threshold_grid": {"minimum": 0.0, "maximum": 1.0, "step": 0.005, "count": 201},
            "tie_break_order": [
                "higher_f_latency",
                "higher_early_f1",
                "higher_speed",
                "lower_erde_50",
                "higher_threshold",
            ],
            "folds": threshold_records,
        },
        metric_root / "fold_thresholds.json",
    )

    oof_final = pd.concat(oof_final_frames, ignore_index=True).sort_values("user_id")
    oof_prefix = pd.concat(oof_prefix_frames, ignore_index=True).sort_values(
        ["user_id", "position"]
    )
    if oof_final["user_id"].duplicated().any():
        raise AssertionError("OOF final predictions contain duplicate users")
    actual_oof_ids = set(oof_final["user_id"].astype(str))
    if actual_oof_ids != expected_test_ids or actual_oof_ids != set(store.users):
        raise AssertionError("OOF coverage does not match the complete user cohort")
    prefix_ids = set(oof_prefix["user_id"].astype(str))
    if prefix_ids != actual_oof_ids:
        raise AssertionError("OOF prefix and final user coverage differ")
    oof_final_path = prediction_root / "oof_final_predictions.parquet"
    oof_prefix_path = prediction_root / "oof_prefix_predictions.parquet"
    _write_predictions(oof_final, oof_final_path)
    _write_predictions(oof_prefix, oof_prefix_path)

    point = metrics_from_final_predictions(oof_final)
    confidence_cfg = config.section("metrics")["confidence_interval"]
    if str(confidence_cfg["method"]) != "user_bootstrap":
        raise ValueError("formal evaluation requires user_bootstrap confidence intervals")
    intervals, bootstrap_records = bootstrap_metric_intervals(
        oof_final["label"].to_numpy(dtype=np.int8),
        oof_final["probability"].to_numpy(dtype=np.float64),
        oof_final["final_prediction"].to_numpy(dtype=np.int8),
        oof_final["alert_position"].to_numpy(dtype=np.float64),
        iterations=int(confidence_cfg["iterations"]),
        level=float(confidence_cfg["level"]),
        seed=int(config.section("cross_validation")["fold_seed"]),
    )
    pd.DataFrame(bootstrap_records).to_csv(metric_root / "bootstrap_metrics.csv", index=False)
    oof_metrics = {
        "config_hash": config.config_hash,
        "users": len(oof_final),
        "positive_users": int(oof_final["label"].sum()),
        "negative_users": int((1 - oof_final["label"]).sum()),
        "primary_estimate": "five outer test folds; mean seed probability within fold",
        "classification_decision": "full-sequence probability compared with fold validation threshold",
        "early_decision": f"first position completing {consecutive} consecutive prefix probabilities at or above the fold validation threshold",
        "point_estimate": point,
        "confidence_interval": {
            "method": "user_percentile_bootstrap",
            "level": float(confidence_cfg["level"]),
            "iterations": int(confidence_cfg["iterations"]),
            "seed": int(config.section("cross_validation")["fold_seed"]),
            "intervals": intervals,
        },
        "metric_sources": METRIC_SOURCE_URLS,
    }
    write_json(oof_metrics, metric_root / "oof_metrics.json")

    manifest = {
        "status": "pass",
        "config_hash": config.config_hash,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "folds": int(config.section("cross_validation")["n_splits"]),
        "seeds": seeds,
        "checkpoints_validated": len(checkpoint_records),
        "checkpoint_records": checkpoint_records,
        "threshold_source": "validation_seed_ensemble_only",
        "test_labels_used_for_selection": False,
        "seed_aggregation": "arithmetic_mean_of_probabilities_within_fold",
        "oof_users": len(oof_final),
        "oof_unique_users": int(oof_final["user_id"].nunique()),
        "oof_complete_cohort": actual_oof_ids == set(store.users),
        "oof_final_predictions": str(oof_final_path.relative_to(config.project_root)),
        "oof_prefix_predictions": str(oof_prefix_path.relative_to(config.project_root)),
        "oof_final_sha256": _file_sha256(oof_final_path),
        "oof_prefix_sha256": _file_sha256(oof_prefix_path),
        "metric_sources": METRIC_SOURCE_URLS,
        "artifacts": {
            "fold_thresholds": "outputs/metrics/fold_thresholds.json",
            "seed_run_metrics": "outputs/metrics/seed_run_metrics.csv",
            "fold_metrics": "outputs/metrics/fold_metrics.csv",
            "fold_metric_summary": "outputs/metrics/fold_metric_summary.csv",
            "bootstrap_metrics": "outputs/metrics/bootstrap_metrics.csv",
            "oof_metrics": "outputs/metrics/oof_metrics.json",
        },
    }
    write_json(manifest, output_root / "reports" / "evaluation_manifest.json")
    return manifest


def validate_evaluation_artifacts(config: ExperimentConfig) -> dict[str, Any]:
    """Independently validate threshold provenance, OOF coverage, and metric replay."""

    output_root = config.path("outputs_dir")
    prediction_root = output_root / "predictions"
    metric_root = output_root / "metrics"
    manifest_path = output_root / "reports" / "evaluation_manifest.json"
    thresholds_path = metric_root / "fold_thresholds.json"
    oof_metrics_path = metric_root / "oof_metrics.json"
    required = [manifest_path, thresholds_path, oof_metrics_path]
    required.extend(
        [
            prediction_root / "oof_final_predictions.parquet",
            prediction_root / "oof_prefix_predictions.parquet",
        ]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evaluation artifacts: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    saved_metrics = json.loads(oof_metrics_path.read_text(encoding="utf-8"))
    if not all(
        payload.get("config_hash") == config.config_hash
        for payload in (manifest, threshold_payload, saved_metrics)
    ):
        raise AssertionError("evaluation artifact config hashes do not match")

    consecutive = int(config.section("decision")["consecutive_positive_predictions"])
    threshold_replay: list[dict[str, float | int | bool]] = []
    all_test_ids: set[str] = set()
    for record in threshold_payload["folds"]:
        fold = int(record["fold"])
        split = _load_split(config, fold)
        validation_final = pd.read_parquet(
            prediction_root / "fold_ensembles" / f"fold_{fold}_validation_final.parquet"
        )
        validation_prefix = pd.read_parquet(
            prediction_root / "fold_ensembles" / f"fold_{fold}_validation_prefix.parquet"
        )
        test_final = pd.read_parquet(
            prediction_root / "fold_ensembles" / f"fold_{fold}_test_final.parquet"
        )
        validation_ids = set(str(value) for value in split["validation_user_ids"])
        test_ids = set(str(value) for value in split["test_user_ids"])
        if set(validation_final["user_id"].astype(str)) != validation_ids:
            raise AssertionError(f"fold {fold} validation prediction membership mismatch")
        if set(test_final["user_id"].astype(str)) != test_ids:
            raise AssertionError(f"fold {fold} test prediction membership mismatch")
        if validation_ids & test_ids:
            raise AssertionError(f"fold {fold} validation/test membership overlap")
        replay = select_threshold(
            _risk_sequences(validation_final, validation_prefix), consecutive
        )
        locked = float(record["threshold"])
        replayed = float(replay["threshold"])
        threshold_replay.append(
            {
                "fold": fold,
                "locked": locked,
                "replayed": replayed,
                "exact_match": locked == replayed,
            }
        )
        all_test_ids.update(test_ids)
    if not all(bool(record["exact_match"]) for record in threshold_replay):
        raise AssertionError("one or more locked thresholds cannot be replayed from validation data")

    oof_final_path = prediction_root / "oof_final_predictions.parquet"
    oof_prefix_path = prediction_root / "oof_prefix_predictions.parquet"
    oof_final = pd.read_parquet(oof_final_path)
    oof_prefix = pd.read_parquet(oof_prefix_path)
    oof_ids = set(oof_final["user_id"].astype(str))
    if oof_final["user_id"].duplicated().any() or oof_ids != all_test_ids:
        raise AssertionError("OOF users are duplicated or do not equal the union of outer test folds")
    prefix_counts = oof_prefix.groupby("user_id").size().sort_index()
    expected_counts = oof_final.set_index("user_id")["length"].astype(int).sort_index()
    if not prefix_counts.equals(expected_counts):
        raise AssertionError("OOF prefix counts do not equal saved sequence lengths")
    if not (
        np.isfinite(oof_final["probability"]).all()
        and np.isfinite(oof_prefix["probability"]).all()
        and oof_final["probability"].between(0.0, 1.0).all()
        and oof_prefix["probability"].between(0.0, 1.0).all()
    ):
        raise AssertionError("OOF probabilities are non-finite or outside [0, 1]")
    last_prefix = (
        oof_prefix.sort_values(["user_id", "position"])
        .groupby("user_id")
        .tail(1)
        .set_index("user_id")["probability"]
        .sort_index()
    )
    final_probability = oof_final.set_index("user_id")["probability"].sort_index()
    final_prefix_max_abs_difference = float(
        np.max(np.abs(final_probability.to_numpy() - last_prefix.to_numpy()))
    )
    if final_prefix_max_abs_difference > 1e-7:
        raise AssertionError("final probabilities do not match the last valid prefixes")

    replayed_metrics = metrics_from_final_predictions(oof_final)
    metric_differences: dict[str, float] = {}
    for section, values in replayed_metrics.items():
        for name, value in values.items():
            saved_value = saved_metrics["point_estimate"][section][name]
            difference = 0.0 if value is None and saved_value is None else abs(float(value) - float(saved_value))
            metric_differences[f"{section}.{name}"] = difference
    maximum_metric_difference = max(metric_differences.values(), default=0.0)
    if maximum_metric_difference > 1e-12:
        raise AssertionError("saved point metrics cannot be exactly replayed from OOF predictions")
    if _file_sha256(oof_final_path) != manifest["oof_final_sha256"]:
        raise AssertionError("OOF final prediction checksum mismatch")
    if _file_sha256(oof_prefix_path) != manifest["oof_prefix_sha256"]:
        raise AssertionError("OOF prefix prediction checksum mismatch")

    seed_files = list((prediction_root / "seed_level").glob("*.parquet"))
    fold_files = list((prediction_root / "fold_ensembles").glob("*.parquet"))
    expected_seed_files = (
        int(config.section("cross_validation")["n_splits"])
        * len(config.section("training")["seeds"])
        * 4
    )
    expected_fold_files = int(config.section("cross_validation")["n_splits"]) * 4
    if len(seed_files) != expected_seed_files or len(fold_files) != expected_fold_files:
        raise AssertionError("seed-level or fold-ensemble prediction file count mismatch")

    report = {
        "status": "pass",
        "config_hash": config.config_hash,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_replayed_from_validation": threshold_replay,
        "test_labels_used_for_threshold_selection": False,
        "oof_users": len(oof_final),
        "oof_unique_users": int(oof_final["user_id"].nunique()),
        "oof_prefix_rows": len(oof_prefix),
        "expected_oof_prefix_rows": int(oof_final["length"].sum()),
        "final_prefix_max_abs_difference": final_prefix_max_abs_difference,
        "maximum_recomputed_metric_abs_difference": maximum_metric_difference,
        "probabilities_finite_and_bounded": True,
        "seed_prediction_files": len(seed_files),
        "fold_ensemble_prediction_files": len(fold_files),
        "checksums_match_manifest": True,
    }
    write_json(report, output_root / "reports" / "evaluation_validation.json")
    return report
