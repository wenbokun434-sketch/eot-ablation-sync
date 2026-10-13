"""Fold-local time normalization without validation/test leakage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .reproducibility import write_json


def _user_set_hash(user_ids: Iterable[str]) -> str:
    stable = "\n".join(sorted(str(value) for value in user_ids)) + "\n"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TimeNormalizer:
    quantile: float
    log_gap_quantile: float

    def transform(self, delta_days: np.ndarray) -> np.ndarray:
        values = np.asarray(delta_days, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Time gaps must be finite and non-negative.")
        transformed = np.log1p(values)
        clipped = np.clip(transformed, 0.0, self.log_gap_quantile)
        return (clipped / self.log_gap_quantile).astype(np.float32)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "raw_unit": "days",
            "transform": "log1p",
            "clipping_quantile": self.quantile,
            "log_gap_quantile": self.log_gap_quantile,
            "output_min": 0.0,
            "output_max": 1.0,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TimeNormalizer":
        return cls(
            quantile=float(payload["clipping_quantile"]),
            log_gap_quantile=float(payload["log_gap_quantile"]),
        )


def fit_time_normalizer(
    posts: pd.DataFrame,
    train_user_ids: Iterable[str],
    quantile: float,
) -> tuple[TimeNormalizer, dict[str, Any]]:
    """Fit on non-initial gaps of training users only."""

    train_users = sorted(str(value) for value in train_user_ids)
    train_set = set(train_users)
    selected = posts.loc[
        posts["user_id"].isin(train_set) & posts["sequence_index"].gt(0), "delta_days"
    ].to_numpy(dtype=np.float64)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1].")
    if selected.size == 0 or np.any(~np.isfinite(selected)) or np.any(selected < 0):
        raise ValueError("Training fold has no valid non-negative time gaps.")
    log_values = np.log1p(selected)
    scale = float(np.quantile(log_values, quantile))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    normalizer = TimeNormalizer(quantile=quantile, log_gap_quantile=scale)
    details = {
        "fit_user_count": len(train_users),
        "fit_user_ids_sha256": _user_set_hash(train_users),
        "fit_non_initial_gap_count": int(selected.size),
        "raw_gap_days_min": float(selected.min()),
        "raw_gap_days_median": float(np.median(selected)),
        "raw_gap_days_max": float(selected.max()),
        "transformed_values_clipped": int((log_values > scale).sum()),
    }
    return normalizer, details


def create_all_time_stats(config: ExperimentConfig) -> list[dict[str, Any]]:
    posts = pd.read_parquet(
        config.path("processed_dir") / "post_index.parquet",
        columns=["user_id", "sequence_index", "delta_days"],
    )
    time_cfg = config.section("time")
    quantile = float(time_cfg["clipping_quantile"])
    output_dir = config.path("time_stats_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    n_folds = int(config.section("cross_validation")["n_splits"])
    for fold in range(n_folds):
        split_path = config.path("splits_dir") / f"fold_{fold}.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        normalizer, details = fit_time_normalizer(posts, split["train_user_ids"], quantile)
        record = {
            "fold": fold,
            "config_hash": config.config_hash,
            "fit_split": "train",
            **normalizer.to_dict(),
            **details,
        }
        write_json(record, output_dir / f"fold_{fold}.json")
        records.append(record)
    write_json(
        {
            "status": "pass",
            "config_hash": config.config_hash,
            "folds": records,
            "validation_and_test_reuse_train_statistics": True,
        },
        config.path("outputs_dir") / "reports" / "time_normalization_report.json",
    )
    return records


def load_time_normalizer(config: ExperimentConfig, fold: int) -> TimeNormalizer:
    path = config.path("time_stats_dir") / f"fold_{fold}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config.config_hash:
        raise ValueError(f"Stale time statistics for fold {fold}")
    return TimeNormalizer.from_dict(payload)
