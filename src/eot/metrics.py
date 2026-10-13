"""Leakage-safe classification and eRisk early-detection metrics.

The ERDE implementation follows the official eRisk 2017 definition.  The
latency-weighted F score follows the official eRisk 2021 overview, including
the published latency parameter p=0.0078.  Positions are one-based counts of
observed writings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ERISK_LATENCY_P = 0.0078
DEFAULT_THRESHOLD_GRID = np.linspace(0.0, 1.0, 201, dtype=np.float64)


@dataclass(frozen=True)
class RiskSequence:
    """One user's ordered prefix probabilities and binary ground truth."""

    user_id: str
    label: int
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float64)
        if values.ndim != 1 or values.size < 1:
            raise ValueError("probabilities must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("probabilities must be finite and lie in [0, 1]")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        object.__setattr__(self, "probabilities", values)


def first_alert_position(
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    consecutive: int = 1,
) -> int | None:
    """Return the first one-based position completing a consecutive alert run."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if consecutive < 1:
        raise ValueError("consecutive must be at least 1")
    run = 0
    for position, probability in enumerate(values, start=1):
        run = run + 1 if probability >= threshold else 0
        if run >= consecutive:
            return position
    return None


def alert_positions(
    sequences: Sequence[RiskSequence],
    thresholds: float | Sequence[float] | np.ndarray,
    consecutive: int = 1,
) -> np.ndarray:
    """Compute one-based alert positions, using NaN for users never alerted."""

    if np.isscalar(thresholds):
        threshold_values = np.full(len(sequences), float(thresholds), dtype=np.float64)
    else:
        threshold_values = np.asarray(thresholds, dtype=np.float64)
        if threshold_values.shape != (len(sequences),):
            raise ValueError("thresholds must have one value per sequence")
    positions = np.full(len(sequences), np.nan, dtype=np.float64)
    for index, (sequence, threshold) in enumerate(zip(sequences, threshold_values, strict=True)):
        position = first_alert_position(sequence.probabilities, threshold, consecutive)
        if position is not None:
            positions[index] = float(position)
    return positions


def classification_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float | None = None,
    predictions: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate configured final-sequence classification metrics."""

    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_score.shape != y_true.shape or y_true.size == 0:
        raise ValueError("labels and probabilities must be non-empty aligned vectors")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(y_score)) or np.any(y_score < 0.0) or np.any(y_score > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if predictions is None:
        if threshold is None:
            raise ValueError("provide threshold or explicit predictions")
        y_pred = (y_score >= float(threshold)).astype(np.int8)
    else:
        y_pred = np.asarray(predictions, dtype=np.int8)
        if y_pred.shape != y_true.shape or not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("predictions must be an aligned binary vector")
    has_both_classes = np.unique(y_true).size == 2
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred))
            if has_both_classes
            else float("nan")
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_both_classes else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def erde_latency_cost(position: float | np.ndarray, cutoff: int) -> float | np.ndarray:
    """Official lc_o(k) = 1 - 1 / (1 + exp(k-o)), evaluated stably."""

    difference = np.asarray(position, dtype=np.float64) - float(cutoff)
    result = 1.0 / (1.0 + np.exp(-np.clip(difference, -700.0, 700.0)))
    return float(result) if result.ndim == 0 else result


def latency_penalty(
    position: float | np.ndarray, p: float = ERISK_LATENCY_P
) -> float | np.ndarray:
    """Official penalty(k) with penalty(1)=0."""

    values = np.asarray(position, dtype=np.float64)
    result = np.tanh(float(p) * (values - 1.0) / 2.0)
    return float(result) if result.ndim == 0 else result


def early_detection_metrics(
    labels: Sequence[int] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    *,
    latency_p: float = ERISK_LATENCY_P,
) -> dict[str, float | None]:
    """Compute ERDE5/50 and latency-weighted F from frozen alert positions."""

    y_true = np.asarray(labels, dtype=np.int8)
    alert_at = np.asarray(positions, dtype=np.float64)
    if y_true.ndim != 1 or alert_at.shape != y_true.shape or y_true.size == 0:
        raise ValueError("labels and positions must be non-empty aligned vectors")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("labels must be binary")
    finite = np.isfinite(alert_at)
    if np.any(alert_at[finite] < 1.0):
        raise ValueError("finite alert positions must be one-based positive values")
    decisions = finite.astype(np.int8)
    positive_prevalence = float(np.mean(y_true))

    def erde(cutoff: int) -> float:
        costs = np.zeros(y_true.size, dtype=np.float64)
        false_positive = (y_true == 0) & finite
        false_negative = (y_true == 1) & ~finite
        true_positive = (y_true == 1) & finite
        costs[false_positive] = positive_prevalence
        costs[false_negative] = 1.0
        costs[true_positive] = np.asarray(
            erde_latency_cost(alert_at[true_positive], cutoff), dtype=np.float64
        )
        return float(np.mean(costs))

    precision = float(precision_score(y_true, decisions, zero_division=0))
    recall = float(recall_score(y_true, decisions, zero_division=0))
    early_f1 = float(f1_score(y_true, decisions, zero_division=0))
    true_positive_positions = alert_at[(y_true == 1) & finite]
    if true_positive_positions.size:
        median_latency = float(np.median(true_positive_positions))
        # The official definition takes the median of the per-user penalties.
        # For an even number of true positives this is not, in general, equal
        # to applying the nonlinear penalty after taking the median position.
        penalties = np.asarray(
            latency_penalty(true_positive_positions, latency_p), dtype=np.float64
        )
        speed = float(1.0 - np.median(penalties))
    else:
        median_latency = None
        speed = 0.0
    return {
        "erde_5": erde(5),
        "erde_50": erde(50),
        "median_latency": median_latency,
        "speed": speed,
        "f_latency": float(early_f1 * speed),
        "early_precision": precision,
        "early_recall": recall,
        "early_f1": early_f1,
    }


def evaluate_sequences(
    sequences: Sequence[RiskSequence],
    thresholds: float | Sequence[float] | np.ndarray,
    consecutive: int,
) -> tuple[np.ndarray, dict[str, float | None]]:
    """Apply an alert policy and evaluate it without accessing other data."""

    positions = alert_positions(sequences, thresholds, consecutive)
    labels = np.asarray([sequence.label for sequence in sequences], dtype=np.int8)
    return positions, early_detection_metrics(labels, positions)


def select_threshold(
    sequences: Sequence[RiskSequence],
    consecutive: int,
    candidates: Iterable[float] = DEFAULT_THRESHOLD_GRID,
) -> dict[str, float | int | None]:
    """Select a threshold by validation F_latency with deterministic tie-breaks."""

    if not sequences:
        raise ValueError("at least one validation sequence is required")
    candidate_values = np.unique(np.asarray(list(candidates), dtype=np.float64))
    if candidate_values.size == 0 or np.any(candidate_values < 0.0) or np.any(candidate_values > 1.0):
        raise ValueError("threshold candidates must lie in [0, 1]")
    best: dict[str, float | int | None] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for threshold in candidate_values:
        _, metrics = evaluate_sequences(sequences, float(threshold), consecutive)
        record: dict[str, float | int | None] = {
            "threshold": float(threshold),
            "consecutive": int(consecutive),
            **metrics,
        }
        key = (
            float(metrics["f_latency"]),
            float(metrics["early_f1"]),
            float(metrics["speed"]),
            -float(metrics["erde_50"]),
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = record
    if best is None:
        raise AssertionError("threshold selection produced no result")
    return best


def bootstrap_metric_intervals(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    *,
    iterations: int,
    level: float,
    seed: int,
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, float | int]]]:
    """User-level percentile bootstrap for all final and early metrics."""

    if iterations < 1 or not 0.0 < level < 1.0:
        raise ValueError("invalid bootstrap settings")
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(probabilities, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.int8)
    alert_at = np.asarray(positions, dtype=np.float64)
    if not (y_score.shape == y_pred.shape == alert_at.shape == y_true.shape):
        raise ValueError("bootstrap arrays must be aligned")
    generator = np.random.default_rng(seed)
    records: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        indices = generator.integers(0, y_true.size, size=y_true.size)
        classification = classification_metrics(
            y_true[indices], y_score[indices], predictions=y_pred[indices]
        )
        early = early_detection_metrics(y_true[indices], alert_at[indices])
        records.append(
            {
                "iteration": iteration,
                **classification,
                **{
                    name: float(value) if value is not None else float("nan")
                    for name, value in early.items()
                },
            }
        )
    alpha = (1.0 - level) / 2.0
    metric_names = [name for name in records[0] if name != "iteration"]
    intervals: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        valid = values[np.isfinite(values)]
        intervals[name] = {
            "lower": float(np.quantile(valid, alpha)) if valid.size else float("nan"),
            "upper": float(np.quantile(valid, 1.0 - alpha)) if valid.size else float("nan"),
            "valid_iterations": int(valid.size),
        }
    return intervals, records
