"""Leakage-safe user-level cross-validation splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ExperimentConfig
from .reproducibility import write_json


def _class_counts(user_ids: list[str], labels: dict[str, int]) -> dict[str, int]:
    counts = Counter(labels[user_id] for user_id in user_ids)
    return {"control_0": counts[0], "depression_1": counts[1]}


def _validate_disjoint(train: list[str], validation: list[str], test: list[str]) -> None:
    train_set, validation_set, test_set = set(train), set(validation), set(test)
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise AssertionError("train, validation and test user IDs must be pairwise disjoint")
    if len(train_set) != len(train) or len(validation_set) != len(validation) or len(test_set) != len(test):
        raise AssertionError("A split contains duplicate user IDs")


def create_user_splits(config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create fixed outer folds and a stratified validation split within each train fold."""

    users_path = config.path("processed_dir") / "user_sequences.parquet"
    users = pd.read_parquet(users_path, columns=["user_id", "label"]).sort_values("user_id")
    if users["user_id"].duplicated().any():
        raise ValueError("user_sequences.parquet contains duplicate users")
    user_ids = users["user_id"].astype(str).to_numpy()
    y = users["label"].astype(int).to_numpy()
    cv_cfg = config.section("cross_validation")
    n_splits = int(cv_cfg["n_splits"])
    fold_seed = int(cv_cfg["fold_seed"])
    validation_fraction = float(cv_cfg["validation_fraction_of_outer_train"])
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(cv_cfg["shuffle"]),
        random_state=fold_seed,
    )
    label_by_user = dict(zip(user_ids.tolist(), y.tolist(), strict=True))
    folds: list[dict[str, Any]] = []
    test_coverage: Counter[str] = Counter()
    for fold, (outer_train_indices, test_indices) in enumerate(splitter.split(user_ids, y)):
        outer_users = user_ids[outer_train_indices]
        outer_labels = y[outer_train_indices]
        train_users, validation_users = train_test_split(
            outer_users,
            test_size=validation_fraction,
            random_state=fold_seed + fold,
            shuffle=True,
            stratify=outer_labels,
        )
        train = sorted(train_users.tolist())
        validation = sorted(validation_users.tolist())
        test = sorted(user_ids[test_indices].tolist())
        _validate_disjoint(train, validation, test)
        if set(train) | set(validation) | set(test) != set(user_ids.tolist()):
            raise AssertionError(f"Fold {fold} does not cover every user exactly once")
        test_coverage.update(test)
        folds.append(
            {
                "fold": fold,
                "config_hash": config.config_hash,
                "strategy": "stratified_user_kfold_with_inner_validation",
                "fold_seed": fold_seed,
                "inner_validation_seed": fold_seed + fold,
                "validation_fraction_of_outer_train": validation_fraction,
                "train_user_ids": train,
                "validation_user_ids": validation,
                "test_user_ids": test,
                "counts": {
                    "train": {"users": len(train), **_class_counts(train, label_by_user)},
                    "validation": {
                        "users": len(validation),
                        **_class_counts(validation, label_by_user),
                    },
                    "test": {"users": len(test), **_class_counts(test, label_by_user)},
                },
            }
        )
    if set(test_coverage) != set(user_ids.tolist()) or any(count != 1 for count in test_coverage.values()):
        raise AssertionError("Outer test folds must cover every user exactly once")
    report = {
        "status": "pass",
        "config_hash": config.config_hash,
        "strategy": "stratified user-level 5-fold CV with inner validation",
        "n_folds": n_splits,
        "shared_across_training_seeds": list(config.section("training")["seeds"]),
        "all_sets_pairwise_disjoint": True,
        "every_user_in_one_outer_test_fold": True,
        "fold_counts": [{"fold": item["fold"], **item["counts"]} for item in folds],
    }
    return folds, report


def write_user_splits(
    folds: list[dict[str, Any]], report: dict[str, Any], config: ExperimentConfig
) -> None:
    splits_dir = config.path("splits_dir")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        write_json(fold, splits_dir / f"fold_{fold['fold']}.json")
    write_json(report, config.path("outputs_dir") / "reports" / "split_report.json")
