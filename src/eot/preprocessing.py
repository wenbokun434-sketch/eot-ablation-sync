"""Independent construction of EOT user sequences from raw eRisk JSON."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .raw_data import build_event_text, iter_raw_events, parse_created_utc, read_labels


POST_COLUMNS = [
    "embedding_index",
    "user_id",
    "label",
    "sequence_index",
    "post_id",
    "kind",
    "created_utc",
    "timestamp_unix",
    "delta_seconds",
    "delta_days",
    "text",
    "text_sha256",
    "source_file",
    "submission_id",
    "parent_id",
]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preprocess_dataset(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build earliest target-only sequences with traceable post metadata."""

    root = config.path("raw_data_root")
    paths = config.section("paths")
    data_cfg = config.section("data")
    labels_path = root / str(paths["labels_file"])
    conversations_path = root / str(paths["conversations_dir"])
    labels, label_errors = read_labels(labels_path)
    if label_errors:
        raise ValueError(f"Malformed label file: {label_errors[:3]}")
    if len({record.user_id for record in labels}) != len(labels):
        raise ValueError("Duplicate user IDs in the label file.")
    label_by_user = {record.user_id: record.label for record in labels}
    max_posts = int(data_cfg["max_posts"])
    remove_markers = {str(item).lower() for item in data_cfg["remove_markers"]}
    normalize_urls = bool(data_cfg["normalize_urls"])
    normalize_users = bool(data_cfg["normalize_user_mentions"])

    post_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    raw_target_messages = 0
    valid_before_truncation = 0

    for user_id in sorted(label_by_user):
        source_path = conversations_path / f"{user_id}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing raw user file: {source_path}")
        candidates: list[dict[str, Any]] = []
        for raw_order, event in enumerate(iter_raw_events(source_path, user_id)):
            if event.target is not True:
                continue
            raw_target_messages += 1
            if event.user_id != user_id:
                dropped["user_id_mismatch"] += 1
                continue
            timestamp = parse_created_utc(event.payload.get("created_utc"))
            if timestamp is None:
                dropped["invalid_timestamp"] += 1
                continue
            post_id = event.post_id
            if not post_id:
                dropped["missing_post_id"] += 1
                continue
            text = build_event_text(
                event,
                remove_markers,
                normalize_urls=normalize_urls,
                normalize_user_mentions=normalize_users,
            )
            if not text:
                dropped["empty_after_cleaning"] += 1
                continue
            candidates.append(
                {
                    "user_id": user_id,
                    "label": int(label_by_user[user_id]),
                    "post_id": post_id,
                    "kind": event.kind,
                    "created_utc": timestamp,
                    "timestamp_unix": int(timestamp.timestamp()),
                    "text": text,
                    "text_sha256": _text_hash(text),
                    "source_file": event.source_file,
                    "submission_id": event.payload.get("submission_id"),
                    "parent_id": event.payload.get("parent_id"),
                    "_raw_order": raw_order,
                }
            )
        candidates.sort(
            key=lambda row: (
                row["timestamp_unix"],
                row["_raw_order"],
                row["kind"],
                row["post_id"],
            )
        )
        seen_ids: set[str] = set()
        unique_candidates: list[dict[str, Any]] = []
        for row in candidates:
            if row["post_id"] in seen_ids:
                dropped["duplicate_post_id"] += 1
                continue
            seen_ids.add(row["post_id"])
            unique_candidates.append(row)
        valid_before_truncation += len(unique_candidates)
        selected = unique_candidates[:max_posts]
        dropped["truncated_valid_messages"] += max(0, len(unique_candidates) - len(selected))
        previous_timestamp: int | None = None
        for sequence_index, row in enumerate(selected):
            timestamp_unix = int(row["timestamp_unix"])
            delta_seconds = 0 if previous_timestamp is None else timestamp_unix - previous_timestamp
            if delta_seconds < 0:
                raise AssertionError(f"Non-monotonic timestamps after sorting for {user_id}")
            row["sequence_index"] = sequence_index
            row["delta_seconds"] = float(delta_seconds)
            row["delta_days"] = float(delta_seconds / 86400.0)
            row.pop("_raw_order", None)
            post_rows.append(row)
            previous_timestamp = timestamp_unix
        user_rows.append(
            {
                "user_id": user_id,
                "label": int(label_by_user[user_id]),
                "raw_target_count": len(candidates),
                "valid_target_count": len(unique_candidates),
                "sequence_length": len(selected),
                "was_truncated": len(unique_candidates) > max_posts,
                "first_created_utc": selected[0]["created_utc"] if selected else None,
                "last_created_utc": selected[-1]["created_utc"] if selected else None,
            }
        )

    posts = pd.DataFrame(post_rows)
    if posts.empty:
        raise ValueError("Preprocessing produced no valid target messages.")
    posts.insert(0, "embedding_index", range(len(posts)))
    posts = posts[POST_COLUMNS]
    users = pd.DataFrame(user_rows).sort_values("user_id").reset_index(drop=True)
    if users["sequence_length"].eq(0).any():
        missing = users.loc[users["sequence_length"].eq(0), "user_id"].tolist()
        raise ValueError(f"Users without a valid sequence: {missing[:10]}")
    if not posts.groupby("user_id")["timestamp_unix"].apply(
        lambda values: values.is_monotonic_increasing
    ).all():
        raise AssertionError("At least one user sequence is not time-monotonic.")
    first_gaps = posts.loc[posts["sequence_index"].eq(0), "delta_seconds"]
    if not first_gaps.eq(0).all():
        raise AssertionError("The first time gap must be zero for every user.")

    report = {
        "status": "pass",
        "config_hash": config.config_hash,
        "selection": {
            "target_only": True,
            "order": str(data_cfg["sequence_selection"]),
            "max_posts": max_posts,
            "padding_side": str(data_cfg["padding_side"]),
        },
        "counts": {
            "users": len(users),
            "control_users": int((users["label"] == 0).sum()),
            "depression_users": int((users["label"] == 1).sum()),
            "raw_target_messages": raw_target_messages,
            "valid_messages_before_truncation": valid_before_truncation,
            "kept_messages": len(posts),
            "truncated_users": int(users["was_truncated"].sum()),
            "minimum_sequence_length": int(users["sequence_length"].min()),
            "median_sequence_length": float(users["sequence_length"].median()),
            "maximum_sequence_length": int(users["sequence_length"].max()),
        },
        "dropped": dict(sorted(dropped.items())),
        "integrity": {
            "all_label_users_retained": len(users) == len(labels),
            "timestamps_monotonic": True,
            "first_gap_zero": True,
            "post_ids_unique": bool(posts["post_id"].is_unique),
            "source_fields_present": True,
        },
    }
    return posts, users, report


def write_preprocessed(
    posts: pd.DataFrame,
    users: pd.DataFrame,
    processed_dir: Path,
) -> tuple[Path, Path]:
    """Write stable parquet artifacts for downstream stages."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    posts_path = processed_dir / "post_index.parquet"
    users_path = processed_dir / "user_sequences.parquet"
    posts.to_parquet(posts_path, index=False, compression="zstd")
    users.to_parquet(users_path, index=False, compression="zstd")
    return posts_path, users_path
