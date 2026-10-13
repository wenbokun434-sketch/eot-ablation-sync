"""Read-only validation of the raw eRisk Task 2 corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from .config import ExperimentConfig
from .raw_data import build_event_text, iter_raw_events, parse_created_utc, read_labels


REQUIRED_FIELDS = {
    "submission": ("user_id", "target", "created_utc", "submission_id"),
    "comment": ("user_id", "target", "created_utc", "comment_id"),
}


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def audit_raw_dataset(config: ExperimentConfig) -> dict[str, Any]:
    """Audit labels, files, event schema and target-message validity."""

    root = config.path("raw_data_root")
    paths = config.section("paths")
    data_cfg = config.section("data")
    labels_path = root / str(paths["labels_file"])
    conversations_path = root / str(paths["conversations_dir"])
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not conversations_path.is_dir():
        raise FileNotFoundError(f"Conversations directory not found: {conversations_path}")

    labels, label_errors = read_labels(labels_path)
    label_counter = Counter(record.user_id for record in labels)
    duplicate_label_ids = sorted(user_id for user_id, count in label_counter.items() if count > 1)
    label_by_user = {record.user_id: record.label for record in labels}
    files = sorted(conversations_path.glob("subject_*.json"))
    file_by_user = {path.stem: path for path in files}
    label_ids = set(label_by_user)
    file_ids = set(file_by_user)

    event_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    invalid_json_files: list[dict[str, str]] = []
    target_user_mismatches: list[dict[str, str | None]] = []
    invalid_target_timestamps: list[dict[str, str | None]] = []
    duplicate_target_ids: list[str] = []
    target_ids: set[str] = set()
    valid_target_counts: dict[str, int] = {user_id: 0 for user_id in label_ids}
    remove_markers = {str(item).lower() for item in data_cfg["remove_markers"]}

    for path in files:
        owner = path.stem
        try:
            events = iter_raw_events(path, owner)
            for event in events:
                event_counts[f"{event.kind}_all"] += 1
                for field in REQUIRED_FIELDS[event.kind]:
                    if field not in event.payload or event.payload[field] is None:
                        missing_fields[f"{event.kind}.{field}"] += 1
                if event.target is not True:
                    event_counts[f"{event.kind}_non_target"] += 1
                    continue
                event_counts[f"{event.kind}_target"] += 1
                if event.user_id != owner:
                    event_counts["target_user_mismatch"] += 1
                    if len(target_user_mismatches) < 50:
                        target_user_mismatches.append(
                            {
                                "source_file": event.source_file,
                                "owner_user_id": owner,
                                "event_user_id": event.user_id,
                                "post_id": event.post_id,
                            }
                        )
                timestamp = parse_created_utc(event.payload.get("created_utc"))
                if timestamp is None:
                    event_counts["target_invalid_timestamp"] += 1
                    if len(invalid_target_timestamps) < 50:
                        invalid_target_timestamps.append(
                            {"source_file": event.source_file, "post_id": event.post_id}
                        )
                text = build_event_text(
                    event,
                    remove_markers,
                    bool(data_cfg["normalize_urls"]),
                    bool(data_cfg["normalize_user_mentions"]),
                )
                if not text:
                    event_counts["target_empty_after_cleaning"] += 1
                if event.post_id is None:
                    event_counts["target_missing_post_id"] += 1
                elif event.post_id in target_ids:
                    event_counts["target_duplicate_post_id"] += 1
                    if len(duplicate_target_ids) < 50:
                        duplicate_target_ids.append(event.post_id)
                else:
                    target_ids.add(event.post_id)
                if (
                    event.user_id == owner
                    and owner in label_by_user
                    and timestamp is not None
                    and bool(text)
                    and event.post_id is not None
                ):
                    valid_target_counts[owner] += 1
        except Exception as exc:
            invalid_json_files.append({"source_file": path.name, "error": str(exc)})

    class_counts = Counter(record.label for record in labels)
    valid_counts = list(valid_target_counts.values())
    critical_issues = {
        "malformed_label_rows": len(label_errors),
        "duplicate_label_user_ids": len(duplicate_label_ids),
        "labels_without_json": len(label_ids - file_ids),
        "json_without_labels": len(file_ids - label_ids),
        "invalid_json_files": len(invalid_json_files),
        "target_user_mismatches": event_counts["target_user_mismatch"],
        "users_without_valid_target_messages": sum(value == 0 for value in valid_counts),
    }
    status = "pass" if not any(critical_issues.values()) else "review_required"
    return {
        "status": status,
        "config_hash": config.config_hash,
        "dataset": {
            "root": str(root),
            "labels_file": labels_path.name,
            "conversations_dir": conversations_path.name,
        },
        "users": {
            "label_rows": len(labels),
            "unique_label_users": len(label_ids),
            "json_files": len(files),
            "class_counts": {"control_0": class_counts[0], "depression_1": class_counts[1]},
            "labels_without_json": sorted(label_ids - file_ids),
            "json_without_labels": sorted(file_ids - label_ids),
            "duplicate_label_user_ids": duplicate_label_ids,
            "users_without_valid_target_messages": sorted(
                user_id for user_id, count in valid_target_counts.items() if count == 0
            ),
            "valid_target_message_count": _summary(valid_counts),
        },
        "events": dict(sorted(event_counts.items())),
        "schema": {
            "missing_required_fields": dict(sorted(missing_fields.items())),
            "label_errors": label_errors[:50],
            "invalid_json_files": invalid_json_files[:50],
            "target_user_mismatch_examples": target_user_mismatches,
            "invalid_target_timestamp_examples": invalid_target_timestamps,
            "duplicate_target_post_id_examples": duplicate_target_ids,
            "example_limit": 50,
        },
        "critical_issues": critical_issues,
        "handling_rules": {
            "target_selection": "keep only events whose target value is boolean true",
            "user_validation": "a target event must match the subject id in its source filename",
            "submission_text": "clean non-empty title and body separately, then join with [SEP]",
            "comment_text": "clean body",
            "invalid_text": "drop events empty after cleaning; retain users for reporting",
            "invalid_timestamp": "drop event and report it",
            "invalid_or_duplicate_id": "drop event and report it",
        },
    }
