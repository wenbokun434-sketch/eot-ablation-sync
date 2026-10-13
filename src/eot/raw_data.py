"""Raw eRisk Task 2 contracts shared by auditing and preprocessing."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)\S+")
USER_PATTERN = re.compile(r"(?i)(?<!\w)(?:/u/|u/)[A-Za-z0-9_-]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class LabelRecord:
    user_id: str
    label: int


@dataclass(frozen=True)
class RawEvent:
    source_file: str
    owner_user_id: str
    kind: str
    payload: dict[str, Any]

    @property
    def user_id(self) -> str | None:
        value = self.payload.get("user_id")
        return str(value) if value is not None else None

    @property
    def target(self) -> Any:
        return self.payload.get("target")

    @property
    def post_id(self) -> str | None:
        key = "submission_id" if self.kind == "submission" else "comment_id"
        value = self.payload.get(key)
        return str(value) if value is not None else None


def read_labels(path: Path) -> tuple[list[LabelRecord], list[str]]:
    """Read whitespace-separated ``user_id label`` rows and report malformed rows."""

    records: list[LabelRecord] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                errors.append(f"line {line_number}: expected 2 fields")
                continue
            user_id, raw_label = parts
            try:
                label = int(raw_label)
            except ValueError:
                errors.append(f"line {line_number}: invalid label {raw_label!r}")
                continue
            if label not in (0, 1):
                errors.append(f"line {line_number}: label must be 0 or 1")
                continue
            records.append(LabelRecord(user_id=user_id, label=label))
    return records, errors


def iter_raw_events(path: Path, owner_user_id: str) -> Iterator[RawEvent]:
    """Yield every submission and comment from one user JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, list):
        raise ValueError(f"Top-level JSON value is not a list: {path}")
    for item_index, item in enumerate(document):
        if not isinstance(item, dict):
            raise ValueError(f"Item {item_index} is not an object: {path}")
        submission = item.get("submission")
        if submission is not None:
            if not isinstance(submission, dict):
                raise ValueError(f"submission at item {item_index} is not an object: {path}")
            yield RawEvent(path.name, owner_user_id, "submission", submission)
        comments = item.get("comments", [])
        if comments is None:
            comments = []
        if not isinstance(comments, list):
            raise ValueError(f"comments at item {item_index} is not a list: {path}")
        for comment_index, comment in enumerate(comments):
            if not isinstance(comment, dict):
                raise ValueError(
                    f"comment {comment_index} at item {item_index} is not an object: {path}"
                )
            yield RawEvent(path.name, owner_user_id, "comment", comment)


def parse_created_utc(value: Any) -> datetime | None:
    """Parse the timestamp formats observed in eRisk into timezone-aware UTC."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S%z"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def clean_component(
    value: Any,
    remove_markers: set[str],
    normalize_urls: bool = True,
    normalize_user_mentions: bool = True,
) -> str:
    """Normalize one title/body component without inventing missing content."""

    if value is None:
        return ""
    text = html.unescape(str(value)).strip()
    if text.lower() in remove_markers:
        return ""
    if normalize_urls:
        text = URL_PATTERN.sub("<URL>", text)
    if normalize_user_mentions:
        text = USER_PATTERN.sub("<USER>", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def build_event_text(
    event: RawEvent,
    remove_markers: set[str],
    normalize_urls: bool = True,
    normalize_user_mentions: bool = True,
) -> str:
    """Build submission ``title [SEP] body`` or comment body text."""

    kwargs = {
        "remove_markers": remove_markers,
        "normalize_urls": normalize_urls,
        "normalize_user_mentions": normalize_user_mentions,
    }
    if event.kind == "submission":
        title = clean_component(event.payload.get("title"), **kwargs)
        body = clean_component(event.payload.get("body"), **kwargs)
        return " [SEP] ".join(part for part in (title, body) if part)
    return clean_component(event.payload.get("body"), **kwargs)
