"""Lazy fixed-width user sequence tensors backed by the embedding memmap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .embeddings import embedding_spec, validate_embedding_cache
from .time_features import TimeNormalizer


@dataclass(frozen=True)
class UserRows:
    user_id: str
    label: int
    embedding_indices: np.ndarray
    delta_days: np.ndarray


class SequenceStore:
    """Validate and index formal post/embedding artifacts once."""

    def __init__(self, config: ExperimentConfig) -> None:
        posts = pd.read_parquet(
            config.path("processed_dir") / "post_index.parquet",
            columns=[
                "embedding_index",
                "user_id",
                "label",
                "sequence_index",
                "post_id",
                "text_sha256",
                "delta_days",
            ],
        ).sort_values("embedding_index")
        spec = embedding_spec(config, posts)
        embedding_path = config.path("embeddings_dir") / "post_embeddings.npy"
        metadata_path = config.path("embeddings_dir") / "post_embeddings.meta.json"
        validate_embedding_cache(embedding_path, metadata_path, spec)
        self.embeddings = np.load(embedding_path, mmap_mode="r")
        self.max_posts = int(config.section("data")["max_posts"])
        self.embedding_dim = int(self.embeddings.shape[1])
        self.users: dict[str, UserRows] = {}
        for user_id, frame in posts.groupby("user_id", sort=False):
            frame = frame.sort_values("sequence_index")
            sequence_indices = frame["sequence_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(sequence_indices, np.arange(len(frame))):
                raise ValueError(f"Non-contiguous sequence_index for {user_id}")
            labels = frame["label"].unique()
            if len(labels) != 1:
                raise ValueError(f"Inconsistent labels for {user_id}")
            self.users[str(user_id)] = UserRows(
                user_id=str(user_id),
                label=int(labels[0]),
                embedding_indices=frame["embedding_index"].to_numpy(dtype=np.int64),
                delta_days=frame["delta_days"].to_numpy(dtype=np.float64),
            )


class UserSequenceDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Create right-padded tensors on access, converting cached features to float32."""

    def __init__(
        self,
        store: SequenceStore,
        user_ids: Iterable[str],
        time_normalizer: TimeNormalizer,
    ) -> None:
        self.store = store
        self.user_ids = [str(value) for value in user_ids]
        missing = sorted(set(self.user_ids) - set(store.users))
        if missing:
            raise KeyError(f"Unknown user IDs: {missing[:10]}")
        self.time_normalizer = time_normalizer

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.store.users[self.user_ids[index]]
        length = len(record.embedding_indices)
        if length < 1 or length > self.store.max_posts:
            raise ValueError(f"Invalid sequence length {length} for {record.user_id}")
        embeddings = np.zeros(
            (self.store.max_posts, self.store.embedding_dim), dtype=np.float32
        )
        embeddings[:length] = self.store.embeddings[record.embedding_indices].astype(np.float32)
        delta = np.zeros(self.store.max_posts, dtype=np.float32)
        delta[:length] = self.time_normalizer.transform(record.delta_days)
        mask = np.zeros(self.store.max_posts, dtype=np.bool_)
        mask[:length] = True
        if delta[0] != 0.0 or np.any(delta < 0) or np.any(delta > 1):
            raise AssertionError(f"Invalid normalized time gaps for {record.user_id}")
        return {
            "user_id": record.user_id,
            "post_embeddings": torch.from_numpy(embeddings),
            "delta_t": torch.from_numpy(delta),
            "mask": torch.from_numpy(mask),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(float(record.label), dtype=torch.float32),
        }
