"""Frozen emotion-model feature extraction with strict cache validation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ExperimentConfig
from .reproducibility import write_json


def corpus_fingerprint(posts: pd.DataFrame) -> str:
    """Hash the exact embedding row, post ID and cleaned-text hash order."""

    digest = hashlib.sha256()
    for row in posts[["embedding_index", "post_id", "text_sha256"]].itertuples(index=False):
        digest.update(f"{row.embedding_index}\t{row.post_id}\t{row.text_sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def embedding_spec(config: ExperimentConfig, posts: pd.DataFrame) -> dict[str, Any]:
    encoder = config.section("encoder")
    spec = {
        "model_name": str(encoder["model_name"]),
        "revision": str(encoder["revision"]),
        "max_tokens": int(config.section("data")["max_tokens"]),
        "pooling": str(encoder["pooling"]),
        "dtype": str(encoder["embedding_dtype"]),
        "row_count": len(posts),
        "corpus_fingerprint": corpus_fingerprint(posts),
    }
    stable = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["cache_fingerprint"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return spec


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_embedding_cache(
    embedding_path: Path,
    metadata_path: Path,
    expected_spec: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    """Reject stale, incomplete or dimensionally inconsistent embedding caches."""

    metadata = _read_metadata(metadata_path)
    if metadata is None or not embedding_path.is_file():
        raise FileNotFoundError("Embedding array or metadata is missing.")
    if metadata.get("spec") != expected_spec:
        raise ValueError("Embedding cache fingerprint/configuration does not match current posts.")
    if require_complete and metadata.get("status") != "complete":
        raise ValueError(f"Embedding cache is not complete: {metadata.get('status')}")
    array = np.load(embedding_path, mmap_mode="r")
    expected_shape = tuple(metadata.get("shape", []))
    if array.shape != expected_shape or array.shape[0] != expected_spec["row_count"]:
        raise ValueError(f"Embedding array shape {array.shape} does not match metadata {expected_shape}.")
    if str(array.dtype) != str(expected_spec["dtype"]):
        raise ValueError(f"Embedding dtype {array.dtype} does not match {expected_spec['dtype']}.")
    return metadata


@torch.inference_mode()
def encode_posts(
    config: ExperimentConfig,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    """Encode every selected post and support safe resume of an interrupted run."""

    posts_path = config.path("processed_dir") / "post_index.parquet"
    posts = pd.read_parquet(
        posts_path,
        columns=["embedding_index", "user_id", "post_id", "text", "text_sha256"],
    ).sort_values("embedding_index")
    expected_indices = np.arange(len(posts), dtype=np.int64)
    if not np.array_equal(posts["embedding_index"].to_numpy(), expected_indices):
        raise ValueError("post_index embedding_index must be contiguous and ordered from zero.")
    spec = embedding_spec(config, posts)
    encoder_cfg = config.section("encoder")
    output_dir = config.path("embeddings_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "post_embeddings.npy"
    metadata_path = output_dir / "post_embeddings.meta.json"
    index_path = output_dir / "post_embedding_index.parquet"

    existing = _read_metadata(metadata_path)
    if existing is not None and existing.get("status") == "complete":
        validate_embedding_cache(embedding_path, metadata_path, spec)
        if existing.get("config_hash") != config.config_hash:
            existing["config_hash"] = config.config_hash
            write_json(existing, metadata_path)
        progress(f"validated existing cache: {embedding_path}")
        return embedding_path, metadata_path
    if existing is not None and existing.get("spec") != spec:
        raise ValueError(
            "An incomplete embedding cache has a different fingerprint. "
            "Move it aside or remove it before restarting."
        )

    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(encoder_cfg["model_name"]),
        revision=str(encoder_cfg["revision"]),
        local_files_only=bool(encoder_cfg.get("local_files_only", False)),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(encoder_cfg["model_name"]),
        revision=str(encoder_cfg["revision"]),
        local_files_only=bool(encoder_cfg.get("local_files_only", False)),
    ).to(selected_device)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    expected_dim = int(config.section("projection")["input_dim"])
    if hidden_size != expected_dim:
        raise ValueError(f"Encoder hidden size {hidden_size} != projection input {expected_dim}")
    dtype = np.dtype(str(encoder_cfg["embedding_dtype"]))
    completed_rows = int(existing.get("completed_rows", 0)) if existing else 0
    if completed_rows:
        embeddings = open_memmap(embedding_path, mode="r+", dtype=dtype)
        if embeddings.shape != (len(posts), hidden_size):
            raise ValueError("Incomplete embedding array has an unexpected shape.")
    else:
        embeddings = open_memmap(
            embedding_path,
            mode="w+",
            dtype=dtype,
            shape=(len(posts), hidden_size),
        )
        posts[["embedding_index", "user_id", "post_id", "text_sha256"]].to_parquet(
            index_path, index=False, compression="zstd"
        )
    metadata = {
        "status": "in_progress",
        "config_hash": config.config_hash,
        "spec": spec,
        "shape": [len(posts), hidden_size],
        "completed_rows": completed_rows,
        "device": str(selected_device),
        "started_at_utc": (existing or {}).get(
            "started_at_utc", datetime.now(timezone.utc).isoformat()
        ),
        "index_file": index_path.name,
        "batch_order": "stable_cleaned_text_length",
    }
    write_json(metadata, metadata_path)
    actual_batch_size = int(batch_size or encoder_cfg.get("batch_size", 64))
    started = time.monotonic()
    texts = posts["text"].tolist()
    # Stable length bucketing only changes compute order. It sharply reduces padding on CPU,
    # while embeddings are still written at their canonical embedding_index rows.
    encode_order = np.argsort(posts["text"].str.len().to_numpy(), kind="stable")
    completed_batches = int((existing or {}).get("completed_batches", 0))
    total_batches = (len(texts) + actual_batch_size - 1) // actual_batch_size
    max_tokens = int(config.section("data")["max_tokens"])
    for batch_number in range(completed_batches, total_batches):
        start = batch_number * actual_batch_size
        end = min(start + actual_batch_size, len(texts))
        row_indices = encode_order[start:end]
        tokens = tokenizer(
            [texts[index] for index in row_indices],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        tokens = {name: tensor.to(selected_device) for name, tensor in tokens.items()}
        outputs = model(**tokens, output_hidden_states=True)
        first_token = outputs.hidden_states[-1][:, 0, :]
        embeddings[row_indices] = first_token.detach().to("cpu", dtype=torch.float32).numpy().astype(dtype)
        if end == len(texts) or (batch_number + 1) % 25 == 0:
            embeddings.flush()
            metadata["completed_rows"] = end
            metadata["completed_batches"] = batch_number + 1
            metadata["elapsed_seconds"] = time.monotonic() - started
            write_json(metadata, metadata_path)
            progress(f"encoded {end}/{len(texts)} posts")
    embeddings.flush()
    metadata.update(
        {
            "status": "complete",
            "completed_rows": len(posts),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    write_json(metadata, metadata_path)
    validate_embedding_cache(embedding_path, metadata_path, spec)
    return embedding_path, metadata_path
