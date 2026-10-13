"""Reusable training primitives and the required two-epoch smoke run."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import SequenceStore, UserSequenceDataset
from .losses import EOTLoss, compute_positive_weight
from .model import EOTModel
from .reproducibility import set_global_seed, write_json
from .time_features import load_time_normalizer


def _to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    max_sequence_length: int | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    result: dict[str, torch.Tensor | list[str]] = {"user_id": batch["user_id"]}
    for name in ("post_embeddings", "delta_t", "mask", "length", "label"):
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}")
        result[name] = value.to(device, non_blocking=device.type == "cuda")
    if max_sequence_length is not None:
        limit = int(max_sequence_length)
        result["post_embeddings"] = result["post_embeddings"][:, :limit]
        result["delta_t"] = result["delta_t"][:, :limit]
        result["mask"] = result["mask"][:, :limit]
        result["length"] = torch.clamp(result["length"], max=limit)
    return result


def train_epoch(
    model: EOTModel,
    loader: DataLoader,
    loss_fn: EOTLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
    max_sequence_length: int | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "final_loss": 0.0, "prefix_loss": 0.0, "users": 0}
    for raw_batch in loader:
        batch = _to_device(raw_batch, device, max_sequence_length)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["post_embeddings"], batch["delta_t"], batch["mask"])
        loss, parts = loss_fn(
            output["final_logits"], output["prefix_logits"], batch["label"], batch["length"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        users = int(batch["label"].shape[0])
        totals["loss"] += float(loss.detach()) * users
        totals["final_loss"] += float(parts["final_loss"]) * users
        totals["prefix_loss"] += float(parts["prefix_loss"]) * users
        totals["users"] += users
    return {key: value / totals["users"] for key, value in totals.items() if key != "users"}


@torch.inference_mode()
def evaluate_loader(
    model: EOTModel,
    loader: DataLoader,
    loss_fn: EOTLoss,
    device: torch.device,
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    user_ids: list[str] = []
    losses: list[float] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device, max_sequence_length)
        output = model(batch["post_embeddings"], batch["delta_t"], batch["mask"])
        loss, _ = loss_fn(
            output["final_logits"], output["prefix_logits"], batch["label"], batch["length"]
        )
        losses.extend([float(loss)] * int(batch["label"].shape[0]))
        user_ids.extend(str(value) for value in batch["user_id"])
        labels.extend(batch["label"].detach().cpu().tolist())
        probabilities.extend(torch.sigmoid(output["final_logits"]).detach().cpu().tolist())
    pr_auc = float(average_precision_score(labels, probabilities)) if len(set(labels)) > 1 else float("nan")
    return {
        "loss": float(np.mean(losses)),
        "pr_auc": pr_auc,
        "labels": labels,
        "probabilities": probabilities,
        "user_ids": user_ids,
    }


def _balanced_ids(user_ids: Iterable[str], store: SequenceStore, per_class: int) -> list[str]:
    by_class = {0: [], 1: []}
    for user_id in user_ids:
        by_class[store.users[user_id].label].append(user_id)
    if min(len(by_class[0]), len(by_class[1])) < per_class:
        raise ValueError("Not enough users from each class for a balanced smoke run.")
    return sorted(by_class[0])[:per_class] + sorted(by_class[1])[:per_class]


def run_smoke_training(
    config: ExperimentConfig,
    fold: int = 0,
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Train two epochs on a small balanced sample, save and reload a checkpoint."""

    set_global_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = json.loads(
        (config.path("splits_dir") / f"fold_{fold}.json").read_text(encoding="utf-8")
    )
    store = SequenceStore(config)
    normalizer = load_time_normalizer(config, fold)
    train_ids = _balanced_ids(split["train_user_ids"], store, per_class=2)
    validation_ids = _balanced_ids(split["validation_user_ids"], store, per_class=2)
    train_dataset = UserSequenceDataset(store, train_ids, normalizer)
    validation_dataset = UserSequenceDataset(store, validation_ids, normalizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
    model = EOTModel.from_config(config).to(device)
    labels = [store.users[user_id].label for user_id in train_ids]
    loss_cfg = config.section("loss")
    loss_fn = EOTLoss(
        compute_positive_weight(labels),
        config.section("pooling")["checkpoints"],
        float(loss_cfg["final_weight"]),
        float(loss_cfg["prefix_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=float(config.section("training")["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, 3):
        train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            float(config.section("training")["gradient_clip_norm"]),
            max_sequence_length=16,
        )
        validation_metrics = evaluate_loader(
            model, validation_loader, loss_fn, device, max_sequence_length=16
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "validation_pr_auc": validation_metrics["pr_auc"],
            }
        )
    temporary_checkpoint = config.path("outputs_dir") / "tmp" / "smoke_checkpoint.pt"
    temporary_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_hash": config.config_hash,
            "fold": fold,
            "seed": seed,
            "epochs": 2,
        },
        temporary_checkpoint,
    )
    reloaded = EOTModel.from_config(config).to(device)
    checkpoint = torch.load(temporary_checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    after_reload = evaluate_loader(
        reloaded, validation_loader, loss_fn, device, max_sequence_length=16
    )
    before_reload = evaluate_loader(
        model, validation_loader, loss_fn, device, max_sequence_length=16
    )
    reload_matches = np.allclose(
        before_reload["probabilities"], after_reload["probabilities"], atol=1e-7
    )
    temporary_checkpoint.unlink()
    report = {
        "status": "pass" if reload_matches and history[-1]["train_loss"] < history[0]["train_loss"] else "review_required",
        "config_hash": config.config_hash,
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "users": {"train": train_ids, "validation": validation_ids},
        "sequence_prefix_length": 16,
        "history": history,
        "loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
        "checkpoint_reload_matches": bool(reload_matches),
        "temporary_checkpoint_removed": not temporary_checkpoint.exists(),
    }
    write_json(report, config.path("outputs_dir") / "reports" / "smoke_training.json")
    return report


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_formal_run(
    config: ExperimentConfig,
    fold: int,
    seed: int,
    device_name: str | None = None,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one fold/seed with resumable last state and a single retained best model."""

    configured_seeds = [int(value) for value in config.section("training")["seeds"]]
    if seed not in configured_seeds:
        raise ValueError(f"Seed {seed} is not one of the configured formal seeds {configured_seeds}")
    n_folds = int(config.section("cross_validation")["n_splits"])
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}]")
    set_global_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = json.loads(
        (config.path("splits_dir") / f"fold_{fold}.json").read_text(encoding="utf-8")
    )
    if split.get("config_hash") != config.config_hash:
        raise ValueError(f"Fold {fold} was generated from a different configuration")
    store = SequenceStore(config)
    normalizer = load_time_normalizer(config, fold)
    train_dataset = UserSequenceDataset(store, split["train_user_ids"], normalizer)
    validation_dataset = UserSequenceDataset(store, split["validation_user_ids"], normalizer)
    training_cfg = config.section("training")
    batch_size = int(training_cfg["batch_users"])
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = EOTModel.from_config(config).to(device)
    train_labels = [store.users[user_id].label for user_id in split["train_user_ids"]]
    positive_weight = compute_positive_weight(train_labels)
    loss_cfg = config.section("loss")
    loss_fn = EOTLoss(
        positive_weight,
        config.section("pooling")["checkpoints"],
        float(loss_cfg["final_weight"]),
        float(loss_cfg["prefix_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    run_name = f"fold_{fold}_seed_{seed}"
    checkpoint_dir = config.path("outputs_dir") / "checkpoints" / run_name
    log_dir = config.path("outputs_dir") / "logs" / run_name
    prediction_dir = config.path("outputs_dir") / "predictions"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    manifest_path = log_dir / "run.json"
    history_path = log_dir / "history.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete" and manifest.get("config_hash") == config.config_hash:
            if not best_path.is_file():
                raise FileNotFoundError(f"Completed manifest exists without {best_path}")
            print(f"skip completed run: {run_name}")
            return manifest
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = int(max_epochs_override or training_cfg["max_epochs"])
    patience_limit = int(training_cfg["early_stopping_patience"])
    start_epoch = 1
    best_score = -math.inf
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config_hash") != config.config_hash:
            raise ValueError(f"Refusing stale resume checkpoint: {last_path}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        patience = int(state["patience"])
        history = list(state["history"])
        print(f"resume {run_name} at epoch {start_epoch}")
    started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.monotonic()
    for epoch in range(start_epoch, max_epochs + 1):
        generator = torch.Generator().manual_seed(seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        epoch_start = time.monotonic()
        train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            float(training_cfg["gradient_clip_norm"]),
        )
        validation_metrics = evaluate_loader(model, validation_loader, loss_fn, device)
        score = float(validation_metrics["pr_auc"])
        improved = math.isfinite(score) and score > best_score + 1e-12
        if improved:
            best_score = score
            best_epoch = epoch
            patience = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "config_hash": config.config_hash,
                    "fold": fold,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_pr_auc": score,
                    "positive_weight": positive_weight,
                },
                best_path,
            )
        else:
            patience += 1
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_final_loss": train_metrics["final_loss"],
            "train_prefix_loss": train_metrics["prefix_loss"],
            "validation_loss": validation_metrics["loss"],
            "validation_pr_auc": score,
            "improved": improved,
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(epoch_record)
        write_json({"run_name": run_name, "history": history}, history_path)
        _atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config_hash": config.config_hash,
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "patience": patience,
                "history": history,
            },
            last_path,
        )
        print(
            f"{run_name} epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_pr_auc={score:.6f} best={best_score:.6f} patience={patience}/{patience_limit}",
            flush=True,
        )
        if patience >= patience_limit:
            break
    if not best_path.is_file():
        raise RuntimeError(f"No valid best checkpoint was produced for {run_name}")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_validation = evaluate_loader(model, validation_loader, loss_fn, device)
    pd.DataFrame(
        {
            "user_id": final_validation["user_ids"],
            "label": np.asarray(final_validation["labels"], dtype=np.int8),
            "probability": np.asarray(final_validation["probabilities"], dtype=np.float32),
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
        }
    ).to_parquet(
        prediction_dir / f"{run_name}_validation.parquet", index=False, compression="zstd"
    )
    if last_path.exists():
        last_path.unlink()
    manifest = {
        "status": "complete",
        "run_name": run_name,
        "config_hash": config.config_hash,
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - run_start,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_score,
        "positive_weight": positive_weight,
        "train_users": len(train_dataset),
        "validation_users": len(validation_dataset),
        "early_stopped": len(history) < max_epochs,
        "best_checkpoint": str(best_path.relative_to(config.project_root)),
        "validation_predictions": str(
            (prediction_dir / f"{run_name}_validation.parquet").relative_to(config.project_root)
        ),
    }
    write_json(manifest, manifest_path)
    return manifest
