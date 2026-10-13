"""Inner optimization training with epoch-boundary, provenance-checked recovery.

The original model, train_epoch and evaluate_loader are reused. Only G trains
and only E early-stops; H/C/T are never evaluated here. Final training requires
the O13/O14 lock interface and is deliberately not an executable O07 stage.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
import os
import platform
import random

import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader

from eot.config import ExperimentConfig
from eot.dataset import SequenceStore
from eot.model import EOTModel
from eot.reproducibility import collect_environment, set_global_seed
from eot.training import evaluate_loader, train_epoch
from .config import (OptimizationConfig, canonical_json, file_sha256,
                     load_optimization_config, project_path)
from .data import InnerDataContext, load_inner_context
from .losses import build_loss
from .runs import RunIdentity, make_run_identity, source_fingerprint
from .splits import digest

TRAINING_SOURCES = (
    "src/eot/config.py", "src/eot/dataset.py", "src/eot/embeddings.py",
    "src/eot/model.py", "src/eot/losses.py", "src/eot/time_features.py",
    "src/eot/training.py", "src/eot/reproducibility.py",
    "src/eot/optimization/config.py", "src/eot/optimization/runs.py",
    "src/eot/optimization/splits.py", "src/eot/optimization/data.py",
    "src/eot/optimization/losses.py", "src/eot/optimization/training.py",
)


@dataclass(frozen=True)
class InnerTrainingPlan:
    config: OptimizationConfig
    context: InnerDataContext
    identity: RunIdentity
    _inputs_json: str

    @property
    def binding(self) -> dict:
        return {"schema_version": 1, "run_id": self.identity.run_id,
            **self.identity.metadata, "split_content_sha256": self.context.split_sha256,
            "statistics_sha256": self.context.statistics_sha256,
            "training_statistics": self.context.statistics,
            "train_user_ids": self.context.user_ids("train"),
            "early_stop_user_ids": self.context.user_ids("early_stop"),
            "training": self.config.section("training"), "loss": self.config.values["loss"],
            "input_sha256": json.loads(self._inputs_json), "early_stop_role": "early_stop",
            "early_stop_rule": "E AP > best AP + 1e-12; ties retain earlier epoch"}


def _legacy_read_config(config: OptimizationConfig) -> ExperimentConfig:
    """Read adapter for SequenceStore/model only; never pass to legacy writers."""
    return ExperimentConfig(config.base_values, config.project_root / "configs/eot_main.yaml",
                            config.project_root, config.base_config_hash)


def prepare_inner_run(config: OptimizationConfig, *, stage: str, outer_fold: int,
                      inner_fold: int, seed: int) -> InnerTrainingPlan:
    """Read-only preflight. Creates no directories, model or training process."""
    if stage not in {"screening", "stability"}:
        raise ValueError("O07 supports screening/stability; smoke and final need later stage adapters")
    fresh = load_optimization_config(config.source, project_root=config.project_root)
    if fresh != config:
        raise ValueError("Config object differs from its locked, current file")
    cfg = config.section("training")
    required = {"optimizer": "adamw", "learning_rate": .0008, "weight_decay": .01,
        "batch_users": 128, "mixed_precision": False, "max_epochs": 50,
        "early_stopping_metric": "pr_auc", "early_stopping_patience": 8,
        "gradient_clip_norm": 1.0, "seeds": [42, 43, 44]}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError("Training constants differ from protocol v1")
    # Check stage/fold/seed before loading data or hashing the embedding array.
    make_run_identity(config, stage=stage, outer_fold=outer_fold, inner_fold=inner_fold,
                      seed=seed, code_sha256="0" * 64, split_sha256="0" * 64)
    context = load_inner_context(config, outer_fold, inner_fold)
    protection = json.loads(project_path(config.project_root,
        "outputs/optimization/v1/protocol/baseline_protection.json").read_text(encoding="utf-8"))
    protected = {r["path"]: r["sha256"] for r in protection["records"] if r["anchor"] == "project"}
    inputs = {}
    for name in ("data/processed/post_index.parquet", "data/processed/user_sequences.parquet",
                 "data/embeddings/post_embeddings.npy", "data/embeddings/post_embeddings.meta.json"):
        actual = file_sha256(project_path(config.project_root, name))
        if actual != protected.get(name):
            raise ValueError(f"Frozen training input changed: {name}")
        inputs[name] = actual
    identity = make_run_identity(config, stage=stage, outer_fold=outer_fold, inner_fold=inner_fold,
        seed=seed, code_sha256=source_fingerprint(config.project_root, TRAINING_SOURCES),
        split_sha256=digest({"split": context.split_sha256, "statistics": context.statistics_sha256}))
    return InnerTrainingPlan(config, context, identity, canonical_json(inputs))


@contextmanager
def _run_lock(identity):
    """OS releases this advisory lock on process death; no stale PID lock removal."""
    path = identity.artifact_path("writer.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This optimization run already has a writer") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This optimization run already has a writer") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_write(identity, relative, payload, *, tensor=False):
    path = identity.artifact_path(relative)
    temporary = identity.artifact_path(relative + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        if tensor:
            torch.save(payload, handle)
        else:
            handle.write((canonical_json(payload) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _runtime(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "device": str(device),
        "platform": platform.platform(), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "packages": {name: version(name) for name in
                     ("torchdiffeq", "pandas", "transformers", "PyYAML", "pyarrow")},
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def _rng_state(device):
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def _restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _commit(plan, state):
    # Alternate slots. Never overwrite the checkpoint referenced by run.json;
    # an interruption before the atomic pointer update leaves the old commit valid.
    relative = f"checkpoints/state_{state['epoch'] % 2}.pt"
    path = _atomic_write(plan.identity, relative, state, tensor=True)
    manifest = {"schema_version": 1, "binding": plan.binding,
        "checkpoint": relative, "checkpoint_sha256": file_sha256(path),
        "status": state["status"], "stop_reason": state["stop_reason"],
        "epoch": state["epoch"], "best_epoch": state["best_epoch"],
        "best_score": state["best_score"], "patience": state["patience"],
        "runtime": state["runtime"], "positive_weight": state["positive_weight"],
        "early_stop_labels_sha256": state["early_stop_labels_sha256"], "history": state["history"]}
    _atomic_write(plan.identity, "run.json", manifest)
    return manifest


def read_checkpoint(plan: InnerTrainingPlan) -> tuple[dict, dict]:
    """Verify checksum and the full expected binding before loading any weights."""
    manifest = json.loads(plan.identity.artifact_path("run.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("binding") != plan.binding:
        raise ValueError("Resume manifest does not match config/source/split/statistics/stage/seed")
    if manifest.get("checkpoint") not in {"checkpoints/state_0.pt", "checkpoints/state_1.pt"}:
        raise ValueError("Unexpected checkpoint path")
    path = plan.identity.artifact_path(manifest["checkpoint"])
    if file_sha256(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Checkpoint checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("binding") != plan.binding:
        raise ValueError("Checkpoint belongs to a different optimization run")
    for key in ("epoch", "best_epoch", "best_score", "patience", "status", "stop_reason",
                "runtime", "history", "positive_weight", "early_stop_labels_sha256"):
        if state.get(key) != manifest.get(key):
            raise ValueError(f"Checkpoint/manifest mismatch: {key}")
    if (type(state["epoch"]) is not int or not 1 <= state["best_epoch"] <= state["epoch"] <= 50
            or [h["epoch"] for h in state["history"]] != list(range(1, state["epoch"] + 1))
            or not math.isfinite(state["best_score"])):
        raise ValueError("Invalid checkpoint training progress")
    stopped = state["patience"] >= plan.config.section("training")["early_stopping_patience"]
    complete = stopped or state["epoch"] == plan.config.section("training")["max_epochs"]
    expected_reason = "early_stopping" if stopped else "max_epochs" if complete else "paused"
    if state["status"] != ("complete" if complete else "paused") or state["stop_reason"] != expected_reason:
        raise ValueError("Invalid completion status")
    return state, manifest


def load_trained_model(plan: InnerTrainingPlan, *, which: str = "best", device_name="cpu"):
    """Load a checked best/latest model without training or changing artifacts."""
    if which not in {"best", "latest"}:
        raise ValueError("which must be best or latest")
    state, manifest = read_checkpoint(plan)
    model = EOTModel.from_config(_legacy_read_config(plan.config)).to(torch.device(device_name))
    model.load_state_dict(state["best_model_state_dict" if which == "best" else "model_state_dict"])
    return model.eval(), manifest


def _fit_inner(plan, store, *, device_name, resume, stop_after_epoch):
    """Internal engine; production callers always use the preflighted public entry."""
    cfg = plan.config.section("training")
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int
            or not 1 <= stop_after_epoch <= cfg["max_epochs"]):
        raise ValueError("stop_after_epoch must be an absolute epoch in 1..50; it does not change budget")
    if resume and not plan.identity.artifact_path("run.json").is_file():
        raise FileNotFoundError("No committed checkpoint for this exact run; cannot resume")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU/CUDA training is supported")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with _run_lock(plan.identity):
        exists = plan.identity.artifact_path("run.json").exists()
        if exists and not resume:
            raise FileExistsError("This run exists; use resume=True to continue or verify completion")
        train_ids = plan.context.user_ids("train")
        early_ids = plan.context.user_ids("early_stop")
        train_users = pd.DataFrame({"user_id": train_ids, "label": [store.users[u].label for u in train_ids]})
        loss_fn = build_loss(plan.config, train_users, train_ids).to(device)
        stats = plan.context.statistics
        if (loss_fn.fit_labels_sha256 != stats["fit_labels_sha256"]
                or loss_fn.base_positive_weight != stats["base_positive_weight"]):
            raise ValueError("Loss weights differ from actual G training statistics")
        early_labels = [store.users[u].label for u in early_ids]
        if set(early_labels) != {0, 1}:
            raise ValueError("Early-stop users require both classes")
        early_labels_sha = digest([[u, int(y)] for u, y in zip(early_ids, early_labels)])
        train_dataset = plan.context.dataset(store, "train")
        early_dataset = plan.context.dataset(store, "early_stop")
        # No holdout, calibration or outer-test dataset is instantiated here.
        seed = plan.identity.metadata["seed"]
        set_global_seed(seed)
        runtime = _runtime(device)
        state, manifest = read_checkpoint(plan) if exists else (None, None)
        if state is not None:
            if state["runtime"] != runtime:
                raise ValueError("Resume runtime differs; exact continuation requires the same environment/device")
            if state["early_stop_labels_sha256"] != early_labels_sha:
                raise ValueError("Early-stop label source changed")
            if state["positive_weight"] != float(loss_fn.positive_weight.item()):
                raise ValueError("Checkpoint positive weight changed")
            if state["status"] == "complete":
                return manifest
            if stop_after_epoch is not None and stop_after_epoch <= state["epoch"]:
                return manifest
        model = EOTModel.from_config(_legacy_read_config(plan.config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        if state is None:
            environment = collect_environment(plan.config.project_root, plan.config.config_hash)
            epoch = best_epoch = patience = 0
            best_score, best_state, history = -math.inf, None, []
        else:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            _restore_rng(state["rng_state"])
            environment = state["environment"]
            epoch, best_epoch, patience = state["epoch"], state["best_epoch"], state["patience"]
            best_score, best_state, history = state["best_score"], state["best_model_state_dict"], state["history"]
        early_loader = DataLoader(early_dataset, batch_size=cfg["batch_users"], shuffle=False,
                                  num_workers=0, pin_memory=device.type == "cuda")
        limit = cfg["max_epochs"] if stop_after_epoch is None else stop_after_epoch
        for epoch in range(epoch + 1, limit + 1):
            train_loader = DataLoader(train_dataset, batch_size=cfg["batch_users"], shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch), num_workers=0, pin_memory=device.type == "cuda")
            trained = train_epoch(model, train_loader, loss_fn, optimizer, device, cfg["gradient_clip_norm"])
            evaluated = evaluate_loader(model, early_loader, loss_fn, device)
            score = float(evaluated["pr_auc"])
            if (not all(math.isfinite(float(x)) for x in (*trained.values(), evaluated["loss"], score))
                    or not np.isfinite(evaluated["probabilities"]).all()):
                raise FloatingPointError("Non-finite training/early-stop result; checkpoint not committed")
            if evaluated["user_ids"] != early_ids or evaluated["labels"] != early_labels:
                raise ValueError("Evaluation did not use exactly the early-stop users/labels")
            improved = score > best_score + 1e-12
            if improved:
                best_score, best_epoch, patience, best_state = score, epoch, 0, _cpu_model_state(model)
            else:
                patience += 1
            history.append({"epoch": epoch, "train_loss": trained["loss"],
                "train_final_loss": trained["final_loss"], "train_prefix_loss": trained["prefix_loss"],
                "early_stop_loss": float(evaluated["loss"]), "early_stop_pr_auc": score, "improved": improved})
            stopped = patience >= cfg["early_stopping_patience"]
            complete = stopped or epoch == cfg["max_epochs"]
            state = {"binding": plan.binding, "epoch": epoch, "best_epoch": best_epoch,
                "best_score": best_score, "patience": patience, "history": history,
                "status": "complete" if complete else "paused",
                "stop_reason": "early_stopping" if stopped else "max_epochs" if complete else "paused",
                "model_state_dict": _cpu_model_state(model), "best_model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(), "rng_state": _rng_state(device),
                "runtime": runtime, "environment": environment,
                "positive_weight": float(loss_fn.positive_weight.item()),
                "early_stop_labels_sha256": early_labels_sha}
            manifest = _commit(plan, state)
            print(f"{plan.identity.run_id[:12]} epoch={epoch} loss={trained['loss']:.6f} "
                  f"early_stop_AP={score:.6f} best_epoch={best_epoch} patience={patience}/8", flush=True)
            if complete:
                break
        return manifest


def train_inner_run(config: OptimizationConfig, *, stage: str, outer_fold: int, inner_fold: int,
                    seed: int, device_name: str | None = None, resume: bool = False,
                    stop_after_epoch: int | None = None) -> dict:
    plan = prepare_inner_run(config, stage=stage, outer_fold=outer_fold, inner_fold=inner_fold, seed=seed)
    store = SequenceStore(_legacy_read_config(config))
    return _fit_inner(plan, store, device_name=device_name, resume=resume, stop_after_epoch=stop_after_epoch)
