"""O14 frozen final training, C calibration, sealed locks and one outer-test evaluation.

This module is the only O14 executable adapter. It reuses the original model and
training primitives, calls the O13 ``require_final_plan`` gate before touching any
final run, fits final time statistics on S only, and keeps all O14 artifacts under
``outputs/optimization/v1/final/``. Outer-test cohorts remain unreachable until
all five second (post-calibration) locks verify.
"""
from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eot.dataset import SequenceStore, UserSequenceDataset
from eot.model import EOTModel
from eot.time_features import TimeNormalizer
from eot.reproducibility import collect_environment, set_global_seed
from eot.training import _to_device, train_epoch

from .comparison import compare_objects, evaluate_selection, pool_fold_frames
from .config import (OptimizationConfig, canonical_json, file_sha256,
                     load_optimization_config, project_path)
from .data import _inputs, fit_training_statistics
from .decisions import (SelectionPredictions, load_selection_scope,
                        select_dual_thresholds)
from .evaluation import (DECISION_COLUMNS, FINAL_COLUMNS, PREFIX_COLUMNS, PredictionBundle,
                         SeedRun, build_bundle, decision_frame, evaluate_frame,
                         frame_digest, legacy_oof_decision_frame, legacy_oof_recompute)
from .losses import build_loss
from .runs import make_run_identity, source_fingerprint
from .splits import checked_ids, digest
from .training import (_atomic_write, _cpu_model_state, _legacy_read_config,
                       _restore_rng, _rng_state, _run_lock, _runtime)

FINAL_ROOT = "outputs/optimization/v1/final"
O13_LOCK = "outputs/optimization/v1/locked/final_lock.json"
O13_AUDIT = "outputs/optimization/v1/locked/audit_o13.py"
PROTECTION = "outputs/optimization/v1/protocol/baseline_protection.json"
FINAL_SEEDS = (42, 43, 44)
BUILDING = "building"
FINAL_SOURCES = (
    "src/eot/config.py", "src/eot/dataset.py", "src/eot/embeddings.py",
    "src/eot/losses.py", "src/eot/model.py", "src/eot/reproducibility.py",
    "src/eot/time_features.py", "src/eot/training.py",
    "src/eot/optimization/config.py", "src/eot/optimization/runs.py",
    "src/eot/optimization/splits.py", "src/eot/optimization/data.py",
    "src/eot/optimization/losses.py", "src/eot/optimization/training.py",
    "src/eot/optimization/decisions.py", "src/eot/optimization/evaluation.py",
    "src/eot/optimization/comparison.py", "src/eot/optimization/final_training.py",
)


def _final_path(config: OptimizationConfig, relative: str) -> Path:
    return project_path(config.project_root, f"{FINAL_ROOT}/{relative}", allowed_prefix=FINAL_ROOT)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _protected_records(config: OptimizationConfig) -> dict:
    records = _load_json(project_path(config.project_root, PROTECTION))["records"]
    return {row["path"]: row for row in records if row["anchor"] == "project"}


def _load_o13_gate(root: Path):
    """Load the O13 read-only gate from an explicit project root, never cwd."""
    path = project_path(Path(root), O13_AUDIT)
    if not path.is_file():
        raise FileNotFoundError(f"Missing O13 gate: {O13_AUDIT}")
    spec = importlib.util.spec_from_file_location("eot_o13_gate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load O13 gate from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(config: OptimizationConfig, relative: str, value: dict) -> Path:
    path = _final_path(config, relative)
    content = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite differing O14 evidence: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    return path


def _write_frame(config: OptimizationConfig, relative: str, frame: pd.DataFrame,
                 columns: Sequence[str]) -> Path:
    path = _final_path(config, relative)
    expected = frame_digest(frame, columns)
    if path.exists():
        if frame_digest(pd.read_parquet(path), columns) != expected:
            raise ValueError(f"Refusing to overwrite differing O14 frame: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[list(columns)].to_parquet(path, index=False, compression="zstd")
    return path


@dataclass(frozen=True)
class FinalDataContext:
    """S/C memberships and S-fitted final training statistics. T is not stored."""

    outer_fold: int
    search_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    split_sha256: str
    statistics_sha256: str
    _statistics_json: str
    input_sha256: dict

    @property
    def statistics(self) -> dict:
        return json.loads(self._statistics_json)

    @property
    def time_normalizer(self) -> TimeNormalizer:
        return TimeNormalizer.from_dict(self.statistics["time_normalizer"])

    @property
    def base_positive_weight(self) -> float:
        return float(self.statistics["base_positive_weight"])

    def user_ids(self, role: str) -> list[str]:
        if role == "search":
            return list(self.search_ids)
        if role == "calibration":
            return list(self.calibration_ids)
        raise ValueError("Final adapter stores only search and calibration memberships")

    def dataset(self, store: SequenceStore, role: str) -> UserSequenceDataset:
        return UserSequenceDataset(store, self.user_ids(role), self.time_normalizer)


@dataclass(frozen=True)
class FinalTrainingPlan:
    """Read-only plan produced after the O13 gate; no directory is created here."""

    config: OptimizationConfig
    context: FinalDataContext
    identity: object
    parent_lock_sha256: str
    parent_lock_file_sha256: str
    _o13_record_json: str
    _inputs_json: str

    @property
    def o13_record(self) -> dict:
        return json.loads(self._o13_record_json)

    @property
    def binding(self) -> dict:
        context = self.context
        return {
            "schema_version": 1,
            "run_id": self.identity.run_id,
            **self.identity.metadata,
            "o13_parent_lock_sha256": self.parent_lock_sha256,
            "o13_parent_lock_file_sha256": self.parent_lock_file_sha256,
            "o13_final_plan": self.o13_record,
            "outer_split_sha256": context.input_sha256[f"data/splits/fold_{context.outer_fold}.json"],
            "search_user_ids": list(context.search_ids),
            "calibration_user_ids": list(context.calibration_ids),
            "statistics_sha256": context.statistics_sha256,
            "training_statistics": context.statistics,
            "training": self.config.section("training"),
            "loss": self.config.values["loss"],
            "input_sha256": self.input_sha256,
            "training_schedule": "fresh_init_fixed_epochs_no_early_stop_no_c_epoch_selection",
            "test_access": "closed_until_all_five_post_calibration_locks",
        }

    @property
    def input_sha256(self) -> dict:
        return json.loads(self._inputs_json)


def prepare_final_run(config: OptimizationConfig, *, outer_fold: int, seed: int) -> FinalTrainingPlan:
    """Read-only O14 preflight: O13 gate, frozen inputs, S statistics and run identity."""
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in 0..4")
    if seed not in FINAL_SEEDS:
        raise ValueError("seed must be one of 42, 43, 44")
    fresh = load_optimization_config(config.source, project_root=config.project_root)
    if fresh != config:
        raise ValueError("Config object differs from its locked, current file")
    if config.candidate_id != "A":
        raise ValueError("O13 locked A/R2 fallback for every fold; other candidates are not enabled in O14")

    gate = _load_o13_gate(config.project_root)
    record = gate.require_final_plan(outer_fold, config.candidate_id, seed, purpose="training")
    required = {
        "fresh_initialization": True,
        "final_early_stopping": False,
        "checkpoint_transfer": False,
        "statistics_fit_role": "S",
        "config": config.source.resolve().relative_to(config.project_root.resolve()).as_posix(),
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"O13 plan mismatch for {key!r}")
    epochs = record.get("epochs")
    if type(epochs) is not int or not 1 <= epochs <= 50:
        raise ValueError("O13 locked final epochs must be an integer in 1..50")

    lock_path = project_path(config.project_root, O13_LOCK)
    lock_wrapper = _load_json(lock_path)
    if lock_wrapper.get("sha256") != gate.digest(lock_wrapper.get("lock")):
        raise ValueError("O13 final lock wrapper is inconsistent")
    parent_file_sha = file_sha256(lock_path)

    users, posts, outers, input_hashes = _inputs(config)
    outer_relative = f"data/splits/fold_{outer_fold}.json"
    outer = outers[outer_fold]
    if outer.get("config_hash") != config.base_config_hash:
        raise ValueError("Stale original split config hash")
    search_ids = tuple(checked_ids(outer["train_user_ids"], "search"))
    calibration_ids = tuple(checked_ids(outer["validation_user_ids"], "calibration"))
    if set(search_ids) & set(calibration_ids):
        raise ValueError("Search and calibration users must be disjoint")
    inventory = {str(value) for value in users["user_id"]}
    if not set(search_ids) | set(calibration_ids) <= inventory:
        raise ValueError("Final roles contain unknown users")

    # The final statistics are fitted afresh on S and never inherit inner values.
    statistics = fit_training_statistics(
        users, posts, search_ids,
        quantile=float(config.base_values["time"]["clipping_quantile"]),
        max_posts=int(config.base_values["data"]["max_posts"]))
    if list(statistics["fit_user_ids"]) != list(search_ids):
        raise ValueError("Final statistics were not fitted on exactly S")
    statistics_sha = digest(statistics)

    protected = _protected_records(config)
    for name in ("data/embeddings/post_embeddings.npy",
                 "data/embeddings/post_embeddings.meta.json"):
        row = protected.get(name)
        if row is None:
            raise ValueError(f"Missing protected input: {name}")
        actual = file_sha256(project_path(config.project_root, name))
        if actual != row["sha256"]:
            raise ValueError(f"Frozen training input changed: {name}")
        input_hashes[name] = actual

    split_material = {
        "kind": "o14_final_fold",
        "outer_fold": outer_fold,
        "search_user_ids": list(search_ids),
        "calibration_user_ids": list(calibration_ids),
        "outer_split_sha256": input_hashes[outer_relative],
        "statistics_sha256": statistics_sha,
        "o13_parent_lock_sha256": lock_wrapper["sha256"],
    }
    identity = make_run_identity(
        config, stage="final", outer_fold=outer_fold, inner_fold=None, seed=seed,
        code_sha256=source_fingerprint(config.project_root, FINAL_SOURCES),
        split_sha256=digest(split_material))
    context = FinalDataContext(
        outer_fold=outer_fold, search_ids=search_ids,
        calibration_ids=calibration_ids, split_sha256=digest(split_material),
        statistics_sha256=statistics_sha,
        _statistics_json=canonical_json(statistics), input_sha256=dict(input_hashes))
    return FinalTrainingPlan(config, context, identity, lock_wrapper["sha256"],
                             parent_file_sha, canonical_json(record),
                             canonical_json(input_hashes))


# ---------------------------------------------------------------------------
# Final training and checkpoint recovery
# ---------------------------------------------------------------------------


def _commit_final(plan: FinalTrainingPlan, state: dict) -> dict:
    relative = f"checkpoints/state_{state['epoch'] % 2}.pt"
    path = _atomic_write(plan.identity, relative, state, tensor=True)
    manifest = {
        "schema_version": 1,
        "binding": plan.binding,
        "checkpoint": relative,
        "checkpoint_sha256": file_sha256(path),
        "status": state["status"],
        "stop_reason": state["stop_reason"],
        "epoch": state["epoch"],
        "target_epochs": state["target_epochs"],
        "runtime": state["runtime"],
        "positive_weight": state["positive_weight"],
        "history": state["history"],
    }
    _atomic_write(plan.identity, "run.json", manifest)
    return manifest


def read_final_checkpoint(plan: FinalTrainingPlan) -> tuple[dict, dict]:
    """Verify a final run/checkpoint against the O13-bound run identity."""
    manifest_path = plan.identity.artifact_path("run.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("No committed final checkpoint for this exact O13 plan")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("binding") != plan.binding:
        raise ValueError("Final manifest does not match O13/config/source/statistics identity")
    if manifest.get("checkpoint") not in {"checkpoints/state_0.pt", "checkpoints/state_1.pt"}:
        raise ValueError("Unexpected final checkpoint path")
    path = plan.identity.artifact_path(manifest["checkpoint"])
    if file_sha256(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Final checkpoint checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("binding") != plan.binding:
        raise ValueError("Final checkpoint belongs to a different O14 run")
    for key in ("epoch", "target_epochs", "status", "stop_reason", "runtime",
                "positive_weight", "history"):
        if state.get(key) != manifest.get(key):
            raise ValueError(f"Final checkpoint/manifest mismatch: {key}")
    target = plan.o13_record["epochs"]
    epoch = state["epoch"]
    if (type(target) is not int or target != state["target_epochs"] or not 1 <= target <= 50
            or type(epoch) is not int or not 1 <= epoch <= target
            or [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1))):
        raise ValueError("Invalid final training progress")
    if state["status"] == "complete":
        if epoch != target or state["stop_reason"] != "fixed_epochs":
            raise ValueError("Invalid completed final run status")
    elif state["status"] == "paused":
        if epoch >= target or state["stop_reason"] != "paused":
            raise ValueError("Invalid paused final run status")
    else:
        raise ValueError("Final run must be complete or paused")
    return state, manifest


def load_final_model(plan: FinalTrainingPlan, *, device_name: str = "cpu"):
    """Load one complete fixed-epoch final model; never resume or write."""
    state, manifest = read_final_checkpoint(plan)
    if state["status"] != "complete":
        raise ValueError("A complete final checkpoint is required for calibration/evaluation")
    device = torch.device(device_name)
    model = EOTModel.from_config(_legacy_read_config(plan.config)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state, manifest


def _fit_final(plan: FinalTrainingPlan, store: SequenceStore, *, device_name: str | None,
               resume: bool, stop_after_epoch: int | None = None) -> dict:
    """Fresh fixed-epoch S training. No C/E set is instantiated or evaluated."""
    cfg = plan.config.section("training")
    target_epochs = plan.o13_record["epochs"]
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int
            or not 1 <= stop_after_epoch <= target_epochs):
        raise ValueError("stop_after_epoch must lie in 1..O13 locked epochs")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU/CUDA final training is supported")

    with _run_lock(plan.identity):
        run_path = plan.identity.artifact_path("run.json")
        exists = run_path.exists()
        if exists and not resume:
            raise FileExistsError("This final run exists; use resume=True to continue")
        train_ids = plan.context.user_ids("search")
        train_users = pd.DataFrame({
            "user_id": train_ids,
            "label": [int(store.users[user_id].label) for user_id in train_ids],
        })
        loss_fn = build_loss(plan.config, train_users, train_ids).to(device)
        stats = plan.context.statistics
        if (loss_fn.fit_labels_sha256 != stats["fit_labels_sha256"]
                or float(loss_fn.base_positive_weight) != plan.context.base_positive_weight):
            raise ValueError("Loss weights differ from final S statistics")
        train_dataset = plan.context.dataset(store, "search")
        seed = plan.identity.metadata["seed"]
        set_global_seed(seed)
        runtime = _runtime(device)
        state, manifest = read_final_checkpoint(plan) if exists else (None, None)
        if state is not None:
            if state["runtime"] != runtime:
                raise ValueError("Resume runtime differs; exact final continuation requires the same environment")
            if state["status"] == "complete":
                return manifest
            if stop_after_epoch is not None and stop_after_epoch <= state["epoch"]:
                return manifest
        model = EOTModel.from_config(_legacy_read_config(plan.config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"])
        if state is None:
            environment = collect_environment(plan.config.project_root, plan.config.config_hash)
            epoch = 0
            history: list[dict] = []
        else:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            _restore_rng(state["rng_state"])
            environment = state["environment"]
            epoch = state["epoch"]
            history = state["history"]
        limit = target_epochs if stop_after_epoch is None else stop_after_epoch
        for current in range(epoch + 1, limit + 1):
            loader = DataLoader(train_dataset, batch_size=cfg["batch_users"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed + current),
                                num_workers=0, pin_memory=device.type == "cuda")
            trained = train_epoch(model, loader, loss_fn, optimizer, device,
                                  cfg["gradient_clip_norm"])
            if not all(math.isfinite(float(value)) for value in trained.values()):
                raise FloatingPointError("Non-finite final training result; checkpoint not committed")
            history.append({"epoch": current, "train_loss": float(trained["loss"]),
                            "train_final_loss": float(trained["final_loss"]),
                            "train_prefix_loss": float(trained["prefix_loss"])})
            complete = current == target_epochs
            state = {
                "binding": plan.binding,
                "epoch": current,
                "target_epochs": target_epochs,
                "status": "complete" if complete else "paused",
                "stop_reason": "fixed_epochs" if complete else "paused",
                "model_state_dict": _cpu_model_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": _rng_state(device),
                "runtime": runtime,
                "environment": environment,
                "positive_weight": float(loss_fn.positive_weight.item()),
                "history": history,
            }
            manifest = _commit_final(plan, state)
            print(f"final {plan.identity.run_id[:12]} outer={plan.context.outer_fold} "
                  f"seed={seed} epoch={current}/{target_epochs} loss={trained['loss']:.6f} "
                  f"complete={complete}", flush=True)
            if complete:
                break
        return manifest


def train_final_run(config: OptimizationConfig, *, outer_fold: int, seed: int,
                    device_name: str | None = None, resume: bool = False,
                    stop_after_epoch: int | None = None) -> dict:
    """Public O14 entry: preflight the O13 lock, then train one fixed-epoch final run."""
    plan = prepare_final_run(config, outer_fold=outer_fold, seed=seed)
    store = SequenceStore(_legacy_read_config(config))
    return _fit_final(plan, store, device_name=device_name, resume=resume,
                      stop_after_epoch=stop_after_epoch)


# ---------------------------------------------------------------------------
# Model inference on C or T (T is only reachable after the five-lock gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ensemble:
    final: pd.DataFrame
    prefix: pd.DataFrame
    provenance: dict

    def user_ids(self) -> list[str]:
        return [str(value) for value in self.final["user_id"]]

    def labels(self) -> list[int]:
        return [int(value) for value in self.final["label"]]

    def sequences(self):
        grouped = {str(key): frame for key, frame in self.prefix.groupby("user_id", sort=False)}
        return [_risk_sequence(str(row.user_id), int(row.label),
                               grouped[str(row.user_id)]["probability"].to_numpy(dtype=np.float64))
                for row in self.final.itertuples(index=False)]


def _risk_sequence(user_id: str, label: int, probabilities: np.ndarray):
    from eot.metrics import RiskSequence
    return RiskSequence(user_id, label, probabilities)


@torch.inference_mode()
def predict_final_ids(plan: FinalTrainingPlan, store: SequenceStore, user_ids,
                      *, device_name: str = "cpu") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every valid prefix for explicit user IDs; requires a complete final model."""
    ids = checked_ids(user_ids, "final prediction users")
    model, _, _ = load_final_model(plan, device_name=device_name)
    device = next(model.parameters()).device
    dataset = UserSequenceDataset(store, ids, plan.context.time_normalizer)
    loader = DataLoader(dataset, batch_size=plan.config.section("training")["batch_users"],
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    final_rows: list[dict] = []
    prefix_rows: list[dict] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(batch["post_embeddings"], batch["delta_t"], batch["mask"])
        probabilities = torch.sigmoid(output["prefix_logits"]).float().cpu().numpy()
        lengths = batch["length"].detach().cpu().numpy().astype(int)
        labels = batch["label"].detach().cpu().numpy()
        for index, raw_user_id in enumerate(batch["user_id"]):
            user_id = str(raw_user_id)
            length = int(lengths[index])
            label = int(labels[index])
            valid = probabilities[index, :length]
            final_rows.append({"user_id": user_id, "label": label, "length": length,
                               "probability": float(valid[-1])})
            for position, value in enumerate(valid, start=1):
                prefix_rows.append({"user_id": user_id, "label": label, "length": length,
                                    "position": position, "probability": float(value)})
    final = pd.DataFrame(final_rows)[list(FINAL_COLUMNS)].sort_values("user_id", kind="stable").reset_index(drop=True)
    prefix = pd.DataFrame(prefix_rows)[list(PREFIX_COLUMNS)].sort_values(
        ["user_id", "position"], kind="stable").reset_index(drop=True)
    if list(final["user_id"]) != ids or list(prefix["user_id"].drop_duplicates()) != ids:
        raise ValueError("Prediction output does not cover exactly the requested users")
    return final, prefix


def _ensemble_frames(final_frames: Sequence[pd.DataFrame], prefix_frames: Sequence[pd.DataFrame],
                     seeds: Sequence[int]) -> _Ensemble:
    if len(final_frames) != len(seeds) or len(prefix_frames) != len(seeds):
        raise ValueError("One final/prefix frame is required per seed")
    reference = final_frames[0][list(FINAL_COLUMNS)].copy().sort_values(
        "user_id", kind="stable").reset_index(drop=True)
    prefix_reference = prefix_frames[0][list(PREFIX_COLUMNS)].copy().sort_values(
        ["user_id", "position"], kind="stable").reset_index(drop=True)
    final_stack, prefix_stack = [], []
    for seed, final_frame, prefix_frame in zip(seeds, final_frames, prefix_frames):
        current_final = final_frame[list(FINAL_COLUMNS)].copy().sort_values(
            "user_id", kind="stable").reset_index(drop=True)
        current_prefix = prefix_frame[list(PREFIX_COLUMNS)].copy().sort_values(
            ["user_id", "position"], kind="stable").reset_index(drop=True)
        if (list(current_final["user_id"]) != list(reference["user_id"])
                or not np.array_equal(current_final["label"].to_numpy(), reference["label"].to_numpy())
                or not np.array_equal(current_final["length"].to_numpy(), reference["length"].to_numpy())
                or list(current_prefix["user_id"]) != list(prefix_reference["user_id"])
                or not np.array_equal(current_prefix["position"].to_numpy(), prefix_reference["position"].to_numpy())):
            raise ValueError(f"Seed {seed} predictions are not aligned")
        final_stack.append(current_final["probability"].to_numpy(dtype=np.float64))
        prefix_stack.append(current_prefix["probability"].to_numpy(dtype=np.float64))
    ensemble_final = reference.copy()
    ensemble_final["probability"] = np.mean(np.vstack(final_stack), axis=0)
    ensemble_prefix = prefix_reference.copy()
    ensemble_prefix["probability"] = np.mean(np.vstack(prefix_stack), axis=0)
    provenance = {"seeds": list(seeds),
                  "final_sha256": frame_digest(ensemble_final, FINAL_COLUMNS),
                  "prefix_sha256": frame_digest(ensemble_prefix, PREFIX_COLUMNS),
                  "users": len(ensemble_final)}
    return _Ensemble(ensemble_final, ensemble_prefix, provenance)


# ---------------------------------------------------------------------------
# C calibration and sealed post-calibration locks
# ---------------------------------------------------------------------------


def _save_local_bundle(config: OptimizationConfig, relative_directory: str,
                       bundle: PredictionBundle) -> dict:
    _write_frame(config, f"{relative_directory}/final.parquet", bundle.final, FINAL_COLUMNS)
    _write_frame(config, f"{relative_directory}/prefix.parquet", bundle.prefix, PREFIX_COLUMNS)
    wrapper = {"sha256": digest(bundle.provenance), "bundle": bundle.provenance}
    _write_json(config, f"{relative_directory}/bundle.json", wrapper)
    files = {}
    for name in ("final.parquet", "prefix.parquet", "bundle.json"):
        files[name] = file_sha256(_final_path(config, f"{relative_directory}/{name}"))
    return {"relative_directory": relative_directory, "files_sha256": files,
            "provenance_sha256": digest(bundle.provenance)}


def _seed_run_record(plan: FinalTrainingPlan, manifest: dict) -> tuple[SeedRun, dict]:
    run_json = plan.identity.artifact_path("run.json")
    checkpoint = plan.identity.artifact_path(manifest["checkpoint"])
    record = {
        "seed": int(plan.identity.metadata["seed"]),
        "run_id": plan.identity.run_id,
        "run_directory": plan.identity.directory.relative_to(plan.config.project_root).as_posix(),
        "run_json": run_json.relative_to(plan.config.project_root).as_posix(),
        "run_json_sha256": file_sha256(run_json),
        "checkpoint": checkpoint.relative_to(plan.config.project_root).as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "epochs": int(plan.o13_record["epochs"]),
        "statistics_sha256": plan.context.statistics_sha256,
    }
    return SeedRun(record["seed"], record["run_id"], record["run_json_sha256"],
                   record["checkpoint_sha256"]), record


def calibrate_final_fold(config: OptimizationConfig, *, outer_fold: int,
                         device_name: str = "cpu") -> dict:
    """Fit the actual final models' C thresholds and seal one fold's second lock."""
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in 0..4")
    store = SequenceStore(_legacy_read_config(config))
    final_frames, prefix_frames, runs, run_records = [], [], [], []
    for seed in FINAL_SEEDS:
        plan = prepare_final_run(config, outer_fold=outer_fold, seed=seed)
        state, manifest = read_final_checkpoint(plan)
        if state["status"] != "complete" or state["epoch"] != plan.o13_record["epochs"]:
            raise ValueError(f"Final run is not complete for outer {outer_fold} seed {seed}")
        final, prefix = predict_final_ids(
            plan, store, plan.context.user_ids("calibration"), device_name=device_name)
        _write_frame(config, f"calibration/outer_{outer_fold}/seed_{seed}/final.parquet",
                     final, FINAL_COLUMNS)
        _write_frame(config, f"calibration/outer_{outer_fold}/seed_{seed}/prefix.parquet",
                     prefix, PREFIX_COLUMNS)
        seed_run, run_record = _seed_run_record(plan, manifest)
        runs.append(seed_run)
        run_records.append(run_record)
        final_frames.append(final)
        prefix_frames.append(prefix)

    bundle = build_bundle(
        config, stage="final", outer_fold=outer_fold, inner_fold=None,
        role="calibration", seeds=FINAL_SEEDS, runs=runs,
        final_frames=final_frames, prefix_frames=prefix_frames)
    bundle_record = _save_local_bundle(
        config, f"calibration/outer_{outer_fold}/bundle", bundle)

    scope = load_selection_scope(config, role="calibration", outer_fold=outer_fold)
    seed_models = tuple(sorted((seed, record["checkpoint_sha256"])
                               for seed, record in zip(FINAL_SEEDS, run_records)))
    reference = SelectionPredictions(
        candidate_id=config.candidate_id, config_sha256=config.config_hash,
        seed_model_sha256=seed_models, sequences=tuple(bundle.sequences()))
    selection = select_dual_thresholds(config, scope, reference, reference)
    selection_path = selection.save(
        config, f"final/calibration/outer_{outer_fold}/threshold_selection.json")
    evaluation = evaluate_selection(config, selection, a_bundle=bundle)
    summary_path = _write_json(
        config, f"calibration/outer_{outer_fold}/calibration_summary.json",
        evaluation.record)

    manifest = selection.manifest
    r1, r2, candidate = manifest["R1"], manifest["R2"], manifest["candidate"]
    candidate_usable = (
        candidate.get("threshold_final") is not None
        and candidate.get("status") != "infeasible"
        and int(candidate["counts"]["correct"]) >= int(r2["counts"]["correct"])
        and int(candidate["counts"]["tp"]) > int(r2["counts"]["tp"]))
    if config.candidate_id == "A" or not candidate_usable:
        fallback = "A_reference_R2"
        opt_source = "A_reference_R2"
    else:
        fallback = f"candidate_{config.candidate_id}"
        opt_source = f"candidate_{config.candidate_id}"

    lock = {
        "schema_version": 1,
        "task": "O14",
        "outer_fold": outer_fold,
        "selected_candidate": config.candidate_id,
        "o13_parent_lock_sha256": None,  # filled from the first plan below
        "o13_parent_lock_file_sha256": None,
        "final_runs": run_records,
        "selection": {
            "cohort_user_ids_sha256": digest(scope.metadata["user_ids"]),
            "cohort_users": len(scope.metadata["user_ids"]),
        },
        "calibration": {
            "bundle": bundle_record,
            "threshold_selection": {
                "relative_path": selection_path.relative_to(config.project_root).as_posix(),
                "file_sha256": file_sha256(selection_path),
                "content_sha256": digest(manifest),
            },
            "calibration_summary": {
                "relative_path": summary_path.relative_to(config.project_root).as_posix(),
                "file_sha256": file_sha256(summary_path),
                "content_sha256": digest(evaluation.record),
            },
            "R1": r1,
            "R2": r2,
            "candidate": candidate,
            "fallback": fallback,
            "opt_source": opt_source,
        },
        "test_access": "closed_until_all_five_post_calibration_locks_verified",
    }
    # Every final plan in this fold shares the same parent O13 lock.
    first_plan = prepare_final_run(config, outer_fold=outer_fold, seed=FINAL_SEEDS[0])
    lock["o13_parent_lock_sha256"] = first_plan.parent_lock_sha256
    lock["o13_parent_lock_file_sha256"] = first_plan.parent_lock_file_sha256
    lock["calibration"]["class_counts"] = {
        "negative": int(np.sum(bundle.labels() == 0)),
        "positive": int(np.sum(bundle.labels() == 1)),
    }
    _write_json(config, f"locks/outer_{outer_fold}.json",
                {"sha256": digest(lock), "lock": lock})
    return {"status": "pass", "outer_fold": outer_fold, "lock": lock,
            "lock_sha256": digest(lock)}


def _second_lock(config: OptimizationConfig, outer_fold: int) -> dict:
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in 0..4")
    wrapper = _load_json(_final_path(config, f"locks/outer_{outer_fold}.json"))
    lock = wrapper.get("lock")
    if not isinstance(lock, dict) or wrapper.get("sha256") != digest(lock):
        raise ValueError("Post-calibration lock wrapper is inconsistent")
    if lock.get("task") != "O14" or lock.get("outer_fold") != outer_fold:
        raise ValueError("Post-calibration lock identity differs")
    if lock.get("test_access") != "closed_until_all_five_post_calibration_locks_verified":
        raise ValueError("Post-calibration lock does not keep test access closed")
    for record in lock.get("final_runs", []):
        if file_sha256(project_path(config.project_root, record["run_json"])) != record["run_json_sha256"]:
            raise ValueError("Final run manifest changed after calibration")
        if file_sha256(project_path(config.project_root, record["checkpoint"])) != record["checkpoint_sha256"]:
            raise ValueError("Final checkpoint changed after calibration")
    calibration = lock["calibration"]
    for name in ("threshold_selection", "calibration_summary"):
        row = calibration[name]
        if file_sha256(project_path(config.project_root, row["relative_path"])) != row["file_sha256"]:
            raise ValueError(f"Calibration evidence changed: {name}")
    for filename, expected in calibration["bundle"]["files_sha256"].items():
        if file_sha256(_final_path(config, f"{calibration['bundle']['relative_directory']}/{filename}")) != expected:
            raise ValueError("Calibration bundle changed after locking")
    scope = load_selection_scope(config, role="calibration", outer_fold=outer_fold)
    if digest(scope.metadata["user_ids"]) != lock["selection"]["cohort_user_ids_sha256"]:
        raise ValueError("Calibration cohort membership changed")
    return lock


def require_outer_test_gate(config: OptimizationConfig) -> list[dict]:
    """Return all five second locks only after each is current and structurally valid."""
    locks = [_second_lock(config, outer_fold) for outer_fold in range(5)]
    if sorted(lock["outer_fold"] for lock in locks) != list(range(5)):
        raise ValueError("Second-lock folds are incomplete or duplicated")
    gate = _load_o13_gate(config.project_root)
    for lock in locks:
        outer_fold = lock["outer_fold"]
        expected_plan = gate.require_final_plan(
            outer_fold, config.candidate_id, lock["final_runs"][0]["seed"], purpose="training")
        if expected_plan["epochs"] != lock["final_runs"][0]["epochs"]:
            raise ValueError("Second lock no longer agrees with the O13 plan")
    return locks


def load_outer_test_cohorts(config: OptimizationConfig) -> list[tuple[int, list[str], str]]:
    """Read only frozen T membership, after every second lock has verified."""
    require_outer_test_gate(config)
    protected = _protected_records(config)
    cohorts = []
    seen: set[str] = set()
    for outer_fold in range(5):
        relative = f"data/splits/fold_{outer_fold}.json"
        row = protected.get(relative)
        if row is None:
            raise ValueError(f"Outer split is not protected: {relative}")
        path = project_path(config.project_root, relative)
        actual = file_sha256(path)
        if actual != row["sha256"]:
            raise ValueError("Protected outer split changed")
        outer = json.loads(path.read_text(encoding="utf-8"))
        ids = checked_ids(outer["test_user_ids"], "outer_test")
        if seen & set(ids):
            raise ValueError("Outer-test users overlap across folds")
        seen |= set(ids)
        cohorts.append((outer_fold, ids, actual))
    if len(seen) != 909:
        raise ValueError(f"Outer-test OOF cohort must contain 909 unique users, found {len(seen)}")
    return cohorts


# ---------------------------------------------------------------------------
# One outer-test evaluation, opened only by all five second locks
# ---------------------------------------------------------------------------


def evaluate_final_test(config: OptimizationConfig, *, device_name: str = "cpu") -> dict:
    """Run the single authorized outer-test evaluation after the five-lock gate."""
    cohorts = load_outer_test_cohorts(config)
    locks = require_outer_test_gate(config)
    store = SequenceStore(_legacy_read_config(config))
    per_fold = {name: [] for name in ("R1", "R2", "OPT")}
    fold_records = []
    per_seed_metrics = {}

    for outer_fold, test_ids, outer_sha in cohorts:
        lock = locks[outer_fold]
        seed_finals, seed_prefixes = [], []
        seed_frames_sha = {}
        for seed in FINAL_SEEDS:
            plan = prepare_final_run(config, outer_fold=outer_fold, seed=seed)
            state, manifest = read_final_checkpoint(plan)
            if state["status"] != "complete":
                raise ValueError("Final model is incomplete; outer test remains closed")
            run_record = next((row for row in lock["final_runs"] if row["seed"] == seed), None)
            if run_record is None or run_record["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                raise ValueError("Final checkpoint differs from the sealed second lock")
            final, prefix = predict_final_ids(plan, store, test_ids, device_name=device_name)
            _write_frame(config, f"evaluation/outer_{outer_fold}/seed_{seed}/final.parquet",
                         final, FINAL_COLUMNS)
            _write_frame(config, f"evaluation/outer_{outer_fold}/seed_{seed}/prefix.parquet",
                         prefix, PREFIX_COLUMNS)
            seed_finals.append(final)
            seed_prefixes.append(prefix)
            seed_frames_sha[str(seed)] = {
                "final_sha256": frame_digest(final, FINAL_COLUMNS),
                "prefix_sha256": frame_digest(prefix, PREFIX_COLUMNS),
            }
        ensemble = _ensemble_frames(seed_finals, seed_prefixes, FINAL_SEEDS)
        _write_frame(config, f"evaluation/outer_{outer_fold}/bundle/final.parquet",
                     ensemble.final, FINAL_COLUMNS)
        _write_frame(config, f"evaluation/outer_{outer_fold}/bundle/prefix.parquet",
                     ensemble.prefix, PREFIX_COLUMNS)

        calibration = lock["calibration"]
        decisions = {
            "R1": decision_frame(
                ensemble,
                threshold_final=calibration["R1"]["threshold_final"],
                threshold_early=calibration["R1"]["threshold_early"]),
            "R2": decision_frame(
                ensemble,
                threshold_final=calibration["R2"]["threshold_final"],
                threshold_early=calibration["R2"]["threshold_early"]),
            "OPT": decision_frame(
                ensemble,
                threshold_final=calibration["candidate"]["threshold_final"],
                threshold_early=calibration["candidate"]["threshold_early"])
            if calibration["opt_source"] != "A_reference_R2"
            else None,
        }
        if decisions["OPT"] is None:
            decisions["OPT"] = decisions["R2"].copy()
        for name, frame in decisions.items():
            _write_frame(config, f"evaluation/outer_{outer_fold}/decisions/{name}.parquet",
                         frame, DECISION_COLUMNS)
            per_fold[name].append(frame)
        fold_seed_metrics = {}
        for seed, final, prefix in zip(FINAL_SEEDS, seed_finals, seed_prefixes):
            seed_ensemble = _Ensemble(
                final, prefix, {"seeds": [seed], "final_sha256": seed_frames_sha[str(seed)]["final_sha256"],
                                "prefix_sha256": seed_frames_sha[str(seed)]["prefix_sha256"],
                                "users": len(final)})
            seed_ensemble.provenance["seed_frames_sha256"] = seed_frames_sha[str(seed)]
            seed_frame = decision_frame(
                seed_ensemble,
                threshold_final=calibration["R2"]["threshold_final"],
                threshold_early=calibration["R2"]["threshold_early"])
            fold_seed_metrics[str(seed)] = evaluate_frame(seed_frame)
        per_seed_metrics[str(outer_fold)] = fold_seed_metrics
        fold_records.append({
            "outer_fold": outer_fold,
            "test_user_ids_sha256": digest(test_ids),
            "test_users": len(test_ids),
            "outer_split_sha256": outer_sha,
            "bundle_sha256": {
                "final": frame_digest(ensemble.final, FINAL_COLUMNS),
                "prefix": frame_digest(ensemble.prefix, PREFIX_COLUMNS),
            },
            "decisions_sha256": {name: frame_digest(frame, DECISION_COLUMNS)
                                 for name, frame in decisions.items()},
            "metrics": {name: evaluate_frame(frame) for name, frame in decisions.items()},
        })

    frames = {}
    for name in ("R1", "R2", "OPT"):
        pooled = pool_fold_frames(per_fold[name], roles=["outer_test"] * len(per_fold[name]),
                                  name=f"OOF_{name}")
        frames[name] = pooled
        _write_frame(config, f"evaluation/oof_{name}.parquet", pooled, DECISION_COLUMNS)

    r0_frame = legacy_oof_decision_frame(config)
    if (list(r0_frame["user_id"]) == list(frames["OPT"]["user_id"])
            and np.array_equal(r0_frame["label"].to_numpy(), frames["OPT"]["label"].to_numpy())):
        frames["R0"] = r0_frame
        _write_frame(config, "evaluation/oof_R0_readonly.parquet", r0_frame, DECISION_COLUMNS)

    comparison_frames = {name: frame for name, frame in frames.items()}
    comparisons = compare_objects(comparison_frames)
    objects = {name: evaluate_frame(frame) for name, frame in frames.items()}
    summary = {
        "task": "O14",
        "status": "pass",
        "outer_test_gate": "all_five_post_calibration_locks_verified",
        "users": int(len(frames["OPT"])),
        "unique_users": int(frames["OPT"]["user_id"].nunique()),
        "objects": objects,
        "per_fold": fold_records,
        "per_seed": per_seed_metrics,
        "comparisons": comparisons,
        "legacy_oof_replay": legacy_oof_recompute(config),
        "uncertainty_caveat": (
            "conditional on fitted final models, O13 selection and locked C thresholds; "
            "does not include retraining or reselection uncertainty"),
    }
    _write_json(config, "evaluation/o14_evaluation_summary.json", summary)
    return summary


def final_run_status(config: OptimizationConfig, *, outer_fold: int, seed: int) -> dict:
    """Cheap controller-facing status; never creates a run or reads test data."""
    try:
        plan = prepare_final_run(config, outer_fold=outer_fold, seed=seed)
    except FileNotFoundError:
        return {"complete": False, "reason": "not_started"}
    run_path = plan.identity.artifact_path("run.json")
    if not run_path.is_file():
        return {"complete": False, "reason": "not_started",
                "run_id": plan.identity.run_id,
                "directory": plan.identity.directory.relative_to(config.project_root).as_posix()}
    try:
        state, manifest = read_final_checkpoint(plan)
    except Exception as exc:
        return {"complete": False, "reason": "invalid", "error": f"{type(exc).__name__}: {exc}",
                "run_id": plan.identity.run_id,
                "directory": plan.identity.directory.relative_to(config.project_root).as_posix()}
    return {
        "complete": state["status"] == "complete",
        "reason": state["status"],
        "run_id": plan.identity.run_id,
        "directory": plan.identity.directory.relative_to(config.project_root).as_posix(),
        "epoch": state["epoch"],
        "target_epochs": state["target_epochs"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "latest_train_loss": state["history"][-1]["train_loss"] if state["history"] else None,
    }
