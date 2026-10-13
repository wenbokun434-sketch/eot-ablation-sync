"""Probe the continuous-time ODE block of trained EOT checkpoints (inference only).

This script never trains and never modifies the model. It replays the formal
checkpoints on their own outer-test users and records, for every user:

* the per-post ODE propagation magnitude ``||z_after - z_before||`` (GRU excluded);
* the normalized time gap ``d_i`` that produced it;
* the RK4 sub-step path length, so a curved trajectory can be told from a straight one;
* a counterfactual replay in which every gap is replaced by one constant value while
  post content is held identical, which isolates the contribution of timing.

``raw_rk4`` re-implements the fixed-step RK4 integration torchdiffeq performs for
``step_size=0.25`` and asserts it matches the model's own ``_propagate`` output, so the
recorded trajectories are provably the ones the model computes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eot.config import ExperimentConfig
from eot.dataset import SequenceStore, UserSequenceDataset
from eot.evaluation import _file_sha256
from eot.model import EOTModel, _DurationScaledODEFunc
from eot.notebook_support import notebook_config
from eot.ode_analysis import validate_probe_records
from eot.reproducibility import write_json
from eot.time_features import load_time_normalizer


def _load_split(config: ExperimentConfig, fold: int) -> dict[str, Any]:
    path = config.path("splits_dir") / f"fold_{fold}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config.config_hash or int(payload.get("fold", -1)) != fold:
        raise ValueError(f"Stale or mismatched split file for fold {fold}")
    return payload


def _load_checkpoint_model(
    config: ExperimentConfig, fold: int, seed: int, device: torch.device
) -> tuple[EOTModel, dict[str, Any], Path]:
    path = config.path("outputs_dir") / "checkpoints" / f"fold_{fold}_seed_{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing formal checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = (config.config_hash, fold, seed)
    observed = (
        checkpoint.get("config_hash"),
        int(checkpoint.get("fold", -1)),
        int(checkpoint.get("seed", -1)),
    )
    if observed != expected:
        raise ValueError(f"checkpoint identity mismatch for {path}: {observed} != {expected}")
    model = EOTModel.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint, path


def raw_rk4(
    ode_func: torch.nn.Module,
    state: torch.Tensor,
    duration: torch.Tensor,
    step_size: float,
    *,
    collect_substeps: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Fixed-step RK4 on tau in [0, 1] with the duration-scaled vector field."""

    scaled = _DurationScaledODEFunc(ode_func, duration)
    n_steps = int(round(1.0 / step_size))
    if not np.isclose(n_steps * step_size, 1.0):
        raise ValueError(f"step_size={step_size} does not tile [0, 1] exactly")
    zero = torch.zeros((), device=state.device, dtype=state.dtype)
    current = state
    substeps: list[torch.Tensor] = [current] if collect_substeps else []
    for _ in range(n_steps):
        k1 = scaled(zero, current)
        k2 = scaled(zero, current + 0.5 * step_size * k1)
        k3 = scaled(zero, current + 0.5 * step_size * k2)
        k4 = scaled(zero, current + step_size * k3)
        current = current + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if collect_substeps:
            substeps.append(current)
    return current, (substeps if collect_substeps else None)


def replay_encoder(
    model: EOTModel,
    embeddings: torch.Tensor,
    delta_t: torch.Tensor,
    mask: torch.Tensor,
    *,
    verify_reference: bool,
) -> dict[str, torch.Tensor]:
    """Replay ``ContinuousTimeEncoder`` while recording per-step ODE diagnostics."""

    encoder = model.continuous_time
    batch, length, _ = embeddings.shape
    valid = mask.bool()
    content = model.projection(embeddings)
    state = encoder.initial_state.unsqueeze(0).expand(batch, -1).contiguous()

    displacement = torch.zeros((batch, length), dtype=state.dtype, device=state.device)
    path_length = torch.zeros((batch, length), dtype=state.dtype, device=state.device)
    states = torch.zeros((batch, length, state.shape[-1]), dtype=state.dtype, device=state.device)
    max_reference_error = 0.0

    for index in range(length):
        duration = delta_t[:, index].unsqueeze(-1)
        before = state
        propagated, substeps = raw_rk4(
            encoder.ode_func, state, duration, encoder.step_size, collect_substeps=True
        )
        if verify_reference:
            reference = encoder._propagate(before, duration)
            error = float((propagated - reference).abs().max().item())
            scale = max(float((reference - before).abs().max().item()), 1e-8)
            max_reference_error = max(max_reference_error, error / scale)
            if error / scale > 1e-3:
                raise AssertionError(
                    f"raw RK4 replay diverges from model._propagate by {error:.3e} "
                    f"({error / scale:.3e} of the propagation magnitude)"
                )
        assert substeps is not None
        displacement[:, index] = (propagated - before).norm(dim=-1)
        stacked = torch.stack(substeps, dim=0)
        path_length[:, index] = (stacked[1:] - stacked[:-1]).norm(dim=-1).sum(dim=0)
        updated = encoder.gru(content[:, index], propagated)
        state = torch.where(valid[:, index].unsqueeze(-1), updated, state)
        states[:, index] = state

    return {
        "states": states,
        "displacement": displacement,
        "path_length": path_length,
        "max_reference_error": torch.tensor(max_reference_error, device=state.device),
    }


def _user_frame(
    *,
    user_id: str,
    fold: int,
    seed: int,
    label: int,
    length: int,
    probability: float,
    real: dict[str, torch.Tensor],
    counterfactual: dict[str, torch.Tensor],
    delta_t: torch.Tensor,
    constant: float,
) -> dict[str, Any]:
    real_final = real["states"][0, length - 1]
    fake_final = counterfactual["states"][0, length - 1]
    shift = float((fake_final - real_final).norm().item())
    denominator = float(real_final.norm().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            real_final.unsqueeze(0), fake_final.unsqueeze(0)
        ).item()
    )
    return {
        "user_id": user_id,
        "fold": fold,
        "seed": seed,
        "label": label,
        "length": length,
        "probability": probability,
        "mean_displacement": float(real["displacement"][0, :length].mean().item()),
        "max_displacement": float(real["displacement"][0, :length].max().item()),
        "mean_delta": float(delta_t[0, :length].mean().item()),
        "max_delta": float(delta_t[0, :length].max().item()),
        "mean_path_length": float(real["path_length"][0, :length].mean().item()),
        "mean_state_norm": denominator,
        "counterfactual_constant": constant,
        "counterfactual_final_shift": shift,
        "counterfactual_final_relative_shift": shift / max(denominator, 1e-8),
        "counterfactual_final_cosine": cosine,
        "counterfactual_mean_displacement": float(
            counterfactual["displacement"][0, :length].mean().item()
        ),
    }


def collect(
    config: ExperimentConfig,
    device: torch.device,
    checkpoints: str,
    limit: int | None = None,
) -> dict[str, Any]:
    folds = list(range(int(config.section("cross_validation")["n_splits"])))
    if limit is not None:
        folds = folds[:1]
    seeds = [int(value) for value in config.section("training")["seeds"]]
    if checkpoints == "seed42":
        seeds = seeds[:1]
    batch_users = int(config.section("training")["batch_users"])
    store = SequenceStore(config)
    step_rows: list[pd.DataFrame] = []
    user_rows: list[pd.DataFrame] = []
    errors: list[float] = []
    checkpoint_records: list[dict[str, Any]] = []
    all_users: set[str] = set()
    state_frames: list[np.ndarray] = []
    state_meta: list[pd.DataFrame] = []

    for fold in folds:
        split = _load_split(config, fold)
        test_users = [str(value) for value in split["test_user_ids"]]
        if limit is not None:
            test_users = test_users[:limit]
        all_users.update(test_users)
        normalizer = load_time_normalizer(config, fold)
        dataset = UserSequenceDataset(store, test_users, normalizer)
        loader = DataLoader(dataset, batch_size=batch_users, shuffle=False, num_workers=0)
        graded = np.concatenate(
            [
                dataset[index]["delta_t"].numpy()[1 : int(dataset[index]["length"])]
                for index in range(len(dataset))
            ]
        )
        constant = float(np.median(graded))
        if not constant > 0.0:
            raise AssertionError("counterfactual constant must be positive")

        for seed in seeds:
            model, checkpoint, path = _load_checkpoint_model(config, fold, seed, device)
            model.eval()
            verify_first_batch = True
            for batch in loader:
                user_ids = [str(value) for value in batch["user_id"]]
                embeddings = batch["post_embeddings"].to(device)
                delta_t = batch["delta_t"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].numpy().astype(np.int8)
                lengths = batch["length"].numpy().astype(np.int16)

                # Verify the raw RK4 replay against the model's own integrator on the
                # first batch only; both are deterministic, so later batches would add
                # nothing but cost 33% more ODE evaluations.
                with torch.enable_grad():
                    real = replay_encoder(
                        model, embeddings, delta_t, mask, verify_reference=verify_first_batch
                    )
                verify_first_batch = False
                errors.append(float(real["max_reference_error"].item()))

                with torch.no_grad():
                    final_logits = model(embeddings, delta_t, mask)["final_logits"]
                probabilities = torch.sigmoid(final_logits).detach().cpu().numpy()

                # Cache the GRU-updated states so case trajectories can be drawn later
                # without running the model a second time.
                padded = torch.full(
                    real["states"].shape,
                    float("nan"),
                    dtype=real["states"].dtype,
                    device=real["states"].device,
                )
                for row, length in enumerate(lengths):
                    padded[row, : int(length)] = real["states"][row, : int(length)]
                state_frames.append(padded.detach().cpu().numpy().astype(np.float16))
                state_meta.append(
                    pd.DataFrame(
                        {
                            "user_id": user_ids,
                            "fold": fold,
                            "seed": seed,
                            "label": labels.astype(np.int8),
                            "length": lengths.astype(np.int16),
                        }
                    )
                )

                counterfactual_delta = delta_t.clone()
                counterfactual_delta[mask.bool()] = constant
                counterfactual_delta[:, 0] = 0.0
                counterfactual = replay_encoder(
                    model, embeddings, counterfactual_delta, mask, verify_reference=False
                )

                for row, user_id in enumerate(user_ids):
                    length = int(lengths[row])
                    user_rows.append(
                        pd.DataFrame(
                            [
                                _user_frame(
                                    user_id=user_id,
                                    fold=fold,
                                    seed=seed,
                                    label=int(labels[row]),
                                    length=length,
                                    probability=float(probabilities[row]),
                                    real={
                                        key: value[row : row + 1]
                                        for key, value in real.items()
                                        if value.ndim >= 2
                                    },
                                    counterfactual={
                                        key: value[row : row + 1]
                                        for key, value in counterfactual.items()
                                        if value.ndim >= 2
                                    },
                                    delta_t=delta_t[row : row + 1].detach().cpu(),
                                    constant=constant,
                                )
                            ]
                        )
                    )
                    step_rows.append(
                        pd.DataFrame(
                            {
                                "user_id": user_id,
                                "fold": fold,
                                "seed": seed,
                                "label": int(labels[row]),
                                "position": np.arange(1, length + 1, dtype=np.int16),
                                "delta_t": delta_t[row, :length].detach().cpu().numpy(),
                                "displacement": real["displacement"][row, :length]
                                .detach()
                                .cpu()
                                .numpy(),
                                "path_length": real["path_length"][row, :length]
                                .detach()
                                .cpu()
                                .numpy(),
                            }
                        )
                    )
            checkpoint_records.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "path": str(path.relative_to(config.project_root)),
                    "sha256": _file_sha256(path),
                    "best_epoch": int(checkpoint.get("epoch", -1)),
                    "validation_pr_auc": float(
                        checkpoint.get("validation_pr_auc", float("nan"))
                    ),
                    "test_users": len(test_users),
                }
            )
            print(
                f"[probe] fold={fold} seed={seed} users={len(test_users)} "
                f"max_rk4_error={max(errors):.2e}",
                flush=True,
            )

    steps = pd.concat(step_rows, ignore_index=True)
    users = pd.concat(user_rows, ignore_index=True)
    for column in ("delta_t", "displacement", "path_length"):
        steps[column] = steps[column].astype(np.float32)

    graded_steps = steps.loc[steps.position > 1]
    correlation = float(
        pd.Series(graded_steps.delta_t).corr(
            pd.Series(graded_steps.displacement), method="spearman"
        )
    )
    summary = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash,
        "checkpoints": checkpoint_records,
        "max_raw_rk4_vs_model_error": max(errors),
        "rk4_check": (
            "max |raw_rk4 - model._propagate| relative to the propagation magnitude; "
            "float32 accumulation only"
        ),
        "partial_run": limit is not None,
        "rows": {
            "step_level": int(len(steps)),
            "user_level": int(len(users)),
            "distinct_users": len(all_users),
        },
        "gap_distribution": {
            "graded_rows": int(len(graded_steps)),
            "share_delta_below_0.01": float((graded_steps.delta_t < 0.01).mean()),
            "share_delta_below_0.05": float((graded_steps.delta_t < 0.05).mean()),
            "share_delta_above_0.5": float((graded_steps.delta_t > 0.5).mean()),
            "median_delta": float(graded_steps.delta_t.median()),
            "quantiles_delta": {
                str(q): float(graded_steps.delta_t.quantile(q))
                for q in (0.25, 0.5, 0.75, 0.95, 0.99)
            },
            "quantiles_displacement": {
                str(q): float(graded_steps.displacement.quantile(q))
                for q in (0.25, 0.5, 0.75, 0.95, 0.99)
            },
            "median_state_norm": float(users.mean_state_norm.median()),
            "median_relative_displacement": float(
                graded_steps.displacement.median() / max(float(users.mean_state_norm.median()), 1e-8)
            ),
            "quantiles_path_over_displacement": {
                str(q): float(
                    (graded_steps.path_length / graded_steps.displacement.clip(lower=1e-12))
                    .quantile(q)
                )
                for q in (0.25, 0.5, 0.75, 0.95)
            },
            "share_delta_below_1e_6": float((graded_steps.delta_t < 1e-6).mean()),
            "share_displacement_below_1e_3": float(
                (graded_steps.displacement < 1e-3).mean()
            ),
            "spearman_delta_vs_displacement": correlation,
            "spearman_delta_vs_path_length": float(
                pd.Series(graded_steps.delta_t).corr(
                    pd.Series(graded_steps.path_length), method="spearman"
                )
            ),
        },
        "counterfactual": {
            "median_final_relative_shift": float(
                users.counterfactual_final_relative_shift.median()
            ),
            "median_final_cosine": float(users.counterfactual_final_cosine.median()),
            "median_mean_displacement_ratio": float(
                (
                    users.counterfactual_mean_displacement
                    / users.mean_displacement.clip(lower=1e-12)
                ).median()
            ),
        },
        "coverage": {
            "test_users": len(all_users),
            "oof_users": int(
                len(
                    pd.read_parquet(
                        config.path("outputs_dir") / "predictions/oof_final_predictions.parquet",
                        columns=["user_id"],
                    )
                )
            )
            if (config.path("outputs_dir") / "predictions/oof_final_predictions.parquet").is_file()
            else None,
        },
    }
    return {
        "steps": steps,
        "users": users,
        "summary": summary,
        "state_frames": state_frames,
        "state_meta": state_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoints", default="all", choices=["all", "seed42"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = notebook_config(Path(".").resolve())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = collect(config, device, args.checkpoints, args.limit)
    out_dir = Path(args.out_dir) if args.out_dir else config.path("outputs_dir") / "ode_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["steps"].to_parquet(out_dir / "ode_step_records.parquet", index=False)
    result["users"].to_parquet(out_dir / "ode_user_records.parquet", index=False)
    np.save(
        out_dir / "ode_state_cache.npy", np.concatenate(result["state_frames"], axis=0)
    )
    pd.concat(result["state_meta"], ignore_index=True).to_parquet(
        out_dir / "ode_state_index.parquet", index=False
    )
    write_json(result["summary"], out_dir / "ode_probe_summary.json")
    report = validate_probe_records(config, out_dir)
    write_json(report, config.path("outputs_dir") / "reports" / "ode_probe_validation.json")
    if report["status"] != "pass":
        raise AssertionError(f"probe validation failed: {report['failures']}")

    summary = result["summary"]
    print(json.dumps(summary["gap_distribution"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["counterfactual"], indent=2, ensure_ascii=False))
    print(f"max |raw RK4 - model._propagate| = {summary['max_raw_rk4_vs_model_error']:.3e}")


if __name__ == "__main__":
    main()
