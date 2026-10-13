"""Validate all formal training runs and write a compact completion summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eot.config import load_config
from eot.model import EOTModel
from eot.reproducibility import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    rows: list[dict[str, object]] = []
    seeds = [int(value) for value in config.section("training")["seeds"]]
    folds = range(int(config.section("cross_validation")["n_splits"]))
    expected_validation_users = 0
    for fold in folds:
        split = json.loads(
            (config.path("splits_dir") / f"fold_{fold}.json").read_text(encoding="utf-8")
        )
        validation_users = len(split["validation_user_ids"])
        expected_validation_users += validation_users * len(seeds)
        for seed in seeds:
            run_name = f"fold_{fold}_seed_{seed}"
            run_path = config.path("outputs_dir") / "logs" / run_name / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run.get("status") != "complete" or run.get("config_hash") != config.config_hash:
                raise ValueError(f"Invalid or stale run manifest: {run_path}")
            checkpoint_path = config.project_root / str(run["best_checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("config_hash") != config.config_hash:
                raise ValueError(f"Stale checkpoint: {checkpoint_path}")
            model = EOTModel.from_config(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            prediction_path = config.project_root / str(run["validation_predictions"])
            predictions = pd.read_parquet(prediction_path)
            if len(predictions) != validation_users:
                raise ValueError(f"Wrong validation row count in {prediction_path}")
            if not predictions["probability"].between(0.0, 1.0).all():
                raise ValueError(f"Invalid probabilities in {prediction_path}")
            if (checkpoint_path.parent / "last.pt").exists():
                raise ValueError(f"Completed run retained last.pt: {checkpoint_path.parent}")
            rows.append(
                {
                    "run_name": run_name,
                    "fold": fold,
                    "seed": seed,
                    "device": run["device"],
                    "epochs_completed": run["epochs_completed"],
                    "best_epoch": run["best_epoch"],
                    "best_validation_pr_auc": run["best_validation_pr_auc"],
                    "elapsed_seconds": run["elapsed_seconds"],
                    "train_users": run["train_users"],
                    "validation_users": run["validation_users"],
                }
            )
    frame = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    if len(frame) != 15 or set(frame["device"]) != {"cuda"}:
        raise ValueError("Expected exactly 15 completed CUDA runs")
    metrics_dir = config.path("outputs_dir") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "training_run_summary.csv", index=False)
    report = {
        "status": "pass",
        "config_hash": config.config_hash,
        "completed_runs": len(frame),
        "expected_runs": 15,
        "all_devices_cuda": True,
        "all_checkpoints_reload": True,
        "all_validation_predictions_valid": True,
        "temporary_last_checkpoints_remaining": 0,
        "validation_prediction_rows": int(frame["validation_users"].sum()),
        "expected_validation_prediction_rows": expected_validation_users,
        "elapsed_seconds_total": float(frame["elapsed_seconds"].sum()),
        "elapsed_minutes_total": float(frame["elapsed_seconds"].sum() / 60.0),
        "best_validation_pr_auc_mean": float(frame["best_validation_pr_auc"].mean()),
        "best_validation_pr_auc_std": float(frame["best_validation_pr_auc"].std(ddof=1)),
        "best_epoch_median": float(np.median(frame["best_epoch"])),
        "runs": frame.to_dict(orient="records"),
    }
    write_json(report, config.path("outputs_dir") / "reports" / "training_completion.json")
    print(json.dumps({key: report[key] for key in report if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
