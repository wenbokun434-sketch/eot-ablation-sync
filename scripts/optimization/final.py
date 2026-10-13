"""O14 CLI: fixed-epoch final training, C calibration, locked outer-test evaluation.

Run from the project root. Training/calibration never reads outer-test labels;
evaluate refuses to start until all five second locks verify.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from eot.optimization.config import load_optimization_config
from eot.optimization.final import (calibrate_final_fold, evaluate_final_test,
                                     final_run_status, prepare_final_run,
                                     require_outer_test_gate, train_final_run)


def _config(args):
    return load_optimization_config(args.config, project_root=Path(args.root).resolve())


def _emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="project root (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="read-only O13 gate and final-plan preflight")
    prepare.add_argument("--config", default="configs/optimization/A.yaml")
    prepare.add_argument("--outer-fold", type=int, required=True)
    prepare.add_argument("--seed", type=int, required=True)

    train = sub.add_parser("train", help="train or resume one fixed-epoch final run")
    train.add_argument("--config", default="configs/optimization/A.yaml")
    train.add_argument("--outer-fold", type=int, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default=None)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-epoch", type=int, default=None)

    calibrate = sub.add_parser("calibrate", help="C calibration and one sealed second lock")
    calibrate.add_argument("--config", default="configs/optimization/A.yaml")
    calibrate.add_argument("--outer-fold", type=int, required=True)
    calibrate.add_argument("--device", default="cpu")

    evaluate = sub.add_parser("evaluate", help="one outer-test evaluation after all five locks")
    evaluate.add_argument("--config", default="configs/optimization/A.yaml")
    evaluate.add_argument("--device", default="cpu")

    status = sub.add_parser("status", help="read-only status of one final run")
    status.add_argument("--config", default="configs/optimization/A.yaml")
    status.add_argument("--outer-fold", type=int, required=True)
    status.add_argument("--seed", type=int, required=True)

    gate = sub.add_parser("verify-gate", help="verify all five post-calibration locks")
    gate.add_argument("--config", default="configs/optimization/A.yaml")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    config = _config(args)
    if args.command == "prepare":
        plan = prepare_final_run(config, outer_fold=args.outer_fold, seed=args.seed)
        _emit({"status": "pass", "task": "O14", "run_id": plan.identity.run_id,
               "run_directory": plan.identity.directory.relative_to(root).as_posix(),
               "outer_fold": args.outer_fold, "seed": args.seed,
               "search_users": len(plan.context.search_ids),
               "calibration_users": len(plan.context.calibration_ids),
               "epochs": plan.o13_record["epochs"],
               "statistics_sha256": plan.context.statistics_sha256,
               "o13_parent_lock_sha256": plan.parent_lock_sha256})
    elif args.command == "train":
        manifest = train_final_run(config, outer_fold=args.outer_fold, seed=args.seed,
                                   device_name=args.device, resume=args.resume,
                                   stop_after_epoch=args.stop_after_epoch)
        _emit({"status": manifest["status"], "task": "O14",
               "outer_fold": args.outer_fold, "seed": args.seed,
               "run_id": manifest["binding"]["run_id"],
               "epoch": manifest["epoch"], "target_epochs": manifest["target_epochs"],
               "checkpoint": manifest["checkpoint"],
               "checkpoint_sha256": manifest["checkpoint_sha256"]})
    elif args.command == "calibrate":
        result = calibrate_final_fold(config, outer_fold=args.outer_fold,
                                      device_name=args.device)
        _emit({"status": result["status"], "task": "O14",
               "outer_fold": result["outer_fold"], "lock_sha256": result["lock_sha256"]})
    elif args.command == "evaluate":
        summary = evaluate_final_test(config, device_name=args.device)
        _emit({"status": summary["status"], "task": "O14",
               "users": summary["users"], "unique_users": summary["unique_users"],
               "objects": sorted(summary["objects"]),
               "summary": "outputs/optimization/v1/final/evaluation/o14_evaluation_summary.json"})
    elif args.command == "status":
        _emit(final_run_status(config, outer_fold=args.outer_fold, seed=args.seed))
    elif args.command == "verify-gate":
        locks = require_outer_test_gate(config)
        _emit({"status": "pass", "task": "O14", "locks": len(locks),
               "outer_folds": [lock["outer_fold"] for lock in locks]})
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
