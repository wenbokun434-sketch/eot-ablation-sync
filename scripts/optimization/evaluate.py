"""Evaluate or compare optimization objects; never selects a threshold or seed.

Subcommands
-----------
legacy    Replay the frozen original OOF metrics from the protected predictions.
validate  Reload and re-check one saved prediction bundle directory.
evaluate  Score one locked threshold record over its own prediction bundles.
compare   Paired user bootstrap between two saved decision frames.
"""
import argparse
import json
import math
import sys

import pandas as pd

from eot.optimization.comparison import compare_frames
from eot.optimization.config import canonical_json, load_optimization_config, project_path
from eot.optimization.decisions import ThresholdSelection
from eot.optimization.evaluation import (DECISION_COLUMNS, evaluate_frame, legacy_oof_recompute,
                                         load_bundle, validate_bundle_artifacts)
from eot.optimization.splits import digest

REPORT_ROOT = "outputs/optimization/v1/reports"


def _frame(config, relative):
    path = project_path(config.project_root, relative, allowed_prefix="outputs/optimization/v1")
    frame = pd.read_parquet(path)
    if set(frame.columns) != set(DECISION_COLUMNS):
        raise ValueError(f"Saved frame must be a decision frame: {sorted(DECISION_COLUMNS)}")
    return frame


def _safe(value):
    """Strict-JSON view: non-finite numbers become null instead of NaN tokens."""
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(config, name, record):
    path = project_path(config.project_root, f"{REPORT_ROOT}/{name}.json",
                        allowed_prefix="outputs/optimization/v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _safe(record)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"Refusing to overwrite a different report: {name}")
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/optimization/A.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    legacy = subparsers.add_parser("legacy", help="Replay the frozen original OOF metrics")
    legacy.add_argument("--report", default="o08_legacy_recompute")

    validate = subparsers.add_parser("validate", help="Re-check a saved prediction bundle")
    validate.add_argument("--directory", required=True,
                          help="Bundle directory relative to the project root")

    evaluate = subparsers.add_parser("evaluate", help="Score a saved decision frame")
    evaluate.add_argument("--frame", required=True, help="Decision frame parquet inside the optimization root")
    evaluate.add_argument("--report", default=None)

    compare = subparsers.add_parser("compare", help="Paired bootstrap between two saved decision frames")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--reference-id", default="reference")
    compare.add_argument("--candidate-id", default="candidate")
    compare.add_argument("--iterations", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    if args.command == "legacy":
        record = legacy_oof_recompute(config)
        print(canonical_json({"status": record["status"], "users": record["users"],
                              "counts": record["counts"], "accuracy": record["accuracy"],
                              "maximum_metric_abs_difference": record["maximum_metric_abs_difference"],
                              "report": str(_write(config, args.report, record))}))
    elif args.command == "validate":
        record = validate_bundle_artifacts(config, args.directory)
        print(canonical_json(record))
    elif args.command == "evaluate":
        frame = _frame(config, args.frame)
        record = {"schema_version": 1, "frame": args.frame, "frame_sha256": digest(
            [[str(v) for v in frame["user_id"]], [int(v) for v in frame["label"]],
             [float(v) for v in frame["probability"]], [int(v) for v in frame["final_prediction"]]]),
            "metrics": evaluate_frame(frame)}
        if args.report:
            record["report"] = str(_write(config, args.report, record))
        print(canonical_json(_safe(record)))
    else:
        reference, candidate = _frame(config, args.reference), _frame(config, args.candidate)
        record = compare_frames(reference, candidate, reference_id=args.reference_id,
                                candidate_id=args.candidate_id, iterations=args.iterations, seed=args.seed)
        if args.report:
            record["report"] = str(_write(config, args.report, record))
        print(canonical_json(_safe({"status": "pass", "reference": args.reference_id,
                              "candidate": args.candidate_id, "users": record["users"],
                              "point_differences": {k: v for k, v in record["point_differences"].items()
                                                    if k != "counts"},
                              "joint_support": record["joint_support"],
                              "indices_sha256": record["bootstrap"]["indices_sha256"],
                              "report": record.get("report")})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
