"""O10 screening entry: run one outer fold, a set of outer folds, or scan the
screening manifest. All artifacts stay under outputs/optimization/v1/screening/.
"""
import argparse
from pathlib import Path

from eot.optimization.config import canonical_json
from eot.optimization.screening import (OUTER_FOLDS, run_outer, run_screening,
                                        screening_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--device", default=None, help="cpu, cuda or an explicit cuda index")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run-outer", help="Run all 12 jobs of one outer fold in order")
    one.add_argument("--outer-fold", type=int, choices=OUTER_FOLDS, required=True)
    many = sub.add_parser("run-folds", help="Sequentially run selected outer folds")
    many.add_argument("--outer-folds", default=",".join(str(fold) for fold in OUTER_FOLDS))
    scan = sub.add_parser("manifest", help="Read-only completion scan; writes screening_manifest.json")
    scan.add_argument("--no-write", action="store_true", help="Print only; do not write the manifest")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "run-outer":
        report = run_outer(args.outer_fold, root=root, device_name=args.device)
        print(canonical_json({"status": "pass", "outer_fold": report["outer_fold"],
                              "advanced": report["advanced"],
                              "fallback_to_A": report["fallback_to_A"],
                              "report": report["report"]}))
        return 0
    if args.command == "run-folds":
        folds = tuple(int(value.strip()) for value in args.outer_folds.split(",") if value.strip())
        manifest = run_screening(root=root, device_name=args.device, outer_folds=folds)
        print(canonical_json({"status": manifest["status"],
                              "completed_runs": manifest["completed_runs"],
                              "expected_runs": manifest["expected_runs"],
                              "manifest": "outputs/optimization/v1/screening/screening_manifest.json"}))
        return 0
    manifest = screening_manifest(root=root, write=not args.no_write)
    print(canonical_json({"status": manifest["status"],
                          "completed_runs": manifest["completed_runs"],
                          "expected_runs": manifest["expected_runs"],
                          "missing_runs": manifest["missing_runs"][:10],
                          "missing_predictions": manifest["missing_predictions"][:10],
                          "missing_selections": manifest["missing_selections"][:10],
                          "missing_reports": manifest["missing_reports"][:10]}))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
