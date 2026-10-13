"""Execute exactly one O11 outer fold, or replay its saved report."""
import argparse
from pathlib import Path
from eot.optimization.stability import build_report, run_outer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = (build_report(args.root.resolve(), args.outer_fold) if args.report_only else
              run_outer(args.root.resolve(), args.outer_fold, args.device))
    print({"outer_fold": args.outer_fold, "stable": result["ranked_stable"],
           "O12_condition": result["O12_condition"]}, flush=True)
