"""Run one gated O12 outer fold; never begins O13."""
import argparse
from pathlib import Path
from eot.optimization.conditional import run_outer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(run_outer(args.root.resolve(), args.outer_fold, args.device), flush=True)
