"""Run the required two-epoch real-feature smoke training."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.training import run_smoke_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = run_smoke_training(load_config(args.config), args.fold, args.seed, args.device)
    print(report)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
