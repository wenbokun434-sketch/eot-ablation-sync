"""Run one or all formal fold/seed EOT training jobs."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.training import train_formal_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.all:
        jobs = [
            (fold, int(seed))
            for fold in range(int(config.section("cross_validation")["n_splits"]))
            for seed in config.section("training")["seeds"]
        ]
    else:
        if args.fold is None or args.seed is None:
            parser.error("provide --fold and --seed, or use --all")
        jobs = [(args.fold, args.seed)]
    for fold, seed in jobs:
        print(train_formal_run(config, fold, seed, args.device, args.max_epochs))


if __name__ == "__main__":
    main()
