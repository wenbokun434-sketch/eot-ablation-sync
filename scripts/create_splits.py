"""Create fixed, leakage-safe user-level folds."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.splitting import create_user_splits, write_user_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    folds, report = create_user_splits(config)
    write_user_splits(folds, report, config)
    for item in report["fold_counts"]:
        print(item)


if __name__ == "__main__":
    main()
