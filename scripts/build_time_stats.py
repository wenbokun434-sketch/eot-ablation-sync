"""Fit fold-local time normalization parameters."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.time_features import create_all_time_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    records = create_all_time_stats(load_config(args.config))
    for record in records:
        print(
            f"fold={record['fold']} train_users={record['fit_user_count']} "
            f"q99_log_gap={record['log_gap_quantile']:.6f}"
        )


if __name__ == "__main__":
    main()
