"""Validate all formal T10 prediction and metric artifacts."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.evaluation import validate_evaluation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    report = validate_evaluation_artifacts(load_config(args.config))
    print(
        f"status={report['status']} oof_users={report['oof_users']} "
        f"prefix_rows={report['oof_prefix_rows']} "
        f"max_metric_difference={report['maximum_recomputed_metric_abs_difference']}"
    )


if __name__ == "__main__":
    main()
