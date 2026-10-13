"""Run formal validation-threshold selection and outer-fold test evaluation."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.evaluation import run_formal_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    manifest = run_formal_evaluation(load_config(args.config), device_name=args.device)
    print(
        f"status={manifest['status']} device={manifest['device']} "
        f"checkpoints={manifest['checkpoints_validated']} oof_users={manifest['oof_users']}"
    )


if __name__ == "__main__":
    main()
