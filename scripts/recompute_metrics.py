"""Recompute OOF metrics from the saved formal prediction artifact."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from eot.config import load_config
from eot.evaluation import metrics_from_final_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    path = config.path("outputs_dir") / "predictions" / "oof_final_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing OOF predictions; run scripts/evaluate.py first: {path}")
    metrics = metrics_from_final_predictions(pd.read_parquet(path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
