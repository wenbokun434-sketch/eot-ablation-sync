"""Audit the raw dataset without creating training data."""

from __future__ import annotations

import argparse

from eot.audit import audit_raw_dataset
from eot.config import load_config
from eot.reproducibility import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output = config.path("outputs_dir") / "reports" / "raw_data_audit.json"
    report = audit_raw_dataset(config)
    write_json(report, output)
    print(f"status={report['status']} output={output}")


if __name__ == "__main__":
    main()
