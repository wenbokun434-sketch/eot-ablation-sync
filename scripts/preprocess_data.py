"""Build target-only user sequences directly from raw eRisk JSON."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.preprocessing import preprocess_dataset, write_preprocessed
from eot.reproducibility import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    posts, users, report = preprocess_dataset(config)
    posts_path, users_path = write_preprocessed(posts, users, config.path("processed_dir"))
    report_path = config.path("outputs_dir") / "reports" / "preprocessing_report.json"
    write_json(report, report_path)
    print(f"posts={posts_path}")
    print(f"users={users_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
