"""Write a reproducible environment report for the EOT experiment."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.reproducibility import collect_environment, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output = config.path("outputs_dir") / "reports" / "environment.json"
    write_json(collect_environment(config.project_root, config.config_hash), output)
    print(output)


if __name__ == "__main__":
    main()
