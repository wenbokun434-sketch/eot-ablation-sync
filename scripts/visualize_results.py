"""Generate the T11 publication figures and tables from frozen formal results."""
import argparse
from eot.config import load_config
from eot.visualization import build_visualizations
from eot.reporting import validate_visualizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.validate_only:
        report = build_visualizations(config, args.device)
        print(f"status={report['status']} figures={report['figure_count']} tables={report['table_count']}")
    validation = validate_visualizations(config)
    print(f"artifact validation={validation['status']}")


if __name__ == "__main__":
    main()
