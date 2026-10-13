"""Render the ODE mechanism figures from the frozen probe records (no training)."""
import argparse
from pathlib import Path

from eot.notebook_support import notebook_config
from eot.ode_analysis import build_ode_figures, validate_probe_records
from eot.reproducibility import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    config = notebook_config(Path(args.root).resolve())
    validation = validate_probe_records(config, config.path("outputs_dir") / "ode_probe")
    if validation["status"] != "pass":
        raise AssertionError(f"probe validation failed: {validation['failures']}")
    report = build_ode_figures(config)
    write_json(
        report,
        config.path("outputs_dir") / "reports" / "ode_visualization_manifest.json",
    )
    print(
        f"status={report['status']} figures={report['figure_count']} "
        f"spearman={report['stats']['spearman_delta_vs_displacement']:.4f}"
    )


if __name__ == "__main__":
    main()
