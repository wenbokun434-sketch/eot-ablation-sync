"""Build the final notebook and optionally execute it in a fresh Python kernel."""
import argparse
from pathlib import Path
from eot.notebook_document import build_notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_notebook(root, execute=args.execute)
    print(f"status={result['status']} executed_cells={result.get('code_cells_executed', 0)}")


if __name__ == "__main__":
    main()
