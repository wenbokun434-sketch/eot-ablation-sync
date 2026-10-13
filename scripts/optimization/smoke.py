"""O09 smoke entry: train A-D for two small epochs, recover/reload, select dual
thresholds on H_j and evaluate with O08. Writes only under
``outputs/optimization/v1/smoke/``.
"""
import argparse
from pathlib import Path

from eot.optimization.config import canonical_json
from eot.optimization.smoke import SMOKE_CANDIDATES, prepare_smoke, run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--candidates", default=",".join(SMOKE_CANDIDATES),
                        help="comma-separated subset of A,B,C,D; must include A")
    parser.add_argument("--device", default=None, help="cpu, cuda or an explicit cuda index")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read-only preflight; no model, training or output artifact")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    candidates = tuple(part.strip() for part in args.candidates.split(",") if part.strip())
    if args.dry_run:
        preparation = prepare_smoke(root)
        print(canonical_json({
            "status": "preflight_only", "root": str(root),
            "subset_counts": preparation.subset.counts,
            "selection_users": len(preparation.scope.metadata["user_ids"]),
            "plans": {candidate: {
                "run_id": plan.identity.run_id,
                "directory": str(plan.identity.directory.relative_to(root))}
                for candidate, plan in preparation.plans.items() if candidate in candidates},
        }))
        return 0
    report = run_smoke(candidates, project_root=root, device_name=args.device)
    print(canonical_json({
        "status": report["status"], "task": report["task"],
        "candidates": {candidate: {
            "run_id": item["run_id"], "epochs": item["epochs"],
            "reload_matched": item["reload"]["matched"],
            "candidate_status": item["thresholds"]["candidate_status"]}
            for candidate, item in report["candidates"].items()},
        "checks": report["checks"], "report": "outputs/optimization/v1/smoke/smoke_acceptance.json",
    }))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
