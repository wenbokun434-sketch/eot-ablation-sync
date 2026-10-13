"""Train/resume one optimization inner run; no batch dispatch or final training."""
import argparse

from eot.optimization.config import canonical_json, load_optimization_config
from eot.optimization.training import prepare_inner_run, train_inner_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["screening", "stability"], required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="Pause at this absolute epoch; the 50-epoch protocol budget is unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Read-only provenance preflight; no model or training")
    args = parser.parse_args()
    config = load_optimization_config(args.config)
    job = dict(stage=args.stage, outer_fold=args.outer_fold, inner_fold=args.inner_fold, seed=args.seed)
    if args.dry_run:
        plan = prepare_inner_run(config, **job)
        print(canonical_json({"status": "preflight_only", "run_id": plan.identity.run_id,
            "directory": str(plan.identity.directory), "binding": plan.binding}))
    else:
        print(canonical_json(train_inner_run(config, **job, device_name=args.device,
            resume=args.resume, stop_after_epoch=args.stop_after_epoch)))


if __name__ == "__main__":
    main()
