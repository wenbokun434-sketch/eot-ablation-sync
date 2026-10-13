"""O11 only: fixed O10 promotions, additional seeds, and within-fold ensembles.

Training, loss, thresholds and metrics are delegated to frozen O07/O06/O08.
Seed 42 is read from screening, never retrained. No calibration/test access.
"""
from functools import cmp_to_key
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eot.dataset import SequenceStore
from eot.evaluation import predict_dataset
from . import screening as old
from . import training
from .config import canonical_json, file_sha256, project_path
from .decisions import SelectionPredictions, load_selection_scope, select_dual_thresholds
from .evaluation import (PredictionBundle, SeedRun, build_bundle, decision_frame,
                         evaluate_frame)
from .splits import digest

ROOT = "outputs/optimization/v1/stability"
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(root, relative, record):
    path = project_path(root, f"{ROOT}/{relative}", allowed_prefix=ROOT)
    content = canonical_json(record) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace different O11 evidence: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return path


def promotion(row):
    """Exact O10 count rule; also used on the three-seed ensemble."""
    per = row["per_fold"]
    if len(per) != 3 or any(p["candidate"] is None for p in per):
        return False
    refs, candidates = [p["R2"]["counts"] for p in per], [p["candidate"]["counts"] for p in per]
    return (sum(c["correct"] for c in candidates) >= sum(r["correct"] for r in refs)
            and sum(c["tp"] for c in candidates) > sum(r["tp"] for r in refs)
            and sum(c["correct"] >= r["correct"] and c["tp"] >= r["tp"]
                    for c, r in zip(candidates, refs)) >= 2)


def required_candidates(root):
    audit = read(root / "outputs/optimization/v1/screening/o10_acceptance.json")
    manifest = root / "outputs/optimization/v1/screening/screening_manifest.json"
    if (audit.get("status") != "pass" or audit.get("runs") != 60
            or file_sha256(manifest) != audit.get("screening_manifest_sha256")):
        raise ValueError("A current passing O10 acceptance is required")
    for name, sha in audit["artifact_sha256"].items():
        if file_sha256(root / name) != sha:
            raise ValueError(f"O10 accepted source changed: {name}")
    result = {}
    for f in range(5):
        report = read(root / f"outputs/optimization/v1/screening/reports/outer_{f}.json")
        if report.get("task") != "O10" or report.get("outer_fold") != f:
            raise ValueError("Invalid O10 outer report")
        advanced = []
        for c in ("B", "C", "D"):
            row = {"per_fold": [{"R2": report["R2"][str(j)],
                    "candidate": report["candidates"][c]["per_fold"][str(j)]["metrics"]}
                    for j in range(3)]}
            passed = promotion(row)
            if passed != report["candidates"][c]["advances_to_O11"]:
                raise ValueError("O10 promotion does not match frozen rule")
            if passed:
                advanced.append(c)
        if advanced != report["advanced"]:
            raise ValueError("O10 candidate list mismatch")
        result[f] = ["A", *advanced]
    return result


def prediction_path(root, f, j, c, seed):
    stage = "screening" if seed == 42 else "stability"
    return root / f"outputs/optimization/v1/{stage}/evaluation/outer_{f}/inner_{j}/{c}/seed_{seed}/prediction_manifest.json"


def plan_for(config, f, j, seed):
    return training.prepare_inner_run(config, stage="screening" if seed == 42 else "stability",
                                      outer_fold=f, inner_fold=j, seed=seed)


def load_prediction(config, f, j, seed, plan=None):
    root = config.project_root
    plan = plan or plan_for(config, f, j, seed)
    run = old._read_manifest(plan)
    record = read(prediction_path(root, f, j, config.candidate_id, seed))
    expected = {"seed": seed, "outer_fold": f, "inner_fold": j,
                "candidate_id": config.candidate_id, "run_id": plan.identity.run_id,
                "checkpoint_sha256": run["checkpoint_sha256"],
                "run_manifest_sha256": file_sha256(plan.identity.artifact_path("run.json")),
                "best_epoch": run["best_epoch"]}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Prediction provenance differs from complete training run")
    prefix_root = f"outputs/optimization/v1/{'screening' if seed == 42 else 'stability'}"
    frames = []
    for key in ("final", "prefix"):
        path = project_path(root, record[key], allowed_prefix=prefix_root)
        if file_sha256(path) != record[key + "_sha256"]:
            raise ValueError("Prediction checksum mismatch")
        frames.append(pd.read_parquet(path))
    final, prefix = frames
    old._check_prediction_pair(final, prefix, plan.context.user_ids("holdout"))
    users = pd.read_parquet(root / "data/processed/user_sequences.parquet",
                            columns=["user_id", "label"],
                            filters=[("user_id", "in", plan.context.user_ids("holdout"))])
    labels = dict(zip(users.user_id.astype(str), users.label.astype(int)))
    if any(labels[str(row.user_id)] != int(row.label) for row in final.itertuples()):
        raise ValueError("Prediction labels differ from actual holdout labels")
    if digest([[str(r.user_id), int(r.label)] for r in final.itertuples()]) != record["labels_sha256"]:
        raise ValueError("Prediction label checksum mismatch")
    return record, final, prefix


def export(config, f, j, seed, plan, store, device):
    if seed not in (43, 44):
        raise ValueError("O11 only exports new seeds 43/44")
    path = prediction_path(config.project_root, f, j, config.candidate_id, seed)
    if path.exists():
        return load_prediction(config, f, j, seed, plan)[0]
    model, run = training.load_trained_model(plan, device_name=str(old._device(device)))
    final, prefix = predict_dataset(model, plan.context.dataset(store, "holdout"),
        device=old._device(device), batch_size=int(config.section("training")["batch_users"]))
    old._check_prediction_pair(final, prefix, plan.context.user_ids("holdout"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "seed": seed, "outer_fold": f, "inner_fold": j,
              "candidate_id": config.candidate_id, "run_id": plan.identity.run_id,
              "checkpoint_sha256": run["checkpoint_sha256"], "best_epoch": run["best_epoch"],
              "run_manifest_sha256": file_sha256(plan.identity.artifact_path("run.json")),
              "labels_sha256": digest([[str(r.user_id), int(r.label)] for r in final.itertuples()]),
              "role": "inner_holdout"}
    for name, frame in (("final", final), ("prefix", prefix)):
        target = path.parent / f"{name}.parquet"
        if target.exists():
            if not pd.read_parquet(target).equals(frame):
                raise ValueError("Partial prediction export differs on recovery")
        else:
            frame.to_parquet(target, index=False, compression="zstd")
        record[name] = target.relative_to(config.project_root).as_posix()
        record[name + "_sha256"] = file_sha256(target)
    write(config.project_root, path.relative_to(config.project_root / ROOT).as_posix(), record)
    return load_prediction(config, f, j, seed, plan)[0]


def bundles(config, f, j):
    loaded = [load_prediction(config, f, j, s) for s in SEEDS]
    runs = [SeedRun(s, r["run_id"], r["run_manifest_sha256"], r["checkpoint_sha256"])
            for s, (r, _, _) in zip(SEEDS, loaded)]
    ensemble = build_bundle(config, stage="stability", outer_fold=f, inner_fold=j,
        role="inner_holdout", seeds=SEEDS, runs=runs,
        final_frames=[x[1] for x in loaded], prefix_frames=[x[2] for x in loaded])
    result = {"ensemble": (ensemble, tuple((s, r["checkpoint_sha256"]) for s, (r, _, _) in zip(SEEDS, loaded)))}
    for seed, (record, final, prefix) in zip(SEEDS, loaded):
        result[str(seed)] = (PredictionBundle(canonical_json(record), final, prefix),
                            ((seed, record["checkpoint_sha256"]),))
    return result


def summarize(folds):
    report = {"per_fold": folds}
    for name in ("R1", "R2", "candidate"):
        metrics = [f[name] for f in folds]
        report[name] = None if any(m is None for m in metrics) else {
            "counts": {k: sum(m["counts"][k] for m in metrics)
                       for k in ("tp", "tn", "fp", "fn", "correct", "users")},
            "mean_inner_ap": float(np.mean([m["pr_auc"] for m in metrics]))}
        if report[name] is not None:
            counts = report[name]["counts"]
            report[name].update(accuracy=counts["correct"] / counts["users"],
                                recall=counts["tp"] / (counts["tp"] + counts["fn"]))
    report["promotion_rule_pass"] = promotion(report)
    report["tradeoff_vs_R2"] = None if report["candidate"] is None else {
        **{f"delta_{k}": report["candidate"]["counts"][k] - report["R2"]["counts"][k]
           for k in ("tp", "tn", "fp", "fn", "correct")},
        **{f"delta_{k}": report["candidate"][k] - report["R2"][k] for k in ("accuracy", "recall")}}
    return report


def stability_rule(modes):
    good_seeds = [s for s in map(str, SEEDS) if modes[s]["candidate"] is not None
                 and modes[s]["candidate"]["counts"]["correct"] >= modes[s]["R2"]["counts"]["correct"]
                 and modes[s]["candidate"]["counts"]["tp"] >= modes[s]["R2"]["counts"]["tp"]]
    return {"stable": bool(modes["ensemble"]["promotion_rule_pass"] and len(good_seeds) >= 2),
            "noninferior_seeds": good_seeds}


def ranked_stable(candidates):
    def compare(a, b):
        for key in ("recall", "accuracy", "mean_inner_ap"):
            diff = candidates[a]["modes"]["ensemble"]["candidate"][key] - candidates[b]["modes"]["ensemble"]["candidate"][key]
            if abs(diff) > 1e-12:
                return -1 if diff > 0 else 1
        return (a > b) - (a < b)
    return sorted([c for c in candidates if c != "A" and candidates[c]["stable"]], key=cmp_to_key(compare))


def ensemble_evidence(root, f, j, c, bundle, persist):
    base = root / ROOT / f"evaluation/outer_{f}/inner_{j}/{c}/ensemble"
    for name, frame in (("final", bundle.final), ("prefix", bundle.prefix)):
        path = base / f"{name}.parquet"
        if path.exists():
            if not pd.read_parquet(path).equals(frame):
                raise ValueError("Saved ensemble probabilities differ from seed average")
        elif persist:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            raise ValueError("Missing ensemble probabilities")
    record = {"provenance": bundle.provenance,
              "final_sha256": file_sha256(base / "final.parquet"),
              "prefix_sha256": file_sha256(base / "prefix.parquet")}
    relative = base.relative_to(root / ROOT).as_posix() + "/ensemble.json"
    if persist:
        write(root, relative, record)
    elif read(root / ROOT / relative) != record:
        raise ValueError("Saved ensemble provenance differs")


def build_report(root, f, *, persist=True):
    required = required_candidates(root)[f]
    configs = old.candidate_configs(root)
    results = {c: {m: [] for m in ("42", "43", "44", "ensemble")} for c in required}
    evidence = {}
    for j in range(3):
        predictions = {c: bundles(configs[c], f, j) for c in required}
        scope = load_selection_scope(configs["A"], role="inner_holdout", outer_fold=f, inner_fold=j)
        for c in required:
            ensemble_evidence(root, f, j, c, predictions[c]["ensemble"][0], persist)
            for mode in results[c]:
                a, a_hashes = predictions["A"][mode]
                q, q_hashes = predictions[c][mode]
                selection = select_dual_thresholds(configs[c], scope,
                    SelectionPredictions("A", configs["A"].config_hash, a_hashes, tuple(a.sequences())),
                    SelectionPredictions(c, configs[c].config_hash, q_hashes, tuple(q.sequences())))
                relative = f"thresholds/outer_{f}/inner_{j}/{c}/{mode}.json"
                record = selection.manifest
                if persist:
                    selection.save(configs[c], f"{ROOT}/{relative}")
                else:
                    stored = read(root / ROOT / relative)
                    if stored != {"sha256": digest(record), "selection": record}:
                        raise ValueError("Saved threshold differs from independent replay")
                evidence[relative] = file_sha256(root / ROOT / relative)
                row = {"inner_fold": j}
                for name, bundle in (("R1", a), ("R2", a), ("candidate", q)):
                    t = record[name]
                    row[name] = None if t["threshold_final"] is None else old._metric_summary(evaluate_frame(
                        decision_frame(bundle, threshold_final=t["threshold_final"], threshold_early=t["threshold_early"])))
                row["tradeoff_vs_R2"] = None if row["candidate"] is None else old._tradeoff(row["R2"], row["candidate"])
                results[c][mode].append(row)
    candidates = {}
    for c in required:
        modes = {m: summarize(rows) for m, rows in results[c].items()}
        candidates[c] = {"modes": modes, **stability_rule(modes)}
    stable = ranked_stable(candidates)
    weights = [c for c in stable if c in ("C", "D")]
    branch = ({"branch": "E", "weight_candidate": weights[0]} if "B" in stable and weights else
              {"branch": "F1"} if not stable else {"branch": "skip"})
    report = {"task": "O11", "outer_fold": f, "required_candidates": required,
              "seeds": list(SEEDS), "candidates": candidates, "ranked_stable": stable,
              "O12_condition": branch, "O12_executed": False,
              "threshold_sha256": evidence, "outer_test_labels_accessed": False}
    if persist:
        write(root, f"reports/outer_{f}.json", report)
    return report


def run_outer(root, f, device="cuda"):
    required = required_candidates(root)[f]
    configs = old.candidate_configs(root)
    store = SequenceStore(training._legacy_read_config(configs["A"]))
    for j in range(3):
        for c in required:
            # Validate the existing seed before spending additional training budget.
            load_prediction(configs[c], f, j, 42)
            for seed in (43, 44):
                plan = plan_for(configs[c], f, j, seed)
                print(f"O11 train outer={f} inner={j} candidate={c} seed={seed}", flush=True)
                old.ensure_screening_run(plan, store, device_name=device)
                export(configs[c], f, j, seed, plan, store, device)
    return build_report(root, f)
