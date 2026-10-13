"""O12 conditional orchestration: gate, three-seed training, replay and F2 check."""
from pathlib import Path
import numpy as np
import pandas as pd
from eot.dataset import SequenceStore
from eot.evaluation import predict_dataset
from . import screening as old, stability as s, training
from .conditional_config import OUTPUT, branch_for, f2_trigger, gate, read, verify_o11, write
from .conditional_training import fit, prepare
from .config import canonical_json, file_sha256, project_path
from .decisions import SelectionPredictions, load_selection_scope, select_dual_thresholds
from .evaluation import PredictionBundle, SeedRun, build_bundle, decision_frame, evaluate_frame
from .splits import digest


def prediction_path(root, f, j, candidate, seed):
    return project_path(root, f"{OUTPUT}/evaluation/outer_{f}/inner_{j}/{candidate}/seed_{seed}/prediction_manifest.json",
                        allowed_prefix=OUTPUT)


def load_prediction(root, f, j, candidate, seed, plan=None):
    plan = plan or prepare(root, f, j, candidate, seed)
    run = old._read_manifest(plan)
    record = read(prediction_path(root, f, j, candidate, seed))
    expected = {"role": "inner_holdout", "outer_fold": f, "inner_fold": j,
                "candidate_id": candidate, "seed": seed, "run_id": plan.identity.run_id,
                "checkpoint_sha256": run["checkpoint_sha256"],
                "run_manifest_sha256": file_sha256(plan.identity.artifact_path("run.json")),
                "best_epoch": run["best_epoch"]}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Conditional prediction provenance differs from completed run")
    frames = []
    for key in ("final", "prefix"):
        path = project_path(root, record[key], allowed_prefix=OUTPUT)
        if file_sha256(path) != record[key + "_sha256"]:
            raise ValueError("Conditional prediction checksum differs")
        frames.append(pd.read_parquet(path))
    final, prefix = frames
    ids = plan.context.user_ids("holdout")
    old._check_prediction_pair(final, prefix, ids)
    users = pd.read_parquet(root / "data/processed/user_sequences.parquet", columns=["user_id", "label"],
                            filters=[("user_id", "in", ids)])
    labels = dict(zip(users.user_id.astype(str), users.label.astype(int)))
    if any(labels[str(r.user_id)] != int(r.label) for r in final.itertuples()):
        raise ValueError("Conditional labels differ from actual H_j")
    if digest([[str(r.user_id), int(r.label)] for r in final.itertuples()]) != record["labels_sha256"]:
        raise ValueError("Conditional labels checksum differs")
    return record, final, prefix


def export(root, f, j, candidate, seed, plan, store, device):
    path = prediction_path(root, f, j, candidate, seed)
    if path.exists():
        return load_prediction(root, f, j, candidate, seed, plan)[0]
    model, run = training.load_trained_model(plan, device_name=str(old._device(device)))
    final, prefix = predict_dataset(model, plan.context.dataset(store, "holdout"),
        device=old._device(device), batch_size=int(plan.config.section("training")["batch_users"]))
    old._check_prediction_pair(final, prefix, plan.context.user_ids("holdout"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "role": "inner_holdout", "outer_fold": f, "inner_fold": j,
              "candidate_id": candidate, "seed": seed, "run_id": plan.identity.run_id,
              "checkpoint_sha256": run["checkpoint_sha256"], "best_epoch": run["best_epoch"],
              "run_manifest_sha256": file_sha256(plan.identity.artifact_path("run.json")),
              "labels_sha256": digest([[str(r.user_id), int(r.label)] for r in final.itertuples()])}
    for name, frame in (("final", final), ("prefix", prefix)):
        target = path.parent / f"{name}.parquet"
        if target.exists():
            if not pd.read_parquet(target).equals(frame):
                raise ValueError("Partial conditional export differs on recovery")
        else:
            frame.to_parquet(target, index=False, compression="zstd")
        record[name] = target.relative_to(root).as_posix()
        record[name + "_sha256"] = file_sha256(target)
    write(root, path.relative_to(root / OUTPUT).as_posix(), record)
    return load_prediction(root, f, j, candidate, seed, plan)[0]


def bundles(root, f, j, candidate, config):
    loaded = [load_prediction(root, f, j, candidate, seed) for seed in s.SEEDS]
    runs = [SeedRun(seed, r["run_id"], r["run_manifest_sha256"], r["checkpoint_sha256"])
            for seed, (r, _, _) in zip(s.SEEDS, loaded)]
    # O08's stability evaluation stage is the frozen three-seed averaging API.
    # Actual training stage remains conditional in every run and extra provenance.
    bundle = build_bundle(config, stage="stability", outer_fold=f, inner_fold=j,
        role="inner_holdout", seeds=s.SEEDS, runs=runs,
        final_frames=[x[1] for x in loaded], prefix_frames=[x[2] for x in loaded])
    bundle = PredictionBundle(canonical_json({**bundle.provenance, "training_stage": "conditional"}),
                              bundle.final, bundle.prefix)
    result = {"ensemble": (bundle, tuple((seed, r["checkpoint_sha256"])
                                          for seed, (r, _, _) in zip(s.SEEDS, loaded)))}
    for seed, (record, final, prefix) in zip(s.SEEDS, loaded):
        result[str(seed)] = (PredictionBundle(canonical_json(record), final, prefix),
                            ((seed, record["checkpoint_sha256"]),))
    return result


def ensemble_evidence(root, f, j, candidate, bundle, persist):
    base = project_path(root, f"{OUTPUT}/evaluation/outer_{f}/inner_{j}/{candidate}/ensemble", allowed_prefix=OUTPUT)
    for name, frame in (("final", bundle.final), ("prefix", bundle.prefix)):
        path = base / f"{name}.parquet"
        if path.exists():
            if not pd.read_parquet(path).equals(frame):
                raise ValueError("O12 ensemble differs from arithmetic seed mean")
        elif persist:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            raise ValueError("Missing O12 ensemble")
    record = {"provenance": bundle.provenance, "final_sha256": file_sha256(base / "final.parquet"),
              "prefix_sha256": file_sha256(base / "prefix.parquet")}
    relative = base.relative_to(root / OUTPUT).as_posix() + "/ensemble.json"
    if persist:
        write(root, relative, record)
    elif read(root / OUTPUT / relative) != record:
        raise ValueError("O12 ensemble provenance differs")


def report_candidate(root, f, candidate, *, persist=True):
    config, trigger = gate(root, f, candidate)
    a_config = training.load_optimization_config(root / "configs/optimization/A.yaml")
    rows = {m: [] for m in ("42", "43", "44", "ensemble")}
    evidence = {}
    for j in range(3):
        reference = s.bundles(a_config, f, j)
        predictions = bundles(root, f, j, candidate, config)
        ensemble_evidence(root, f, j, candidate, predictions["ensemble"][0], persist)
        scope = load_selection_scope(config, role="inner_holdout", outer_fold=f, inner_fold=j)
        for mode in rows:
            a, a_hashes = reference[mode]
            q, q_hashes = predictions[mode]
            selection = select_dual_thresholds(config, scope,
                SelectionPredictions("A", a_config.config_hash, a_hashes, tuple(a.sequences())),
                SelectionPredictions(candidate, config.config_hash, q_hashes, tuple(q.sequences())))
            record = selection.manifest
            relative = f"thresholds/outer_{f}/inner_{j}/{candidate}/{mode}.json"
            if persist:
                selection.save(config, f"{OUTPUT}/{relative}")
            elif read(root / OUTPUT / relative) != {"sha256": digest(record), "selection": record}:
                raise ValueError("O12 threshold replay differs")
            evidence[relative] = file_sha256(root / OUTPUT / relative)
            row = {"inner_fold": j}
            for name, bundle in (("R1", a), ("R2", a), ("candidate", q)):
                t = record[name]
                row[name] = None if t["threshold_final"] is None else old._metric_summary(evaluate_frame(
                    decision_frame(bundle, threshold_final=t["threshold_final"], threshold_early=t["threshold_early"])))
            row["tradeoff_vs_R2"] = None if row["candidate"] is None else old._tradeoff(row["R2"], row["candidate"])
            rows[mode].append(row)
    modes = {m: s.summarize(value) for m, value in rows.items()}
    report = {"task": "O12", "outer_fold": f, "candidate_id": candidate, "config_sha256": config.config_hash,
              "trigger": trigger, "modes": modes, **s.stability_rule(modes),
              "threshold_sha256": evidence, "outer_test_labels_accessed": False}
    relative = f"reports/outer_{f}/{candidate}.json"
    if persist:
        write(root, relative, report)
    elif read(root / OUTPUT / relative) != report:
        raise ValueError("O12 candidate report differs from probability replay")
    return report


def run_candidate(root, f, candidate, store, device):
    config, trigger = gate(root, f, candidate)
    write(root, f"gates/outer_{f}/{candidate}.json", trigger)
    for j in range(3):
        for seed in s.SEEDS:
            plan = prepare(root, f, j, candidate, seed)
            print(f"O12 train outer={f} inner={j} candidate={candidate} seed={seed}", flush=True)
            path = plan.identity.artifact_path("run.json")
            if not path.exists() or read(path)["status"] != "complete":
                fit(plan, store, device_name=device, resume=path.exists())
            old._read_manifest(plan)
            export(root, f, j, candidate, seed, plan, store, device)
    report_candidate(root, f, candidate)
    replay = report_candidate(root, f, candidate, persist=False)
    write(root, f"gates/outer_{f}/{candidate}_verified.json", {
        "report_sha256": file_sha256(root / OUTPUT / f"reports/outer_{f}/{candidate}.json"),
        "status": "replayed", "runs": 9})
    return replay


def run_outer(root, f, device="cuda"):
    branch = branch_for(verify_o11(root)[f])
    results = {}
    if branch["branch"] != "skip":
        a = training.load_optimization_config(root / "configs/optimization/A.yaml")
        store = SequenceStore(training._legacy_read_config(a))
        candidate = branch["branch"]
        report = run_candidate(root, f, candidate, store, device)
        results[candidate] = file_sha256(root / OUTPUT / f"reports/outer_{f}/{candidate}.json")
        if candidate == "F1" and f2_trigger(report):
            run_candidate(root, f, "F2", store, device)
            results["F2"] = file_sha256(root / OUTPUT / f"reports/outer_{f}/F2.json")
    record = {"task": "O12", "status": "complete", "outer_fold": f, "initial_branch": branch,
              "candidate_reports": results, "runs": len(results) * 9,
              "F2_triggered": "F2" in results,
              "reason": "condition_not_triggered_verified_skip" if not results else
                        "F1_stability_and_AP_gate_passed" if "F2" in results else
                        "F1_stability_or_AP_gate_failed" if "F1" in results else "combination_path_only",
              "outer_test_labels_accessed": False}
    write(root, f"reports/outer_{f}.json", record)
    return record
