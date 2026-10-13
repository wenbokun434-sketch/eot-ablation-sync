"""O02 document and baseline audit only; no training, splitting or threshold search.

snapshot: record existing protected project and raw files, refusing replacement.
verify: rehash protection records, verify the frozen protocol and write acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/optimization/v1/protocol"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def verify_legacy():
    expected = {}
    def add(relative, sha):
        name = Path(relative).as_posix()
        if name in expected:
            assert expected[name] == sha, f"Conflicting legacy digest: {name}"
        expected[name] = sha
    vis = read(ROOT / "outputs/reports/visualization_manifest.json")
    for section in ("sources", "outputs", "code"):
        for name, sha in vis[section].items():
            add(name, sha)
    evaluation = read(ROOT / "outputs/reports/evaluation_manifest.json")
    for row in evaluation["checkpoint_records"]:
        add(row["path"], row["sha256"])
    add(evaluation["oof_final_predictions"], evaluation["oof_final_sha256"])
    add(evaluation["oof_prefix_predictions"], evaluation["oof_prefix_sha256"])
    notebook = read(ROOT / "outputs/reports/notebook_validation.json")
    for name, sha in notebook["code"].items():
        add(name, sha)
    for name_key, hash_key in (("notebook", "notebook_sha256"), ("html_preview", "html_sha256"),
                                ("analysis_source", "analysis_source_sha256")):
        add(notebook[name_key], notebook[hash_key])
    for name, sha in expected.items():
        assert digest(ROOT / name) == sha, f"Legacy artifact differs: {name}"
    return len(expected)


def snapshot():
    target = OUT / "baseline_protection.json"
    if target.exists():
        raise FileExistsError("Baseline already frozen; use verify, never replace the baseline.")
    verified = verify_legacy()
    raw = Path(read(ROOT / "outputs/reports/raw_data_audit.json")["dataset"]["root"])
    assert raw.is_dir(), "Raw dataset is unavailable; snapshot must not silently omit it."
    records = []
    for directory in ("configs", "src", "scripts", "tests", "data", "outputs", "notebooks", "docs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in {"optimization", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if relative.parts[:2] == ("outputs", "tmp"):
                continue
            records.append({"anchor": "project", "path": relative.as_posix(),
                            "bytes": path.stat().st_size, "sha256": digest(path)})
    for name in ("TODO.md", "DESIGN.md", "EXPERIMENT_PLAN.md", "MANIFEST.md", "pyproject.toml",
                 "requirements.txt", "requirements-cuda.txt"):
        path = ROOT / name
        if path.is_file():
            records.append({"anchor": "project", "path": name, "bytes": path.stat().st_size,
                            "sha256": digest(path)})
    raw_files = sorted(path for path in raw.rglob("*") if path.is_file())
    for index, path in enumerate(raw_files):
        records.append({"anchor": "raw", "path": path.relative_to(raw).as_posix(),
                        "bytes": path.stat().st_size, "sha256": digest(path)})
        if (index + 1) % 200 == 0:
            print(f"raw snapshot {index + 1}/{len(raw_files)}", flush=True)
    keys = [(r["anchor"], r["path"]) for r in records]
    assert len(keys) == len(set(keys))
    write(target, {"status": "frozen", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "raw_root": str(raw), "project_root": str(ROOT), "hash_algorithm": "sha256",
                   "legacy_manifest_entries_verified": verified, "records": records,
                   "mutable_control_files_excluded": ["../优化TODO.md", "README.md"],
                   "excluded_ephemeral_directories": ["__pycache__", ".pytest_cache", "outputs/tmp"],
                   "note": "Existing files only; new optimization paths are excluded. Raw data is read-only."},
          exclusive=True)
    print(f"snapshot frozen: {len(records)} files; legacy manifest entries verified: {verified}", flush=True)


def verify():
    protected = read(OUT / "baseline_protection.json")
    anchors = {"project": ROOT, "raw": Path(protected["raw_root"])}
    for index, row in enumerate(protected["records"]):
        path = anchors[row["anchor"]] / row["path"]
        assert path.is_file() and path.stat().st_size == row["bytes"], row["path"]
        assert digest(path) == row["sha256"], row["path"]
        if (index + 1) % 300 == 0:
            print(f"protected files verified {index + 1}/{len(protected['records'])}", flush=True)
    raw_actual = {p.relative_to(anchors["raw"]).as_posix() for p in anchors["raw"].rglob("*") if p.is_file()}
    raw_expected = {r["path"] for r in protected["records"] if r["anchor"] == "raw"}
    assert raw_actual == raw_expected, "Raw dataset file inventory changed."
    legacy_count = verify_legacy()
    protocol = read(OUT / "protocol_v1.json")
    assert protocol["protocol_id"] == "eot_threshold_loss_v1"
    assert protocol["status"] == "locked_before_optimization_training"
    assert protocol["seeds"] == [42, 43, 44]
    assert protocol["thresholds"]["grid"] == {"min": 0.0, "max": 1.0, "step": 0.005, "count": 201}
    assert protocol["data_roles"]["search"] == "original_train_user_ids"
    assert protocol["data_roles"]["calibration"] == "original_validation_user_ids"
    assert protocol["data_roles"]["test"] == "original_test_user_ids"
    assert protocol["inner_cv"]["folds"] == 3
    assert protocol["final_fit"]["calibration_used_for_epoch_selection"] is False
    assert protocol["final_fit"]["transfer_inner_threshold"] is False
    assert protocol["selection"]["cross_outer_selection"] is False
    assert protocol["selection"]["max_optional_candidates_per_outer"] == 2
    assert protocol["execution"]["test_gate"] == "all_five_outer_folds_have_post_calibration_locks"
    assert protocol["execution"]["max_unique_inner_training_runs"] == 270
    for name, alpha, weight in (("A", 1.0, .5), ("B", 1.0, .25), ("C", .5, .5), ("D", 1.5, .5)):
        assert protocol["candidates"][name]["alpha"] == alpha
        assert protocol["candidates"][name]["prefix_weight"] == weight
    universe, tests, counts = None, [], []
    for fold in range(5):
        split = read(ROOT / f"data/splits/fold_{fold}.json")
        groups = [set(split[key]) for key in ("train_user_ids", "validation_user_ids", "test_user_ids")]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3))
        current = set.union(*groups)
        universe = current if universe is None else universe
        assert current == universe and len(universe) == 909
        assert split["counts"]["validation"]["depression_1"] == 10
        tests.extend(groups[2])
        counts.append({"fold": fold, **split["counts"]})
    assert len(tests) == len(set(tests)) == 909
    doc = ROOT / "docs/optimization/PROTOCOL_v1.md"
    body = doc.read_text(encoding="utf-8")
    for section in range(1, 13):
        assert f"## {section}. " in body
    todo = (ROOT.parent / "优化TODO.md").read_text(encoding="utf-8")
    completed = re.findall(r"^- \[x\] \*\*(O\d{2})", todo, re.M)
    assert completed in (["O01"], ["O01", "O02"])
    locked_paths = [doc, OUT / "protocol_v1.json", OUT / "baseline_protection.json", Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in locked_paths}
    lock_path = OUT / "protocol_lock.json"
    if lock_path.exists():
        assert read(lock_path)["sha256"] == hashes, "O02 lock changed; a new version is required."
    else:
        write(lock_path, {"protocol_id": protocol["protocol_id"], "sha256": hashes,
                          "created_at_utc": datetime.now(timezone.utc).isoformat()}, exclusive=True)
    acceptance = {"status": "pass", "task": "O02", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                  "protected_files_verified": len(protected["records"]), "raw_files_verified": len(raw_expected),
                  "legacy_manifest_entries_verified": legacy_count, "protocol_lock_sha256": digest(lock_path),
                  "data_roles_and_outer_membership_verified": True, "protocol_schema_checks": "pass",
                  "documentation_sections": 12, "outer_fold_counts": counts,
                  "training_executed": False, "threshold_search_executed": False,
                  "optimization_business_code_created": False,
                  "claim_boundary": "Protocol and baseline integrity verified; no optimization result yet."}
    write(OUT / "o02_acceptance.json", acceptance)
    print(json.dumps({key: acceptance[key] for key in ("status", "task", "protected_files_verified",
                     "raw_files_verified", "legacy_manifest_entries_verified", "training_executed")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["snapshot", "verify"])
    args = parser.parse_args()
    snapshot() if args.mode == "snapshot" else verify()
