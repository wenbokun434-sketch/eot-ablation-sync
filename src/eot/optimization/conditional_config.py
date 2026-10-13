"""O12 config and gate adapters; O03 config loader and old hashes stay frozen."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import yaml
from .config import (canonical_json, file_sha256, load_optimization_config, project_path,
                     _UniqueLoader)
from .splits import digest

OUTPUT = "outputs/optimization/v1/conditional"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(root, relative, record):
    path = project_path(root, f"{OUTPUT}/{relative}", allowed_prefix=OUTPUT)
    content = canonical_json(record) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace different O12 evidence: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return path


def verify_o11(root, *, all_frozen=False):
    out = root / "outputs/optimization/v1/stability"
    audit = read(out / "o11_acceptance.json")
    if (audit.get("task") != "O11" or audit.get("status") != "pass"
            or audit.get("new_runs") != 48 or audit.get("reused_runs") != 24):
        raise ValueError("O12 requires complete O11 acceptance")
    if file_sha256(out / "reports/o11/preflight.json") != audit["preflight_sha256"]:
        raise ValueError("O11 preflight changed")
    if "- [x] **O11" not in (root / "优化TODO.md").read_text(encoding="utf-8"):
        raise ValueError("O11 must be accepted and checked first")
    if set(audit["report_sha256"]) != set(map(str, range(5))):
        raise ValueError("O11 reports are incomplete")
    reports = {}
    for f, sha in audit["report_sha256"].items():
        path = out / f"reports/outer_{f}.json"
        if file_sha256(path) != sha:
            raise ValueError("Accepted O11 report changed")
        reports[int(f)] = read(path)
    if all_frozen:
        for name, sha in read(out / "reports/o11/preflight.json")["frozen_sha256"].items():
            if file_sha256(root / name) != sha:
                raise ValueError(f"Frozen O11 source/evidence changed: {name}")
    return reports


def branch_for(report):
    stable = report["ranked_stable"]
    weights = [c for c in stable if c in ("C", "D")]
    branch = ({"branch": "E", "weight_candidate": weights[0]} if "B" in stable and weights
              else {"branch": "F1"} if not stable else {"branch": "skip"})
    if branch != report["O12_condition"]:
        raise ValueError("O11 branch disagrees with locked trigger")
    return branch


def f2_trigger(report):
    if report.get("candidate_id") != "F1":
        raise ValueError("F2 requires F1 evidence")
    ensemble = report["modes"]["ensemble"]
    return bool(report["stable"] and ensemble["candidate"] is not None
                and ensemble["candidate"]["mean_inner_ap"] + 1e-12 >= ensemble["R2"]["mean_inner_ap"])


def load_config(root, candidate, *, weight_candidate=None):
    base = load_optimization_config(root / "configs/optimization/A.yaml")
    if candidate not in ("E", "F1", "F2"):
        raise ValueError("O12 candidates are only E, F1 and F2")
    if candidate == "E":
        if weight_candidate not in ("C", "D"):
            raise ValueError("Combination requires a stable weight candidate")
        settings = dict(final_loss="weighted_bce", alpha=.5 if weight_candidate == "C" else 1.5,
                        prefix_weight=.25, gamma=0.0)
        filename = f"E_{weight_candidate}"
    else:
        option = base.protocol["optional"][candidate]
        settings = {k: option[k] for k in ("final_loss", "alpha", "prefix_weight", "gamma")}
        filename = candidate
    path = project_path(root, f"configs/optimization/{filename}.yaml", allowed_prefix="configs/optimization")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    expected = yaml.load(base.source.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    expected.update(candidate_id=candidate, loss=settings)
    if (raw != expected or type(raw.get("schema_version")) is not int
            or any(type(raw["loss"][k]) not in (int, float) for k in ("alpha", "prefix_weight", "gamma"))):
        raise ValueError("Conditional config differs from the exact locked option")
    loss = replace(base.loss, **settings)
    effective = base.values
    effective.update(candidate_id=candidate, loss=asdict(loss),
                     implementation="O12_conditional_v1")
    return replace(base, source=path, candidate_id=candidate, loss=loss,
                   source_sha256=file_sha256(path), config_hash=digest(effective),
                   _effective_json=canonical_json(effective))


def gate(root, f, candidate):
    if type(f) is not int or f not in range(5):
        raise ValueError("Invalid outer fold")
    reports = verify_o11(root)
    branch = branch_for(reports[f])
    source = root / f"outputs/optimization/v1/stability/reports/outer_{f}.json"
    if candidate == "F2":
        if branch["branch"] != "F1":
            raise ValueError("F2 cannot run on a combination or skipped path")
        source = root / OUTPUT / f"reports/outer_{f}/F1.json"
        report = read(source)
        if report.get("outer_fold") != f or not f2_trigger(report):
            raise ValueError("F2 trigger is not satisfied")
        # The controller must have fully replayed F1 before unlocking F2.
        seal = read(root / OUTPUT / f"gates/outer_{f}/F1_verified.json")
        if (seal.get("report_sha256") != file_sha256(source)
                or seal.get("status") != "replayed" or seal.get("runs") != 9):
            raise ValueError("F1 report has no current replay verification")
    elif candidate != branch["branch"]:
        raise ValueError("Candidate is not triggered in this outer fold")
    config = load_config(root, candidate, weight_candidate=branch.get("weight_candidate"))
    record = {"task": "O12", "outer_fold": f, "candidate_id": candidate,
              "config_sha256": config.config_hash, "source": source.relative_to(root).as_posix(),
              "source_sha256": file_sha256(source), "branch": branch,
              "O11_acceptance_sha256": file_sha256(root / "outputs/optimization/v1/stability/o11_acceptance.json")}
    return config, record
