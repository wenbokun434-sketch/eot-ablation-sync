"""Train-only statistics and verified, candidate-independent optimization data.

Build metadata with ``python -m eot.optimization.data --config
configs/optimization/A.yaml``. Existing differing artifacts are never overwritten.
Final-stage training/calibration gates belong to O13/O14; this adapter exposes
only train, early_stop and holdout datasets for the inner stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from eot.time_features import TimeNormalizer, fit_time_normalizer
from .config import (OptimizationConfig, PROTOCOL_DIR, PROTOCOL_ID, canonical_json,
                     file_sha256, load_optimization_config, project_path)
from .splits import checked_ids, counts, digest, label_map, make_inner_splits

DATA_ROOT = "data/optimization/v1"
POST_COLUMNS = ["user_id", "label", "sequence_index", "delta_days"]


def fit_training_statistics(users: pd.DataFrame, posts: pd.DataFrame, train_user_ids,
                            *, quantile: float = 0.99, max_posts: int = 256) -> dict:
    """Fit on actual gradient-training users, with complete ID/gap provenance.

    The base class ratio is shared by candidates. Multiplication by candidate
    alpha belongs to the loss implementation (O05), not to split generation.
    """
    ids = checked_ids(train_user_ids, "fit")
    labels = label_map(users, ids)
    selected = posts.loc[posts.user_id.isin(ids), POST_COLUMNS].copy()
    if set(selected.user_id) != set(ids):
        raise ValueError("Missing posts for fit users")
    selected = selected.sort_values(["user_id", "sequence_index"], kind="stable")
    for uid, group in selected.groupby("user_id", sort=False):
        indices = group.sequence_index.to_numpy()
        gaps = group.delta_days.to_numpy(dtype=np.float64)
        if (not 1 <= len(group) <= max_posts
                or not np.array_equal(indices, np.arange(len(group)))
                or not np.isfinite(gaps).all() or np.any(gaps < 0) or gaps[0] != 0):
            raise ValueError("Invalid fit sequence order, length or time gaps")
        if not group.label.eq(labels[uid]).all():
            raise ValueError("Fit post labels disagree with user labels")
    class_counts = counts(labels, ids)
    if min(class_counts["positive"], class_counts["negative"]) <= 0:
        raise ValueError("Training statistics require both classes")
    normalizer, details = fit_time_normalizer(selected, ids, quantile)
    ordered_input = [[u, int(y), int(i), float(gap)]
                     for u, y, i, gap in selected.itertuples(index=False, name=None)]
    return {"fit_user_ids": ids, **details,
            "fit_labels_sha256": digest([[u, labels[u]] for u in ids]),
            "fit_post_metadata_sha256": digest(ordered_input),
            "class_counts": class_counts,
            "base_positive_weight": class_counts["negative"] / class_counts["positive"],
            "time_normalizer": normalizer.to_dict()}


def _binding(config: OptimizationConfig) -> dict:
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID,
            "protocol_sha256": config.protocol_sha256,
            "protocol_lock_sha256": config.protocol_lock_sha256,
            "base_config_hash": config.base_config_hash}


def _inputs(config: OptimizationConfig):
    # Recheck the frozen inputs before reading any data; no legacy output writes.
    protection = json.loads(project_path(config.project_root,
                            f"{PROTOCOL_DIR}/baseline_protection.json").read_text(encoding="utf-8"))
    protected = {r["path"]: r["sha256"] for r in protection["records"] if r["anchor"] == "project"}
    paths = config.base_values["paths"]
    names = [f"{paths['processed_dir']}/user_sequences.parquet",
             f"{paths['processed_dir']}/post_index.parquet",
             *(f"{paths['splits_dir']}/fold_{f}.json" for f in range(5))]
    hashes = {}
    for name in names:
        actual = file_sha256(project_path(config.project_root, name))
        if actual != protected.get(name):
            raise ValueError(f"Frozen data input changed: {name}")
        hashes[name] = actual
    users = pd.read_parquet(project_path(config.project_root, names[0]), columns=["user_id", "label"])
    posts = pd.read_parquet(project_path(config.project_root, names[1]), columns=POST_COLUMNS)
    outers = [json.loads(project_path(config.project_root, n).read_text(encoding="utf-8")) for n in names[2:]]
    if any(o.get("config_hash") != config.base_config_hash for o in outers):
        raise ValueError("Stale original split config hash")
    return users, posts, outers, hashes


def _split_record(config, users, outer, f, hashes):
    return {**_binding(config), "source_sha256": hashes,
            "split_implementation": {"sklearn": sklearn.__version__, "numpy": np.__version__},
            **make_inner_splits(users, outer, f)}


def _stats_record(config, users, posts, split, j):
    return {**_binding(config), "outer_fold": split["outer_fold"], "inner_fold": j,
            "fit_role": "train", "split_sha256": digest(split),
            **fit_training_statistics(users, posts, split["inner"][j]["roles"]["train"],
                                      quantile=config.base_values["time"]["clipping_quantile"],
                                      max_posts=config.base_values["data"]["max_posts"])}


def _payload(value) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _save_all(root: Path, artifacts: dict[str, dict]) -> None:
    """Preflight every output before creating any; reruns must be byte-identical."""
    pending = []
    for relative, value in artifacts.items():
        path = project_path(root, relative, allowed_prefix=DATA_ROOT)
        content = _payload(value)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite differing optimization data: {relative}")
        else:
            pending.append((path, content))
    for path, content in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)


def build_optimization_data(config: OptimizationConfig) -> dict:
    users, posts, outers, hashes = _inputs(config)
    if Counter(u for o in outers for u in o["test_user_ids"]) != Counter({u: 1 for u in users.user_id}):
        raise ValueError("Original outer tests must cover each user once")
    artifacts, summaries = {}, []
    for f, outer in enumerate(outers):
        split = _split_record(config, users, outer, f, hashes)
        artifacts[f"{DATA_ROOT}/splits/outer_{f}.json"] = split
        for j in range(3):
            stats = _stats_record(config, users, posts, split, j)
            artifacts[f"{DATA_ROOT}/time_stats/outer_{f}_inner_{j}.json"] = stats
            summaries.append({"outer_fold": f, "inner_fold": j,
                              "counts": split["inner"][j]["counts"],
                              "base_positive_weight": stats["base_positive_weight"],
                              "log_gap_quantile": stats["time_normalizer"]["log_gap_quantile"]})
    manifest = {**_binding(config), "status": "pass", "training_started": False,
                "outer_folds": 5, "inner_splits": 15,
                "calibration_test_labels_used_for": "membership audit counts only",
                "final_train_statistics": "fit afresh on S during final training; never reuse inner statistics",
                "sha256": {p: hashlib.sha256(_payload(v)).hexdigest() for p, v in artifacts.items()},
                "summary": summaries}
    artifacts[f"{DATA_ROOT}/manifest.json"] = manifest
    _save_all(config.project_root, artifacts)
    return manifest


@dataclass(frozen=True)
class InnerDataContext:
    """Loaded via load_inner_context: split/stat hashes checked against live inputs."""
    outer_fold: int
    inner_fold: int
    split_sha256: str
    statistics_sha256: str
    _roles_json: str
    _statistics_json: str

    @property
    def statistics(self) -> dict:
        return json.loads(self._statistics_json)

    @property
    def time_normalizer(self) -> TimeNormalizer:
        return TimeNormalizer.from_dict(self.statistics["time_normalizer"])

    @property
    def base_positive_weight(self) -> float:
        return self.statistics["base_positive_weight"]

    def user_ids(self, role: str) -> list[str]:
        if role not in {"train", "early_stop", "holdout"}:
            raise ValueError("Inner adapter exposes only train, early_stop and holdout")
        return json.loads(self._roles_json)[role]

    def dataset(self, store, role: str):
        # Reuse the original sequence ordering, right padding, masks and transform.
        from eot.dataset import UserSequenceDataset
        return UserSequenceDataset(store, self.user_ids(role), self.time_normalizer)


def load_inner_context(config: OptimizationConfig, outer_fold: int, inner_fold: int) -> InnerDataContext:
    if (type(outer_fold) is not int or outer_fold not in range(5)
            or type(inner_fold) is not int or inner_fold not in range(3)):
        raise ValueError("Invalid outer/inner fold")
    manifest = json.loads(project_path(config.project_root, f"{DATA_ROOT}/manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(k) != v for k, v in _binding(config).items()):
        raise ValueError("Stale optimization data manifest")
    users, posts, outers, hashes = _inputs(config)
    expected_split = _split_record(config, users, outers[outer_fold], outer_fold, hashes)
    expected_stats = _stats_record(config, users, posts, expected_split, inner_fold)
    for name, expected in ((f"splits/outer_{outer_fold}.json", expected_split),
                           (f"time_stats/outer_{outer_fold}_inner_{inner_fold}.json", expected_stats)):
        relative = f"{DATA_ROOT}/{name}"
        path = project_path(config.project_root, relative, allowed_prefix=DATA_ROOT)
        if (file_sha256(path) != manifest["sha256"].get(relative)
                or path.read_bytes() != _payload(expected)):
            raise ValueError(f"Stale or mismatched split/statistics: {name}")
    return InnerDataContext(outer_fold, inner_fold, digest(expected_split), digest(expected_stats),
                            canonical_json(expected_split["inner"][inner_fold]["roles"]),
                            canonical_json(expected_stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = build_optimization_data(load_optimization_config(args.config))
    print(canonical_json({"status": manifest["status"], "inner_splits": manifest["inner_splits"],
                          "manifest": f"{DATA_ROOT}/manifest.json"}))


if __name__ == "__main__":
    main()
