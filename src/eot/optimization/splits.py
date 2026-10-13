"""Protocol-v1 user splits. No model fitting or threshold selection."""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import canonical_json


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def checked_ids(values, name: str) -> list[str]:
    ids = list(values)
    if not ids or any(not isinstance(u, str) or not u or "\n" in u for u in ids):
        raise ValueError(f"{name}: expected nonempty string user IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate user IDs")
    return sorted(ids)


def label_map(users: pd.DataFrame, ids) -> dict[str, int]:
    """Inspect labels of requested users only; heldout labels cannot enter a fit."""
    ids = checked_ids(ids, "label users")
    selected = users.loc[users.user_id.isin(ids), ["user_id", "label"]]
    if selected.user_id.duplicated().any() or set(selected.user_id) != set(ids):
        raise ValueError("Missing or duplicate requested users")
    if not selected.label.isin([0, 1]).all() or selected.label.map(lambda x: isinstance(x, bool)).any():
        raise ValueError("Requested labels must be binary 0/1")
    return {u: int(y) for u, y in selected.itertuples(index=False, name=None)}


def counts(labels: dict, ids) -> dict:
    positive = sum(labels[u] for u in ids)
    return {"users": len(ids), "negative": len(ids) - positive, "positive": positive}


def disjoint(roles: dict[str, list[str]]) -> None:
    for role, ids in roles.items():
        checked_ids(ids, role)
    for left, right in combinations(roles, 2):
        if set(roles[left]) & set(roles[right]):
            raise ValueError(f"Overlapping user roles: {left}, {right}")


def make_inner_splits(users: pd.DataFrame, outer: dict, outer_fold: int) -> dict:
    """Sort S, use seeds 10042+f and 20042+10*f+j exactly as locked in O02.

    Calibration/test labels are read only for descriptive counts. They never
    enter either splitter. This function does not inspect time gaps.
    """
    if type(outer_fold) is not int or outer_fold not in range(5) or outer.get("fold") != outer_fold:
        raise ValueError("Invalid outer fold identity")
    roles = {"search": checked_ids(outer["train_user_ids"], "search"),
             "calibration": checked_ids(outer["validation_user_ids"], "calibration"),
             "test": checked_ids(outer["test_user_ids"], "test")}
    disjoint(roles)
    all_ids = sum(roles.values(), [])
    if set(all_ids) != set(users.user_id):
        raise ValueError("Outer roles must partition the complete user inventory")
    labels = label_map(users, all_ids)
    search = np.asarray(roles["search"])
    y = np.asarray([labels[u] for u in search])
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=10042 + outer_fold)
    inner = []
    for j, (remainder, holdout) in enumerate(splitter.split(search, y)):
        g, e = train_test_split(search[remainder], test_size=0.125,
                                stratify=y[remainder], shuffle=True,
                                random_state=20042 + 10 * outer_fold + j)
        current = {"train": sorted(g.tolist()), "early_stop": sorted(e.tolist()),
                   "holdout": sorted(search[holdout].tolist())}
        disjoint({**current, "calibration": roles["calibration"], "test": roles["test"]})
        if set(sum(current.values(), [])) != set(search):
            raise ValueError("Inner roles do not partition S")
        summary = {name: counts(labels, ids) for name, ids in current.items()}
        if any(min(c["positive"], c["negative"]) == 0 for c in summary.values()):
            raise ValueError("Every inner role must contain both classes")
        inner.append({"inner_fold": j, "early_stop_seed": 20042 + 10 * outer_fold + j,
                      "roles": current, "counts": summary,
                      "user_ids_sha256": {name: digest(ids) for name, ids in current.items()}})
    coverage = Counter(u for item in inner for u in item["roles"]["holdout"])
    if coverage != Counter({u: 1 for u in search}):
        raise ValueError("Each search user must be held out exactly once")
    return {"outer_fold": outer_fold, "inner_split_seed": 10042 + outer_fold,
            "inner_folds": 3, "early_stop_fraction": 0.125,
            "outer_roles": roles, "outer_counts": {k: counts(labels, v) for k, v in roles.items()},
            "inner": inner, "pairwise_disjoint": True, "holdout_coverage_exactly_once": True}
