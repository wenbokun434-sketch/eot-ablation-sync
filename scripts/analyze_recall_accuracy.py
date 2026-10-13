"""Post-hoc threshold diagnostics using frozen predictions; never train or tune on test.

Run from any directory with Python, NumPy and pandas. Alternative final-classification
thresholds are chosen within each original validation fold. These validation users
also supported checkpoint selection; results are exploratory, not a fresh evaluation.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/recall_accuracy"
GRID = np.linspace(0.0, 1.0, 201)
POLICIES = ("original", "final_f1", "final_accuracy", "recall70_accuracy", "fixed_050")


def score(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / len(y),
                recall=recall, precision=tp / (tp + fp) if tp + fp else 0.0,
                f1=2 * tp / (2 * tp + fp + fn),
                balanced_accuracy=(recall + specificity) / 2)


def select(validation, original):
    rows = [{"threshold": float(t), **score(validation.label, validation.probability >= t)}
            for t in GRID]
    return {
        "original": float(original),
        "final_f1": max(rows, key=lambda r: (r["f1"], r["accuracy"], r["threshold"]))["threshold"],
        "final_accuracy": max(rows, key=lambda r: (r["accuracy"], r["recall"], r["threshold"]))["threshold"],
        "recall70_accuracy": max((r for r in rows if r["recall"] >= 0.7),
                                 key=lambda r: (r["accuracy"], r["recall"], r["threshold"]))["threshold"],
        "fixed_050": 0.5,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT / "outputs/predictions/oof_final_predictions.parquet",
                    ROOT / "outputs/metrics/fold_thresholds.json"]
    test = pd.read_parquet(source_paths[0]).sort_values("user_id").reset_index(drop=True)
    assert test.user_id.is_unique and len(test) == 909
    assert np.array_equal(test.final_prediction, (test.probability >= test.threshold).astype(int))
    locked = json.loads(source_paths[1].read_text(encoding="utf-8"))
    chosen, validation_frames = {}, {}
    # Finish all validation-only selection before scoring any alternative on test.
    for item in locked["folds"]:
        fold = item["fold"]
        path = ROOT / f"outputs/predictions/fold_ensembles/fold_{fold}_validation_final.parquet"
        source_paths.append(path)
        validation = pd.read_parquet(path)
        assert validation.user_id.is_unique
        assert set(validation.user_id).isdisjoint(set(test.loc[test.fold == fold, "user_id"]))
        assert (validation.fold == fold).all()
        chosen[fold] = select(validation, item["threshold"])
        validation_frames[fold] = validation
    threshold_rows, fold_rows, pooled_rows = [], [], []
    predictions = {}
    for policy in POLICIES:
        threshold = test.fold.map({f: v[policy] for f, v in chosen.items()}).to_numpy()
        pred = (test.probability.to_numpy() >= threshold).astype(int)
        predictions[policy] = pred
        pooled_rows.append({"policy": policy, **score(test.label, pred)})
        for fold, values in chosen.items():
            val = validation_frames[fold]
            threshold_rows.append({"policy": policy, "fold": fold, "threshold": values[policy],
                                   "users": len(val), "positive_users": int(val.label.sum()),
                                   **score(val.label, val.probability >= values[policy])})
            mask = (test.fold == fold).to_numpy()
            fold_rows.append({"policy": policy, "fold": fold, "threshold": values[policy],
                              **score(test.label.to_numpy()[mask], pred[mask])})
    # Same resampled users in both policies: conditional paired user bootstrap.
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(test), size=(2000, len(test)))
    y = test.label.to_numpy()
    positive_n = (y[indices] == 1).sum(axis=1)
    baseline = predictions["original"]
    paired = []
    for policy, pred in predictions.items():
        if policy == "original":
            continue
        correctness_delta = (pred == y).astype(int) - (baseline == y).astype(int)
        positive_delta = ((pred - baseline) * (y == 1))
        accuracy_deltas = correctness_delta[indices].mean(axis=1)
        recall_deltas = positive_delta[indices].sum(axis=1) / positive_n
        paired.append({
            "policy": policy,
            "delta_accuracy": float(correctness_delta.mean()),
            "delta_accuracy_ci_lower": float(np.quantile(accuracy_deltas, .025)),
            "delta_accuracy_ci_upper": float(np.quantile(accuracy_deltas, .975)),
            "delta_recall": float(positive_delta.sum() / y.sum()),
            "delta_recall_ci_lower": float(np.quantile(recall_deltas, .025)),
            "delta_recall_ci_upper": float(np.quantile(recall_deltas, .975)),
            "corrected_errors": int(((baseline != y) & (pred == y)).sum()),
            "new_errors": int(((baseline == y) & (pred != y)).sum()),
        })
    fn = test.loc[(test.label == 1) & (test.final_prediction == 0)]
    user_path = ROOT / "data/processed/user_sequences.parquet"
    source_paths.append(user_path)
    users = pd.read_parquet(user_path, columns=["user_id", "was_truncated"])
    joined = test.merge(users, on="user_id", validate="one_to_one")
    assert len(joined) == len(test)
    groups = {"positive": joined.loc[joined.label == 1],
              "false_negative": joined.loc[(joined.label == 1) & (joined.final_prediction == 0)]}
    diagnostic = {
        "sequence_information": {
            name: {"users": len(frame), "truncated_users": int(frame.was_truncated.sum()),
                   "users_below_32_posts": int((frame.length < 32).sum())}
            for name, frame in groups.items()
        },
        "false_negative_users": len(fn),
        "false_negative_probability_quantiles": fn.probability.quantile([0, .25, .5, .75, 1]).to_dict(),
        "false_negative_score_at_least_050": int((fn.probability >= .5).sum()),
        "false_negative_already_early_alerted": int(fn.alert_prediction.sum()),
        "baseline_by_fold": fold_rows[:5],
        "positive_sequence_lengths": test.loc[test.label == 1].length.describe().to_dict(),
        "false_negative_sequence_lengths": fn.length.describe().to_dict(),
    }
    for name, rows in [("oof_policy_comparison", pooled_rows), ("validation_thresholds", threshold_rows),
                       ("test_fold_comparison", fold_rows), ("paired_differences", paired)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    manifest = {
        "analysis_type": "post_hoc_exploratory_final_classification_only",
        "config_hash": locked["config_hash"],
        "training_executed": False, "formal_predictions_modified": False,
        "selection": "Each original validation fold only; grid 0 to 1 step .005; fixed_050 is a reference.",
        "policies": {
            "final_f1": "Maximize final F1, then accuracy, then threshold.",
            "final_accuracy": "Maximize final accuracy, then recall, then threshold.",
            "recall70_accuracy": "Require validation recall >= .70; maximize accuracy, then recall, then threshold.",
        },
        "limitations": [
            "Existing test results informed the question; this is not a preregistered confirmatory experiment.",
            "Validation folds were also used for checkpoint selection and have only ten positives each.",
            "Multiple policies are reported; intervals have no multiplicity adjustment.",
            "Bootstrap conditions on frozen models and chosen thresholds, without reselection or retraining.",
            "Never pool all folds' validation records to select one threshold: cross-fold train/test users overlap.",
            "Early alert policies remain unchanged and are not rescored with final thresholds.",
        ],
        "bootstrap": {"method": "paired_user_percentile", "iterations": 2000, "seed": 42},
        "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(pooled_rows).to_string(index=False))
    print(pd.DataFrame(threshold_rows).pivot(index="fold", columns="policy", values="threshold").to_string())
    print(pd.DataFrame(paired).to_string(index=False))
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
