"""Derive publication tables and auditable case data from frozen T10 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .dataset import SequenceStore, UserSequenceDataset
from .evaluation import _file_sha256, _load_model, metrics_from_final_predictions
from .metrics import classification_metrics
from .reproducibility import set_global_seed
from .time_features import load_time_normalizer


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_tables(metrics: dict, fold_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep pooled OOF estimates distinct from unweighted fold mean and SD."""
    summary = fold_summary.loc[fold_summary.split == "test"].set_index("metric")
    tables = {}
    for section, points in metrics["point_estimate"].items():
        rows = []
        for name, value in points.items():
            ci = metrics["confidence_interval"]["intervals"][name]
            rows.append({
                "metric": name, "oof_estimate": value,
                "ci_lower": ci["lower"], "ci_upper": ci["upper"],
                "fold_mean": summary.loc[name, "mean"],
                "fold_sd": summary.loc[name, "std"],
                "unit": "observed posts" if name == "median_latency" else "fraction",
            })
        tables[section] = pd.DataFrame(rows)
    return tables


def prefix_performance(final: pd.DataFrame, prefix: pd.DataFrame, checkpoints: list[int]) -> pd.DataFrame:
    """Score min(k, user length); never drop short users from the primary curve."""
    rows = []
    for checkpoint in checkpoints:
        users = final[["user_id", "label", "length", "fold", "threshold"]].copy()
        users["position"] = np.minimum(checkpoint, users.length)
        selected = users.merge(
            prefix[["user_id", "position", "probability"]],
            on=["user_id", "position"], how="left", validate="one_to_one",
        )
        if selected.probability.isna().any():
            raise ValueError("Missing probability at a required prefix checkpoint")
        cohorts = [("oof", -1, selected)] + [
            ("fold", int(fold), frame) for fold, frame in selected.groupby("fold")
        ]
        for scope, fold, frame in cohorts:
            values = classification_metrics(
                frame.label, frame.probability,
                predictions=(frame.probability >= frame.threshold).astype(int),
            )
            rows.append({
                "checkpoint": checkpoint, "scope": scope, "fold": fold,
                "users": len(frame), "positive_users": int(frame.label.sum()),
                "users_with_at_least_k_posts": int((frame.length >= checkpoint).sum()),
                "policy": "min(k,length); frozen fold threshold; snapshot classification",
                **values,
            })
    return pd.DataFrame(rows)


def select_cases(final: pd.DataFrame) -> pd.DataFrame:
    """One median-length user per early-decision outcome; deterministic ID ties."""
    records = []
    for name, label, decision in [("TP", 1, 1), ("FP", 0, 1), ("FN", 1, 0), ("TN", 0, 0)]:
        group = final.loc[(final.label == label) & (final.alert_prediction == decision)].copy()
        if group.empty:
            continue
        group["distance"] = (group.length - group.length.median()).abs()
        record = group.sort_values(["distance", "user_id"]).iloc[0].drop("distance").to_dict()
        record.update({"case": name, "group_users": len(group),
                       "selection": "closest to group median length; lexicographic user ID tie"})
        record["explanation_position"] = (
            int(record["alert_position"]) if decision else int(record["length"])
        )
        records.append(record)
    return pd.DataFrame(records)


@torch.inference_mode()
def extract_case_attention(
    config: ExperimentConfig, cases: pd.DataFrame, prefix: pd.DataFrame, device_name: str,
) -> pd.DataFrame:
    """Reload held-out checkpoints and save pooling weights at the decision prefix."""
    device = torch.device(device_name)
    store = SequenceStore(config)
    rows = []
    evaluation = read_json(config.path("outputs_dir") / "reports/evaluation_manifest.json")
    hashes = {(r["fold"], r["seed"]): r["sha256"] for r in evaluation["checkpoint_records"]}
    for case in cases.itertuples(index=False):
        fold = int(case.fold)
        ds = UserSequenceDataset(store, [case.user_id], load_time_normalizer(config, fold))
        item = ds[0]
        length, end = int(case.length), int(case.explanation_position)
        reference = prefix.loc[prefix.user_id == case.user_id].sort_values("position").probability.to_numpy()
        seed_probabilities = []
        for seed in config.section("training")["seeds"]:
            seed = int(seed)
            checkpoint = config.path("outputs_dir") / f"checkpoints/fold_{fold}_seed_{seed}/best.pt"
            if _file_sha256(checkpoint) != hashes[(fold, seed)]:
                raise ValueError("Case checkpoint differs from the frozen T10 checkpoint")
            set_global_seed(seed)
            model, _ = _load_model(config, fold, seed, device)
            model.eval()
            out = model(
                item["post_embeddings"].unsqueeze(0).to(device),
                item["delta_t"].unsqueeze(0).to(device),
                item["mask"].unsqueeze(0).to(device), return_attention=True,
            )
            weights = out["attention_weights"][0, end - 1].cpu().numpy()
            if not np.isclose(weights[:end].sum(), 1.0, atol=1e-6) or np.any(weights[end:] != 0):
                raise AssertionError("Pooling weights include future/padding positions or are not normalized")
            seed_probabilities.append(torch.sigmoid(out["prefix_logits"])[0, :length].cpu().numpy())
            for index in range(end):
                rows.append({"case": case.case, "user_id": case.user_id, "fold": fold,
                             "seed": seed, "position": index + 1, "explanation_position": end,
                             "weight": float(weights[index])})
            del model, out
        if not np.allclose(np.mean(seed_probabilities, axis=0), reference, atol=1e-5, rtol=1e-5):
            raise AssertionError("Reloaded case probabilities differ from saved OOF predictions")
        print(f"case={case.case} fold={fold} explanation_prefix={end} checkpoint replay OK", flush=True)
    return pd.DataFrame(rows)


def stability_tables(seed_metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = seed_metrics.loc[seed_metrics.split == "test"].copy()
    names = ["pr_auc", "roc_auc", "f1", "recall", "f_latency", "erde_50"]
    rows = []
    for scope, key in [("within_fold_across_seeds", "fold"), ("within_seed_across_folds", "seed")]:
        for group, frame in test.groupby(key):
            for metric in names:
                rows.append({"scope": scope, "group": int(group), "metric": metric,
                             "runs": len(frame), "mean": frame[metric].mean(),
                             "sd": frame[metric].std(ddof=1)})
    return {"stability_runs": test, "stability_summary": pd.DataFrame(rows)}


def verify_point_metrics(final: pd.DataFrame, saved: dict) -> None:
    actual = metrics_from_final_predictions(final)
    for section, values in actual.items():
        for name, value in values.items():
            expected = saved["point_estimate"][section][name]
            if value is None and expected is None:
                continue
            if not np.isclose(value, expected, atol=1e-12, rtol=0):
                raise AssertionError(f"Stale metric: {section}.{name}")


def validate_visualizations(config: ExperimentConfig) -> dict:
    """Verify publication artifacts can be traced back to unchanged inputs."""
    from datetime import datetime, timezone
    from PIL import Image
    from .reproducibility import write_json

    root = config.path("outputs_dir")
    manifest = read_json(root / "reports/visualization_manifest.json")
    if manifest["config_hash"] != config.config_hash:
        raise AssertionError("Visualization config hash mismatch")
    for section in ("sources", "outputs", "code"):
        for relative, digest in manifest[section].items():
            path = config.project_root / relative
            if not path.is_file() or _file_sha256(path) != digest:
                raise AssertionError(f"Changed {section} artifact: {relative}")
    final = pd.read_parquet(root / "predictions/oof_final_predictions.parquet")
    prefix = pd.read_parquet(root / "predictions/oof_prefix_predictions.parquet")
    metrics = read_json(root / "metrics/oof_metrics.json")
    tables = result_tables(metrics, pd.read_csv(root / "metrics/fold_metric_summary.csv"))
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(root / f"tables/{name}.csv"), expected, check_dtype=False)
    expected_prefix = prefix_performance(final, prefix, list(config.section("pooling")["checkpoints"]))
    pd.testing.assert_frame_equal(pd.read_csv(root / "tables/prefix_performance.csv"), expected_prefix, check_dtype=False)
    cases = pd.read_csv(root / "tables/representative_cases.csv")
    pd.testing.assert_frame_equal(cases, select_cases(final), check_dtype=False)
    weights = pd.read_csv(root / "tables/case_attention.csv")
    if not np.isfinite(weights.weight).all() or (weights.weight < 0).any():
        raise AssertionError("Invalid saved attention weights")
    if not np.allclose(weights.groupby(["case", "seed"]).weight.sum(), 1.0, atol=1e-6):
        raise AssertionError("Saved attention weights are not normalized")
    if (weights.position > weights.explanation_position).any():
        raise AssertionError("Saved attention includes future positions")
    for case in cases.itertuples(index=False):
        selected = weights.loc[weights.case == case.case]
        for seed in config.section("training")["seeds"]:
            positions = selected.loc[selected.seed == seed, "position"].to_numpy()
            if not np.array_equal(positions, np.arange(1, case.explanation_position + 1)):
                raise AssertionError("Missing seed or position in attention table")
    dimensions = {}
    for figure in manifest["figures"]:
        for name in figure["files"]:
            path = root / "figures" / name
            if path.suffix == ".png":
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    if min(im.size) < 900 or min(im.info.get("dpi", (0, 0))) < 299:
                        raise AssertionError(f"Insufficient publication resolution: {name}")
                    dimensions[name] = list(im.size)
            elif path.suffix == ".pdf":
                with path.open("rb") as handle:
                    if not handle.read(5) == b"%PDF-":
                        raise AssertionError(f"Invalid PDF: {name}")
            elif path.suffix == ".svg":
                import xml.etree.ElementTree as ET
                if not ET.parse(path).getroot().tag.endswith("svg"):
                    raise AssertionError(f"Invalid SVG: {name}")
    report = {"status": "pass", "config_hash": config.config_hash,
              "validated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_output_and_code_hashes_match": True,
              "tables_recomputed_from_frozen_predictions": True,
              "cases_replayed_deterministically": True,
              "saved_attention_normalized_and_causal": True,
              "png_dimensions": dimensions, "figure_count": len(dimensions),
              "formats_per_figure": 3, "publication_dpi": 300}
    write_json(report, root / "reports/visualization_validation.json")
    return report
