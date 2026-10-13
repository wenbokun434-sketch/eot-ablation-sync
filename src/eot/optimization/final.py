"""O14 public adapter: frozen final training plus calibrated O14 calibration.

``final_training.py`` is the actual training adapter whose source hash is bound
into every final run identity. This module re-exports that implementation and
overrides calibration with the O14-compatible path/NaN handling; it does not
modify any frozen O06 module or class.
"""
from .final_training import *  # noqa: F401,F403
from .final_training import (  # noqa: F401
    _Ensemble, _commit_final, _ensemble_frames, _final_path, _fit_final,
    _load_o13_gate, _protected_records, _save_local_bundle, _second_lock,
    _seed_run_record, _write_frame, _write_json, calibrate_final_fold,
    evaluate_final_test, final_run_status, load_outer_test_cohorts,
    predict_final_ids, prepare_final_run, read_final_checkpoint,
    require_outer_test_gate, train_final_run,
)

import math as _math

import numpy as _np
import pandas as _pd

from .comparison import evaluate_selection
from .config import file_sha256 as _file_sha256
from .decisions import (SelectionPredictions, load_selection_scope,
                        select_dual_thresholds)
from .evaluation import build_bundle, frame_digest as _original_frame_digest
from .splits import digest as _digest


# Decision frames use NaN for "no alert position". The frozen JSON digest
# rejects NaN, so O14 persists and hashes them with a deterministic sentinel.
# Finite frames still use the original digest bytes.
def _safe_frame_digest(frame, columns):
    try:
        return _original_frame_digest(frame, columns)
    except ValueError:
        rows = []
        for row in frame[list(columns)].itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, float) and _math.isnan(value):
                    values.append("__nan__")
                elif isinstance(value, float) and _math.isinf(value):
                    values.append("__inf__" if value > 0 else "__-inf__")
                elif isinstance(value, _np.generic):
                    values.append(value.item())
                else:
                    values.append(value)
            rows.append(values)
        return _digest(rows)


def _o14_write_frame(config, relative, frame, columns):
    path = _final_path(config, relative)
    expected = _safe_frame_digest(frame, columns)
    if path.exists():
        if _safe_frame_digest(_pd.read_parquet(path), columns) != expected:
            raise ValueError(f"Refusing to overwrite differing O14 frame: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[list(columns)].to_parquet(path, index=False, compression="zstd")
    return path


# Patch only the frozen final-training globals used by its evaluation function.
import eot.optimization.final_training as _final_training

_final_training._write_frame = _o14_write_frame
_final_training.frame_digest = _safe_frame_digest


def calibrate_final_fold(config, *, outer_fold: int, device_name: str = "cpu") -> dict:
    """Fit actual final models on C and seal one fold's post-calibration lock.

    This is the corrected O14 adapter: the frozen helper used a short
    ``final/...`` path and called the ``labels`` property as a method. Both are
    fixed here without touching any frozen module/class.
    """
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in 0..4")
    store = SequenceStore(_legacy_read_config(config))
    final_frames, prefix_frames, runs, run_records = [], [], [], []
    for seed in FINAL_SEEDS:
        plan = prepare_final_run(config, outer_fold=outer_fold, seed=seed)
        state, manifest = read_final_checkpoint(plan)
        if state["status"] != "complete" or state["epoch"] != plan.o13_record["epochs"]:
            raise ValueError(f"Final run is not complete for outer {outer_fold} seed {seed}")
        final, prefix = predict_final_ids(
            plan, store, plan.context.user_ids("calibration"), device_name=device_name)
        _o14_write_frame(config, f"calibration/outer_{outer_fold}/seed_{seed}/final.parquet",
                         final, FINAL_COLUMNS)
        _o14_write_frame(config, f"calibration/outer_{outer_fold}/seed_{seed}/prefix.parquet",
                         prefix, PREFIX_COLUMNS)
        seed_run, run_record = _seed_run_record(plan, manifest)
        runs.append(seed_run)
        run_records.append(run_record)
        final_frames.append(final)
        prefix_frames.append(prefix)

    bundle = build_bundle(
        config, stage="final", outer_fold=outer_fold, inner_fold=None,
        role="calibration", seeds=FINAL_SEEDS, runs=runs,
        final_frames=final_frames, prefix_frames=prefix_frames)
    bundle_record = _save_local_bundle(
        config, f"calibration/outer_{outer_fold}/bundle", bundle)

    scope = load_selection_scope(config, role="calibration", outer_fold=outer_fold)
    seed_models = tuple(sorted((seed, record["checkpoint_sha256"])
                               for seed, record in zip(FINAL_SEEDS, run_records)))
    reference = SelectionPredictions(
        candidate_id=config.candidate_id, config_sha256=config.config_hash,
        seed_model_sha256=seed_models, sequences=tuple(bundle.sequences()))
    selection = select_dual_thresholds(config, scope, reference, reference)
    selection_path = selection.save(
        config, f"outputs/optimization/v1/final/calibration/outer_{outer_fold}/threshold_selection.json")
    evaluation = evaluate_selection(config, selection, a_bundle=bundle)
    summary_path = _write_json(
        config, f"calibration/outer_{outer_fold}/calibration_summary.json",
        evaluation.record)

    manifest = selection.manifest
    r1, r2, candidate = manifest["R1"], manifest["R2"], manifest["candidate"]
    candidate_usable = (
        candidate.get("threshold_final") is not None
        and candidate.get("status") != "infeasible"
        and int(candidate["counts"]["correct"]) >= int(r2["counts"]["correct"])
        and int(candidate["counts"]["tp"]) > int(r2["counts"]["tp"]))
    if config.candidate_id == "A" or not candidate_usable:
        fallback = "A_reference_R2"
        opt_source = "A_reference_R2"
    else:
        fallback = f"candidate_{config.candidate_id}"
        opt_source = f"candidate_{config.candidate_id}"

    labels = _np.asarray(bundle.labels, dtype=_np.int64)
    lock = {
        "schema_version": 1,
        "task": "O14",
        "outer_fold": outer_fold,
        "selected_candidate": config.candidate_id,
        "o13_parent_lock_sha256": None,
        "o13_parent_lock_file_sha256": None,
        "final_runs": run_records,
        "selection": {
            "cohort_user_ids_sha256": _digest(scope.metadata["user_ids"]),
            "cohort_users": len(scope.metadata["user_ids"]),
        },
        "calibration": {
            "bundle": bundle_record,
            "threshold_selection": {
                "relative_path": selection_path.relative_to(config.project_root).as_posix(),
                "file_sha256": _file_sha256(selection_path),
                "content_sha256": _digest(manifest),
            },
            "calibration_summary": {
                "relative_path": summary_path.relative_to(config.project_root).as_posix(),
                "file_sha256": _file_sha256(summary_path),
                "content_sha256": _digest(evaluation.record),
            },
            "R1": r1,
            "R2": r2,
            "candidate": candidate,
            "fallback": fallback,
            "opt_source": opt_source,
            "class_counts": {
                "negative": int(_np.sum(labels == 0)),
                "positive": int(_np.sum(labels == 1)),
            },
        },
        "test_access": "closed_until_all_five_post_calibration_locks_verified",
    }
    first_plan = prepare_final_run(config, outer_fold=outer_fold, seed=FINAL_SEEDS[0])
    lock["o13_parent_lock_sha256"] = first_plan.parent_lock_sha256
    lock["o13_parent_lock_file_sha256"] = first_plan.parent_lock_file_sha256
    _write_json(config, f"locks/outer_{outer_fold}.json",
                {"sha256": _digest(lock), "lock": lock})
    return {"status": "pass", "outer_fold": outer_fold, "lock": lock,
            "lock_sha256": _digest(lock)}
