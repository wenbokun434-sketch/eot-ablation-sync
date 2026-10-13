"""Portable result-mode setup and presentation helpers for the final notebook."""
from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, load_config
from .evaluation import _file_sha256
from .reporting import read_json, validate_visualizations


def require_files(root: Path, artifacts: dict[str, str]) -> None:
    missing = [f"- {name}\n  恢复命令：{command}" for name, command in artifacts.items()
               if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少实验产物，请在项目根目录执行相应命令后重新运行：\n" + "\n".join(missing))


def notebook_config(root: Path, run_training: bool = False) -> ExperimentConfig:
    """Restore the original path only to reconstruct the saved hash in result mode.

    No raw records are accessed in result mode. Training mode always requires the
    user's environment variable and checks its actual contents before work starts.
    """
    source = root / "configs/eot_main.yaml"
    if run_training:
        if not os.environ.get("ERISK_TASK2_ROOT"):
            raise ValueError('开启 RUN_TRAINING 前请设置 os.environ["ERISK_TASK2_ROOT"] = "原始数据根目录"')
        config = load_config(source)
        require_files(config.path("raw_data_root"), {
            str(config.section("paths")["labels_file"]): '设置 ERISK_TASK2_ROOT 为包含标签文件和 all_combined 的目录',
        })
        if not (config.path("raw_data_root") / str(config.section("paths")["conversations_dir"])).is_dir():
            raise FileNotFoundError("缺少原始 all_combined 目录。请检查 ERISK_TASK2_ROOT 后运行 python scripts/audit_raw_data.py")
        return config
    require_files(root, {"outputs/reports/raw_data_audit.json": "python scripts/audit_raw_data.py"})
    audit = read_json(root / "outputs/reports/raw_data_audit.json")
    original = os.environ.get("ERISK_TASK2_ROOT")
    try:
        os.environ["ERISK_TASK2_ROOT"] = audit["dataset"]["root"]
        config = load_config(source)
    finally:
        if original is None:
            os.environ.pop("ERISK_TASK2_ROOT", None)
        else:
            os.environ["ERISK_TASK2_ROOT"] = original
    if config.config_hash != audit["config_hash"]:
        raise ValueError("当前 YAML 与冻结结果配置不一致。请恢复原配置，或在独立实验目录开启 RUN_TRAINING 重建产物。")
    return config


def preflight_results(config: ExperimentConfig) -> pd.DataFrame:
    root = config.project_root
    artifacts = {
        "outputs/reports/environment.json": "python scripts/report_environment.py",
        "outputs/reports/preprocessing_report.json": "python scripts/preprocess_data.py",
        "data/processed/post_index.parquet": "python scripts/preprocess_data.py",
        "data/processed/user_sequences.parquet": "python scripts/preprocess_data.py",
        "data/embeddings/post_embeddings.npy": "python scripts/extract_embeddings.py --device cuda",
        "data/embeddings/post_embeddings.meta.json": "python scripts/extract_embeddings.py --device cuda",
        "outputs/reports/smoke_training.json": "python scripts/smoke_train.py",
        "outputs/predictions/oof_final_predictions.parquet": "python scripts/evaluate.py --device cuda",
        "outputs/predictions/oof_prefix_predictions.parquet": "python scripts/evaluate.py --device cuda",
        "outputs/reports/evaluation_manifest.json": "python scripts/evaluate.py --device cuda",
        "outputs/metrics/oof_metrics.json": "python scripts/evaluate.py --device cuda",
        "outputs/metrics/fold_thresholds.json": "python scripts/evaluate.py --device cuda",
        "outputs/reports/visualization_manifest.json": "python scripts/visualize_results.py --device cuda",
    }
    for fold in range(int(config.section("cross_validation")["n_splits"])):
        artifacts[f"data/splits/fold_{fold}.json"] = "python scripts/create_splits.py"
        artifacts[f"data/time_stats/fold_{fold}.json"] = "python scripts/build_time_stats.py"
        for seed in config.section("training")["seeds"]:
            cmd = f"python scripts/train.py --fold {fold} --seed {seed} --device cuda"
            artifacts[f"outputs/checkpoints/fold_{fold}_seed_{seed}/best.pt"] = cmd
            artifacts[f"outputs/logs/fold_{fold}_seed_{seed}/run.json"] = cmd
            artifacts[f"outputs/logs/fold_{fold}_seed_{seed}/history.json"] = cmd
    require_files(root, artifacts)
    manifest = read_json(root / "outputs/reports/visualization_manifest.json")
    require_files(root, {name: "python scripts/visualize_results.py --device cuda" for name in manifest["outputs"]})
    # Existing T11 validator replays tables and verifies source, figure and code hashes.
    validate_visualizations(config)
    raw_available = config.path("raw_data_root").is_dir()
    return pd.DataFrame([
        {"检查": "结果、图表与源码哈希", "状态": "通过", "说明": "默认仅读取正式产物；已核验 T11 来源"},
        {"检查": "原始数据目录", "状态": "可访问" if raw_available else "未挂载",
         "说明": "结果浏览无需原始数据；重建请设置 ERISK_TASK2_ROOT 并运行 python scripts/audit_raw_data.py"},
    ])


def configuration_table(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for section, values in config.values.items():
        if section == "paths":
            continue
        for key, value in values.items():
            rows.append({"模块": section, "参数": key, "值": str(value)})
    return pd.DataFrame(rows)


def split_table(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    test_users: list[str] = []
    for fold in range(int(config.section("cross_validation")["n_splits"])):
        split = read_json(config.path("splits_dir") / f"fold_{fold}.json")
        if split["config_hash"] != config.config_hash:
            raise ValueError("划分配置哈希不一致")
        groups = [set(split[f"{name}_user_ids"]) for name in ("train", "validation", "test")]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("同折 train/validation/test 用户重叠")
        test_users.extend(split["test_user_ids"])
        for name, counts in split["counts"].items():
            rows.append({"fold": fold, "split": name, **counts})
    if len(test_users) != len(set(test_users)):
        raise AssertionError("外层测试用户重复")
    return pd.DataFrame(rows)


def training_table(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for fold in range(int(config.section("cross_validation")["n_splits"])):
        for seed in config.section("training")["seeds"]:
            run = read_json(config.path("outputs_dir") / f"logs/fold_{fold}_seed_{seed}/run.json")
            if run["status"] != "complete" or run["config_hash"] != config.config_hash:
                raise AssertionError("训练未完成或配置不匹配")
            rows.append({key: run[key] for key in (
                "fold", "seed", "device", "epochs_completed", "best_epoch", "best_validation_pr_auc", "elapsed_seconds",
            )})
    return pd.DataFrame(rows)


def source_browser(root: Path) -> str:
    """Embed the canonical core source as collapsible read-only display output."""
    modules = ["config", "reproducibility", "raw_data", "audit", "preprocessing", "splitting",
               "embeddings", "time_features", "dataset", "model", "losses", "training",
               "metrics", "evaluation", "reporting", "visualization"]
    parts = ["<p>以下展示直接读取 src/eot/ 的正式源码；执行仍调用这些模块。</p>"]
    for name in modules:
        path = root / f"src/eot/{name}.py"
        parts.append(f"<details><summary><strong>src/eot/{name}.py</strong> · SHA256 {_file_sha256(path)[:12]}</summary>"
                     f"<pre style='max-height:600px;overflow:auto;white-space:pre'>{html.escape(path.read_text(encoding='utf-8'))}</pre></details>")
    return "\n".join(parts)
