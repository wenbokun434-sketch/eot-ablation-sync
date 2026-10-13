"""Publication figures and result report generated exclusively from formal artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .config import ExperimentConfig
from .evaluation import _file_sha256, validate_evaluation_artifacts
from .reporting import (
    extract_case_attention, prefix_performance, read_json, result_tables,
    select_cases, stability_tables, verify_point_metrics,
)
from .reproducibility import write_json


COLORS = ["#126E82", "#D47A35", "#7062A0", "#429B79", "#C34D62"]
LABELS = {0: "Control", 1: "Depression"}


class FigureWriter:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def save(self, fig, name: str, caption: str, sources: list[str]) -> None:
        paths = []
        for extension in ("png", "pdf", "svg"):
            path = self.root / f"{name}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            if not path.is_file() or path.stat().st_size < 1000:
                raise AssertionError(f"Empty figure: {path}")
            paths.append(path.name)
        self.records.append({"name": name, "caption": caption, "files": paths, "sources": sources})
        plt.close(fig)


def _figure(title: str, subtitle: str, rows=1, cols=1, size=(11, 4.8)):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    fig.suptitle(title, x=0.07, ha="left", fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.07, 0.925, subtitle, fontsize=9, color="#56616B")
    fig.subplots_adjust(left=0.075, right=0.965, top=0.82, bottom=0.15, wspace=0.28, hspace=0.55)
    return fig, axes


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.17)
    ax.set_axisbelow(True)


def _dataset_figures(writer, final, users, posts):
    truncated = int(users.was_truncated.sum())
    fig, axes = _figure("01  |  Cohort and observed histories", f"Independent EOT cohort; earliest 256 valid target-user writings; {truncated} histories truncated", cols=2)
    ax = axes[0, 0]
    counts = final.label.value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar([LABELS[0], LABELS[1]], counts, color=COLORS[:2], width=0.5)
    ax.bar_label(bars, labels=[f"{n:,} ({n/len(final):.1%})" for n in counts], padding=5)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set(xlabel="User label", ylabel="Number of users")
    ax = axes[0, 1]
    for label in (0, 1):
        group = final.loc[final.label == label]
        ax.hist(group.length, bins=np.linspace(0, 256, 17), histtype="step", linewidth=2,
                weights=np.full(len(group), 100 / len(group)), color=COLORS[label],
                label=f"{LABELS[label]} (n={len(group)})")
    ax.set(xlabel="Observed writings per user (posts)", ylabel="Users within class (%)")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        _clean(ax)
    writer.save(fig, "01_cohort", "独立用户类别分布与截断后的有效序列长度分布。直方图按类别归一化。",
                ["predictions/oof_final_predictions.parquet", "data/processed/user_sequences.parquet"])

    gaps = posts.loc[posts.sequence_index > 0, "delta_days"].to_numpy()
    fig, axes = _figure("02  |  Irregular time between writings", f"{len(gaps):,} non-initial gaps; zero gaps retained; units: days", cols=2)
    for ax, values, label in zip(axes.flat, [gaps, np.log1p(gaps)],
                               ["Raw time gap (days; symlog axis)", "log(1 + gap in days)"]):
        values = np.sort(values)
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=COLORS[0], linewidth=2)
        ax.set(xlabel=label, ylabel="Cumulative fraction of gaps", ylim=(0, 1.02))
        _clean(ax)
    axes[0, 0].set_xscale("symlog", linthresh=1e-3)
    axes[0, 0].set_xlim(left=0)
    axes[0, 0].set_xticks([0, 0.001, 0.1, 1, 10, 100, 1000])
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0, 1].set_xlim(left=0)
    writer.save(fig, "02_time_gaps", "相邻有效帖时间间隔的经验累积分布；排除人为设置为零的首帖间隔，保留真实零间隔，无尾部裁剪。",
                ["data/processed/post_index.parquet"])


def _training_figure(writer, histories, runs, folds, seeds):
    fig, axes = _figure("03  |  Training and checkpoint selection", "Each column is one fold; colored lines are seeds; markers identify the saved best checkpoint",
                        rows=2, cols=len(folds), size=(16, 7.5))
    for col, fold in enumerate(folds):
        for i, seed in enumerate(seeds):
            h = histories.loc[(histories.fold == fold) & (histories.seed == seed)]
            best = int(runs.loc[(runs.fold == fold) & (runs.seed == seed), "best_epoch"].iloc[0])
            axes[0, col].plot(h.epoch, h.train_loss, color=COLORS[i], label=f"Seed {seed}")
            axes[0, col].plot(h.epoch, h.validation_loss, color=COLORS[i], linestyle="--", alpha=0.65)
            axes[1, col].plot(h.epoch, h.validation_pr_auc, color=COLORS[i])
            best_row = h.loc[h.epoch == best].iloc[0]
            axes[1, col].scatter([best], [best_row.validation_pr_auc], color=COLORS[i], s=28, zorder=5)
        axes[0, col].set_title(f"Fold {fold}", fontsize=11)
        axes[0, col].set_xlabel("Epoch")
        axes[1, col].set(xlabel="Epoch", ylim=(0, 1.02))
        axes[0, col].set_ylim(bottom=0)
        axes[0, col].legend(fontsize=7)
        for row in range(2):
            _clean(axes[row, col])
    axes[0, 0].set_ylabel("Weighted total loss\nsolid: train / dashed: validation")
    axes[1, 0].set_ylabel("Validation PR-AUC (AP)")
    writer.save(fig, "03_training", "15 次正式训练曲线。实线训练损失、虚线验证损失；圆点为按验证 AP 选出的最佳 epoch。早停后不补齐曲线。",
                ["logs/fold_*_seed_*/history.json", "logs/fold_*_seed_*/run.json"])


def _classification_figures(writer, final, metrics, fold_metrics):
    fig, axes = _figure("04  |  Held-out discrimination", f"OOF n={len(final):,}; three-seed mean probability within each test fold; thin lines: individual folds", cols=2)
    for fold, frame in final.groupby("fold"):
        fpr, tpr, _ = roc_curve(frame.label, frame.probability)
        precision, recall, _ = precision_recall_curve(frame.label, frame.probability)
        axes[0, 0].plot(fpr, tpr, color=COLORS[fold], alpha=0.35, linewidth=1)
        axes[0, 1].step(recall, precision, where="post", color=COLORS[fold], alpha=0.35, linewidth=1)
    fpr, tpr, _ = roc_curve(final.label, final.probability)
    precision, recall, _ = precision_recall_curve(final.label, final.probability)
    point = metrics["point_estimate"]["classification"]
    ci = metrics["confidence_interval"]["intervals"]
    axes[0, 0].plot(fpr, tpr, color=COLORS[0], linewidth=2.5,
                   label=f"OOF AUC {point['roc_auc']:.3f}\n95% CI [{ci['roc_auc']['lower']:.3f}, {ci['roc_auc']['upper']:.3f}]")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#9AA3AA", label="Chance")
    axes[0, 1].step(recall, precision, where="post", color=COLORS[0], linewidth=2.5,
                   label=f"OOF AP {point['pr_auc']:.3f}\n95% CI [{ci['pr_auc']['lower']:.3f}, {ci['pr_auc']['upper']:.3f}]")
    axes[0, 1].axhline(final.label.mean(), linestyle="--", color="#9AA3AA", label=f"Prevalence {final.label.mean():.3f}")
    axes[0, 0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[0, 1].set(xlabel="Recall", ylabel="Precision")
    for ax in axes.flat:
        ax.set(xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(loc="lower right" if ax is axes[0, 0] else "lower left", fontsize=8)
        _clean(ax)
    writer.save(fig, "04_roc_pr", "OOF ROC 和 PR 曲线；细线为五折，粗线为合并 OOF。PR-AUC 使用 average precision (AP)，非梯形积分。区间对应单个标量指标，不是逐点曲线置信带。",
                ["predictions/oof_final_predictions.parquet", "metrics/oof_metrics.json"])

    fig, axes = _figure("05  |  Classification and early alerts", f"Same {len(final):,} held-out users; fold-specific thresholds locked on validation; counts and row percentages", cols=2)
    for ax, column, title in zip(axes.flat, ["final_prediction", "alert_prediction"], ["Final-sequence classification", "First sustained early alert"]):
        cm = confusion_matrix(final.label, final[column], labels=[0, 1])
        fractions = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}\n{fractions[i,j]:.1%}", ha="center", va="center",
                        fontsize=14, color="white" if fractions[i,j] > 0.55 else "#243343")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=[LABELS[0], LABELS[1]],
               yticklabels=[LABELS[0], LABELS[1]], xlabel="Predicted label", ylabel="True label", title=title)
        ax.grid(False)
    writer.save(fig, "05_confusion", "完整序列分类与连续两次越阈告警的混淆矩阵。颜色按真实类别逐行归一化，数字同时给出人数与比例。",
                ["predictions/oof_final_predictions.parquet"])


def _prefix_figure(writer, performance):
    oof = performance.loc[performance.scope == "oof"]
    fig, axes = _figure("06  |  Performance as more writings arrive", "Fixed OOF cohort; use position min(k, length); frozen fold thresholds; bands: fold mean +/- 1 SD", cols=3, size=(13, 5.2))
    for ax, metric, title in zip(axes.flat, ["pr_auc", "f1", "recall"], ["PR-AUC (AP)", "Snapshot F1", "Snapshot recall"]):
        folds = performance.loc[performance.scope == "fold"].groupby("checkpoint")[metric].agg(["mean", "std"])
        ax.fill_between(folds.index, np.maximum(0, folds["mean"] - folds["std"]),
                        np.minimum(1, folds["mean"] + folds["std"]), alpha=0.14, color=COLORS[0])
        ax.plot(folds.index, folds["mean"], linestyle="--", color=COLORS[0], alpha=0.6, label="Fold mean +/- SD")
        ax.plot(oof.checkpoint, oof[metric], "o-", color=COLORS[0], label="Pooled OOF")
        ax.set_xscale("log", base=2)
        ax.set(xticks=oof.checkpoint, xlabel="Observed-writing budget k (posts)", ylabel=title, ylim=(0, 1.02))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=7)
        _clean(ax)
    counts = ";  ".join(f"{int(r.checkpoint)}: {int(r.users_with_at_least_k_posts)}" for r in oof.itertuples())
    fig.text(0.07, 0.025, f"Users with at least k posts (k: n): {counts}", fontsize=8, color="#56616B")
    fig.subplots_adjust(bottom=0.23)
    writer.save(fig, "06_prefix_performance", "固定 OOF 人群的前缀性能：短用户采用 min(k,L) 的最后概率，未丢弃也未补造帖子。F1/Recall 是当前前缀分类而非不可撤销告警。阴影为五折均值±标准差，不是置信区间。",
                ["predictions/oof_prefix_predictions.parquet", "tables/prefix_performance.csv"])


def _latency_figure(writer, final):
    detected = final.loc[(final.label == 1) & (final.alert_prediction == 1), "alert_position"].sort_values()
    positive = int(final.label.sum())
    fig, axes = _figure("07  |  Early detection timing", f"True positive alerts: {len(detected)}/{positive}; missed positive users: {positive-len(detected)}; delay measured in observed posts", cols=2)
    axes[0, 0].hist(detected, bins=[1, 2, 4, 8, 16, 32, 64, 128, 257], color=COLORS[0], edgecolor="white")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].axvline(detected.median(), color=COLORS[1], linestyle="--", label=f"TP median: {detected.median():g} posts")
    axes[0, 0].set(xlabel="First alert position (posts)", ylabel="True positive users")
    axes[0, 0].legend(fontsize=8)
    x = np.r_[0, detected.to_numpy(), 256]
    y = np.r_[0, np.arange(1, len(detected) + 1) / positive, len(detected) / positive]
    axes[0, 1].step(x, y, where="post", color=COLORS[0], linewidth=2)
    axes[0, 1].set(xlabel="Observed-writing budget (posts)", ylabel="Fraction of ALL positive users detected", ylim=(0, 1), xlim=(0, 256))
    for ax in axes.flat:
        _clean(ax)
    writer.save(fig, "07_detection_latency", f"左图仅真阳性告警延迟分布；右图以全体真实正类为分母，未检测用户保留在分母中。中位数第{detected.median():g}帖仅描述检出的正类。",
                ["predictions/oof_final_predictions.parquet"])


def _case_figures(writer, cases, case_data, attention):
    fig, axes = _figure("08  |  Illustrative risk trajectories", "One median-length case per early outcome; selected deterministically; markers show the first sustained alert", rows=2, cols=2, size=(12, 8))
    for ax, case in zip(axes.flat, cases.itertuples(index=False)):
        data = case_data.loc[case_data.case == case.case].sort_values("position")
        ax.plot(data.elapsed_days, data.probability, color=COLORS[0], linewidth=1.6, marker=".", markersize=2)
        ax.axhline(case.threshold, linestyle="--", color=COLORS[1], label=f"Threshold {case.threshold:.3f}")
        if pd.notna(case.alert_position):
            alert = data.loc[data.position == case.alert_position].iloc[0]
            ax.scatter([alert.elapsed_days], [alert.probability], marker="*", s=120, color=COLORS[4], zorder=5, label=f"Alert at post {int(case.alert_position)}")
        ax.set(title=f"{case.case} | Fold {case.fold} | {case.length} posts", xlabel="Days since first observed writing",
               ylabel="Prefix risk probability", ylim=(0, 1.04))
        ax.legend(fontsize=8, loc="lower right")
        _clean(ax)
    writer.save(fig, "08_risk_trajectories", "按早期 TP/FP/FN/TN 各组序列长度中位数最近、用户 ID 升序破同分选例；横轴为真实经过天数。案例用于描述，不能代替总体统计。",
                ["tables/representative_cases.csv", "tables/case_trajectories.csv"])

    fig, axes = _figure("09  |  Pooling weights and time gaps", "Weights at alert prefix (TP/FP) or final prefix (FN/TN); mean across seeds; descriptive attention, not causal attribution", rows=2, cols=2, size=(12, 8))
    for ax, case in zip(axes.flat, cases.itertuples(index=False)):
        weights = attention.loc[attention.case == case.case].groupby("position").weight.agg(["mean", "std"])
        gaps = case_data.loc[(case_data.case == case.case) & (case_data.position <= case.explanation_position)].sort_values("position")
        ax.plot(weights.index, weights["mean"], color=COLORS[0], label="Mean pooling weight")
        ax.fill_between(weights.index, np.maximum(0, weights["mean"]-weights["std"]),
                        weights["mean"]+weights["std"], color=COLORS[0], alpha=0.15, label="Across-seed +/- SD")
        ax.set(title=f"{case.case} | Pooling at post {case.explanation_position}", xlabel="Observed writing position (posts)", ylabel="Normalized pooling weight", ylim=(0, None))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        twin = ax.twinx()
        twin.plot(gaps.position, gaps.delta_days, color=COLORS[1], linestyle=":", alpha=0.85, label="Raw gap (days)")
        twin.set_yscale("symlog", linthresh=0.01)
        twin.set_ylabel("Gap (days; symlog)", color=COLORS[1], fontsize=8)
        twin.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        if case.case == cases.iloc[0]["case"]:
            fig.legend(h1+h2, l1+l2, fontsize=9, loc="upper left",
                       bbox_to_anchor=(0.07, 0.885), ncol=3, frameon=False)
        _clean(ax)
    fig.subplots_adjust(wspace=0.42)
    writer.save(fig, "09_pooling_time", "重新载入对应外层测试折的三个最佳模型，保存告警时刻或最终时刻的 Attention Pooling 权重。权重和为1、未来及 padding 权重为0，并核对重载概率与 OOF 一致。阴影为种子标准差；该权重不是 Transformer 自注意力矩阵，也不是因果贡献。",
                ["tables/case_attention.csv", "tables/case_trajectories.csv", "checkpoints/fold_*_seed_*/best.pt"])


def _stability_figure(writer, seed_metrics, fold_metrics):
    test = seed_metrics.loc[seed_metrics.split == "test"]
    ensemble = fold_metrics.loc[fold_metrics.split == "test"]
    fig, axes = _figure("10  |  Variation across folds and seeds", "Individual seed runs use their own validation thresholds; diamonds: three-seed probability ensemble", cols=3, size=(13, 4.8))
    for ax, metric, label in zip(axes.flat, ["pr_auc", "f1", "f_latency"], ["PR-AUC (AP)", "Final F1", "F-latency"]):
        for i, (seed, frame) in enumerate(test.groupby("seed")):
            ax.scatter(frame.fold + (i-1)*0.1, frame[metric], color=COLORS[i], s=35, label=f"Seed {seed}")
        ax.scatter(ensemble.fold, ensemble[metric], marker="D", s=40, color="#25313E", label="Ensemble")
        ax.set(xlabel="Outer test fold", ylabel=label, xticks=ensemble.fold, ylim=(0, 1.03))
        ax.legend(fontsize=7)
        _clean(ax)
    writer.save(fig, "10_stability", "五折三种子测试指标与同折集成结果。集成不是选最佳种子；各种子的阈值来自各自验证预测。",
                ["metrics/seed_run_metrics.csv", "metrics/fold_metrics.csv"])


def _markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def build_visualizations(config: ExperimentConfig, device_name: str = "cuda") -> dict:
    """Validate T10, derive immutable-data tables, render the T11 figure set."""
    validation = validate_evaluation_artifacts(config)
    root = config.path("outputs_dir")
    tables_root = root / "tables"
    tables_root.mkdir(exist_ok=True)
    metrics = read_json(root / "metrics/oof_metrics.json")
    final = pd.read_parquet(root / "predictions/oof_final_predictions.parquet")
    prefix = pd.read_parquet(root / "predictions/oof_prefix_predictions.parquet")
    verify_point_metrics(final, metrics)
    sources = [root / "predictions/oof_final_predictions.parquet", root / "predictions/oof_prefix_predictions.parquet",
               root / "metrics/oof_metrics.json", root / "metrics/fold_thresholds.json",
               root / "metrics/fold_metric_summary.csv", root / "metrics/fold_metrics.csv", root / "metrics/seed_run_metrics.csv"]
    posts_path = config.path("processed_dir") / "post_index.parquet"
    users_path = config.path("processed_dir") / "user_sequences.parquet"
    sources.extend([posts_path, users_path])
    posts = pd.read_parquet(posts_path, columns=["user_id", "sequence_index", "delta_days", "timestamp_unix"])
    users = pd.read_parquet(users_path)
    folds = list(range(int(config.section("cross_validation")["n_splits"])))
    seeds = [int(s) for s in config.section("training")["seeds"]]
    histories, runs = [], []
    for fold in folds:
        for seed in seeds:
            name = f"fold_{fold}_seed_{seed}"
            run_path, history_path = root / f"logs/{name}/run.json", root / f"logs/{name}/history.json"
            run = read_json(run_path)
            if run["config_hash"] != config.config_hash or run["status"] != "complete":
                raise ValueError(f"Stale or incomplete run: {name}")
            runs.append(run)
            histories.append(pd.DataFrame(read_json(history_path)["history"]).assign(fold=fold, seed=seed))
            sources.extend([run_path, history_path, root / f"checkpoints/{name}/best.pt"])
        sources.extend([config.path("splits_dir") / f"fold_{fold}.json", config.path("time_stats_dir") / f"fold_{fold}.json"])
    histories, runs = pd.concat(histories, ignore_index=True), pd.DataFrame(runs)
    fold_metrics = pd.read_csv(root / "metrics/fold_metrics.csv")
    seed_metrics = pd.read_csv(root / "metrics/seed_run_metrics.csv")
    tables = result_tables(metrics, pd.read_csv(root / "metrics/fold_metric_summary.csv"))
    tables.update(stability_tables(seed_metrics))
    performance = prefix_performance(final, prefix, list(config.section("pooling")["checkpoints"]))
    if not np.all(performance.loc[performance.scope == "oof", "users"] == len(final)):
        raise AssertionError("Prefix evaluation cohort changed")
    max_prefix = performance.loc[(performance.scope == "oof") & (performance.checkpoint == config.section("data")["max_posts"])].iloc[0]
    for metric, value in metrics["point_estimate"]["classification"].items():
        if not np.isclose(max_prefix[metric], value, atol=1e-12):
            raise AssertionError(f"Last prefix does not match final metric {metric}")
    cases = select_cases(final)
    case_data = prefix.merge(cases[["user_id", "case"]], on="user_id", validate="many_to_one")
    post_times = posts.assign(position=posts.sequence_index + 1).drop(columns="sequence_index")
    case_data = case_data.merge(post_times, on=["user_id", "position"], validate="one_to_one")
    case_data["elapsed_days"] = (case_data.timestamp_unix - case_data.groupby("user_id").timestamp_unix.transform("min")) / 86400
    attention = extract_case_attention(config, cases, prefix, device_name)
    tables.update({"prefix_performance": performance, "representative_cases": cases,
                   "case_trajectories": case_data, "case_attention": attention,
                   "training_history": histories, "training_runs": runs, "fold_results": fold_metrics})
    for name, frame in tables.items():
        frame.to_csv(tables_root / f"{name}.csv", index=False, encoding="utf-8")
    print("Tables and case checkpoint replay validated; rendering figures", flush=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 11, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.edgecolor": "#87949F", "text.color": "#243343", "axes.labelcolor": "#243343"})
    writer = FigureWriter(root / "figures")
    _dataset_figures(writer, final, users, posts)
    _training_figure(writer, histories, runs, folds, seeds)
    _classification_figures(writer, final, metrics, fold_metrics)
    _prefix_figure(writer, performance)
    _latency_figure(writer, final)
    _case_figures(writer, cases, case_data, attention)
    _stability_figure(writer, seed_metrics, fold_metrics)
    report = ["# EOT 正式结果与图表（T11）", "",
              f"配置哈希：`{config.config_hash}`。OOF 用户 {len(final)} 名，正类 {int(final.label.sum())} 名。",
              "", "本报告读取 T10 冻结的五折测试预测；每折三个种子概率取均值，验证集锁定的阈值直接用于对应测试折。所有表格和图均由正式代码生成。",
              "", "## 主分类结果", "", _markdown_table(tables["classification"]), "", "## 早期检测结果", "", _markdown_table(tables["early_detection"]), "",
              "OOF 是合并用户后的点估计；fold_mean/fold_sd 是五个测试折指标的等权均值/样本标准差。CI 是既定模型、划分和阈值条件下的 2,000 次用户 bootstrap 95% 区间，不包含重新训练造成的波动。", "",
              "## 阅读与引用说明", "",
              "- PR-AUC 采用 average precision (AP)。ERDE 使用 0–1 数值，换算百分比时乘100。",
              "- 前缀曲线保留固定 OOF 用户集合；短序列结束后沿用最后一次概率，同时报告仍有至少 k 帖的用户数。",
              "- 最终分类使用完整有效序列概率；早期告警为连续两次前缀越阈，一旦告警即保留该决策。",
              "- 真阳性延迟中位数仅针对成功告警用户；漏报在累计检测比例图中保留在正类分母。",
              "- 早期指标基于最多256条清洗后的目标用户帖子，是本实验的离线协议，不能直接视为官方完整流式榜单成绩。",
              "- 四个案例由固定中位长度规则选择，覆盖早期 TP/FP/FN/TN；仅作描述。Pooling 权重不是因果解释或临床证据。", "",
              "## 结果观察", ""]
    cls = metrics["point_estimate"]["classification"]
    early = metrics["point_estimate"]["early_detection"]
    weakest = fold_metrics.loc[fold_metrics.split == "test"].sort_values(["recall", "fold"]).iloc[0]
    report.extend([
        f"最终分类 Precision={cls['precision']:.4f}、Recall={cls['recall']:.4f}；当前阈值策略呈现较高精确率和较低召回率。",
        f"早期告警检出 {int(((final.label == 1) & (final.alert_prediction == 1)).sum())}/{int(final.label.sum())} 名正类，F-latency={early['f_latency']:.4f}；检出者中位延迟 {early['median_latency']:g} 帖。",
        f"折间波动明显：Fold {int(weakest.fold)} 的最终 Recall={weakest.recall:.4f}，对应验证阈值={weakest.threshold:.3f}。阈值由很小的验证正类样本确定，报告应保留折间结果，不能只引用合并 AUC。",
        "当前测试结果用于最终报告；后续如需调参，应预先定义新的验证协议并单独记录实验版本。", "", "## 图表", "",
    ])
    for record in writer.records:
        name = record["name"]
        report.extend([f"### {name}", "", record["caption"], "",
                       f"![{name}](../figures/{name}.png)", "",
                       f"[矢量 PDF](../figures/{name}.pdf) · [可编辑 SVG](../figures/{name}.svg)", ""])
    report.extend(["## 可重复执行", "", "在项目根目录设置原有 ERISK_TASK2_ROOT、PYTHONPATH=src 后执行：", "",
                   "```powershell", "python scripts/visualize_results.py --device cuda", "```", "",
                   "汇总表：`outputs/tables/`。生成清单：`outputs/reports/visualization_manifest.json`。"])
    report_path = root / "reports/T11_RESULTS.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    # Contact sheet is a small review artifact made from the exported scientific figures.
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    for ax, record in zip(axes.flat, writer.records):
        ax.imshow(plt.imread(writer.root / record["files"][0]))
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(writer.root / "figure_overview.png", dpi=110, facecolor="white")
    plt.close(fig)
    output_paths = [tables_root / f"{name}.csv" for name in tables]
    output_paths.extend(writer.root / f for record in writer.records for f in record["files"])
    output_paths.extend([writer.root / "figure_overview.png", report_path])
    manifest = {
        "status": "pass", "config_hash": config.config_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures": writer.records, "figure_count": len(writer.records), "formats": ["png", "pdf", "svg"],
        "dpi": 300, "table_count": len(tables), "oof_users": len(final),
        "t10_validation_status": validation["status"], "prefix_fixed_cohort": True,
        "last_prefix_metrics_match_final": True, "case_checkpoint_replay_matches_oof": True,
        "attention_future_and_padding_excluded": True,
        "sources": {str(p.relative_to(config.project_root)): _file_sha256(p) for p in sources},
        "outputs": {str(p.relative_to(config.project_root)): _file_sha256(p) for p in output_paths},
        "code": {str(p.relative_to(config.project_root)): _file_sha256(p) for p in [Path(__file__), Path(__file__).with_name("reporting.py")]},
    }
    write_json(manifest, root / "reports/visualization_manifest.json")
    return manifest
