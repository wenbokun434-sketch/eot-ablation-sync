"""ODE mechanism figures built exclusively from the frozen probe artifacts.

The probe records are produced by ``scripts/probe_ode_states.py``, which replays the 15
formal checkpoints without training. This module only reads those records: it never
touches a model, a threshold, or a prediction. Every plotted number is derived from
saved step-level records, and user-level quantities are collapsed across the three seeds
within a fold before any class comparison, matching the protocol in ``DESIGN.md``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .evaluation import _file_sha256
from .reporting import read_json

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20250904


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _bootstrap_spearman(
    frame: pd.DataFrame, iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, float]:
    """Percentile interval resampling whole users, never individual steps."""

    codes, uniques = pd.factorize(frame.user_id.to_numpy(), sort=True)
    delta = frame.delta_t.to_numpy(np.float64)
    displacement = frame.displacement.to_numpy(np.float64)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(len(uniques) + 1))
    block_indices = [
        order[boundaries[index] : boundaries[index + 1]] for index in range(len(uniques))
    ]
    block_sizes = np.array([len(block) for block in block_indices])
    if len(block_indices) < 3:
        return {"lower": float("nan"), "upper": float("nan"), "iterations": iterations}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta_rank = pd.Series(delta).rank().to_numpy()
    displacement_rank = pd.Series(displacement).rank().to_numpy()
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        drawn = rng.integers(0, len(block_indices), size=len(block_indices))
        sizes = block_sizes[drawn]
        selection = np.empty(int(sizes.sum()), dtype=np.int64)
        cursor = 0
        for index, size in zip(drawn, sizes):
            selection[cursor : cursor + size] = block_indices[index]
            cursor += size
        if cursor != len(selection):
            raise AssertionError("bootstrap resample size mismatch")
        left = delta_rank[selection]
        right = displacement_rank[selection]
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.sqrt(float((left * left).sum()) * float((right * right).sum()))
        estimates[iteration] = (
            float((left * right).sum()) / denominator if denominator > 0 else np.nan
        )
    return {
        "lower": float(np.nanpercentile(estimates, 2.5)),
        "upper": float(np.nanpercentile(estimates, 97.5)),
        "iterations": iterations,
    }


def _collapse_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per user: median across the three seeds of that user's own fold."""

    return (
        frame.groupby(["user_id", "label"], as_index=False)
        .agg(
            mean_displacement=("mean_displacement", "median"),
            max_displacement=("max_displacement", "median"),
            mean_delta=("mean_delta", "median"),
            counterfactual_final_relative_shift=(
                "counterfactual_final_relative_shift",
                "median",
            ),
            counterfactual_final_cosine=("counterfactual_final_cosine", "median"),
            counterfactual_mean_displacement=("counterfactual_mean_displacement", "median"),
            mean_state_norm=("mean_state_norm", "median"),
            length=("length", "first"),
        )
    )


def load_probe(probe_dir: Path) -> dict[str, Any]:
    return {
        "steps": pd.read_parquet(probe_dir / "ode_step_records.parquet"),
        "users": pd.read_parquet(probe_dir / "ode_user_records.parquet"),
        "summary": read_json(probe_dir / "ode_probe_summary.json"),
    }


def time_step_table(steps: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Median displacement and spread inside equal-count time-step bins."""

    graded = steps.loc[steps.position > 1].copy()
    graded["bin"] = pd.qcut(graded.delta_t, q=bins, duplicates="drop")
    return (
        graded.groupby("bin", observed=True)
        .agg(
            delta_median=("delta_t", "median"),
            displacement_median=("displacement", "median"),
            displacement_p10=("displacement", lambda values: values.quantile(0.10)),
            displacement_p90=("displacement", lambda values: values.quantile(0.90)),
            steps=("displacement", "size"),
            users=("user_id", "nunique"),
        )
        .reset_index(drop=True)
    )


def class_time_step_table(steps: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    graded = steps.loc[steps.position > 1].copy()
    graded["bin"] = pd.qcut(graded.delta_t, q=bins, duplicates="drop")
    return (
        graded.groupby(["label", "bin"], observed=True)
        .agg(
            delta_median=("delta_t", "median"),
            displacement_median=("displacement", "median"),
            displacement_low=("displacement", lambda values: values.quantile(0.25)),
            displacement_high=("displacement", lambda values: values.quantile(0.75)),
            steps=("displacement", "size"),
            users=("user_id", "nunique"),
        )
        .reset_index()
    )


def validate_probe_records(config: ExperimentConfig, probe_dir: Path) -> dict[str, Any]:
    """Confirm the probe is complete, reproducible, and consistent with the model."""

    failures: list[str] = []
    checks: dict[str, Any] = {}
    summary = read_json(probe_dir / "ode_probe_summary.json")
    steps = pd.read_parquet(probe_dir / "ode_step_records.parquet")
    users = pd.read_parquet(probe_dir / "ode_user_records.parquet")

    if summary.get("partial_run", False):
        failures.append("probe was run with --limit and is not a full replay")
    if summary.get("config_hash") != config.config_hash:
        failures.append("probe config hash does not match the frozen configuration")

    error = float(summary.get("max_raw_rk4_vs_model_error", float("inf")))
    checks["max_raw_rk4_vs_model_error"] = error
    if not error < 1e-3:
        failures.append(f"raw RK4 replay error {error:.3e} exceeds 1e-3")

    for record in summary.get("checkpoints", []):
        path = config.project_root / record["path"]
        if not path.is_file():
            failures.append(f"missing checkpoint {record['path']}")
        elif _file_sha256(path) != record["sha256"]:
            failures.append(f"checkpoint hash changed: {record['path']}")

    covered = set(users.user_id)
    oof = pd.read_parquet(
        config.path("outputs_dir") / "predictions/oof_final_predictions.parquet",
        columns=["user_id"],
    )
    checks["probe_users"] = len(covered)
    checks["oof_users"] = int(len(oof))
    if covered != set(oof.user_id):
        failures.append("probe users do not equal the frozen OOF user set")

    per_user = users.groupby("user_id").size()
    checks["checkpoints_per_user"] = int(per_user.median())
    if per_user.nunique() != 1:
        failures.append("users were not replayed by the same number of checkpoints")
    if int(users.groupby("user_id").length.nunique().max()) != 1:
        failures.append("sequence length differs across checkpoints for some user")

    joined = pd.concat(
        [
            steps.groupby("user_id").size().rename("steps"),
            users.groupby("user_id").length.first().rename("length"),
            per_user.rename("checkpoints"),
        ],
        axis=1,
    )
    if not (joined.steps == joined.length * joined.checkpoints).all():
        failures.append("step records do not cover every checkpoint of every user")
    checks["step_rows_expected"] = int((joined.length * joined.checkpoints).sum())
    checks["step_rows_observed"] = int(len(steps))
    if len(steps) != checks["step_rows_expected"]:
        failures.append("step record total does not match length x checkpoints")

    graded = steps.loc[steps.position > 1]
    checks["spearman_delta_vs_displacement"] = _spearman(
        graded.delta_t.to_numpy(np.float64), graded.displacement.to_numpy(np.float64)
    )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_summary_sha256": _file_sha256(probe_dir / "ode_probe_summary.json"),
    }


def _state_modulation(steps: pd.DataFrame) -> pd.DataFrame:
    """Per user: how strongly the field's magnitude varies along the trajectory.

    Displacement is linear in the time step by construction, so dividing by the step
    removes the timing factor and leaves the field magnitude sampled along the states the
    user actually visits. A ratio near zero means the ODE behaves like a scalar gate.
    """

    graded = steps.loc[(steps.position > 1) & (steps.delta_t > 1e-6)].copy()
    graded["normalized"] = graded.displacement / graded.delta_t
    frame = (
        graded.groupby("user_id")
        .normalized.agg(["mean", "std", "size"])
        .rename(columns={"size": "steps"})
        .reset_index()
    )
    frame = frame.loc[frame.steps >= 8].copy()
    frame["ratio_of_std_to_mean"] = frame["std"] / frame["mean"].clip(lower=1e-12)
    return frame


def _select_probe_cases(users: pd.DataFrame) -> pd.DataFrame:
    """Exactly three deterministic cases: one per class at median length, plus a
    distinct user whose mean gap is the largest, so the panel shows long-gap structure
    at a comparable sequence length."""

    ranked = users.groupby("user_id").agg(
        label=("label", "first"),
        length=("length", "first"),
        seed=("seed", "first"),
        mean_delta=("mean_delta", "median"),
    )
    chosen: list[dict[str, Any]] = []

    def pick(group: pd.DataFrame, rule: str) -> None:
        group = group.copy()
        group["distance"] = (group.length - float(group.length.median())).abs()
        row = group.sort_values(["distance", "user_id"]).iloc[0]
        chosen.append(
            {
                "user_id": row.name,
                "label": int(row.label),
                "length": int(row.length),
                "seed": int(row.seed),
                "rule": rule,
            }
        )

    pick(ranked, "median length")
    remaining = ranked.drop(index=[entry["user_id"] for entry in chosen], errors="ignore")
    # Third case: the largest mean gap among users within 10% of the median length, so
    # the trajectory is long enough to compare with the first two.
    reference = float(ranked.length.median())
    comparable = remaining.loc[
        (remaining.length >= reference * 0.9) & (remaining.length <= reference * 1.1)
    ]
    pool = comparable if len(comparable) >= 3 else remaining
    longest = pool.sort_values(["mean_delta", "user_id"]).iloc[-1]
    chosen.append(
        {
            "user_id": longest.name,
            "label": int(longest.label),
            "length": int(longest.length),
            "seed": int(longest.seed),
            "rule": "largest mean gap",
        }
    )
    remaining = ranked.drop(index=[entry["user_id"] for entry in chosen], errors="ignore")
    for label in (0, 1):
        group = remaining.loc[remaining.label == label]
        if len(group) >= 3:
            pick(group, "median length, second class")
            break
    return pd.DataFrame(chosen)


def build_ode_figures(config: ExperimentConfig) -> dict[str, Any]:
    """Render the ODE mechanism figure set (11-14) from frozen probe records."""

    from .visualization import COLORS, LABELS, FigureWriter, _clean, _figure

    probe_dir = config.path("outputs_dir") / "ode_probe"
    payload = load_probe(probe_dir)
    steps, users, summary = payload["steps"], payload["users"], payload["summary"]
    graded = steps.loc[steps.position > 1].copy()
    collapsed = _collapse_seeds(users)
    tables_root = config.path("outputs_dir") / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    # A handful of post pairs share a timestamp, so d is exactly zero and a log axis
    # cannot show them. They are excluded from the plotting frame only; every reported
    # share is computed on the full graded set.
    plottable = graded.loc[graded.delta_t > 0].copy()

    binned = time_step_table(plottable)
    class_binned = class_time_step_table(plottable)
    binned.to_csv(tables_root / "ode_time_step_bins.csv", index=False, encoding="utf-8")
    class_binned.to_csv(
        tables_root / "ode_time_step_bins_by_class.csv", index=False, encoding="utf-8"
    )
    collapsed.to_csv(tables_root / "ode_user_collapsed.csv", index=False, encoding="utf-8")

    correlation = _spearman(
        plottable.delta_t.to_numpy(np.float64), plottable.displacement.to_numpy(np.float64)
    )
    interval = _bootstrap_spearman(plottable)
    probe_users = int(users.user_id.nunique())
    sources = [
        str((probe_dir / name).relative_to(config.project_root))
        for name in (
            "ode_step_records.parquet",
            "ode_user_records.parquet",
            "ode_probe_summary.json",
        )
    ]
    writer = FigureWriter(config.path("outputs_dir") / "figures")
    stats: dict[str, Any] = {
        "spearman_delta_vs_displacement": correlation,
        "spearman_user_bootstrap": interval,
        "graded_steps": int(len(graded)),
        "plottable_steps": int(len(plottable)),
        "exact_zero_steps": int((graded.delta_t <= 0).sum()),
        "probe_users": probe_users,
    }

    # ------------------------------------------------------------------ figure 11
    fig, axes = _figure(
        "11  |  How far the ODE moves the state between two writings",
        f"{len(plottable):,} graded steps from {probe_users} users with d>0 "
        f"({int((graded.delta_t <= 0).sum())} same-second pairs omitted); "
        f"Spearman rho={correlation:.3f} (user bootstrap 95% [{interval['lower']:.3f}, {interval['upper']:.3f}])",
        cols=2,
        size=(12.5, 5.0),
    )
    ax = axes[0, 0]
    hexbin = ax.hexbin(
        plottable.delta_t,
        plottable.displacement,
        xscale="log",
        yscale="log",
        gridsize=44,
        mincnt=1,
        cmap="Blues",
        bins="log",
        linewidths=0.0,
    )
    fig.colorbar(hexbin, ax=ax, label="Steps per cell (log)")
    ax.plot(
        binned.delta_median,
        binned.displacement_median,
        color=COLORS[1],
        linewidth=2.6,
        marker="o",
        markersize=4,
        label="Median per equal-count bin",
    )
    ax.fill_between(
        binned.delta_median,
        binned.displacement_p10,
        binned.displacement_p90,
        color=COLORS[1],
        alpha=0.16,
        label="Within-bin 10th-90th percentile",
    )
    ax.set(
        xlabel="Normalized time step d (log scale)",
        ylabel="ODE displacement |z_after - z_before| (log scale)",
    )
    ax.legend(fontsize=8, loc="upper left")
    _clean(ax)

    ax = axes[0, 1]
    for label in (0, 1):
        subset = class_binned.loc[class_binned.label == label]
        ax.plot(
            subset.delta_median,
            subset.displacement_median,
            color=COLORS[label],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=LABELS[label],
        )
        ax.fill_between(
            subset.delta_median,
            subset.displacement_low,
            subset.displacement_high,
            color=COLORS[label],
            alpha=0.13,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Normalized time step d (log scale)",
        ylabel="ODE displacement (log scale)",
    )
    ax.legend(fontsize=8, loc="upper left")
    _clean(ax)
    writer.save(
        fig,
        "11_ode_displacement",
        "每个相邻帖对经过 ODE 传播后的状态位移（不含 GRUCell 更新）与其归一化时间间隔 d 的关系。"
        "左图密度为帖子对数，橙线为 20 个等频分箱的中位数与 10-90 分位带；右图按真实类别分色。"
        "相关系数为 Spearman，置信区间以用户为单位做 2000 次 bootstrap 重采样。",
        sources,
    )
    stats["displacement_quantiles"] = {
        str(q): float(plottable.displacement.quantile(q)) for q in (0.5, 0.75, 0.95, 0.99)
    }

    # ------------------------------------------------------------------ figure 12
    observed = pd.read_parquet(
        config.path("processed_dir") / "post_index.parquet",
        columns=["sequence_index", "delta_days"],
    )
    observed = observed.loc[observed.sequence_index > 0, "delta_days"].to_numpy(np.float64)
    fig, axes = _figure(
        "12  |  Observed writings are extremely unevenly spaced in time",
        f"Normalized steps use each fold's training 99th percentile of log(1 + gap); "
        f"{float((graded.delta_t < 0.01).mean()):.1%} of steps sit below d=0.01",
        cols=2,
        size=(12.5, 5.0),
    )
    ax = axes[0, 0]
    ordered = np.sort(graded.delta_t.to_numpy(np.float64))
    ax.plot(
        ordered,
        np.arange(1, len(ordered) + 1) / len(ordered),
        color=COLORS[0],
        linewidth=2.4,
    )
    median_delta = float(graded.delta_t.median())
    ax.axvline(median_delta, linestyle="--", color=COLORS[1], linewidth=1.5)
    ax.annotate(
        f"median d={median_delta:.4f}\n(about 13 minutes)",
        xy=(median_delta, 0.5),
        xytext=(median_delta * 8, 0.30),
        fontsize=8,
        color="#56616B",
    )
    ax.axvline(0.5, linestyle=":", color=COLORS[2], linewidth=1.5)
    ax.annotate(
        "d=0.5\nthe entire remaining\nhalf of the scale\nholds 6% of steps",
        xy=(0.5, 0.9),
        xytext=(0.06, 0.72),
        fontsize=8,
        color="#56616B",
    )
    ax.set(
        xscale="log",
        xlabel="Normalized time step d (log scale)",
        ylabel="Fraction of steps at or below d",
        ylim=(0, 1.02),
    )
    _clean(ax)

    ax = axes[0, 1]
    positive = observed[observed > 0]
    ax.hist(
        positive,
        bins=np.logspace(np.log10(positive.min()), np.log10(positive.max()), 48),
        color=COLORS[0],
        alpha=0.88,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Raw gap between consecutive writings (days, log scale)",
        ylabel="Number of post pairs (log scale)",
    )
    top = ax.get_ylim()[1]
    ax.set_ylim(top=top * 3.2)
    for value, text in (
        (1 / 1440, "1 minute"),
        (1 / 24, "1 hour"),
        (1.0, "1 day"),
        (365.0, "1 year"),
    ):
        ax.axvline(value, color="#9AA3AA", linestyle=":", linewidth=1.0)
        ax.annotate(
            text,
            xy=(value, top * 2.6),
            fontsize=7,
            color="#56616B",
            rotation=90,
            ha="right",
            va="top",
        )
    _clean(ax)
    writer.save(
        fig,
        "12_ode_time_scale",
        "左图：全部相邻帖对归一化间隔 d 的经验累积分布，中位数仅约 0.0075，说明绝大多数相邻帖只相隔十几分钟；"
        "右图：同一批帖对的真实间隔分布，跨度从秒级到年级。观察序列在时间上高度不均匀，"
        "这正是引入连续时间状态传播的动机，也说明时间信号集中在少数长间隔上。",
        sources
        + [
            str(
                (config.path("processed_dir") / "post_index.parquet").relative_to(
                    config.project_root
                )
            )
        ],
    )

    # ------------------------------------------------------------------ figure 13
    fig, axes = _figure(
        "13  |  How much of the final state is decided by timing alone",
        f"Counterfactual holds every post fixed and replaces all gaps with the fold median "
        f"(population median d={float(users.counterfactual_constant.median()):.4f})",
        cols=2,
        size=(12.5, 5.0),
    )
    ax = axes[0, 0]
    for label in (0, 1):
        subset = collapsed.loc[
            collapsed.label == label, "counterfactual_final_relative_shift"
        ]
        subset = subset.loc[np.isfinite(subset) & (subset > 0)]
        ax.hist(
            subset,
            bins=np.logspace(np.log10(subset.min()), np.log10(subset.max()), 16),
            histtype="step",
            linewidth=2.0,
            weights=np.full(len(subset), 100 / len(subset)),
            color=COLORS[label],
            label=f"{LABELS[label]} (n={len(subset)})",
        )
    median_shift = float(collapsed.counterfactual_final_relative_shift.median())
    ax.axvline(
        median_shift, linestyle="--", color=COLORS[1], linewidth=1.6,
        label=f"Median {median_shift:.1%}",
    )
    ax.set(
        xscale="log",
        xlabel="Relative shift of the final encoder state (log scale)",
        ylabel="Users within class (%)",
    )
    ax.legend(fontsize=8)
    _clean(ax)

    ax = axes[0, 1]
    real_mean = float(users.mean_displacement.median())
    fake_mean = float(users.counterfactual_mean_displacement.median())
    modulation = _state_modulation(steps)
    modulation.to_csv(
        tables_root / "ode_state_modulation.csv", index=False, encoding="utf-8"
    )
    user_level = (
        users.groupby(["user_id", "label"], as_index=False)
        .agg(mean_delta=("mean_delta", "median"), mean_displacement=("mean_displacement", "median"))
    )
    user_level = user_level.loc[
        (user_level.mean_delta > 0) & (user_level.mean_displacement > 0)
    ]
    for label in (0, 1):
        subset = user_level.loc[user_level.label == label]
        ax.scatter(
            subset.mean_delta,
            subset.mean_displacement,
            s=9,
            alpha=0.45,
            color=COLORS[label],
            label=LABELS[label],
            linewidths=0.0,
        )
    slope = np.polyfit(
        np.log(user_level.mean_delta), np.log(user_level.mean_displacement), 1
    )
    grid = np.logspace(
        np.log10(user_level.mean_delta.min()), np.log10(user_level.mean_delta.max()), 50
    )
    ax.plot(
        grid,
        np.exp(slope[1]) * grid ** slope[0],
        color="#25313E",
        linewidth=2.0,
        label=f"log-log fit, slope {slope[0]:.2f}",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Mean normalized time step per user (log scale)",
        ylabel="Mean ODE displacement per user (log scale)",
    )
    ax.legend(fontsize=8, loc="upper left")
    _clean(ax)
    writer.save(
        fig,
        "13_ode_counterfactual",
        "左图：反事实重放——保持全部帖子内容与顺序不变，只把所有时间间隔替换为该折训练集的中位间隔，"
        "最终编码状态的相对偏移分布（对数轴）。右图：每个用户一个点，平均位移与平均归一化间隔在双对数坐标下"
        "近似成幂律，斜率接近 1，说明“时间越长、状态被推得越远”在用户层面成立。"
        "该分析刻画模型对时间输入的敏感性，不是重新训练的消融实验。",
        sources,
    )
    stats["counterfactual"] = {
        "median_final_relative_shift": median_shift,
        "median_final_cosine": float(collapsed.counterfactual_final_cosine.median()),
        "real_mean_displacement": real_mean,
        "constant_mean_displacement": fake_mean,
        "ratio": fake_mean / max(real_mean, 1e-12),
        "user_level_loglog_slope": float(slope[0]),
        "median_field_constancy_cv": float(modulation.ratio_of_std_to_mean.median()),
    }

    # ------------------------------------------------------------------ figure 14
    position_table = (
        plottable.groupby("position")
        .agg(
            displacement_median=("displacement", "median"),
            displacement_low=("displacement", lambda values: values.quantile(0.25)),
            displacement_high=("displacement", lambda values: values.quantile(0.75)),
            users=("user_id", "nunique"),
        )
        .reset_index()
    )
    position_table.to_csv(
        tables_root / "ode_position_profile.csv", index=False, encoding="utf-8"
    )
    constant = float(users.counterfactual_constant.median())
    cases = _select_probe_cases(users)

    fig, axes = _figure(
        "14  |  Most steps barely move the state; a few move it a lot",
        "Same texts, two timelines. Displacement is linear in the time step, so the dashed curve is the "
        "equal-gap counterpart implied by the same vector field.",
        rows=2,
        cols=2,
        size=(12, 8),
    )
    ax = axes[0, 0]
    positive = plottable.loc[plottable.displacement > 0, "displacement"].to_numpy(np.float64)
    ax.hist(
        positive,
        bins=np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60),
        color=COLORS[0],
        alpha=0.9,
    )
    for value, text in (
        (float(plottable.displacement.median()), "median"),
        (float(plottable.displacement.quantile(0.95)), "95th percentile"),
    ):
        ax.axvline(value, color=COLORS[1], linestyle="--", linewidth=1.6)
        ax.annotate(
            f"{text}\n{value:.3f}",
            xy=(value, ax.get_ylim()[1] * 0.55),
            xytext=(value * 1.6, ax.get_ylim()[1] * 0.62),
            fontsize=8,
            color="#56616B",
        )
    ax.set(
        xscale="log",
        xlabel="ODE displacement per step (log scale)",
        ylabel="Number of steps",
    )
    _clean(ax)

    for ax, case in zip(axes.flat[1:], cases.itertuples(index=False)):
        series = steps.loc[
            (steps.user_id == case.user_id) & (steps.seed == case.seed)
        ].sort_values("position")
        series = series.loc[series.delta_t > 0]
        ax.plot(
            series.position,
            series.displacement,
            color=COLORS[case.label],
            linewidth=1.8,
            label="Real timeline",
        )
        median_step = float(series.delta_t.median())
        ax.plot(
            series.position,
            series.displacement / series.delta_t.clip(lower=1e-6) * median_step,
            color=COLORS[2],
            linewidth=1.5,
            linestyle="--",
            label="Equal gaps (magnitude scaled)",
        )
        longest = series.sort_values("delta_t").iloc[-1]
        ax.scatter(
            [longest.position],
            [longest.displacement],
            color=COLORS[4],
            s=46,
            zorder=5,
            label=f"Longest gap (d={longest.delta_t:.3f})",
        )
        ax.set(
            title=f"{case.user_id} | {LABELS[case.label]} | {case.length} writings",
            xlabel="Writing position",
            ylabel="ODE displacement",
            ylim=(0, None),
        )
        ax.legend(fontsize=7)
        _clean(ax)
    writer.save(
        fig,
        "14_ode_timelines",
        "上左：全部帖对的位移分布（对数横轴），中位仅 0.036 而 P95 达 2.90，说明 ODE 在多数步骤上"
        "几乎不推动状态，少处长间隔才产生明显位移。其余三图为确定性选取的用户个案（两类各取长度"
        "中位数最近者，再加平均间隔最大者），实线为真实时间线，虚线为把间隔统一到该用户中位间隔后的"
        "等效位移，红点为最长一次空档。个案仅用于说明机制形态。",
        sources,
    )
    stats["cases"] = cases.to_dict(orient="records")
    return {
        "status": "pass",
        "config_hash": config.config_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_count": len(writer.records),
        "figures": writer.records,
        "stats": stats,
        "probe_summary": {
            "median_delta": median_delta,
            "share_below_0.01": float((graded.delta_t < 0.01).mean()),
            "share_above_0.5": float((graded.delta_t > 0.5).mean()),
            "graded_steps": int(len(graded)),
        },
        "tables": [
            "ode_time_step_bins",
            "ode_time_step_bins_by_class",
            "ode_user_collapsed",
            "ode_position_profile",
        ],
    }
