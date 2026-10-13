"""Frozen-group, post-hoc high-minus-low contrasts and sensitivity checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ('R0', 'R1', 'R2', 'ACC_V2')
METRICS = ('recall', 'balanced_accuracy', 'f1')
GROUPS = ('low', 'medium', 'high')
SCENARIOS = ('all', 'span_ge_1_day', 'length_256', 'length_256_span_ge_1_day')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def from_counts(counts):
    """Input last axis TP,TN,FP,FN; output recall, balanced accuracy, F1."""
    tp, tn, fp, fn = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    recall = np.divide(tp, tp+fn, out=np.full_like(tp, np.nan), where=tp+fn > 0)
    specificity = np.divide(tn, tn+fp, out=np.full_like(tn, np.nan), where=tn+fp > 0)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp), where=2*tp+fp+fn > 0)
    return np.stack([recall, (recall+specificity)/2, f1], axis=-1)


def indicators(labels, decisions):
    y = np.asarray(labels)[:, None]
    d = np.asarray(decisions)
    require(d.ndim == 2 and len(d) == len(y), 'Misaligned decisions')
    require(bool(np.isin(y, [0, 1]).all() and np.isin(d, [0, 1]).all()), 'Nonbinary labels or decisions')
    return np.stack([(y == 1) & (d == 1), (y == 0) & (d == 0),
                     (y == 0) & (d == 1), (y == 1) & (d == 0)], axis=-1).astype(np.int64)


def resample_metrics(labels, decisions, iterations, seed):
    """Multinomial user counts equal ordinary bootstrap with replacement.

    The same user multiplicities are applied to all strategies. Groups use
    independent random streams. Only fixed-decision confusion metrics are needed.
    """
    z = indicators(labels, decisions)
    n, strategies, _ = z.shape
    require(n > 0, 'Empty bootstrap group')
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, strategies, len(METRICS)))
    for start in range(0, iterations, 256):
        size = min(256, iterations-start)
        weights = rng.multinomial(n, np.full(n, 1/n), size=size)
        counts = (weights @ z.reshape(n, -1)).reshape(size, strategies, 4)
        draws[start:start+size] = from_counts(counts)
    return from_counts(z.sum(axis=0)), draws


def scenario_mask(users, scenario):
    if scenario == 'all':
        return np.ones(len(users), dtype=bool)
    if scenario == 'span_ge_1_day':
        return (users.span_days >= 1).to_numpy()
    if scenario == 'length_256':
        return (users.sequence_length == 256).to_numpy()
    if scenario == 'length_256_span_ge_1_day':
        return ((users.sequence_length == 256) & (users.span_days >= 1)).to_numpy()
    raise ValueError(f'Unknown scenario: {scenario}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    require(args.bootstrap >= 10000, 'At least 10000 draws required for adjusted interval tails')
    previous = ROOT/'outputs/analysis/posting_frequency'
    out = ROOT/'outputs/analysis/posting_frequency_robustness'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'acceptance.json', {'status': 'incomplete'})
    inputs = {}

    def source(path):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    acceptance = read_json(source(previous/'acceptance.json'))
    require(acceptance['status'] == 'pass', 'Previous stage not accepted')
    for name, expected in acceptance['output_sha256'].items():
        require(sha(source(previous/name)) == expected, f'Previous output changed: {name}')
    manifest = read_json(source(previous/'source_manifest.json'))
    for name, expected in manifest['input_sha256'].items():
        require(sha(source(ROOT/name)) == expected, f'Original source changed: {name}')
    for name, key in [('scripts/evaluate_posting_frequency.py', 'script_sha256'),
                      ('tests/test_posting_frequency.py', 'tests_sha256')]:
        require(sha(source(ROOT/name)) == manifest[key], f'Previous code changed: {name}')
    protocol_previous = read_json(source(previous/'protocol.json'))
    u = pd.read_csv(source(previous/'user_frequency_groups.csv')).sort_values('user_id').reset_index(drop=True)
    require(len(u) == 909 and u.user_id.nunique() == 909 and u.label.sum() == 102, 'Cohort changed')
    q1, q2 = protocol_previous['boundaries_posts_per_day']
    expected = pd.cut(u.posts_per_day, [-np.inf, q1, q2, np.inf], labels=GROUPS).astype(str)
    require(np.array_equal(expected, u.frequency_group), 'Frozen group assignments disagree')
    run_id = protocol_previous['source_run_id']
    decisions = []
    for strategy in STRATEGIES:
        f = pd.read_parquet(source(ROOT/f'outputs/optimization/v2/evaluation/runs/{run_id}/oof_{strategy}.parquet'))
        f = f.sort_values('user_id').reset_index(drop=True)
        require(np.array_equal(f.user_id, u.user_id) and np.array_equal(f.label, u.label), 'OOF alignment failed')
        require(np.array_equal(f.length, u.sequence_length), 'OOF input lengths disagree')
        require(np.array_equal(f.final_prediction, (f.probability >= f.threshold_final).astype(int)), 'Frozen decisions mismatch')
        decisions.append(f.final_prediction.to_numpy())
    decisions = np.stack(decisions, axis=1)
    old_metrics = pd.read_csv(source(previous/'metrics.csv')).set_index(['strategy', 'group'])
    # Fix protocol before calculating any new subgroup effects.
    protocol = dict(status='post-hoc exploratory; not independent confirmation',
                    contrast='high frequency minus low frequency', strategies=list(STRATEGIES),
                    metrics=list(METRICS), scenarios=list(SCENARIOS),
                    rules={'all': 'all 909 original users', 'span_ge_1_day': 'span_days >= 1',
                           'length_256': 'sequence_length == 256',
                           'length_256_span_ge_1_day': 'sequence_length == 256 and span_days >= 1'},
                    frequency_boundaries=[q1, q2], regroup_after_filtering=False,
                    bootstrap=dict(iterations=args.bootstrap, seed=args.seed,
                                   method='independent ordinary user bootstrap within low/high groups; same draws across strategies',
                                   implementation='multinomial user multiplicities, batched',
                                   invalid_draws='skip pairs where either sampled group lacks a class; do not redraw'),
                    nominal_interval=.95, primary_family_comparisons=12,
                    primary_adjusted_interval=1-.05/12,
                    multiplicity='Bonferroni-adjusted percentile bootstrap intervals for all-cohort 4 strategies x 3 metrics only; approximate coverage',
                    sensitivity='nominal 95% descriptive intervals only; not additional significance claims',
                    new_training_runs=0, new_inference_runs=0, threshold_changes=0)
    write_json(out/'protocol.json', protocol)
    memberships, points, contrasts = [], [], []
    labels = u.label.to_numpy()
    for scenario_index, scenario in enumerate(SCENARIOS):
        selection = scenario_mask(u, scenario)
        memberships.append(u.loc[selection, ['user_id', 'fold', 'label', 'frequency_group', 'sequence_length', 'span_days']].assign(scenario=scenario))
        group_points, group_draws = {}, {}
        for group_index, group in enumerate(GROUPS):
            mask = selection & (u.frequency_group == group).to_numpy()
            n, positives = int(mask.sum()), int(labels[mask].sum())
            require(n > positives > 0, f'Single-class or empty group: {scenario}/{group}')
            point = from_counts(indicators(labels[mask], decisions[mask]).sum(axis=0))
            group_points[group] = point
            for index, strategy in enumerate(STRATEGIES):
                row = dict(scenario=scenario, group=group, strategy=strategy, users=n,
                           positives=positives, negatives=n-positives, positive_rate=positives/n,
                           median_posts=float(u.loc[mask, 'sequence_length'].median()),
                           median_span_days=float(u.loc[mask, 'span_days'].median()))
                row.update(dict(zip(METRICS, map(float, point[index]))))
                points.append(row)
                if scenario == 'all':
                    for metric_index, metric in enumerate(METRICS):
                        require(np.isclose(point[index, metric_index], old_metrics.loc[(strategy, group), metric], rtol=0, atol=1e-12), 'Prior metric reproduction failed')
            if group != 'medium':
                _, group_draws[group] = resample_metrics(labels[mask], decisions[mask], args.bootstrap,
                                                         np.random.SeedSequence([args.seed, scenario_index, group_index]))
        delta = group_points['high']-group_points['low']
        draws = group_draws['high']-group_draws['low']
        # Use one shared validity mask: both classes must exist in both sampled groups.
        valid = np.isfinite(draws).all(axis=(1, 2))
        require(int(valid.sum()) >= args.bootstrap*.95, 'Too many single-class bootstrap samples')
        for si, strategy in enumerate(STRATEGIES):
            for mi, metric in enumerate(METRICS):
                lo, hi = np.quantile(draws[valid, si, mi], [.025, .975])
                row = dict(scenario=scenario, strategy=strategy, metric=metric,
                           difference=float(delta[si, mi]), lower=float(lo), upper=float(hi),
                           valid_draws=int(valid.sum()), skipped_draws=int((~valid).sum()))
                if scenario == 'all':
                    tail = .05/(2*12)
                    adjlo, adjhi = np.quantile(draws[valid, si, mi], [tail, 1-tail])
                    row.update(adjusted_lower=float(adjlo), adjusted_upper=float(adjhi),
                               adjusted_excludes_zero=bool(adjlo > 0 or adjhi < 0))
                else:
                    row.update(adjusted_lower=np.nan, adjusted_upper=np.nan, adjusted_excludes_zero=None)
                contrasts.append(row)
        print(f'{scenario}: users={int(selection.sum())}; bootstrap={int(valid.sum())}/{args.bootstrap}', flush=True)
    membership = pd.concat(memberships, ignore_index=True)
    points, contrasts = pd.DataFrame(points), pd.DataFrame(contrasts)
    membership.to_csv(out/'sensitivity_membership.csv', index=False, encoding='utf-8-sig')
    points.to_csv(out/'subgroup_metrics.csv', index=False)
    contrasts.to_csv(out/'high_minus_low.csv', index=False)
    plot(out, contrasts)
    report(out, points, contrasts, protocol)
    require(all(sha(ROOT/name) == h for name, h in inputs.items()), 'Protected source changed during analysis')
    write_json(out/'source_manifest.json', dict(input_sha256=inputs, script_sha256=sha(Path(__file__)),
               tests_sha256=sha(ROOT/'tests/test_posting_frequency_robustness.py'),
               python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    write_json(out/'acceptance.json', dict(status='pass', prior_point_estimates_reproduced=True,
               groups_and_thresholds_unchanged=True, sources_unchanged=True,
               contrasts=len(contrasts), primary_family_comparisons=12,
               bootstrap_iterations=args.bootstrap, new_training_runs=0, new_inference_runs=0,
               output_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'acceptance.json'}))
    print(f'PASS: {out}', flush=True)


def plot(out, contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    colors = ['#245A81', '#AD6B26', '#2A8069', '#916099']
    for ax, metric in zip(axes, METRICS):
        rows = contrasts.loc[(contrasts.scenario == 'all') & (contrasts.metric == metric)].set_index('strategy').loc[list(STRATEGIES)]
        for i, (s, color) in enumerate(zip(STRATEGIES, colors)):
            r = rows.loc[s]
            ax.hlines(i, r.adjusted_lower*100, r.adjusted_upper*100, color=color, linewidth=1.3)
            ax.hlines(i, r.lower*100, r.upper*100, color=color, linewidth=5)
            ax.plot(r.difference*100, i, 'o', color=color)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.set(yticks=range(4), yticklabels=STRATEGIES, title=metric.replace('_', ' ').title(),
               xlabel='High minus low (percentage points)', ylim=(-.7, 3.7))
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Exploratory frequency contrasts | All 909 users\nThick: 95% bootstrap CI; thin: Bonferroni-adjusted CI (12 comparisons)', fontsize=12)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'high_minus_low.{ext}', dpi=180)
    plt.close(fig)


def report(out, points, contrasts, protocol):
    names = {'all': '原始全体', 'span_ge_1_day': '跨度至少1天', 'length_256': '固定256条',
             'length_256_span_ge_1_day': '固定256条且跨度至少1天'}
    lines = ['# 发帖频率分组：差异与稳健性分析', '',
             '本轮属于查看过分组结果后的补充探索，不是独立确认。频率边界、测试用户归属、模型预测和阈值全部沿用上一轮。', '',
             '## 方法', '',
             '- 主要比较：高频减低频，四种方案各比较 Recall、Balanced Accuracy 和 F1，共 12 项。未选择中频配对作为额外主比较。',
             f'- 在高、低频组内独立进行 {protocol["bootstrap"]["iterations"]} 次普通用户级 bootstrap，seed={protocol["bootstrap"]["seed"]}。同一组内四方案共用抽样用户。以多项分布生成用户重复次数，等价于有放回逐用户抽样。',
             '- 报告普通 95% percentile 区间及 Bonferroni 校正的 99.5833% 区间，后者针对全体用户的 12 项主要比较。bootstrap 覆盖率是近似的，校正不消除事后探索性质。',
             '- 若一次抽样中任一组缺少一类，则整对跳过且不补抽，有效次数保存在 CSV。',
             '- 敏感性分析固定三个规则：跨度至少1天、输入恰好256条、两者同时满足；不重分频率组。只给描述性的95%区间，不进行额外显著性宣称。', '',
             '## 全体用户：高频减低频', '',
             '| 方案 | 指标 | 差值（百分点） | 95%区间 | 校正区间 | 校正区间是否排除0 |',
             '|---|---|---:|---|---|---|']
    primary = contrasts.loc[contrasts.scenario == 'all']
    for r in primary.itertuples():
        lines.append(f'| {r.strategy} | {r.metric} | {r.difference*100:+.2f} | [{r.lower*100:+.2f}, {r.upper*100:+.2f}] | [{r.adjusted_lower*100:+.2f}, {r.adjusted_upper*100:+.2f}] | {"是" if r.adjusted_excludes_zero else "否"} |')
    detected = primary.loc[primary.adjusted_excludes_zero == True]
    if detected.empty:
        lines += ['', '**12 项主要比较的校正区间均包含 0。本轮没有获得经多重比较校正后排除零差异的证据；这不等于证明两组表现相同。**']
    else:
        lines += ['', '校正区间排除 0 的项目：' + '、'.join(f'{r.strategy}/{r.metric}' for r in detected.itertuples()) + '。这些仍是条件于现有模型的事后探索结果，不能当作独立确认。']
    lines += ['', '![高低频差异](high_minus_low.png)', '', '## 敏感性分析样本构成', '',
              '| 筛选规则 | 组别 | 人数 | 正类 | 帖数中位数 | 跨度中位数（天） |', '|---|---|---:|---:|---:|---:|']
    for r in points.loc[points.strategy == 'R0'].itertuples():
        lines.append(f'| {names[r.scenario]} | {r.group} | {r.users} | {r.positives} | {r.median_posts:.0f} | {r.median_span_days:.2f} |')
    lines += ['', '## 敏感性分析：高低频召回率差值', '',
              '| 规则 | 方案 | 高减低（百分点） | 95%区间 |', '|---|---|---:|---|']
    for r in contrasts.loc[contrasts.metric == 'recall'].itertuples():
        lines.append(f'| {names[r.scenario]} | {r.strategy} | {r.difference*100:+.2f} | [{r.lower*100:+.2f}, {r.upper*100:+.2f}] |')
    lines += ['', '### 对上一轮 R2 高频召回较高现象的追查', '']
    r2 = contrasts.loc[(contrasts.strategy == 'R2') & (contrasts.metric == 'recall')].set_index('scenario')
    for scenario in SCENARIOS:
        row = r2.loc[scenario]
        lines.append(f'- {names[scenario]}：高低频召回率差值为 {row.difference*100:+.2f} 个百分点，普通95%区间 [{row.lower*100:+.2f}, {row.upper*100:+.2f}]。')
    fixed = points.loc[(points.scenario == 'length_256') & (points.strategy == 'R2')].set_index('group')
    lines += [f'- 固定256条后，低频组召回率 {fixed.loc["low", "recall"]:.2%}（正类 {int(fixed.loc["low", "positives"])} 人），高频组 {fixed.loc["high", "recall"]:.2%}（正类 {int(fixed.loc["high", "positives"])} 人）。筛选同时改变了人群，不能把差值变化直接解释为排除了混杂后的纯频率效应。',
              '- 敏感性分析的区间未纳入上述12项主比较校正；即使普通区间不含0，也仅作为待独立验证的探索性线索。',
              '', '全部方案的 F1、Balanced Accuracy 差值及区间见 high_minus_low.csv；筛选后的低、中、高频点估计见 subgroup_metrics.csv。', '',
              '## 结论边界', '',
              '- 固定256条使所保留用户的输入数量相同，但减少样本且改变人群构成；不能称为完全控制混杂，也不能将变化完全归因于帖数。',
              '- 排除跨度不足1天的用户用于检查短窗口集中发帖的影响；未删除原数据，也未改动任何原预测。',
              '- 频率=(帖数−1)/跨度。在固定帖数时，频率与跨度直接关联；本实验无法独立区分两者的作用。',
              '- 区间以固定模型、阈值、折分配及频率边界为条件，不包含重训方差、模型间训练集重叠引起的依赖或分界估计的不确定性。',
              '- 本轮没有检验各方案彼此的差值，不应根据其区间简单宣布某方案显著优于另一方案。',
              '- 这是自然用户群体的关联分析，不是对同一用户改变发帖频率的干预，也不证明时间模块的增益。', '',
              '## 复现', '', '在 eot_experiment 目录执行：', '', '```powershell',
              'python -B -m unittest discover -s tests -p test_posting_frequency_robustness.py -v',
              'python -B scripts/evaluate_posting_frequency_robustness.py --bootstrap 20000 --seed 42', '```', '',
              'protocol.json 保存分析规则；source_manifest.json 追溯上一轮与原始证据；acceptance.json 保存本轮核验及输出哈希。', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
