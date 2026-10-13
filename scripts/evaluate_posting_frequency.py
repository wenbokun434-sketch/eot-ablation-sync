"""Read-only, post-hoc frequency subgroup evaluation of frozen OOF predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ('R0', 'R1', 'R2', 'ACC_V2')
GROUPS = ('low', 'medium', 'high')
METRICS = ('accuracy', 'recall', 'precision', 'f1', 'balanced_accuracy', 'roc_auc', 'pr_auc')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def group_users(users):
    u = users.sort_values('user_id').copy().reset_index(drop=True)
    require(not u.user_id.duplicated().any(), 'Duplicate user IDs')
    u['span_days'] = (pd.to_datetime(u.last_created_utc, utc=True) -
                      pd.to_datetime(u.first_created_utc, utc=True)).dt.total_seconds() / 86400
    require(bool(((u.sequence_length >= 2) & (u.span_days > 0)).all()),
            'Undefined frequency: fewer than two posts or nonpositive span; review before evaluation')
    u['posts_per_day'] = (u.sequence_length - 1) / u.span_days
    require(bool(np.isfinite(u.posts_per_day).all()), 'Nonfinite frequency')
    q1, q2 = u.posts_per_day.quantile([1/3, 2/3]).to_numpy()
    require(q1 < q2, 'Tied quantiles: cannot define three distinct frequency intervals')
    # Identical frequencies always share a group; do not force equal sizes by user ID.
    u['frequency_group'] = pd.cut(u.posts_per_day, [-np.inf, q1, q2, np.inf], labels=GROUPS)
    require(not u.frequency_group.isna().any(), 'Missing group assignment')
    require(set(u.frequency_group.astype(str)) == set(GROUPS), 'Empty frequency group')
    return u, [float(q1), float(q2)]


def metrics(y, probability, pred):
    y, probability, pred = np.asarray(y), np.asarray(probability), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    both = bool((y == 1).any() and (y == 0).any())
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp+tn)/len(y),
                recall=recall, precision=tp/(tp+fp) if tp+fp else 0.0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                balanced_accuracy=(recall+tn/(tn+fp))/2 if both else np.nan,
                roc_auc=float(roc_auc_score(y, probability)) if both else np.nan,
                pr_auc=float(average_precision_score(y, probability)) if both else np.nan)


def bootstrap(frame, iterations, seed):
    y = frame.label.to_numpy()
    p = frame.probability.to_numpy()
    d = frame.final_prediction.to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        idx = rng.integers(0, len(frame), len(frame))
        if np.unique(y[idx]).size != 2:
            continue
        draws.append(metrics(y[idx], p[idx], d[idx]))
    require(len(draws) > 0, 'No two-class bootstrap replicates')
    return {name: dict(zip(('lower', 'upper'), map(float, np.quantile(
                [draw[name] for draw in draws], [.025, .975])))) for name in METRICS}, len(draws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    require(args.bootstrap >= 100, 'Use at least 100 bootstrap replicates')
    out = ROOT / 'outputs/analysis/posting_frequency'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'acceptance.json', dict(status='incomplete', reason='Evaluation started; pass is written only after all checks'))
    inputs = {}

    def source(relative):
        path = ROOT / relative
        inputs[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
        return path

    evaluation = 'outputs/optimization/v2/evaluation'
    run_id = read_json(source(f'{evaluation}/run.json'))['run_id']
    run = f'{evaluation}/runs/{run_id}'
    seal = read_json(source(f'{evaluation}/seal.json'))
    # Verify all existing sealed V2 evidence, not only the selected prediction files.
    for relative, expected in seal['artifacts'].items():
        require(sha(source(relative)) == expected, f'Sealed artifact changed: {relative}')
    baseline = read_json(source(f'{run}/summary.json'))
    u, boundaries = group_users(pd.read_parquet(source('data/processed/user_sequences.parquet')))
    require(len(u) == 909 and int(u.label.sum()) == 102, 'Unexpected cohort')
    posts = pd.read_parquet(source('data/processed/post_index.parquet'),
                            columns=['user_id', 'timestamp_unix', 'sequence_index', 'kind'])
    require(set(posts.user_id) == set(u.user_id), 'Post/user ID mismatch')
    agg = posts.groupby('user_id').timestamp_unix.agg(['size', 'min', 'max', 'nunique'])
    aligned = agg.reindex(u.user_id)
    require(np.array_equal(aligned['size'], u.sequence_length), 'Post counts disagree')
    for col, stat in [('first_created_utc', 'min'), ('last_created_utc', 'max')]:
        expected = pd.to_datetime(u[col], utc=True).astype('int64').to_numpy() // 10**9
        require(np.array_equal(aligned[stat].to_numpy(), expected), f'{col} mismatch')
    for _, frame in posts.groupby('user_id'):
        ordered = frame.sort_values('sequence_index')
        require(np.array_equal(ordered.sequence_index, np.arange(len(frame))), 'Invalid post order')
        require(ordered.timestamp_unix.is_monotonic_increasing, 'Nonmonotonic timestamps')
    u['duplicate_timestamp_events'] = (aligned['size']-aligned['nunique']).to_numpy()
    # Flags are descriptive; retain all valid users and never winsorize model inputs.
    u['short_span_under_one_day'] = u.span_days < 1
    u['above_frequency_p99'] = u.posts_per_day > u.posts_per_day.quantile(.99)
    fold_by_user = {}
    for fold in range(5):
        original = read_json(source(f'data/splits/fold_{fold}.json'))
        opt = read_json(source(f'data/optimization/v1/splits/outer_{fold}.json'))
        test = set(original['test_user_ids'])
        require(not (test & set(original['train_user_ids'])), 'Original train/test overlap')
        require(not (test & set(original['validation_user_ids'])), 'Original validation/test overlap')
        require(test == set(opt['outer_roles']['test']), 'Outer test membership changed')
        for role in ('search', 'calibration'):
            require(not (test & set(opt['outer_roles'][role])), f'Outer {role}/test overlap')
        for inner in opt['inner']:
            for ids in inner['roles'].values():
                require(not (test & set(ids)), 'Outer test user in inner development data')
        require(not (test & set(fold_by_user)), 'User appears in multiple test folds')
        fold_by_user.update({uid: fold for uid in test})
    require(set(fold_by_user) == set(u.user_id), 'Test fold coverage mismatch')
    u['fold'] = u.user_id.map(fold_by_user)
    predictions = {}
    for strategy in STRATEGIES:
        f = pd.read_parquet(source(f'{run}/oof_{strategy}.parquet')).sort_values('user_id').reset_index(drop=True)
        require(not f.user_id.duplicated().any() and list(f.user_id) == list(u.user_id), 'OOF ID mismatch')
        require(np.array_equal(f.label, u.label), 'OOF label mismatch')
        require(np.array_equal(f.length, u.sequence_length), 'OOF sequence length mismatch')
        require(bool(np.isfinite(f.probability).all() and f.probability.between(0, 1).all()), 'Invalid probabilities')
        require(bool(np.isfinite(f.threshold_final).all() and f.threshold_final.between(0, 1).all()), 'Invalid thresholds')
        require(np.array_equal(f.final_prediction, (f.probability >= f.threshold_final).astype(int)), 'Saved decision/threshold mismatch')
        for fold in range(5):
            part = pd.read_parquet(source(f'{run}/outer_{fold}/{strategy}.parquet')).sort_values('user_id').reset_index(drop=True)
            require(set(part.user_id) == set(u.loc[u.fold == fold, 'user_id']), 'Fold predictions not test-only')
            pd.testing.assert_frame_equal(f.loc[f.user_id.isin(part.user_id)].reset_index(drop=True)[part.columns], part)
        f['frequency_group'] = u.frequency_group
        f['fold'] = u.fold
        predictions[strategy] = f

    protocol = dict(research_status='post-hoc exploratory subgroup analysis', source_run_id=run_id,
                    frequency='(model input sequence_length - 1) / elapsed calendar days',
                    includes=['submission', 'comment'], max_posts=256,
                    boundaries_posts_per_day=boundaries, boundary_rule='low <= q1; q1 < medium <= q2; high > q2',
                    grouping='pooled OOF frequency terciles, no labels/predictions used; ties stay together',
                    bootstrap=dict(iterations=args.bootstrap, seed=args.seed, level=.95,
                                   method='ordinary user resampling within each fixed group; skip single-class draws without replacement draws'),
                    pr_auc_definition='average precision (AP), not trapezoidal PR area',
                    undefined_precision='0 when no positive predictions',
                    preserved='original per-fold probabilities, final decisions and thresholds',
                    new_training_runs=0, new_inference_runs=0)
    # Persist grouping before any performance calculations.
    write_json(out/'protocol.json', protocol)
    u.to_csv(out/'user_frequency_groups.csv', index=False, encoding='utf-8-sig')
    stats = u.groupby('frequency_group', observed=True).agg(
        users=('label', 'size'), positives=('label', 'sum'), positive_rate=('label', 'mean'),
        frequency_min=('posts_per_day', 'min'), frequency_median=('posts_per_day', 'median'),
        frequency_max=('posts_per_day', 'max'), posts_median=('sequence_length', 'median'),
        span_days_median=('span_days', 'median'), truncated_users=('was_truncated', 'sum'),
        short_span_users=('short_span_under_one_day', 'sum'),
        duplicate_timestamp_users=('duplicate_timestamp_events', lambda s: int((s>0).sum())))
    stats['negatives'] = stats.users-stats.positives
    stats.to_csv(out/'group_statistics.csv', encoding='utf-8-sig')
    u.groupby(['fold', 'frequency_group'], observed=True).label.agg(['size', 'sum']).to_csv(out/'fold_group_counts.csv')
    u.loc[u.short_span_under_one_day | u.above_frequency_p99 | (u.duplicate_timestamp_events > 0)].to_csv(
        out/'frequency_diagnostics.csv', index=False, encoding='utf-8-sig')
    rows, ci_rows, fold_rows = [], [], []
    for strategy, f in predictions.items():
        overall = metrics(f.label, f.probability, f.final_prediction)
        for name in METRICS:
            reference = baseline['metrics'][strategy]
            expected = reference['accuracy'] if name == 'accuracy' else reference['classification'][name]
            require(np.isclose(overall[name], expected), f'Original metric mismatch: {strategy}/{name}')
        for name in ('tp', 'tn', 'fp', 'fn'):
            require(overall[name] == baseline['metrics'][strategy]['counts'][name], 'Original confusion counts mismatch')
        total = np.zeros(4, dtype=int)
        for group in ('all', *GROUPS):
            part = f if group == 'all' else f.loc[f.frequency_group == group]
            point = metrics(part.label, part.probability, part.final_prediction)
            rows.append(dict(strategy=strategy, group=group, users=len(part), positives=int(part.label.sum()), **point))
            if group != 'all':
                total += [point[k] for k in ('tp', 'tn', 'fp', 'fn')]
            ci, valid = bootstrap(part, args.bootstrap, args.seed)
            for name in METRICS:
                ci_rows.append(dict(strategy=strategy, group=group, metric=name, estimate=point[name],
                                    **ci[name], valid_draws=valid, skipped_draws=args.bootstrap-valid))
            print(f'{strategy}/{group}: n={len(part)}, bootstrap={valid}/{args.bootstrap}', flush=True)
        require(np.array_equal(total, [overall[k] for k in ('tp', 'tn', 'fp', 'fn')]), 'Confusion matrix reconstruction failed')
        for (fold, group), part in f.groupby(['fold', 'frequency_group'], observed=True):
            fold_rows.append(dict(strategy=strategy, fold=int(fold), group=str(group), users=len(part),
                                  positives=int(part.label.sum()), **metrics(part.label, part.probability, part.final_prediction)))
    results, cis = pd.DataFrame(rows), pd.DataFrame(ci_rows)
    results.to_csv(out/'metrics.csv', index=False)
    cis.to_csv(out/'confidence_intervals.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv', index=False)
    plot_results(out, results, cis)
    report(out, protocol, stats, results, cis)
    require(all(sha(ROOT/relative) == digest for relative, digest in inputs.items()), 'Source changed during evaluation')
    write_json(out/'source_manifest.json', dict(input_sha256=inputs,
               script_sha256=sha(Path(__file__)),
               tests_sha256=sha(ROOT/'tests/test_posting_frequency.py'), python=sys.version,
               numpy=np.__version__, pandas=pd.__version__))
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'acceptance.json'}
    write_json(out/'acceptance.json', dict(status='pass', users=len(u), unique_users=int(u.user_id.nunique()),
               strategies=list(STRATEGIES), sealed_source_artifacts_verified=len(seal['artifacts']),
               test_membership_verified=True, same_groups_for_all_strategies=True,
               confusion_matrices_reconstruct=True, original_metrics_reproduced=True,
               input_hashes_unchanged=True, new_training_runs=0, new_inference_runs=0, output_sha256=outputs))
    print(f'PASS: {out}', flush=True)


def plot_results(out, results, cis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#245A81', '#AD6B26', '#2A8069', '#916099']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, ['accuracy', 'recall', 'f1', 'balanced_accuracy']):
        for offset, (strategy, color) in enumerate(zip(STRATEGIES, colors)):
            sub = cis.loc[(cis.strategy == strategy) & (cis.metric == metric)].set_index('group').loc[list(GROUPS)]
            x = np.arange(3) + (offset-1.5)*.13
            ax.vlines(x, sub.lower, sub.upper, color=color, alpha=.65)
            ax.plot(x, sub.estimate, 'o', color=color, label=strategy, markersize=5)
        sizes = results.loc[results.strategy == STRATEGIES[0]].set_index('group').users
        ax.set(xticks=np.arange(3), xticklabels=[f'{g.title()} (n={sizes[g]})' for g in GROUPS],
               ylim=(0, 1.03), title=metric.replace('_', ' ').title())
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle('Posting frequency subgroups | Frozen OOF predictions\n95% user bootstrap intervals; exploratory analysis', fontsize=13)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'frequency_performance.{ext}', dpi=180)
    plt.close(fig)


def report(out, protocol, stats, results, cis):
    lines = ['# 发帖频率分层评估', '',
             '本实验为已有测试结果上的探索性分析。复用冻结的折外测试预测，未训练、未推理、未调阈值。', '',
             '## 口径与分组', '',
             '频率 =（模型输入帖子数 − 1）/ 首末帖相隔天数。包括有效主帖和评论，取最早最多 256 条；不是完整历史频率或活跃日频率。',
             f'三分位边界：{protocol["boundaries_posts_per_day"][0]:.9f}、{protocol["boundaries_posts_per_day"][1]:.9f} 条/天。相同频率不拆组，分界值归较低组。',
             '分组依据全体 OOF 用户的频率，未用标签或性能选择边界；属于事后描述，不能视为新独立测试集。', '',
             '| 组别 | 人数 | 正类 | 正类比例 | 帖数中位数 | 跨度中位数（天） | 截断人数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g, r in stats.iterrows():
        lines.append(f'| {g} | {int(r.users)} | {int(r.positives)} | {r.positive_rate:.2%} | {r.posts_median:.0f} | {r.span_days_median:.2f} | {int(r.truncated_users)} |')
    lines += ['', '## 分组结果', '', '| 方案 | 组别 | Accuracy | Recall | Precision | F1 | Balanced Accuracy | ROC-AUC | PR-AUC(AP) | TP/TN/FP/FN |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results.itertuples():
        lines.append(f'| {r.strategy} | {r.group} | {r.accuracy:.2%} | {r.recall:.2%} | {r.precision:.2%} | {r.f1:.2%} | {r.balanced_accuracy:.2%} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.tp}/{r.tn}/{r.fp}/{r.fn} |')
    lines += ['', '## 主要观察', '']
    for strategy in STRATEGIES:
        r = results.loc[results.strategy == strategy].set_index('group').loc[list(GROUPS)]
        recall = '、'.join(f'{v:.2%}' for v in r.recall)
        acc = '、'.join(f'{v:.2%}' for v in r.accuracy)
        trend = '点估计随频率组别升高而上升' if r.recall.is_monotonic_increasing else '并非随频率组别单调上升'
        lines.append(f'- {strategy}：低/中/高频准确率为 {acc}；召回率为 {recall}，{trend}。')
    lines += ['- R2 高频组召回率点估计较高，但高低频召回率区间存在重叠；本轮未检验差值，不能据此宣称统计显著差异。',
              '- 三组全预测为负类时的准确率分别为 ' + '、'.join(f'{1-r:.2%}' for r in stats.positive_rate) + '，说明高频组较高的准确率需要结合类别比例解读。',
              '- R1、R2、ACC_V2 复用相同概率，故组内 ROC-AUC 与 PR-AUC 相同；不同分类表现来自其冻结的决策阈值。',
              '- 本轮可支持“各频率群体表现存在描述性差异”，不能支持“用户增加发帖就会提升预测效果”。',
              '', '![分组性能与置信区间](frequency_performance.png)', '', '## 召回率不确定性', '',
              '| 方案 | 组别 | Recall | 95% 区间 | 有效抽样次数 |', '|---|---|---:|---|---:|']
    for r in cis.loc[(cis.metric == 'recall') & (cis.group != 'all')].itertuples():
        lines.append(f'| {r.strategy} | {r.group} | {r.estimate:.2%} | [{r.lower:.2%}, {r.upper:.2%}] | {r.valid_draws} |')
    lines += ['', '## 解读与限制', '',
              '- 四种方案分别报告；R0 与后续方案的训练/校准流程不同，不能把所有差异归因于阈值。',
              '- 三组正类比例不同，准确率和 PR-AUC 受类别比例影响；应同时看召回率、平衡准确率与混淆矩阵。',
              '- 高频组只有 19 名正类，一名正类预测变化约影响 5.26 个百分点的召回率。不能仅凭点估计认定稳定优劣。',
              '- 低频组与其他组的输入帖数、观察跨度不同，本实验不控制这些混杂因素，也不证明发帖频率的因果效应或时间模块增益。',
              '- 短窗口集中发帖和重复时间戳保留并另列诊断表；频率是收集到的有效事件的观测密度，不等同于平台完整活动率。',
              f'- 置信区间采用组内普通用户级 bootstrap，{protocol["bootstrap"]["iterations"]} 次，seed={protocol["bootstrap"]["seed"]}；单类抽样跳过、不补抽。各方案使用相同用户排序与抽样种子。',
              '- 区间以既有模型、折分配、阈值和分组边界为条件，不包含重训或边界估计的不确定性；未作显著性检验。',
              '- PR-AUC 使用 average precision (AP)；无预测正类时 precision 按 0 计。早期检测不在本轮范围内。', '',
              '## 复现与验收', '', '在 eot_experiment 目录执行：', '',
              '```powershell', 'python -B scripts/evaluate_posting_frequency.py --bootstrap 2000 --seed 42',
              'python -B -m unittest discover -s tests -p test_posting_frequency.py -v', '```', '',
              'acceptance.json 记录覆盖、测试归属、指标还原及文件保护核验；source_manifest.json 保存来源与哈希。',
              'metrics.csv 为点估计，confidence_intervals.csv 为全部指标区间，fold_metrics.csv 为逐折补充结果；单类折分组的 AUC/平衡准确率为空。',
              'user_frequency_groups.csv 含逐用户分组；frequency_diagnostics.csv 含短跨度、重复时间戳或高于 P99 的用户，均未删改。', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
