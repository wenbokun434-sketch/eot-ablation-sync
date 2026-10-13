"""V02 fixed two-fold cross-evaluation, exclusively on stored inner H users.

python -m eot.optimization.threshold_v2.validation build
python -m eot.optimization.threshold_v2.validation audit
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from eot.optimization.evaluation import DECISION_COLUMNS, decision_frame, evaluate_frame
from eot.optimization.splits import digest
from .selection import fit_policies
from .sources import Sources, read, sha, subset, verify_protected

POLICIES = ('R1', 'R2', 'ACC_V2')
CODE = ('__init__.py', 'selection.py', 'sources.py', 'validation.py')


def encoded(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def store_json(path, record, persist):
    text = encoded(record)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Existing artifact cannot be replayed: {path}')
    elif persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    else:
        raise FileNotFoundError(path)


def store_frame(path, frame, persist):
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
    elif persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression='zstd')
    else:
        raise FileNotFoundError(path)


def frame_hash(frame):
    # Null is a missing early alert, not a non-finite model probability.
    rows = [[None if pd.isna(x) else x.item() if isinstance(x, np.generic) else x for x in row]
            for row in frame[list(DECISION_COLUMNS)].itertuples(index=False, name=None)]
    return digest(rows)


def differences(candidate, reference):
    return {'accuracy': candidate['accuracy'] - reference['accuracy'],
            'recall': candidate['classification']['recall'] - reference['classification']['recall'],
            **{key: candidate['counts'][key] - reference['counts'][key]
               for key in ('correct', 'tp', 'tn', 'fp', 'fn')}}


def evaluate_half(parent, fit_ids, evaluation_ids):
    """Freeze a selection using only Q, then use P to measure it."""
    if set(fit_ids) & set(evaluation_ids) or set(fit_ids) | set(evaluation_ids) != set(parent.user_ids):
        raise ValueError('Fit/evaluation cohorts must partition the inner H')
    fit = subset(parent, fit_ids)
    selection = fit_policies(fit.sequences(), expected_fit_ids=fit_ids)
    selection_sha = digest(selection)
    evaluation = subset(parent, evaluation_ids)
    frames, metrics = {}, {}
    for name, policy in selection['policies'].items():
        frame = decision_frame(evaluation, threshold_final=policy['threshold_final'],
                               threshold_early=policy['threshold_early'], consecutive=2)
        frames[name] = frame
        metrics[name] = evaluate_frame(frame)
    if digest(selection) != selection_sha:
        raise AssertionError('Evaluation altered the locked fit result')
    early_cols = ['user_id', 'threshold_early', 'alert_prediction', 'alert_position']
    for name in ('R2', 'ACC_V2'):
        pd.testing.assert_frame_equal(frames[name][early_cols], frames['R1'][early_cols])
    return {'fit': selection, 'fit_sha256': selection_sha,
            'evaluation_user_ids': evaluation.user_ids,
            'evaluation_labels_sha256': digest([[r.user_id, int(r.label)] for r in evaluation.final.itertuples()]),
            'evaluation_predictions_sha256': evaluation.predictions_sha256(),
            'metrics': metrics,
            'differences': {f'ACC_V2_vs_{name}': differences(metrics['ACC_V2'], metrics[name]) for name in ('R1', 'R2')},
            'decision_sha256': {name: frame_hash(frame) for name, frame in frames.items()}}, frames


def identity(root, sources):
    directory = Path(__file__).parent
    return {'protocol_lock_sha256': sources.lock_hash, 'stage': 'V02_inner_cross_evaluation',
            'sources_manifest_sha256': sha(root / 'outputs/optimization/v2/protocol/sources.json'),
            'code': {p.relative_to(root).as_posix(): sha(p) for p in [directory / name for name in CODE]},
            'environment': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'pyarrow')},
            'seeds': [42, 43, 44], 'candidate': 'A'}


def report(summary):
    lines = ['# V02 内部交叉评价报告', '',
             '范围：15 组内层 H、30 个评价半折。每组在拟合半集选早期及最终阈值，再在另一半评价；三种策略共用模型概率、Q/P 成员和 Q 早期阈值。', '',
             '**本报告不是外层测试成绩。** 不同外层 S 相互重叠，下表不得再合并为独立 OOF；没有读取 C/T 标签、没有训练或模型推理。', '',
             'R1：共用 Q 的早期阈值；R2：旧 TP 优先策略；ACC_V2：正确人数优先、正确人数和 TP 均受同 Q 的 R1 下限约束。校准约束不保证评价集无下降。', '',
             '| 外层 | 策略 | 用户数 | 正确数 | TP | FP | FN | Accuracy | Recall |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for outer in summary['outer_results']:
        for name in POLICIES:
            m = outer['metrics'][name]
            c = m['counts']
            lines.append(f"| {outer['outer_fold']} | {name} | {c['users']} | {c['correct']} | {c['tp']} | {c['fp']} | {c['fn']} | {m['accuracy']:.4%} | {m['classification']['recall']:.4%} |")
    lines += ['', '## 新策略相对旧 R2 的变化', '',
              '| 外层 | Δ正确人数 | ΔTP | ΔFP | ΔFN | ΔAccuracy（百分点） | ΔRecall（百分点） |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for outer in summary['outer_results']:
        d = outer['differences']['ACC_V2_vs_R2']
        lines.append(f"| {outer['outer_fold']} | {d['correct']:+d} | {d['tp']:+d} | {d['fp']:+d} | {d['fn']:+d} | {d['accuracy']*100:+.4f} | {d['recall']*100:+.4f} |")
    lines += ['', '全部正负变化均保留。内部结果不触发新候选或修改规则；V02 技术验收通过才可进入 V03，届时在各外层 C 上选择实际最终阈值，部署早期阈值仍保持 v1 原值。', '',
              '各半折成员、拟合标签/概率哈希、201 点搜索表、选择及回退原因、评价指标在 outer_f/inner_j/half_h/result.json；三种评价决策为同目录 parquet。',
              '每个外层汇总决策及全部分类/早期指标在 outer_f 下；summary.json 保存源文件、代码、环境、协议与运行身份。不同策略仅最终阈值不同，早期决策须逐用户一致。', '',
              '复现：在项目根目录设置 PYTHONPATH=src 后，运行 python -m eot.optimization.threshold_v2.validation audit；仅读取已有内层预测，不训练、不推理。', '']
    return '\n'.join(lines)


def compute(root, persist):
    sources = Sources(root)
    binding = identity(root, sources)
    run_id = digest(binding)
    base = root / 'outputs/optimization/v2/validation/runs' / run_id
    results, half_count = [], 0
    for f in range(5):
        outer_frames = {name: [] for name in POLICIES}
        for j in range(3):
            parent, split = sources.inner(f, j)
            splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=30042 + 10*f + j)
            seen = []
            for h, (q, p) in enumerate(splitter.split(parent.final.user_id, parent.final.label)):
                fit_ids = parent.final.iloc[q].user_id.tolist()
                evaluation_ids = parent.final.iloc[p].user_id.tolist()
                item, frames = evaluate_half(parent, fit_ids, evaluation_ids)
                item = {'outer_fold': f, 'inner_fold': j, 'half': h,
                        'split_seed': 30042+10*f+j, 'run_id': run_id,
                        'parent_provenance_sha256': digest(parent.provenance),
                        'split_manifest_sha256': sources.used[f'data/optimization/v1/splits/outer_{f}.json'], **item}
                half_dir = base / f'outer_{f}/inner_{j}/half_{h}'
                store_json(half_dir / 'result.json', item, persist)
                for name, frame in frames.items():
                    store_frame(half_dir / f'{name}.parquet', frame, persist)
                    outer_frames[name].append(frame)
                seen += evaluation_ids
                half_count += 1
            if sorted(seen) != parent.user_ids or len(seen) != len(set(seen)):
                raise AssertionError('Every H user must be evaluated exactly once')
            print(f'V02 {"build" if persist else "replay"}: outer={f}, inner={j}, both halves passed', flush=True)
        metrics = {}
        for name, frames in outer_frames.items():
            pooled = pd.concat(frames, ignore_index=True).sort_values('user_id', kind='stable').reset_index(drop=True)
            if pooled.user_id.duplicated().any() or pooled.user_id.tolist() != sorted(split['outer_roles']['search']):
                raise AssertionError('Outer internal coverage must equal S exactly once')
            store_frame(base / f'outer_{f}/{name}.parquet', pooled, persist)
            m = evaluate_frame(pooled)
            # evaluate_frame reports the first row's threshold; pooled users have different
            # fit-half thresholds, so do not present that scalar as a global threshold.
            m.pop('thresholds')
            metrics[name] = m
        result = {'outer_fold': f, 'metrics': metrics,
                  'differences': {f'ACC_V2_vs_{name}': differences(metrics['ACC_V2'], metrics[name]) for name in ('R1', 'R2')}}
        store_json(base / f'outer_{f}/summary.json', result, persist)
        results.append(result)
    if half_count != 30:
        raise AssertionError('Expected 30 evaluation halves')
    summary = {'task': 'V02', 'status': 'complete_pending_acceptance', 'run_id': run_id,
               'binding': binding, 'inner_groups': 15, 'evaluation_halves': half_count,
               'outer_results': results, 'input_files_sha256': sources.used,
               'new_training_runs': 0, 'new_inference_runs': 0, 'calibration_test_labels_read': False,
               'cross_outer_pooling': False, 'actual_final_thresholds_generated': False}
    store_json(base / 'summary.json', summary, persist)
    text = report(summary)
    path = base / 'REPORT.md'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise AssertionError('Report content differs from replay')
    elif persist:
        path.write_text(text, encoding='utf-8', newline='\n')
    else:
        raise FileNotFoundError(path)
    return base, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'audit'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    protection = verify_protected(root)
    print('Protocol and all original protected files verified', flush=True)
    base, summary = compute(root, persist=args.action == 'build')
    pointer = root / 'outputs/optimization/v2/validation/run.json'
    store_json(pointer, {'run_id': summary['run_id'], 'directory': base.relative_to(root).as_posix()}, args.action == 'build')
    if args.action == 'audit':
        if protection != verify_protected(root):
            raise AssertionError('Protection metadata changed during replay')
        tests = root / 'outputs/optimization/v2/validation/tests.json'
        test_record = read(tests)
        if test_record.get('status') != 'pass' or test_record.get('failures') or test_record.get('errors'):
            raise AssertionError('Successful V02 tests are required')
        for relative, expected in test_record['code_sha256'].items():
            if sha(root / relative) != expected:
                raise AssertionError('Tested code no longer matches')
        artifacts = [p for p in base.rglob('*') if p.is_file()]
        acceptance = {'task': 'V02', 'status': 'pass', 'run_id': summary['run_id'],
                      'protection': protection, 'replay': 'all_30_halves_and_all_metrics_exact',
                      'tests_sha256': sha(tests), 'tests': test_record['tests'],
                      'test_code_sha256': test_record['code_sha256'],
                      'inner_groups': 15, 'evaluation_halves': 30,
                      'calibration_test_labels_read': False, 'actual_final_thresholds_generated': False,
                      'later_tasks_started': False,
                      'artifacts': {p.relative_to(root).as_posix(): sha(p) for p in sorted(artifacts)}}
        store_json(root / 'outputs/optimization/v2/validation/v02_acceptance.json', acceptance, True)
        print(encoded({k:v for k,v in acceptance.items() if k not in ('artifacts', 'test_code_sha256')}), flush=True)
    else:
        print(f'Build complete, run_id={summary["run_id"]}; audit required before ticking V02', flush=True)


if __name__ == '__main__':
    main()
