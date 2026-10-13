"""V01 document/source audit only. Standard library; no training or threshold search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/optimization/v2/protocol'
V1 = ROOT / 'outputs/optimization/v1'
DOC = ROOT / 'docs/optimization/v2/PROTOCOL_v2.md'
TODO = ROOT / '阈值优化TODO_v2.md'

RULES = {
    'protocol_id': 'eot_accuracy_first_threshold_v2', 'version': 2,
    'research_status': 'post_hoc_exploratory_not_independent_confirmation',
    'new_training_runs': 0, 'new_model_inference_runs': 0,
    'candidate': 'A', 'seeds': [42, 43, 44],
    'aggregation': 'arithmetic_mean_probabilities_within_fold',
    'selection': {
        'grid_source': 'eot.metrics.DEFAULT_THRESHOLD_GRID', 'min': 0, 'max': 1,
        'step': 0.005, 'count': 201, 'positive_rule': 'probability >= threshold',
        'reference': 'R1_same_selection_users_same_A_probabilities',
        'constraints': ['correct >= R1.correct', 'tp >= R1.tp'],
        'maximize_lexicographic': ['correct', 'tp', 'precision_exact_fraction', 'threshold'],
        'zero_predicted_positive_precision': 0,
        'unchanged_when': 'best.correct == R1.correct and best.tp == R1.tp',
        'unchanged_action': 'return_reference_early_threshold',
        'empty_feasible_or_invalid_input': 'raise_and_stop',
        'test_metric_floor': None,
    },
    'internal_validation': {
        'outer_folds': [0, 1, 2, 3, 4], 'inner_folds': [0, 1, 2],
        'split': 'StratifiedKFold', 'n_splits': 2, 'shuffle': True,
        'sort': 'user_id', 'seed_formula': '30042 + 10*outer_fold + inner_fold',
        'evaluation_halves': 30,
        'early_threshold': 'select_on_fit_half_only_using_original_F_latency_rule',
        'policies': ['R1', 'v1_TP_first', 'v2_correct_first'],
        'apply_to_evaluation_half_without_refitting': True,
        'whole_H_threshold_reuse': False, 'cross_outer_selection': False,
        'performance_based_policy_switch': False,
    },
    'calibration': {
        'role': 'C_per_outer_fold', 'users': 91, 'positives': 10,
        'early_threshold': 'v1_final_lock.calibration.R1.threshold_early_unchanged',
        'inner_threshold_transfer': False, 'cross_outer_pooling': False,
        'retrain_or_recalibrate_probabilities': False,
    },
    'evaluation': {
        'object': 'ACC_V2', 'gate': 'all_five_v2_locks_and_parent_lock_valid',
        'unique_users': 909, 'primary_comparison': 'ACC_V2_vs_R2',
        'other_comparisons': ['ACC_V2_vs_R1', 'ACC_V2_vs_R0'],
        'early_decisions': 'identical_to_v1_R1_R2',
        'bootstrap': {'unit': 'users', 'paired': True, 'iterations': 2000,
                      'seed': 42, 'level': 0.95, 'method': 'percentile',
                      'single_class': 'skip_report_counts_no_redraw'},
        'dual_improvement': 'accuracy_delta > 0 and recall_delta > 0',
        'performance_required_for_technical_acceptance': False,
        'reselection_after_test': False,
    },
    'tasks': ['V01', 'V02', 'V03', 'V04'],
    'code_root': 'src/eot/optimization/threshold_v2',
    'output_root': 'outputs/optimization/v2',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def create(path, value):
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(content)


def check_documents():
    text = TODO.read_text(encoding='utf-8')
    tasks = re.findall(r'^- \[([ x])\] \*\*(V\d{2}) ', text, re.M)
    if tasks not in ([(' ', f'V{i:02}') for i in range(1, 5)],
                     [('x', 'V01')] + [(' ', f'V{i:02}') for i in range(2, 5)]):
        raise AssertionError('V01 requires an ordered four-task list with no later task checked')
    sections = re.split(r'^- \[[ x]\] \*\*V\d{2} ', text, flags=re.M)[1:]
    for section in sections:
        for key in ('修改目标', '允许修改的范围', '不允许破坏的逻辑', '验收标准'):
            if key not in section:
                raise AssertionError(f'Task requirement missing: {key}')
    doc = DOC.read_text(encoding='utf-8')
    if len(re.findall(r'^## [1-8]\. ', doc, re.M)) != 8:
        raise AssertionError('Protocol must contain eight sections')
    for token in ('TP >= TP_ref', 'TP+TN >= K_ref', '(正确人数, TP, Precision, threshold)',
                  'unchanged_reference', '30042+10*f+j', '2000', 'ACC_V2_vs_R2', '新增训练次数'):
        if token not in doc:
            raise AssertionError(f'Protocol rule missing: {token}')
    old = (ROOT / '优化TODO.md').read_text(encoding='utf-8')
    if re.findall(r'^- \[([ x])\] \*\*(O\d{2}) ', old, re.M) != [('x', f'O{i:02}') for i in range(1, 16)]:
        raise AssertionError('Old O01-O15 checklist is not complete')


def groups():
    result = []
    for f in range(5):
        for j in range(3):
            base = V1 / f'stability/evaluation/outer_{f}/inner_{j}/A/ensemble'
            result.append({'role': 'inner_holdout', 'outer_fold': f, 'inner_fold': j,
                           'files': [(base / name).relative_to(ROOT).as_posix()
                                     for name in ('final.parquet', 'prefix.parquet', 'ensemble.json')]})
        base = V1 / f'final/calibration/outer_{f}/bundle'
        result.append({'role': 'calibration', 'outer_fold': f,
                       'files': [(base / name).relative_to(ROOT).as_posix()
                                 for name in ('final.parquet', 'prefix.parquet', 'bundle.json')] +
                                [(V1 / f'final/locks/outer_{f}.json').relative_to(ROOT).as_posix()]})
    for group in result:
        for name in group['files']:
            if not (ROOT / name).is_file():
                raise AssertionError(f'Missing reusable input: {name}')
    return result


def capture():
    cache = {}
    def actual(path):
        path = path.resolve()
        if path not in cache:
            cache[path] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return cache[path]
    # Verify, rather than silently replacing, all earlier recorded hashes.
    a = read(V1 / 'delivery/o15_acceptance.json')
    if a['status'] != 'pass':
        raise AssertionError('O15 not accepted')
    for relative, expected in a['artifacts'].items():
        if actual(ROOT / relative)['sha256'] != expected:
            raise AssertionError(f'O15 delivery mismatch: {relative}')
    old_sources = read(V1 / 'delivery/source_manifest.json')
    for relative, expected in old_sources.items():
        if actual(ROOT / relative) != expected:
            raise AssertionError(f'O15 frozen source mismatch: {relative}')
    protection = read(V1 / 'protocol/baseline_protection.json')
    anchors = {'project': ROOT, 'raw': Path(protection['raw_root']).resolve()}
    protected = []
    for row in protection['records']:
        observed = actual(anchors[row['anchor']] / row['path'])
        if observed != {'sha256': row['sha256'], 'bytes': row['bytes']}:
            raise AssertionError(f'Original protection mismatch: {row["path"]}')
        protected.append({k:row[k] for k in ('anchor', 'path', 'sha256', 'bytes')})
    paths = {ROOT / name for name in old_sources}
    paths.update(ROOT / name for name in a['artifacts'])
    paths.update([ROOT / 'README.md', ROOT / '优化TODO.md', ROOT / 'TODO.md', V1 / 'delivery/o15_acceptance.json'])
    v1_files = sorted(p for p in V1.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(v1_files)
    records = {p.relative_to(ROOT).as_posix(): actual(p) for p in sorted(paths)}
    return {'project_files': records, 'original_protection': protected,
            'raw_root': protection['raw_root'],
            'v1_members': [p.relative_to(ROOT).as_posix() for p in v1_files],
            'reusable_prediction_groups': groups(),
            'verified_old_delivery_files': len(a['artifacts']),
            'verified_old_source_files': len(old_sources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'verify'])
    args = parser.parse_args()
    check_documents()
    if args.action == 'snapshot':
        if OUT.exists():
            raise FileExistsError('Protocol directory already exists; refusing to overwrite')
        evidence = capture()
        OUT.mkdir(parents=True)
        create(OUT / 'rules.json', RULES)
        create(OUT / 'sources.json', evidence)
        targets = [DOC, Path(__file__), OUT / 'rules.json', OUT / 'sources.json']
        create(OUT / 'protocol_lock.json', {'protocol_id': RULES['protocol_id'],
               'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in targets}})
    else:
        lock = read(OUT / 'protocol_lock.json')
        for relative, expected in lock['files'].items():
            if sha(ROOT / relative) != expected:
                raise AssertionError(f'v2 protocol hash mismatch: {relative}')
        if read(OUT / 'rules.json') != RULES:
            raise AssertionError('Machine rules differ from frozen audit specification')
        evidence = capture()
        if evidence != read(OUT / 'sources.json'):
            raise AssertionError('Frozen input content or v1 membership changed')
    result = {'task': 'V01', 'status': 'pass', 'action': args.action,
              'project_files_verified': len(evidence['project_files']),
              'original_protected_files_verified': len(evidence['original_protection']),
              'inner_groups': 15, 'calibration_groups': 5,
              'new_training_runs': 0, 'new_inference_runs': 0,
              'new_threshold_searches': 0, 'new_test_evaluations': 0,
              'later_tasks_started': False,
              'lock_sha256': sha(OUT / 'protocol_lock.json')}
    if args.action == 'verify' and not (OUT / 'v01_acceptance.json').exists():
        create(OUT / 'v01_acceptance.json', {
            **result, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'scikit_learn': version('scikit-learn'),
            'todo_status_before_single_tick': 'V01-V04 unchecked'})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
