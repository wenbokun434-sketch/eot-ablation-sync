"""V03: C-only calibration and five-fold lock. No test prediction reader.

python -m eot.optimization.threshold_v2.calibration build
python -m eot.optimization.threshold_v2.calibration audit
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eot.optimization.evaluation import (FINAL_COLUMNS, PREFIX_COLUMNS, decision_frame,
                                        evaluate_frame, frame_digest)
from eot.optimization.splits import digest
from .selection import select_accuracy_first
from .sources import Sources, bundle, read, sha, verify_protected
from .validation import store_json, store_frame, frame_hash, encoded, differences

BASE = Path('outputs/optimization/v2')
CODE = ('__init__.py', 'selection.py', 'sources.py', 'validation.py', 'calibration.py')


def require_v02(root):
    path = root / BASE / 'validation/v02_acceptance.json'
    acceptance = read(path)
    if acceptance['task'] != 'V02' or acceptance['status'] != 'pass' or acceptance['evaluation_halves'] != 30:
        raise ValueError('Complete V02 acceptance required')
    for relative, expected in {**acceptance['artifacts'], **acceptance['test_code_sha256']}.items():
        if sha(root / relative) != expected:
            raise ValueError(f'V02 frozen artifact/code changed: {relative}')
    if sha(root / BASE / 'validation/tests.json') != acceptance['tests_sha256']:
        raise ValueError('V02 test evidence changed')
    return sha(path)


def identity(root):
    source = Sources(root)
    directory = Path(__file__).parent
    return {'protocol_lock_sha256': source.lock_hash, 'stage': 'V03_C_calibration',
            'candidate': 'A', 'seeds': [42, 43, 44], 'v02_acceptance_sha256': require_v02(root),
            'code': {p.relative_to(root).as_posix(): sha(p) for p in [directory / n for n in CODE]},
            'environment': {n: version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'pyarrow')}}


def load_calibration(source, f, *, role='calibration'):
    if role != 'calibration' or type(f) is not int or f not in range(5):
        raise ValueError('V03 permits only one specified calibration fold')
    split = source.json(f'data/optimization/v1/splits/outer_{f}.json')
    ids = sorted(split['outer_roles']['calibration'])
    if len(ids) != 91 or len(set(ids)) != 91:
        raise ValueError('C must contain 91 unique users')
    if set(ids) & (set(split['outer_roles']['search']) | set(split['outer_roles']['test'])):
        raise ValueError('Calibration users overlap S/T')
    group = next(g for g in source.manifest['reusable_prediction_groups']
                 if g['role'] == 'calibration' and g['outer_fold'] == f)
    for relative in group['files']:
        source.checked(relative)
    old_relative = f'outputs/optimization/v1/final/locks/outer_{f}.json'
    wrapper = source.json(old_relative)
    old = wrapper['lock']
    if digest(old) != wrapper['sha256'] or old['outer_fold'] != f or old['selected_candidate'] != 'A':
        raise ValueError('Invalid frozen v1 final lock')
    if sha(source.checked('outputs/optimization/v1/locked/final_lock.json')) != old['o13_parent_lock_file_sha256']:
        raise ValueError('v1 parent plan no longer matches')
    directory = f'outputs/optimization/v1/final/calibration/outer_{f}/bundle'
    meta = source.json(directory + '/bundle.json')
    provenance = meta['bundle']
    if digest(provenance) != meta['sha256'] or meta['sha256'] != old['calibration']['bundle']['provenance_sha256']:
        raise ValueError('C bundle provenance hash differs')
    for key, expected in {'role': 'calibration', 'stage': 'final', 'outer_fold': f,
                           'inner_fold': None, 'candidate_id': 'A', 'seeds': [42,43,44],
                           'users': 91, 'class_counts': {'positive':10,'negative':81},
                           'aggregation':'arithmetic_mean_probabilities_across_seeds_within_fold'}.items():
        if provenance[key] != expected:
            raise ValueError(f'C bundle role/membership differs: {key}')
    for name, expected in old['calibration']['bundle']['files_sha256'].items():
        if sha(source.checked(directory + '/' + name)) != expected:
            raise ValueError('Locked C bundle file differs')
    parent = bundle(pd.read_parquet(source.checked(directory + '/final.parquet')),
                    pd.read_parquet(source.checked(directory + '/prefix.parquet')), ids, provenance)
    if int(parent.final.label.sum()) != 10:
        raise ValueError('C must contain 10 positive users')
    for name, columns in (('final', FINAL_COLUMNS), ('prefix', PREFIX_COLUMNS)):
        if frame_digest(getattr(parent,name),columns) != provenance[name+'_sha256']:
            raise ValueError('C frame content hash differs')
    if digest(ids) != provenance['cohort_user_ids_sha256'] or digest(ids) != old['selection']['cohort_user_ids_sha256']:
        raise ValueError('C cohort hash differs')
    if digest([[r.user_id,int(r.label)] for r in parent.final.itertuples()]) != provenance['labels_sha256']:
        raise ValueError('C labels hash differs')
    if parent.predictions_sha256() != provenance['predictions_sha256']:
        raise ValueError('C prefix prediction hash differs')
    if sorted(r['seed'] for r in old['final_runs']) != [42,43,44]:
        raise ValueError('Exactly three final seeds required')
    seed_bundles = []
    for seed in (42,43,44):
        row = next(r for r in old['final_runs'] if r['seed'] == seed)
        link = next(r for r in provenance['seed_runs'] if r['seed'] == seed)
        expected_link = {'seed':seed, 'run_id':row['run_id'], 'run_sha256':row['run_json_sha256'],
                         'checkpoint_sha256':row['checkpoint_sha256']}
        if link != expected_link:
            raise ValueError('C seed provenance differs from locked final model')
        run = source.json(row['run_json'])
        checkpoint = source.checked(row['checkpoint'])
        if sha(checkpoint) != row['checkpoint_sha256'] or sha(source.checked(row['run_json'])) != row['run_json_sha256']:
            raise ValueError('Final checkpoint/run hash differs')
        if run['status'] != 'complete' or run['epoch'] != row['epochs'] or run['checkpoint_sha256'] != row['checkpoint_sha256']:
            raise ValueError('Final run incomplete or mismatched')
        seed_directory = f'outputs/optimization/v1/final/calibration/outer_{f}/seed_{seed}'
        seed_bundles.append(bundle(pd.read_parquet(source.checked(seed_directory+'/final.parquet')),
                                   pd.read_parquet(source.checked(seed_directory+'/prefix.parquet')), ids))
    for name in ('final','prefix'):
        reference = getattr(parent,name)
        for seed_bundle in seed_bundles:
            pd.testing.assert_frame_equal(getattr(seed_bundle,name).drop(columns='probability'),
                                          reference.drop(columns='probability'))
        if not np.array_equal(np.mean([getattr(b,name).probability.to_numpy() for b in seed_bundles],axis=0),
                              reference.probability.to_numpy()):
            raise ValueError('Saved C ensemble is not exact three-seed mean')
    return parent, old


def calibrate(root, f, binding):
    source = Sources(root)
    parent, old = load_calibration(source,f)
    reference = old['calibration']['R1']
    r = reference['threshold_early']
    if reference['threshold_final'] != r:
        raise ValueError('R1 should use the same early/final threshold')
    selection = select_accuracy_first(parent.sequences(),r,expected_ids=parent.user_ids)
    frames, metrics = {}, {}
    for name, final_threshold in (('R1',r), ('R2',old['calibration']['R2']['threshold_final']),
                                   ('ACC_V2',selection['threshold_final'])):
        frames[name] = decision_frame(parent,threshold_final=final_threshold,threshold_early=r,consecutive=2)
        metrics[name] = evaluate_frame(frames[name])
    for name in ('R1','R2'):
        expected = old['calibration'][name]['counts']
        for key in ('tp','tn','fp','fn','correct'):
            if metrics[name]['counts'][key] != expected[key]:
                raise ValueError(f'Frozen C reference counts cannot be replayed: {name}/{key}')
        if metrics[name]['classification']['precision'] != expected['precision']:
            raise ValueError('Frozen calibration precision cannot be replayed')
    if metrics['ACC_V2']['counts'] != selection['counts']:
        raise ValueError('Last-prefix selection and exported decision counts differ')
    for name in ('R2','ACC_V2'):
        pd.testing.assert_frame_equal(frames[name][['user_id','alert_prediction','alert_position','threshold_early']],
                                      frames['R1'][['user_id','alert_prediction','alert_position','threshold_early']])
    run_id = digest(binding)
    output = {'task':'V03','run_id':run_id,'outer_fold':f,'label_role':'calibration',
              'users':91,'positive_users':10,'user_ids':parent.user_ids,
              'cohort_sha256':digest(parent.user_ids), 'labels_sha256':parent.provenance['labels_sha256'],
              'predictions_sha256':parent.predictions_sha256(), 'selection':selection,
              'metrics':metrics,'differences':{f'ACC_V2_vs_{n}':differences(metrics['ACC_V2'],metrics[n]) for n in ('R1','R2')},
              'decision_sha256':{n:frame_hash(frame) for n,frame in frames.items()},
              'old_final_threshold':old['calibration']['R2']['threshold_final'],
              'old_early_threshold':r,'new_early_threshold':selection['threshold_early'],
              'v1_final_lock_sha256':source.used[f'outputs/optimization/v1/final/locks/outer_{f}.json'],
              'final_runs':old['final_runs'], 'input_files_sha256':source.used}
    return output,frames


def child_record(output, binding, artifact_hashes):
    return {'task':'V03','schema_version':1,'outer_fold':output['outer_fold'],
            'run_id':digest(binding),'binding_sha256':digest(binding),
            'protocol_lock_sha256':binding['protocol_lock_sha256'],
            'candidate':'A','seeds':[42,43,44],'label_role':'calibration',
            'threshold_final':output['selection']['threshold_final'],
            'threshold_early':output['new_early_threshold'], 'status':output['selection']['status'],
            'cohort_sha256':output['cohort_sha256'],'selection_sha256':digest(output['selection']),
            'calibration_result_sha256':digest(output), 'artifacts':artifact_hashes,
            'input_files_sha256':output['input_files_sha256'], 'test_labels_accessed':False}


def wrapped(value):
    return {'lock':value,'sha256':digest(value)}


def validate_lock_structure(parent, children):
    """Fail before any test reader can be called; independently testable envelope checks."""
    if digest(parent['lock']) != parent['sha256']:
        raise ValueError('Parent lock checksum mismatch')
    p=parent['lock']
    if p['task'] != 'V03' or p['status'] != 'all_five_calibration_locks_complete' or set(p['children']) != {str(f) for f in range(5)}:
        raise ValueError('Incomplete five-fold parent lock')
    if set(children) != set(range(5)):
        raise ValueError('Five unique child folds required')
    for f,w in children.items():
        if digest(w['lock']) != w['sha256']:
            raise ValueError('Child lock checksum mismatch')
        c=w['lock']
        if c['outer_fold'] != f or c['run_id'] != p['run_id'] or c['binding_sha256'] != digest(p['binding']):
            raise ValueError('Child fold/binding mismatch')
        if c['label_role'] != 'calibration' or c['test_labels_accessed']:
            raise ValueError('Wrong label role in calibration lock')
        if c['protocol_lock_sha256'] != p['binding']['protocol_lock_sha256']:
            raise ValueError('Child protocol differs')


def require_final_locks(root, *, require_acceptance=True):
    """Read-only V04 gate: verifies all locks and replays C, never opens T."""
    root=Path(root).resolve()
    parent=read(root/BASE/'locks/final_lock.json')
    children={f:read(root/BASE/f'locks/runs/{parent["lock"]["run_id"]}/outer_{f}.json') for f in range(5)}
    validate_lock_structure(parent,children)
    p=parent['lock']
    if p['binding'] != identity(root) or p['run_id'] != digest(p['binding']):
        raise ValueError('Current source/code differs from lock binding')
    for relative,expected in p['report_files'].items():
        if sha(root/relative) != expected:
            raise ValueError('Calibration summary/report changed')
    for f,w in children.items():
        entry=p['children'][str(f)]
        expected_path=(BASE/f'locks/runs/{p["run_id"]}/outer_{f}.json').as_posix()
        if entry['path'] != expected_path or sha(root/expected_path) != entry['file_sha256']:
            raise ValueError('Missing or changed child file')
        c=w['lock']
        for relative,expected in c['artifacts'].items():
            if sha(root/relative) != expected:
                raise ValueError('Calibration artifact changed')
        output,_=calibrate(root,f,p['binding'])
        if child_record(output,p['binding'],c['artifacts']) != c:
            raise ValueError('Calibration child cannot be replayed from C')
    if require_acceptance:
        acceptance=read(root/BASE/'calibration/v03_acceptance.json')
        if acceptance['status'] != 'pass' or acceptance['parent_lock_sha256'] != sha(root/BASE/'locks/final_lock.json'):
            raise ValueError('V03 acceptance does not authorize this lock')
        for relative,expected in acceptance['test_code_sha256'].items():
            if sha(root/relative) != expected:
                raise ValueError('V03 tested code changed')
        if sha(root/BASE/'calibration/tests.json') != acceptance['tests_sha256']:
            raise ValueError('V03 test evidence changed')
    return {f:w['lock'] for f,w in children.items()}


def build(root):
    binding=identity(root);run_id=digest(binding);rows=[];children={}
    base=root/BASE/f'calibration/runs/{run_id}'
    for f in range(5):
        result,frames=calibrate(root,f,binding)
        directory=base/f'outer_{f}'
        store_json(directory/'result.json',result,True)
        for name,frame in frames.items():store_frame(directory/f'{name}.parquet',frame,True)
        artifacts={p.relative_to(root).as_posix():sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
        child=child_record(result,binding,artifacts)
        path=root/BASE/f'locks/runs/{run_id}/outer_{f}.json'
        store_json(path,wrapped(child),True)
        children[str(f)]={'path':path.relative_to(root).as_posix(),'file_sha256':sha(path)}
        rows.append({'outer_fold':f,'early_unchanged':result['old_early_threshold'],
                     'old_final':result['old_final_threshold'],'new_final':child['threshold_final'],
                     'status':child['status'],'R1_counts':result['metrics']['R1']['counts'],
                     'R2_counts':result['metrics']['R2']['counts'],'ACC_V2_counts':result['selection']['counts']})
        print(f'Calibrated and sealed outer {f}: final={child["threshold_final"]}, early={child["threshold_early"]}, {child["status"]}',flush=True)
    store_json(base/'summary.json',{'task':'V03','run_id':run_id,'folds':rows,'test_labels_accessed':False},True)
    lines=['# V03 五折实际阈值校准与锁定','',
           '下表仅来自各折 C（91 人、10 名正类），不是外层测试成绩。模型和三种子概率不变；部署早期阈值保持 v1 原值。', '',
           '| 外层 | 旧最终阈值 R2 | 新最终阈值 | 早期阈值（不变） | 状态 | C 正确人数 R1/R2/v2 | C TP R1/R2/v2 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        counts=[row[k+'_counts'] for k in ('R1','R2','ACC_V2')]
        lines.append(f"| {row['outer_fold']} | {row['old_final']:.3f} | {row['new_final']:.3f} | {row['early_unchanged']:.3f} | {row['status']} | {'/'.join(str(c['correct']) for c in counts)} | {'/'.join(str(c['tp']) for c in counts)} |")
    lines+=['','每折均保存 201 点搜索表、参考计数、来源/代码/协议哈希和校准决策。无计数改善时返回 R1 阈值；不得将校准收益当作测试保证。','',
             '总锁为 outputs/optimization/v2/locks/final_lock.json；只有全部五折锁及 V03 验收均有效，V04 入口才可读取测试预测。本轮未执行 V04。','']
    report='\n'.join(lines)
    path=base/'REPORT.md'
    if path.exists():
        if path.read_text(encoding='utf-8') != report:raise ValueError('Existing report differs')
    else:path.write_text(report,encoding='utf-8',newline='\n')
    parent={'task':'V03','status':'all_five_calibration_locks_complete','run_id':run_id,'binding':binding,
            'children':children,'test_labels_accessed':False,
            'report_files':{p.relative_to(root).as_posix():sha(p) for p in (base/'summary.json',base/'REPORT.md')}}
    store_json(root/BASE/'locks/final_lock.json',wrapped(parent),True)
    print('Five locks built; V03 acceptance still required before V04 access',flush=True)


def audit(root):
    protection=verify_protected(root)
    tests=read(root/BASE/'calibration/tests.json')
    if tests['status'] != 'pass' or tests['failures'] or tests['errors']:
        raise ValueError('V03 tests must pass')
    for relative,expected in tests['code_sha256'].items():
        if sha(root/relative) != expected:raise ValueError('V03 tested code differs')
    locks=require_final_locks(root,require_acceptance=False)
    if protection != verify_protected(root):raise ValueError('Protection changed during replay')
    parent_path=root/BASE/'locks/final_lock.json';parent=read(parent_path)['lock']
    artifacts=dict(parent['report_files'])
    for child in locks.values():artifacts.update(child['artifacts'])
    artifacts.update({row['path']:row['file_sha256'] for row in parent['children'].values()})
    artifacts[parent_path.relative_to(root).as_posix()]=sha(parent_path)
    acceptance={'task':'V03','status':'pass','parent_lock_sha256':sha(parent_path),'run_id':parent['run_id'],
                'calibration_folds':5,'users_per_fold':91,'positives_per_fold':10,
                'all_thresholds_replayed':True,'early_thresholds_unchanged':True,
                'test_labels_accessed':False,'new_training_runs':0,'new_inference_runs':0,
                'later_tasks_started':False,'protection':protection,
                'tests':tests['tests'],'tests_sha256':sha(root/BASE/'calibration/tests.json'),
                'test_code_sha256':tests['code_sha256'],'artifacts':artifacts}
    store_json(root/BASE/'calibration/v03_acceptance.json',acceptance,True)
    print(encoded({k:v for k,v in acceptance.items() if k not in ('artifacts','test_code_sha256')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['build','audit'])
    args=parser.parse_args();root=Path(__file__).resolve().parents[4]
    if args.action=='build':
        verify_protected(root)
        build(root)
    else:audit(root)


if __name__=='__main__':main()
