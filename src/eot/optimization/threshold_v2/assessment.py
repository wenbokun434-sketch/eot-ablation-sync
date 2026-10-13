"""V04 one sealed evaluation, followed only by identical-input read-only replay.

python -m eot.optimization.threshold_v2.assessment build
python -m eot.optimization.threshold_v2.assessment audit
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from eot.optimization.comparison import paired_bootstrap, pool_fold_frames
from eot.optimization.evaluation import (decision_frame, evaluate_frame, frame_digest,
                                         FINAL_COLUMNS, PREFIX_COLUMNS, legacy_oof_decision_frame)
from eot.optimization.config import load_optimization_config
from eot.optimization.splits import digest
from . import calibration
from .sources import Sources, bundle, read, sha, verify_protected
from .validation import store_json, store_frame, encoded, differences, frame_hash

BASE = Path('outputs/optimization/v2')
OLD = 'outputs/optimization/v1/final/evaluation'
NAMES = ('R0','R1','R2','ACC_V2')
EARLY = ['user_id','label','length','threshold_early','alert_prediction','alert_position']
CODE = (*calibration.CODE, 'assessment.py')


def pooled_metrics(frame):
    result=evaluate_frame(frame)
    result.pop('thresholds')  # Each user retains their fold's threshold.
    return result


def paired_users(reference,candidate,expected_ids=None):
    for frame in (reference,candidate):
        if frame.user_id.duplicated().any() or frame.user_id.isna().any():
            raise ValueError('Duplicate or null evaluation users')
    if reference.user_id.tolist()!=candidate.user_id.tolist() or reference.label.tolist()!=candidate.label.tolist():
        raise ValueError('Paired users/labels differ')
    if expected_ids is not None and candidate.user_id.tolist()!=sorted(expected_ids):
        raise ValueError('Missing or unauthorized test users')


def conclusion(delta):
    a,r=delta['correct'],delta['tp']
    if a>0 and r>0:return '双提升'
    if a>0 and r==0:return '准确率提高、召回持平'
    if a*r<0:return '取舍变化'
    if a==0 and r>0:return '准确率持平、召回提高'
    if a==0 and r==0:return '两项持平'
    if a<0 and r<0:return '两项下降'
    return '准确率下降、召回持平' if a<0 else '准确率持平、召回下降'


class TestSources:
    """All five locks and V03 acceptance checked before any T content access."""
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.locks=calibration.require_final_locks(self.root)
        self.source=Sources(self.root)
        self.summary=self.source.json(OLD+'/o14_evaluation_summary.json')

    def frame(self,relative):
        return pd.read_parquet(self.source.checked(relative))

    def fold(self,f):
        if type(f) is not int or f not in range(5):raise ValueError('Invalid test fold')
        split=self.source.json(f'data/optimization/v1/splits/outer_{f}.json')
        ids=sorted(split['outer_roles']['test'])
        if set(ids)&(set(split['outer_roles']['search'])|set(split['outer_roles']['calibration'])):
            raise ValueError('Test role overlaps development roles')
        directory=OLD+f'/outer_{f}'
        parent=bundle(self.frame(directory+'/bundle/final.parquet'),self.frame(directory+'/bundle/prefix.parquet'),ids)
        old=next(row for row in self.summary['per_fold'] if row['outer_fold']==f)
        if digest(ids)!=old['test_user_ids_sha256'] or len(ids)!=old['test_users']:
            raise ValueError('Frozen test membership differs')
        for name,columns in (('final',FINAL_COLUMNS),('prefix',PREFIX_COLUMNS)):
            if frame_digest(getattr(parent,name),columns)!=old['bundle_sha256'][name]:
                raise ValueError('Frozen ensemble differs')
        seeds={s:bundle(self.frame(directory+f'/seed_{s}/final.parquet'),
                        self.frame(directory+f'/seed_{s}/prefix.parquet'),ids) for s in (42,43,44)}
        for name in ('final','prefix'):
            frame=getattr(parent,name)
            for b in seeds.values():
                pd.testing.assert_frame_equal(getattr(b,name).drop(columns='probability'),frame.drop(columns='probability'),check_exact=True)
            if not np.array_equal(np.mean([getattr(b,name).probability.to_numpy() for b in seeds.values()],axis=0),frame.probability.to_numpy()):
                raise ValueError('Test probabilities differ from fixed three-seed average')
        references={n:self.frame(directory+f'/decisions/{n}.parquet') for n in ('R1','R2')}
        old_lock=self.source.json(f'outputs/optimization/v1/final/locks/outer_{f}.json')['lock']
        for n,frame in references.items():
            threshold=old_lock['calibration'][n]
            expected=decision_frame(parent,threshold_final=threshold['threshold_final'],threshold_early=threshold['threshold_early'],consecutive=2)
            # v1 persisted int64 labels; the validated bundle normalizes to int8.
            # Compare every value exactly while allowing this storage-width difference.
            pd.testing.assert_frame_equal(frame,expected,check_exact=True,check_dtype=False)
            if evaluate_frame(frame)!=old['metrics'][n]:raise ValueError('Frozen fold metrics differ')
        return parent,seeds,references


def binding(root,reader):
    return {'stage':'V04_single_outer_evaluation','protocol_lock_sha256':reader.source.lock_hash,
            'parent_lock_sha256':sha(root/BASE/'locks/final_lock.json'),
            'v03_acceptance_sha256':sha(root/BASE/'calibration/v03_acceptance.json'),
            'source_manifest_sha256':sha(root/BASE/'protocol/sources.json'),
            'code':{(Path('src/eot/optimization/threshold_v2')/n).as_posix():sha(Path(__file__).parent/n) for n in CODE},
            'environment':calibration.identity(root)['environment'],
            'bootstrap':{'iterations':2000,'seed':42,'level':0.95},'primary_comparison':'ACC_V2_vs_R2'}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def report(summary):
    lines=['# V04 准确率优先最终阈值：外层评价','',
           '本次为查看过 v1 测试结果后的补充探索，不是独立确认。复用已有模型和概率，无训练、无推理、无测试后调参。',
           '技术验收与收益结论分开：验收证据见 v04_acceptance.json；性能结论如下。', '',
           '## 909 人配对 OOF 结果','',
           '| 策略 | 正确数 | TP | TN | FP | FN | Accuracy | Recall | Precision | F1 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for n in NAMES:
        m=summary['metrics'][n];c=m['counts'];k=m['classification']
        lines.append(f"| {n} | {c['correct']} | {c['tp']} | {c['tn']} | {c['fp']} | {c['fn']} | {m['accuracy']:.4%} | {k['recall']:.4%} | {k['precision']:.4%} | {k['f1']:.4%} |")
    lines+=['','R2 为主要比较对象，R1 为同概率共用早期阈值的参考，R0 为历史模型与流程，R0 差异不能全归因于阈值。', '',
            '| 比较 | Δ正确数 | ΔTP | ΔFP | ΔFN | ΔAccuracy（百分点） | ΔRecall（百分点） | 结论 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,row in summary['comparisons'].items():
        d=row['differences']
        lines.append(f"| {name} | {d['correct']:+d} | {d['tp']:+d} | {d['fp']:+d} | {d['fn']:+d} | {d['accuracy']*100:+.4f} | {d['recall']*100:+.4f} | {row['conclusion']} |")
    lines+=['','## 配对不确定性','',
            '三组均使用 2000 次用户级配对 bootstrap、seed=42、95% percentile 区间。单类抽样跳过且不补抽；完整逐指标区间及有效次数见下表。区间以当前模型与阈值为条件，不包含重训或重选不确定性，仅作描述。','',
            '| 比较 | 指标 | 差值（原比例） | 下界 | 上界 | 有效次数 |',
            '| --- | --- | --- | --- | --- | --- |']
    def fmt(x):return 'NA' if x is None else f'{x:.6f}'
    for name,row in summary['comparisons'].items():
        for metric,item in row['bootstrap']['differences'].items():
            lines.append(f"| {name} | {metric} | {fmt(item['point_difference'])} | {fmt(item['lower'])} | {fmt(item['upper'])} | {item['valid_iterations']} |")
        lines+=['',f"{name}：单类跳过 {row['bootstrap']['skipped_single_class_iterations']} 次，有效抽样 {row['bootstrap']['valid_iterations']} 次。",'']
        # Repeat header after the per-comparison note to keep valid Markdown tables.
        if name!=list(summary['comparisons'])[-1]:lines+=['| 比较 | 指标 | 差值（原比例） | 下界 | 上界 | 有效次数 |','| --- | --- | --- | --- | --- | --- |']
    lines+=['','## 逐折与种子描述','',
            '| 外层 | 最终阈值 | 早期阈值 | 策略/种子 | 用户 | 正确 | TP | FP | FN | Accuracy | Recall |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for fold in summary['folds']:
        for n,m in {**fold['metrics'],**{f'ACC_V2 seed {s}':m for s,m in fold['seeds'].items()}}.items():
            c=m['counts'];t=m['thresholds']
            lines.append(f"| {fold['outer_fold']} | {t['final']:.3f} | {t['early']:.3f} | {n} | {c['users']} | {c['correct']} | {c['tp']} | {c['fp']} | {c['fn']} | {m['accuracy']:.4%} | {m['classification']['recall']:.4%} |")
    lines+=['','三个种子均在各折同一组已锁集成阈值下评价，未为种子单独选阈值或择优。完整单种子 OOF 指标保存在 summary.json 的 pooled_seed_metrics。', '',
            '## 全部分类与早期指标','', '| 策略 | 分类指标 | 早期指标 |','| --- | --- | --- |']
    for n,m in summary['metrics'].items():
        lines.append(f"| {n} | {'; '.join(k+'='+fmt(v) for k,v in m['classification'].items())} | {'; '.join(k+'='+fmt(v) for k,v in m['early_detection'].items())} |")
    lines+=['','新策略的早期告警、告警位置、早期阈值和全部早期指标与 R1/R2 逐用户一致。PR-AUC、ROC-AUC、Brier 等基于概率的指标也因概率不变而保持一致。', '',
            '逐折全部指标、各组误报/漏报差值、种子结果、决策哈希、来源哈希、版本及三组全部区间均在 summary.json；逐用户决策在 evaluation/runs/<run_id>/ 下。', '',
            '## 复现与结论边界','',
            '在项目根目录使用 PowerShell：', '', '```powershell',
            "$env:PYTHONPATH=(Resolve-Path src).Path", "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:PYTHONUTF8='1'",
            'python -m eot.optimization.threshold_v2.assessment audit','```','',
            '首次 build 封存评价；重复 build 拒绝。audit 只重算相同输入并核对已封存产物，不改变阈值或覆盖结果。专项测试命令为 python tests/optimization/threshold_v2/test_v04.py。', '',
            '规则、模型、划分和 C 阈值已固定；无论收益如何，本轮不再修改。独立确认需要新的保留数据。','']
    return '\n'.join(lines)


def compute(root,reader,identity,persist):
    run_id=digest(identity);base=root/BASE/'evaluation/runs'/run_id
    collected={n:[] for n in NAMES};seed_frames={s:[] for s in (42,43,44)};folds=[]
    r0=reader.frame(OLD+'/oof_R0_readonly.parquet')
    config=load_optimization_config(root/'configs/optimization/A.yaml',project_root=root)
    pd.testing.assert_frame_equal(r0,legacy_oof_decision_frame(config),check_exact=True)
    for f in range(5):
        parent,seeds,frames=reader.fold(f);lock=reader.locks[f]
        frames['ACC_V2']=decision_frame(parent,threshold_final=lock['threshold_final'],threshold_early=lock['threshold_early'],consecutive=2)
        frames['R0']=r0[r0.user_id.isin(parent.user_ids)].reset_index(drop=True)
        metrics={};seed_metrics={}
        for n,frame in frames.items():
            paired_users(frames['R1'],frame,parent.user_ids)
            if n in ('R2','ACC_V2'):
                pd.testing.assert_frame_equal(frame[EARLY],frames['R1'][EARLY],check_exact=True,check_dtype=False)
            store_frame(base/f'outer_{f}/{n}.parquet',frame,persist)
            metrics[n]=evaluate_frame(frame);collected[n].append(frame)
        for n in ('R2','ACC_V2'):
            if metrics[n]['early_detection']!=metrics['R1']['early_detection']:raise ValueError('Early metrics changed')
        for s,b in seeds.items():
            frame=decision_frame(b,threshold_final=lock['threshold_final'],threshold_early=lock['threshold_early'],consecutive=2)
            store_frame(base/f'outer_{f}/seed_{s}.parquet',frame,persist)
            seed_metrics[str(s)]=evaluate_frame(frame);seed_frames[s].append(frame)
        fold={'outer_fold':f,'users':parent.user_ids,'metrics':metrics,'seeds':seed_metrics,
              'differences':{f'ACC_V2_vs_{n}':differences(metrics['ACC_V2'],metrics[n]) for n in ('R2','R1','R0')},
              'decision_sha256':{n:frame_hash(fr) for n,fr in frames.items()}}
        store_json(base/f'outer_{f}/result.json',fold,persist);folds.append(fold)
        print(f'V04 {"evaluate" if persist else "replay"}: outer {f} passed',flush=True)
    pooled={n:pool_fold_frames(v,roles=['outer_test']*5,name=n) for n,v in collected.items()}
    for n,frame in pooled.items():
        paired_users(r0,frame)
        if len(frame)!=909:raise ValueError('Expected exactly 909 OOF users')
        store_frame(base/f'oof_{n}.parquet',frame,persist)
        if n in ('R1','R2'):
            pd.testing.assert_frame_equal(frame,reader.frame(OLD+f'/oof_{n}.parquet'),check_exact=True)
        if n!='ACC_V2' and evaluate_frame(frame)!=reader.summary['objects'][n]:raise ValueError('Frozen pooled metrics differ')
    seed_metrics={}
    for s,frames in seed_frames.items():
        frame=pool_fold_frames(frames,roles=['outer_test']*5,name=f'seed{s}')
        store_frame(base/f'oof_seed_{s}.parquet',frame,persist);seed_metrics[str(s)]=pooled_metrics(frame)
    metrics={n:pooled_metrics(fr) for n,fr in pooled.items()};comparisons={}
    for n in ('R2','R1','R0'):
        delta=differences(metrics['ACC_V2'],metrics[n])
        print(f'V04 paired bootstrap ACC_V2 vs {n}: 2000 iterations',flush=True)
        bootstrap=clean(paired_bootstrap(pooled[n],pooled['ACC_V2'],iterations=2000,seed=42,level=.95))
        comparisons[f'ACC_V2_vs_{n}']={'differences':delta,'conclusion':conclusion(delta),'bootstrap':bootstrap}
    summary={'task':'V04','run_id':run_id,'binding':identity,'users':909,'unique_users':909,
             'research_status':'post_hoc_exploratory_not_independent_confirmation',
             'primary_comparison':'ACC_V2_vs_R2','folds':folds,'metrics':metrics,'pooled_seed_metrics':seed_metrics,
             'comparisons':comparisons,'input_files_sha256':reader.source.used,
             'early_decisions_unchanged':True,'new_training_runs':0,'new_inference_runs':0,
             'decision_sha256':{n:frame_hash(fr) for n,fr in pooled.items()}}
    store_json(base/'summary.json',summary,persist)
    delivery=root/BASE/'delivery';store_json(delivery/'summary.json',summary,persist)
    path=delivery/'REPORT.md';text=report(summary)
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:raise ValueError('Sealed report differs')
    elif persist:
        with path.open('x',encoding='utf-8',newline='\n') as handle:handle.write(text)
    else:raise FileNotFoundError(path)
    return base,summary


def run(root,action):
    root=Path(root).resolve();pointer=root/BASE/'evaluation/run.json'
    if action=='build' and pointer.exists():raise ValueError('Evaluation already started/sealed; only audit is permitted')
    protection=verify_protected(root)
    reader=TestSources(root);identity=binding(root,reader);run_id=digest(identity)
    tests_path=root/BASE/'evaluation/tests.json';tests=read(tests_path)
    if tests['status']!='pass' or tests['failures'] or tests['errors']:raise ValueError('V04 tests required')
    for relative,expected in tests['code_sha256'].items():
        if sha(root/relative)!=expected:raise ValueError('V04 tested code changed')
    store_json(pointer,{'run_id':run_id,'binding':identity},action=='build')
    base,summary=compute(root,reader,identity,action=='build')
    if verify_protected(root)!=protection:raise ValueError('Protected sources changed')
    artifacts={p.relative_to(root).as_posix():sha(p) for p in sorted(base.rglob('*')) if p.is_file()}
    for path in (pointer,root/BASE/'delivery/REPORT.md',root/BASE/'delivery/summary.json'):
        artifacts[path.relative_to(root).as_posix()]=sha(path)
    seal={'run_id':run_id,'binding_sha256':digest(identity),'artifacts':artifacts}
    store_json(root/BASE/'evaluation/seal.json',seal,action=='build')
    if action=='audit':
        acceptance={'task':'V04','status':'pass','run_id':run_id,'users':909,'unique_users':909,
                    'protection':protection,'five_fold_locks_verified':True,'early_decisions_unchanged':True,
                    'full_metric_and_bootstrap_replay':'exact','bootstrap_iterations_per_comparison':2000,
                    'comparisons':3,'seed':42,'tests':tests['tests'],'tests_sha256':sha(tests_path),
                    'test_code_sha256':tests['code_sha256'],'artifacts':artifacts,
                    'evaluation_seal_sha256':sha(root/BASE/'evaluation/seal.json'),
                    'new_training_runs':0,'new_inference_runs':0,
                    'performance_conclusions':{n:v['conclusion'] for n,v in summary['comparisons'].items()}}
        store_json(root/BASE/'delivery/v04_acceptance.json',acceptance,True)
        print(encoded({k:v for k,v in acceptance.items() if k not in ('artifacts','test_code_sha256')}),flush=True)
    else:print('V04 sealed; read-only audit required before ticking TODO',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['build','audit'])
    run(Path(__file__).resolve().parents[4],parser.parse_args().action)
