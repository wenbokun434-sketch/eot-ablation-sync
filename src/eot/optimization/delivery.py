"""O15 result-only delivery. Never trains, infers, or selects thresholds.

Run from the project with PYTHONPATH=src:
    python -m eot.optimization.delivery --build
    python -m eot.optimization.delivery --audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

BASE = Path('outputs/optimization/v1')
OUT = BASE / 'delivery'
NOTEBOOK = Path('notebooks/eot_optimization.ipynb')
NAMES = ('R0', 'R1', 'R2', 'OPT')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False) + '\n', encoding='utf-8')


def equal(actual, expected, where='root'):
    """Recursive numerical comparison; metadata and shapes must match exactly."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f'{where}: dictionary keys differ')
        for key in expected:
            equal(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{where}: list length differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f'{where}.{i}')
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, atol=1e-12, rtol=0, equal_nan=False):
            raise AssertionError(f'{where}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(f'{where}: values differ')


def load_results(root):
    """Recompute all published point metrics from frozen user predictions."""
    from .config import load_optimization_config
    from .evaluation import evaluate_frame, legacy_oof_decision_frame
    root = Path(root).resolve()
    summary = read(root / BASE / 'final/evaluation/o14_evaluation_summary.json')
    config = load_optimization_config(root / 'configs/optimization/A.yaml', project_root=root)
    frames = {'R0': legacy_oof_decision_frame(config)}
    for name in NAMES[1:]:
        frames[name] = pd.read_parquet(root / BASE / f'final/evaluation/oof_{name}.parquet')
    for name, frame in frames.items():
        if len(frame) != 909 or frame.user_id.duplicated().any():
            raise AssertionError(f'{name}: invalid OOF coverage')
        equal(frame.user_id.tolist(), frames['R0'].user_id.tolist(), name + '.users')
        equal(frame.label.tolist(), frames['R0'].label.tolist(), name + '.labels')
        result = evaluate_frame(frame)
        equal(result, summary['objects'][name], name)
    if not frames['OPT'].equals(frames['R2']):
        raise AssertionError('Expected all-fold A/R2 fallback is no longer identical')
    return summary, frames


def tables(root, summary):
    """Expose every candidate, including unsuccessful and untriggered branches."""
    main, early, folds, seeds, candidates, choices, intervals = [], [], [], [], [], [], []
    for name in NAMES:
        m = summary['objects'][name]
        main.append({'object': name, **m['counts'], **m['classification']})
        early.append({'object': name, **m['early_detection']})
    for fold in summary['per_fold']:
        for name, m in fold['metrics'].items():
            folds.append({'outer_fold': fold['outer_fold'], 'object': name,
                          **m['counts'], **m['classification'], **m['early_detection'],
                          'threshold_final': m['thresholds']['final'],
                          'threshold_early': m['thresholds']['early']})
    for fold, values in summary['per_seed'].items():
        for seed, m in values.items():
            seeds.append({'outer_fold': fold, 'seed': seed, **m['counts'],
                          **m['classification'], **m['early_detection']})
    for f in range(5):
        screen = read(root / BASE / f'screening/reports/outer_{f}.json')
        for name, c in screen['candidates'].items():
            m = c.get('merged') or {}
            candidates.append({'stage': 'O10', 'outer_fold': f, 'candidate': name,
                               'mode': '42', 'passes': c['advances_to_O11'],
                               'status': 'baseline' if name == 'A' else 'evaluated',
                               **{k: m.get(k) for k in ('accuracy', 'recall', 'pr_auc')},
                               **m.get('counts', {})})
        stable = read(root / BASE / f'stability/reports/outer_{f}.json')
        conditional = read(root / BASE / f'conditional/reports/outer_{f}.json')
        groups = [('O11', name, c) for name, c in stable['candidates'].items()]
        for name in conditional['candidate_reports']:
            c = read(root / BASE / f'conditional/reports/outer_{f}/{name}.json')
            groups.append(('O12', name, c))
        for stage, name, c in groups:
            for mode, v in c['modes'].items():
                m = v.get('candidate') or {}
                candidates.append({'stage': stage, 'outer_fold': f, 'candidate': name,
                                   'mode': mode, 'passes': v['promotion_rule_pass'],
                                   'stable': c['stable'], 'noninferior_seeds': ','.join(c['noninferior_seeds']),
                                   'status': 'baseline' if name == 'A' else 'evaluated',
                                   'accuracy': m.get('accuracy'), 'recall': m.get('recall'),
                                   'mean_inner_ap': m.get('mean_inner_ap'), **m.get('counts', {})})
        for name in ('E', 'F1', 'F2'):
            if name not in conditional['candidate_reports']:
                candidates.append({'stage': 'O12', 'outer_fold': f, 'candidate': name,
                                   'mode': 'none', 'status': 'not_triggered',
                                   'reason': conditional['reason'] if name == 'F2' else
                                   'mutually_exclusive_branch_' + conditional['initial_branch']['branch']})
        lock = read(root / BASE / f'locked/outer_{f}.json')['fold']
        seal = read(root / BASE / f'final/locks/outer_{f}.json')['lock']
        r = seal['calibration']['R2']
        choices.append({'outer_fold': f, 'O10_advanced': ','.join(screen['advanced']) or 'none',
                        'O11_stable': ','.join(stable['ranked_stable']) or 'none',
                        'O12_branch': conditional['initial_branch']['branch'],
                        'F2_triggered': conditional['F2_triggered'],
                        'selected': lock['selected_candidate'], 'reason': lock['fallback_reason'],
                        'epochs_42_43_44': '/'.join(str(m['epochs']) for m in lock['models']),
                        'threshold_final': r['threshold_final'], 'threshold_early': r['threshold_early'],
                        'OPT_source': seal['calibration']['opt_source']})
    for name, comparison in summary['comparisons']['comparisons'].items():
        for metric, item in comparison['bootstrap']['differences'].items():
            intervals.append({'comparison': name, 'metric': metric, **item})
    return {k: pd.DataFrame(v) for k, v in dict(main=main, early=early, folds=folds,
            seeds=seeds, candidates=candidates, choices=choices, intervals=intervals).items()}


def md_table(frame):
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '—'
        return f'{v:.6f}' if isinstance(v, float) else str(v).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                       '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                      ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def make_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main = data['main'].set_index('object').loc[list(NAMES)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    colors = ['#68788a', '#9a6b3a', '#287c96', '#32865b']
    for ax, metric, title in zip(axes[:2], ('recall', 'accuracy'), ('Recall (%)', 'Accuracy (%)')):
        values = main[metric] * 100
        bars = ax.bar(NAMES, values, color=colors)
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set(ylim=(0, 105), title=title, ylabel='Percent')
        ax.spines[['top', 'right']].set_visible(False)
    x = np.arange(4)
    axes[2].bar(x - .18, main.fp, width=.36, label='False positives', color='#b76145')
    axes[2].bar(x + .18, main.fn, width=.36, label='False negatives', color='#546981')
    axes[2].set(xticks=x, xticklabels=NAMES, title='OOF errors (909 users)', ylabel='Users')
    axes[2].legend(frameon=False)
    axes[2].spines[['top', 'right']].set_visible(False)
    fig.suptitle('Locked optimization results | OPT equals R2; dual improvement not achieved', fontsize=13)
    return fig


def report_text(data, summary, protocol_href='../../../../docs/optimization/PROTOCOL_v1.md'):
    main = data['main'].set_index('object')
    delta = summary['comparisons']['comparisons']['OPT_vs_R0']['point_differences']
    intro = f'''# EOT 阈值与 Loss 优化最终报告（O15）

**结论：未实现 Recall 与 Accuracy 双提升。** 五个外层均未找到稳定的新 loss 候选，按锁定协议回退 A/R2，OPT 与 R2 的概率、决策和指标完全相同。流程交付与优化成功分别判断。

相对历史 R0，OPT Recall 变化 {delta['recall'] * 100:+.2f} 个百分点，Accuracy 变化 {delta['accuracy'] * 100:+.2f} 个百分点；误报由 {int(main.loc['R0','fp'])} 增至 {int(main.loc['OPT','fp'])}，漏报由 {int(main.loc['R0','fn'])} 降至 {int(main.loc['OPT','fn'])}。这属于召回与误报/准确率的取舍，不能记为新 loss 收益。

## 方法与数据边界

这是查看过原测试结果后、在新运行前锁定的事后补充实验，不是独立确认性研究。协议见 [PROTOCOL_v1.md]({protocol_href})。保留原五折测试成员、最早 256 帖、256 tokens、冻结 encoder、帖子顺序、时间间隔、模型结构和因果/padding mask。

每外层原 train 构成 S（636/637 人），原 validation 构成 C（91 人、10 名正类），测试 T 为 181/182 人。S 内三折留出 H；其余再分 12.5% 作早停 E，实际梯度集合 G 独立拟合时间统计和类别权重。E 仅以 AP 早停，H 用于内层筛选。训练保持 AdamW、lr=0.0008、weight_decay=0.01、batch=128、最多 50 epoch、patience=8、clip=1、非混合精度。

加权 BCE 为最终分支加 λ 倍用户平均前缀分支，正类权重为 α×训练负/正人数比。A=(1,0.5)，B=(1,0.25)，C=(0.5,0.5)，D=(1.5,0.5)。F1 仅最终分支使用 γ=1 Focal，前缀保持 BCE。组合 E 和 F2（γ=2）仅按预定条件触发；本次均未触发，不能记为已训练。

O10 种子 42 比较 A–D；晋级要求三个 H 均可行、合并正确人数不少于 R2 且 TP 更多，并且至少两个 H 的正确人数与 TP 均不低于 R2。O11 对 A 和晋级组补齐 43/44，先平均概率再选阈值，并要求至少两个单种子不劣。五折均无稳定 BCE，故各运行 F1；F1 未稳定，未触发 F2。完成 60 次初筛、48 次新增稳定性训练、45 次 Focal，即 153 次唯一内层训练；种子 42 的复用不重复计数。

O13 五折均锁定 A；种子 42/43/44 的最终 epoch 分别取对应三个内层最佳 epoch 的中位数。在 S 从头重训 15 次，之后在 C 上平均三种子概率并校准，校准后不重训。五份第二锁全部完成后才进行唯一一次外层测试评估。本交付仅重算已保存预测，不重新推理、训练或搜索测试阈值。

最终阈值在 0–1、步长 0.005 的网格上，约束选择集正确人数不少于同集 R1，依次最大化 TP、正确人数、Precision、阈值。此约束只在选择集成立，不保证测试 Accuracy 不下降。早期阈值仍按原 F-latency 规则选取；连续两次越阈及 ERDE/F-latency 公式不变。

## 四个报告对象

R0 是历史正式模型与原训练流程；R1 是新协议 A 模型、早期阈值同时用于最终分类；R2 与 R1 共用概率，仅分离最终阈值；OPT 是逐外层锁定选择及回退流程。主要比较 OPT 对 R2；R2 对 R1 描述阈值变化；OPT 对 R0 混有训练流程变化，不能全部归因于 loss。

主表指标为比例（0–1），counts 为人数。全体 OOF 覆盖相同 909 人，其中正类 102 人，每人一次。汇总概率上的 AP/ROC-AUC 是跨折描述值。不存在一个统一的 OOF 阈值，实际阈值见逐折表。

'''
    sections = [intro, '## 最终分类指标\n\n' + md_table(data['main'][['object', 'users', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'recall', 'precision', 'f1']]),
                '## 补充分类指标\n\n' + md_table(data['main'][['object', 'pr_auc', 'roc_auc', 'balanced_accuracy', 'mcc', 'brier_score', 'macro_f1']]),
                '## 早期检测指标\n\n' + md_table(data['early']),
                '## 逐折锁定与校准\n\n' + md_table(data['choices']),
                '## 全部内层候选（非最终测试成绩）\n\nA 的 passes=false 不表示基线训练失败，而是其不作为新 loss 晋级；未晋级/不稳定候选全部保留。空值表示该阶段不适用或未运行，不能作零分解释。\n\n' + md_table(data['candidates']),
                '## 逐外层最终结果\n\n' + md_table(data['folds']),
                '## 单种子描述性结果\n\n这里沿用锁定的集成阈值描述各种子，不单独挑选最优种子。\n\n' + md_table(data['seeds']),
                '## 配对差值与不确定性\n\n差值方向为候选减参考，单位为原指标比例；2000 次用户级配对 bootstrap、seed=42、percentile 95% 区间，逐指标有效次数列于表内。两个主终点的支持性判断要求各自 2.5% 分位数均大于零，本次不满足。区间以已训练模型、已完成选择及 C 阈值为条件，不包含重新训练/重新选择的不确定性。\n\n' + md_table(data['intervals']),
                '''## 限制与结论边界

每折 C 仅 10 名正类，阈值与回退存在明显抽样不确定性；内层成绩经过阈值和候选筛选，不能当作独立性能。原测试结果的先前暴露不能由本次交叉验证撤销。需要新的保留数据或预先约定评估才能独立确认，不能因当前结果再在同一测试集上选择候选。

## 复现与来源

所有表由 `src/eot/optimization/delivery.py` 读取冻结产物生成，指标调用原优化评估实现，不复制指标或 loss 公式。全部来源含逐阶段完整 JSON、运行、checkpoint 和代码哈希见 `source_manifest.json`；本轮交付哈希见 `o15_acceptance.json`。可读图为 `results.png`，矢量版为 `results.svg`；表格同时提供同名 CSV。Notebook 为 `notebooks/eot_optimization.ipynb`，默认 RUN_TRAINING=False，且不包含训练调用。

在项目根目录执行（需要项目依赖及 notebook 可选依赖）：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m eot.optimization.delivery --build
python -m eot.optimization.delivery --audit
```

build 会重算点估计、生成报告/图/Notebook 并在新内核执行；audit 会重放 O14 校准及预测来源、重算所有配对区间、核验保护文件和交付哈希。均不启动训练。已有 O14 验收文件只做内容相等验证，不改写旧验收材料。
''']
    return '\n\n'.join(sections) + '\n'


def source_paths(root):
    paths = set()
    for folder in ('src/eot', 'configs', 'scripts', 'tests', 'docs/optimization',
                   'data/optimization/v1', str(BASE)):
        for p in (root / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(root / OUT):
                paths.add(p)
    return sorted(paths)


def snapshot(root):
    """Hash every optimization artifact (including checkpoints), and all code."""
    return {p.relative_to(root).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in source_paths(root)}


def build(root):
    import nbformat as nb
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    summary, _ = load_results(root)
    data = tables(root, summary)
    for name, frame in data.items():
        frame.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig', lineterminator='\n')
    report = report_text(data, summary)
    (out / 'REPORT.md').write_text(report, encoding='utf-8')
    fig = make_figure(data)
    fig.savefig(out / 'results.png', dpi=180)
    fig.savefig(out / 'results.svg')
    import matplotlib.pyplot as plt
    plt.close(fig)
    notebook = nb.v4.new_notebook()
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.cells = [nb.v4.new_markdown_cell('# EOT 优化交付与复现\n\nO15；结果模式，默认不训练。运行所有单元会从保存预测重算指标，不调用模型或重新选阈值。详细来源及复现命令见下方报告。'),
        nb.v4.new_code_cell("from pathlib import Path\nimport sys\nRUN_TRAINING = False\nif RUN_TRAINING:\n    raise RuntimeError('本 Notebook 仅支持结果复现；训练须另走冻结协议。')\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/eot/optimization/delivery.py').is_file())\nsys.path.insert(0, str(ROOT / 'src'))\nfrom eot.optimization.delivery import load_results, tables, make_figure, report_text\nfrom IPython.display import display, Markdown\nsummary, frames = load_results(ROOT)\ndata = tables(ROOT, summary)\nprint('PASS: 4 objects, 909 unique users each; point metrics replayed; no training')"),
        nb.v4.new_code_cell("display(data['main'])\ndisplay(data['early'])"),
        nb.v4.new_code_cell("from io import BytesIO\nfrom IPython.display import Image\nfig = make_figure(data)\nimage_buffer = BytesIO()\nfig.savefig(image_buffer, format='png', dpi=180)\ndisplay(Image(data=image_buffer.getvalue()))\nimport matplotlib.pyplot as plt\nplt.close(fig)"),
        nb.v4.new_code_cell("display(data['choices'])\ndisplay(data['candidates'])"),
        nb.v4.new_code_cell("display(Markdown(report_text(data, summary, protocol_href='../docs/optimization/PROTOCOL_v1.md')))")]
    for i, cell in enumerate(notebook.cells):
        cell.id = f'o15-{i:02d}'
    with tempfile.TemporaryDirectory(prefix='kernel_', dir=out) as directory:
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        manager.connection_file = str(Path(directory) / 'connection.json')
        env = os.environ.copy()
        env['IPYTHONDIR'] = str(Path(directory) / 'ipython')
        env['JUPYTER_RUNTIME_DIR'] = directory
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                                resources={'metadata': {'path': str(root / 'notebooks')}})
        try:
            client.execute(env=env)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    nb.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    image_count = sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs)
    if any(c.execution_count is None for c in cells) or image_count != 1:
        raise AssertionError('Unexecuted notebook cell or missing embedded result figure')
    nb.write(notebook, root / NOTEBOOK)
    body, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    body = body.replace('href="../docs/optimization/PROTOCOL_v1.md"',
                        'href="../../../../docs/optimization/PROTOCOL_v1.md"')
    (out / 'eot_optimization.html').write_text(body, encoding='utf-8')
    save(out / 'notebook_execution.json', {'status': 'pass', 'fresh_kernel': True,
         'interpreter': sys.executable, 'code_cells': len(cells), 'training_executed': False,
         'notebook_sha256': sha(root / NOTEBOOK), 'html_sha256': sha(out / 'eot_optimization.html')})
    print('Notebook fresh-kernel execution passed; hashing source artifacts...', flush=True)
    save(out / 'source_manifest.json', snapshot(root))
    print('Build complete. Run --audit before marking O15.', flush=True)


def audit(root):
    """Full read-only historical audit plus paired bootstrap and delivery checks."""
    import nbformat
    from .comparison import compare_frames
    out = root / OUT
    expected = read(out / 'source_manifest.json')
    equal(snapshot(root), expected, 'source_manifest')
    print(f'Source manifest verified: {len(expected)} files', flush=True)
    # Reuse O14 audit unchanged, but intercept its sole write as an equality check.
    old_argv = sys.argv[:]
    target = (root / BASE / 'final/o14_acceptance.json').resolve()
    def verify_write(path, text, *args, **kwargs):
        if path.resolve() != target or path.read_text(encoding='utf-8') != text:
            raise AssertionError(f'Historical audit attempted a non-identical write: {path}')
        return len(text)
    try:
        sys.argv = ['audit_o14.py', '--root', str(root)]
        with patch.object(Path, 'write_text', verify_write):
            module = runpy.run_path(str(root / 'tests/optimization/audit_o14.py'), run_name='o15_readonly_o14')
            if module['main']() != 0:
                raise AssertionError('O14 replay failed')
    finally:
        sys.argv = old_argv
    print('Historical O14 replay passed without writing; recomputing paired intervals...', flush=True)
    summary, frames = load_results(root)
    for name, expected_comparison in summary['comparisons']['comparisons'].items():
        ref, candidate = expected_comparison['reference_id'], expected_comparison['candidate_id']
        replay = compare_frames(frames[ref], frames[candidate], reference_id=ref, candidate_id=candidate)
        equal(replay, expected_comparison, name)
        print(f'Paired bootstrap replay passed: {name} (2000 iterations)', flush=True)
    data = tables(root, summary)
    equal((out / 'REPORT.md').read_text(encoding='utf-8'), report_text(data, summary), 'report')
    for name, frame in data.items():
        equal((out / f'{name}.csv').read_text(encoding='utf-8-sig'),
              frame.to_csv(index=False, lineterminator='\n'), name)
    if not (out / '../../../../docs/optimization/PROTOCOL_v1.md').resolve().is_file():
        raise AssertionError('Broken report protocol link')
    notebook = nbformat.read(root / NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    errors = [o for c in cells for o in c.outputs if o.output_type == 'error']
    images = sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs)
    execution = read(out / 'notebook_execution.json')
    if errors or not images or [c.execution_count for c in cells] != list(range(1, len(cells)+1)):
        raise AssertionError('Notebook execution/figure validation failed')
    equal(sha(root / NOTEBOOK), execution['notebook_sha256'], 'notebook_hash')
    equal(sha(out / 'eot_optimization.html'), execution['html_sha256'], 'html_hash')
    # Recheck source evidence after the replay; no old source/result may change.
    equal(snapshot(root), expected, 'post_audit_sources')
    artifacts = [p for p in out.iterdir() if p.is_file() and p.name != 'o15_acceptance.json']
    artifacts += [root / NOTEBOOK, root / 'src/eot/optimization/delivery.py', root / 'README.md']
    acceptance = {'task': 'O15', 'status': 'pass', 'optimization_success': False,
                  'conclusion': '未实现双指标提升；OPT 与 R2 相同，相对 R0 召回上升但准确率下降',
                  'source_files_verified': len(expected), 'o14_readonly_replay': 'pass',
                  'protected_files': read(target)['protected_files'], 'oof_users': 909,
                  'point_metrics_replay': 'pass', 'paired_bootstrap_replay': 'pass',
                  'bootstrap_iterations': 2000, 'fresh_kernel': True, 'code_cells': len(cells),
                  'error_outputs': len(errors), 'embedded_figures': images, 'training_executed': False,
                  'artifacts': {p.relative_to(root).as_posix(): sha(p) for p in sorted(artifacts)}}
    save(out / 'o15_acceptance.json', acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k != 'artifacts'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build', action='store_true')
    group.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    (build if args.build else audit)(root)


if __name__ == '__main__':
    main()
