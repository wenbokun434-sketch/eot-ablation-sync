"""Build a self-contained Chinese report from accepted frequency analyses only."""
from __future__ import annotations

import base64
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/analysis'
OUT = BASE/'posting_frequency_delivery'
GROUPS = ('low', 'medium', 'high')
STRATEGIES = ('R0', 'R1', 'R2', 'ACC_V2')
CN = dict(low='低频', medium='中频', high='高频', all='全体',
          span_ge_1_day='跨度至少1天', length_256='固定256条',
          length_256_span_ge_1_day='固定256条且跨度至少1天')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def require(value, message):
    if not value:
        raise ValueError(message)


def pct(value):
    return f'{float(value):.2%}'


def interval(row, adjusted=False):
    prefix = 'adjusted_' if adjusted else ''
    return f'[{row[prefix+"lower"]*100:+.2f}, {row[prefix+"upper"]*100:+.2f}]'


class Report:
    def __init__(self):
        self.md, self.html, self.tables = [], [], []

    def heading(self, text, level=2):
        self.md += ['#'*level+' '+text, '']
        self.html.append(f'<h{level}>{escape(text)}</h{level}>')

    def paragraph(self, text):
        self.md += [text, '']
        self.html.append(f'<p>{escape(text)}</p>')

    def table(self, headers, rows):
        rows = [[str(v) for v in row] for row in rows]
        require(all(len(row) == len(headers) for row in rows), 'Unequal table columns')
        self.tables.append(rows)
        self.md += ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
        self.md += ['| '+' | '.join(row)+' |' for row in rows] + ['']
        self.html.append('<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{escape(v)}</th>' for v in headers)+
                         '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(v)}</td>' for v in row)+'</tr>' for row in rows)+
                         '</tbody></table></div>')

    def figure(self, filename, caption):
        self.md += [f'![{caption}]({filename})', '', caption, '']
        data = base64.b64encode((OUT/filename).read_bytes()).decode('ascii')
        self.html.append(f'<figure><img alt="{escape(caption)}" src="data:image/png;base64,{data}"><figcaption>{escape(caption)}</figcaption></figure>')


class HTMLAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells, self.images, self.resources = [], [], []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'td':
            self.current = ''
        if tag == 'img':
            self.images.append(attrs.get('src', ''))
        if tag in ('script', 'link', 'iframe'):
            self.resources.append(attrs)

    def handle_data(self, data):
        if self.current is not None:
            self.current += data

    def handle_endtag(self, tag):
        if tag == 'td':
            self.cells.append(self.current)
            self.current = None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/'acceptance.json', {'status': 'incomplete'})
    inputs = {}

    def protect(path):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    for stage in ('posting_frequency', 'posting_frequency_robustness'):
        folder = BASE/stage
        acceptance = read_json(protect(folder/'acceptance.json'))
        require(acceptance['status'] == 'pass', f'{stage}: not accepted')
        for name, expected in acceptance['output_sha256'].items():
            require(sha(protect(folder/name)) == expected, f'{stage}/{name}: modified')
        manifest = read_json(protect(folder/'source_manifest.json'))
        for name, expected in manifest['input_sha256'].items():
            require(sha(protect(ROOT/name)) == expected, f'Source modified: {name}')
        stem = 'evaluate_posting_frequency' if stage == 'posting_frequency' else 'evaluate_posting_frequency_robustness'
        require(sha(protect(ROOT/f'scripts/{stem}.py')) == manifest['script_sha256'], 'Source script changed')
        test = 'test_posting_frequency' if stage == 'posting_frequency' else 'test_posting_frequency_robustness'
        require(sha(protect(ROOT/f'tests/{test}.py')) == manifest['tests_sha256'], 'Source tests changed')
    p1, p2 = BASE/'posting_frequency', BASE/'posting_frequency_robustness'
    protocol = read_json(p1/'protocol.json')
    robust_protocol = read_json(p2/'protocol.json')
    stats = pd.read_csv(p1/'group_statistics.csv').set_index('frequency_group').loc[list(GROUPS)]
    metrics = pd.read_csv(p1/'metrics.csv')
    cis = pd.read_csv(p1/'confidence_intervals.csv')
    contrasts = pd.read_csv(p2/'high_minus_low.csv')
    sensitivities = pd.read_csv(p2/'subgroup_metrics.csv')
    require(int(stats.users.sum()) == 909 and int(stats.positives.sum()) == 102, 'Cohort mismatch')
    require(protocol['boundaries_posts_per_day'] == robust_protocol['frequency_boundaries'], 'Boundaries differ')
    require(len(metrics) == 16 and len(contrasts) == 48, 'Unexpected results shape')
    # Independently verify quoted differences against the corresponding point estimates.
    for row in contrasts.itertuples():
        pair = sensitivities.loc[(sensitivities.scenario == row.scenario) & (sensitivities.strategy == row.strategy)].set_index('group')
        require(np.isclose(pair.loc['high', row.metric]-pair.loc['low', row.metric], row.difference, atol=1e-12, rtol=0), 'Difference does not match points')
    for filename, source in [('frequency_performance.png', p1/'frequency_performance.png'),
                              ('high_minus_low.png', p2/'high_minus_low.png'),
                              ('group_statistics.csv', p1/'group_statistics.csv'),
                              ('metrics.csv', p1/'metrics.csv'),
                              ('confidence_intervals.csv', p1/'confidence_intervals.csv'),
                              ('high_minus_low.csv', p2/'high_minus_low.csv'),
                              ('subgroup_metrics.csv', p2/'subgroup_metrics.csv')]:
        shutil.copyfile(source, OUT/filename)
        require(sha(source) == sha(OUT/filename), 'Copied artifact mismatch')

    r = Report()
    r.heading('不同发帖频率下的模型预测能力', 1)
    r.paragraph('补充实验汇总｜冻结模型与阈值｜用户级折外测试预测｜探索性分析')
    r.heading('给老师的简要结论')
    r.paragraph('已在不重新训练、不重新推理的条件下，完成低、中、高发帖频率用户的分层评估。909 名用户按观测频率分为三组，每组303人。四种既有方案分别报告，避免将模型版本或阈值策略混为一谈。')
    recall = metrics.loc[(metrics.strategy == 'R2') & metrics['group'].isin(GROUPS)].set_index('group').loc[list(GROUPS), 'recall']
    primary = contrasts.loc[contrasts.scenario == 'all']
    require(not primary.adjusted_excludes_zero.eq(True).any(), 'Conclusion changed: review report wording')
    r.paragraph(f'R2 的低、中、高频召回率分别为 {"、".join(map(pct, recall))}，呈上升的点估计趋势；原方案 R0 的召回率并不随频率单调上升。高频减低频的12项主要比较在多重比较校正后的区间均包含0，因此尚不能确认稳定的组间差异，也不能把结果解释为“发帖越频繁，模型必然越准确”。')
    r.heading('1. 实验问题与评估协议')
    r.paragraph('研究问题：在保持已训练模型、用户历史输入和分类决策规则不变时，模型对不同观测发帖频率用户的预测表现有何差异？本实验按频率细分原有测试用户，不重新划分训练集与测试集。每位用户只使用其原测试折产生的预测；909人的折外测试预测合并后，每人恰好出现一次。')
    r.paragraph('频率定义为（有效输入帖子数−1）÷首末帖间隔天数，即平均相邻事件间隔的倒数。帖子包含目标用户的有效主帖与评论，最多使用时间最早的256条；这是模型输入窗口内的观测事件密度，不是完整平台历史频率，也不是按活跃日计算的频率。')
    q1, q2 = protocol['boundaries_posts_per_day']
    r.paragraph(f'频率三分位边界固定为 {q1:.9f} 和 {q2:.9f} 条/天：低频≤第一边界，中频介于两边界之间且包含第二边界，高频>第二边界。边界仅由频率确定，未根据预测效果优化。后续筛选仍沿用同一分组。')
    r.table(['方案', '说明'], [
        ['R0', '历史原模型与评估流程'], ['R1', '后续模型概率，最终分类共用早期阈值的参考方案'],
        ['R2', '后续既有校准方案，保留其冻结的最终分类阈值'],
        ['ACC_V2', '准确率优先的既有阈值方案']])
    r.paragraph('R1、R2、ACC_V2 使用相同概率，决策阈值不同，因此其同组 ROC-AUC 和 PR-AUC 相同；R0 与后续流程的差异不能全部归因于阈值。本报告只分析最终分类表现，未新增早期检测评估。')
    r.heading('2. 三组用户构成')
    r.table(['组别', '用户数', '正类数', '正类比例', '输入帖数中位数', '跨度中位数/天'],
            [[CN[g], int(v.users), int(v.positives), pct(v.positive_rate), f'{v.posts_median:.0f}', f'{v.span_days_median:.2f}'] for g, v in stats.iterrows()])
    r.paragraph(f'共有 {int(stats.truncated_users.sum())} 人的历史被截断到最多256条；{int(stats.short_span_users.sum())} 人的输入窗口不足1天，均保留在主分析中。三组全预测为负类时的准确率分别为 {"、".join(pct(1-x) for x in stats.positive_rate)}，因此高频组的高准确率必须结合类别比例解读。')
    r.heading('3. 主要分组结果')
    r.table(['方案', '组别', 'Accuracy', 'Recall', 'Precision', 'F1', 'Balanced Accuracy'],
            [[v.strategy, CN[v.group], *[pct(getattr(v, k)) for k in ('accuracy','recall','precision','f1','balanced_accuracy')]]
             for v in metrics.loc[metrics['group'] != 'all'].itertuples()])
    r.paragraph('图1展示各组点估计及普通95%用户级 bootstrap 区间，每组抽样2000次。高频组仅有19名正类，一名正类预测变化对应约5.26个百分点召回率变化；不能仅凭点估计排序认定稳定优劣。')
    r.figure('frequency_performance.png', '图1　不同发帖频率组的最终分类表现与95%区间。')
    r.heading('4. 高频与低频是否有足够差异证据？')
    r.paragraph('直接计算“高频减低频”的差值，而不是仅凭两组各自区间是否重叠来判断。组内普通用户 bootstrap 20000次，高低频独立抽样，同组内四方案共用抽样用户。四方案×三个指标（Recall、Balanced Accuracy、F1）构成12项主要比较，同时报告普通95%区间和Bonferroni校正后的99.5833%区间。其覆盖率是近似的，校正不能消除事后探索性质。下表所有差值和区间单位均为百分点。')
    r.table(['方案', '指标', '高频−低频', '普通95%区间', '多重比较校正区间'],
            [[v.strategy, v.metric, f'{v.difference*100:+.2f}', interval(v._asdict()), interval(v._asdict(), True)] for v in primary.itertuples()])
    r.paragraph('12项校正区间均包含0。正确表述是“当前证据不足以确认组间差异”，而不是“已证明频率对性能没有影响”。R2 的 Balanced Accuracy 普通95%差值区间虽不含0，但校正区间包含0，不应单独挑选该结果宣布显著。')
    r.figure('high_minus_low.png', '图2　高频减低频的性能差值；粗线为普通95%区间，细线为12项比较校正后的区间。')
    r.heading('5. 敏感性检查')
    r.paragraph('补充检查输入帖子数量和短窗口集中发帖的影响：分别保留跨度至少1天、输入恰好256条，以及同时满足两条件的用户。所有筛选规则在本轮计算差值前固定，分组边界和预测不变；这些分析发生在已查看主分析结果之后，只提供探索性的普通95%区间，不作额外显著性宣称。')
    r.table(['规则', '组别', '人数', '正类人数'],
            [[CN[v.scenario], CN[v.group], v.users, v.positives] for v in sensitivities.loc[sensitivities.strategy == 'R2'].itertuples()])
    r.table(['规则', '方案', '召回率高频−低频/百分点', '普通95%区间'],
            [[CN[v.scenario], v.strategy, f'{v.difference*100:+.2f}', interval(v._asdict())]
             for v in contrasts.loc[contrasts.metric == 'recall'].itertuples()])
    r2 = contrasts.loc[(contrasts.strategy == 'R2') & (contrasts.metric == 'recall')].set_index('scenario')
    r.paragraph(f'R2 排除不足1天的窗口后，召回率差值由 {r2.loc["all", "difference"]*100:+.2f} 变为 {r2.loc["span_ge_1_day", "difference"]*100:+.2f} 个百分点，方向和幅度接近。固定256条后，差值为 {r2.loc["length_256", "difference"]*100:+.2f} 个百分点，普通95%区间为 {interval(r2.loc["length_256"])}；但低、高频组正类只有17和14人，筛选同时改变了人群构成。这是待验证线索，不能称为完全排除混杂后的频率效应。')
    r.heading('6. 结论与限制')
    r.paragraph('本实验完成了老师提出的“不重训、按发帖频率细分测试用户”的验证任务。观察到不同频率群体的分类表现存在描述性差异，R2高频组表现较好；但全体用户的多重比较校正结果尚不支持确认稳定差异。需要将频率分组分析作为对模型适用范围的补充描述，而非时间建模模块有效性的消融证据。')
    r.paragraph('主要限制包括：高频正类样本较少；组间正类比例、历史长度、观察跨度不同；固定帖数时频率与跨度存在直接数学关系；数据只反映被收集和清洗后的事件。所有区间以既有模型、阈值、折分配和分界为条件，不包含重训方差或训练集重叠带来的依赖。自然群体间比较不能说明同一用户增加发帖会提高模型预测能力。')
    r.heading('7. 可用于论文或汇报的表述')
    paper = (f'为考察模型对不同发帖频率用户的预测表现，我们在保持模型参数、输入序列及原分类阈值不变的条件下，'
             f'按输入历史的观测发帖频率对909名用户的折外测试预测进行三分位分层，各层包含303名用户。'
             f'R2方案在低、中、高频组的召回率分别为 {"、".join(map(pct, recall))}。'
             f'其高低频召回率差值为 {r2.loc["all", "difference"]*100:.2f} 个百分点，普通95% bootstrap区间为 {interval(r2.loc["all"])} 个百分点。'
             '针对四种既有方案的Recall、Balanced Accuracy和F1所作的12项主要比较，在Bonferroni校正后的区间均包含0。'
             '因此，结果提示不同频率用户群体可能具有不同的预测表现，但现有数据尚不足以确认稳定差异；该分析属于事后探索，不能支持因果解释或单独证明时间模块的增益。')
    r.paragraph(paper)
    (OUT/'paper_paragraph.txt').write_text(paper+'\n', encoding='utf-8')
    r.heading('8. 数据、复现与验收')
    r.paragraph('本报告完全读取已验收产物，未重训、未重新推理、未重新选择阈值或计算新的性能实验。报告使用的表格保存在同目录的 group_statistics.csv、metrics.csv、confidence_intervals.csv、high_minus_low.csv 和 subgroup_metrics.csv；图像与来源一致。HTML将两张图嵌入文件，可单独离线分享。')
    r.paragraph('在 eot_experiment 目录运行 python -B scripts/build_posting_frequency_delivery.py 可重建报告。source_manifest.json 记录两轮报告及原始证据的哈希；acceptance.json 记录数字一致性、离线资源和原文件保护核验。')
    css = '''body{margin:0;background:#f3f5f7;color:#203040;font:16px/1.8 "Microsoft YaHei","Noto Sans SC",sans-serif}main{max-width:1120px;margin:32px auto;padding:44px 48px;background:white;border-top:6px solid #245a81}h1{font-size:30px;line-height:1.4;margin-top:0}h2{font-size:23px;margin-top:40px;border-bottom:1px solid #dce4eb;padding-bottom:8px}p{margin:16px 0}.table-wrap{overflow:auto;margin:22px 0}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}th{background:#eaf0f5;text-align:left}th,td{padding:10px 12px;border-bottom:1px solid #dce4eb}tbody tr:nth-child(even){background:#f7f9fb}figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#526477;margin-top:10px}@media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:0;border:0;max-width:none}h2{break-after:avoid}tr,figure{break-inside:avoid}table{font-size:10px;white-space:normal}th,td{padding:5px}img{max-height:220mm;object-fit:contain}}'''
    doc = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>不同发帖频率下的模型预测能力</title><style>'+css+'</style></head><body><main>'+''.join(r.html)+'</main></body></html>'
    (OUT/'REPORT.md').write_text('\n'.join(r.md), encoding='utf-8')
    (OUT/'REPORT.html').write_text(doc, encoding='utf-8')
    audit = HTMLAudit()
    audit.feed(doc)
    expected_cells = [cell for table in r.tables for row in table for cell in row]
    require(audit.cells == expected_cells, 'HTML table content mismatch')
    require(len(audit.images) == 2 and not audit.resources, 'Unexpected external resources')
    for src, filename in zip(audit.images, ['frequency_performance.png', 'high_minus_low.png']):
        require(src.startswith('data:image/png;base64,'), 'Image not embedded')
        require(base64.b64decode(src.split(',', 1)[1]) == (OUT/filename).read_bytes(), 'Embedded image mismatch')
    require(all(sha(ROOT/p) == h for p,h in inputs.items()), 'Source changed during delivery')
    write_json(OUT/'source_manifest.json', dict(input_sha256=inputs, builder_sha256=sha(Path(__file__))))
    write_json(OUT/'acceptance.json', dict(status='pass', upstream_stages_verified=2,
               source_files_verified=len(inputs), source_hashes_unchanged=True,
               copied_artifacts_identical=True, high_low_differences_verified=48,
               html_table_cells_verified=len(expected_cells), embedded_images_verified=2,
               external_html_resources=0, new_training_runs=0, new_inference_runs=0,
               new_metric_experiments=0,
               output_sha256={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'acceptance.json'}))
    print(f'PASS: {OUT}; {len(expected_cells)} HTML table cells; {len(inputs)} protected files')


if __name__ == '__main__':
    main()
