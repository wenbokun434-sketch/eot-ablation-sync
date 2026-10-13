"""Build and execute the final presentation notebook without duplicating core logic."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

from .evaluation import _file_sha256
from .reporting import read_json
from .reproducibility import write_json


def make_notebook():
    import nbformat as nbf
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md(r"""
# Emotion–ODE–Transformer：完整实验

**用户历史帖子 → 情绪特征 → MLP → 连续时间 ODE–GRU → 内容/时间融合 → 因果 Transformer → 用户级分类与早期告警**

本 notebook 整合本实验独立生成的数据审计、方法、15 次正式训练、五折 OOF 测试结果、图表和正式源码。核心实现唯一保存在 `src/eot/`；本文件负责解释、调度和展示。

默认 `RUN_TRAINING = False`。选择 **Restart Kernel and Run All** 可重新读取并核验已保存结果，图表会嵌入输出。首次浏览不要求 CUDA 或挂载原始数据，但需要本项目的正式产物和 Python 依赖。

目录：① 配置与运行开关；② 原始审计及独立预处理；③ 用户划分与情绪特征；④ 连续时间模型与训练；⑤ 测试评估与结果；⑥ 十组正式图表；⑦ 结论边界与源码。

复现环境：先按 README 安装 CUDA PyTorch（若需训练），然后在项目根目录执行 `python -m pip install -e ".[notebook]"`。缺少文件时，检查单元会给出相应恢复命令。
""")
    code('''
RUN_TRAINING = False
DEVICE = "cuda"  # 仅重建/训练时使用；默认结果模式不进行 GPU 推理。

from pathlib import Path
import sys
sys.dont_write_bytecode = True
candidates = [Path.cwd(), Path.cwd() / "eot_experiment", *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / "configs/eot_main.yaml").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("请从 eot_experiment/ 或 notebooks/ 启动 notebook；需要 configs/eot_main.yaml 和 src/eot/。")
sys.path.insert(0, str(ROOT / "src"))
print("运行模式：" + ("重建流程（已完成且配置一致的训练会跳过）" if RUN_TRAINING else "读取并核验正式结果"))
''')
    md("""
## 1. 配置与环境

结果模式使用原始审计报告中保存的数据目录字符串重建配置哈希，不打开原始帖子，也不修改系统环境。训练模式要求显式设置 `ERISK_TASK2_ROOT`。

如需从零复现：将完整项目复制到独立目录，保留 `configs/`、`src/` 和 `scripts/`，设置原始数据环境变量并把开关改为 `True`，顺序执行全部单元。已存在的配置不匹配产物会拒绝加载；已完成的同配置训练自动跳过。冻结情绪模型当前配置为本地缓存读取，须预先准备配置指定 revision 的模型缓存。
""")
    code('''
import pandas as pd
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 120)
import torch
from IPython.display import display, Markdown, HTML, Image
from eot.notebook_support import (
    notebook_config, preflight_results, configuration_table, split_table,
    training_table, source_browser,
)
from eot.reporting import read_json, validate_visualizations, verify_point_metrics
from eot.reproducibility import collect_environment, set_global_seed, write_json

config = notebook_config(ROOT, run_training=RUN_TRAINING)
OUT = config.path("outputs_dir")
if RUN_TRAINING:
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请按 requirements-cuda.txt 配置后重启内核，或显式将 DEVICE 改为 cpu。")
    set_global_seed(int(config.section("cross_validation")["fold_seed"]))
    write_json(collect_environment(ROOT, config.config_hash), OUT / "reports/environment.json")
else:
    display(preflight_results(config))

print("配置哈希：", config.config_hash)
environment = read_json(OUT / "reports/environment.json")
display(pd.DataFrame(environment["packages"].items(), columns=["正式训练依赖", "版本"]))
display(pd.DataFrame(environment["cuda"]["devices"])[["name", "total_memory_bytes"]])
display(configuration_table(config))
''')
    md("""
## 2. 原始数据审计与独立预处理

数据来自 eRisk Task 2 原始用户 JSON，筛选 `target=true` 且用户 ID 与所属文件一致的帖子；submission 合并 title/body，comment 使用 body。规范 URL、用户提及与空白后按时间稳定排序，只保留每位用户最早的 256 条有效帖子。

审计发现的异常必须展示：原始审计状态可能为 `review_required`，预处理报告记录对应剔除原因及用户完整性；不会把原始数据异常隐藏成“全部无异常”。
""")
    code('''
from eot.audit import audit_raw_dataset
from eot.preprocessing import preprocess_dataset, write_preprocessed

if RUN_TRAINING:
    audit = audit_raw_dataset(config)
    write_json(audit, OUT / "reports/raw_data_audit.json")
    # 用户不匹配记录由下一阶段显式剔除，其余结构性异常先停止。
    unresolved = {k: v for k, v in audit["critical_issues"].items() if v and k != "target_user_mismatches"}
    if unresolved:
        raise ValueError(f"需要先解决原始数据审计异常：{unresolved}")
    posts, users, preprocessing = preprocess_dataset(config)
    write_preprocessed(posts, users, config.path("processed_dir"))
    write_json(preprocessing, OUT / "reports/preprocessing_report.json")

audit = read_json(OUT / "reports/raw_data_audit.json")
preprocessing = read_json(OUT / "reports/preprocessing_report.json")
print("原始审计状态：", audit["status"], "；预处理状态：", preprocessing["status"])
display(pd.DataFrame(audit["critical_issues"].items(), columns=["审计项", "数量"]))
display(pd.DataFrame(preprocessing["dropped"].items(), columns=["剔除原因", "数量"]))
display(pd.DataFrame(preprocessing["counts"].items(), columns=["独立数据统计", "值"]))
display(pd.DataFrame(preprocessing["integrity"].items(), columns=["完整性检查", "通过"]))
assert preprocessing["status"] == "pass" and all(preprocessing["integrity"].values())
assert audit["critical_issues"]["target_user_mismatches"] == preprocessing["dropped"].get("user_id_mismatch", 0)
''')
    md("""
## 3. 五折用户划分与情绪特征

用户为不可拆分样本单位。外层采用分层五折，每个外层训练集再划出 12.5% 作验证；三个随机种子共享划分。仅使用当前训练折拟合时间尺度、类别权重；验证集用于最佳 epoch 和阈值选择。

固定 revision 的 `j-hartmann/emotion-english-distilroberta-base` 对每帖最多 256 tokens 编码，取最后一层首标记隐藏表示（768维），以 float16 缓存。它不是七类情绪概率；训练时转换为 float32，文本编码器冻结。
""")
    code('''
from eot.splitting import create_user_splits, write_user_splits
from eot.embeddings import encode_posts

if RUN_TRAINING:
    folds, split_report = create_user_splits(config)
    write_user_splits(folds, split_report, config)
    encode_posts(config, DEVICE)

display(split_table(config))
embedding_meta = read_json(config.path("embeddings_dir") / "post_embeddings.meta.json")
display(pd.DataFrame(embedding_meta["spec"].items(), columns=["Embedding 缓存约束", "值"]))
print("缓存形状：", embedding_meta["shape"], "；状态：", embedding_meta["status"])
''')
    md(r"""
## 4. 时间变量、张量与模型结构

第一帖间隔置零，其余以天为单位计算真实相邻间隔。训练折的非首帖 `log1p(Δt)` 的 99% 分位数为尺度 $q$，输入时间为 $\tau_i=\min(\log(1+\Delta t_i),q)/q$。验证/测试复用训练尺度。

张量：帖子特征 `[B,N,768]`，投影 `[B,N,128]`，时间与有效 mask `[B,N]`。实现按下式演化：

$$e_i=\mathrm{MLP}(h_i),\qquad \frac{dz(s)}{ds}=\tau_i f_\theta(z(s)),\quad s\in[0,1]$$
$$\tilde z_i=\mathrm{ODESolve}(z_{i-1},\tau_i),\qquad z_i=\mathrm{GRUCell}(e_i,\tilde z_i)$$

初始状态为可学习向量。ODE 使用无 dropout 的自治向量场，RK4 固定步长 0.25；GRU 吸收当前内容。可学习 gate 融合投影内容与时间状态，再经过 LayerNorm、位置编码和两层因果 Transformer。

Transformer 同时应用 causal/padding mask。前缀 Attention Pooling 对每个有效前缀输出一个用户表示，MLP 产生 prefix logit；最终 logit 取最后有效前缀。未来帖不能改变早期预测。
""")
    code('''
from eot.time_features import create_all_time_stats
from eot.dataset import SequenceStore
from eot.model import EOTModel, count_parameters

if RUN_TRAINING:
    create_all_time_stats(config)

time_stats = [read_json(config.path("time_stats_dir") / f"fold_{fold}.json")
              for fold in range(int(config.section("cross_validation")["n_splits"]))]
display(pd.DataFrame(time_stats)[["fold", "fit_split", "fit_user_count", "log_gap_quantile", "transformed_values_clipped"]])
store = SequenceStore(config)  # 验证 embedding 的内容指纹与本实验帖子索引一致。
print("有效用户数：", len(store.users), "；embedding 维度：", store.embedding_dim)
model = EOTModel.from_config(config)  # 仅检查结构/参数量，此对象不用于正式评估。
display(pd.DataFrame(count_parameters(model).items(), columns=["序列模型参数量（不含冻结编码器）", "数量"]))
print(model)
del model, store
''')
    md(r"""
## 5. 冒烟训练与正式训练

总损失为 $\mathcal L=\mathcal L_{final}+0.5\mathcal L_{prefix}$；均采用训练折类别权重的 BCEWithLogitsLoss。前缀监督位置为 `{1,2,4,8,16,32,64,128,256}`，先在用户内平均再跨用户平均。

AdamW、batch=128、learning rate=8e-4、weight decay=0.01、gradient clip=1。最多50轮，验证 AP 连续8轮未改善早停，保留最佳 checkpoint；正式运行共5折×3种子（42/43/44）。正式训练使用 CUDA；情绪缓存的最初提取设备单独记录。

默认读取训练日志。`RUN_TRAINING=True` 会执行两轮小样本冒烟训练，并按 fold/seed 顺序调用正式训练入口；现有已完成运行按配置哈希校验后自动跳过。
""")
    code('''
from eot.training import run_smoke_training, train_formal_run

if RUN_TRAINING:
    smoke = run_smoke_training(config, device_name=DEVICE)
    if smoke["status"] != "pass":
        raise RuntimeError("冒烟训练未通过，停止正式训练。")
    for fold in range(int(config.section("cross_validation")["n_splits"])):
        for seed in config.section("training")["seeds"]:
            train_formal_run(config, fold, int(seed), DEVICE)

smoke = read_json(OUT / "reports/smoke_training.json")
display(pd.DataFrame(smoke["history"]))
print("Checkpoint 重载一致：", smoke["checkpoint_reload_matches"])
runs = training_table(config)
display(runs)
print(f"正式运行数：{len(runs)}；累计训练用时：{runs.elapsed_seconds.sum()/60:.2f} 分钟")
''')
    md(r"""
## 6. 验证阈值锁定与五折 OOF 评估

同折三种子在概率层面取均值。仅在对应验证集以步长0.005扫描 `[0,1]`，最大化 F-latency，再将阈值冻结用于外层测试。每个种子的独立阈值另用于稳定性报告。外层 OOF 每个用户只出现一次。

最终分类：最后有效前缀概率越阈。早期告警：连续两个前缀概率越阈时触发，一旦告警便保留决策。二者的 F1 和混淆矩阵分开报告。

ERDE 按误报/漏报/真阳性延迟计费；F-latency 为早期 F1 乘以 speed，speed 为1减去真阳性延迟惩罚的中位数，参数 p=0.0078。具体公式只在 `src/eot/metrics.py` 实现。来源：[eRisk 2017](https://tec.citius.usc.es/ir/pdf/eRisk2017.pdf)、[eRisk 2021](https://www.ceur-ws.org/Vol-2936/paper-72.pdf)。
""")
    code('''
from eot.evaluation import run_formal_evaluation, validate_evaluation_artifacts
from eot.visualization import build_visualizations

if RUN_TRAINING:
    run_formal_evaluation(config, device_name=DEVICE)
    build_visualizations(config, device_name=DEVICE)

evaluation_check = validate_evaluation_artifacts(config)
figure_check = validate_visualizations(config)
print("评估验收：", evaluation_check["status"], "；图表验收：", figure_check["status"])
print("OOF 用户：", evaluation_check["oof_unique_users"], "；有效前缀：", evaluation_check["oof_prefix_rows"])
thresholds = read_json(OUT / "metrics/fold_thresholds.json")
display(pd.DataFrame(thresholds["folds"])[["fold", "threshold", "consecutive_positive_predictions", "source", "validation_user_count", "test_user_count"]])
final = pd.read_parquet(OUT / "predictions/oof_final_predictions.parquet")
metrics = read_json(OUT / "metrics/oof_metrics.json")
verify_point_metrics(final, metrics)
''')
    md("""
## 7. 正式结果表

`oof_estimate` 是合并用户后的点估计，`fold_mean`/`fold_sd` 为五个测试折等权均值/样本标准差。PR-AUC 使用 average precision (AP)。95% CI 为2,000次用户级 percentile bootstrap；它以当前训练结果与阈值为条件，不包含重新训练的不确定性。

ERDE 以0–1数值报告，转成百分比需乘100。延迟单位为观察帖子数。
""")
    code('''
display(Markdown("### 完整序列分类"))
display(pd.read_csv(OUT / "tables/classification.csv").round(4))
display(Markdown("### 早期检测"))
display(pd.read_csv(OUT / "tables/early_detection.csv").round(4))
display(Markdown("### 各折测试结果"))
fold_results = pd.read_csv(OUT / "tables/fold_results.csv")
display(fold_results.loc[fold_results.split == "test", ["fold", "threshold", "pr_auc", "roc_auc", "f1", "recall", "f_latency", "erde_50"]].round(4))
display(Markdown("### 15次运行与稳定性"))
display(pd.read_csv(OUT / "tables/stability_runs.csv")[["fold", "seed", "threshold", "pr_auc", "f1", "f_latency"]].round(4))
display(pd.read_csv(OUT / "tables/stability_summary.csv").round(4))
''')
    md("""
## 8. 正式可视化

以下10组图直接读取已保存的300 dpi PNG并嵌入 notebook 输出；同名矢量 PDF/SVG 可在 `outputs/figures/` 获取。英文图表便于论文排版，中文图注说明评估口径。

前缀曲线始终保留全部 OOF 用户，使用第 `min(k,L)` 帖概率；短序列结束后沿用最后概率，另列出仍有至少k帖的用户数。案例按早期 TP/FP/FN/TN 组内长度中位数最近规则固定选择；pooling 权重在告警前缀或最终前缀提取，不包含未来或 padding。
""")
    code('''
figure_manifest = read_json(OUT / "reports/visualization_manifest.json")
display(pd.read_csv(OUT / "tables/representative_cases.csv")[["case", "fold", "length", "threshold", "alert_position", "explanation_position", "group_users"]])
prefix_table = pd.read_csv(OUT / "tables/prefix_performance.csv")
display(prefix_table.loc[prefix_table.scope == "oof", ["checkpoint", "users", "users_with_at_least_k_posts", "pr_auc", "f1", "recall"]].round(4))
''')
    for index in range(10):
        code(f'''
figure = figure_manifest["figures"][{index}]
display(Markdown("### " + figure["name"] + "\\n\\n" + figure["caption"]))
display(Image(filename=str(OUT / "figures" / figure["files"][0]), width=1100))
''')
    md("""
## 9. 主要发现与适用边界

这里报告固定实验的测试结果，不用观察到的测试表现继续选择阈值。真阳性中位告警位置只描述检出者，必须同时报告漏报数。当前离线协议只处理每用户最早256条清洗后的目标帖子，不能直接作为官方完整流式榜单成绩。

类别不平衡和每折较少的验证正类用户会使阈值及召回率波动；应并列引用 OOF、折间结果和用户置信区间。Attention Pooling 权重是模型内部的描述性权重，不等同于因果贡献或临床证据。
""")
    code('''
c = metrics["point_estimate"]["classification"]
e = metrics["point_estimate"]["early_detection"]
positive = int(final.label.sum())
detected = int(((final.label == 1) & (final.alert_prediction == 1)).sum())
display(Markdown(
    f"- 完整序列：AP **{c['pr_auc']:.4f}**，ROC-AUC **{c['roc_auc']:.4f}**，F1 **{c['f1']:.4f}**。\\n"
    f"- Precision **{c['precision']:.4f}**、Recall **{c['recall']:.4f}**，当前策略召回仍有限。\\n"
    f"- 早期检出 **{detected}/{positive}**，漏报 **{positive-detected}**；检出者中位第 **{e['median_latency']:g}** 帖告警。\\n"
    f"- F-latency **{e['f_latency']:.4f}**，ERDE5 **{e['erde_5']:.4f}**，ERDE50 **{e['erde_50']:.4f}**。"
))
''')
    analysis_path = Path(__file__).resolve().parents[2] / "docs/RESULTS_ANALYSIS.md"
    analysis = analysis_path.read_text(encoding="utf-8")
    analysis = analysis.replace("\n## ", "\n### ")
    md(analysis.replace(
        "# 实验结果分析：时间间隔是否能提高准确率",
        "## 9.1 时间间隔是否能提高准确率", 1,
    ))
    md("""
## 10. 正式源码与复现入口

展开下列模块可查看本次交付所调用的完整核心代码。源码由文件读取后嵌入为只读展示；notebook 中没有第二套模型、指标或绘图实现。

常用入口（项目根目录）：`python scripts/train.py --all --device cuda`、`python scripts/evaluate.py --device cuda`、`python scripts/visualize_results.py --device cuda`。完整重建按本 notebook 的 `RUN_TRAINING` 分支顺序执行。

交付验证：`python scripts/build_notebook.py --execute` 在新内核执行默认结果模式，并验证所有代码单元无错误、全部十组图已嵌入。
""")
    code('''
display(HTML(source_browser(ROOT)))
print("全部结果单元执行完成；T10 指标已重算，T11 图表和数据已核验。")
''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (EOT)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "eot": {"task": "T12", "default_run_training": False, "core_implementation": "src/eot/"},
    })
    for index, cell in enumerate(notebook.cells):
        cell.id = f"eot-{index:03d}"
    nbf.validate(notebook)
    return notebook


def build_notebook(root: Path, execute: bool = False) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from nbconvert import HTMLExporter

    notebook = make_notebook()
    path = root / "notebooks/eot_complete_experiment.ipynb"
    path.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    if execute:
        # Force the current interpreter in memory; no global kernel installation.
        (root / "outputs/tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="notebook_kernel_", dir=root / "outputs/tmp") as directory:
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            manager.connection_file = str(Path(directory) / "connection.json")
            env = os.environ.copy()
            env.pop("ERISK_TASK2_ROOT", None)  # Exercise the portable result-mode fallback.
            env["IPYTHONDIR"] = str(Path(directory) / "ipython")
            env["JUPYTER_RUNTIME_DIR"] = directory
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                                    resources={"metadata": {"path": str(root / "notebooks")}})
            # The client does not own a supplied manager: always shut it down explicitly.
            try:
                client.execute(env=env)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [output for cell in code_cells for output in cell.outputs if output.output_type == "error"]
        images = sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
        if errors or any(cell.execution_count is None for cell in code_cells) or images != 10:
            raise AssertionError("Notebook execution is incomplete or missing one of the ten embedded figures")
    elif path.is_file():
        raise FileExistsError("已存在最终 notebook；请使用 --execute 生成并验证新版本，避免覆盖为未执行文件。")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    if not execute:
        return {"status": "built_unexecuted", "path": str(path)}
    html_path = root / "outputs/reports/eot_complete_experiment.html"
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(notebook)
    html_path.write_text(body, encoding="utf-8")
    report = {
        "status": "pass", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": read_json(root / "outputs/reports/raw_data_audit.json")["config_hash"],
        "default_run_training": False, "fresh_kernel_execution": True,
        "tested_without_erisk_environment_variable": True,
        "code_cells_executed": len(code_cells), "error_outputs": len(errors), "embedded_png_figures": images,
        "elapsed_seconds": time.monotonic() - started,
        "notebook": str(path.relative_to(root)), "notebook_sha256": _file_sha256(path),
        "html_preview": str(html_path.relative_to(root)), "html_sha256": _file_sha256(html_path),
        "retraining_executed": False,
        "code": {str(p.relative_to(root)): _file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("notebook_support.py"))},
        "analysis_source": "docs/RESULTS_ANALYSIS.md",
        "analysis_source_sha256": _file_sha256(root / "docs/RESULTS_ANALYSIS.md"),
    }
    write_json(report, root / "outputs/reports/notebook_validation.json")
    return report
