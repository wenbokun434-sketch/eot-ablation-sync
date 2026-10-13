# EOT Ablation Experiment Synchronization

以 Emotion–ODE–Transformer（EOT）为参照，同步 eRisk 2025 Task 2 用户级抑郁检测实验的代码、配置和评估方法。

此仓库包含原实验实现、主配置、合成数据测试和同步说明。原始数据、用户 ID、标签名单、固定划分文件、embedding、checkpoint、逐用户预测及带输出的 Notebook 均不公开。代码公开不改变 eRisk 数据的使用限制；同伴须自行取得数据访问资格。

## 先统一协议

- **主实验**：`configs/eot_main.yaml`，使用验证集 PR-AUC 早停和 F-latency 阈值选择；建议作为结构消融的共同起点。
- **Loss／阈值优化 v1**：`configs/optimization/`、`src/eot/optimization/`、`docs/optimization/PROTOCOL_v1.md`。其最终训练轮数、校准集角色和双阈值规则不同于主实验。
- **阈值优化 v2**：`src/eot/optimization/threshold_v2/`、`docs/optimization/v2/PROTOCOL_v2.md`。复用 v1 模型概率，只改变最终阈值。

v1／v2 历史流程需要私下提供的冻结协议锁、预测及源文件；单独克隆代码不能重放历史结果。公开测试仅包含不依赖受限产物的主实验测试。不要把不同协议的结果作为单一模块消融比较。

同步条件见 [消融同步清单](docs/ABLATION_SYNC.md)，文件范围见 [公开文件说明](docs/PUBLICATION_SCOPE.md)。本仓库尚未新增或运行结构消融；已有模型代码保持原样。

## 安装

建议 Python 3.12；部分优化代码使用 Python 3.11 起提供的 `hashlib.file_digest`。历史主实验版本见 `docs/reference_environment.json`，它是历史记录，不是当前环境探测。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebook]"
python -m unittest discover -s tests -v
python scripts/verify_public_package.py
```

需要匹配原 CUDA 环境时，先按 `requirements-cuda.txt` 的说明安装 CUDA 版 PyTorch；该文件不是完整依赖锁。

## 从原始数据运行主实验

在仓库根目录运行，将路径替换为本人有权访问的 Task 2 数据目录：

```powershell
$env:ERISK_TASK2_ROOT = "D:/path/to/final-eriskt2-dataset-with-ground-truth"
python scripts/report_environment.py
python scripts/audit_raw_data.py
python scripts/preprocess_data.py
python scripts/create_splits.py
python scripts/build_time_stats.py
```

首次生成后，应将自己的划分与私下同步的参考划分进行成员核验，确认一致再训练。不要只比较随机种子。

原配置 `encoder.local_files_only: true`。首次使用须先缓存指定 revision 的模型与 tokenizer：

```powershell
python scripts/cache_encoder.py
python scripts/extract_embeddings.py --device cuda
python scripts/smoke_train.py
python scripts/train.py --all --device cuda
python scripts/evaluate.py --device cuda
python scripts/validate_evaluation.py
python scripts/recompute_metrics.py
```

正式主配置为 5 折 × 3 个种子（42／43／44）。每次消融使用单独工作目录或克隆目录，以免覆盖其他组的缓存、训练记录和预测。

## 私下同步固定划分

在此公开仓库根目录运行：

```powershell
python scripts/export_private_sync.py --source "D:/path/to/eot_experiment"
```

该命令只复制 5 个固定划分文件和 5 个时间统计文件到被 Git 忽略的 `private-sync/`，同时生成校验清单。这些文件可能包含用户标识或本地路径，仅向有权访问同一数据的同伴私下传递。对方核验后复制到自己的 `data/splits/` 和 `data/time_stats/`。训练派生的配置哈希可能因本地路径不同而变化；应先核验真实成员、参数和数据，再重新生成当地运行记录，不能伪造旧哈希。

本仓库没有另行指定代码开源许可证；公开可见不等于授予额外授权。
