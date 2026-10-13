# EOT 实验代码设计约束

## 1. 分层边界

- `configs/` 只保存可复现参数，不包含业务实现。
- `src/eot/` 是唯一正式实现；脚本和 notebook 只能调用这里的代码，不得复制模型或指标公式。
- `scripts/` 只负责解析命令行参数和调用 `src/eot/`。
- `tests/` 使用合成数据或临时目录，不读取同伴基线产物。
- `data/` 只保存本实验从原始 eRisk JSON 独立生成的派生数据。
- `outputs/` 只保存可由配置和正式代码重建的报告、模型、预测、指标与图形。

## 2. 数据与防泄漏约束

- 用户是不可拆分的最小样本单位；同一用户不得跨 train、validation、test。
- 原始帖子必须以 `target=true` 和文件所属用户共同校验。
- 时间缩放分位数、类别权重、early stopping 与阈值只能使用当前折训练/验证数据。
- 测试标签不得参与超参数选择、归一化、阈值选择或模型选择。
- 每条派生帖子保留 `user_id`、`post_id`、`kind`、`created_utc` 与源文件名，支持溯源。

## 3. 模型接口约束

- `EOTModel.forward` 输入为帖子 embedding、归一化时间间隔和有效位置 mask。
- ODE 只负责相邻帖子之间的连续状态传播，GRUCell 负责吸收当前帖子。
- padding 位置不能推进 ODE、更新 GRU 或参与 attention/pooling。
- Transformer 必须同时使用 causal mask 和 padding mask。
- 模型返回最终 logit、逐位置前缀 logit及可选 pooling 权重。

## 4. 产物与可复现约束

- 所有 JSON 使用 UTF-8、排序键和稳定缩进。
- embedding 缓存必须同时保存模型 revision、token 长度、pooling、帖子顺序和文本哈希。
- checkpoint 必须包含模型参数、配置哈希、fold、seed、epoch 和验证指标。
- 所有随机过程由统一种子入口控制。
- 正式图表只能读取保存的预测和指标，不允许手工录入数值。

## 5. 评估约束

- 每个 fold/seed 的阈值只能由对应验证集前缀预测选择，测试预测不得反向影响阈值。
- 主结果先在同一 fold 内对三个 seed 的概率取算术平均，再用该 fold 的验证集集成概率锁定阈值。
- 完整序列分类使用最后一个有效前缀的概率；早期告警使用连续两次前缀概率越阈，并分别保存 `final_prediction` 与 `alert_prediction`。
- 外层五折测试结果合并为 OOF 时，每个用户必须且只能出现一次。
- ERDE 和 F-latency 公式只在 `src/eot/metrics.py` 实现；指标必须可由保存的 OOF 预测独立重算。
- 置信区间以用户为抽样单位，不得按帖子或前缀行 bootstrap。

## 6. 正式可视化约束

- `reporting.py` 负责表格、前缀取样、确定性案例选择和产物校验；`visualization.py` 负责科研图表与报告生成。
- 绘图输入必须通过 T10 配置哈希、预测校验和与指标重算；绘图不得修改模型或选择新阈值。
- 前缀主曲线使用固定 OOF 用户集合，取第 `min(k, L)` 帖概率，显式报告仍有至少 k 帖的用户数。
- 合并 OOF、折均值、折标准差、用户 bootstrap 区间必须使用各自准确的名称，不互相替代。
- 案例按早期 TP/FP/FN/TN 分组，以离组内长度中位数最近为规则，用户 ID 升序破同分。
- Pooling 权重在告警前缀或最终前缀计算；检查权重归一化、无未来/padding，以及重载概率与冻结 OOF 一致。
- 图表导出 300 dpi PNG 及矢量 PDF/SVG；保存来源、输出和代码哈希，输出独立验收报告。

## 7. Notebook 约束

- notebook 是最后一个任务，不是第二套实现。
- 所有核心单元调用 `src/eot/`；仅保留解释、配置、调度、结果表和可视化代码。
- 默认 `RUN_TRAINING = False`，打开时读取已存在的正式产物。
- 缺少原始数据、embedding、checkpoint 或预测时必须给出明确的下一条命令。
- `notebook_document.py` 负责文档编排、全新内核执行和静态导出；`notebook_support.py` 负责结果模式预检、配置恢复、摘要及只读源码展示。
- 结果模式不打开原始帖子、不训练或重新推理，使用已保存审计记录恢复配置哈希；训练模式必须显式提供并校验原始数据目录。
- 最终 notebook 必须保存全部执行输出和 10 组正式图，附带 HTML 及包含配置、源码、交付文件哈希的验收报告；重新执行仍以完整项目为依赖。
