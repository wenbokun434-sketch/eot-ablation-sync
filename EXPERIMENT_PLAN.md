# EOT 完整实验执行计划

## 完成标准

实验只有在以下条件全部满足后才视为完成：

- 从原始 eRisk Task 2 数据独立生成用户序列。
- 数据划分、时间归一化、类别权重和阈值选择均无测试集泄漏。
- Emotion DistilRoBERTa、MLP、ODE-GRU、融合层、因果 Transformer 和分类头均有可运行实现。
- 五折、三个随机种子的训练与评估全部成功完成。
- 保存 out-of-fold 预测、前缀预测、指标、checkpoint 和可视化。
- 最终 notebook 可以从头到尾执行，并复现论文中的主要表格和图形。

## 统一实验约束

- 数据单位是用户，不允许按帖子随机划分。
- 目标消息以 `target=true` 判定。
- 主输入为每个用户时间上最早的 256 条有效消息。
- 文本编码器重新运行，不加载同伴基线的 embedding。
- 每折时间归一化参数只从该折训练用户计算。
- 模型选择只使用验证集，测试集只用于锁定后的最终评估。
- 所有正式运行必须记录 fold、seed、配置哈希、模型 revision 和代码版本。

## Step 1：环境与配置锁定

- [ ] 创建独立 Python 环境并记录 Python、PyTorch、Transformers 和 torchdiffeq 版本。
- [ ] 检查 CPU、GPU、显存和可用磁盘空间。
- [ ] 检查 `configs/eot_main.yaml` 中不存在个人桌面绝对路径。
- [ ] 固定 Emotion DistilRoBERTa 的模型 revision。
- [ ] 设置随机种子、确定性选项和日志目录。
- [ ] 生成环境报告 `outputs/reports/environment.json`。

验收：重新启动环境后能够读取配置并导入全部依赖。

## Step 2：原始数据审计

- [ ] 读取标签文件并验证用户 ID 唯一。
- [ ] 检查 JSON 文件数和标签用户数一致。
- [ ] 统计 Depression 与 Control 用户数。
- [ ] 检查 `submission`、`comments`、`target`、`created_utc` 和消息 ID 字段。
- [ ] 记录空文本、删除文本、无效时间戳和异常 `user_id`。
- [ ] 生成 `outputs/reports/raw_data_audit.json`。

验收：所有异常均有明确处理规则，且没有静默丢弃用户。

## Step 3：独立数据预处理

- [ ] 以 `target=true` 提取目标用户消息。
- [ ] submission 构造为 `title [SEP] body`，comment 使用 `body`。
- [ ] HTML 解码并规范 URL、用户提及、空格和换行。
- [ ] 删除空文本、`[removed]` 和 `[deleted]`。
- [ ] 按用户和 `created_utc` 稳定排序。
- [ ] 计算原始时间间隔和用户级数据统计。
- [ ] 保留每名用户最早的 256 条有效消息，短序列右侧 padding。
- [ ] 保存 `data/processed/user_sequences.parquet` 和 `data/processed/post_index.parquet`。
- [ ] 保存 `outputs/reports/preprocessing_report.json`。

验收：时间单调、标签完整、用户数不因序列长度而减少、帖子 ID 可追溯回原始 JSON。

## Step 4：用户级数据划分

- [ ] 使用五折用户级分层交叉验证。
- [ ] 在每个外层训练折中独立划出验证用户。
- [ ] 固定 fold 文件，三个随机种子共享相同 fold 划分。
- [ ] 检查 train、validation 和 test 用户交集为空。
- [ ] 保存 `data/splits/fold_0.json` 至 `fold_4.json`。
- [ ] 输出每折的用户数和类别数。

验收：任何用户的全部帖子只出现在一个集合中。

## Step 5：独立生成帖子情绪向量

- [ ] 从 `post_index.parquet` 读取本实验帖子文本。
- [ ] 使用固定 revision 的 Emotion DistilRoBERTa 编码。
- [ ] 设置最大 token 长度为 256，并采用首标记隐藏表示。
- [ ] 以 float16 保存 embedding，以 float32 送入训练模型。
- [ ] 保存 embedding 行号与帖子 ID 的映射。
- [ ] 保存模型名、revision、token 长度、pooling 和文本哈希。
- [ ] 实现缓存校验，配置不一致时拒绝加载旧缓存。

验收：随机抽样帖子可从 embedding 索引准确反查，且重复运行生成相同形状和顺序。

## Step 6：构建序列张量与时间变量

- [ ] 构建帖子向量张量、位置张量、原始时间间隔和 padding mask。
- [ ] 每折只使用训练用户计算 `log1p` 时间间隔的 99% 分位数。
- [ ] 将时间尺度变换到非负区间零到一。
- [ ] 验证集和测试集复用对应训练折参数。
- [ ] 保存每折 `data/time_stats/fold_x.json`。
- [ ] 检查首条消息时间间隔为零。
- [ ] 检查 padding 位置不会更新 ODE 或 GRU 状态。

验收：所有张量形状一致，时间变量有限、非负且无 NaN。

## Step 7：实现 EOT 模型

- [ ] 实现 `PostProjection`。
- [ ] 实现无 Dropout 的 `ODEFunc`。
- [ ] 使用 `torchdiffeq.odeint` 实现批量连续时间传播。
- [ ] 实现 `GRUCell` 当前消息更新。
- [ ] 实现内容与时间状态的门控融合。
- [ ] 实现位置编码、causal mask 和 padding mask。
- [ ] 实现两层 Transformer Encoder。
- [ ] 实现支持多个前缀的 Attention Pooling。
- [ ] 实现 MLP Classification Head。
- [ ] 输出最终 logit、前缀 logit 和可选 attention 权重。

验收：前向传播形状正确；修改未来帖子不会影响更早位置的隐藏状态和预测。

## Step 8：单元测试与小规模冒烟训练

- [ ] 测试时间间隔为零时 ODE 状态不自然漂移。
- [ ] 测试 padding 前后预测一致。
- [ ] 测试 causal mask 不存在未来信息泄漏。
- [ ] 测试只有一条有效消息的边界情况。
- [ ] 测试极大时间间隔经过截断后保持稳定。
- [ ] 使用少量用户执行两个 epoch 的冒烟训练。
- [ ] 检查 loss 下降、梯度有限、checkpoint 可以重新加载。

验收：全部自动化测试通过，冒烟训练无 NaN、无 shape 错误、无内存异常。

## Step 9：正式训练

- [ ] 对五个 fold 分别运行 seed 42、43、44，共 15 次训练。
- [ ] 使用类别加权 BCE、最终预测损失和前缀预测损失。
- [ ] 使用 AdamW、weight decay 和梯度裁剪。
- [ ] 依据验证集 PR-AUC 执行 early stopping。
- [ ] 每次运行只保留最佳 checkpoint 和精简训练历史。
- [ ] 将配置、fold、seed、最佳 epoch 和验证指标写入运行清单。

验收：15 次运行均具有最佳 checkpoint、训练日志和对应验证预测。

## Step 10：锁定阈值并评估测试折

- [ ] 在每次运行的验证集选择分类阈值。
- [ ] 在验证集确定连续越阈次数等 alert 规则。
- [ ] 锁定阈值后生成测试折最终预测。
- [ ] 保存用户级最终概率和各前缀风险概率。
- [ ] 计算 PR-AUC、ROC-AUC、Precision、Recall、F1、Macro-F1、MCC 和 Brier Score。
- [ ] 计算 ERDE5、ERDE50、median latency、speed 和延迟加权 F 值。
- [ ] 合并全部测试折得到 out-of-fold 预测。
- [ ] 计算折间均值、标准差和用户级 bootstrap 置信区间。

验收：指标可以从保存的预测文件重新计算，测试标签未参与阈值或模型选择。

## Step 11：生成正式可视化和结果摘要

- [ ] 绘制训练损失与验证 PR-AUC 曲线。
- [ ] 绘制 ROC 曲线和 Precision–Recall 曲线。
- [ ] 绘制混淆矩阵。
- [ ] 绘制不同前缀长度下的 PR-AUC、F1 和 Recall。
- [ ] 绘制正确阳性用户的检测延迟分布。
- [ ] 绘制代表性用户的风险概率时间轨迹。
- [ ] 绘制代表性用户的 attention 权重与发帖时间间隔。
- [ ] 输出主指标表、早期检测表和运行稳定性表。
- [ ] 检查所有图包含标题、坐标名称、图例和单位，并适合论文排版。

验收：所有图表都由正式预测和指标文件生成，不手工填写实验数值。

## Step 12：整合实验代码和结果到单一 notebook

- [ ] 创建 `notebooks/eot_complete_experiment.ipynb`。
- [ ] 按顺序整合实验背景、配置、数据审计、预处理、模型结构、训练方法、评估和可视化。
- [ ] notebook 调用 `src/` 中的正式实现，不复制出第二套模型代码。
- [ ] 默认读取已完成的 checkpoint 和结果，避免每次打开都重新训练 15 个模型。
- [ ] 提供显式的 `RUN_TRAINING` 开关以支持完整复现。
- [ ] 展示模型结构、参数量、数据统计、主要指标表和全部正式图形。
- [ ] 清除调试输出、临时单元格、硬编码个人路径和无关分析。
- [ ] 重启 kernel 并从头运行全部非训练单元格。
- [ ] 验证 notebook 在缺少缓存或 checkpoint 时给出清晰提示。
- [ ] 将执行成功且输出完整的 notebook 作为本实验最终交付物。
