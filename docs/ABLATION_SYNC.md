# 以 EOT 为基准的消融同步协议

## 必须一致的条件

| 项目 | 参照条件 |
|---|---|
| 任务 | eRisk 2025 Task 2，Depression=1，Control=0，用户级预测 |
| 用户范围 | 原主实验 909 用户，102 正类、807 负类；核验真实用户名单 |
| 消息 | `target=true` 且用户身份匹配；submission=`title [SEP] body`，comment=`body` |
| 清洗 | 原 `raw_data.py` 和 `preprocessing.py` 实现；URL、提及、删除文本、去重和稳定排序相同 |
| 截断 | 最早 256 条有效消息，每条最多 256 tokens，右 padding |
| 划分 | 私下同步参考 fold_0–4.json 的实际成员；不得按帖子划分 |
| 编码器 | j-hartmann/emotion-english-distilroberta-base，revision=0e1cd914e3d46199ed785853e12b57304e04178b，冻结，首 token 768 维隐藏表示 |
| 时间 | 天、log1p、当前梯度训练集 q99 缩放、[0,1] 裁剪；首帖与 padding=0 |
| 网络 | 非消融模块保持主配置；投影 768→256→128，ODE–GRU，门控融合，因果 Transformer，前缀 attention pooling |
| 优化 | AdamW，lr=0.0008，weight decay=0.01，batch=128，gradient clip=1，mixed precision=false |
| 主实验早停 | Validation PR-AUC，最多 50 epochs，patience=8 |
| 主实验损失 | 加权 BCE，final + 0.5×prefix；pos_weight 仅由训练集计算；前缀损失先按用户平均 |
| 前缀训练检查点 | 1,2,4,8,16,32,64,128,256；早期告警按实际有效前缀预测规则计算 |
| 种子 | 每组 42/43/44，共 15 次正式训练 |
| 集成 | 同折同用户同前缀的三种子概率算术平均，随后选阈值 |
| 主实验阈值 | 每组自己的验证集最大化 F-latency；0–1、步长 0.005；连续 2 次越阈；测试锁定后评价 |
| 指标 | 直接调用 `metrics.py` 和 `evaluation.py`；PR-AUC 使用 average precision；最终分类与早期告警分开 |
| 汇总 | 五折 OOF 每用户一次；折均值/标准差与 OOF 指标分开；不挑最佳 seed |

阈值选择方法相同，不要求各模型使用相同数值阈值。编码器、损失或网络模块若是该组明确的消融因素，可以改变；其余条件保持不变。主实验与 v1/v2 的训练和校准角色不能混合。

## 建议预先约定的组

| 名称 | 唯一改动 | 解释 |
|---|---|---|
| full | 原始真实间隔 EOT | 共同参照 |
| fixed_gap | 非首帖有效间隔改为训练集拟合的固定值，其他结构保留 | 检验真实不规则间隔 |
| zero_gap | 时间间隔全部为 0，GRU/融合/Transformer 保留 | 关闭 ODE 时间传播 |
| without_state_branch | 移除 ODE–GRU 分支并提前约定替代连接 | 检验整个状态分支，不能解释为仅去掉时间输入 |

上表是待实施方案，仓库不声称已有这些训练或结果。固定间隔必须只从当前训练用户计算；首帖/padding 保持 0。每组从头训练，推理时清零属于敏感性检查。

## 每组提交结果

保存配置、代码 commit、fold 成员校验、seed、环境、最佳 epoch、checkpoint 来源、最终概率、各有效前缀概率、验证集阈值和告警位置。逐用户结果私下共享。

比较 Accuracy、Recall、F1、AP、Balanced Accuracy、ROC-AUC 及早期检测指标；报告相对 full 的差值。对相同 OOF 用户进行 2000 次配对 bootstrap，95% percentile 区间，同次抽样共享用户索引。当前主实验已查看测试结果，后续同划分消融应注明为事后补充分析。
