# 公开代码范围

公开：`src/` 正式 Python 实现、`scripts/` CLI、主配置和 loss 候选配置、不依赖受限产物的主实验测试、网络说明和协议文档。

未公开：原始 JSON、社交媒体文本、标签名单、用户/帖子 ID 文件、处理后的数据、真实固定 fold 文件、embedding、模型权重、逐用户预测、冻结优化锁、原 Notebook 输出和其他个人文件。含真实案例 ID 的 `ode_notebook.py` 及其专用生成入口也不包含在公开副本中。

`source_manifest.json` 列出从本机原实验逐字节复制的文件及 SHA-256；`.gitattributes` 禁用换行转换以保留这些哈希。新编写的仓库 README、同步说明和辅助脚本不列入该源文件清单。`verify_public_package.py` 会检查源文件一致性及待上传文件范围。

`EXPERIMENT_PLAN.md` 的任务勾选是历史计划，不代表本公开副本的执行状态。v1/v2 的源代码可以查看，但重放历史运行依赖私下同步的冻结产物；不得删掉来源检查后声称复现。
