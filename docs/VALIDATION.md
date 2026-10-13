# 公开代码副本验证

验证日期：2026-10-13。

- 在独立公开副本中执行 `python -m unittest discover -s tests -v`，33 项测试全部通过。
- `python -m compileall -q src scripts tests` 通过。
- `scripts/verify_public_package.py` 核验 106 个原样复制文件的 SHA-256，并检查 Git 文件范围和常见凭据格式。
- 对待上传文本与原实验实际用户名单进行本地匹配，未发现真实用户 ID；用户名单未写入公开仓库。
- `private-sync/`、`data/`、`outputs/` 的 Git 忽略规则已验证。
- 私下同步导出工具成功生成 5 个划分文件、5 个时间统计文件及校验清单；这些文件不上传。

验证不包含重新训练 15 个模型、下载编码器、执行完整受限数据流水线或重放 v1/v2 历史优化。结构消融尚未实施或运行。
