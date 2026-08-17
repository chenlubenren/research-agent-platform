# GCNII 真实完整测试结果

这是 `/wiki` 与 `/idea` 轻量化改造后的真实全流程结果，不是模拟页面或人工编造示例。

## 测试说明

- 测试日期：2026 年 8 月 3 日。
- 输入：公开论文 `2007.02133-GCNII.pdf` 和一段 GNN 研究目标。
- 流程：PDF → Wiki 论文总结 → 候选 Idea → 同候选批评 → 最终 Idea → 证据校验 → Wiki 回写。
- 本轮真实质量验收通道：`codex-cli-gpt-5.6-sol`。
- 说明：该模型通道仅用于本次真实生成质量验收，没有替换项目正式的 GLM、DeepSeek 等 OpenAI-compatible（OpenAI 兼容）接口。
- 全流程耗时：292.612 秒，约 4 分 53 秒。
- 证据校验：通过；无无效 Evidence ID（证据编号），无缺失页码，无模型回退。

## 建议查看顺序

1. `最终完整测试报告.md`：先看测试结论、修复内容和验证范围。
2. `FINAL_IDEA.md`：最终交付给研究人员的研究 Idea。
3. `IDEA_VERIFICATION.md`：批评、反证、风险和可行性核验。
4. `IDEA_CANDIDATES.md`：候选生成、比较与推荐过程。
5. `WIKI_PAPER_SUMMARY.md`：一篇论文一份 Markdown 的 Wiki 沉淀结果。
6. `RESEARCH_CONTRACT.md`：交给后续实验方案模块的研究契约。

## 文件说明

| 文件 | 内容 |
|---|---|
| `参考论文-GCNII.pdf` | 本次真实测试输入论文 |
| `WIKI_PAPER_SUMMARY.md` | 带页码证据的单篇论文 Wiki 总结 |
| `WIKI_QUERY_PACK.md` | 提供给 Idea 阶段的紧凑检索证据包 |
| `IDEA_CANDIDATES.md` | 候选 Idea、比较和推荐结果 |
| `IDEA_VERIFICATION.md` | 对锁定候选的批评核验结果 |
| `FINAL_IDEA.md` | 九个中文栏目的最终研究 Idea |
| `RESEARCH_CONTRACT.md` | 后续 `/plan` 模块的交接信息 |
| `IDEA_TRACE.json` | 三阶段模型、耗时、回退和证据校验轨迹 |
| `RUN_SUMMARY.json` | 完整运行摘要 |
| `MODEL_CALL_SUMMARY.json` | 各次真实模型调用的耗时和输入输出规模 |

## 已完成验证

```text
Idea 定向测试：21 passed
全项目回归：109 passed, 16 warnings
```

16 个警告来自已有的 FastAPI/Starlette 依赖弃用提示和 Matplotlib 中文字体缺字，与本次 Wiki/Idea 修改无关。
