---
name: platform-idea-finalization
description: Consolidate a locked candidate and critique into a paper-structured, evidence-linked Chinese research idea.
---

# Final Idea Consolidation

## Goal

Produce a concise paper-style final Idea for graduate and doctoral researchers. Keep the locked candidate and critic verdict. Reuse only writing principles from `research-refine`, `research-review`, `invention-structuring`, and `kill-argument`; do not add stages.

## Rules

1. Never switch ideas. Preserve the locked candidate, criticism, uncertainty, and strongest rejection.
2. Use one dominant contribution. Explain the important problem, evidence-backed gap, hypothesis, minimal mechanism, value, and falsifiable outcome.
3. Cite only supplied Evidence IDs and pages. Separate paper facts, grounded inference, proposed design, and unconfirmed settings.
4. State whether the gap is author-explicit or inferred. Never claim first, full novelty, SOTA, universal compatibility, or preserved theory without proof. If fewer than five complete readable papers are available, or the candidate is direct author Future Work, mark `status: blocked_preliminary` and do not present it as a new-paper contribution.
5. Write clear Simplified Chinese. Explain why the mechanism may work; avoid slogans and scattered English.
6. Related work must synthesize the supplied Cross-Paper Evidence Matrix, compare at least two distinct papers when coverage allows, and admit limited coverage. Metadata-only references are not evidence.
7. The experiment section must use a few numbered Markdown subheadings such as `### 5.1`. Each block explicitly states 目的、数据集、对照组、评价指标、步骤和输出, and labels 论文已有设置、本项目沿用设置、本项目新增设置、新增设置的文献依据、尚未验证的假设、失败判据. Include ablations, failure criteria, and capacity-matched controls when supported or needed; label proposed or unconfirmed settings and never invent results or hyperparameters.
8. Use 高节点度节点, not 高阶节点, when referring to node degree.
9. Include a section named `创新性判定` with current status, Future Work overlap, closest prior work, concrete difference, coverage, unresolved evidence gaps, and whether real experiments are recommended. Include scope, strongest rejection, unresolved risks, and an abandon-or-revise condition.
10. Return one Markdown document only. Do not mention skills, prompts, agents, execution metadata, or append `白话总结`.

## Output Shape

Use this exact title and order:

1. `# 最终研究 Idea：<具体、克制、可讨论的中文标题>`
2. `## 摘要`
3. `## 1. 引言`
4. `## 2. 相关工作`
5. `## 3. 研究问题与核心假设`
6. `## 4. 方法思路`
7. `## 5. 实验方案`
8. `## 6. 预期贡献与可证伪预测`
9. `## 7. 局限、风险与不确定性`
10. `## 8. 结论与下一步`
11. `## 参考文献与证据`
