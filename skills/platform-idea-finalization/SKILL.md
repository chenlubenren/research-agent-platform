---
name: platform-idea-finalization
description: Consolidate an idea candidate and critique into a professional, readable, evidence-linked Chinese research idea.
---

# Final Idea Consolidation

## Goal

Produce a professional, readable final Idea for graduate and doctoral researchers. It must be understandable to a new group member, rigorous enough for discussion, and ready for `/plan` without becoming a full experiment plan.

Reuse only these writing principles, without adding stages: `research-refine` for a fixed problem and minimal mechanism; `research-review` for claim-evidence alignment; `invention-structuring` for problem-solution-value logic; `kill-argument` for the strongest rejection.

## Rules

1. Preserve the locked candidate, critic verdict, criticism, and uncertainty; never switch ideas.
2. Explain: important problem -> evidence-backed gap -> hypothesis -> minimal mechanism -> value -> falsifiable outcome.
3. Keep one dominant contribution, not a feature list.
4. Cite only supplied IDs. Separate paper facts, grounded inference, and unverified assumptions.
5. Say whether the gap is author-explicit or inferred. Never claim first, full novelty, SOTA, universal compatibility, or preserved theory without proof.
6. Write clear Simplified Chinese. Start with one sentence and a short plain-language explanation. Explain why the mechanism may work.
7. Do not invent formulas, symbols, datasets, metrics, hyperparameters, results, or frameworks absent from evidence. Leave implementation and full experiments to `/plan`.
8. Include scope, strongest rejection, unresolved risks, and an abandon-or-revise condition.
9. Return one complete Markdown document only; no repeated draft or outer code fence.
10. After the title, use only the nine required Chinese sections. Do not mention skills, prompts, agents, execution metadata, or append a separate `白话总结`.

## Output Shape

Use this exact title and order:

1. `# 最终研究 Idea：<具体、克制、可讨论的中文标题>`
2. `## 一句话研究 Idea`
3. `## 研究背景与核心问题`
4. `## 现有研究不足与可切入空白`
5. `## 核心假设与方法思路`
6. `## 预期创新与学术价值`
7. `## 可证伪预测`
8. `## 证据依据`
9. `## 适用边界、风险与不确定性`
10. `## 交给实验方案模块的下一步`
