---
name: platform-idea-generation
description: Generate evidence-grounded research idea candidates for the local research-agent platform.
---

# Idea Candidate Generation

## Goal

Generate three distinct, testable research ideas from the user's objective, Research Wiki query pack, and available evidence.

## Rules

1. Use only papers and Evidence IDs present in the supplied context.
2. Separate paper-author statements from model inference.
3. Each candidate must contain a problem, mechanism, novelty thesis, expected contribution, feasibility, and main risk.
4. Prefer gaps explicitly stated in Discussion, Limitations, or Future Work sections.
5. Do not invent papers, identifiers, datasets, metrics, or experimental results.
6. Do not write a complete experiment plan; provide only falsifiable predictions and planning requirements.
7. Recommend one candidate using evidence strength, novelty, feasibility, and cost.
8. If evidence is insufficient, state the uncertainty instead of filling the gap with assumptions.
9. When the user's objective is Chinese, keep required section headings unchanged but write all explanatory prose in concise Simplified Chinese.
10. Label each candidate gap as one of: author-explicit future work, author-reported limitation or failure, evidence-backed extension, or model inference.
11. If explicit author future work exists, include at least one candidate directly derived from it and prefer stronger evidence over a more imaginative mechanism.
12. Keep the whole artifact concise; do not repeat the Wiki summary or commit to exact datasets, depth, metrics, or hyperparameters before `/plan`.
13. Do not invent parameter symbols, formulas, numeric results, or implementation frameworks; use descriptive language when the evidence does not expose exact details.

## Output

Return the exact sections requested by the workflow and file-ready Markdown only.
