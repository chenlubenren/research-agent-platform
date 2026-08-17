---
name: platform-idea-generation
description: Generate evidence-grounded research idea candidates for the local research-agent platform.
---

# Idea Candidate Generation

## Goal

Generate three distinct, testable research ideas from the user's objective, Research Wiki query pack, and available evidence.

## Rules

1. Use only papers and Evidence IDs present in the supplied context. Read the Cross-Paper Evidence Matrix and Evidence Coverage before proposing candidates.
2. Separate paper-author statements from model inference.
3. Each candidate must contain a problem, mechanism, novelty thesis, expected contribution, feasibility, and main risk.
4. Prefer gaps explicitly stated in Discussion, Limitations, or Future Work sections.
5. Do not invent papers, identifiers, datasets, metrics, or experimental results.
6. Do not write a complete experiment plan; provide only falsifiable predictions and planning requirements.
7. Recommend one candidate using evidence strength, novelty, feasibility, and cost. A direct extension of author Future Work cannot be recommended as a new-paper contribution.
8. If evidence is insufficient, state the uncertainty instead of filling the gap with assumptions.
9. When the user's objective is Chinese, keep required section headings unchanged but write all explanatory prose in concise Simplified Chinese.
10. Label each candidate gap as one of: author-explicit future work, author-reported limitation or failure, evidence-backed extension, or model inference.
11. Generate one direct replication, one cross-paper combination, and one mechanism-or-question candidate. For every candidate include candidate type, supporting Paper IDs, Evidence IDs, author Future Work overlap, closest prior work, concrete difference, added research question, expected experiment, maximum innovation risk, and `novelty_status`.
12. Mark `novelty_status: direct_extension` when the candidate restates author Future Work, an already implemented method, a simple method-name combination, a dataset-only change, or a hyperparameter-only change. Do not recommend direct_extension as a top-journal candidate.
13. If fewer than five complete readable papers are present in Evidence Coverage, state that evidence is insufficient for a top-journal candidate and do not claim novelty.
14. Keep the whole artifact concise; do not repeat the Wiki summary or commit to exact datasets, depth, metrics, or hyperparameters before `/plan`.
15. Do not invent parameter symbols, formulas, numeric results, or implementation frameworks; use descriptive language when the evidence does not expose exact details.

## Output

Return the exact sections requested by the workflow and file-ready Markdown only.
