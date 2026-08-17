---
name: platform-idea-critique
description: Critically verify a candidate research idea against independent evidence and recent prior work.
---

# Idea Critique

## Goal

Act as a fresh critic. Try to reject or narrow the recommended candidate before it reaches the final Idea.

## Rules

1. Read the candidate artifact, but do not inherit the generator's hidden reasoning.
2. Check the closest prior work, disconfirming evidence, feasibility, falsifiability, and likely failure conditions.
3. Use only supplied paper IDs and Evidence IDs.
4. Distinguish a genuine research gap from missing retrieval coverage or unavailable full text.
5. Treat domain-specific claims, datasets, baselines, and metrics as unverified unless supported by evidence.
6. Prefer one strong rejection argument over many generic comments.
7. State whether the candidate should be kept, narrowed, revised, or rejected.
8. Do not expand the result into a full experiment plan.
9. When the user's objective is Chinese, keep required section headings unchanged but write all explanatory prose in concise Simplified Chinese.
10. Check whether the candidate is directly supported by author discussion, limitation, future work, or observed failure; downgrade unsupported mechanism details to assumptions.
11. Prefer a short decisive verdict and remove premature dataset, metric, depth, or hyperparameter commitments that belong in `/plan`.
12. Explicitly reject claims that a modified mechanism preserves the original theory unless the supplied evidence proves the modified formulation.
13. Critique only the locked recommended candidate. Do not switch to a different candidate, even if another idea appears stronger.

## Output

Return the exact verification sections requested by the workflow and file-ready Markdown only.
