---
name: platform-research-wiki
description: Maintain the platform's single session Wiki with original PDFs, per-paper summaries, retrieval packs, and minimal relationship edges.
---

# Research Wiki

## Goal

Maintain one practical Research Wiki for human reading and Idea retrieval.

## Rules

1. Preserve the original PDF and create one Markdown summary per paper.
2. Bind factual summaries to page-linked Evidence IDs whenever available.
3. Clearly separate author statements, extracted evidence, and model inference.
4. Update an existing paper identified by file content instead of creating duplicates.
5. Keep `index.md` human-readable and `query_pack.md` focused on the current topic.
6. Record only minimal, evidence-bearing relationships in `relations.jsonl`.
7. Preserve failed or rejected Idea information when it can prevent repetition.
8. Do not claim that a graph database, external Wiki helper, MCP tool, or independent reviewer was used unless the runtime actually used it.
9. Prioritize evidence from Discussion, Limitations, Conclusion, Future Work, ablations, and explicit failure observations instead of reading only the first pages.

## Output

Return the exact Wiki sections requested by the workflow and file-ready Markdown only.
