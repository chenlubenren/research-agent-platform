---
name: figure-orchestrator
description: Route one research figure from evidence through FigureContract, visual design, an editable renderer, deterministic QA, and delivery. Use for /fig requests involving data plots, architecture or workflow diagrams, rich mechanism figures, reference-style reproduction, or a manual WPS image-to-PPT handoff.
---

# Figure Orchestrator

Produce one complete research Figure per request. A Figure may contain multiple panels, but never expand silently into an entire paper figure set.

## Workflow

1. Freeze sources before drawing. Prefer an explicit path, then the latest upload batch, then bounded workspace discovery.
2. Create `figures/FIGURE_CONTRACT.json`. Treat it as the semantic contract: claim, panels, entities, directed relations, label allowlist, evidence, language, size, and editability.
3. Route by explicit editability before diagram type:
   - numeric comparison, trend, or distribution with data -> `matplotlib`;
   - non-data default -> `academic_svg`;
   - explicit node dragging, connector-following, or structure edits -> `drawio`;
   - reference-image decomposition -> optional Edit Banana service, then non-canonical `drawio` draft.
4. Create `figures/VISUAL_STYLE_SPEC.json` and `figures/LAYOUT_PLAN.json`.
5. Measure text with an explicit font, run ELK (or record fallback), and create one shared `DiagramRenderSpec`.
6. Load exactly one renderer skill: `paper-figure`, `academic-svg`, or `drawio-figure`.
7. Render an authoritative editable source plus SVG, PDF, and PNG derivatives.
8. Run deterministic QA before delivery. Never use a subjective model score to acquit hard semantic or geometry errors.
9. Write `figures/generated/FIGURE_DELIVERY.json` with source, exports, layout engine, provenance, editability, warnings, and QA status.

## Hard Rules

- Keep FigureContract distinct from DiagramRenderSpec. The former describes truth; the latter stores measured coordinates shared by Academic SVG and Draw.io.
- Never route to a data plot only because a CSV exists.
- Never bake structural labels or arrows into an ImageGen bitmap.
- Never call ImageGen unless the user explicitly requests a custom illustration asset. It may not generate labels, arrows, or primary structure.
- Keep all labels within the contract allowlist and every arrow aligned with a declared relation.
- Mark output `needs_human_visual_review` until a person inspects the rendered preview at final size.
- Treat WPS conversion as manual, external, non-canonical, and privacy-gated.

Read [references/contracts-and-routing.md](references/contracts-and-routing.md) when extending schemas or routing. Read [references/quality-gate.md](references/quality-gate.md) when changing acceptance behavior.

## Provenance

Workflow concepts are informed by nature-figure, SciPilot, and K-Dense scientific-visualization. Do not copy or vendor upstream repositories without a separate license review.
