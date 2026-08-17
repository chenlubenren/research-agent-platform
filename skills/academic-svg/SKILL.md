---
name: academic-svg
description: Render renderer-neutral DiagramRenderSpec files as publication-oriented editable SVG.
---

# Academic SVG

Use only after `figure-orchestrator` has frozen a FigureContract, LayoutPlan, and DiagramRenderSpec.

- Treat the SVG as the authoritative editable source.
- Keep panels, nodes, icons, text, and edges in separate groups with stable IDs.
- Use native `text` and `tspan` elements and standard SVG markers.
- Do not introduce rasterized labels, arrows, gradients, shadows, or `foreignObject`.
- Render PNG/PDF from the exact SVG and pass the shared figure QA gate.
