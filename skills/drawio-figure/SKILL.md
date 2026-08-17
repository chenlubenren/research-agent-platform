---
name: drawio-figure
description: Generate structurally editable academic diagrams as native Draw.io XML from a shared DiagramRenderSpec. Use when users explicitly need node dragging, structure changes, icon replacement, or connectors that follow moved nodes.
---

# Draw.io Research Figure

Build Draw.io as the authoritative source only for explicit structural-editing routes. Academic SVG remains the default non-data backend. Generate Draw.io XML and SVG from the same renderer-neutral specification so preview and editable source do not drift.

## Workflow

1. Read `FIGURE_CONTRACT.json`, `LAYOUT_PLAN.json`, and `VISUAL_STYLE_SPEC.json`.
2. Reuse the already measured and laid-out DiagramRenderSpec; never calculate a second Draw.io-only layout.
3. Keep every structural element separate:
   - panel background and title;
   - node background, icon, and label;
   - arrow and arrow label;
   - optional decorative asset.
4. Emit valid `mxfile/mxGraphModel` XML. Use native vertex and edge cells; never flatten the full figure into a background image.
5. Emit an equivalent SVG from the same specification and render PNG/PDF from that SVG.
6. Validate entity coverage, arrow direction, label allowlist, bounds, overlap, XML parsing, fonts, and contrast.

## Style Rules

- Use restrained pastel fills, strong outlines, rounded corners, colored panel tabs, and optional diagonal hatching.
- Use semantic color accents rather than one uniform monochrome icon treatment.
- Keep labels concise and editable. Wrap them inside their node bounds.
- Use thick, high-contrast arrows with visible heads; avoid crossings.
- Use locally cached, license-recorded Iconify SVGs as replaceable image cells. ImageGen remains explicit opt-in only.

Read [references/drawio-contract.md](references/drawio-contract.md) before modifying XML generation. The bundled icon inventory is project-authored and recorded in `assets/asset-sources.json`.
