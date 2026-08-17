# Contracts and Routing

## Layering

- FigureContract: scientific claim, entities, directed relations, labels, evidence, and editability.
- VisualStyleSpec: palette, typography, hatching, icon density, arrows, and panels.
- LayoutPlan: reading direction, panel/node order, grouping, ports, and edge types without coordinates.
- DiagramRenderSpec: measured text, coordinates, ports, waypoints, and shared renderer primitives.
- Authoritative source: Python, Academic SVG, or Draw.io XML.

## Routing Precedence

1. Explicit format or editing behavior.
2. Reference-image decomposition request.
3. Explicit data-plot request.
4. Other non-data requests default to Academic SVG.

Mechanism and framework intent always outrank incidental tabular files in the workspace.
