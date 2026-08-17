# Figure Quality Gate

Hard failure conditions:

- a rendered label is absent from the FigureContract allowlist;
- a declared entity or relation is missing;
- an arrow reverses its declared source and target;
- a node leaves the canvas or materially overlaps another node;
- SVG, PDF, PNG, or the canonical source is invalid;
- a data-figure script does not reference the frozen source.

Warnings do not acquit hard failures. Subjective visual review remains advisory, and publication readiness requires human inspection of the rendered preview.
