# Draw.io Output Contract

The document root must be `mxfile > diagram > mxGraphModel > root` with cells `0` and `1`.

- Structural nodes are vertex cells.
- Labels are independent text vertex cells.
- Icons are independent vertex cells.
- Connections are edge cells with explicit `source` and `target` IDs.
- Geometry uses absolute canvas coordinates; edge geometry is relative.
- The SVG preview and Draw.io XML must be derived from the same RenderSpec.

Do not place a full-figure bitmap behind transparent editable labels. That is a false-editability anti-pattern.
