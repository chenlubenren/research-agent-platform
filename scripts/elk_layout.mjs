import ELK from "elkjs/lib/elk.bundled.js";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const request = JSON.parse(input);
const elk = new ELK();
const graph = {
  id: "root",
  layoutOptions: {
    "elk.algorithm": "layered",
    "elk.direction": request.direction || "RIGHT",
    "elk.spacing.nodeNode": "36",
    "elk.layered.spacing.nodeNodeBetweenLayers": "64",
    "elk.edgeRouting": "ORTHOGONAL",
  },
  children: request.nodes.map((node) => ({
    id: node.id,
    width: node.width,
    height: node.height,
  })),
  edges: request.edges.map((edge) => ({
    id: edge.id,
    sources: [edge.source],
    targets: [edge.target],
  })),
};
const result = await elk.layout(graph);
process.stdout.write(JSON.stringify({
  nodes: (result.children || []).map(({ id, x, y }) => ({ id, x, y })),
}));
