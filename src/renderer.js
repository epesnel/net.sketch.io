const COLORS = {
  Input: { fill: '#e8eaf6', stroke: '#5c6bc0', text: '#283593' },
  Output: { fill: '#e8eaf6', stroke: '#5c6bc0', text: '#283593' },
  Conv2D: { fill: '#e3f2fd', stroke: '#42a5f5', text: '#1565c0' },
  Conv1D: { fill: '#e3f2fd', stroke: '#42a5f5', text: '#1565c0' },
  Conv3D: { fill: '#e3f2fd', stroke: '#42a5f5', text: '#1565c0' },
  Conv2DTranspose: { fill: '#e3f2fd', stroke: '#42a5f5', text: '#1565c0' },
  Conv1DTranspose: { fill: '#e3f2fd', stroke: '#42a5f5', text: '#1565c0' },
  DepthwiseConv2D: { fill: '#fff3e0', stroke: '#ffa726', text: '#e65100' },
  SeparableConv2D: { fill: '#fff3e0', stroke: '#ffa726', text: '#e65100' },
  Dense: { fill: '#e8f5e9', stroke: '#66bb6a', text: '#2e7d32' },
  MaxPooling2D: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  MaxPooling1D: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  AveragePooling2D: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  AveragePooling1D: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  GlobalAveragePooling2D: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  GlobalMaxPooling2D: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  UpSampling2D: { fill: '#e8f5e9', stroke: '#66bb6a', text: '#2e7d32' },
  UpSampling1D: { fill: '#e8f5e9', stroke: '#66bb6a', text: '#2e7d32' },
  BatchNormalization: { fill: '#f5f5f5', stroke: '#bdbdbd', text: '#616161' },
  LayerNormalization: { fill: '#f5f5f5', stroke: '#bdbdbd', text: '#616161' },
  GroupNormalization: { fill: '#f5f5f5', stroke: '#bdbdbd', text: '#616161' },
  Dropout: { fill: '#f5f5f5', stroke: '#bdbdbd', text: '#616161' },
  SpatialDropout2D: { fill: '#f5f5f5', stroke: '#bdbdbd', text: '#616161' },
  Activation: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  ReLU: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  LeakyReLU: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  PReLU: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  ELU: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  Softmax: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  Flatten: { fill: '#eceff1', stroke: '#78909c', text: '#37474f' },
  Reshape: { fill: '#eceff1', stroke: '#78909c', text: '#37474f' },
  Concatenate: { fill: '#fff8e1', stroke: '#ffb300', text: '#ff6f00' },
  Add: { fill: '#e8f5e9', stroke: '#66bb6a', text: '#2e7d32' },
  Subtract: { fill: '#fce4ec', stroke: '#ef5350', text: '#c62828' },
  Multiply: { fill: '#f3e5f5', stroke: '#ab47bc', text: '#6a1b9a' },
  Average: { fill: '#e0f2f1', stroke: '#26a69a', text: '#00695c' },
  Resize: { fill: '#e8f5e9', stroke: '#66bb6a', text: '#2e7d32' },
  SpaceToDepth: { fill: '#ede7f6', stroke: '#7e57c2', text: '#4527a0' },
  DepthToSpace: { fill: '#ede7f6', stroke: '#7e57c2', text: '#4527a0' },
  Pad: { fill: '#eceff1', stroke: '#78909c', text: '#37474f' },
  Reduce: { fill: '#eceff1', stroke: '#78909c', text: '#37474f' },
  Slice: { fill: '#eceff1', stroke: '#78909c', text: '#37474f' },
  Embedding: { fill: '#f3e5f5', stroke: '#ab47bc', text: '#6a1b9a' },
  LSTM: { fill: '#ede7f6', stroke: '#7e57c2', text: '#4527a0' },
  GRU: { fill: '#ede7f6', stroke: '#7e57c2', text: '#4527a0' },
  BinOp: { fill: '#eceff1', stroke: '#78909c', text: '#37474f' },
};
const DEFAULT_COLOR = { fill: '#eceff1', stroke: '#78909c', text: '#37474f' };

const IDENTITY_TYPES = new Set([
  'BatchNormalization', 'LayerNormalization', 'GroupNormalization',
  'Dropout', 'SpatialDropout2D', 'Activation', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'Softmax',
]);

export class GraphRenderer {
  constructor(container) {
    this.container = container;
    this.svg = null;
    this.positions = {};
    this.nodeElements = {};
  }

  mergeActivations(graph) {
    const merged = new Set();
    const actMap = new Map();
    for (const node of graph.nodes) {
      if (!IDENTITY_TYPES.has(node.type)) continue;
      const inEdges = graph.edges.filter(e => e.to === node.id);
      if (inEdges.length !== 1) continue;
      const parentId = inEdges[0].from;
      const parent = graph.nodes.find(n => n.id === parentId);
      if (!parent || IDENTITY_TYPES.has(parent.type)) continue;
      const parentOut = graph.edges.filter(e => e.from === parentId);
      if (parentOut.length !== 1) continue;
      merged.add(node.id);
      actMap.set(parentId, node.type);
    }
    const newEdges = [];
    for (const e of graph.edges) {
      if (merged.has(e.to)) continue;
      if (merged.has(e.from)) {
        const parentId = graph.edges.find(pe => pe.to === e.from)?.from;
        if (parentId) newEdges.push({ from: parentId, to: e.to });
      } else {
        newEdges.push(e);
      }
    }
    const newNodes = graph.nodes.filter(n => !merged.has(n.id));
    for (const n of newNodes) {
      if (actMap.has(n.id)) n.activation = actMap.get(n.id);
    }
    return { nodes: newNodes, edges: newEdges };
  }

  render(graph) {
    if (!graph || graph.nodes.length === 0) {
      this.container.innerHTML = '<div class="empty-state">Paste TF/Keras code on the left and click Analyze</div>';
      return;
    }
    this.container.innerHTML = '';
    graph = this.mergeActivations(graph);
    this._lastGraph = graph;

    const layout = this.computeLayout(graph);
    const { positions, width, height } = layout;
    this.positions = positions;

    const pad = 60;
    const totalH = height;
    const svgW = width + pad * 2;
    const svgH = totalH + pad * 2;

    this._edgeOffsets = null;
    this.svg = this.createSVG(svgW, svgH);
    this.addDefs();

    const mainG = this.el('g', { transform: `translate(${pad}, ${pad})` });
    this.svg.appendChild(mainG);
    for (const edge of graph.edges) this.drawEdgeDagre(mainG, edge, graph, positions);
    for (const node of graph.nodes) this.drawNode(mainG, node, positions[node.id]);


    this.container.appendChild(this.svg);
    this.addZoomPan();
    this.addInteractivity(graph);
    this.addStats(graph);
  }

  computeLayout(graph) {
    const nodeIdx = new Map(graph.nodes.map((n, i) => [n.id, i]));
    let maxRes = 1, maxCh = 1;
    for (const n of graph.nodes) {
      const s = n.outputShape;
      if (s && s.length >= 4) {
        if (typeof s[1] === 'number' && s[1] > maxRes) maxRes = s[1];
        if (typeof s[3] === 'number' && s[3] > maxCh) maxCh = s[3];
      }
    }

    // Main path for edge styling
    const adj = new Map();
    for (const e of graph.edges) {
      if (!adj.has(e.from)) adj.set(e.from, []);
      adj.get(e.from).push(e.to);
    }
    const mainPath = new Set();
    const mainEdges = new Set();
    const findLongest = (nodeId, visited) => {
      if (visited.has(nodeId)) return [];
      visited.add(nodeId);
      const children = adj.get(nodeId) || [];
      if (children.length === 0) return [nodeId];
      let best = [];
      for (const c of children) {
        const p = findLongest(c, new Set(visited));
        if (p.length > best.length) best = p;
      }
      return [nodeId, ...best];
    };
    if (graph.nodes.length > 0) {
      const path = findLongest(graph.nodes[0].id, new Set());
      for (const id of path) mainPath.add(id);
      for (let i = 0; i < path.length - 1; i++) mainEdges.add(path[i] + ':' + path[i + 1]);
    }
    this._mainPath = mainPath;
    this._mainEdges = mainEdges;

    // Use dagre for layout
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'LR', nodesep: 8, ranksep: 14, marginx: 20, marginy: 20, align: 'UL' });
    g.setDefaultEdgeLabel(function() { return {}; });

    for (const node of graph.nodes) {
      const d = this.nodeDims(node, maxRes, maxCh);
      g.setNode(node.id, { width: d.w, height: d.h, node: node });
    }
    for (const e of graph.edges) {
      g.setEdge(e.from, e.to);
    }

    dagre.layout(g);

    const positions = {};
    g.nodes().forEach(id => {
      const n = g.node(id);
      if (n) {
        positions[id] = { x: n.x - n.width / 2, y: n.y - n.height / 2, w: n.width, h: n.height, node: n.node };
      }
    });

    let maxX = 0, maxY = 0;
    for (const p of Object.values(positions)) {
      maxX = Math.max(maxX, p.x + p.w);
      maxY = Math.max(maxY, p.y + p.h);
    }

    this._nodeIdx = nodeIdx;
    this._maxRes = maxRes;
    this._branchNodes = [];
    this._dagreGraph = g;
    return { positions, width: maxX + 40, height: maxY + 40 };
  }

  nodeDims(node, maxRes, maxCh) {
    const isIdentity = IDENTITY_TYPES.has(node.type);
    const isActivation = ['ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'Activation', 'Softmax'].includes(node.type);
    if (isActivation) return { w: 24, h: 28 };
    if (isIdentity) return { w: 14, h: 44 };
    if (['Add', 'Subtract', 'Multiply', 'Average'].includes(node.type)) return { w: 22, h: 22 };
    if (node.type === 'Concatenate') return { w: 22, h: 22 };
    if (node.type === 'Slice') return { w: 12, h: 44 };
    if (node.type === 'Flatten') return { w: 22, h: 52 };
    if (node.type === 'Dense') return { w: 22, h: 60 };

    const s = node.outputShape;
    const res = (s && s.length >= 4 && typeof s[1] === 'number') ? s[1] : maxRes;
    const ch = (s && s.length >= 4 && typeof s[3] === 'number') ? s[3] : 16;
    const resFrac = res / Math.max(1, maxRes);
    const hShape = Math.round(40 + resFrac * 110);

    const isPool = ['MaxPooling2D','MaxPooling1D','AveragePooling2D','AveragePooling1D','GlobalAveragePooling2D','GlobalMaxPooling2D','UpSampling2D','UpSampling1D','Resize'].includes(node.type);
    if (isPool) {
      const inRes = node.inputShapes?.[0]?.[1];
      const inFrac = (typeof inRes === 'number' ? inRes : res) / Math.max(1, maxRes);
      const hPool = Math.round(40 + Math.max(resFrac, inFrac) * 110);
      return { w: 22, h: Math.max(40, hPool) };
    }

    const chFrac = ch / Math.max(1, maxCh);
    const w = Math.round(Math.max(14, 12 + Math.sqrt(chFrac) * 20));
    return { w, h: hShape };
  }

  drawNode(parent, node, pos) {
    if (!pos) return;
    const { x, y, w, h } = pos;
    const c = COLORS[node.type] || DEFAULT_COLOR;
    const isIdentity = IDENTITY_TYPES.has(node.type);
    const isOp = ['Add', 'Subtract', 'Multiply', 'Average', 'Concatenate'].includes(node.type);
    const font = "'IBM Plex Sans', 'Inter', 'Helvetica Neue', Arial, sans-serif";

    const g = this.el('g', { class: 'graph-node', 'data-id': node.id, transform: `translate(${x}, ${y})` });

    const isDown = ['MaxPooling2D','MaxPooling1D','AveragePooling2D','AveragePooling1D','GlobalAveragePooling2D','GlobalMaxPooling2D'].includes(node.type);
    const isUp = ['UpSampling2D','UpSampling1D'].includes(node.type);
    const isResize = node.type === 'Resize';
    if (isDown || isUp || isResize) {
      const inS = node.inputShapes?.[0], outS = node.outputShape;
      const inRes = (inS && inS.length >= 4 && typeof inS[1] === 'number') ? inS[1] : null;
      const outRes = (outS && outS.length >= 4 && typeof outS[1] === 'number') ? outS[1] : null;
      const mr = this._maxRes || 1;
      const hIn = inRes !== null ? Math.max(20, 40 + (inRes / mr) * 110) : h;
      const hOut = outRes !== null ? Math.max(20, 40 + (outRes / mr) * 110) : h;
      const padL = (h - Math.min(hIn, h)) / 2, padR = (h - Math.min(hOut, h)) / 2;
      const tL = padL, tR = padR;
      const bL = h - padL, bR = h - padR;
      g.appendChild(this.el('polygon', { points: `0,${tL} ${w},${tR} ${w},${bR} 0,${bL}`, fill: c.fill, stroke: c.stroke, 'stroke-width': 1.2, 'stroke-linejoin': 'round' }));
      const down = (inRes || 0) > (outRes || 0);
      const cx = w / 2, aLen = h * 0.3;
      const aTop = h / 2 - aLen / 2, aBot = h / 2 + aLen / 2;
      if (down) {
        g.appendChild(this.el('line', { x1: cx, y1: aTop, x2: cx, y2: aBot, stroke: c.text, 'stroke-width': 1.2, 'stroke-linecap': 'round' }));
        g.appendChild(this.el('path', { d: `M${cx - 3},${aBot - 4} L${cx},${aBot} L${cx + 3},${aBot - 4}`, fill: 'none', stroke: c.text, 'stroke-width': 1.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
      } else {
        g.appendChild(this.el('line', { x1: cx, y1: aBot, x2: cx, y2: aTop, stroke: c.text, 'stroke-width': 1.2, 'stroke-linecap': 'round' }));
        g.appendChild(this.el('path', { d: `M${cx - 3},${aTop + 4} L${cx},${aTop} L${cx + 3},${aTop + 4}`, fill: 'none', stroke: c.text, 'stroke-width': 1.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
      }
    } else if (['ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'Activation', 'Softmax'].includes(node.type)) {
      const actName = node.type === 'Activation' ? 'Act' : node.type;
      const actLbl = this.el('text', { x: w / 2, y: -4, 'text-anchor': 'middle', fill: c.text, 'font-size': '6px', 'font-weight': '600', 'font-family': font });
      actLbl.textContent = actName;
      g.appendChild(actLbl);
      g.appendChild(this.el('rect', { x: 0, y: 0, width: w, height: h, rx: 4, ry: 4, fill: 'none', stroke: c.stroke, 'stroke-width': 1 }));
      const px = 5, py = 4, iw = w - px * 2, ih = h - py * 2;
      const ox = px + iw * 0.4, oy = py + ih * 0.6;
      g.appendChild(this.el('line', { x1: px, y1: oy, x2: px + iw, y2: oy, stroke: '#aaa', 'stroke-width': 0.6 }));
      g.appendChild(this.el('line', { x1: ox, y1: py + ih, x2: ox, y2: py, stroke: '#aaa', 'stroke-width': 0.6 }));
      g.appendChild(this.el('path', { d: `M${px + iw - 1},${oy - 1.5} L${px + iw + 0.5},${oy} L${px + iw - 1},${oy + 1.5}`, fill: 'none', stroke: '#aaa', 'stroke-width': 0.6 }));
      g.appendChild(this.el('path', { d: `M${ox - 1.5},${py + 1} L${ox},${py - 0.5} L${ox + 1.5},${py + 1}`, fill: 'none', stroke: '#aaa', 'stroke-width': 0.6 }));
      let shapePath;
      if (node.type === 'ReLU' || node.type === 'Activation') {
        shapePath = `M${px + 1},${oy} L${ox},${oy} L${px + iw - 1},${py + 2}`;
      } else if (node.type === 'LeakyReLU' || node.type === 'PReLU') {
        shapePath = `M${px + 1},${oy - ih * 0.1} L${ox},${oy} L${px + iw - 1},${py + 2}`;
      } else if (node.type === 'ELU') {
        shapePath = `M${px + 1},${oy - 1} Q${ox - 2},${oy} ${ox},${oy} L${px + iw - 1},${py + 2}`;
      } else {
        shapePath = `M${px + 1},${py + ih - 2} C${px + iw * 0.25},${py + ih - 2} ${px + iw * 0.75},${py + 2} ${px + iw - 1},${py + 2}`;
      }
      g.appendChild(this.el('path', { d: shapePath, fill: 'none', stroke: c.stroke, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
    } else if (isOp) {
      const r = Math.min(w, h) / 2;
      const cx = w / 2, cy = h / 2;
      g.appendChild(this.el('circle', { cx, cy, r, fill: c.fill, stroke: c.stroke, 'stroke-width': 1.5 }));
      const sym = node.type === 'Add' ? '+' : node.type === 'Subtract' ? '−' : node.type === 'Multiply' ? '×' : node.type === 'Concatenate' ? 'C' : 'ø';
      const symEl = this.el('text', {
        x: cx, y: cy, 'text-anchor': 'middle', 'dominant-baseline': 'central',
        fill: c.text, 'font-size': node.type === 'Concatenate' ? '9px' : '12px', 'font-weight': '600', 'font-family': font,
      });
      symEl.textContent = sym;
      g.appendChild(symEl);
    } else if (node.type === 'Dense') {
      const topLbl = this.el('text', { x: w / 2, y: -4, 'text-anchor': 'middle', fill: c.text, 'font-size': '6.5px', 'font-weight': '600', 'font-family': font });
      topLbl.textContent = 'Dense';
      g.appendChild(topLbl);
      g.appendChild(this.el('rect', {
        x: 0, y: 0, width: w, height: h, rx: 5, ry: 5,
        fill: 'none', stroke: c.stroke, 'stroke-width': 1.2,
      }));
      const botLbl = this.el('text', { x: w / 2, y: h + 9, 'text-anchor': 'middle', fill: c.text, 'font-size': '5.5px', 'font-family': font, opacity: 0.7 });
      botLbl.textContent = this.shortLabel(node);
      g.appendChild(botLbl);
      const inC = node.inputShapes?.[0]?.[node.inputShapes[0]?.length - 1] || 16;
      const outC = node.config?.units || 10;
      const maxDim = Math.max(inC, outC, 1);
      const nL = Math.max(2, Math.min(7, Math.round(2 + 5 * inC / maxDim)));
      const nR = Math.max(2, Math.min(7, Math.round(2 + 5 * outC / maxDim)));
      const nr = 1.8;
      const pad = 5, xL = w * 0.3, xR = w * 0.7;
      const gapL = (h - pad * 2) / (nL + 1), gapR = (h - pad * 2) / (nR + 1);
      for (let i = 0; i < nL; i++) {
        const yl = pad + gapL * (i + 1);
        for (let j = 0; j < nR; j++) {
          const yr2 = pad + gapR * (j + 1);
          g.appendChild(this.el('line', { x1: xL, y1: yl, x2: xR, y2: yr2, stroke: '#888', 'stroke-width': 0.6, opacity: 0.6 }));
        }
      }
      for (let i = 0; i < nL; i++) {
        const yl = pad + gapL * (i + 1);
        g.appendChild(this.el('circle', { cx: xL, cy: yl, r: nr, fill: '#a8d8a8', stroke: '#5a9a5a', 'stroke-width': 0.8 }));
      }
      for (let j = 0; j < nR; j++) {
        const yr2 = pad + gapR * (j + 1);
        g.appendChild(this.el('circle', { cx: xR, cy: yr2, r: nr, fill: '#a8c8e8', stroke: '#5a8ab8', 'stroke-width': 0.8 }));
      }
    } else if (node.type === 'Conv2D' || node.type === 'Conv1D' || node.type === 'Conv3D' || node.type === 'Conv2DTranspose') {
      const topLbl = this.el('text', { x: w / 2, y: -4, 'text-anchor': 'middle', fill: c.text, 'font-size': '6.5px', 'font-weight': '600', 'font-family': font });
      topLbl.textContent = this.shortTypeName(node.type);
      g.appendChild(topLbl);
      g.appendChild(this.el('rect', { x: 0, y: 0, width: w, height: h, rx: 5, ry: 5, fill: c.fill, stroke: c.stroke, 'stroke-width': 1.2 }));
      const cw = w * 0.38, ch2 = h * 0.5, d = w * 0.25;
      const bx = (w - cw - d) / 2 + 1, by = (h - ch2 - d) / 2 + d;
      g.appendChild(this.el('path', { d: `M${bx},${by} L${bx+cw},${by} L${bx+cw},${by+ch2} L${bx},${by+ch2} Z`, fill: c.stroke, opacity: 0.18, stroke: c.stroke, 'stroke-width': 0.7 }));
      g.appendChild(this.el('path', { d: `M${bx},${by} L${bx+d},${by-d} L${bx+cw+d},${by-d} L${bx+cw},${by} Z`, fill: c.stroke, opacity: 0.08, stroke: c.stroke, 'stroke-width': 0.4 }));
      g.appendChild(this.el('path', { d: `M${bx+cw},${by} L${bx+cw+d},${by-d} L${bx+cw+d},${by+ch2-d} L${bx+cw},${by+ch2} Z`, fill: c.stroke, opacity: 0.12, stroke: c.stroke, 'stroke-width': 0.4 }));
      const k = this.numCfg(node.config?.kernel_size, 3);
      const kSz = Math.min(cw, ch2) * Math.min(k / 6, 0.7);
      const kx = bx + 1, ky = by + ch2 - kSz - 1;
      g.appendChild(this.el('rect', { x: kx, y: ky, width: kSz, height: kSz, fill: '#ef5350', opacity: 0.5, stroke: '#c62828', 'stroke-width': 0.6, rx: 0.5 }));
      g.appendChild(this.el('path', { d: `M${kx},${ky} L${kx+d},${ky-d} L${kx+kSz+d},${ky-d} L${kx+kSz},${ky} Z`, fill: '#ef5350', opacity: 0.2, stroke: '#c62828', 'stroke-width': 0.3 }));
      g.appendChild(this.el('path', { d: `M${kx+kSz},${ky} L${kx+kSz+d},${ky-d} L${kx+kSz+d},${ky+kSz-d} L${kx+kSz},${ky+kSz} Z`, fill: '#ef5350', opacity: 0.25, stroke: '#c62828', 'stroke-width': 0.3 }));
      const kLbl = this.el('text', { x: bx + cw / 2, y: by + ch2 + 7, 'text-anchor': 'middle', fill: c.text, 'font-size': '5px', 'font-family': font, opacity: 0.7 });
      kLbl.textContent = k + '×' + k;
      g.appendChild(kLbl);
      const fLbl = this.el('text', { x: w / 2, y: h + 8, 'text-anchor': 'middle', fill: c.text, 'font-size': '5.5px', 'font-family': font, opacity: 0.7 });
      fLbl.textContent = (node.config?.filters || '?') + ' filters';
      g.appendChild(fLbl);
    } else if (node.type === 'DepthwiseConv2D' || node.type === 'SeparableConv2D') {
      const topLbl = this.el('text', { x: w / 2, y: -4, 'text-anchor': 'middle', fill: c.text, 'font-size': '6px', 'font-weight': '600', 'font-family': font });
      topLbl.textContent = 'DW-CONV';
      g.appendChild(topLbl);
      g.appendChild(this.el('rect', { x: 0, y: 0, width: w, height: h, rx: 5, ry: 5, fill: c.fill, stroke: c.stroke, 'stroke-width': 1.2 }));
      const nSlice = 4, cw = w * 0.38, ch2 = h * 0.42, d = w * 0.2;
      const bx = (w - cw - d) / 2, by = (h - ch2 - d) / 2 + d;
      const sliceD = d / nSlice;
      for (let s = nSlice - 1; s >= 0; s--) {
        const sx = bx + s * sliceD, sy = by - s * sliceD;
        const op = 0.3 + s * 0.12;
        g.appendChild(this.el('path', { d: `M${sx},${sy} L${sx+cw},${sy} L${sx+cw},${sy+ch2} L${sx},${sy+ch2} Z`, fill: c.stroke, opacity: op, stroke: c.stroke, 'stroke-width': s === 0 ? 0.7 : 0.3 }));
      }
      const k = this.numCfg(node.config?.kernel_size, 3);
      const kSz = Math.min(cw, ch2) * Math.min(k / 6, 0.7);
      const kx = bx + 1, ky = by + ch2 - kSz - 1;
      g.appendChild(this.el('rect', { x: kx, y: ky, width: kSz, height: kSz, fill: '#ef5350', opacity: 0.5, stroke: '#c62828', 'stroke-width': 0.6, rx: 0.5 }));
      const botLbl = this.el('text', { x: w / 2, y: h + 8, 'text-anchor': 'middle', fill: c.text, 'font-size': '5.5px', 'font-family': font, opacity: 0.7 });
      botLbl.textContent = k + '×' + k;
      g.appendChild(botLbl);
    } else if (node.type === 'Flatten') {
      const topLbl = this.el('text', { x: w / 2, y: -4, 'text-anchor': 'middle', fill: c.text, 'font-size': '6.5px', 'font-weight': '600', 'font-family': font });
      topLbl.textContent = 'Flatten';
      g.appendChild(topLbl);
      g.appendChild(this.el('rect', { x: 0, y: 0, width: w, height: h, rx: 5, ry: 5, fill: c.fill, stroke: c.stroke, 'stroke-width': 1.2 }));
      const colors = ['#42a5f5', '#66bb6a', '#ffa726', '#ef5350'];
      const cs = 5, gap = 1, arrowLen = 4;
      const gridH = 2 * cs + gap;
      const vecH = 4 * cs + 3 * gap;
      const totalH = gridH + arrowLen + 4 + vecH;
      const startY = (h - totalH) / 2;
      const gridW = 2 * cs + gap, gridX = (w - gridW) / 2;
      for (let r = 0; r < 2; r++) for (let cc = 0; cc < 2; cc++) {
        g.appendChild(this.el('rect', { x: gridX + cc * (cs + gap), y: startY + r * (cs + gap), width: cs, height: cs, rx: 1, fill: colors[r * 2 + cc], opacity: 0.7 }));
      }
      const arrowTop = startY + gridH + 2, arrowBot = arrowTop + arrowLen;
      const ax = w / 2;
      g.appendChild(this.el('line', { x1: ax, y1: arrowTop, x2: ax, y2: arrowBot, stroke: c.stroke, 'stroke-width': 0.8 }));
      g.appendChild(this.el('path', { d: `M${ax - 1.5},${arrowBot - 2} L${ax},${arrowBot} L${ax + 1.5},${arrowBot - 2}`, fill: 'none', stroke: c.stroke, 'stroke-width': 0.8 }));
      const vecY = arrowBot + 2;
      const vecX = (w - cs) / 2;
      for (let i = 0; i < 4; i++) {
        g.appendChild(this.el('rect', { x: vecX, y: vecY + i * (cs + gap), width: cs, height: cs, rx: 1, fill: colors[i], opacity: 0.7 }));
      }
    } else {
      const rx = isIdentity ? 3 : 5;
      g.appendChild(this.el('rect', {
        x: 0, y: 0, width: w, height: h, rx, ry: rx,
        fill: c.fill, stroke: c.stroke, 'stroke-width': 1.2,
      }));

      const typeName = isIdentity ? this.shortType(node.type) : this.shortTypeName(node.type);
      const paramStr = isIdentity ? '' : this.shortLabel(node);
      const parts = [typeName, paramStr].filter(Boolean);
      const label = parts.join(', ');
      const textG = this.el('g', { transform: `translate(${w / 2}, ${h / 2}) rotate(-90)` });
      const txt = this.el('text', {
        x: 0, y: 0, 'text-anchor': 'middle', 'dominant-baseline': 'central',
        fill: c.text, 'font-size': isIdentity ? '8px' : '8px', 'font-weight': '600',
        'font-family': font, 'letter-spacing': '0.2px',
      });
      txt.textContent = label;
      textG.appendChild(txt);
      g.appendChild(textG);
    }

    const title = this.el('title');
    title.textContent = this.tooltip(node);
    g.appendChild(title);

    parent.appendChild(g);
    this.nodeElements[node.id] = g;
  }

  precomputeSkipRoutes(graph, positions) {
    this._skipArcY = new Map();
    const skips = [];
    for (const edge of graph.edges) {
      const fi = this._nodeIdx?.get(edge.from) ?? 0;
      const ti = this._nodeIdx?.get(edge.to) ?? 0;
      const span = Math.abs(ti - fi);
      const isMainEdge = this._mainEdges && this._mainEdges.has(edge.from + ':' + edge.to);
      const needsSkip = !isMainEdge && span > 1;
      if (needsSkip) {
        const from = positions[edge.from], to = positions[edge.to];
        if (!from || !to) continue;
        const minI = Math.min(fi, ti), maxI = Math.max(fi, ti);
        let maxBottom = 0, minTop = Infinity;
        for (let i = minI; i <= maxI; i++) {
          const n = graph.nodes[i];
          if (!n) continue;
          const p = positions[n.id];
          if (p) { maxBottom = Math.max(maxBottom, p.y + p.h); minTop = Math.min(minTop, p.y); }
        }
        const minX = Math.min(from.x, to.x);
        const maxX = Math.max(from.x + from.w, to.x + to.w);
        skips.push({ edge, baseY: maxBottom, minX, maxX, span });
      }
    }
    const assignLanes = (list) => {
      list.sort((a, b) => a.span - b.span);
      const lanes = [];
      for (const s of list) {
        let lane = 0;
        for (let l = 0; l < lanes.length; l++) {
          const conflict = lanes[l].some(o => !(s.maxX < o.minX - 10 || s.minX > o.maxX + 10));
          if (!conflict) { lane = l; break; }
          lane = l + 1;
        }
        if (!lanes[lane]) lanes[lane] = [];
        lanes[lane].push(s);
      }
      return lanes;
    };

    const TRAP_H = 36, LINE_GAP = 8, BASE_PAD = 12;

    const lanes = assignLanes(skips);

    // Check which lanes have branch nodes (need more height)
    const branchNodesByEdge = new Map();
    if (this._branchNodes) {
      for (const bn of this._branchNodes) {
        const inEdge = graph.edges.find(e => e.to === bn.node.id);
        const outEdge = graph.edges.find(e => e.from === bn.node.id);
        if (!inEdge || !outEdge) continue;
        for (const s of skips) {
          if (s.edge.from === inEdge.from || s.edge.to === outEdge.to) {
            branchNodesByEdge.set(s.edge, bn);
            break;
          }
        }
      }
    }

    let curY = 0;
    const laneY = [];
    for (let l = 0; l < lanes.length; l++) {
      const hasLabel = lanes[l].some(s => s.edge.label);
      const hasBranch = lanes[l].some(s => branchNodesByEdge.has(s.edge));
      const branchH = hasBranch ? 50 : 0;
      const laneH = Math.max(hasLabel ? TRAP_H + LINE_GAP : LINE_GAP, branchH + LINE_GAP);
      laneY.push(curY + laneH / 2);
      curY += laneH;
    }
    for (const s of skips) {
      const lane = lanes.findIndex(l => l.includes(s));
      this._skipArcY.set(s.edge, s.baseY + BASE_PAD + laneY[lane]);
    }
    let maxArcY = 0;
    for (const y of this._skipArcY.values()) maxArcY = Math.max(maxArcY, y);
    this._skipTotalHeight = maxArcY > 0 ? maxArcY + BASE_PAD : 0;

    // Position branch nodes below main path, laid out horizontally
    if (this._branchNodes && this._branchNodes.length > 0) {
      // Find the main path Y bottom
      let mainBottom = 0;
      for (const [id, p] of Object.entries(positions)) {
        if (!p.branch) mainBottom = Math.max(mainBottom, p.y + p.h);
      }
      const branchY = mainBottom + 20;

      // Find the X range: between the first and last main-path ancestor/descendant
      const inMap = new Map(), outMap = new Map();
      for (const e of graph.edges) {
        if (!outMap.has(e.from)) outMap.set(e.from, []);
        outMap.get(e.from).push(e.to);
        if (!inMap.has(e.to)) inMap.set(e.to, []);
        inMap.get(e.to).push(e.from);
      }

      // Find main-path anchor for branches: nearest main ancestor and descendant
      const findMainAncestor = (nid, visited) => {
        if (!visited) visited = new Set();
        if (visited.has(nid)) return null;
        visited.add(nid);
        if (this._mainPath.has(nid) && positions[nid] && !positions[nid].branch) return nid;
        for (const p of (inMap.get(nid) || [])) {
          const r = findMainAncestor(p, visited);
          if (r) return r;
        }
        return null;
      };
      const findMainDescendant = (nid, visited) => {
        if (!visited) visited = new Set();
        if (visited.has(nid)) return null;
        visited.add(nid);
        if (this._mainPath.has(nid) && positions[nid] && !positions[nid].branch) return nid;
        for (const c of (outMap.get(nid) || [])) {
          const r = findMainDescendant(c, visited);
          if (r) return r;
        }
        return null;
      };

      // Layout branch nodes left to right
      let branchX = 0;
      for (const bn of this._branchNodes) {
        const ancestor = findMainAncestor(bn.node.id);
        const descendant = findMainDescendant(bn.node.id);
        const aPos = ancestor ? positions[ancestor] : null;
        const dPos = descendant ? positions[descendant] : null;

        if (aPos && dPos) {
          // Center this branch node between its anchors
          const rangeStart = aPos.x + aPos.w;
          const rangeEnd = dPos.x;
          const rangeMid = (rangeStart + rangeEnd) / 2;
          branchX = Math.max(branchX, rangeMid - bn.d.w / 2);
        }

        positions[bn.node.id] = { x: branchX, y: branchY, w: bn.d.w, h: bn.d.h, node: bn.node, branch: true };
        branchX += bn.d.w + 4;
      }
    }
  }

  drawEdgeDagre(parent, edge, graph, positions) {
    const from = positions[edge.from], to = positions[edge.to];
    if (!from || !to) return;
    const isMainEdge = this._mainEdges && this._mainEdges.has(edge.from + ':' + edge.to);

    const dagreEdge = this._dagreGraph.edge(edge.from, edge.to);
    const isFromCircle = ['Add','Subtract','Multiply','Average','Concatenate'].includes(from.node?.type);
    const isToCircle = ['Add','Subtract','Multiply','Average','Concatenate'].includes(to.node?.type);
    const x1 = isFromCircle ? from.x + from.w / 2 + Math.min(from.w, from.h) / 2 : from.x + from.w;
    const y1 = from.y + from.h / 2;
    const x2 = isToCircle ? to.x + to.w / 2 - Math.min(to.w, to.h) / 2 : to.x;
    const y2 = to.y + to.h / 2;

    const allPts = [{ x: x1, y: y1 }];
    if (dagreEdge && dagreEdge.points) {
      for (const p of dagreEdge.points) allPts.push(p);
    }
    allPts.push({ x: x2, y: y2 });

    let d;
    if (allPts.length <= 3) {
      const mx = (x1 + x2) / 2;
      if (Math.abs(y2 - y1) < 3) d = `M${x1},${y1} L${x2},${y2}`;
      else d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    } else {
      d = `M${x1},${y1}`;
      for (let i = 1; i < allPts.length - 1; i++) {
        const cur = allPts[i], next = allPts[i + 1];
        const mx = (cur.x + next.x) / 2, my = (cur.y + next.y) / 2;
        if (i === 1) d += ` Q${cur.x},${cur.y} ${mx},${my}`;
        else d += ` T${mx},${my}`;
      }
      d += ` L${x2},${y2}`;
    }

    const path = this.el('path', {
      d, fill: 'none',
      stroke: '#546e7a',
      'stroke-width': isMainEdge ? 1.4 : 1.1,
      'marker-end': 'url(#arrow)',
      opacity: isMainEdge ? 0.7 : 0.5,
      class: 'graph-edge',
      'data-from': edge.from, 'data-to': edge.to,
    });
    parent.appendChild(path);
  }

  drawEdge(parent, edge, graph, positions) {
    const from = positions[edge.from], to = positions[edge.to];
    if (!from || !to) return;

    const fromIdx = this._nodeIdx?.get(edge.from) ?? 0;
    const toIdx = this._nodeIdx?.get(edge.to) ?? 0;
    const span = Math.abs(toIdx - fromIdx);
    const isMainEdge = this._mainEdges && this._mainEdges.has(edge.from + ':' + edge.to);
    const isSkip = !isMainEdge && span > 1;

    let x1, y1, x2, y2, d;

    if (!this._edgeOffsets) {
      this._edgeOffsets = new Map();
      const allAtNode = new Map();
      for (const e of graph.edges) {
        const me = this._mainEdges && this._mainEdges.has(e.from + ':' + e.to);
        if (me) continue;
        if (!allAtNode.has(e.from)) allAtNode.set(e.from, { out: [], in: [] });
        if (!allAtNode.has(e.to)) allAtNode.set(e.to, { out: [], in: [] });
        allAtNode.get(e.from).out.push(e);
        allAtNode.get(e.to).in.push(e);
      }
      for (const [nid, info] of allAtNode) {
        const total = info.out.length + info.in.length;
        if (total <= 1) continue;
        let idx = 0;
        for (const e of info.in) {
          const o = this._edgeOffsets.get(e) || { fromOff: 0, toOff: 0 };
          o.toOff = (idx - (total-1)/2) * 10;
          this._edgeOffsets.set(e, o);
          idx++;
        }
        for (const e of info.out) {
          const o = this._edgeOffsets.get(e) || { fromOff: 0, toOff: 0 };
          o.fromOff = (idx - (total-1)/2) * 10;
          this._edgeOffsets.set(e, o);
          idx++;
        }
      }
    }
    const off = this._edgeOffsets.get(edge) || { fromOff: 0, toOff: 0 };

    if (from.branch || to.branch) {
      if (from.branch && !to.branch) {
        x1 = from.x + from.w; y1 = from.y + from.h / 2;
        x2 = to.x + to.w / 2 + off.toOff; y2 = to.y + to.h;
        d = `M${x1},${y1} H${x2} V${y2}`;
      } else if (!from.branch && to.branch) {
        x1 = from.x + from.w / 2 + off.fromOff; y1 = from.y + from.h;
        x2 = to.x; y2 = to.y + to.h / 2;
        d = `M${x1},${y1} V${y2} H${x2}`;
      } else {
        x1 = from.x + from.w; y1 = from.y + from.h / 2;
        x2 = to.x; y2 = to.y + to.h / 2;
        d = `M${x1},${y1} H${x2}`;
      }
    } else if (isSkip) {
      x1 = from.x + from.w / 2 + off.fromOff;
      y1 = from.y + from.h;
      x2 = to.x + to.w / 2 + off.toOff;
      y2 = to.y + to.h;
      const arcY = this._skipArcY?.get(edge) ?? (Math.max(y1, y2) + 20);
      d = `M${x1},${y1} V${arcY} H${x2} V${y2}`;
    } else {
      x1 = from.x + from.w;
      y1 = from.y + from.h / 2 + off.fromOff;
      x2 = to.x;
      y2 = to.y + to.h / 2 + off.toOff;
      if (Math.abs(y2 - y1) < 3) {
        d = `M${x1},${y1} H${x2}`;
      } else {
        const midX = x1 + (x2 - x1) * 0.5;
        d = `M${x1},${y1} H${midX} V${y2} H${x2}`;
      }
    }

    const path = this.el('path', {
      d, fill: 'none',
      stroke: isSkip ? '#546e7a' : '#78909c',
      'stroke-width': isSkip ? 1.4 : 1.3,
      'marker-end': 'url(#arrow)',
      opacity: isSkip ? 0.6 : 0.75,
      class: 'graph-edge',
      'data-from': edge.from, 'data-to': edge.to,
    });
    parent.appendChild(path);

  }

  addInteractivity(graph) {
    const self = this;
    for (const g of this.svg.querySelectorAll('.graph-node')) {
      g.addEventListener('mouseenter', () => {
        const shape = g.querySelector('rect, circle');
        if (shape) {
          shape.setAttribute('stroke-width', '2.5');
          shape.setAttribute('filter', 'url(#hover-shadow)');
        }
        const id = g.getAttribute('data-id');
        for (const edge of self.svg.querySelectorAll('.graph-edge')) {
          if (edge.getAttribute('data-from') === id || edge.getAttribute('data-to') === id) {
            edge.setAttribute('stroke-width', '2.5');
            edge.setAttribute('opacity', '0.9');
          }
        }
      });
      g.addEventListener('mouseleave', () => {
        const shape = g.querySelector('rect, circle');
        if (shape) {
          shape.setAttribute('stroke-width', '1.5');
          shape.removeAttribute('filter');
        }
        for (const edge of self.svg.querySelectorAll('.graph-edge')) {
          const isMain = self._mainEdges && self._mainEdges.has(edge.getAttribute('data-from') + ':' + edge.getAttribute('data-to'));
          edge.setAttribute('stroke-width', isMain ? '1.4' : '1.1');
          edge.setAttribute('opacity', isMain ? '0.7' : '0.5');
        }
      });
      g.addEventListener('click', (e) => {
        e.stopPropagation();
        const node = graph.nodes.find(n => n.id === g.getAttribute('data-id'));
        if (node) { self.showNodeDetail(node); self.highlightCode(node); }
      });
    }
    this.svg.addEventListener('click', () => { self.hideNodeDetail(); self.clearHighlight(); });
  }

  showNodeDetail(node) {
    const panel = document.getElementById('info-panel');
    if (!panel) return;
    this._detailNode = node;
    if (typeof window !== 'undefined') window._selectedNodeId = node.id;

    const macs = this.estimateMACs(node);
    let html = '<div class="info-title">' + node.type + (node.activation ? ' + ' + node.activation : '') + '</div>';

    html += '<div class="info-signature">' + this.buildSignature(node) + '</div>';

    html += '<div class="info-section"><div class="info-section-title">Shape</div>';
    html += '<div class="info-row"><span class="info-label">Output</span><span class="info-value">' + this.formatShape(node.outputShape) + '</span></div>';
    if (node.inputShapes && node.inputShapes.length > 0) {
      html += '<div class="info-row"><span class="info-label">Input</span><span class="info-value">' + node.inputShapes.map(s => this.formatShape(s)).join(', ') + '</span></div>';
    }
    html += '</div>';

    html += '<div class="info-section"><div class="info-section-title">Compute</div>';
    html += '<div class="info-row"><span class="info-label">Parameters</span><span class="info-value highlight">' + this.formatNum(node.params || 0) + '</span></div>';
    html += '<div class="info-row"><span class="info-label">MACs</span><span class="info-value highlight">' + this.formatNum(macs) + '</span></div>';
    html += '</div>';

    const cfg = node.config || {};
    const cfgKeys = Object.keys(cfg).filter(k => cfg[k] !== undefined && cfg[k] !== null);
    if (cfgKeys.length > 0) {
      const editable = new Set(['filters','kernel_size','units','strides','padding','pool_size','size','rate','block_size','output_dim','alpha','use_bias']);
      const selectFields = { padding: ['same','valid'], activation: ['relu','sigmoid','tanh','softmax','linear','swish','elu','selu'] };

      let sharedCount = 0;
      const dynFields = new Set();
      if (node.defLine > 0 && this._lastGraph) {
        for (const other of this._lastGraph.nodes) {
          if (other.id !== node.id && other.defLine === node.defLine) sharedCount++;
        }
        const editor = document.getElementById('code-editor');
        if (editor) {
          const srcLine = editor.value.split('\n')[node.defLine - 1] || '';
          for (const k of cfgKeys) {
            if (!editable.has(k)) continue;
            const re = new RegExp('\\b' + k + '\\s*=\\s*([^,)]+)');
            const m = srcLine.match(re);
            if (m) {
              const val = m[1].trim();
              if (/[a-zA-Z_]/.test(val) && !/^['"]/.test(val) && val !== 'True' && val !== 'False') {
                dynFields.add(k);
              }
            } else {
              const posMap = { Conv2D: ['filters','kernel_size'], Conv2DTranspose: ['filters','kernel_size'], Dense: ['units'], DepthwiseConv2D: ['kernel_size'], MaxPooling2D: ['pool_size'], UpSampling2D: ['size'] };
              const positionals = posMap[node.type] || [];
              const posIdx = positionals.indexOf(k);
              if (posIdx >= 0) {
                const paren = srcLine.indexOf('(');
                if (paren >= 0) {
                  const argsStr = srcLine.substring(paren + 1);
                  const parts = argsStr.split(',');
                  if (posIdx < parts.length) {
                    const pval = parts[posIdx].trim().replace(/[)]/g, '');
                    if (/[a-zA-Z_]/.test(pval) && !/^['"]/.test(pval) && !/^\(/.test(pval)) {
                      dynFields.add(k);
                    }
                  }
                }
              }
            }
          }
        }
      }

      html += '<div class="info-section"><div class="info-section-title">Config</div>';

      if (sharedCount > 0) {
        html += '<div class="info-warn">Shared definition — editing affects ' + (sharedCount + 1) + ' layers</div>';
      }

      for (const k of cfgKeys) {
        const v = cfg[k];
        html += '<div class="info-row"><span class="info-label">' + k + '</span>';
        const canEdit = node.defLine > 0 && editable.has(k) && !dynFields.has(k);
        if (canEdit) {
          if (selectFields[k]) {
            html += '<select class="info-input" data-field="' + k + '">';
            for (const opt of selectFields[k]) {
              html += '<option value="' + opt + '"' + (String(v) === opt ? ' selected' : '') + '>' + opt + '</option>';
            }
            html += '</select>';
          } else if (typeof v === 'boolean') {
            html += '<select class="info-input" data-field="' + k + '"><option value="True"' + (v ? ' selected' : '') + '>True</option><option value="False"' + (!v ? ' selected' : '') + '>False</option></select>';
          } else {
            const displayVal = Array.isArray(v) ? v.join(', ') : v;
            html += '<input class="info-input" data-field="' + k + '" value="' + displayVal + '" />';
          }
        } else {
          html += '<span class="info-value">' + (typeof v === 'object' ? JSON.stringify(v) : v) + '</span>';
          if (dynFields.has(k)) html += '<span class="info-dyn" title="Computed from variable">expr</span>';
        }
        html += '</div>';
      }
      html += '</div>';
    }

    panel.innerHTML = html;

    const self = this;
    for (const input of panel.querySelectorAll('.info-input')) {
      input.addEventListener('change', function() {
        if (typeof updateSourceValue === 'function') {
          updateSourceValue(self._detailNode, this.dataset.field, this.value);
        }
      });
    }
  }

  buildSignature(node) {
    const TF_PREFIX = {
      Conv2D: 'layers.Conv2D', Conv1D: 'layers.Conv1D', Conv2DTranspose: 'layers.Conv2DTranspose',
      DepthwiseConv2D: 'layers.DepthwiseConv2D', SeparableConv2D: 'layers.SeparableConv2D',
      Dense: 'layers.Dense', MaxPooling2D: 'layers.MaxPooling2D', AveragePooling2D: 'layers.AveragePooling2D',
      UpSampling2D: 'layers.UpSampling2D', BatchNormalization: 'layers.BatchNormalization',
      Dropout: 'layers.Dropout', Flatten: 'layers.Flatten', Reshape: 'layers.Reshape',
      Embedding: 'layers.Embedding', LSTM: 'layers.LSTM', GRU: 'layers.GRU',
      PReLU: 'layers.PReLU', LeakyReLU: 'layers.LeakyReLU',
      Concatenate: 'tf.concat', Add: 'tf.add', Resize: 'tf.image.resize',
      SpaceToDepth: 'tf.nn.space_to_depth', DepthToSpace: 'tf.nn.depth_to_space',
      Pad: 'tf.pad', Reduce: 'tf.reduce_mean',
    };
    const prefix = TF_PREFIX[node.type] || node.type;
    const cfg = node.config || {};
    const parts = Object.entries(cfg)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => k + '=' + this.formatPyVal(v));
    return prefix + '(' + parts.join(', ') + ')';
  }

  formatPyVal(v) {
    if (typeof v === 'string') return "'" + v + "'";
    if (typeof v === 'boolean') return v ? 'True' : 'False';
    if (Array.isArray(v)) return '(' + v.join(', ') + ')';
    return String(v);
  }

  hideNodeDetail() {
    const panel = document.getElementById('info-panel');
    if (panel) panel.innerHTML = '<div class="info-empty">Click a node to inspect</div>';
  }

  highlightCode(node) {
    this.clearHighlight();
    const scroll = document.getElementById('editor-scroll');
    const lineNumEl = document.getElementById('line-numbers');
    if (!scroll) return;

    const line = node.defLine > 0 ? node.defLine : (node.srcLine || 0);
    const matchLine = line - 1;
    if (matchLine < 0) return;

    const lineH = 20;
    const padTop = 10;
    const barTop = padTop + matchLine * lineH;

    const bar = document.createElement('div');
    bar.className = 'hl-line-bg';
    bar.id = 'hl-bar';
    bar.style.top = barTop + 'px';
    scroll.appendChild(bar);

    scroll.scrollTop = Math.max(0, barTop - scroll.clientHeight / 2);

    if (lineNumEl) {
      const prev = lineNumEl.querySelector('.hl');
      if (prev) prev.classList.remove('hl');
      const spans = lineNumEl.querySelectorAll('span');
      if (spans[matchLine]) spans[matchLine].classList.add('hl');
    }
  }

  clearHighlight() {
    const bar = document.getElementById('hl-bar');
    if (bar) bar.remove();
    const lineNumEl = document.getElementById('line-numbers');
    if (lineNumEl) {
      const prev = lineNumEl.querySelector('.hl');
      if (prev) prev.classList.remove('hl');
    }
  }

  numCfg(v, dflt) {
    if (typeof v === 'number') return v;
    if (Array.isArray(v)) return typeof v[0] === 'number' ? v[0] : dflt;
    if (v && typeof v === 'object' && v.items) {
      const first = v.items[0];
      return typeof first === 'number' ? first : (first?.value != null ? Number(first.value) : dflt);
    }
    return dflt;
  }

  dim(shape, idx) {
    if (!shape || idx >= shape.length) return 0;
    return typeof shape[idx] === 'number' ? shape[idx] : 0;
  }

  estimateMACs(node) {
    const s = node.outputShape;
    const cfg = node.config || {};
    if (!s || s.length < 2) return 0;
    const outH = this.dim(s, 1);
    const outW = s.length >= 4 ? this.dim(s, 2) : 1;
    const outC = s.length >= 4 ? this.dim(s, 3) : this.dim(s, s.length - 1);
    const inShapes = node.inputShapes || [];
    const inC = inShapes[0] ? this.dim(inShapes[0], inShapes[0].length - 1) : 0;

    switch (node.type) {
      case 'Conv2D': case 'SeparableConv2D': {
        const k = this.numCfg(cfg.kernel_size, 3);
        return outH * outW * outC * k * k * inC;
      }
      case 'Conv2DTranspose': {
        const k = this.numCfg(cfg.kernel_size, 3);
        return outH * outW * outC * k * k * inC;
      }
      case 'DepthwiseConv2D': {
        const k = this.numCfg(cfg.kernel_size, 3);
        return outH * outW * inC * k * k;
      }
      case 'Dense': return this.numCfg(cfg.units, 0) * inC;
      case 'Conv1D': {
        const k = this.numCfg(cfg.kernel_size, 3);
        return outH * outC * k * inC;
      }
      case 'LSTM': { const u = this.numCfg(cfg.units, 0); return 4 * outH * (inC * u + u * u); }
      case 'GRU': { const u = this.numCfg(cfg.units, 0); return 3 * outH * (inC * u + u * u); }
      default: return 0;
    }
  }

  applyView(mode) {
    if (!this._lastGraph || !this.nodeElements) return;
    const graph = this._lastGraph;

    if (mode === 'type') {
      for (const node of graph.nodes) {
        const el = this.nodeElements[node.id];
        if (!el) continue;
        const c = COLORS[node.type] || DEFAULT_COLOR;
        const shape = el.querySelector('rect, circle');
        if (shape) { shape.setAttribute('fill', c.fill); shape.setAttribute('stroke', c.stroke); }
        const txt = el.querySelector('text');
        if (txt) txt.setAttribute('fill', c.text);
      }
      this._removeLegend();
      return;
    }

    let values = [];
    for (const node of graph.nodes) {
      const v = mode === 'macs' ? this.estimateMACs(node) : (node.params || 0);
      values.push({ node, v });
    }
    const maxV = Math.max(1, ...values.map(x => x.v));

    for (const { node, v } of values) {
      const el = this.nodeElements[node.id];
      if (!el) continue;
      const t = maxV > 0 ? v / maxV : 0;
      const fill = this._heatColor(t);
      const stroke = this._heatColor(Math.min(1, t + 0.2));
      const shape = el.querySelector('rect, circle');
      if (shape) { shape.setAttribute('fill', fill); shape.setAttribute('stroke', stroke); }
      const txt = el.querySelector('text');
      if (txt) txt.setAttribute('fill', t > 0.5 ? '#fff' : '#333');
    }
    this._showLegend(mode, maxV);
  }

  _heatColor(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.5) {
      const r = Math.round(255);
      const g = Math.round(255 - t * 2 * 80);
      const b = Math.round(255 - t * 2 * 200);
      return `rgb(${r},${g},${b})`;
    }
    const r = Math.round(255 - (t - 0.5) * 2 * 60);
    const g = Math.round(175 - (t - 0.5) * 2 * 140);
    const b = Math.round(55 - (t - 0.5) * 2 * 55);
    return `rgb(${r},${g},${b})`;
  }

  _showLegend(mode, maxV) {
    this._removeLegend();
    const panel = document.getElementById('graph-panel');
    if (!panel) return;
    const el = document.createElement('div');
    el.id = 'heatmap-legend';
    el.innerHTML = '<div class="legend-bar"></div><div class="legend-labels"><span>0</span><span>' + this.formatNum(maxV) + (mode === 'macs' ? ' MACs' : ' params') + '</span></div>';
    panel.appendChild(el);
  }

  _removeLegend() {
    const old = document.getElementById('heatmap-legend');
    if (old) old.remove();
  }

  addStats(graph) {
    let totalParams = 0, totalMACs = 0;
    for (const n of graph.nodes) {
      totalParams += n.params || 0;
      totalMACs += this.estimateMACs(n);
    }
    const old = document.getElementById('global-stats-bar');
    if (old) old.remove();
    const bar = document.createElement('div');
    bar.className = 'stats-bar';
    bar.id = 'global-stats-bar';
    const inputNode = graph.nodes.find(n => n.type === 'Input');
    const inputPixels = inputNode && inputNode.outputShape ? (inputNode.outputShape[1] || 1) * (inputNode.outputShape[2] || 1) : 1;
    const macsPerPx = inputPixels > 0 ? (totalMACs / inputPixels) : 0;
    bar.innerHTML = '<div class="stats-item"><span class="stats-value">' + graph.nodes.length + '</span><span class="stats-label">Layers</span></div>'
      + '<div class="stats-item"><span class="stats-value">' + this.formatNum(totalParams) + '</span><span class="stats-label">Parameters</span></div>'
      + '<div class="stats-item"><span class="stats-value">' + this.formatNum(totalMACs) + '</span><span class="stats-label">MACs</span></div>'
      + '<div class="stats-item"><span class="stats-value">' + this.formatNum(Math.round(macsPerPx)) + '</span><span class="stats-label">MACs/px</span></div>'
      + '<div class="stats-item" id="stats-detail" style="margin-left:auto"><span class="stats-label">Click a node for details</span></div>';
    document.body.appendChild(bar);
  }

  exportSVG() {
    if (!this.svg) return null;
    const clone = this.svg.cloneNode(true);
    clone.setAttribute('viewBox', `0 0 ${this.naturalW} ${this.naturalH}`);
    clone.setAttribute('width', this.naturalW);
    clone.setAttribute('height', this.naturalH);
    clone.style.width = ''; clone.style.height = '';
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', '100%'); bg.setAttribute('height', '100%'); bg.setAttribute('fill', '#ffffff');
    clone.insertBefore(bg, clone.firstChild);
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = `text { font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif; }`;
    clone.insertBefore(style, clone.firstChild);
    const serializer = new XMLSerializer();
    return serializer.serializeToString(clone);
  }

  async exportSVGWithFonts() {
    if (!this.svg) return null;
    let fontCSS = '';
    try {
      const resp = await fetch('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      const css = await resp.text();
      const urls = [...css.matchAll(/url\(([^)]+)\)/g)];
      let inlined = css;
      for (const m of urls) {
        try {
          const fontResp = await fetch(m[1]);
          const blob = await fontResp.blob();
          const b64 = await new Promise(r => { const rd = new FileReader(); rd.onload = () => r(rd.result); rd.readAsDataURL(blob); });
          inlined = inlined.replace(m[1], b64);
        } catch(e) {}
      }
      fontCSS = inlined;
    } catch(e) {}
    const clone = this.svg.cloneNode(true);
    clone.setAttribute('viewBox', `0 0 ${this.naturalW} ${this.naturalH}`);
    clone.setAttribute('width', this.naturalW);
    clone.setAttribute('height', this.naturalH);
    clone.style.width = ''; clone.style.height = '';
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', '100%'); bg.setAttribute('height', '100%'); bg.setAttribute('fill', '#ffffff');
    clone.insertBefore(bg, clone.firstChild);
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = fontCSS + `\ntext { font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif; }`;
    clone.insertBefore(style, clone.firstChild);
    return new XMLSerializer().serializeToString(clone);
  }

  // ─── SVG helpers ──────────────────────────────────────

  createSVG(w, h) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.display = 'block';
    this.vb = { x: 0, y: 0, w: w, h: h };
    this.naturalW = w;
    this.naturalH = h;
    this.applyViewBox(svg);
    return svg;
  }

  applyViewBox(svg) {
    svg = svg || this.svg;
    svg.setAttribute('viewBox', `${this.vb.x} ${this.vb.y} ${this.vb.w} ${this.vb.h}`);
  }

  zoomFit() {
    if (!this.svg) return;
    this.vb = { x: 0, y: 0, w: this.naturalW, h: this.naturalH };
    this.applyViewBox();
  }

  zoomBy(factor) {
    if (!this.svg) return;
    const cx = this.vb.x + this.vb.w / 2;
    const cy = this.vb.y + this.vb.h / 2;
    this.vb.w *= factor;
    this.vb.h *= factor;
    this.vb.x = cx - this.vb.w / 2;
    this.vb.y = cy - this.vb.h / 2;
    this.applyViewBox();
  }

  addZoomPan() {
    const svg = this.svg;
    const self = this;
    let isPanning = false, startX = 0, startY = 0, startVB = null;

    const controls = document.createElement('div');
    controls.id = 'zoom-controls';
    controls.innerHTML = '<button class="zoom-btn" id="zoom-in" title="Zoom in">+</button>'
      + '<button class="zoom-btn" id="zoom-out" title="Zoom out">&minus;</button>'
      + '<button class="zoom-btn zoom-fit" id="zoom-fit" title="Fit to view">&#x2922;</button>';
    this.container.appendChild(controls);
    document.getElementById('zoom-in').addEventListener('click', () => self.zoomBy(1 / 1.3));
    document.getElementById('zoom-out').addEventListener('click', () => self.zoomBy(1.3));
    document.getElementById('zoom-fit').addEventListener('click', () => self.zoomFit());

    svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const scale = e.deltaY > 0 ? 1.06 : 1 / 1.06;
      const rect = svg.getBoundingClientRect();
      const mx = (e.clientX - rect.left) / rect.width;
      const my = (e.clientY - rect.top) / rect.height;
      const px = this.vb.x + mx * this.vb.w;
      const py = this.vb.y + my * this.vb.h;
      this.vb.w *= scale;
      this.vb.h *= scale;
      this.vb.x = px - mx * this.vb.w;
      this.vb.y = py - my * this.vb.h;
      this.applyViewBox();
    }, { passive: false });

    svg.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      isPanning = true;
      startX = e.clientX;
      startY = e.clientY;
      startVB = { ...this.vb };
      svg.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      const rect = svg.getBoundingClientRect();
      const dx = (e.clientX - startX) / rect.width * startVB.w;
      const dy = (e.clientY - startY) / rect.height * startVB.h;
      this.vb.x = startVB.x - dx;
      this.vb.y = startVB.y - dy;
      this.applyViewBox();
    });

    window.addEventListener('mouseup', () => {
      isPanning = false;
      svg.style.cursor = 'grab';
    });

    svg.style.cursor = 'grab';
  }

  addDefs() {
    const defs = this.el('defs');
    const m1 = this.el('marker', { id: 'arrow', viewBox: '0 0 10 8', refX: 10, refY: 4, markerWidth: 7, markerHeight: 6, orient: 'auto-start-reverse' });
    m1.appendChild(this.el('polygon', { points: '0,0.5 10,4 0,7.5', fill: '#546e7a' }));
    defs.appendChild(m1);
    const filter = this.el('filter', { id: 'hover-shadow', x: '-20%', y: '-20%', width: '140%', height: '140%' });
    filter.appendChild(this.el('feDropShadow', { dx: 0, dy: 1, stdDeviation: 2, 'flood-opacity': 0.15 }));
    defs.appendChild(filter);
    this.svg.appendChild(defs);
  }

  el(tag, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  // ─── Label helpers ────────────────────────────────────

  shortLabel(node) {
    const t = node.type;
    const c = node.config || {};
    const s = node.outputShape;
    const inC = node.inputShapes?.[0]?.[node.inputShapes[0]?.length - 1];
    const outC = s?.[s?.length - 1];
    const ks = (c.kernel_size != null) ? this.numCfg(c.kernel_size, 0) : 0;

    if (t === 'Input') return node.label || 'Input';
    if (t === 'Output') return 'Output';

    if (t === 'Conv2D' || t === 'Conv1D' || t === 'Conv3D') {
      const k = ks || 3;
      return `(${k}), ${typeof inC === 'number' ? inC : '?'}→${c.filters || '?'}`;
    }
    if (t === 'Conv2DTranspose') {
      const k = ks || 3;
      return `(${k})T, ${typeof inC === 'number' ? inC : '?'}→${c.filters || '?'}`;
    }
    if (t === 'DepthwiseConv2D') {
      const k = ks || 3;
      return `(${k}), ${typeof inC === 'number' ? inC : '?'}`;
    }
    if (t === 'Dense') return `${typeof inC === 'number' ? inC : '?'}→${c.units || '?'}`;
    if (t === 'SpaceToDepth' || t === 'DepthToSpace') {
      return `(${typeof inC === 'number' ? inC : '?'})→(${typeof outC === 'number' ? outC : '?'})`;
    }
    if (t === 'Resize') {
      const rh = s?.[1], rw = s?.[2];
      return typeof rh === 'number' ? `↓ ${rh}×${rw}` : '↓ resize';
    }
    if (t === 'Slice') return node.label || '[ ]';
    if (t === 'Concatenate') return typeof outC === 'number' ? 'C ' + outC : 'C';
    if (t === 'MaxPooling2D' || t === 'MaxPooling1D') return '↓Pool';
    if (t === 'AveragePooling2D') return '↓AvgP';
    if (t === 'UpSampling2D') return `↑${this.fmtVal(c.size)}`;
    if (t === 'Flatten') return 'Flat';
    if (t === 'Reshape') return 'Resh';
    if (t === 'Pad') return 'Pad';
    if (t === 'Embedding') return `Emb ${c.output_dim || ''}`;
    if (t === 'LSTM') return `LSTM ${c.units || ''}`;
    if (t === 'GRU') return `GRU ${c.units || ''}`;
    if (t === 'GlobalAveragePooling2D') return 'GAP';
    if (t === 'GlobalMaxPooling2D') return 'GMP';
    if (t === 'Reduce') return 'μ';
    return node.label || t;
  }

  shortType(type) {
    const map = { BatchNormalization: 'BN', LayerNormalization: 'LN', GroupNormalization: 'GN',
      Dropout: 'Drop', SpatialDropout2D: 'SDrop', Activation: 'Act', ReLU: 'ReLU',
      LeakyReLU: 'LReLU', PReLU: 'PReLU', ELU: 'ELU', Softmax: 'Softmax' };
    return map[type] || type;
  }

  shortTypeName(type) {
    const map = {
      Conv2D: 'CONV', Conv1D: 'CONV', Conv3D: 'CONV', Conv2DTranspose: 'CONV-T',
      DepthwiseConv2D: 'DW-CONV', SeparableConv2D: 'SEP-CONV',
      Dense: 'DENSE', MaxPooling2D: 'POOL', AveragePooling2D: 'POOL',
      GlobalAveragePooling2D: 'GAP', GlobalMaxPooling2D: 'GMP',
      UpSampling2D: 'UPSAMPLE', Flatten: 'FLATTEN', Reshape: 'RESHAPE',
      SpaceToDepth: 'S2D', DepthToSpace: 'D2S',
      Resize: 'RESIZE', Slice: 'SLICE', Pad: 'PAD', Reduce: 'REDUCE',
      Input: 'INPUT', Output: 'OUTPUT',
      Embedding: 'EMBED', LSTM: 'LSTM', GRU: 'GRU',
    };
    return map[type] || type.toUpperCase();
  }

  formatShape(shape) {
    if (!shape) return '';
    return shape.map(d => d === null ? '?' : d).join('×');
  }

  tooltip(node) {
    const lines = [node.type];
    if (node.config && Object.keys(node.config).length > 0) {
      for (const [k, v] of Object.entries(node.config)) {
        if (v !== undefined && v !== null) lines.push(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`);
      }
    }
    if (node.outputShape) lines.push(`shape: ${this.formatShape(node.outputShape)}`);
    if (node.params > 0) lines.push(`params: ${this.formatNum(node.params)}`);
    return lines.join('\n');
  }

  formatNum(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
  }

  fmtVal(v) {
    if (Array.isArray(v)) return v.join('×');
    return v ?? '';
  }
}
