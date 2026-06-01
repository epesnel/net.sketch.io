const COLORS = {
  Input: '#546e7a', Output: '#546e7a',
  Conv2D: '#1a237e', Conv1D: '#1a237e', Conv3D: '#1a237e',
  Conv2DTranspose: '#283593', Conv1DTranspose: '#283593',
  DepthwiseConv2D: '#4e342e', SeparableConv2D: '#4e342e',
  Dense: '#4a148c',
  MaxPooling2D: '#37474f', MaxPooling1D: '#37474f',
  AveragePooling2D: '#455a64', AveragePooling1D: '#455a64',
  GlobalAveragePooling2D: '#37474f', GlobalMaxPooling2D: '#37474f',
  UpSampling2D: '#37474f', UpSampling1D: '#37474f',
  BatchNormalization: '#546e7a', LayerNormalization: '#607d8b', GroupNormalization: '#607d8b',
  Dropout: '#78909c', SpatialDropout2D: '#78909c',
  Activation: '#880e4f', ReLU: '#880e4f', LeakyReLU: '#880e4f', PReLU: '#880e4f', ELU: '#880e4f', Softmax: '#880e4f',
  Flatten: '#455a64', Reshape: '#546e7a',
  Concatenate: '#37474f', Add: '#37474f', Subtract: '#37474f', Multiply: '#37474f', Average: '#37474f',
  Resize: '#00695c', SpaceToDepth: '#1a237e', DepthToSpace: '#1a237e',
  Pad: '#546e7a', Reduce: '#546e7a', Slice: '#546e7a',
  Embedding: '#4a148c', LSTM: '#311b92', GRU: '#311b92',
  BinOp: '#37474f',
};

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
    g.setGraph({ rankdir: 'LR', nodesep: 8, ranksep: 12, marginx: 20, marginy: 20, align: 'UL' });
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
    this._branchNodes = [];
    this._dagreGraph = g;
    return { positions, width: maxX + 40, height: maxY + 40 };
  }

  nodeDims(node, maxRes, maxCh) {
    const s = node.outputShape;
    const isIdentity = IDENTITY_TYPES.has(node.type);
    if (isIdentity) return { w: 8, h: 40 };
    if (['Add', 'Subtract', 'Multiply', 'Average'].includes(node.type)) return { w: 24, h: 24 };
    if (node.type === 'Concatenate') return { w: 24, h: 24 };
    if (node.type === 'Slice') return { w: 14, h: 40 };
    const res = (s && s.length >= 4 && typeof s[1] === 'number') ? s[1] : maxRes;
    const ch = (s && s.length >= 4 && typeof s[3] === 'number') ? s[3] : 16;
    const resFrac = res / Math.max(1, maxRes);
    let h = Math.max(70, Math.round(80 + resFrac * 100));
    let w = Math.max(18, Math.min(32, 14 + Math.sqrt(ch / Math.max(1, maxCh)) * 18));
    if (node.type === 'Input' || node.type === 'Output') { w = 22; }
    if (node.type === 'Resize') { w = 30; }
    if (node.type === 'SpaceToDepth' || node.type === 'DepthToSpace') { w = 22; }
    return { w: Math.round(w), h: Math.round(h) };
  }

  drawNode(parent, node, pos) {
    if (!pos) return;
    const { x, y, w, h } = pos;
    const color = COLORS[node.type] || '#607d8b';
    const isIdentity = IDENTITY_TYPES.has(node.type);
    const isOp = ['Add', 'Subtract', 'Multiply', 'Average', 'Concatenate'].includes(node.type);

    const g = this.el('g', { class: 'graph-node', 'data-id': node.id, transform: `translate(${x}, ${y})` });
    const isResize = node.type === 'Resize';

    if (isResize) {
      const inH = node.inputShapes?.[0]?.[1];
      const outH = node.outputShape?.[1];
      const isDown = (typeof inH === 'number' && typeof outH === 'number') ? outH < inH : true;
      const inset = h * 0.22;
      const tL = isDown ? 0 : inset;
      const tR = isDown ? inset : 0;
      const bL = isDown ? h : h - inset;
      const bR = isDown ? h - inset : h;
      const pts = `0,${tL} ${w},${tR} ${w},${bR} 0,${bL}`;
      const shadow = this.el('polygon', { points: `1,${tL+1} ${w+1},${tR+1} ${w+1},${bR+1} 1,${bL+1}`, fill: 'rgba(0,0,0,0.08)', 'pointer-events': 'none' });
      g.appendChild(shadow);
      const poly = this.el('polygon', { points: pts, fill: color, opacity: 0.92 });
      g.appendChild(poly);
      const textG = this.el('g', { transform: `translate(${w/2}, ${h/2}) rotate(-90)` });
      const txt = this.el('text', { x: 0, y: 0, 'text-anchor': 'middle', 'dominant-baseline': 'central', fill: '#fff', 'font-size': '8px', 'font-weight': '600', 'font-family': "'Inter', 'SF Pro Text', system-ui, sans-serif" });
      txt.textContent = this.shortLabel(node);
      textG.appendChild(txt);
      g.appendChild(textG);
    } else if (isOp) {
      const r = Math.min(w, h) / 2;
      const cx = w / 2, cy = h / 2;
      const shadow = this.el('circle', { cx: cx + 1, cy: cy + 1, r, fill: 'rgba(0,0,0,0.1)', 'pointer-events': 'none' });
      g.appendChild(shadow);
      const circle = this.el('circle', { cx, cy, r, fill: color, opacity: 0.92, stroke: 'none' });
      g.appendChild(circle);
      const sym = node.type === 'Add' ? '+' : node.type === 'Subtract' ? '−' : node.type === 'Multiply' ? '×' : node.type === 'Concatenate' ? 'C' : 'ø';
      const fontSize = node.type === 'Concatenate' ? '10px' : '16px';
      const symEl = this.el('text', {
        x: cx, y: cy, 'text-anchor': 'middle', 'dominant-baseline': 'central',
        fill: '#fff', 'font-size': fontSize, 'font-weight': '700', 'font-family': "'Inter', 'SF Pro Text', system-ui, sans-serif",
      });
      symEl.textContent = sym;
      g.appendChild(symEl);
    } else {
      const shadow = this.el('rect', {
        x: 1, y: 1, width: w, height: h, rx: isIdentity ? 4 : 6, ry: isIdentity ? 4 : 6,
        fill: 'rgba(0,0,0,0.1)', stroke: 'none', 'pointer-events': 'none',
      });
      g.appendChild(shadow);
      const rect = this.el('rect', {
        x: 0, y: 0, width: w, height: h, rx: isIdentity ? 4 : 6, ry: isIdentity ? 4 : 6,
        fill: color, opacity: isIdentity ? 0.7 : 0.92, stroke: 'none',
      });
      g.appendChild(rect);

      {
        const typeName = isIdentity ? this.shortType(node.type) : this.shortTypeName(node.type);
        const paramStr = isIdentity ? '' : this.shortLabel(node);
        const parts = [typeName, paramStr].filter(Boolean);
        const label = parts.join(', ');

        const textG = this.el('g', { transform: `translate(${w/2}, ${h/2}) rotate(-90)` });
        const txt = this.el('text', {
          x: 0, y: 0, 'text-anchor': 'middle', 'dominant-baseline': 'central',
          fill: '#fff', 'font-size': '8px', 'font-weight': isIdentity ? '400' : '600', 'font-family': "'Inter', 'SF Pro Text', system-ui, sans-serif",
        });
        txt.textContent = label;
        textG.appendChild(txt);
        g.appendChild(textG);
      }
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

    const x1 = from.x + from.w, y1 = from.y + from.h / 2;
    const x2 = to.x, y2 = to.y + to.h / 2;
    const dx = x2 - x1;

    let d;
    if (Math.abs(y2 - y1) < 3 && isMainEdge) {
      d = `M${x1},${y1} L${x2},${y2}`;
    } else if (isMainEdge) {
      const mx = x1 + dx / 2;
      d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    } else {
      let maxBottom = Math.max(from.y + from.h, to.y + to.h);
      const minX = Math.min(from.x, to.x);
      const maxX = Math.max(from.x + from.w, to.x + to.w);
      for (const p of Object.values(positions)) {
        if (p.x + p.w > minX && p.x < maxX) {
          maxBottom = Math.max(maxBottom, p.y + p.h);
        }
      }
      const dropY = maxBottom + 20 + Math.min(Math.abs(dx) * 0.03, 25);
      const r = 8;
      d = `M${x1},${y1} L${x1},${dropY - r} Q${x1},${dropY} ${x1 + r},${dropY} L${x2 - r},${dropY} Q${x2},${dropY} ${x2},${dropY - r} L${x2},${y2}`;
    }

    const path = this.el('path', {
      d, fill: 'none',
      stroke: isMainEdge ? '#78909c' : '#90a4ae',
      'stroke-width': isMainEdge ? 1.5 : 1.2,
      'marker-end': 'url(#arrow)',
      opacity: isMainEdge ? 0.8 : 0.55,
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
        const shape = g.querySelector('rect, circle, polygon');
        if (shape) {
          shape.setAttribute('opacity', '1');
          shape.setAttribute('stroke', '#fff');
          shape.setAttribute('stroke-width', '2');
        }
        const id = g.getAttribute('data-id');
        for (const edge of self.svg.querySelectorAll('.graph-edge')) {
          if (edge.getAttribute('data-from') === id || edge.getAttribute('data-to') === id) {
            edge.setAttribute('stroke-width', '3');
            edge.setAttribute('opacity', '1');
          }
        }
      });
      g.addEventListener('mouseleave', () => {
        const shape = g.querySelector('rect, circle, polygon');
        const node = graph.nodes.find(n => n.id === g.getAttribute('data-id'));
        const isId = node && IDENTITY_TYPES.has(node.type);
        if (shape) {
          shape.setAttribute('opacity', isId ? '0.7' : '0.92');
          shape.setAttribute('stroke', 'none');
        }
        for (const edge of self.svg.querySelectorAll('.graph-edge')) {
          const isMain = self._mainEdges && self._mainEdges.has(edge.getAttribute('data-from') + ':' + edge.getAttribute('data-to'));
          edge.setAttribute('stroke-width', isMain ? '1.5' : '1.2');
          edge.setAttribute('opacity', isMain ? '0.8' : '0.55');
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
      html += '<div class="info-section"><div class="info-section-title">Config</div>';
      for (const k of cfgKeys) {
        const v = cfg[k];
        html += '<div class="info-row"><span class="info-label">' + k + '</span>';
        if (node.defLine > 0 && editable.has(k)) {
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

    const matchLine = (node.srcLine || 0) - 1;
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
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = `
      text { font-family: system-ui, -apple-system, sans-serif; }
      .graph-node { cursor: pointer; }
    `;
    clone.insertBefore(style, clone.firstChild);
    const serializer = new XMLSerializer();
    return serializer.serializeToString(clone);
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

  addZoomPan() {
    const svg = this.svg;
    let isPanning = false, startX = 0, startY = 0, startVB = null;

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
    const marker = this.el('marker', { id: 'arrow', viewBox: '0 0 10 6', refX: 10, refY: 3, markerWidth: 8, markerHeight: 6, orient: 'auto-start-reverse' });
    const poly = this.el('polygon', { points: '0 0, 10 3, 0 6', fill: '#78909c' });
    marker.appendChild(poly);
    defs.appendChild(marker);
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
