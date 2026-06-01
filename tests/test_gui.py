#!/usr/bin/env python3.11
"""
Test graph topology coherence.
Verifies that the generated graph makes sense: edges connect correctly,
main path exists, branch nodes connect back, no orphans.

Usage: python3.11 tests/test_gui.py
"""
import os, sys, re, json, subprocess

G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; D = '\033[90m'; B = '\033[1m'; N = '\033[0m'

def build_bundle():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for f in ['src/tokenizer.js', 'src/parser.js', 'src/interpreter.js', 'src/renderer.js']:
        with open(os.path.join(root, f)) as fh:
            code = re.sub(r'^export ', '', fh.read(), flags=re.MULTILINE)
        parts.append(code)
    return '\n'.join(parts)

JS_BUNDLE = build_bundle()

def run_js(code, input_shape):
    js = JS_BUNDLE + '\n'
    js += f'var code = {json.dumps(code)};\n'
    js += f'var inputShape = {json.dumps(input_shape)};\n'
    js += """var tokens = new Tokenizer(code).tokenize();
var ast = new Parser(tokens).parseModule();
var interp = new Interpreter();
var graph = interp.analyze(ast, inputShape);
var nodes = [];
for (var i = 0; i < graph.nodes.length; i++) {
  var n = graph.nodes[i];
  nodes.push({id: n.id, type: n.type, shape: n.outputShape, params: n.params || 0, label: n.label || ''});
}
var edges = [];
for (var i = 0; i < graph.edges.length; i++) {
  edges.push({from: graph.edges[i].from, to: graph.edges[i].to, label: graph.edges[i].label || ''});
}

// Run layout + skip routing (renderer needs a fake container)
var positions = null;
var mainPath = null;
try {
  var fakeContainer = {innerHTML:'', appendChild:function(){}, querySelectorAll:function(){return[]}};
  var r = new GraphRenderer(fakeContainer);
  var mg = r.mergeActivations(graph);
  var layout = r.computeLayout(mg);
  r.precomputeSkipRoutes(mg, layout.positions);
  positions = {};
  for (var id in layout.positions) {
    var p = layout.positions[id];
    positions[id] = {x: Math.round(p.x), y: Math.round(p.y), w: Math.round(p.w), h: Math.round(p.h), branch: !!p.branch};
  }
  mainPath = r._mainPath ? Array.from(r._mainPath) : [];
} catch(e) {}

JSON.stringify({nodes: nodes, edges: edges, warnings: interp.warnings, positions: positions, mainPath: mainPath});
"""
    try:
        out = subprocess.run(['osascript', '-l', 'JavaScript', '-e', js],
                           capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None, out.stderr.strip()[:200]
        return json.loads(out.stdout.strip()), None
    except Exception as e:
        return None, str(e)[:200]


def check_topology(name, graph):
    """Check graph topology coherence. Returns list of (pass, message) tuples."""
    checks = []
    nodes = graph['nodes']
    edges = graph['edges']
    node_ids = {n['id'] for n in nodes}
    node_map = {n['id']: n for n in nodes}

    # 1. All edges reference existing nodes
    bad_edges = [e for e in edges if e['from'] not in node_ids or e['to'] not in node_ids]
    checks.append((len(bad_edges) == 0, f"edges valid: {len(bad_edges)} dangling"))

    # 2. Input and Output exist
    input_nodes = [n for n in nodes if n['type'] == 'Input']
    output_nodes = [n for n in nodes if n['type'] == 'Output']
    checks.append((len(input_nodes) == 1, f"1 Input node (got {len(input_nodes)})"))
    checks.append((len(output_nodes) == 1, f"1 Output node (got {len(output_nodes)})"))

    # 3. No orphan nodes (every non-Input node has at least 1 input edge)
    in_degree = {n['id']: 0 for n in nodes}
    out_degree = {n['id']: 0 for n in nodes}
    for e in edges:
        in_degree[e['to']] = in_degree.get(e['to'], 0) + 1
        out_degree[e['from']] = out_degree.get(e['from'], 0) + 1
    orphans = [nid for nid, deg in in_degree.items() if deg == 0 and node_map[nid]['type'] != 'Input']
    checks.append((len(orphans) == 0, f"no orphans: {[node_map[o]['type'] for o in orphans]}" if orphans else "no orphans"))

    # 4. No dead ends (every non-Output node has at least 1 output edge)
    dead = [nid for nid, deg in out_degree.items() if deg == 0 and node_map[nid]['type'] != 'Output']
    checks.append((len(dead) == 0, f"no dead ends: {[node_map[d]['type'] for d in dead]}" if dead else "no dead ends"))

    # 5. Path exists from Input to Output
    if input_nodes and output_nodes:
        adj = {}
        for e in edges:
            adj.setdefault(e['from'], []).append(e['to'])
        visited = set()
        stack = [input_nodes[0]['id']]
        while stack:
            nid = stack.pop()
            if nid in visited: continue
            visited.add(nid)
            for c in adj.get(nid, []):
                stack.append(c)
        reachable = output_nodes[0]['id'] in visited
        checks.append((reachable, "Input→Output path exists" if reachable else "NO path Input→Output"))

    # 6. Shape consistency: for each edge, from.shape and to.shape should be compatible
    shape_issues = []
    for e in edges:
        fn = node_map.get(e['from'])
        tn = node_map.get(e['to'])
        if not fn or not tn: continue
        fs, ts = fn.get('shape'), tn.get('shape')
        if not fs or not ts: continue
        # Check batch dim matches
        if len(fs) > 0 and len(ts) > 0 and fs[0] is not None and ts[0] is not None and fs[0] != ts[0]:
            shape_issues.append(f"{fn['type']}→{tn['type']}: batch {fs[0]}≠{ts[0]}")
    checks.append((len(shape_issues) == 0, f"shapes ok" if not shape_issues else f"shape issues: {shape_issues[:3]}"))

    # 7. Multi-input nodes (Add, Concatenate) have ≥2 inputs
    for n in nodes:
        if n['type'] in ('Add', 'Subtract', 'Multiply', 'Concatenate'):
            in_count = sum(1 for e in edges if e['to'] == n['id'])
            checks.append((in_count >= 2, f"{n['type']}({n['id']}) has {in_count} inputs (need ≥2)"))

    # ─── Visual checks (if positions available) ────────
    positions = graph.get('positions')
    if positions:
        # 8. No node overlaps (bounding boxes don't intersect for ANY pair)
        all_positioned = [(nid, positions[nid]) for nid in positions]
        overlaps = []
        for i, (id1, p1) in enumerate(all_positioned):
            for j, (id2, p2) in enumerate(all_positioned):
                if j <= i: continue
                margin = 2
                x_overlap = p1['x'] + margin < p2['x'] + p2['w'] - margin and p2['x'] + margin < p1['x'] + p1['w'] - margin
                y_overlap = p1['y'] + margin < p2['y'] + p2['h'] - margin and p2['y'] + margin < p1['y'] + p1['h'] - margin
                if x_overlap and y_overlap:
                    t1 = node_map.get(id1, {}).get('type', '?')
                    t2 = node_map.get(id2, {}).get('type', '?')
                    overlaps.append(f"{t1}({id1})↔{t2}({id2})")
        checks.append((len(overlaps) == 0, f"no overlaps" if not overlaps else f"{len(overlaps)} overlaps: {overlaps[:5]}"))

        # 9. Main path nodes in left-to-right X order
        main_ordered = [(nid, positions[nid]['x']) for nid in graph.get('mainPath', []) if nid in positions and not positions[nid].get('branch')]
        x_sorted = all(main_ordered[i][1] <= main_ordered[i+1][1] for i in range(len(main_ordered)-1))
        checks.append((x_sorted, "main path L→R order" if x_sorted else "main path NOT in L→R order"))

        # 10. Branch nodes below main path
        main_ids_set = set(graph.get('mainPath', []))
        main_nodes = [(nid, positions[nid]) for nid in main_ids_set if nid in positions]
        if main_nodes:
            main_center_y = sum(p['y'] + p['h']/2 for _, p in main_nodes) / len(main_nodes)
            branch_positions = [(nid, positions[nid]) for nid in positions if positions[nid].get('branch')]
            branch_ok = all(p['y'] + p['h']/2 > main_center_y for _, p in branch_positions)
            if branch_positions:
                checks.append((branch_ok, f"branch nodes below main ({len(branch_positions)} branches)"))

        # 11. All nodes have positive dimensions
        zero_dims = [nid for nid, p in positions.items() if p['w'] <= 0 or p['h'] <= 0]
        checks.append((len(zero_dims) == 0, "all nodes have size" if not zero_dims else f"zero-dim nodes: {zero_dims}"))

        # 12. All nodes within reasonable bounds
        max_x = max(p['x'] + p['w'] for p in positions.values())
        max_y = max(p['y'] + p['h'] for p in positions.values())
        min_x = min(p['x'] for p in positions.values())
        min_y = min(p['y'] for p in positions.values())
        bounds_ok = min_x >= -10 and min_y >= -10 and max_x < 20000 and max_y < 5000
        checks.append((bounds_ok, f"bounds ok ({int(max_x)}×{int(max_y)})" if bounds_ok else f"bounds bad: x=[{int(min_x)},{int(max_x)}] y=[{int(min_y)},{int(max_y)}]"))

    return checks


# ─── Test cases ─────────────────────────────────────────

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TESTS = {}

# ProgDownLite
with open(os.path.join(root, 'tests/models/progdownlite.py')) as f:
    TESTS['ProgDownLite'] = {
        'code': f.read() + "\nmodel = ProgDownLite(ratio=2)\n",
        'input': [1, 256, 256, 3],
        'expect': {
            'node_count': 16,
            'edge_count': 17,
            'main_path_types': ['Input', 'Conv2D', 'Conv2D', 'Conv2D', 'Conv2D', 'Conv2D', 'Add', 'Conv2D', 'Resize', 'Conv2D', 'Conv2D', 'Conv2D', 'Conv2D', 'Add', 'Output'],
            'branch_nodes': {'Resize': 1},  # The ResizeBicubic on the skip
        }
    }

# Simple U-Net
TESTS['U-Net skip'] = {
    'code': """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.enc = layers.Conv2D(16, 3, padding='same')
        self.pool = layers.MaxPooling2D(2)
        self.btl = layers.Conv2D(32, 3, padding='same')
        self.up = layers.UpSampling2D(2)
        self.dec = layers.Conv2D(16, 3, padding='same')
        self.out_conv = layers.Conv2D(1, 1, padding='same')
    def call(self, x):
        e = self.enc(x)
        x = self.pool(e)
        x = self.btl(x)
        x = self.up(x)
        x = tf.concat([x, e], axis=-1)
        x = self.dec(x)
        x = self.out_conv(x)
        return x
model = M()
""",
    'input': [1, 32, 32, 3],
    'expect': {
        'node_count': 9,
        'edge_count': 9,
    }
}

# Residual
TESTS['Residual'] = {
    'code': """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.c = layers.Conv2D(3, 3, padding='same')
    def call(self, x): return x + self.c(x)
model = M()
""",
    'input': [1, 8, 8, 3],
    'expect': {
        'node_count': 4,
        'edge_count': 4,
    }
}

# LoResNet
with open(os.path.join(root, 'tests/models/loresnet.py')) as f:
    TESTS['LoResNet_CPU'] = {
        'code': f.read() + "\nmodel = LoResNet_CPU(_ratio=0.5, base_filters=16, num_stages=3)\n",
        'input': [1, 256, 256, 3],
        'expect': {}
    }


# ─── Run ────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print(f" net.sketch.io — Graph topology coherence tests")
    print("=" * 65)

    passed, failed = 0, 0

    for name, test in TESTS.items():
        print(f"\n  {B}{name}{N}")
        graph, err = run_js(test['code'], test['input'])
        if err:
            print(f"    {R}ERROR{N}: {err}")
            failed += 1
            continue

        # Topology checks
        checks = check_topology(name, graph)
        for ok, msg in checks:
            if ok:
                print(f"    {G}✓{N} {msg}")
                passed += 1
            else:
                print(f"    {R}✗{N} {msg}")
                failed += 1

        # Expected counts
        exp = test.get('expect', {})
        if 'node_count' in exp:
            nc = len(graph['nodes'])
            ok = nc == exp['node_count']
            print(f"    {'✓' if ok else '✗'} nodes: {nc} (expect {exp['node_count']})")
            if ok: passed += 1
            else: failed += 1

        if 'edge_count' in exp:
            ec = len(graph['edges'])
            ok = ec == exp['edge_count']
            print(f"    {'✓' if ok else '✗'} edges: {ec} (expect {exp['edge_count']})")
            if ok: passed += 1
            else: failed += 1

        # Main path check
        if 'main_path_types' in exp:
            adj = {}
            for e in graph['edges']:
                adj.setdefault(e['from'], []).append(e['to'])
            node_map = {n['id']: n for n in graph['nodes']}

            # Find longest path
            def longest(nid, vis=None):
                if vis is None: vis = set()
                if nid in vis: return []
                vis.add(nid)
                best = []
                for c in adj.get(nid, []):
                    p = longest(c, set(vis))
                    if len(p) > len(best): best = p
                return [nid] + best

            inp = [n for n in graph['nodes'] if n['type'] == 'Input']
            if inp:
                path = longest(inp[0]['id'])
                path_types = [node_map[nid]['type'] for nid in path]
                ok = path_types == exp['main_path_types']
                if ok:
                    print(f"    {G}✓{N} main path: {len(path)} nodes")
                else:
                    print(f"    {R}✗{N} main path mismatch:")
                    print(f"      {D}got:    {path_types}{N}")
                    print(f"      {D}expect: {exp['main_path_types']}{N}")
                if ok: passed += 1
                else: failed += 1

        # Branch node check
        if 'branch_nodes' in exp:
            inp = [n for n in graph['nodes'] if n['type'] == 'Input']
            if inp:
                path_ids = set(longest(inp[0]['id']))
                branch = [n for n in graph['nodes'] if n['id'] not in path_ids and n['type'] not in ('Input', 'Output')]
                branch_types = {}
                for n in branch:
                    branch_types[n['type']] = branch_types.get(n['type'], 0) + 1
                ok = branch_types == exp['branch_nodes']
                if ok:
                    print(f"    {G}✓{N} branch nodes: {branch_types}")
                else:
                    print(f"    {R}✗{N} branch nodes: got {branch_types}, expect {exp['branch_nodes']}")
                if ok: passed += 1
                else: failed += 1

        # Show warnings
        warnings = graph.get('warnings', [])
        if warnings:
            print(f"    {Y}⚠ {len(warnings)} warnings{N}")

    print("\n" + "─" * 65)
    total = passed + failed
    if failed == 0:
        print(f"  {G}{B}ALL PASSED{N}  {passed}/{total} checks")
    else:
        print(f"  {R}{B}{failed} FAILED{N}  {passed}/{total} checks")
    print("─" * 65)
    sys.exit(1 if failed else 0)
