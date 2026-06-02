#!/usr/bin/env python3
"""Build app.html from src/*.js + style.css — single self-contained file."""
import re, json, sys

# Load dagre first (no export stripping needed)
with open('src/dagre.min.js') as fh:
    dagre_code = fh.read()

files = ['src/tokenizer.js', 'src/parser.js', 'src/interpreter.js', 'src/renderer.js']
parts = [dagre_code]
for f in files:
    with open(f) as fh:
        code = re.sub(r'^export ', '', fh.read(), flags=re.MULTILINE)
    parts.append(code)
bundle = '\n'.join(parts)

with open('style.css') as f:
    css = f.read()

examples = {}

examples["SimpleUNet"] = {"shape": "1, 256, 256, 3", "code": """\
import tensorflow as tf
from tensorflow.keras import layers, Model

class SimpleUNet(Model):
    def __init__(self, base_filters=32, num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        self.enc_convs = []
        self.enc_bns = []
        self.enc_pools = []
        for i in range(num_stages - 1):
            filters = base_filters * (2 ** i)
            self.enc_convs.append(layers.Conv2D(filters, 3, padding='same'))
            self.enc_bns.append(layers.BatchNormalization())
            self.enc_pools.append(layers.MaxPooling2D(2))
        self.bottleneck = layers.Conv2D(base_filters * (2 ** (num_stages - 1)), 3, padding='same')
        self.bottleneck_bn = layers.BatchNormalization()
        self.dec_ups = []
        self.dec_convs = []
        self.dec_bns = []
        for i in range(num_stages - 1):
            filters = base_filters * (2 ** (num_stages - 2 - i))
            self.dec_ups.append(layers.UpSampling2D(2))
            self.dec_convs.append(layers.Conv2D(filters, 3, padding='same'))
            self.dec_bns.append(layers.BatchNormalization())
        self.final_conv = layers.Conv2D(3, 1, padding='same')

    def call(self, x, training=False):
        skips = []
        for i in range(self.num_stages - 1):
            x = self.enc_convs[i](x)
            x = self.enc_bns[i](x)
            skips.append(x)
            x = self.enc_pools[i](x)
        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        for i in range(self.num_stages - 1):
            x = self.dec_ups[i](x)
            x = tf.concat([x, skips[self.num_stages - 2 - i]], axis=-1)
            x = self.dec_convs[i](x)
            x = self.dec_bns[i](x)
        x = self.final_conv(x)
        return x

model = SimpleUNet(base_filters=32, num_stages=4)
"""}

examples["MiniResNet"] = {"shape": "1, 32, 32, 3", "code": """\
import tensorflow as tf
from tensorflow.keras import layers, Model

class ResBlock(Model):
    def __init__(self, filters, strides=(1, 1)):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.use_proj = strides != (1, 1)
        if self.use_proj:
            self.proj = layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False)
            self.proj_bn = layers.BatchNormalization()
        self.add = layers.Add()
        self.relu2 = layers.ReLU()

    def call(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.proj_bn(self.proj(x)) if self.use_proj else x
        return self.relu2(self.add([out, shortcut]))

class MiniResNet(Model):
    def __init__(self):
        super().__init__()
        self.stem = layers.Conv2D(16, 3, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(32, strides=(2, 2))
        self.block3 = ResBlock(64, strides=(2, 2))
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(10)

    def call(self, x):
        x = self.relu(self.bn(self.stem(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.fc(self.gap(x))

model = MiniResNet()
"""}

examples["AutoEncoder"] = {"shape": "1, 32, 32, 3", "code": """\
import tensorflow as tf
from tensorflow.keras import layers, Model

class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = layers.Conv2D(32, 3, strides=(2, 2), padding='same')
        self.enc_bn1 = layers.BatchNormalization()
        self.enc_relu1 = layers.ReLU()
        self.enc_conv2 = layers.Conv2D(64, 3, strides=(2, 2), padding='same')
        self.enc_bn2 = layers.BatchNormalization()
        self.enc_relu2 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.enc_dense = layers.Dense(128)
        self.enc_relu3 = layers.ReLU()
        self.dec_dense = layers.Dense(8 * 8 * 64)
        self.dec_relu = layers.ReLU()
        self.reshape = layers.Reshape((8, 8, 64))
        self.dec_up1 = layers.UpSampling2D(2)
        self.dec_conv1 = layers.Conv2D(32, 3, padding='same')
        self.dec_bn1 = layers.BatchNormalization()
        self.dec_relu1 = layers.ReLU()
        self.dec_up2 = layers.UpSampling2D(2)
        self.dec_conv2 = layers.Conv2D(3, 3, padding='same')

    def call(self, x):
        x = self.enc_relu1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_relu2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_relu3(self.enc_dense(self.flatten(x)))
        x = self.dec_relu(self.dec_dense(x))
        x = self.reshape(x)
        x = self.dec_relu1(self.dec_bn1(self.dec_conv1(self.dec_up1(x))))
        return self.dec_conv2(self.dec_up2(x))

model = AutoEncoder()
"""}

examples["MobileNet-style"] = {"shape": "1, 32, 32, 3", "code": """\
import tensorflow as tf
from tensorflow.keras import layers, Model

class DWSepBlock(Model):
    def __init__(self, filters, strides=(1, 1)):
        super().__init__()
        self.dw = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pw = layers.Conv2D(filters, 1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, x):
        x = self.relu1(self.bn1(self.dw(x)))
        return self.relu2(self.bn2(self.pw(x)))

class MobileNet(Model):
    def __init__(self):
        super().__init__()
        self.stem = layers.Conv2D(16, 3, padding='same', use_bias=False)
        self.stem_bn = layers.BatchNormalization()
        self.stem_relu = layers.ReLU()
        self.b1 = DWSepBlock(16)
        self.b2 = DWSepBlock(32, strides=(2, 2))
        self.b3 = DWSepBlock(64, strides=(2, 2))
        self.b4 = DWSepBlock(128, strides=(2, 2))
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(10)

    def call(self, x):
        x = self.stem_relu(self.stem_bn(self.stem(x)))
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.fc(self.gap(x))

model = MobileNet()
"""}

# For backward compat
example = examples["SimpleUNet"]["code"]

app_js = """
var EXAMPLES = """ + json.dumps(examples) + """;
var currentRenderer = null;
var debounceTimer = null;
function doAnalyze() {
  var code = document.getElementById('code-editor').value;
  var shapeStr = document.getElementById('input-shape').value;
  var graphPanel = document.getElementById('graph-panel');
  var warningsEl = document.getElementById('warnings');
  var inputShape;
  try { inputShape = JSON.parse('[' + shapeStr + ']'); }
  catch(e) { inputShape = [1, 256, 256, 3]; }
  try {
    var tokens = new Tokenizer(code).tokenize();
    var ast = new Parser(tokens).parseModule();
    var interp = new Interpreter();
    var graph = interp.analyze(ast, inputShape);
    warningsEl.textContent = graph.nodes.length + ' nodes, ' + graph.edges.length + ' edges';
    if (interp.warnings.length > 0) warningsEl.textContent += ' | ' + interp.warnings.slice(0, 5).join(' | ');
    currentRenderer = new GraphRenderer(graphPanel);
    currentRenderer.render(graph);
    if (window._selectedNodeId) {
      var sel = graph.nodes.find(function(n) { return n.id === window._selectedNodeId; });
      if (sel) {
        currentRenderer.showNodeDetail(sel);
        currentRenderer.highlightCode(sel);
      }
    }
  } catch (e) {
    warningsEl.textContent = 'Error: ' + e.message;
    graphPanel.innerHTML = '<div class="empty-state">Analysis failed: ' + e.message + '</div>';
  }
}
async function doExport() {
  if (!currentRenderer) return;
  var svgString = await currentRenderer.exportSVGWithFonts();
  if (!svgString) return;
  var blob = new Blob([svgString], { type: 'image/svg+xml' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'model-graph.svg';
  a.click();
}
async function doExportPDF() {
  if (!currentRenderer) return;
  var svgString = await currentRenderer.exportSVGWithFonts();
  if (!svgString) return;
  var w = currentRenderer.naturalW, h = currentRenderer.naturalH;
  var margin = 10;
  var pdfW = w * 0.75 + margin * 2, pdfH = h * 0.75 + margin * 2;
  var orientation = pdfW > pdfH ? 'landscape' : 'portrait';
  var pdf = new jspdf.jsPDF({ orientation: orientation, unit: 'pt', format: [pdfW, pdfH] });
  var el = document.createElement('div');
  el.innerHTML = svgString;
  var svg = el.querySelector('svg');
  svg.setAttribute('width', (pdfW - margin * 2) + 'pt');
  svg.setAttribute('height', (pdfH - margin * 2) + 'pt');
  pdf.svg(svg, { x: margin, y: margin, width: pdfW - margin * 2, height: pdfH - margin * 2 }).then(function() {
    pdf.save('model-graph.pdf');
  });
}
function loadExample(name) {
  var ex = EXAMPLES[name || document.getElementById('example-select').value];
  if (!ex) return;
  var ed = document.getElementById('code-editor');
  ed.value = ex.code;
  document.getElementById('input-shape').value = ex.shape;
  updateHighlight();
  syncEditorSize();
  doAnalyze();
}
function updateSourceValue(node, field, rawValue) {
  var ed = document.getElementById('code-editor');
  var lines = ed.value.split('\\n');
  var lineIdx = ((node.defLine || node.srcLine || 0)) - 1;
  if (lineIdx < 0 || lineIdx >= lines.length) return;
  var line = lines[lineIdx];
  // Try keyword arg: field = value or field=value
  var kwRe = new RegExp('(\\\\b' + field + '\\\\s*=\\\\s*)(' +
    "'[^']*'" + '|"[^"]*"|\\\\([^)]*\\\\)|\\\\[[^\\\\]]*\\\\]|[\\\\w.]+)');
  var m = line.match(kwRe);
  if (m) {
    var pyVal = formatPyVal(field, rawValue, m[2]);
    lines[lineIdx] = line.substring(0, m.index) + m[1] + pyVal + line.substring(m.index + m[0].length);
    ed.value = lines.join('\\n');
    updateHighlight(); syncEditorSize();
    window._hlDefLine = true;
    doAnalyze();
    return;
  }
  // Positional fallback: find the Nth arg in the call parens
  var posMap = {
    Conv2D:['filters','kernel_size','strides','padding'], Conv1D:['filters','kernel_size','strides','padding'],
    Conv2DTranspose:['filters','kernel_size','strides','padding'],
    DepthwiseConv2D:['kernel_size','strides','padding'], SeparableConv2D:['filters','kernel_size','strides','padding'],
    Dense:['units','activation'], MaxPooling2D:['pool_size','strides','padding'],
    UpSampling2D:['size'], Dropout:['rate'], Embedding:['input_dim','output_dim'],
    LSTM:['units'], GRU:['units'], PReLU:['shared_axes'], LeakyReLU:['alpha'],
  };
  var positionals = posMap[node.type] || [];
  var posIdx = positionals.indexOf(field);
  if (posIdx < 0) return;
  var paren = line.indexOf('(');
  if (paren < 0) return;
  var args = splitCallArgs(line, paren + 1);
  if (posIdx < args.length && args[posIdx].text.indexOf('=') < 0) {
    var pyVal = formatPyVal(field, rawValue, args[posIdx].text.trim());
    lines[lineIdx] = line.substring(0, args[posIdx].start) + pyVal + line.substring(args[posIdx].end);
    ed.value = lines.join('\\n');
    updateHighlight(); syncEditorSize();
    window._hlDefLine = true;
    doAnalyze();
  }
}
function formatPyVal(field, raw, old) {
  if (field === 'padding' || field === 'activation') return "'" + raw + "'";
  if (raw === 'True' || raw === 'False') return raw;
  // Detect tuple: old was (x, y) or raw has commas
  if (old && old.trim().charAt(0) === '(') return '(' + raw + ')';
  if (String(raw).indexOf(',') >= 0) return '(' + raw + ')';
  return String(raw);
}
function splitCallArgs(line, start) {
  var depth = 1, args = [], cur = start, i = start;
  while (i < line.length && depth > 0) {
    var ch = line[i];
    if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') { depth--; if (depth === 0) break; }
    else if (ch === ',' && depth === 1) { args.push({text: line.substring(cur, i), start: cur, end: i}); cur = i + 1; }
    else if (ch === "'" || ch === '"') { i++; while (i < line.length && line[i] !== ch) i++; }
    i++;
  }
  if (cur < i) args.push({text: line.substring(cur, i), start: cur, end: i});
  return args;
}
document.getElementById('copy-code').addEventListener('click', function() {
  var btn = this;
  navigator.clipboard.writeText(document.getElementById('code-editor').value).then(function() {
    btn.classList.add('copied');
    setTimeout(function() { btn.classList.remove('copied'); }, 1200);
  });
});
var editor = document.getElementById('code-editor');
var _sel = document.getElementById('example-select');
Object.keys(EXAMPLES).forEach(function(k, i) {
  var opt = document.createElement('option');
  opt.value = k; opt.textContent = k;
  if (i === 0) opt.selected = true;
  _sel.appendChild(opt);
});
editor.value = EXAMPLES[_sel.value].code;
document.getElementById('input-shape').value = EXAMPLES[_sel.value].shape;
editor.oninput = function() { clearTimeout(debounceTimer); debounceTimer = setTimeout(doAnalyze, 1200); };
editor.onkeydown = function(e) {
  if (e.key === 'Tab') { e.preventDefault(); var s = editor.selectionStart, end = editor.selectionEnd; editor.value = editor.value.substring(0, s) + '    ' + editor.value.substring(end); editor.selectionStart = editor.selectionEnd = s + 4; updateHighlight(); syncEditorSize(); }
};
doAnalyze();

// View toggle
document.querySelectorAll('.view-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.view-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    if (currentRenderer) currentRenderer.applyView(btn.dataset.view);
  });
});

// Resize handles
document.querySelectorAll('.resize-handle').forEach(function(handle) {
  var leftId = handle.getAttribute('data-left');
  var rightId = handle.getAttribute('data-right');
  var leftEl = document.getElementById(leftId);
  var rightEl = document.getElementById(rightId);
  if (!leftEl || !rightEl) return;
  var dragging = false, startX = 0, startLW = 0, startRW = 0;
  handle.addEventListener('mousedown', function(e) {
    dragging = true; startX = e.clientX;
    startLW = leftEl.getBoundingClientRect().width;
    startRW = rightEl.getBoundingClientRect().width;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });
  window.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    var dx = e.clientX - startX;
    var newL = Math.max(150, startLW + dx);
    var newR = Math.max(150, startRW - dx);
    leftEl.style.flex = '0 0 ' + newL + 'px';
    rightEl.style.flex = '0 0 ' + newR + 'px';
  });
  window.addEventListener('mouseup', function() {
    if (dragging) { dragging = false; document.body.style.cursor = ''; document.body.style.userSelect = ''; }
  });
});

// Python syntax highlighting (tokenizer approach)
""" + r"""
var _kwSet = new Set('def,class,return,if,elif,else,for,in,while,break,continue,pass,import,from,as,with,try,except,finally,raise,assert,yield,lambda,global,nonlocal,del,and,or,not,is'.split(','));
var _biSet = new Set('super,range,len,int,float,max,min,print,list,tuple,dict,set,type,isinstance,enumerate,zip,reversed,sum,abs,sorted'.split(','));
var _constSet = new Set(['None','True','False']);
function highlightPython(code) {
  var out = '', i = 0, n = code.length;
  function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  while (i < n) {
    var ch = code[i];
    // Comment
    if (ch === '#') {
      var end = code.indexOf('\n', i); if (end < 0) end = n;
      out += '<span class="hl-comment">' + esc(code.slice(i, end)) + '</span>';
      i = end; continue;
    }
    // String
    if (ch === '"' || ch === "'") {
      var q = ch, j = i + 1, triple = false;
      if (code[j] === q && code[j+1] === q) { triple = true; j += 2; }
      while (j < n) {
        if (code[j] === '\\') { j += 2; continue; }
        if (triple && code[j] === q && code[j+1] === q && code[j+2] === q) { j += 3; break; }
        if (!triple && code[j] === q) { j += 1; break; }
        j++;
      }
      out += '<span class="hl-string">' + esc(code.slice(i, j)) + '</span>';
      i = j; continue;
    }
    // Decorator
    if (ch === '@' && (i === 0 || code[i-1] === '\n')) {
      var j = i + 1; while (j < n && /\w/.test(code[j])) j++;
      out += '<span class="hl-decorator">' + esc(code.slice(i, j)) + '</span>';
      i = j; continue;
    }
    // Word
    if (/[a-zA-Z_]/.test(ch)) {
      var j = i + 1; while (j < n && /[a-zA-Z0-9_]/.test(code[j])) j++;
      var w = code.slice(i, j);
      if (_kwSet.has(w)) out += '<span class="hl-keyword">' + w + '</span>';
      else if (_constSet.has(w)) out += '<span class="hl-keyword">' + w + '</span>';
      else if (w === 'self') out += '<span class="hl-self">' + w + '</span>';
      else if (_biSet.has(w)) out += '<span class="hl-builtin">' + w + '</span>';
      else out += esc(w);
      i = j; continue;
    }
    // Number
    if (/[0-9]/.test(ch)) {
      var j = i; while (j < n && /[0-9.eE+\-]/.test(code[j])) j++;
      out += '<span class="hl-number">' + esc(code.slice(i, j)) + '</span>';
      i = j; continue;
    }
    out += esc(ch); i++;
  }
  return out;
}
var highlightEl = document.getElementById('code-highlight');
var lineNumEl = document.getElementById('line-numbers');
var editorScroll = document.getElementById('editor-scroll');
function updateHighlight() {
  if (highlightEl) highlightEl.innerHTML = highlightPython(editor.value) + '\n';
  updateLineNumbers();
}
function updateLineNumbers() {
  if (!lineNumEl) return;
  var n = editor.value.split('\n').length;
  var html = '';
  for (var i = 1; i <= n; i++) html += '<span>' + i + '</span>';
  lineNumEl.innerHTML = html;
}
function syncEditorSize() {
  editor.style.height = editor.scrollHeight + 'px';
  if (highlightEl) highlightEl.style.height = editor.scrollHeight + 'px';
}
editor.addEventListener('input', function() {
  updateHighlight();
  syncEditorSize();
});
editorScroll.addEventListener('scroll', function() {
  lineNumEl.scrollTop = editorScroll.scrollTop;
});
updateHighlight();
syncEditorSize();
""" + """
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<title>net.sketch</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.4/dist/svg2pdf.umd.min.js"></script>
<style>
{css}
</style>
</head>
<body>
<header>
<h1>net<span>.sketch</span></h1>
<div class="header-actions">
  <select id="example-select" onchange="loadExample()"><option value="" disabled>Load example...</option></select>
  <button class="primary" onclick="doAnalyze()">Analyze</button>
  <button onclick="doExport()">Export SVG</button>
  <button onclick="doExportPDF()">Export PDF</button>
</div>
</header>
<main>
<div class="panel editor-panel" id="editor-panel">
  <div class="panel-header">Python Code<button class="copy-btn" id="copy-code" title="Copy code"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button></div>
  <div class="editor-container">
    <div class="line-numbers" id="line-numbers"></div>
    <div class="editor-scroll" id="editor-scroll">
      <pre id="code-highlight" aria-hidden="true"></pre>
      <textarea id="code-editor" spellcheck="false" autocomplete="off"></textarea>
    </div>
  </div>
  <div id="warnings"></div>
  <div class="editor-footer">
    <label for="input-shape">Input shape:</label>
    <input id="input-shape" type="text" value="1, 256, 256, 3" placeholder="batch, H, W, C">
  </div>
</div>
<div class="resize-handle" data-left="editor-panel" data-right="graph-wrapper"></div>
<div class="panel graph-panel-wrapper" id="graph-wrapper">
  <div class="panel-header">Model Graph
    <div class="view-toggle">
      <button class="view-btn active" data-view="type" title="Color by layer type">Type</button>
      <button class="view-btn" data-view="macs" title="MACs heatmap">MACs</button>
      <button class="view-btn" data-view="params" title="Parameter count heatmap">Params</button>
    </div>
  </div>
  <div id="graph-panel"></div>
</div>
<div class="resize-handle" data-left="graph-wrapper" data-right="info-wrapper"></div>
<div class="panel info-panel" id="info-wrapper">
  <div class="panel-header">Details</div>
  <div id="info-panel">
    <div class="info-empty">Click a node to inspect</div>
  </div>
</div>
</main>
<script>
{bundle}
{app_js}
</script>
</body>
</html>"""

with open('app.html', 'w') as f:
    f.write(html)

# Also write as index.html for GitHub Pages
with open('index.html', 'w') as f:
    f.write(html)

print(f'Built app.html + index.html ({len(html)} chars)')
