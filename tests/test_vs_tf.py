#!/usr/bin/env python3.11
"""
Non-regression: TF ground truth vs JS static interpreter.
Run: python3.11 tests/test_vs_tf.py

Uses osascript -l JavaScript to run the JS interpreter (no Node.js needed).
"""
import os, sys, json, subprocess, re, textwrap

# ─── Build JS bundle ────────────────────────────────────
def build_bundle():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for f in ['src/tokenizer.js', 'src/parser.js', 'src/interpreter.js']:
        with open(os.path.join(root, f)) as fh:
            code = re.sub(r'^export ', '', fh.read(), flags=re.MULTILINE)
        parts.append(code)
    return '\n'.join(parts)

JS_BUNDLE = build_bundle()

def run_js(code, input_shape):
    """Run tokenizer→parser→interpreter on Python code string, return {params, output_shape}"""
    js = JS_BUNDLE + '\n'
    js += f'var code = {json.dumps(code)};\n'
    js += f'var inputShape = {json.dumps(input_shape)};\n'
    js += textwrap.dedent("""\
        var tokens = new Tokenizer(code).tokenize();
        var ast = new Parser(tokens).parseModule();
        var interp = new Interpreter();
        var graph = interp.analyze(ast, inputShape);
        var totalParams = 0;
        for (var i = 0; i < graph.nodes.length; i++) totalParams += (graph.nodes[i].params || 0);
        var outNode = null;
        for (var i = 0; i < graph.nodes.length; i++) { if (graph.nodes[i].type === 'Output') outNode = graph.nodes[i]; }
        var typeCounts = {};
        for (var i = 0; i < graph.nodes.length; i++) {
            var t = graph.nodes[i].type;
            typeCounts[t] = (typeCounts[t] || 0) + 1;
        }
        var perNodeShapes = [];
        for (var i = 0; i < graph.nodes.length; i++) {
            perNodeShapes.push({type: graph.nodes[i].type, shape: graph.nodes[i].outputShape, params: graph.nodes[i].params || 0});
        }
        var nodeOrder = [];
        for (var i = 0; i < graph.nodes.length; i++) {
            var n = graph.nodes[i];
            var lbl = n.label || n.type;
            var ch = n.outputShape ? n.outputShape[n.outputShape.length-1] : 0;
            nodeOrder.push(lbl + '(' + ch + ')');
        }
        var result = JSON.stringify({
            params: totalParams,
            output_shape: outNode ? outNode.outputShape : null,
            node_count: graph.nodes.length,
            edge_count: graph.edges.length,
            type_counts: typeCounts,
            nodes: perNodeShapes,
            node_order: nodeOrder,
            warnings: interp.warnings.length
        });
        result;
    """)
    try:
        out = subprocess.run(['osascript', '-l', 'JavaScript', '-e', js],
                           capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return {'params': -1, 'output_shape': None, 'error': out.stderr.strip()[:200]}
        return json.loads(out.stdout.strip())
    except subprocess.TimeoutExpired:
        return {'params': -1, 'output_shape': None, 'error': 'TIMEOUT'}
    except Exception as e:
        return {'params': -1, 'output_shape': None, 'error': str(e)[:200]}

# ─── TF ground truth ────────────────────────────────────
def get_tf_results():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    def record(m, input_shape):
        inp = tf.zeros(input_shape)
        m(inp)
        out_shape = list(m(inp).shape)
        type_counts = {}
        for l in m.layers:
            if not l.built: continue
            t = l.__class__.__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        per_layer = []
        for l in m.layers:
            if not l.built: continue
            lp = sum(v.numpy().size for v in l.trainable_variables)
            per_layer.append({'type': l.__class__.__name__, 'params': lp})
        return {
            'params': m.count_params(),
            'output_shape': out_shape,
            'input_shape': list(input_shape),
            'layer_types': type_counts,
            'per_layer': per_layer,
        }

    results = {}

    class M1(Model):
        def __init__(self):
            super().__init__()
            self.c = layers.Conv2D(16, 3, padding='same')
        def call(self, x): return self.c(x)
    m = M1(); m(tf.zeros((1,32,32,3)))
    results['Single Conv2D'] = record(m, (1,32,32,3))

    class M2(Model):
        def __init__(self):
            super().__init__()
            self.c1 = layers.Conv2D(8, 3, padding='same')
            self.bn = layers.BatchNormalization()
            self.c2 = layers.Conv2D(16, 1, padding='same')
        def call(self, x): return self.c2(self.bn(self.c1(x)))
    m = M2(); m(tf.zeros((1,16,16,3)))
    results['Conv+BN+Conv'] = record(m, (1,16,16,3))

    class M3(Model):
        def __init__(self):
            super().__init__()
            self.c = layers.Conv2D(3, 3, padding='same')
        def call(self, x): return x + self.c(x)
    m = M3(); m(tf.zeros((1,8,8,3)))
    results['Residual Add'] = record(m, (1,8,8,3))

    class M4(Model):
        def __init__(self):
            super().__init__()
            self.dw = layers.DepthwiseConv2D(5, padding='same')
        def call(self, x): return self.dw(x)
    m = M4(); m(tf.zeros((1,16,16,8)))
    results['DepthwiseConv2D'] = record(m, (1,16,16,8))

    class M5(Model):
        def __init__(self):
            super().__init__()
            self.enc = layers.Conv2D(16, 3, padding='same')
            self.pool = layers.MaxPooling2D(2)
            self.btl = layers.Conv2D(32, 3, padding='same')
            self.up = layers.UpSampling2D(2)
            self.dec = layers.Conv2D(16, 3, padding='same')
            self.out_conv = layers.Conv2D(1, 1, padding='same')
        def call(self, x):
            e = self.enc(x); x = self.pool(e); x = self.btl(x); x = self.up(x)
            x = tf.concat([x, e], axis=-1); x = self.dec(x); return self.out_conv(x)
    m = M5(); m(tf.zeros((1,32,32,3)))
    results['U-Net skip'] = record(m, (1,32,32,3))

    class M6(Model):
        def __init__(self, n=3):
            super().__init__()
            self.convs = [layers.Conv2D(8, 3, padding='same') for _ in range(n)]
            self.final = layers.Conv2D(1, 1)
        def call(self, x):
            for c in self.convs: x = c(x)
            return self.final(x)
    m = M6(3); m(tf.zeros((1,8,8,3)))
    results['For loop layers'] = record(m, (1,8,8,3))

    class LightUNet(Model):
        def __init__(self, bf=16, ns=3):
            super().__init__()
            self.num_stages = ns
            self.enc_conv0_pw = layers.Conv2D(bf, 1, padding='same')
            self.enc_conv0_dw = layers.DepthwiseConv2D(3, padding='same')
            self.enc_dws, self.enc_pws, self.enc_channels = [], [], [bf]
            for i in range(1, ns):
                ch = bf*2 if i==ns-1 else bf
                self.enc_dws.append(layers.DepthwiseConv2D(5, padding='same'))
                self.enc_pws.append(layers.Conv2D(ch, 1, padding='same'))
                self.enc_channels.append(ch)
            btl_ch = self.enc_channels[-1]
            self.btl_dw1 = layers.DepthwiseConv2D(5, padding='same')
            self.btl_dw2 = layers.DepthwiseConv2D(3, padding='same')
            self.btl_pw = layers.Conv2D(btl_ch, 1, padding='same')
            self.dec_dws, self.dec_pws = [], []
            for i in range(ns):
                dc = self.enc_channels[ns-1-i] if i<ns-1 else bf
                self.dec_dws.append(layers.DepthwiseConv2D(5, padding='same'))
                self.dec_pws.append(layers.Conv2D(dc, 1, padding='same'))
            self.dec_final = layers.DepthwiseConv2D(3, padding='same')
            self.prelus = [layers.PReLU(shared_axes=[1,2]) for _ in range(4*ns+4)]
        def call(self, x):
            a=0
            x=self.prelus[a](self.enc_conv0_pw(x));a+=1
            x=self.prelus[a](self.enc_conv0_dw(x));a+=1
            skips=[x]
            for i in range(self.num_stages-1):
                x=self.prelus[a](self.enc_dws[i](x));a+=1
                x=self.prelus[a](self.enc_pws[i](x));a+=1
                skips.append(x)
            x=self.prelus[a](self.btl_dw1(x));a+=1
            x=self.prelus[a](self.btl_dw2(x));a+=1
            x=self.prelus[a](self.btl_pw(x));a+=1
            for i in range(self.num_stages):
                skip=skips[self.num_stages-1-i]
                x=self.prelus[a](self.dec_dws[i](x));a+=1
                x=tf.concat([x,skip],axis=-1)
                x=self.prelus[a](self.dec_pws[i](x));a+=1
            x=self.prelus[a](self.dec_final(x));a+=1
            return x
    m = LightUNet(16, 3); m(tf.zeros((1,64,64,6)))
    results['LightUNet'] = record(m, (1,64,64,6))

    class M8(Model):
        def __init__(self):
            super().__init__()
            self.conv = layers.Conv2D(32, 3, strides=2, padding='same')
            self.gap = layers.GlobalAveragePooling2D()
            self.d1 = layers.Dense(64, activation='relu')
            self.d2 = layers.Dense(10)
        def call(self, x): return self.d2(self.d1(self.gap(self.conv(x))))
    m = M8(); m(tf.zeros((1,32,32,3)))
    results['Dense classifier'] = record(m, (1,32,32,3))

    class M9(Model):
        def __init__(self):
            super().__init__()
            self.b1 = layers.Conv2D(8, 1, padding='same')
            self.b2 = layers.Conv2D(8, 3, padding='same')
            self.b3 = layers.Conv2D(8, 5, padding='same')
            self.out = layers.Conv2D(1, 1, padding='same')
        def call(self, x):
            return self.out(tf.concat([self.b1(x), self.b2(x), self.b3(x)], axis=-1))
    m = M9(); m(tf.zeros((1,16,16,3)))
    results['Multi-branch'] = record(m, (1,16,16,3))

    class M10(Model):
        def __init__(self, n=4):
            super().__init__()
            self.stem = layers.Conv2D(16, 3, padding='same')
            self.convs = [layers.Conv2D(16, 3, padding='same') for _ in range(n)]
            self.bns = [layers.BatchNormalization() for _ in range(n)]
            self.head = layers.Conv2D(3, 1, padding='same')
        def call(self, x):
            x = self.stem(x)
            for i in range(4):
                r = x; x = self.bns[i](self.convs[i](x)); x = x + r
            return self.head(x)
    m = M10(4); m(tf.zeros((1,32,32,3)))
    results['Chained res+BN'] = record(m, (1,32,32,3))

    class M11(Model):
        def __init__(self):
            super().__init__()
            self.dws, self.pws, self.bns = [], [], []
            ch = [3,16,32,64]
            for i in range(3):
                self.dws.append(layers.DepthwiseConv2D(3, padding='same'))
                self.pws.append(layers.Conv2D(ch[i+1], 1, padding='same'))
                self.bns.append(layers.BatchNormalization())
        def call(self, x):
            for i in range(3): x = self.bns[i](self.pws[i](self.dws[i](x)))
            return x
    m = M11(); m(tf.zeros((1,16,16,3)))
    results['DW separable'] = record(m, (1,16,16,3))

    # ─── Test 12: Nested sub-model ──────────────────────
    class SubBlock(Model):
        def __init__(self, ch):
            super().__init__()
            self.dw = layers.DepthwiseConv2D(3, padding='same')
            self.pw = layers.Conv2D(ch, 1, padding='same')
        def call(self, x):
            return self.pw(self.dw(x))
    class M12(Model):
        def __init__(self):
            super().__init__()
            self.stem = layers.Conv2D(16, 3, padding='same')
            self.block1 = SubBlock(16)
            self.block2 = SubBlock(32)
            self.head = layers.Conv2D(1, 1, padding='same')
        def call(self, x):
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            return self.head(x)
    m = M12(); m(tf.zeros((1,16,16,3)))
    results['Nested sub-model'] = record(m, (1,16,16,3))

    # ─── Test 13: Helper method in call ─────────────────
    class M13(Model):
        def __init__(self):
            super().__init__()
            self.c1 = layers.Conv2D(16, 3, padding='same')
            self.c2 = layers.Conv2D(16, 3, padding='same')
            self.c3 = layers.Conv2D(8, 1, padding='same')
        def _conv_block(self, x, conv):
            return tf.nn.relu(conv(x))
        def call(self, x):
            x = self._conv_block(x, self.c1)
            x = self._conv_block(x, self.c2)
            return self.c3(x)
    m = M13(); m(tf.zeros((1,8,8,3)))
    results['Helper method'] = record(m, (1,8,8,3))

    # ─── Test 14: Encoder-decoder with varying channels ─
    class M14(Model):
        def __init__(self):
            super().__init__()
            self.enc = []
            self.dec = []
            channels = [16, 32, 64]
            for ch in channels:
                self.enc.append(layers.Conv2D(ch, 3, padding='same'))
            for ch in reversed(channels[:-1]):
                self.dec.append(layers.Conv2D(ch, 3, padding='same'))
            self.final = layers.Conv2D(1, 1, padding='same')
        def call(self, x):
            skips = []
            for conv in self.enc:
                x = conv(x)
                skips.append(x)
            for i, conv in enumerate(self.dec):
                x = tf.concat([x, skips[len(skips) - 2 - i]], axis=-1)
                x = conv(x)
            return self.final(x)
    m = M14(); m(tf.zeros((1,16,16,3)))
    results['Enc-dec varying ch'] = record(m, (1,16,16,3))

    # ─── Test 15: Sequential-style with activation kwarg ─
    class M15(Model):
        def __init__(self):
            super().__init__()
            self.c1 = layers.Conv2D(32, 3, padding='same', activation='relu')
            self.c2 = layers.Conv2D(32, 3, padding='same', activation='relu')
            self.pool = layers.MaxPooling2D(2)
            self.flat = layers.Flatten()
            self.d = layers.Dense(10)
        def call(self, x):
            x = self.c1(x)
            x = self.c2(x)
            x = self.pool(x)
            x = self.flat(x)
            return self.d(x)
    m = M15(); m(tf.zeros((1,8,8,1)))
    results['Conv+Flatten+Dense'] = record(m, (1,8,8,1))

    # ─── Test 16: LightUNet strided+SE (real arch) ──────
    tf.keras.backend.clear_session()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    from loresnet import LightUNet as LU_Real
    m = LU_Real(base_filters=16, num_stages=3, use_se=True, strided=True)
    m(tf.zeros((1, 64, 64, 6)))
    results['LightUNet strided+SE'] = record(m, (1, 64, 64, 6))

    # ─── Test 17: LoResNet_CPU (real arch) ──────────────
    tf.keras.backend.clear_session()
    from loresnet import LoResNet_CPU
    m = LoResNet_CPU(_ratio=0.5, base_filters=16, num_stages=3)
    m(tf.zeros((1, 256, 256, 3)))
    results['LoResNet_CPU'] = record(m, (1, 256, 256, 3))

    # ─── Test 18: HeavyTeacher (real arch) ──────────────
    tf.keras.backend.clear_session()
    from heavy_teacher import HeavyTeacher
    m = HeavyTeacher(_ratio=0.5)
    m(tf.zeros((1, 256, 256, 3)))
    results['HeavyTeacher'] = record(m, (1, 256, 256, 3))

    # ─── Test 19: ProgDownLite (Sequential + custom Layer) ─
    tf.keras.backend.clear_session()
    from progdownlite import ProgDownLite
    m = ProgDownLite(ratio=2)
    m(tf.zeros((1, 64, 64, 3)))
    results['ProgDownLite'] = record(m, (1, 64, 64, 3))

    return results

# ─── JS test codes (must match TF architectures) ────────
JS_CODES = {
"Single Conv2D": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.c = layers.Conv2D(16, 3, padding='same')
    def call(self, x): return self.c(x)
model = M()
""",
"Conv+BN+Conv": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.c1 = layers.Conv2D(8, 3, padding='same')
        self.bn = layers.BatchNormalization()
        self.c2 = layers.Conv2D(16, 1, padding='same')
    def call(self, x):
        x = self.c1(x)
        x = self.bn(x)
        x = self.c2(x)
        return x
model = M()
""",
"Residual Add": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.c = layers.Conv2D(3, 3, padding='same')
    def call(self, x): return x + self.c(x)
model = M()
""",
"DepthwiseConv2D": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.dw = layers.DepthwiseConv2D(5, padding='same')
    def call(self, x): return self.dw(x)
model = M()
""",
"U-Net skip": """import tensorflow as tf
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
"For loop layers": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self, n=3):
        super().__init__()
        self.convs = []
        for i in range(n):
            self.convs.append(layers.Conv2D(8, 3, padding='same'))
        self.final = layers.Conv2D(1, 1)
    def call(self, x):
        for i in range(3):
            x = self.convs[i](x)
        x = self.final(x)
        return x
model = M(n=3)
""",
"LightUNet": """import tensorflow as tf
from tensorflow.keras import layers, Model
class LightUNet(Model):
    def __init__(self, base_filters=16, num_stages=3, use_se=False, strided=False):
        super().__init__()
        bf = base_filters
        self.num_stages = num_stages
        self.use_se = use_se and num_stages > 1
        self.strided = strided and num_stages > 1
        self.enc_conv0_pw = layers.Conv2D(bf, (1, 1), padding='same')
        self.enc_conv0_dw = layers.DepthwiseConv2D((3, 3), padding='same')
        self.enc_dws = []
        self.enc_pws = []
        self.enc_channels = [bf]
        stride = (1, 1)
        for i in range(1, num_stages):
            ch_out = bf * 2 if i == num_stages - 1 else bf
            self.enc_dws.append(layers.DepthwiseConv2D((5, 5), strides=stride, padding='same'))
            self.enc_pws.append(layers.Conv2D(ch_out, (1, 1), padding='same'))
            self.enc_channels.append(ch_out)
        btl_ch = self.enc_channels[-1]
        self.btl_dw1 = layers.DepthwiseConv2D((5, 5), padding='same')
        self.btl_dw2 = layers.DepthwiseConv2D((3, 3), padding='same')
        self.btl_pw = layers.Conv2D(btl_ch, (1, 1), padding='same')
        self.dec_dws = []
        self.dec_pws = []
        for i in range(num_stages):
            if i < num_stages - 1:
                dec_ch = self.enc_channels[num_stages - 1 - i]
            else:
                dec_ch = bf
            self.dec_dws.append(layers.DepthwiseConv2D((5, 5), padding='same'))
            self.dec_pws.append(layers.Conv2D(dec_ch, (1, 1), padding='same'))
        self.dec_final = layers.DepthwiseConv2D((3, 3), padding='same')
        n_act = 4 * num_stages + 4
        self.prelus = [layers.PReLU(shared_axes=[1, 2]) for i in range(n_act)]
    def call(self, x):
        act = 0
        x = self.prelus[act](self.enc_conv0_pw(x)); act += 1
        x = self.prelus[act](self.enc_conv0_dw(x)); act += 1
        skips = [x]
        for i in range(self.num_stages - 1):
            x = self.prelus[act](self.enc_dws[i](x)); act += 1
            x = self.prelus[act](self.enc_pws[i](x)); act += 1
            skips.append(x)
        x = self.prelus[act](self.btl_dw1(x)); act += 1
        x = self.prelus[act](self.btl_dw2(x)); act += 1
        x = self.prelus[act](self.btl_pw(x)); act += 1
        for i in range(self.num_stages):
            skip = skips[self.num_stages - 1 - i]
            x = self.prelus[act](self.dec_dws[i](x)); act += 1
            x = tf.concat([x, skip], axis=-1)
            x = self.prelus[act](self.dec_pws[i](x)); act += 1
        x = self.prelus[act](self.dec_final(x)); act += 1
        return x
model = LightUNet(base_filters=16, num_stages=3, use_se=False, strided=False)
""",
"Dense classifier": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.conv = layers.Conv2D(32, 3, strides=2, padding='same')
        self.gap = layers.GlobalAveragePooling2D()
        self.d1 = layers.Dense(64, activation='relu')
        self.d2 = layers.Dense(10)
    def call(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.d1(x)
        return self.d2(x)
model = M()
""",
"Multi-branch": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.b1 = layers.Conv2D(8, 1, padding='same')
        self.b2 = layers.Conv2D(8, 3, padding='same')
        self.b3 = layers.Conv2D(8, 5, padding='same')
        self.out = layers.Conv2D(1, 1, padding='same')
    def call(self, x):
        a = self.b1(x)
        b = self.b2(x)
        c = self.b3(x)
        x = tf.concat([a, b, c], axis=-1)
        return self.out(x)
model = M()
""",
"Chained res+BN": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self, n=4):
        super().__init__()
        self.stem = layers.Conv2D(16, 3, padding='same')
        self.convs = []
        self.bns = []
        for i in range(n):
            self.convs.append(layers.Conv2D(16, 3, padding='same'))
            self.bns.append(layers.BatchNormalization())
        self.head = layers.Conv2D(3, 1, padding='same')
    def call(self, x):
        x = self.stem(x)
        for i in range(4):
            residual = x
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = x + residual
        return self.head(x)
model = M(n=4)
""",
"DW separable": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.dws = []
        self.pws = []
        self.bns = []
        ch = [3, 16, 32, 64]
        for i in range(3):
            self.dws.append(layers.DepthwiseConv2D(3, padding='same'))
            self.pws.append(layers.Conv2D(ch[i + 1], 1, padding='same'))
            self.bns.append(layers.BatchNormalization())
    def call(self, x):
        for i in range(3):
            x = self.bns[i](self.pws[i](self.dws[i](x)))
        return x
model = M()
""",
"Nested sub-model": """import tensorflow as tf
from tensorflow.keras import layers, Model
class SubBlock(Model):
    def __init__(self, ch):
        super().__init__()
        self.dw = layers.DepthwiseConv2D(3, padding='same')
        self.pw = layers.Conv2D(ch, 1, padding='same')
    def call(self, x):
        return self.pw(self.dw(x))
class M(Model):
    def __init__(self):
        super().__init__()
        self.stem = layers.Conv2D(16, 3, padding='same')
        self.block1 = SubBlock(16)
        self.block2 = SubBlock(32)
        self.head = layers.Conv2D(1, 1, padding='same')
    def call(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)
model = M()
""",
"Helper method": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.c1 = layers.Conv2D(16, 3, padding='same')
        self.c2 = layers.Conv2D(16, 3, padding='same')
        self.c3 = layers.Conv2D(8, 1, padding='same')
    def _conv_block(self, x, conv):
        return tf.nn.relu(conv(x))
    def call(self, x):
        x = self._conv_block(x, self.c1)
        x = self._conv_block(x, self.c2)
        return self.c3(x)
model = M()
""",
"Enc-dec varying ch": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.enc = []
        self.dec = []
        channels = [16, 32, 64]
        for ch in channels:
            self.enc.append(layers.Conv2D(ch, 3, padding='same'))
        for ch in reversed(channels[:-1]):
            self.dec.append(layers.Conv2D(ch, 3, padding='same'))
        self.final = layers.Conv2D(1, 1, padding='same')
    def call(self, x):
        skips = []
        for conv in self.enc:
            x = conv(x)
            skips.append(x)
        for i, conv in enumerate(self.dec):
            x = tf.concat([x, skips[len(skips) - 2 - i]], axis=-1)
            x = conv(x)
        return self.final(x)
model = M()
""",
"Conv+Flatten+Dense": """import tensorflow as tf
from tensorflow.keras import layers, Model
class M(Model):
    def __init__(self):
        super().__init__()
        self.c1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.c2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool = layers.MaxPooling2D(2)
        self.flat = layers.Flatten()
        self.d = layers.Dense(10)
    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.d(x)
model = M()
""",
}

# Load real architecture codes from files and add instantiation lines
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_root, 'tests/models/loresnet.py')) as f:
    _loresnet_code = f.read()
with open(os.path.join(_root, 'tests/models/heavy_teacher.py')) as f:
    _teacher_code = f.read()

JS_CODES['LightUNet strided+SE'] = _loresnet_code + "\nmodel = LightUNet(base_filters=16, num_stages=3, use_se=True, strided=True)\n"
JS_CODES['LoResNet_CPU'] = _loresnet_code + "\nmodel = LoResNet_CPU(_ratio=0.5, base_filters=16, num_stages=3)\n"
JS_CODES['HeavyTeacher'] = _teacher_code + "\nmodel = HeavyTeacher(_ratio=0.5)\n"

with open(os.path.join(_root, 'tests/models/progdownlite.py')) as f:
    _progdown_code = f.read()
JS_CODES['ProgDownLite'] = _progdown_code + "\nmodel = ProgDownLite(ratio=2)\n"

# ─── Main ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print(" net.sketch.io — TF vs JS interpreter non-regression tests")
    print("=" * 65)

    print("\n[1/2] Running TF ground truth...")
    tf_results = get_tf_results()
    print(f"      {len(tf_results)} models built\n")

    G = '\033[92m'  # green
    R = '\033[91m'  # red
    Y = '\033[93m'  # yellow
    C = '\033[96m'  # cyan
    D = '\033[90m'  # dim
    B = '\033[1m'   # bold
    N = '\033[0m'   # reset

    print(f"[2/2] Running JS interpreter (osascript)...\n")
    passed, failed, checks = 0, 0, 0

    for name, tf_gt in tf_results.items():
        code = JS_CODES.get(name)
        if not code:
            print(f"  {Y}SKIP{N}  {name} (no JS code)")
            continue

        print(f"  {B}{name}{N}")
        js_result = run_js(code, tf_gt['input_shape'])

        if 'error' in js_result:
            print(f"    {R}ERROR{N} JS: {js_result['error']}")
            failed += 1
            checks += 1
            print()
            continue

        tf_p = tf_gt['params']
        js_p = js_result['params']
        tf_s = tf_gt['output_shape']
        js_s = js_result['output_shape']
        js_nodes = js_result.get('node_count', '?')
        js_edges = js_result.get('edge_count', '?')
        js_warn = js_result.get('warnings', 0)
        js_types = js_result.get('type_counts', {})
        js_node_list = js_result.get('nodes', [])

        # Params check
        checks += 1
        if tf_p == js_p:
            print(f"    {G}✓{N} params    {D}TF={tf_p:>8,}  JS={js_p:>8,}{N}")
            passed += 1
        else:
            diff = js_p - tf_p
            pct = (diff / tf_p * 100) if tf_p else 0
            print(f"    {R}✗ params    TF={tf_p:>8,}  JS={js_p:>8,}  (diff={diff:+,} = {pct:+.1f}%){N}")
            failed += 1

        # Shape check
        checks += 1
        if tf_s == js_s:
            print(f"    {G}✓{N} shape     {D}{js_s}{N}")
            passed += 1
        else:
            print(f"    {R}✗ shape     TF={tf_s}  JS={js_s}{N}")
            failed += 1

        # Layer types — compare JS vs TF (factual from model.layers)
        # TF model.layers only lists keras layers, not tf ops (Add, Concat, etc.)
        # JS graph includes tf ops as nodes. So we check:
        #   1. Every TF keras layer is present in JS (with correct count)
        #   2. JS may have extra graph ops (Add, Concatenate, etc.) - that's expected
        tf_types = tf_gt.get('layer_types', {})
        js_op_types = {k: v for k, v in js_types.items() if k not in ('Input', 'Output', 'Slice')}
        graph_ops = {'Add', 'Subtract', 'Multiply', 'Concatenate', 'Resize', 'SpaceToDepth', 'DepthToSpace', 'Pad', 'Reduce'}

        # For nested sub-models: TF lists them as a single layer (e.g. SubBlock=2),
        # but JS inlines their ops. So we flatten TF sub-model layers.
        # We check: all non-submodel TF layers appear in JS with same count.
        tf_keras_layers = {k: v for k, v in tf_types.items() if k not in graph_ops}
        js_keras_layers = {k: v for k, v in js_op_types.items() if k not in graph_ops}

        # Check that every TF keras layer type appears in JS
        # (JS may have MORE because sub-models are inlined)
        checks += 1
        all_present = True
        missing = []
        for t, c in tf_keras_layers.items():
            js_c = js_keras_layers.get(t, 0)
            if js_c < c and t not in ('SubBlock',):
                # Skip custom sub-model classes - JS inlines them
                if t[0].isupper() and t not in ('Conv2D','Conv1D','Conv2DTranspose','DepthwiseConv2D','SeparableConv2D',
                    'Dense','MaxPooling2D','AveragePooling2D','UpSampling2D','GlobalAveragePooling2D','GlobalMaxPooling2D',
                    'BatchNormalization','LayerNormalization','Dropout','Flatten','Reshape','PReLU','ReLU','LeakyReLU',
                    'Embedding','LSTM','GRU','ZeroPadding2D'):
                    continue  # custom sub-model, JS inlines it
                missing.append(f"{t}: TF={c} JS={js_c}")
                all_present = False

        if all_present:
            js_str = ' '.join(f"{t}={c}" for t, c in sorted(js_op_types.items()))
            print(f"    {G}✓{N} layers    {D}{js_str}{N}")
            passed += 1
        else:
            print(f"    {R}✗ layers    missing: {', '.join(missing)}{N}")
            failed += 1

        # Edges + info
        print(f"    {D}  nodes={js_nodes} edges={js_edges}{N}")

        if js_warn > 0:
            print(f"    {Y}  ⚠ {js_warn} warnings{N}")

        # Sequence check for LoResNet_CPU
        node_order = js_result.get('node_order', [])
        if name == 'LoResNet_CPU' and node_order:
            checks += 1
            # Expected key sequence: Input → Slice(Y) → Slice(UV) → S2D → Concat(6ch)
            #   → [prefilter] → Conv(pf_head) → Add → Conv(pf_proj) → Resize
            #   → [backbone] → Add → Conv(head) → ... → Output
            order_str = ' → '.join(node_order[:8])
            # Check the node types (labels may have been renamed by variable assignment)
            node_types_only = [n.get('type','') for n in js_result.get('nodes', [])]
            # Concat must be 6ch (S2D_4 + UV_2)
            concat_idx = next((i for i, t in enumerate(node_types_only) if t == 'Concatenate'), -1)
            concat_ch = js_result['nodes'][concat_idx]['shape'][-1] if concat_idx >= 0 and js_result['nodes'][concat_idx].get('shape') else 0
            # First Resize should be after prefilter (after ~20 nodes)
            resize_idx = next((i for i, t in enumerate(node_types_only) if t == 'Resize'), -1)
            # Checks: Input first, then 2 Slices, then S2D, then Concat(6ch), Resize not too early
            has_slices = node_types_only.count('Slice') >= 2
            seq_ok = (node_types_only[0] == 'Input' and has_slices and concat_ch == 6 and resize_idx > 15)
            if seq_ok:
                print(f"    {G}✓{N} sequence  {D}{order_str}...{N}")
                passed += 1
            else:
                print(f"    {R}✗ sequence  first 5: {actual_types}, concat_ch={concat_ch}, resize@{resize_idx}{N}")
                print(f"    {R}  expected: {expect_start}, concat_ch=6, resize>10{N}")
                failed += 1

        print()

    print("─" * 65)
    total = passed + failed
    if failed == 0:
        print(f"  {G}{B}ALL PASSED{N}  {passed}/{total} checks across {len(tf_results)} models")
    else:
        print(f"  {R}{B}{failed} FAILED{N}  {passed}/{total} checks across {len(tf_results)} models")
    print("─" * 65)
    sys.exit(1 if failed else 0)
