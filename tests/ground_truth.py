#!/usr/bin/env python3.11
"""Ground truth param counts from TF — used to validate the JS interpreter.
Run: python3.11 tests/ground_truth.py
Outputs JSON that the JS tests compare against.
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import tensorflow as tf
from tensorflow.keras import layers, Model

results = {}

# ─── Test 1: Single Conv2D ──────────────────────────────
class M1(Model):
    def __init__(self):
        super().__init__()
        self.c = layers.Conv2D(16, 3, padding='same')
    def call(self, x): return self.c(x)

m = M1(); m(tf.zeros((1,32,32,3)))
results['test1'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1,32,32,3))).shape)}

# ─── Test 2: Conv + BN + Conv ───────────────────────────
class M2(Model):
    def __init__(self):
        super().__init__()
        self.c1 = layers.Conv2D(8, 3, padding='same')
        self.bn = layers.BatchNormalization()
        self.c2 = layers.Conv2D(16, 1, padding='same')
    def call(self, x):
        return self.c2(self.bn(self.c1(x)))

m = M2(); m(tf.zeros((1,16,16,3)))
results['test2'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1,16,16,3))).shape)}

# ─── Test 3: Residual Add ───────────────────────────────
class M3(Model):
    def __init__(self):
        super().__init__()
        self.c = layers.Conv2D(3, 3, padding='same')
    def call(self, x): return x + self.c(x)

m = M3(); m(tf.zeros((1,8,8,3)))
results['test3'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1,8,8,3))).shape)}

# ─── Test 4: DepthwiseConv2D ────────────────────────────
class M4(Model):
    def __init__(self):
        super().__init__()
        self.dw = layers.DepthwiseConv2D(5, padding='same')
    def call(self, x): return self.dw(x)

m = M4(); m(tf.zeros((1,16,16,8)))
results['test4'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1,16,16,8))).shape)}

# ─── Test 5: U-Net skip ────────────────────────────────
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
        e = self.enc(x)
        x = self.pool(e)
        x = self.btl(x)
        x = self.up(x)
        x = tf.concat([x, e], axis=-1)
        x = self.dec(x)
        x = self.out_conv(x)
        return x

m = M5(); m(tf.zeros((1,32,32,3)))
results['test5'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1,32,32,3))).shape)}

# ─── Test 6: For loop layers ───────────────────────────
class M6(Model):
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

m = M6(n=3); m(tf.zeros((1,8,8,3)))
results['test6'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1,8,8,3))).shape)}

# ─── Test 7: LightUNet ─────────────────────────────────
class LightUNet(Model):
    def __init__(self, base_filters=16, num_stages=3, use_se=False, strided=False):
        super().__init__()
        bf = base_filters
        self.num_stages = num_stages
        self.use_se = use_se and num_stages > 1
        self.strided = strided and num_stages > 1
        self.enc_conv0_pw = layers.Conv2D(bf, (1, 1), padding='same')
        self.enc_conv0_dw = layers.DepthwiseConv2D((3, 3), padding='same')
        self.enc_dws = []; self.enc_pws = []; self.enc_channels = [bf]
        stride = (2, 2) if self.strided else (1, 1)
        for i in range(1, num_stages):
            ch_out = bf * 2 if i == num_stages - 1 else bf
            self.enc_dws.append(layers.DepthwiseConv2D((5, 5), strides=stride, padding='same'))
            self.enc_pws.append(layers.Conv2D(ch_out, (1, 1), padding='same'))
            self.enc_channels.append(ch_out)
        btl_ch = self.enc_channels[-1]
        self.btl_dw1 = layers.DepthwiseConv2D((5, 5), padding='same')
        self.btl_dw2 = layers.DepthwiseConv2D((3, 3), padding='same')
        self.btl_pw = layers.Conv2D(btl_ch, (1, 1), padding='same')
        self.dec_dws = []; self.dec_pws = []
        for i in range(num_stages):
            dec_ch = self.enc_channels[num_stages - 1 - i] if i < num_stages - 1 else bf
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

m = LightUNet(base_filters=16, num_stages=3, use_se=False, strided=False)
m(tf.zeros((1, 64, 64, 6)))
results['test7'] = {'params': m.count_params(), 'output_shape': list(m(tf.zeros((1, 64, 64, 6))).shape)}

# ─── Print results ──────────────────────────────────────
print("=" * 50)
print("TF GROUND TRUTH")
print("=" * 50)
for k, v in sorted(results.items()):
    print(f"  {k}: params={v['params']}, output={v['output_shape']}")

# Save as JSON for JS tests to consume
with open('tests/ground_truth.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to tests/ground_truth.json")
