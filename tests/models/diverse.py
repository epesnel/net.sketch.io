import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# 1. MiniResNet — small ResNet with 3 residual blocks (16/32/64 filters)
# ---------------------------------------------------------------------------

class ResBlock(Model):
    def __init__(self, filters, strides=(1, 1), name="ResBlock"):
        super(ResBlock, self).__init__(name=name)
        self.conv1 = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.use_proj = strides != (1, 1)
        if self.use_proj:
            self.proj_conv = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.proj_bn = layers.BatchNormalization()
        self.add = layers.Add()
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = inputs
        if self.use_proj:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut)
        x = self.add([x, shortcut])
        x = self.relu2(x)
        return x


class MiniResNet(Model):
    def __init__(self, num_classes=10, name="MiniResNet"):
        super(MiniResNet, self).__init__(name=name)
        # Stem
        self.stem_conv = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)
        self.stem_bn = layers.BatchNormalization()
        self.stem_relu = layers.ReLU()
        # Residual blocks: 16 → 32 → 64 with stride-2 downsampling
        self.block1 = ResBlock(16, strides=(1, 1), name="res_block_1")
        self.block2 = ResBlock(32, strides=(2, 2), name="res_block_2")
        self.block3 = ResBlock(64, strides=(2, 2), name="res_block_3")
        # Head
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs):
        x = self.stem_conv(inputs)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# 2. MobileBlock — MobileNet-style depthwise separable convolutions
# ---------------------------------------------------------------------------

class DepthwiseSepBlock(Model):
    def __init__(self, filters, strides=(1, 1), name="DWSepBlock"):
        super(DepthwiseSepBlock, self).__init__(name=name)
        self.dw_conv = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=False)
        self.dw_bn = layers.BatchNormalization()
        self.dw_relu = layers.ReLU()
        self.pw_conv = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
        self.pw_bn = layers.BatchNormalization()
        self.pw_relu = layers.ReLU()

    def call(self, inputs):
        x = self.dw_conv(inputs)
        x = self.dw_bn(x)
        x = self.dw_relu(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class MobileBlock(Model):
    def __init__(self, num_classes=10, name="MobileBlock"):
        super(MobileBlock, self).__init__(name=name)
        # Initial conv
        self.stem_conv = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)
        self.stem_bn = layers.BatchNormalization()
        self.stem_relu = layers.ReLU()
        # 4 depthwise-separable blocks: 16 → 32 → 64 → 128
        self.dsb1 = DepthwiseSepBlock(16, strides=(1, 1), name="dsb_16")
        self.dsb2 = DepthwiseSepBlock(32, strides=(2, 2), name="dsb_32")
        self.dsb3 = DepthwiseSepBlock(64, strides=(2, 2), name="dsb_64")
        self.dsb4 = DepthwiseSepBlock(128, strides=(2, 2), name="dsb_128")
        # Dense head
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(256)
        self.dense_relu = layers.ReLU()
        self.dense2 = layers.Dense(num_classes)

    def call(self, inputs):
        x = self.stem_conv(inputs)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.dsb1(x)
        x = self.dsb2(x)
        x = self.dsb3(x)
        x = self.dsb4(x)
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dense_relu(x)
        x = self.dense2(x)
        return x


# ---------------------------------------------------------------------------
# 3. TinyAttention — simple self-attention with 1x1 convs for Q/K/V
# ---------------------------------------------------------------------------

class TinyAttention(Model):
    def __init__(self, channels=64, num_classes=10, name="TinyAttention"):
        super(TinyAttention, self).__init__(name=name)
        self.channels = channels
        # Stem
        self.stem_conv = layers.Conv2D(channels, (3, 3), padding='same', use_bias=False)
        self.stem_bn = layers.BatchNormalization()
        self.stem_relu = layers.ReLU()
        # Q, K, V projections (1x1 convolutions)
        self.query_conv = layers.Conv2D(channels, (1, 1), padding='same', name="query")
        self.key_conv = layers.Conv2D(channels, (1, 1), padding='same', name="key")
        self.value_conv = layers.Conv2D(channels, (1, 1), padding='same', name="value")
        # Flatten + attention via Dense (static proxy for matmul)
        self.flatten_q = layers.Flatten()
        self.flatten_k = layers.Flatten()
        self.flatten_v = layers.Flatten()
        self.attn_dense = layers.Dense(channels * 16 * 16, name="attn_mix")
        self.reshape_out = layers.Reshape((16, 16, channels))
        # Output projection
        self.out_conv = layers.Conv2D(channels, (1, 1), padding='same', name="out_proj")
        self.out_bn = layers.BatchNormalization()
        self.out_relu = layers.ReLU()
        # Skip connection
        self.add = layers.Add()
        # Post-attention conv layers
        self.post_conv1 = layers.Conv2D(channels, (3, 3), padding='same', use_bias=False)
        self.post_bn1 = layers.BatchNormalization()
        self.post_relu1 = layers.ReLU()
        self.post_conv2 = layers.Conv2D(channels, (3, 3), padding='same', use_bias=False)
        self.post_bn2 = layers.BatchNormalization()
        self.post_relu2 = layers.ReLU()
        # Head
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs):
        x = self.stem_conv(inputs)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        # Self-attention block
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        q_flat = self.flatten_q(q)
        v_flat = self.flatten_v(v)
        attn_out = self.attn_dense(v_flat)
        attn_out = self.reshape_out(attn_out)
        attn_out = self.out_conv(attn_out)
        attn_out = self.out_bn(attn_out)
        attn_out = self.out_relu(attn_out)
        # Residual connection
        x = self.add([x, attn_out])
        # Post-attention conv layers
        x = self.post_conv1(x)
        x = self.post_bn1(x)
        x = self.post_relu1(x)
        x = self.post_conv2(x)
        x = self.post_bn2(x)
        x = self.post_relu2(x)
        # Head
        x = self.gap(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# 4. UNetMini — 3-stage encoder-decoder with skip connections via Concat
# ---------------------------------------------------------------------------

class UNetMini(Model):
    def __init__(self, name="UNetMini"):
        super(UNetMini, self).__init__(name=name)
        # Encoder stage 1: 64x64 → 32x32
        self.enc1_conv1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)
        self.enc1_bn1 = layers.BatchNormalization()
        self.enc1_relu1 = layers.ReLU()
        self.enc1_conv2 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)
        self.enc1_bn2 = layers.BatchNormalization()
        self.enc1_relu2 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D((2, 2))
        # Encoder stage 2: 32x32 → 16x16
        self.enc2_conv1 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)
        self.enc2_bn1 = layers.BatchNormalization()
        self.enc2_relu1 = layers.ReLU()
        self.enc2_conv2 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)
        self.enc2_bn2 = layers.BatchNormalization()
        self.enc2_relu2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D((2, 2))
        # Encoder stage 3: 16x16 → 8x8
        self.enc3_conv1 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)
        self.enc3_bn1 = layers.BatchNormalization()
        self.enc3_relu1 = layers.ReLU()
        self.enc3_conv2 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)
        self.enc3_bn2 = layers.BatchNormalization()
        self.enc3_relu2 = layers.ReLU()
        self.pool3 = layers.MaxPooling2D((2, 2))
        # Bottleneck: 8x8
        self.btl_conv1 = layers.Conv2D(256, (3, 3), padding='same', use_bias=False)
        self.btl_bn1 = layers.BatchNormalization()
        self.btl_relu1 = layers.ReLU()
        self.btl_conv2 = layers.Conv2D(256, (3, 3), padding='same', use_bias=False)
        self.btl_bn2 = layers.BatchNormalization()
        self.btl_relu2 = layers.ReLU()
        # Decoder stage 3: 8x8 → 16x16
        self.up3 = layers.UpSampling2D((2, 2))
        self.concat3 = layers.Concatenate()
        self.dec3_conv1 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)
        self.dec3_bn1 = layers.BatchNormalization()
        self.dec3_relu1 = layers.ReLU()
        self.dec3_conv2 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)
        self.dec3_bn2 = layers.BatchNormalization()
        self.dec3_relu2 = layers.ReLU()
        # Decoder stage 2: 16x16 → 32x32
        self.up2 = layers.UpSampling2D((2, 2))
        self.concat2 = layers.Concatenate()
        self.dec2_conv1 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)
        self.dec2_bn1 = layers.BatchNormalization()
        self.dec2_relu1 = layers.ReLU()
        self.dec2_conv2 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)
        self.dec2_bn2 = layers.BatchNormalization()
        self.dec2_relu2 = layers.ReLU()
        # Decoder stage 1: 32x32 → 64x64
        self.up1 = layers.UpSampling2D((2, 2))
        self.concat1 = layers.Concatenate()
        self.dec1_conv1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)
        self.dec1_bn1 = layers.BatchNormalization()
        self.dec1_relu1 = layers.ReLU()
        self.dec1_conv2 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)
        self.dec1_bn2 = layers.BatchNormalization()
        self.dec1_relu2 = layers.ReLU()
        # Final 1x1 output
        self.out_conv = layers.Conv2D(1, (1, 1), padding='same', name="output_conv")

    def call(self, inputs):
        # Encoder
        e1 = self.enc1_conv1(inputs)
        e1 = self.enc1_bn1(e1)
        e1 = self.enc1_relu1(e1)
        e1 = self.enc1_conv2(e1)
        e1 = self.enc1_bn2(e1)
        e1 = self.enc1_relu2(e1)
        p1 = self.pool1(e1)

        e2 = self.enc2_conv1(p1)
        e2 = self.enc2_bn1(e2)
        e2 = self.enc2_relu1(e2)
        e2 = self.enc2_conv2(e2)
        e2 = self.enc2_bn2(e2)
        e2 = self.enc2_relu2(e2)
        p2 = self.pool2(e2)

        e3 = self.enc3_conv1(p2)
        e3 = self.enc3_bn1(e3)
        e3 = self.enc3_relu1(e3)
        e3 = self.enc3_conv2(e3)
        e3 = self.enc3_bn2(e3)
        e3 = self.enc3_relu2(e3)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.btl_conv1(p3)
        b = self.btl_bn1(b)
        b = self.btl_relu1(b)
        b = self.btl_conv2(b)
        b = self.btl_bn2(b)
        b = self.btl_relu2(b)

        # Decoder
        d3 = self.up3(b)
        d3 = self.concat3([d3, e3])
        d3 = self.dec3_conv1(d3)
        d3 = self.dec3_bn1(d3)
        d3 = self.dec3_relu1(d3)
        d3 = self.dec3_conv2(d3)
        d3 = self.dec3_bn2(d3)
        d3 = self.dec3_relu2(d3)

        d2 = self.up2(d3)
        d2 = self.concat2([d2, e2])
        d2 = self.dec2_conv1(d2)
        d2 = self.dec2_bn1(d2)
        d2 = self.dec2_relu1(d2)
        d2 = self.dec2_conv2(d2)
        d2 = self.dec2_bn2(d2)
        d2 = self.dec2_relu2(d2)

        d1 = self.up1(d2)
        d1 = self.concat1([d1, e1])
        d1 = self.dec1_conv1(d1)
        d1 = self.dec1_bn1(d1)
        d1 = self.dec1_relu1(d1)
        d1 = self.dec1_conv2(d1)
        d1 = self.dec1_bn2(d1)
        d1 = self.dec1_relu2(d1)

        out = self.out_conv(d1)
        return out


# ---------------------------------------------------------------------------
# 5. AutoEncoder — Conv encoder with Dense bottleneck, conv decoder
# ---------------------------------------------------------------------------

class AutoEncoder(Model):
    def __init__(self, latent_dim=128, name="AutoEncoder"):
        super(AutoEncoder, self).__init__(name=name)
        # Encoder
        self.enc_conv1 = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.enc_bn1 = layers.BatchNormalization()
        self.enc_relu1 = layers.ReLU()
        self.enc_conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.enc_bn2 = layers.BatchNormalization()
        self.enc_relu2 = layers.ReLU()
        self.enc_flatten = layers.Flatten()
        self.enc_dense = layers.Dense(latent_dim, name="latent")
        self.enc_dense_relu = layers.ReLU()
        # Decoder
        self.dec_dense = layers.Dense(8 * 8 * 64)
        self.dec_dense_relu = layers.ReLU()
        self.dec_reshape = layers.Reshape((8, 8, 64))
        self.dec_up1 = layers.UpSampling2D((2, 2))
        self.dec_conv1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)
        self.dec_bn1 = layers.BatchNormalization()
        self.dec_relu1 = layers.ReLU()
        self.dec_up2 = layers.UpSampling2D((2, 2))
        self.dec_conv2 = layers.Conv2D(3, (3, 3), padding='same', name="reconstruction")

    def call(self, inputs):
        # Encode
        x = self.enc_conv1(inputs)
        x = self.enc_bn1(x)
        x = self.enc_relu1(x)
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_relu2(x)
        x = self.enc_flatten(x)
        x = self.enc_dense(x)
        x = self.enc_dense_relu(x)
        # Decode
        x = self.dec_dense(x)
        x = self.dec_dense_relu(x)
        x = self.dec_reshape(x)
        x = self.dec_up1(x)
        x = self.dec_conv1(x)
        x = self.dec_bn1(x)
        x = self.dec_relu1(x)
        x = self.dec_up2(x)
        x = self.dec_conv2(x)
        return x


# ---------------------------------------------------------------------------
# Instantiate all models
# ---------------------------------------------------------------------------

mini_resnet = MiniResNet()
mobile_block = MobileBlock()
tiny_attention = TinyAttention()
unet_mini = UNetMini()
auto_encoder = AutoEncoder()
