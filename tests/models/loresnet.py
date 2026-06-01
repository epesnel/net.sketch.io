import tensorflow as tf
from tensorflow.keras import layers, Model


class LightUNet(Model):
    def __init__(self, base_filters=16, num_stages=3, use_se=True, strided=True, name="LightUNet"):
        super(LightUNet, self).__init__(name=name)
        bf = base_filters
        self.num_stages = num_stages
        self.use_se = use_se and num_stages > 1
        self.strided = strided and num_stages > 1
        self.base_filters = bf
        self.enc_conv0_pw = layers.Conv2D(bf, (1, 1), padding='same', use_bias=True)
        self.enc_conv0_dw = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=True)
        self.enc_dws = []
        self.enc_pws = []
        self.enc_channels = [bf]
        for i in range(1, num_stages):
            if self.strided:
                ch_out = bf * (2 ** i)
            else:
                ch_out = bf * 2 if i == num_stages - 1 else bf
            stride = (2, 2) if self.strided else (1, 1)
            self.enc_dws.append(layers.DepthwiseConv2D((5, 5), strides=stride, padding='same', use_bias=True))
            self.enc_pws.append(layers.Conv2D(ch_out, (1, 1), padding='same', use_bias=True))
            self.enc_channels.append(ch_out)
        btl_ch = self.enc_channels[-1] if num_stages > 1 else bf
        self.btl_dw1 = layers.DepthwiseConv2D((5, 5), padding='same', use_bias=True)
        self.btl_dw2 = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=True)
        self.btl_pw = layers.Conv2D(btl_ch, (1, 1), padding='same', use_bias=True)
        self.dec_dws = []
        self.dec_pws = []
        self.dec_channels = []
        for i in range(num_stages):
            if i < num_stages - 1:
                dec_ch = self.enc_channels[num_stages - 2 - i]
            else:
                dec_ch = bf
            self.dec_channels.append(dec_ch)
            self.dec_dws.append(layers.DepthwiseConv2D((5, 5), padding='same', use_bias=True))
            self.dec_pws.append(layers.Conv2D(dec_ch, (1, 1), padding='same', use_bias=True))
        self.dec_final = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=True)
        if self.use_se:
            r = 4
            self.se_fc1s = []
            self.se_fc2s = []
            for i in range(num_stages - 1):
                ch = self.dec_channels[i]
                self.se_fc1s.append(layers.Dense(max(ch // r, 2), use_bias=False))
                self.se_fc2s.append(layers.Dense(ch, use_bias=False))
        n_act = 4 * num_stages + 4
        self.prelus = [layers.PReLU(shared_axes=[1, 2], name=f'act_{i}') for i in range(n_act)]

    def call(self, inputs, training=None, return_features=False):
        x = inputs
        act = 0
        skips = []
        features = {} if return_features else None
        x = self.prelus[act](self.enc_conv0_pw(x)); act += 1
        x = self.prelus[act](self.enc_conv0_dw(x)); act += 1
        skips.append(x)
        if return_features:
            features['skip0'] = x
        for i in range(self.num_stages - 1):
            x = self.prelus[act](self.enc_dws[i](x)); act += 1
            x = self.prelus[act](self.enc_pws[i](x)); act += 1
            skips.append(x)
            if return_features:
                features[f'skip{i + 1}'] = x
        x = self.prelus[act](self.btl_dw1(x)); act += 1
        x = self.prelus[act](self.btl_dw2(x)); act += 1
        x = self.prelus[act](self.btl_pw(x)); act += 1
        if return_features:
            features['bottleneck'] = x
        for i in range(self.num_stages):
            skip = skips[self.num_stages - 1 - i]
            if self.strided and i > 0:
                x = tf.image.resize(x, tf.shape(skip)[1:3], method='bilinear')
            x = self.prelus[act](self.dec_dws[i](x)); act += 1
            x = tf.concat([x, skip], axis=-1)
            x = self.prelus[act](self.dec_pws[i](x)); act += 1
            if self.use_se and i < self.num_stages - 1:
                se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
                se = tf.nn.relu(self.se_fc1s[i](se))
                se = tf.nn.sigmoid(self.se_fc2s[i](se))
                x = x * se
            if return_features:
                features[f'dec_stage{i}'] = x
            if i == self.num_stages - 1:
                x = self.prelus[act](self.dec_final(x)); act += 1
        if return_features:
            return x, features
        return x


class LoResNet_CPU(Model):
    def __init__(self, _ratio, interp_method='lanczos3',
                 prefilter_bf=6, prefilter_stages=1,
                 base_filters=16, num_stages=3,
                 name="LoResNet_CPU"):
        super(LoResNet_CPU, self).__init__(name=name)
        self.ratio = _ratio
        self.interp_method = interp_method
        self.base_filters = base_filters
        self.prefilter = LightUNet(base_filters=prefilter_bf, num_stages=prefilter_stages, use_se=False, strided=False, name="prefilter")
        self.pf_head = layers.Conv2D(6, (1, 1), padding='same', use_bias=True, name="pf_head")
        self.pf_proj = layers.Conv2D(base_filters, (1, 1), padding='same', use_bias=True, name="pf_proj")
        self.backbone = LightUNet(base_filters=base_filters, num_stages=num_stages, use_se=False, strided=False, name="backbone")
        self.head_conv = layers.Conv2D(6, (1, 1), padding='same', use_bias=True, name="head_conv")

    def normalize(self, x): return (x - 128.0) / 255.0
    def denormalize(self, x): return (x * 255.0) + 128.0

    def call(self, inputs, training=None, return_features=False):
        shape = tf.shape(inputs)
        H_in, W_in = shape[1], shape[2]
        h_out = tf.cast(tf.cast(H_in, tf.float32) * self.ratio, tf.int32)
        w_out = tf.cast(tf.cast(W_in, tf.float32) * self.ratio, tf.int32)
        h_uv_out, w_uv_out = h_out // 2, w_out // 2
        y_in_hr = inputs[..., 0:1]
        uv_in_hr = inputs[:, :H_in // 2, :W_in // 2, 1:3]
        y_proc = self.normalize(y_in_hr)
        uv_proc = self.normalize(uv_in_hr)
        y_s2d = tf.nn.space_to_depth(y_proc, block_size=2)
        yuv_fr = tf.concat([y_s2d, uv_proc], axis=-1)
        features = {} if return_features else None
        pf_out = self.prefilter(yuv_fr)
        pf_residual = self.pf_head(pf_out)
        yuv_fr = yuv_fr + pf_residual
        yuv_pre_proc = self.pf_proj(yuv_fr)
        current_h = tf.shape(yuv_pre_proc)[1]
        current_w = tf.shape(yuv_pre_proc)[2]
        scale = tf.cast(h_uv_out, tf.float32) / tf.cast(current_h, tf.float32)
        target_h = tf.cast(tf.cast(current_h, tf.float32) * scale, tf.int32)
        target_w = tf.cast(tf.cast(current_w, tf.float32) * scale, tf.int32)
        x_ds = tf.image.resize(yuv_pre_proc, [target_h, target_w], method=self.interp_method)
        if return_features:
            x, bb_features = self.backbone(x_ds, return_features=True)
            features.update(bb_features)
        else:
            x = self.backbone(x_ds)
        x = x + x_ds
        out = self.head_conv(x)
        y_s2d_out, uv_out = tf.split(out, [4, 2], axis=-1)
        y_reconstructed = self.denormalize(tf.nn.depth_to_space(y_s2d_out, block_size=2))
        uv_reconstructed = self.denormalize(uv_out)
        padding_h = h_out - h_uv_out
        padding_w = w_out - w_uv_out
        uv_out_padded = tf.pad(uv_reconstructed, paddings=[[0, 0], [0, padding_h], [0, padding_w], [0, 0]], mode='CONSTANT', constant_values=0)
        output = tf.concat([y_reconstructed, uv_out_padded], axis=-1)
        if return_features:
            return {'output': output, 'features': features}
        return output
