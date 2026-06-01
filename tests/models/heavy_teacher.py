import tensorflow as tf
from tensorflow.keras import layers, Model


class HeavyTeacher(Model):
    W = 8

    def __init__(self, _ratio, name="HeavyTeacher"):
        super(HeavyTeacher, self).__init__(name=name)
        self.ratio = _ratio
        W = self.W
        self.pre_conv = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True, name="pre_conv")
        self.pre_dec_conv1 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.pre_dec_conv2 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.pre_dec_conv3 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.pre_dec_conv4 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.pre_dec_conv5 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.pre_dec_out = layers.Conv2D(16 * W, (1, 1), padding='same', use_bias=True, name="pre_dec_out")
        self.pre_dec_residual = layers.Conv2D(6, (1, 1), padding='same', use_bias=True, name="pre_dec_residual")
        self.enc_conv0_1 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv0_2 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv1_1 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv1_2 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv1_3 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv2_1 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv2_2 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.enc_conv2_3 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.btl_conv1 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.btl_conv2 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.btl_conv3 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.dec_conv1_1 = layers.Conv2D(32 * W, (3, 3), padding='same', use_bias=True)
        self.dec_conv1_pw = layers.Conv2D(16 * W, (1, 1), padding='same', use_bias=True)
        self.dec_conv2_1 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.dec_conv2_pw = layers.Conv2D(16 * W, (1, 1), padding='same', use_bias=True)
        self.dec_conv3_1 = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.dec_conv3_pw = layers.Conv2D(16 * W, (1, 1), padding='same', use_bias=True)
        self.dec_conv3_final = layers.Conv2D(16 * W, (3, 3), padding='same', use_bias=True)
        self.head_conv = layers.Conv2D(6, (1, 1), padding='same', use_bias=True, name="head_conv")
        self.prelus = [layers.PReLU(shared_axes=[1, 2], name=f'prelu_{i}') for i in range(27)]

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
        yuv_pre = self.prelus[18](self.pre_conv(yuv_fr))
        yuv_pre = self.prelus[19](self.pre_dec_conv1(yuv_pre))
        yuv_pre = self.prelus[20](self.pre_dec_conv2(yuv_pre))
        yuv_pre = self.prelus[21](self.pre_dec_conv3(yuv_pre))
        yuv_pre = self.prelus[22](self.pre_dec_conv4(yuv_pre))
        yuv_pre = self.prelus[23](self.pre_dec_conv5(yuv_pre))
        yuv_pre = self.pre_dec_out(yuv_pre)
        if return_features:
            features['pre_dec'] = yuv_pre
        pre_residual = self.pre_dec_residual(yuv_pre)
        yuv_enhanced = yuv_fr + pre_residual
        x = tf.image.resize(yuv_pre, size=(h_uv_out, w_uv_out), method='bicubic', antialias=False)
        x = self.prelus[0](self.enc_conv0_1(x))
        x = self.prelus[1](self.enc_conv0_2(x))
        skip0 = x
        if return_features:
            features['skip0'] = skip0
        x = self.prelus[2](self.enc_conv1_1(x))
        x = self.prelus[3](self.enc_conv1_2(x))
        x = self.prelus[4](self.enc_conv1_3(x))
        skip1 = x
        if return_features:
            features['skip1'] = skip1
        x = self.prelus[5](self.enc_conv2_1(x))
        x = self.prelus[6](self.enc_conv2_2(x))
        x = self.prelus[7](self.enc_conv2_3(x))
        skip2 = x
        if return_features:
            features['skip2'] = skip2
        x = self.prelus[8](self.btl_conv1(x))
        x = self.prelus[9](self.btl_conv2(x))
        x = self.prelus[10](self.btl_conv3(x))
        if return_features:
            features['bottleneck'] = x
        x = self.prelus[11](self.dec_conv1_1(x))
        x = tf.concat([x, skip2], axis=-1)
        x = self.prelus[12](self.dec_conv1_pw(x))
        if return_features:
            features['dec_stage1'] = x
        x = self.prelus[13](self.dec_conv2_1(x))
        x = tf.concat([x, skip1], axis=-1)
        x = self.prelus[14](self.dec_conv2_pw(x))
        if return_features:
            features['dec_stage2'] = x
        x = self.prelus[15](self.dec_conv3_1(x))
        x = tf.concat([x, skip0], axis=-1)
        x = self.prelus[16](self.dec_conv3_pw(x))
        x = self.prelus[17](self.dec_conv3_final(x))
        if return_features:
            features['dec_stage3'] = x
        residuals = self.head_conv(x)
        res_y_s2d, res_uv = tf.split(residuals, [4, 2], axis=-1)
        y_enhanced_s2d, uv_enhanced = tf.split(yuv_enhanced, [4, 2], axis=-1)
        y_enhanced = tf.nn.depth_to_space(y_enhanced_s2d, block_size=2)
        y_base = tf.image.resize(y_enhanced, size=(h_out, w_out), method='bicubic', antialias=False)
        uv_base = tf.image.resize(uv_enhanced, size=(h_uv_out, w_uv_out), method='bicubic', antialias=False)
        res_y = tf.nn.depth_to_space(res_y_s2d, block_size=2)
        y_reconstructed = self.denormalize(y_base + res_y)
        uv_reconstructed = self.denormalize(uv_base + res_uv)
        padding_h = h_out - h_uv_out
        padding_w = w_out - w_uv_out
        uv_out_padded = tf.pad(uv_reconstructed, paddings=[[0, 0], [0, padding_h], [0, padding_w], [0, 0]], mode='CONSTANT', constant_values=0)
        output = tf.concat([y_reconstructed, uv_out_padded], axis=-1)
        if return_features:
            return {'output': output, 'features': features}
        return output
