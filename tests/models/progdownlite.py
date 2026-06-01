import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer


class BilinearDownsample(Layer):
    def __init__(self, scale_factor, **kwargs):
        super(BilinearDownsample, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        height = tf.cast(input_shape[1], tf.float32)
        width = tf.cast(input_shape[2], tf.float32)
        new_height = tf.cast(height * self.scale_factor, tf.int32)
        new_width = tf.cast(width * self.scale_factor, tf.int32)
        new_size = (new_height, new_width)
        resized = tf.raw_ops.ResizeBilinear(
            images=inputs, size=new_size, half_pixel_centers=True
        )
        return tf.nn.relu(resized)


class ProgDownLite(Model):
    def __init__(self, ratio):
        super(ProgDownLite, self).__init__()
        self._ratio = 1 / ratio

        self.initial_layers = tf.keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(3, 3, padding='same')
        ])

        self.final_layers = tf.keras.Sequential([
            layers.Conv2D(64, 3, padding='same'),
            BilinearDownsample(scale_factor=self._ratio),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(3, 3, padding='same')
        ])

    def call(self, inputs, training=None):
        e1 = self.initial_layers(inputs)
        y1 = e1 + inputs
        e2 = self.final_layers(y1)
        Fx = tf.raw_ops.ResizeBicubic(
            images=y1,
            size=(tf.cast(y1.shape[1] * self._ratio, tf.int32),
                  tf.cast(y1.shape[2] * self._ratio, tf.int32)),
            half_pixel_centers=True
        )
        down = e2 + Fx
        return down
