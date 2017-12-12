import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply, Merge



class SoftZ(Layer):

    def __init__(self, **kwargs):
        super(SoftZ, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        u = tf.random_uniform(shape=tf.shape(inputs))
        # formula (6). tau is usually set to 0.1
        # In the paper, input is rho(x;v)
        tau = .1
        #noised = tf.sigmoid((1 / tau) * (tf.log(inputs / (1 - inputs)) + tf.log(u / (1 - u))))
        noised = tf.sigmoid((1 / tau) * (inputs + tf.log(u / (1 - u))))
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(SoftZ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskSoftMax(Merge):
    """Layer that multiplies (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output *= inputs[i]
        return output