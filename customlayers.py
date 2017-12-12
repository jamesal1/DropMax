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
        unnoised = tf.sigmoid((1 / tau) * inputs)
        return K.in_train_phase(noised, unnoised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(SoftZ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskSoftMax(Merge):


    def _merge_function(self, inputs):
        output = inputs[0]
        mask = inputs[1]
        e = K.exp(output - K.max(output, axis=1, keepdims=True)) * mask
        s = K.sum(e,axis=1,keepdims=True)
        return e/s