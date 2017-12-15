import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply, Merge

from sparsemax import sparsemax


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

    # def get_config(self):
    #     config = {'stddev': self.stddev}
    #     base_config = super(SoftZ, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


# class MaskSoftMax(Merge):
#
#
#     def merge_function(self, inputs):
#         return None
#         output = inputs[0]
#         mask = inputs[1]
#         raise ValueError(str(len(inputs)))
#         return inputs[2]
#         # e = K.exp(output - K.max(output, axis=1, keepdims=True)) * mask
#         e = K.exp(output - K.max(output, axis=1, keepdims=True))
#         # e = output * mask
#         s = K.sum(e,axis=1,keepdims=True)
#         return e/s



def maskSoftMax(output,zee):
    output = keras.layers.Lambda(lambda x: K.exp(x - K.max(x, axis=1, keepdims=True)))(output)
    next_layer = Multiply()([output,zee])
    next_layer = keras.layers.Lambda(lambda x: x+ 1e-4)(next_layer)
    return keras.layers.Lambda(lambda x: x/K.sum(next_layer,axis=1,keepdims=True),name="softmax")(next_layer)


def SparseMax():
    return keras.layers.Lambda(lambda x: sparsemax(x))
