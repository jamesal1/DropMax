import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply, Merge
from tensorflow.contrib.distributions import  Bernoulli
from sparsemax import sparsemax


class SoftZ(Layer):

    def __init__(self, **kwargs):
        super(SoftZ, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        u = tf.random_uniform(shape=tf.shape(inputs))
        tau = .1
        #noised = tf.sigmoid((1 / tau) * (tf.log(inputs / (1 - inputs)) + tf.log(u / (1 - u))))
        noised = tf.sigmoid((1 / tau) * (inputs + tf.log(u / (1 - u))))
        unnoised = tf.sigmoid((1 / tau) * inputs)
        return K.in_train_phase(noised, unnoised, training=training)


def maskSoftMax(output,zee):
    output = keras.layers.Lambda(lambda x: K.exp(x - K.max(x, axis=1, keepdims=True)))(output)
    next_layer = Multiply()([output,zee])
    next_layer = keras.layers.Lambda(lambda x: x+ 1e-4)(next_layer)
    return keras.layers.Lambda(lambda x: x/K.sum(next_layer,axis=1,keepdims=True),name="softmax")(next_layer)


def SparseMax():
    return keras.layers.Lambda(lambda x: sparsemax(x))

def randomDropLogit(retain):
    def ret(y_true, y_pred):
        bernoulli = Bernoulli(probs=retain)
        b = tf.cast(bernoulli.sample(sample_shape=tf.shape(y_true)), dtype=tf.float32)
        output = y_pred
        mask = tf.maximum(b, y_true)
        output = output * mask
        output = tf.nn.softmax(output)
        loss = keras.losses.categorical_crossentropy(y_true,output)
        return loss
    return ret

def randomDropClass(retain):
    def ret(y_true, y_pred):
        bernoulli = Bernoulli(probs=retain)
        b = tf.cast(bernoulli.sample(sample_shape=tf.shape(y_true)), dtype=tf.float32)
        output = y_pred
        mask = tf.maximum(b, y_true)
        exp_output = tf.exp(output - tf.reduce_max(output, reduction_indices=[1], keep_dims=True))
        exp_output = exp_output * mask + 1e-4
        sum_output = tf.reduce_sum(output, axis=1, keep_dims=True)
        output = exp_output / sum_output
        loss = keras.losses.categorical_crossentropy(y_true,output)
        return loss
    return ret
