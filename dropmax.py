import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

CONFIG = tf.ConfigProto(device_count={'GPU': 4}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)



import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation

from keras.models import Model
from keras.datasets import mnist
from keras import losses
import keras.backend as K

input_shape = (28, 28,1)
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.
x_val = x_train[55000:]
x_train = x_train[:55000]
y_val = y_train[55000:]
y_train = y_train[:55000]


x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)


class SoftZ(Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    # @interfaces.legacy_gaussiannoise_support
    def __init__(self, **kwargs):
        super(SoftZ, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        u = tf.random_uniform(shape=tf.shape(inputs))
        noised = tf.sigmoid((1 / .1) * (tf.log(inputs / (1 - inputs)) + tf.log(u / (1 - u))))
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(SoftZ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential


def getCNNBase():
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.4)(x)
    return input_img,x
input_img,x = getCNNBase()
x = Dense(num_classes, activation='softmax')(x)
base = Model(input_img,x)

def compileRun(model,loss):
    lr = 3e-3
    SGDopt = keras.optimizers.SGD(lr=lr)

    with tf.device('/gpu:' + str(2)):
        model.compile(loss=[loss], optimizer=SGDopt,metrics=['accuracy'])
    model.fit(x_train,y_train_cat,batch_size=128,epochs=10,validation_data=(x_test,y_test_cat))

#compileRun(base,keras.losses.categorical_crossentropy)

inp, next_layer = getCNNBase()
output = Dense(num_classes)(next_layer)
rho = Dense(num_classes,activation='sigmoid',name="rho")(next_layer)
zee = SoftZ()(rho)
next_layer = Multiply()([output,zee])
softMax = Activation("softmax")(next_layer)
model = Model(inp,softMax)

def custom_loss(y_true, y_pred):
    rho = model.get_layer("rho").output
    variational_loss = keras.losses.categorical_crossentropy(y_true,rho)
    q_loss = keras.losses.categorical_crossentropy(y_true,y_pred)
    reg_loss = keras.losses.binary_crossentropy(rho,rho)
    alpha = .5
    return q_loss + variational_loss #- alpha * reg_loss

compileRun(model,keras.losses.categorical_crossentropy)

print(model.predict(x_train[:2]))
print(y_train[:2])