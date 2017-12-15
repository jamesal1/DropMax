"""
This is a code for ICLR 2018 reproducability challenge
http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge-Readings.pdf
Paper: DropMax: Adaptive Stochastic Softmax
https://openreview.net/forum?id=Sy4c-3xRW
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.distributions import  Bernoulli
from random import random
from numpy import array
from numpy import cumsum
from keras.callbacks import LambdaCallback

from keras.models import Sequential
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply, Merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model
from keras.datasets import mnist
from keras import losses
import keras.backend as K

from customlayers import *

#For GPU environment
CONFIG = tf.ConfigProto(device_count={'GPU': 4}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)

#retain_probabilities = [0.2, 0.4, 0.6]


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


def compileRun(model,loss):
    lr = 3e-3
    SGDopt = keras.optimizers.SGD(lr=lr)
    Adamopt = keras.optimizers.Adam(lr=lr)
    def print_help(batch,logs):
        def out_help(name):
            return K.function([model.layers[0].input,K.learning_phase()],
                                          [model.get_layer(name).output])([x_train[:5],True])[0]
        #print(out_help("rho"))
        print(out_help("output"))

        # print(model.get_layer("rho").get_weights())
        # print(model.get_layer("output").get_weights())

    print_weights = LambdaCallback(on_epoch_end=print_help)

    with tf.device('/gpu:' + str(2)):
        model.compile(loss=[loss], optimizer=Adamopt,metrics=['accuracy'])

    model.fit(x_train,y_train_cat,batch_size=128,epochs=10,validation_data=(x_test,y_test_cat),
          callbacks = [print_weights])


def custom_loss(y_true, y_pred):
    rho = model.get_layer("rho").output
    rho = tf.sigmoid(rho)
    variational_loss = keras.losses.categorical_crossentropy(y_true,rho)
    q_loss = keras.losses.categorical_crossentropy(y_true,y_pred)
    reg_loss = keras.losses.binary_crossentropy(rho,rho)
    alpha = 1
    #return q_loss - alpha * reg_loss
    return q_loss + variational_loss #- alpha * reg_loss


def custom_loss_with_drop(y_true, y_pred):
    bernouli = Bernoulli(probs=0.8)
    b = tf.cast(bernouli.sample(sample_shape=tf.shape(y_true)), dtype=tf.float32)

    output = model.get_layer("output").output
    mask = tf.maximum(b, y_true)
    output = output * mask
    output = tf.nn.softmax(output)
    # exp_output = tf.exp(output - tf.reduce_max(output, reduction_indices=[1], keep_dims=True))  # subtract maximum val to prevent it from overflowing
    # lambda x: x + 1e-4, exp_output
    #exp_output = exp_output + 1e-4
    #sum_output = tf.reduce_sum(output, axis=1, keep_dims=True)
    #output = exp_output / sum_output
    loss = keras.losses.categorical_crossentropy(y_true,output)

    return loss


def custom_loss_with_drop_after(y_true, y_pred):
    bernouli = Bernoulli(probs=0.8)
    b = tf.cast(bernouli.sample(sample_shape=tf.shape(y_true)), dtype=tf.float32)

    output = model.get_layer("output").output
    mask = tf.maximum(b, y_true)

    # output = tf.nn.softmax(output)
    exp_output = tf.exp(output - tf.reduce_max(output, reduction_indices=[1], keep_dims=True))  # subtract maximum val to prevent it from overflowing
    exp_output = exp_output + 1e-4
    exp_output = exp_output * mask
    sum_output = tf.reduce_sum(output, axis=1, keep_dims=True)
    output = exp_output / sum_output
    loss = keras.losses.categorical_crossentropy(y_true, output)

    return loss


# main here

## Loading and pre-processing data
input_shape = (28, 28,1)
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#print(x_test.shape)
x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.
x_val = x_train[55000:]
x_train = x_train[:55000]
y_val = y_train[55000:]
y_train = y_train[:55000]
x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)


## Base line implementation (CNN from TF tutorial)
input_img,x = getCNNBase()
x = Dense(num_classes, activation='softmax')(x)
base = Model(input_img,x)


## Dropmax integration
inp, next_layer = getCNNBase()


# rho (p) from Formula (4)
# represents retain probabilities
#rho = Dense(num_classes,activation='sigmoid',name="rho")(next_layer)

#rho = Dense(num_classes,name="rho")(next_layer)

# Instantiate SoftZ object with parameter rho

#zee = SoftZ()(rho)

# output = Dense(num_classes,name="output",activation="relu")(next_layer)
# next_layer = Multiply(name="multiply")([output,zee])
# softMax = Activation("softmax",name="softmax")(next_layer)
# model = Model(inp,softMax)

# To test random dropout
output = Dense(num_classes,name="output",activation="linear")(next_layer)
dropout = Dropout(0.4)(output)
softMax = Activation("softmax",name="softmax")(dropout)


#softMax = MaskSoftMax(name="softmax")([output,zee])
model = Model(inp,softMax)



#compileRun(model,keras.losses.categorical_crossentropy)
compileRun(model,custom_loss_with_drop)

print(model.predict(x_train[:2]))
print(y_train[:2])
