"""
This is a code for ICLR 2018 reproducibility challenge
http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge-Readings.pdf
Paper: DropMax: Adaptive Stochastic Softmax
https://openreview.net/forum?id=Sy4c-3xRW
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#For GPU environment
CONFIG = tf.ConfigProto(device_count={'GPU': 4}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)

# from keras.datasets import cifar100
#
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()


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
from keras.regularizers import l2

from customlayers import *

import sklearn.utils



l2_co = 1e-5



class DatasetModel():


    def __init__(self,datasetFunction,baseNetwork,softmaxLayer):
        dataset = datasetFunction()
        self.input_shape = dataset[0]
        self.num_classes = dataset[1]
        self.x_train = dataset[2]
        self.x_val = dataset[3]
        self.x_test = dataset[4]
        self.y_train_cat = dataset[5]
        self.y_val_cat = dataset[6]
        self.y_test_cat = dataset[7]
        input, next_layer = baseNetwork(self.input_shape)
        next_layer = softmaxLayer(next_layer,self.num_classes,W_reg=l2(l2_co))
        self.model = Model(input,next_layer)

    def compileRun(self,loss,metrics=[]):
        lr = 1e-3
        SGDopt = keras.optimizers.SGD(lr=lr)
        Adamopt = keras.optimizers.Adam(lr=lr)

        def print_help(batch, logs):
            def out_help(name):
                return K.function([self.model.layers[0].input, K.learning_phase()],
                                  [self.model.get_layer(name).output])([self.x_train[:5], True])[0]

            print(out_help("rho"))
            print(out_help("output"))
            # print(model.get_layer("rho").get_weights())
            # print(model.get_layer("output").get_weights())

        print_weights = LambdaCallback(on_epoch_end=print_help)

        with tf.device('/gpu:' + str(3)):
            self.model.compile(loss=[loss], optimizer=Adamopt, metrics=['accuracy']+metrics)

        self.model.fit(self.x_train, self.y_train_cat, batch_size=128, epochs=100, validation_data=(self.x_test, self.y_test_cat),
                  callbacks=[print_weights])

    def getRegLoss(self,alpha = 1):
        def custom_loss(y_true, y_pred):
            rho = self.model.get_layer("rho").output
            rho = tf.sigmoid(rho)
            variational_loss = keras.losses.categorical_crossentropy(y_true, rho)
            q_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
            reg_loss = keras.losses.binary_crossentropy(rho, rho)
            return q_loss + variational_loss   - alpha * reg_loss
        return custom_loss

    def getVariationalLoss(self):
        def custom_loss(y_true, y_pred):
            rho = self.model.get_layer("rho").output
            rho = tf.sigmoid(rho)
            variational_loss = keras.losses.categorical_crossentropy(y_true, rho)
            q_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
            return q_loss + variational_loss
        return custom_loss

    def restrictTrainSize(self,n):
        self.x_train,self.y_train_cat = sklearn.utils.shuffle(self.x_train,self.y_train_cat,random_state=0)[:n]

def getMNISTBase(input_shape):
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (5, 5), activation='relu', padding='same', W_regularizer = l2(l2_co))(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', W_regularizer = l2(l2_co))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024,activation='relu', W_regularizer = l2(l2_co))(x)
    x = Dropout(0.4)(x)
    return input_img,x





# main here

## Loading and pre-processing data

def loadMNIST():
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
    return input_shape,num_classes,x_train,x_val,x_test,y_train_cat,y_val_cat,y_test_cat


# ## Base line implementation (CNN from TF tutorial)
# input_img,x = getMNISTBase()
# x = Dense(num_classes, activation='softmax', W_regularizer = l2(l2_co))(x)
# base = Model(input_img,x)
#
#
# ## Dropmax integration
# inp, next_layer = getMNISTBase()


def adaptiveDropMaxClass(next_layer,num_classes,W_reg):
    rho = Dense(num_classes, name="rho", W_regularizer=W_reg)(next_layer)
    zee = SoftZ()(rho)
    output = Dense(num_classes, name="output", W_regularizer=W_reg)(next_layer)
    softMax = maskSoftMax(output, zee)
    return softMax


def adaptiveDropMaxLogit(next_layer,num_classes,W_reg):
    rho = Dense(num_classes, name="rho", W_regularizer=W_reg)(next_layer)
    zee = SoftZ()(rho)
    output = Dense(num_classes,name="output", W_regularizer=W_reg)(next_layer)
    next_layer = Multiply(name="multiply")([output,zee])
    softMax = Activation("softmax",name="softmax")(next_layer)
    return softMax

def adaptiveDropMaxLogitReLU(next_layer,num_classes,W_reg):
    rho = Dense(num_classes, name="rho", W_regularizer=W_reg)(next_layer)
    zee = SoftZ()(rho)
    output = Dense(num_classes,name="output",activation="relu", W_regularizer=W_reg)(next_layer)
    next_layer = Multiply(name="multiply")([output,zee])
    softMax = Activation("softmax",name="softmax")(next_layer)
    return softMax

def sparseMax(next_layer,num_classes,W_reg):
    return SparseMax()(next_layer)

def baseSoftMax(next_layer,num_classes,W_reg):
    return Dense(num_classes,name="output",activation="softmax", W_regularizer=W_reg)(next_layer)

# output = Dense(num_classes,name="output",activation="relu")(next_layer)
# next_layer = Multiply(name="multiply")([output,zee])
# softMax = Activation("softmax",name="softmax")(next_layer)
# model = Model(inp,softMax)


model = DatasetModel(loadMNIST,getMNISTBase,adaptiveDropMaxClass)
# model.compileRun(keras.losses.categorical_crossentropy)
# model.compileRun(model.getVariationalLoss())
model.compileRun(model.getRegLoss(10))

# input_shape,num_classes,x_train,x_val,x_test,y_train_cat,y_val_cat,y_test_cat = loadMNIST()
# inp, next_layer = getMNISTBase(input_shape)
# softMax=adaptiveDropMaxClass(next_layer,num_classes,l2(l2_co))
# softMax=adaptiveDropMaxLogit(next_layer,num_classes,l2(l2_co))
# softMax=baseSoftMax(next_layer,num_classes,l2(l2_co))
# model = Model(inp,softMax)

# compileRun(model,keras.losses.categorical_crossentropy)
# compileRun(model,custom_loss)
#
# print(model.predict(x_train[:2]))
# print(y_train[:2])