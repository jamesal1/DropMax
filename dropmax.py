"""
This is a code for ICLR 2018 reproducibility challenge
http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge-Readings.pdf
Paper: DropMax: Adaptive Stochastic Softmax
https://openreview.net/forum?id=Sy4c-3xRW
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

import tensorflow as tf
#For GPU environment
CONFIG = tf.ConfigProto(device_count={'GPU': 4}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)



from random import random
from numpy import array
from numpy import cumsum
from keras.callbacks import LambdaCallback

from keras.models import Sequential
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply, Merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model
from keras.datasets import mnist,cifar10,cifar100
from keras import losses
import keras.backend as K
from keras.regularizers import l2

from customlayers import *

import sklearn.utils

import numpy as np


from threading import Thread
run=0

class DatasetModel():


    def __init__(self,datasetFunction,baseNetwork,softmaxLayer,l2_co=0,run=0):
        dataset = datasetFunction(run)
        self.input_shape = dataset[0]
        self.num_classes = dataset[1]
        self.x_train = dataset[2]
        self.x_val = dataset[3]
        self.x_test = dataset[4]
        self.y_train_cat = dataset[5]
        self.y_val_cat = dataset[6]
        self.y_test_cat = dataset[7]
        self.model = baseNetwork(self.input_shape,self.num_classes,softmaxLayer,l2_co)
        # input, next_layer = baseNetwork(self.input_shape)
        # next_layer = softmaxLayer(next_layer,self.num_classes,W_reg=l2(l2_co))
        # self.model = Model(input,next_layer)

    def compileRun(self,loss,lr=1e-3,decay=1e-4,batch_size=50,epochs=30,held="val"):
        Adamopt = keras.optimizers.Adam(lr=lr,decay=decay)
        if isinstance(loss,float):
            loss = self.getRegLoss(loss)
        def print_help(batch, logs):
            def out_help(name):
                return K.function([self.model.layers[0].input, K.learning_phase()],
                                  [self.model.get_layer(name).output])([self.x_train[:5], True])[0]
            # print(out_help("rho"))
            # print(out_help("output"))
            # print(model.get_layer("rho").get_weights())
            # print(model.get_layer("output").get_weights())

        print_weights = LambdaCallback(on_epoch_end=print_help)
        if held =="val":
            held=self.x_val,self.y_val_cat
        else:
            held = self.x_test, self.y_test_cat
        with tf.device('/gpu:' + str(0)):
            self.model.compile(loss=[loss], optimizer=Adamopt, metrics=['accuracy'])

            self.model.fit(self.x_train, self.y_train_cat, batch_size=batch_size, epochs=epochs,
                           validation_data=held,
                      callbacks=[print_weights])

    def getRegLoss(self,alpha = 1):
        def custom_loss(y_true, y_pred):
            rho = self.model.get_layer("rho").output
            rho = tf.sigmoid(rho)
            variational_loss = keras.losses.categorical_crossentropy(y_true, rho)
            q_loss = keras.losses.categorical_crossentropy(y_true, y_pred)

            rho_ent=tf.maximum(rho,y_true)
            # reg_loss = keras.losses.binary_crossentropy(rho_ent, rho_ent)
            reg_loss = keras.losses.binary_crossentropy(rho, rho)
            return q_loss + variational_loss   - alpha * reg_loss
        return custom_loss


    def restrictTrainSize(self,n):
        self.x_train,self.y_train_cat = self.x_train[:n],self.y_train_cat[:n]

def getMNISTBase(input_shape,num_classes,softmax_layer,W_reg):
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer = l2(W_reg))(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer = l2(W_reg))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024,activation='relu', kernel_regularizer = l2(W_reg))(x)
    x = Dropout(0.4)(x)
    return Model(input_img,softmax_layer(x,num_classes,l2(W_reg)))
    return input_img,x

def getCIFAR10Base(input_shape,num_classes,softmax_layer,W_reg):
    with tf.device('/gpu:' + str(0)):
        import cifar10vgg
        m = cifar10vgg.cifar10vgg(W_reg).model
        model = Model(m.layers[0].input, softmax_layer(m.layers[-3].output, num_classes, l2(W_reg)))
        return model

def getCIFAR100Base(input_shape, num_classes, softmax_layer, W_reg):
    with tf.device('/gpu:' + str(0)):
        import cifar100vgg
        m = cifar100vgg.cifar100vgg(W_reg).model
        model = Model(m.layers[0].input, softmax_layer(m.layers[-3].output, num_classes, l2(W_reg)))
        return model


def loadMNIST(run):
    input_shape = (28, 28,1)
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=run)
    x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.
    x_val = x_train[:5000]
    x_train = x_train[5000:]
    y_val = y_train[:5000]
    y_train = y_train[5000:]
    x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    return input_shape,num_classes,x_train,x_val,x_test,y_train_cat,y_val_cat,y_test_cat

def loadCIFAR10(run):
    input_shape=(32,32,3)
    num_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)
    X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train, random_state=run)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    X_val = X_train[:5000]
    X_train = X_train[5000:]
    Y_val = Y_train[:5000]
    Y_train = Y_train[5000:]
    return input_shape,num_classes,X_train,X_val,X_test,Y_train,Y_val,Y_test


def loadCIFAR100(run):
    input_shape=(32,32,3)
    num_classes = 100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)
    X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train, random_state=run)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    X_val = X_train[:5000]
    X_train = X_train[5000:]
    Y_val = Y_train[:5000]
    Y_train = Y_train[5000:]
    return input_shape,num_classes,X_train,X_val,X_test,Y_train,Y_val,Y_test


def adaptiveDropMaxClass(next_layer,num_classes,W_reg):
    rho = Dense(num_classes, name="rho", kernel_regularizer=W_reg)(next_layer)
    zee = SoftZ()(rho)
    output = Dense(num_classes, name="output", kernel_regularizer=W_reg)(next_layer)
    softMax = maskSoftMax(output, zee)
    return softMax

def adaptiveDropMaxLogit(next_layer,num_classes,W_reg,activation="linear"):
    rho = Dense(num_classes, name="rho", kernel_regularizer=W_reg)(next_layer)
    zee = SoftZ()(rho)
    output = Dense(num_classes,name="output", kernel_regularizer=W_reg,activation=activation)(next_layer)
    next_layer = Multiply(name="multiply")([output,zee])
    softMax = Activation("softmax",name="softmax")(next_layer)
    return softMax

def sparseMax(next_layer,num_classes,W_reg):
    output = Dense(num_classes, name="output", kernel_regularizer=W_reg)(next_layer)
    return SparseMax()(output)

def baseSoftMax(next_layer,num_classes,W_reg):
    return Dense(num_classes,name="output",activation="softmax", kernel_regularizer=W_reg)(next_layer)

def randomDropMax(next_layer,num_classes,W_reg):
    return Dense(num_classes, name="output", kernel_regularizer=W_reg)(next_layer)



# model = DatasetModel(loadCIFAR10,getCIFAR10Base,adaptiveDropMaxClass)
# model = DatasetModel(loadCIFAR100,getCIFAR100Base,adaptiveDropMaxClass)


functionLosses=[
                (baseSoftMax,keras.losses.categorical_crossentropy),
                (sparseMax,keras.losses.categorical_crossentropy),
                (adaptiveDropMaxClass,keras.losses.categorical_crossentropy),
                (adaptiveDropMaxClass,0.),
                (adaptiveDropMaxClass,1.),
                (adaptiveDropMaxClass,10.),
                (adaptiveDropMaxLogit,1.),
                (adaptiveDropMaxLogit,10.),
                (adaptiveDropMaxLogit,100.),
                (randomDropMax,randomDropClass(.2)),
                (randomDropMax,randomDropClass(.4)),
                (randomDropMax,randomDropClass(.6)),
                (randomDropMax,randomDropLogit(.2)),
                (randomDropMax,randomDropLogit(.4)),
                (randomDropMax,randomDropLogit(.6))
                ]

def valAllMNIST():
    for train_size,epochs in [(1000,500),(5000,200),(55000,20)]:
        for soft,loss in functionLosses:
            for l2_coef,gpu in [(0,0),(1e-5,1),(1e-4,2),(1e-3,3)]:
                if gpu!=int(sys.argv[1]):
                    continue
                print(l2_coef, soft, loss, train_size)
                model = DatasetModel(loadMNIST, getMNISTBase, soft, l2_coef)
                model.restrictTrainSize(train_size)
                model.compileRun(loss,lr=1e-3,decay=1e-4,batch_size=128,epochs=epochs)

def testAllMNIST():
    optimal1k=[(baseSoftMax,0,keras.losses.categorical_crossentropy),
    (sparseMax,1e-4,keras.losses.categorical_crossentropy),
    (adaptiveDropMaxClass,1e-5,10.),
    (adaptiveDropMaxLogit,1e-5,10.),
    (randomDropMax,1e-5,randomDropClass(.4)),
    (randomDropMax,1e-4,randomDropLogit(.6))]
    optimal5k=[(baseSoftMax,0,keras.losses.categorical_crossentropy),
    (sparseMax,1e-3,keras.losses.categorical_crossentropy),
    (adaptiveDropMaxClass,0,10.),
    (adaptiveDropMaxLogit,1e-5,10.),
    (randomDropMax,1e-4,randomDropClass(.4)),
    (randomDropMax,0,randomDropClass(.4))]
    optimal55k=[(baseSoftMax,1e-3,keras.losses.categorical_crossentropy),
    (sparseMax,0,keras.losses.categorical_crossentropy),
    (adaptiveDropMaxClass,0,10.),
    (adaptiveDropMaxLogit,0,10.),
    (randomDropMax,1e-5,randomDropClass(.6)),
    (randomDropMax,0,randomDropClass(.6))]
    for train_size,epochs,optimal in [(1000,500,optimal1k),(5000,200,optimal5k),(55000,20,optimal55k)]:
        for soft,l2_coef,loss in optimal:
            print(l2_coef, soft, loss, train_size)
            model = DatasetModel(loadMNIST, getMNISTBase, soft, l2_coef)
            model.restrictTrainSize(train_size)
            model.compileRun(loss, lr=1e-3, decay=1e-4, batch_size=128, epochs=epochs,held="test")


valAllMNIST()
testAllMNIST()
# model = DatasetModel(loadCIFAR10,getCIFAR10Base,adaptiveDropMaxClass)
# model = DatasetModel(loadMNIST,getMNISTBase,adaptiveDropMaxClass)
# model = DatasetModel(loadMNIST,getMNISTBase,adaptiveDropMaxLogit)
# model.compileRun(keras.losses.categorical_crossentropy,batch_size=512)
# model.compileRun(keras.losses.categorical_crossentropy,lr=1e-5,batch_size=256)
# model.compileRun(model.getRegLoss(10),batch_size=128)