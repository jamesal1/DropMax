"""
This is a code for ICLR 2018 reproducability challenge
http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge-Readings.pdf
Paper: DropMax: Adaptive Stochastic Softmax
https://openreview.net/forum?id=Sy4c-3xRW
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from random import random
from numpy import array
from numpy import cumsum
from keras.callbacks import LambdaCallback
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Layer, Lambda, Multiply, Merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model
from keras.datasets import mnist
from keras import losses
import keras.backend as K

#For GPU environment
CONFIG = tf.ConfigProto(device_count={'GPU': 4}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)

#Random Seed
seed = 0
random.seed(seed)

#Setup CrossValidation
splits = 5
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
cvscores = []


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
        print(out_help("rho"))
        print(out_help("output"))

        # print(model.get_layer("rho").get_weights())
        # print(model.get_layer("output").get_weights())

    print_weights = LambdaCallback(on_epoch_end=print_help)

    with tf.device('/gpu:' + str(2)):
        model.compile(loss=[loss], optimizer=Adamopt,metrics=['accuracy'])

    model.fit(x_train,y_train_cat,batch_size=128,epochs=100,validation_data=(x_test,y_test_cat),
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



# main here

## Loading and pre-processing data
input_shape = (28, 28,1)
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#print(x_test.shape)

for train, test in kfold.split(X, Y):
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
    rho = Dense(num_classes,name="rho")(next_layer)

    # Instantiate SoftZ object with parameter rho
    zee = SoftZ()(rho)

    output = Dense(num_classes,name="output",activation="relu")(next_layer)
    next_layer = Multiply(name="multiply")([output,zee])
    softMax = Activation("softmax",name="softmax")(next_layer)
    model = Model(inp,softMax)

    # output = Dense(num_classes,name="output",activation="softmax")(next_layer)
    # next_layer = Multiply(name="multiply")([output,zee])
    # softMax = Activation("softmax",name="softmax")(next_layer)
    # model = Model(inp,softMax)



    #compileRun(model,keras.losses.categorical_crossentropy)
    compileRun(model,custom_loss)

    print(model.predict(x_train[:2]))
    print(y_train[:2])
    # cvscores += score

print(sum(cvscores)/len(cvscores))