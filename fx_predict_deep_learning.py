#!/usr/bin/python
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
import numpy as np


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Xt_Yt(X, y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test
df1 = pd.read_csv("USDJPY.csv", names=["time", "open", "high", "low", "close", "volume"], header=None)
h = .02  # step size in the mesh
open_data = pd.to_numeric(df1['open'][1:])
close_data = pd.to_numeric(df1['close'][1:])

model = Sequential()

# Creat series for the output data
predict_data = pd.Series()

# Creat data frame for the input data
input_data = pd.DataFrame(dtype=float)

for i in range(1,close_data.size-21):
    ar = pd.Series()
    predict_data = predict_data.append(pd.Series(data=close_data[i+21]))
    for n in range(i,i+20):
        ar= ar.append(pd.Series(data=close_data[n]), ignore_index=True,verify_integrity=True)


    input_data = input_data.append(ar, ignore_index=True,verify_integrity=True)

X_train, X_test, Y_train, Y_test = create_Xt_Yt(input_data.values, predict_data.values)

model = Sequential()
model.add(Convolution1D(input_shape = (X_train.shape[0], X_train.shape[1]),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(input_shape = (X_train.shape[0], X_train.shape[1]),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(250))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('linear'))

opt = Nadam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
          nb_epoch = 100,
          batch_size = 128,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

print(predict_data.shape)
print(input_data.shape)

print(predict_data)
print(input_data)