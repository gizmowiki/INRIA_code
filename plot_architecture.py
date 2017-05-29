from __future__ import print_function
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import sys
from keras.utils import plot_model
import xml.etree.ElementTree as ET
import random
import pickle
import multiprocessing
import time
import logging
import matlab.engine
import threading
import json
import time
import os
import numpy as np
import cv2
import scipy.io as sio


def getModelFitNoFitNostalgia(img_shape):
    nb_classes = 2
    nb_filters = 96

    pool_size = (2, 2)

    kernel_size = (3, 3)

    input_shape = (img_shape[0], img_shape[1], 1)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape, name="conv_1", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
    model.add(Dropout(0.5))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
    # model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('softmax'))
    # model.load_weights(weightsPath)
    # model.compile(loss='categorical_crossentropy',
    #                     optimizer='rmsprop',
    #                     metrics=['accuracy'])

    return model



def getModelNewFitNoFitNostalgia(img_shape):
    nb_classes = 2
    nb_filters = 96

    pool_size = (2, 2)

    kernel_size = (3, 3)

    input_shape = (img_shape[0], img_shape[1], 1)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape, name="conv_1", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('softmax'))
    # model.load_weights(weightsPath)
    # model.compile(loss='categorical_crossentropy',
    # 			  optimizer='rmsprop',
    # 			  metrics=['accuracy'])

    return model


def getModelPostNostalgia2(img_shape):
	nb_filters = 72
	nb_classes = 2
	pool_size = (2, 2)

	kernel_size = (3, 3)

	input_shape = (img_shape[0], img_shape[1], 1)

	model = Sequential()

	model.add(Convolution2D(96, kernel_size[0], kernel_size[1],
							border_mode='valid',
							input_shape=input_shape, name="conv_1", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
	model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
								 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
	model.add(Dropout(0.25))
	model.add(Convolution2D(96, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
	model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
								 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))
	
	# model.load_weights(weightsPath)
	
	return model


def getModelPostNostalgia1(img_shape):
    nb_filters = 72
    nb_classes = 2
    pool_size = (2, 2)

    kernel_size = (3, 3)

    input_shape = (img_shape[0], img_shape[1], 1)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape, name="conv_1", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('softmax'))

    # model.load_weights(weightsPath)

    return model

def getModelNewNegative(img_shape):
    nb_classes = 2
    nb_filters = 96

    pool_size = (2, 2)

    kernel_size = (3, 3)

    input_shape = (img_shape[0], img_shape[1], 1)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape, name="conv_1", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
    model.add(Dropout(0.75))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('softmax'))
    # model.load_weights(weightsPath)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

img_shape = (64, 128)

model = getModelNewNegative(img_shape)

plot_model(model, to_file='/home/rpandey/model_fit_negatives.png')

model = getModelNewFitNoFitNostalgia(img_shape)

plot_model(model, to_file='/home/rpandey/model_fit_negatives_nostalgia.png')

model = getModelFitNoFitNostalgia(img_shape) 
plot_model(model, to_file='/home/rpandey/model_fit_no_fit_nostalgia.png')

model = getModelPostNostalgia2(img_shape)

plot_model(model, to_file='/home/rpandey/model_fit_ed_bx_post_nostalgia_2.png')

model = getModelPostNostalgia1(img_shape)

plot_model(model, to_file='/home/rpandey/model_fit_ed_bx_post_nostalgia_1.png')

