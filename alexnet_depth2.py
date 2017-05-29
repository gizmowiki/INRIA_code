from __future__ import print_function
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers
from KerasLayers.Custom_layers import LRN2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from skimage import io
import os
import sys
import xml.etree.ElementTree as ET
import random
import pickle
import json
import time
from collections import namedtuple
import cv2
import numpy as np

filelist = []
# base_path = "/dev/shm/people_detect"
# mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'), 0)

NB_CLASS = 2         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005   # L2 regularization factor
USE_BN = True           # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'tf'

def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module conv + BN
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    if batch_norm:
        x = LRN2D()(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def create_model():

    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 64, 128)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (64, 128, 1)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Conv Net Layer 1
    x = conv2D_bn(img_input, 1, 11, 11, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 2 - Conv Net Layer 1
    y = conv2D_bn(img_input, 1, 11, 11, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(y)
    y = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y)

    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 48, 55, 55, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(y)
    y = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y)

    # Channel 1 - Conv Net Layer 3
    x = conv2D_bn(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 2 - Conv Net Layer 3
    y = conv2D_bn(y, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(y)
    y = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y)

    # Channel 1 - Conv Net Layer 4
    x1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    x1 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x1)
    x1 = conv2D_bn(x1, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 2 - Conv Net Layer 4
    y1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    y1 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y1)
    y1 = conv2D_bn(y1, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 1 - Conv Net Layer 5
    x2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    x2 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x2)
    x2 = conv2D_bn(x2, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 2 - Conv Net Layer 5
    y2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    y2 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y2)
    y2 = conv2D_bn(y2, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 1 - Cov Net Layer 6
    x3 = conv2D_bn(x2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x3 = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x3)
    x3 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = conv2D_bn(y2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y3 = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(y3)
    y3 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y3)

    # Channel 1 - Cov Net Layer 7
    x4 = merge([x3, y3], mode='mul', concat_axis=CONCAT_AXIS)
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(DROPOUT)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = merge([x3, y3], mode='mul', concat_axis=CONCAT_AXIS)
    y4 = Flatten()(y4)
    y4 = Dense(2048, activation='relu')(y4)
    y4 = Dropout(DROPOUT)(y4)

    # Channel 1 - Cov Net Layer 8
    x5 = merge([x4, y4], mode='mul')
    x5 = Dense(2048, activation='relu')(x5)
    x5 = Dropout(DROPOUT)(x5)

    # Channel 2 - Cov Net Layer 8
    y5 = merge([x4, y4], mode='mul')
    y5 = Dense(2048, activation='relu')(y5)
    y5 = Dropout(DROPOUT)(y5)

    # Final Channel - Cov Net 9
    xy = merge([x5, y5], mode='mul')
    xy = Dense(output_dim=NB_CLASS,
               activation='softmax')(xy)

    return xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING

def load():
    global filelist
    base_path = "/local/people_detect"
    base_path_neg = "/local/people_detect/"
    # mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'))
    for lines in open(os.path.join(base_path, 'files_positive_tmp.txt')):
        lines = lines.strip()
        filelist.append(lines)
    for lines in open(os.path.join(base_path_neg, 'files_negative_tmp.txt')):
        lines = lines.strip()
        filelist.append(lines)

    random.shuffle(filelist)
    random.shuffle(filelist)



def load_train():
    if len(filelist)!= 0:
        result_X = []
        result_Y = []
	b_size = 1
        num_train_samples = len(filelist[:int(0.8*len(filelist))]) - (len(filelist[:int(0.8*len(filelist))])%b_size)
        while 1:
		for i in range(num_train_samples):
            		img_file_name = filelist[i]
            		img = cv2.imread(img_file_name, 0)
            		if 'positive' in img_file_name:
            		    result_Y.append(1)
            		else:
            		    result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%b_size == 0:
            		    x_train = np.asarray(result_X)
            		    result_X = []
            		    y_train = np.asarray(result_Y)
            		    y_train = np_utils.to_categorical(y_train, 2)
                        # y_train = np.asarray(result_Y)
            		    result_Y = []
            		    x_train = x_train.reshape(x_train.shape[0], 64, 128, 1)
           	 	    yield x_train, y_train


def load_test():
    if len(filelist) != 0:
	b_size = 1
        num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % b_size)
        remaining = len(filelist) - num_train_samples
        num_test_samples = remaining - (remaining%b_size)
        result_X = []
        result_Y = []
	while 1:
        	for i in range(num_train_samples, num_train_samples + num_test_samples):
            		img_file_name = filelist[i]
            		img = cv2.imread(img_file_name, 0)
            		if 'positive' in img_file_name:
                		result_Y.append(1)
            		else:
               			result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%b_size == 0:
                		x_test = np.asarray(result_X)
                		result_X = []
                		y_test = np.asarray(result_Y)
                		y_test = np_utils.to_categorical(y_test, 2)
                		# y_test = np.asarray(result_Y)
                        	result_Y = []
                		x_test = x_test.reshape(x_test.shape[0], 64, 128, 1)
                		yield x_test, y_test


	
load()
b_size = 1
num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % b_size)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%b_size)


batch_size = 1
nb_classes = 2
nb_epoch = 44
xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

model = Model(input=img_input,
                  output=[xy])
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

filepath="/home/rpandey/people_detect/weights4/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

print ("num train:", num_train_samples,"num test:", num_test_samples)
model.fit_generator(load_train(), samples_per_epoch=num_train_samples, nb_epoch=nb_epoch, # verbose=1)
                    verbose=1, validation_data=load_test(), nb_val_samples=num_test_samples, callbacks=callbacks_list)


model.save('/data/stars/share/people_depth/people-depth/fulldata/depth_people_alexnet.h5')


