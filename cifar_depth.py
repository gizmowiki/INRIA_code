from __future__ import print_function
from keras import regularizers
from KerasLayers.Custom_layers import LRN2D
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
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

batch_size = 32

def load_train():
    if len(filelist)!= 0:
        result_X = []
        result_Y = []
	b_size = batch_size
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
	b_size = batch_size
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
b_size = batch_size
num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % b_size)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%b_size)

nb_classes = 2
nb_epoch = 44
# input image dimensions
img_rows, img_cols = 64, 128
# The CIFAR10 images are RGB.
img_channels = 1
input_shape = (img_rows, img_cols, 1)
# model = Model(input=img_input,
#                  output=[xy])
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
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


