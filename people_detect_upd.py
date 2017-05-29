from __future__ import print_function
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from skimage import io
import os
import sys
import xml.etree.ElementTree as ET
import random
import pickle
import json
import time
from collections import namedtuple
# import matplotlib.pyplot as plt
import cv2
import numpy as np

filelist = []
# base_path = "/dev/shm/people_detect"
# mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'), 0)


def load():
    global filelist
    base_path = "/dev/shm/people_detect"
    base_path_neg = "/dev/shm/people_detect/"
    # mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'))
    for lines in open(os.path.join(base_path, 'files_positive.txt')):
        lines = lines.strip()
        filelist.append(lines)
    for lines in open(os.path.join(base_path_neg, 'files_negative.txt')):
        lines = lines.strip()
        filelist.append(lines)

    random.shuffle(filelist)
    random.shuffle(filelist)



def load_train():
    if len(filelist)!= 0:
        result_X = []
        result_Y = []
        num_train_samples = len(filelist[:int(0.8*len(filelist))]) - (len(filelist[:int(0.8*len(filelist))])%64)
        while 1:
		for i in range(num_train_samples):
            		img_file_name = filelist[i]
            		img = cv2.imread(img_file_name, 0)
            		if 'positive' in img_file_name:
            		    result_Y.append(1)
            		else:
            		    result_Y.append(0)
			img = img.astype(np.float32)
			img /= 255
            		result_X.append(img)
            		if (i+1)%64 == 0:
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
        num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
        remaining = len(filelist) - num_train_samples
        num_test_samples = remaining - (remaining%64)
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
            		img = img.astype(np.float32)
			img /= 255
			result_X.append(img)
            		if (i+1)%64 == 0:
                		x_test = np.asarray(result_X)
                		result_X = []
                		y_test = np.asarray(result_Y)
                		y_test = np_utils.to_categorical(y_test, 2)
				# y_test = np.asarray(result_Y)
                		result_Y = []
                		x_test = x_test.reshape(x_test.shape[0], 64, 128, 1)
                		yield x_test, y_test


	
load()
num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%64)


batch_size = 64
nb_classes = 2
nb_epoch = 30

img_rows, img_cols = 64, 128

nb_filters = 72

pool_size = (2, 2)

kernel_size = (3, 3)

input_shape = (img_rows, img_cols, 1)



model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape, name="conv_1", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_classes, init='glorot_normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

filepath="/home/rpandey/people_detect/weights2/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

print ("num train:", num_train_samples,"num test:", num_test_samples)
model.fit_generator(load_train(), samples_per_epoch=num_train_samples, nb_epoch=nb_epoch, # verbose=1)
                    verbose=1, validation_data=load_test(), nb_val_samples=num_test_samples, callbacks=callbacks_list)


model.save('/data/stars/share/people_depth/people-depth/fulldata/depth_people_tmp.h5')

