from __future__ import print_function
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
# from keras.utils.visualize_util import plot
from KerasLayers.Custom_layers import LRN2D
from skimage import io
import os
import sys
import xml.etree.ElementTree as ET
import random
import pickle
import multiprocessing
import time
import logging
import threading
import json
import time
from collections import namedtuple

import cv2
import numpy as np

filelist = []
# base_path = "/dev/shm/people_detect"
# mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'), 0)
NB_CLASS = 2       # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True
DIM_ORDERING = 'tf'

def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,
                 border_mode='same', subsample=(1, 1),
                 activation='relu', LRN2D_norm=True,
                 weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module Convolution + lrn2d
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
                      bias=False,
                      dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    if LRN2D_norm:

        x = LRN2D(alpha=ALPHA, beta=BETA)(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x
def create_model():
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 224, 224)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (224, 224, 1)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(
        img_input, 3, 11, 11, subsample=(
            1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 6
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Channel 1 - Cov Net Layer 8
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Final Channel - Cov Net 9
    x = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING

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


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def load_train():
    if len(filelist)!= 0:
        result_X = []
        result_Y = []
        num_train_samples = len(filelist[:int(0.8*len(filelist))]) - (len(filelist[:int(0.8*len(filelist))])%64)
        while 1:
		for i in range(num_train_samples):
            		img_file_name = filelist[i]
            		img = cv2.imread(img_file_name, 0)
			img = cv2.resize(img, (224,224))
            		if 'positive' in img_file_name:
            		    result_Y.append(1)
            		else:
            		    result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%64 == 0:
            		    x_train = np.asarray(result_X)
            		    result_X = []
            		    y_train = np.asarray(result_Y)
            		    y_train = np_utils.to_categorical(y_train, 2)
                        # y_train = np.asarray(result_Y)
            		    result_Y = []
            		    x_train = x_train.reshape(x_train.shape[0], 224, 224, 1)
           	 	    yield x_train, y_train

load()
batch_index = -64
batch_size = 64
max_q_size = 20
maxproc = 8
processes = []
samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
def load_train_my_generator():
    # batch_index = -64
    # batch_size = 64
    # max_q_size = 20
    # maxproc = 8
    
    # samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
    try:
	queue = multiprocessing.Queue(maxsize=max_q_size)
        def producer():
		result_X = []
		result_Y = []
		global batch_index
		batch_index += batch_size
		for i in range(batch_size):
            		img_file_name = filelist[batch_index+i]
         		img = cv2.imread(img_file_name, 0)
			img = cv2.resize(img, (224,224))
	            	if 'positive' in img_file_name:
	            		result_Y.append(1)
	            	else:
        	    		result_Y.append(0)
            		img = img.astype(np.float32)
			img /= 255
			result_X.append(img)
		# print ("from producer", len(result_Y)     		    		
        	x_train = np.asarray(result_X)
        	y_train = np.asarray(result_Y)
        	y_train = np_utils.to_categorical(y_train, 2)
        	x_train = x_train.reshape(x_train.shape[0], 224, 224, 1)
		queue.put((x_train, y_train))

	def start_process():
		global processes
		for i in range(len(processes), maxproc):
			thread = multiprocessing.Process(target=producer)
			time.sleep(0.01)
			thread.start()
		processes.append(thread)
	while True:
		processes = [p for p in processes if p.is_alive()]
		if len(processes) < maxproc:
			start_process()
		yield queue.get()
    except:
	print("Finishing")
	global processes
	for th in processes:
		th.terminate()
		queue.close()
	raise



@threadsafe_generator
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
			img = cv2.resize(img, (224,224))
            		if 'positive' in img_file_name:
                		result_Y.append(1)
            		else:
               			result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%64 == 0:
                		x_test = np.asarray(result_X)
                		result_X = []
                		y_test = np.asarray(result_Y)
                		y_test = np_utils.to_categorical(y_test, 2)
                		# y_test = np.asarray(result_Y)
                        	result_Y = []
                		x_test = x_test.reshape(x_test.shape[0], 224, 224, 1)
                		yield x_test, y_test


	

num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%64)


batch_size = 64
nb_classes = 2
nb_epoch = 55

x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()
model = Model(input=img_input,
                  output=[x])
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

filepath="/home/rpandey/people_detect/weights5/weights-improvement-{epoch:02d}-{train_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

print ("num train:", num_train_samples,"num test:", num_test_samples)
# model.fit_generator(load_train(), samples_per_epoch=num_train_samples, nb_epoch=nb_epoch, # verbose=1)
#                    verbose=1, workers=11, validation_data=load_test(), nb_val_samples=num_test_samples, callbacks=callbacks_list)
train_loss_min = 1000.3
samples_seen = 0
samples_seen_test = 0
for e in range(nb_epoch):
	progbar = generic_utils.Progbar(samples_per_epoch)
	progbar_test = generic_utils.Progbar(num_test_samples)
	print("epoch %d" % e)
	for X_train, Y_train in load_train_my_generator():
		train_loss = model.train_on_batch(X_train, Y_train)
		progbar.add(batch_size, values=[("train loss", train_loss[0]), ("train_accuracy", train_loss[1])])
		samples_seen += batch_size
		if samples_seen == samples_per_epoch:
			samples_seen = 0
		 	break
	print("Now saving models for epoch", e)
	model.save_weights("/home/rpandey/people_detect/weights6/weights-improvement-epoch{0:02d}.hdf5".format(e))
	for X_test, Y_test in load_test():
		test_loss = model.test_on_batch(X_test, Y_test)
		progbar_test.add(batch_size, values=[("test loss", test_loss[0]), ("test_accuracy", test_loss[1])])
		samples_seen_test += batch_size
		if samples_seen_test == num_test_samples:
			samples_seen_test = 0
			break

model.save('/data/stars/share/people_depth/people-depth/fulldata/depth_people_alexnet_imagenet.h5')


