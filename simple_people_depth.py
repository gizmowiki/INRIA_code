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
from keras.callbacks import TensorBoard
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
def load():
    global filelist
    base_path = "/dev/shm/people_depth"
    base_path_neg = "/local/people_depth/"
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

load()
chunk_index = -16000
chunk_size = 16000
max_q_size = 20
maxproc = 2
processes = []
samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)


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
			img = cv2.resize(img, (64,128))
			img = img.astype(np.float32)
			img -= np.min(img)
			if (np.max(img) - np.min(img))!= 0:
				img /= (np.max(img) - np.min(img))
            		if 'positive' in img_file_name:
            		    result_Y.append(1)
            		else:
            		    result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%chunk_size == 0:
            		    x_train = np.asarray(result_X)
            		    result_X = []
            		    y_train = np.asarray(result_Y)
            		    y_train = np_utils.to_categorical(y_train, 2)
			    # y_train = np.asarray(result_Y)
            		    result_Y = []
            		    x_train = x_train.reshape(x_train.shape[0], 64, 128, 1)
           	 	    yield x_train, y_train


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
		global chunk_index
		chunk_index += chunk_size
		jj = 0
		for i in range(chunk_size):
            		img_file_name = filelist[chunk_index+i]	
         		img = cv2.imread(img_file_name, 2)
			if 'chalearn' in img_file_name:
				img = img.astype(np.float32)
				img /= 255
				img *= (((float(img_file_name.split('_')[-2])+1)*65536/4096)-1)
				img = img.astype(np.uint16)
			if 'positive' in img_file_name:
                                result_Y.append(1)
                        else:
				jj += 1
				if jj%40 == 0:
					try:
						gauss = np.random.normal(0.8,0.3**0.5,img.shape)
						gauss= gauss.astype(np.uint16)
						img += gauss
					except:
						print ("nhi hua")
                                result_Y.append(0)
			img = cv2.resize(img, (128,256))
			img = img.astype(np.float32)
			img /= 65535
			result_X.append(img)
		# print ("from producer", len(result_Y)     		    		
        	x_train = np.asarray(result_X)
        	y_train = np.asarray(result_Y)
        	y_train = np_utils.to_categorical(y_train, 2)
        	x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
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
            		img = cv2.imread(img_file_name, 2)
			if 'chalearn' in img_file_name:
                                img = img.astype(np.float32)
                                img /= 255
                                img *= (((float(img_file_name.split('_')[-2])+1)*65536/4096)-1)
                                img = img.astype(np.uint16)
			img = cv2.resize(img, (128,256))
			img = img.astype(np.float32)
			img /= 65535	
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
                		x_test = x_test.reshape(x_test.shape[0], 128, 256, 1)
                		yield x_test, y_test


	

num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%64)


batch_size = 64
nb_classes = 2
nb_epoch = 66

img_rows, img_cols = 128, 256

nb_filters = 72

pool_size = (2, 2)

kernel_size = (3, 3)

input_shape = (img_rows, img_cols, 1)



model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape, name="conv_1", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_2", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
model.add(Dropout(0.5))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.75))
model.add(Dense(nb_classes, init='glorot_normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

filepath="/home/rpandey/people_detect/weights11/weights-improvement-{epoch:02d}-{train_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

print ("num train:", num_train_samples,"num test:", num_test_samples)
# model.fit_generator(load_train_my_generator(), samples_per_epoch=num_train_samples, nb_epoch=nb_epoch, # verbose=1)
#                     verbose=1, workers=5, validation_data=load_test(), nb_val_samples=num_test_samples, callbacks=callbacks_list)
train_loss_min = 1000.3
samples_seen = 0
samples_seen_test = 0
logspath = "/home/rpandey/people_detect/logs5"
for e in range(nb_epoch):
	random.shuffle(filelist)
	# progbar = generic_utils.Progbar(samples_per_epoch)
	progbar_test = generic_utils.Progbar(num_test_samples)
	print("epoch %d/%d" % (e, nb_epoch))
	chunk_id = 0
	for X_train, Y_train in load_train_my_generator():
		logsdir = os.path.join(logspath, "logs_epoch_{0:02d}".format(e))
		chunk_id += 1
		print ("Now parsing chunk ", chunk_id)
		# train_loss = model.train_on_batch(X_train, Y_train)
		# progbar.add(batch_size, values=[("train loss", train_loss[0]), ("train_accuracy", train_loss[1])])
		tbcallbacks = TensorBoard(log_dir=logsdir, histogram_freq=0, write_graph=True, write_images=True)
		model.fit(X_train, Y_train, batch_size=64, epochs=1, callbacks=[tbcallbacks])
		samples_seen += chunk_size
		if samples_seen == samples_per_epoch:
			samples_seen = 0
		 	break
	print("Now saving models for epoch", e)
	model.save_weights("/home/rpandey/people_detect/weights12/weights-improvement-epoch{0:02d}.hdf5".format(e))
	for X_test, Y_test in load_test():
		test_loss = model.test_on_batch(X_test, Y_test)
		progbar_test.add(batch_size, values=[("test loss", test_loss[0]), ("test_accuracy", test_loss[1])])
		samples_seen_test += batch_size
		if samples_seen_test == num_test_samples:
			samples_seen_test = 0
			break

model.save('/data/stars/share/people_depth/people-depth/fulldata/depth_people_simple_new_data_double_depth_bit.h5')


