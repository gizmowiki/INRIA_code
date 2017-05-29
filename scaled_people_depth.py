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
# from KerasLayers.Custom_layers import LRN2D
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


## Constant Params

fit_generator = True

filelist = []
base_path = "/local/people_depth"
base_path_neg = "/local/people_depth/nyud"



### Functions for parameters
def load():
    global filelist
    # mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'))
    for lines in open(os.path.join(base_path, 'files_positive.txt')):
        lines = lines.strip()
        filelist.append(lines)
    for lines in open(os.path.join(base_path_neg, 'files_negative.txt')):
        lines = lines.strip()
        filelist.append(lines)

    random.shuffle(filelist)
    random.shuffle(filelist)

load()
chunk_size = 16000
maxproc = 2
processes = []
batch_index = 0
batch_index_test = 0
samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
max_q_size = int(samples_per_epoch/chunk_size) * maxproc
chunks = range(0, samples_per_epoch, chunk_size)
j = 0

num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%64)

img_shape = [64, 128]
batch_size = 256
nb_classes = 2
nb_epoch = 66

samples_seen = 0
samples_seen_test = 0
logspath = "/data/stars/user/rpandey/logs_backup/logs_fit_optimized_hung_small_fine+tune"
weightspath = "/data/stars/user/rpandey/weights_backup/weights_fit_optimized_hung_fine+tune"

if not os.path.exists(logspath):
	os.makedirs(logspath)

if not os.path.exists(weightspath):
	os.makedirs(weightspath)

## Constant Params

## Functions

def getModel(img_shape, nb_classes):
	nb_filters = 72

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
	model.add(Dropout(0.25))
	# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
	# model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
	model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
								 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='SGD',
				  metrics=['accuracy'])

	return model


def getModelOptimized(img_shape, nb_classes):
    # nb_classes = 2
    nb_filters = 96
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (img_shape[0], img_shape[1], 1)
    model = Sequential()
    model.add(Convolution2D(nb_filters, 7, 7, border_mode='valid', input_shape=input_shape, name="conv_1", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(2,2), name="maxpool_1"))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Convolution2D(72, 5, 5, name="conv_2", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_filters, 4, 4, name="conv_3", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(2,2), name="maxpool_3"))
    model.add(Convolution2D(72, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='glorot_normal', activation='softmax'))
    # model.load_weights('/home/rpandey/people_detect/weights_fit_optimized/weights-improvement-08-0.015.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model


def getModelSunday(img_shape, nb_classes):
	nb_filters = 72

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
	# model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
	# 							 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.75))
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='SGD',
				  metrics=['accuracy'])

	return model


def getModelMinimal(img_shape, nb_classes):
	nb_filters = 72
	pool_size = (2, 2)
	kernel_size = (3, 3)
	input_shape = (img_shape[0], img_shape[1], 1)
	model = Sequential()
	model.add(Convolution2D(36, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape, name="conv_1", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_1"))
	model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Convolution2D(36, 4, 4, name="conv_2", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_2"))
	model.add(Dropout(0.25))
	model.add(Convolution2D(36, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
	# model.add(Convolution2D(36, 4, 4, name="conv_4", init='glorot_normal'))
	# model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
	# model.add(Dropout(0.5))
	model.add(Convolution2D(36, 4, 4, name="conv_5", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_5"))
	model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))
	#model.load_weights(weightsPath)
	model.compile(loss='categorical_crossentropy',
				  optimizer='SGD',
				  metrics=['accuracy'])

	return model


def getModelNew(img_shape, nb_classes):
	nb_filters = 72

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

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	return model


def getModelPostNostalgia(img_shape, nb_classes):
	nb_filters = 72

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

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	return model


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


def load_train_my_generator():
    try:
		queue = multiprocessing.Queue(maxsize=max_q_size)
		def producer(chunk_index):
			result_X = []
			result_Y = []
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
				img = cv2.resize(img, (img_shape[0],img_shape[1]))
				# img = img.astype(np.float32)
				# img /= 65535
				result_X.append(img)
			# print ("from producer", len(result_Y)
			x_train = np.asarray(result_X)
			y_train = np.asarray(result_Y)
			y_train = np_utils.to_categorical(y_train, 2)
			x_train = x_train.reshape(x_train.shape[0], img_shape[0], img_shape[1], 1)
			queue.put((x_train, y_train))
			del result_X, result_Y, x_train, y_train, img
		processes = []
		def start_process():
			global j
			for i in range(len(processes), maxproc):
				j = (j+1)% len(chunks)
				thread = multiprocessing.Process(target=producer(chunks[j]))
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
		for th in processes:
			th.terminate()
			queue.close()
		raise


def get_data_train(index):
	# print ("batch_index", index)
	result_X = []
	result_Y = []
	jj = 0
	for i in range(index, index+batch_size):
		img_file_name = filelist[i]
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
			if jj%(batch_size/10) == 0:
				try:
					gauss = np.random.normal(0.8,0.3**0.5,img.shape)
					gauss= gauss.astype(np.uint16)
					img += gauss
				except:
					print ("nhi hua")
			result_Y.append(0)
		img = cv2.resize(img, (img_shape[0],img_shape[1]))
		# img = img.astype(np.float32)
		# img /= 65535
		result_X.append(img)

	x_train = np.asarray(result_X)
	result_X = []
	y_train = np.asarray(result_Y)
	y_train = np_utils.to_categorical(y_train, 2)
	result_Y = []
	x_train = x_train.reshape(x_train.shape[0], img_shape[0], img_shape[1], 1)

	return x_train, y_train


@threadsafe_generator
def load_train():
	while 1:
		global batch_index
		batch_index = (batch_index + batch_size)%num_train_samples
		x_train, y_train = get_data_train(batch_index)
		yield x_train, y_train


def get_data_test(index):
	# print ("batch_index", index)
	result_X = []
	result_Y = []
	for i in range(num_train_samples+index, num_train_samples+index+batch_size):
		img_file_name = filelist[i]
		img = cv2.imread(img_file_name, 2)
		if 'chalearn' in img_file_name:
			img = img.astype(np.float32)
			img /= 255
			img *= (((float(img_file_name.split('_')[-2])+1)*65536/4096)-1)
			img = img.astype(np.uint16)
		if 'positive' in img_file_name:
			result_Y.append(1)
		else:
			result_Y.append(0)
		img = cv2.resize(img, (img_shape[0],img_shape[1]))
		# img = img.astype(np.float32)
		# img /= 65535
		result_X.append(img)

	x_train = np.asarray(result_X)
	result_X = []
	y_train = np.asarray(result_Y)
	y_train = np_utils.to_categorical(y_train, 2)
	result_Y = []
	x_train = x_train.reshape(x_train.shape[0], img_shape[0], img_shape[1], 1)

	return x_train, y_train


@threadsafe_generator
def load_test():
	while 1:
		global batch_index_test
		batch_index_test = (batch_index_test + batch_size)%num_test_samples
		x_test, y_test = get_data_train(batch_index_test)
		yield x_test, y_test

## Functions

## Functions calling and execution

model = getModelOptimized(img_shape, nb_classes)
if sys.argv[1] is None:
	print ("num train:", num_train_samples,"num test:", num_test_samples)

	if fit_generator:
		tbcallbacks = TensorBoard(log_dir=logspath, histogram_freq=0, write_graph=True, write_images=True)
		filepath = os.path.join(weightspath, "weights-improvement-{epoch:02d}-{val_loss:.3f}.hdf5")
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
		callbacks_list = [tbcallbacks, checkpoint]
		model.fit_generator(load_train(), steps_per_epoch=(num_train_samples/batch_size), epochs=nb_epoch, # verbose=1)
	                    verbose=1, workers=4, validation_data=load_test(), validation_steps=(num_test_samples/batch_size), callbacks=callbacks_list)
	else:
		for e in range(nb_epoch):
			random.shuffle(filelist)
			j = 0
			# progbar = generic_utils.Progbar(samples_per_epoch)
			progbar_test = generic_utils.Progbar(num_test_samples)
			print("epoch %d/%d" % (e, nb_epoch))
			chunk_id = 0
			for X_train, Y_train in load_train():
				if samples_seen == samples_per_epoch:
					samples_seen = 0
				 	break
				logsdir = os.path.join(logspath, "logs_epoch_{0:02d}".format(e))
				chunk_id += 1
				print ("Now parsing chunk ", chunk_id)

				tbcallbacks = TensorBoard(log_dir=logsdir, histogram_freq=0, write_graph=True, write_images=True)
				model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, callbacks=[tbcallbacks])

				samples_seen += chunk_size
			print("Now saving models for epoch", e)
			model.save_weights(os.path.join(weightspath, "weights-improvement-epoch{0:02d}.hdf5".format(e)))
			for X_test, Y_test in load_test():
				test_loss = model.test_on_batch(X_test, Y_test)
				progbar_test.add(batch_size, values=[("test loss", test_loss[0]), ("test_accuracy", test_loss[1])])
				samples_seen_test += batch_size
				if samples_seen_test == num_test_samples:
					samples_seen_test = 0
					break

	model.save('/data/stars/share/people_depth/people-depth/fulldata/depth_people_simple_optimized_bit.h5')
else:
	print ("Now fine tuning the model with weights loaded %s " % sys.argv[1])
	model.load_weights(sys.argv[1])
	batch_index_v = 0
	mod_result_X = []
	mod_result_Y = []
	no_fine_tune = 0
	for X_train, Y_train in load_train():
		batch_index_v += 1
		print ("Now predicting for batch %d" % batch_index_v)
		if batch_index_v > 5125:
			break
		predict_probability = model.predict(X_train)
		predict_class = []
		# print (predict_probability)
		for item in predict_probability:
			if float(item[1]) <= 0.8:
				predict_class.append(0)
			else:
				predict_class.append(1)
		# predict_class_d = np_utils.to_categorical(predict_class, 2)
		# mod_result_X = []
		# mod_result_Y = []
		prev = len(mod_result_Y)
		for i in range(len(Y_train)):
			if predict_class[i] != Y_train[i][1]:
				mod_result_X.append(X_train[i])
				mod_result_Y.append(Y_train[i][1])
		print ("Total no of false_postives in batch %d is %d. Total till now %d" % (batch_index_v, len(mod_result_Y)-prev, no_fine_tune))
		if len(mod_result_Y) == prev:
			continue
		no_fine_tune += (len(mod_result_Y)-prev)
	print ("Now fine tuning the model with images %d" % no_fine_tune)
	mod_X_train = np.asarray(mod_result_X)
	mod_Y_train = np_utils.to_categorical(mod_result_Y, 2)
	model.fit(mod_X_train, mod_Y_train, epochs=20)
	mod_result_X = []
	mod_result_Y = []
	print ("Now saving model in path: %s " % sys.argv[2])
	model.save_weights(sys.argv[2])

## Functions Calling and executions
