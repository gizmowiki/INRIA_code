from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

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

def AlexNet_convnet(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=(227,227,3))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096,6,6,activation="relu",name="dense_1")(dense_1)
        dense_2 = Convolution2D(4096,1,1,activation="relu",name="dense_2")(dense_1)
        dense_3 = Convolution2D(1000, 1,1,name="dense_3")(dense_2)
        prediction = Softmax4D(axis=1,name="softmax")(dense_3)
    else:
        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000,name='dense_3')(dense_3)
        prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(input=inputs, output=prediction)
    if weights_path:
        model.load_weights(weights_path)

    return model


def AlexNet():
	"""
	Creates a Keras Model mimicking the OverFeat Accurate model.
	The model architecture follows the specifications as mentioned
	below :

	Appendix A : Table 3
	OverFeat : Integrated Recognition, Localization and Detection
	using Convolutional Networks.
	Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob
	Fergus and Yann LeCun.
	"""

	model = Sequential()
	model.add(Convolution2D(96, 7, 7, subsample=(2,2), activation='relu',\
	input_shape=(221, 221, 1)))

	model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

	model.add(Convolution2D(256, 7, 7, activation='relu'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))

	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))

	model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same'))

	model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same'))

	model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

	model.add(Flatten())

	model.add(Dense(4096, activation='relu'))

	model.add(Dense(4096, activation='relu'))

	model.add(Dense(2, activation='softmax'))

	return model
batch_size = 64

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
        num_train_samples = len(filelist[:int(0.8*len(filelist))]) - (len(filelist[:int(0.8*len(filelist))])%batch_size)
        while 1:
		for i in range(num_train_samples):
            		img_file_name = filelist[i]
            		img = cv2.imread(img_file_name, 0)
            		img = cv2.resize(img, (221, 221))
                    	if 'positive' in img_file_name:
            		    result_Y.append(1)
            		else:
            		    result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%batch_size == 0:
            		    x_train = np.asarray(result_X)
            		    result_X = []
            		    y_train = np.asarray(result_Y)
            		    y_train = np_utils.to_categorical(y_train, 2)
                        # y_train = np.asarray(result_Y)
            		    result_Y = []
            		    x_train = x_train.reshape(x_train.shape[0], 221, 221, 1)
           	 	    yield x_train, y_train


def load_test():
    if len(filelist) != 0:
        num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % batch_size)
        remaining = len(filelist) - num_train_samples
        num_test_samples = remaining - (remaining%batch_size)
        result_X = []
        result_Y = []
	while 1:
        	for i in range(num_train_samples, num_train_samples + num_test_samples):
            		img_file_name = filelist[i]
            		img = cv2.imread(img_file_name, 0)
                    	img = cv2.resize(img, (221, 221))
            		if 'positive' in img_file_name:
                		result_Y.append(1)
            		else:
               			result_Y.append(0)
            		result_X.append(img)
            		if (i+1)%batch_size == 0:
                		x_test = np.asarray(result_X)
                		result_X = []
                		y_test = np.asarray(result_Y)
                		y_test = np_utils.to_categorical(y_test, 2)
                		# y_test = np.asarray(result_Y)
                        	result_Y = []
                		x_test = x_test.reshape(x_test.shape[0], 221, 221, 1)
                		yield x_test, y_test


	
load()
num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % batch_size)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%batch_size)



nb_classes = 2
nb_epoch = 44
# xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()
# alexnet = AlexNet_convnet(weights_path='alexnet_weights.h5')
# input = alexnet.input
# img_representation = alexnet.get_layer("dense_2").output
# classifier = Dense(2,name='classifier')(img_representation)
# classifier = Activation("softmax", name="softmax")(classifier)
# model = Model(input=input,output=classifier)
# sgd = SGD(lr=.1, decay=1.e-6, momentum=0.9, nesterov=False)
model = AlexNet()
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


