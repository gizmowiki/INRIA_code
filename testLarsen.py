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
# from skimage import io
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
import os
import scipy.io as sio
import cv2
import matlab.engine
import numpy as np

start_time = time.time()
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/rpandey/people_detect/edge_boxes/edges',nargout=0)
print("Loaded MATLAB engine with time ", str(time.time() - start_time), "seconds")
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


def getData(filelist):
    files = []
    for imgpath in open(filelist):
        imgpath = imgpath.strip()
        files.append(imgpath)
    files.sort()
    return files

def area(bboxa, bbbob):  # returns None if rectangles don't intersect
    dx = min(bboxa[2], bbbob[2]) - max(bboxa[0], bbbob[0])
    dy = min(bboxa[3], bbbob[3]) - max(bboxa[1], bbbob[1])
    if (dx >= 0) and (dy >= 0):
        area_a = (bboxa[3] - bboxa[1] + 1) * (bboxa[2] - bboxa[0] + 1)
        return np.float((dx*dy))/np.float(area_a)
    else:
        return 0

def load_data(imgpath):
	bboxes = eng.getEdgeBoxes(imgpath)
	return bboxes


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

start_time = time.time()
x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

model = Model(input=img_input,
                  output=[x])

model.load_weights('/home/rpandey/people_detect/weights6/weights-improvement-epoch01.hdf5')
print("Loaded keras model and weights with time taken ", str(time.time() - start_time), "seconds")

threshold = 0.5
annotations_base_path = "/data/stars/user/sdas/CAD60/newjoint_positions"
offset = 10
write_base_path = "/home/rpandey/depth_results"
crop_id = 0
files = getData('/data/stars/share/people_depth/people-depth/larsen_inria/filelist_test.txt')
for imgpath in files:
    print ("Parsing image", imgpath)
    img = cv2.imread(imgpath, 0)
    result_X = []
    bboxes = load_data(imgpath)
    img_scaled = img.astype(np.float32)
    img_scaled -= np.min(img_scaled)
    img_scaled /= (np.max(img_scaled) - np.min(img_scaled))
    img_scaled *= 255
    img_scaled = img_scaled.astype(np.uint8)
    # img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    bboxes_use = []
    max_val = -1
    min_val = 2
    for bbox in bboxes:
        if bbox[4]>max_val:
            max_val = bbox[4]
        if bbox[4]<min_val:
            min_val = bbox[4]
    true_max = 0.6*(max_val - min_val)  
    for bbox_t in bboxes:
        bbox = [int(x) for x in bbox_t[:4]]
        if bbox_t[4] >= true_max: 
            bboxes_use.append(bbox)
            crops = img_scaled[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            crops = cv2.resize(crops, (224,224), interpolation = cv2.INTER_CUBIC)
	    crops = crops.astype(np.float32)
	    crops /= 255
            result_X.append(crops)
    X = np.asarray(result_X)
    X = X.reshape(X.shape[0], 224, 224, 1)
    predictions = model.predict(X, batch_size=64)
    print ("img", imgpath)      
    selected = 0
    for i in range(len(predictions)):
        print (predictions[i])
        if predictions[i][1]>=threshold:
            selected += 1
            print ("bbox", bboxes_use[i])
            bbox = bboxes_use[i]
            bbox = [int(x) for x in bbox]
            cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255,255,255))
    
    print ("Total targets from bbox proposal %d and total detected person %d" % (X.shape[0], selected))
    cv2.imshow('finally', img_scaled)
    if cv2.waitKey(100) & 0xFF == ord('q'):
            break

