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
# from keras.utils.visualize_util import plot
from KerasLayers.Custom_layers import LRN2D
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


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


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
    random.shuffle(files)
    random.shuffle(files)
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




start_time = time.time()
batch_size = 64
nb_classes = 2
nb_epoch = 55

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
model.add(Dropout(0.25))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, init='glorot_normal'))
model.add(Activation('softmax'))


model.load_weights('/home/rpandey/people_detect/weights8/weights-improvement-epoch09.hdf5')
print("Loaded keras model and weights with time taken ", str(time.time() - start_time), "seconds")

threshold = 0.99
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
    true_max = 0.7*(max_val - min_val)
    for bbox_t in bboxes:
        bbox = [int(x) for x in bbox_t[:4]]
        if bbox_t[4] >= true_max: 
            bboxes_use.append(bbox)
            crops = img_scaled[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            crops = cv2.resize(crops, (64,128), interpolation = cv2.INTER_CUBIC)
            crops = crops.astype(np.float32)
	    crops /= 255
	    result_X.append(crops)
    print ("no of target test samples by edge box", len(result_X))
    X = np.asarray(result_X)
    X = X.reshape(X.shape[0], 64, 128, 1)
    predictions = model.predict(X, batch_size=64)
    print ("img", imgpath)      
    selected = 0
    total_bb = []
    for i in range(len(predictions)):
        print (predictions[i])
        if predictions[i][1]>=threshold:
            selected += 1
            print ("bbox", bboxes_use[i])
            bbox = bboxes_use[i]
            bbox = [int(x) for x in bbox[:4]]
            bbox[3] += bbox[1]
            bbox[2] += bbox[0]
            total_bb.append(np.asarray(bbox))
    actual_bb = non_max_suppression_fast(np.asarray(total_bb), 0.45)
    for bbox in actual_bb:
	cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,255))
    
    print ("Total targets from bbox proposal %d and total detected person %d" % (X.shape[0], selected))
    cv2.imshow('finally', img_scaled)
    if cv2.waitKey(100) & 0xFF == ord('q'):
            break

