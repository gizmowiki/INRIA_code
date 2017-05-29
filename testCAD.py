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

model.load_weights('/home/rpandey/people_detect/weights6/weights-improvement-epoch08.hdf5')
print("Loaded keras model and weights with time taken ", str(time.time() - start_time), "seconds")

threshold = 0.6
annotations_base_path = "/data/stars/user/sdas/CAD60/newjoint_positions"
offset = 10
write_base_path = "/home/rpandey/depth_results"
crop_id = 0
for data in ["data1", "data2", "data3", "data4"]:
	base_path = os.path.join("/data/stars/share/people_depth/people-depth/cad/", data)
	print("Now parsing for CADA data ", base_path)
	for subfolders in os.listdir(base_path):
		if os.path.isdir(os.path.join(base_path, subfolders)):
			count_images = 0
			for item in os.listdir(os.path.join(base_path, subfolders)):
				count_images += 1
			count_images /= 2
			matfile = os.path.join(annotations_base_path, subfolders, 'joint_positions.mat')
			annotations_data = sio.loadmat(matfile)
			for i in range(count_images):
				imgfilename_depth = os.path.join(base_path, subfolders, 'Depth_'+str(i+1)+'.png')
				img_depth = cv2.imread(imgfilename_depth, 0)
				img_depth = img_depth.astype(np.float32)
				img_depth -= np.min(img_depth)
				img_depth /= (np.max(img_depth) - np.min(img_depth))
				img_depth *= 255
				img_depth = img_depth.astype(np.uint8)
				imgfilename_rgb = os.path.join(base_path, subfolders, 'RGB_'+str(i+1)+'.png')
				img_rgb = cv2.imread(imgfilename_rgb)
				xmin = int(np.min(annotations_data['pos_img'][0][i]) - 2*offset)
				ymin = int(np.min(annotations_data['pos_img'][1][i]) - (2.5*offset))
				xmax = int(np.max(annotations_data['pos_img'][0][i]) + 2*offset)
				ymax = int(np.max(annotations_data['pos_img'][1][i]) + 2*offset)
				if xmin < 0:
					xmin = 0
				if ymin < 0:
					ymin = 0
				if xmax > 320 or xmax == 0:
					xmax = 320
				if ymax > 240 or ymax == 0:
					ymax = 240

				bboxa = [xmin, ymin, xmax, ymax]
				bboxes = load_data(imgfilename_depth)
				bboxes_use = []
                		max_val = -1
               			min_val = 2
               			for bbox in bboxes:
                        		if bbox[4]>max_val:
                                		max_val = bbox[4]
                        		if bbox[4]<min_val:
                                		min_val = bbox[4]
                		true_max = 0.6*(max_val - min_val)
				result_X = []
				for bb in bboxes:
					bb = [int(x) for x in bb]
					crop_img = img_depth[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
					crop_img = cv2.resize(crop_img, (224,224))
					result_X.append(crop_img)
				X = np.asarray(result_X)
				X = X.reshape(X.shape[0], 224, 224, 1)
				predictions = model.predict(X, batch_size=64)
				print (predictions)
				for jj in range(predictions.shape[0]):
					if predictions[jj][1]>=threshold and bboxes[jj][4]>=true_max:
						bb = [int(x) for x in bboxes[jj]]
						# print ("Comparing area between", bboxa, " and ", [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
						overlap = area([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]], bboxa)
						if overlap >= 0.8:
							print("ye selected", overlap)
							cv2.rectangle(img_depth, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0,255,0), thickness=2)
						else:
							# print("Ye nai selected", overlap)
							cv2.rectangle(img_depth, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255,0,0))
				# crop_img = cv2.resize(crop_img, (64, 128))
				filename = os.path.join(write_base_path, '{0:08d}.jpg'.format(crop_id))
				# cv2.imshow("",crop_img)
				cv2.imwrite(filename, img_depth)
				print("Completed writing people depth in", filename)
				crop_id += 1
				# cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0,255,0))
				# cv2.rectangle(img_depth, (xmin, ymin), (xmax, ymax), (255,255,255))
				# vis = np.concatenate((img_rgb, img_depth), axis=1)
				
				# cv2.imshow("depth", img_depth_view)
				# if cv2.waitKey(100) & 0xFF == ord('q'):
				# 	break

