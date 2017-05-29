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



def area(bboxa, bbbob):  # returns None if rectangles don't intersect
    dx = min(bboxa[2], bbbob[2]) - max(bboxa[0], bbbob[0])
    dy = min(bboxa[3], bbbob[3]) - max(bboxa[1], bbbob[1])
    if (dx >= 0) and (dy >= 0):
        area_a = (bboxa[3] - bboxa[1] + 1) * (bboxa[2] - bboxa[0] + 1)
	area_b = (bbbob[3] - bbbob[1] + 1) * (bbbob[2] - bbbob[0] + 1)
        return np.float((dx*dy))/np.float(max(area_a, area_b))
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


model.load_weights('/home/rpandey/people_detect/weights7/weights-improvement-epoch30.hdf5')
print("Loaded keras model and weights with time taken ", str(time.time() - start_time), "seconds")

threshold = 0.999
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
					crop_img = cv2.resize(crop_img, (64,128))
                    			crop_img = crop_img.astype(np.float32)
                    			crop_img /= 255
					result_X.append(crop_img)
				X = np.asarray(result_X)
				X = X.reshape(X.shape[0], 64, 128, 1)
				predictions = model.predict(X, batch_size=64)
				print (predictions.shape)
				sel_b = [320, 240, 0, 0]
				total_bb = []
				for jj in range(predictions.shape[0]):
					if predictions[jj][1]>=threshold and bboxes[jj][4] >=true_max:
						bb = [int(x) for x in bboxes[jj][:4]]
						bb[2] += bb[0]
						bb[3] += bb[1]
						total_bb.append(np.asarray(bb))
				actual_bb = non_max_suppression_fast(np.asarray(total_bb), 0.6)
				for bb in actual_bb:
					overlap = area(bboxa, bb)
					if overlap >= 0.55:
						print ("ye selected", overlap)
						cv2.rectangle(img_depth, (bb[0], bb[1]), (bb[2], bb[3]), (255,255,255), thickness=2)
					else:
						print ("ye nai selected", overlap)
						cv2.rectangle(img_depth, (bb[0], bb[1]), (bb[2], bb[3]), (255,255,255))
					
						# print ("Comparing area between", bboxa, " and ", [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
						# overlap_a = area([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]], bboxa)
						# overlap_b = area(bboxa, [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
						# if overlap_a >= 0.8 or overlap_b >= 0.6:
						# 	print("ye selected", [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
						# 	if bb[0] < sel_b[0]:
						# 		sel_b[0] = bb[0]
						# 	if bb[1] < sel_b[1]:
						# 		sel_b[1] = bb[1]
						# 	if (bb[0] + bb[2]) > sel_b[2]:
						# 		sel_b[2] = bb[0] + bb[2]
						# 	if (bb[1] + bb[3]) > sel_b[3]:
						# 		sel_b[3] = bb[1] + bb[3]
										
							# cv2.putText(img_depth, '{0:.2f}'.format(predictions[jj][1]), (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX,
                           				# 		fontScale=0.3,
                            				# 		color=(0, 255, 255))
							# cv2.rectangle(img_depth, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255,255,255), thickness=2)
						# else:
						# 	print("Ye nai selected", overlap_a)
							# cv2.rectangle(img_depth, (bboxa[0], bboxa[1]), (bboxa[2], bboxa[3]), (0,0,0))
						# 	cv2.rectangle(img_depth, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255,255,255))

				# if sel_b[0]!= 320:
					# cv2.rectangle(img_depth, (bboxa[0], bboxa[1]), (bboxa[2], bboxa[3]), (0,0,0))
				# 	cv2.rectangle(img_depth, (sel_b[0],sel_b[1]), (sel_b[2], sel_b[3]), (255,255,255), thickness=2)
				# crop_img = cv2.resize(crop_img, (64, 128))
				# filename = os.path.join(write_base_path, '{0:08d}.jpg'.format(crop_id))
				# cv2.imshow("",crop_img)
				# cv2.imwrite(filename, img_depth_view)
				# print("Completed writing people depth in", filename)
				# crop_id += 1
				# cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0,255,0))
				# cv2.rectangle(img_depth, (xmin, ymin), (xmax, ymax), (255,255,255))
				# vis = np.concatenate((img_rgb, img_depth), axis=1)
				img_depth = cv2.resize(img_depth, (640,480))				
				cv2.imshow("depth", img_depth)
				if cv2.waitKey(100) & 0xFF == ord('q'):
					break

