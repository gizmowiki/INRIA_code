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
import matlab.engine
import threading
import json
import time
import os
import numpy as np
import cv2
import scipy.io as sio


def area(bboxa, bbbob):  # returns None if rectangles don't intersect
    dx = min(bboxa[2], bbbob[2]) - max(bboxa[0], bbbob[0])
    dy = min(bboxa[3], bbbob[3]) - max(bboxa[1], bbbob[1])
    if (dx >= 0) and (dy >= 0):
        area_a = (bboxa[3] - bboxa[1] + 1) * (bboxa[2] - bboxa[0] + 1)
        area_b = (bbbob[3] - bbbob[1] + 1) * (bbbob[2] - bbbob[0] + 1)
        return np.float((dx*dy))/np.float(max(area_a, area_b))
    else:
        return 0


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
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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


def getData(filelist_path):
    files = []
    for lines in open(filelist_path, 'rb'):
        lines = lines.strip()
        files.append(lines)
    files.sort()
    return files


def processImage(img_path, img, model, eng, image_enhancement, nms_threshold, predict_threshold, ed_bx_threshold,
                 img_shape):
    if image_enhancement[0]:
        img_scaled = cv2.GaussianBlur(img,(5,5),0)
	# img_scaled = cv2.medianBlur(img, 3)
    if image_enhancement[1]:
	# img_scaled = img.astype(np.float32)
	# img_scaled -= 65535
	# img_scaled *= -1
	# img_scaled = img_scaled.astype(np.uint16)
        hist,bins = np.histogram(img.flatten(),65536,[0,65536])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint16')
        img_scaled=cdf[img]
	# img_scaled = cv2.equalizeHist(img)
    if (not image_enhancement[0]) and (not image_enhancement[1]):
	    img_scaled=img
    # img = cv2.equalizeHist(img)
    img_scaled = img_scaled.astype(np.float32)
    img_scaled -= np.min(img_scaled)
    if (np.max(img_scaled) - np.min(img_scaled)) != 0:
         img_scaled /= (np.max(img_scaled) - np.min(img_scaled))
    img_scaled *= 65535
    img_scaled = img_scaled.astype(np.uint16)
    # print ("max_min", img_scaled.max(), img_scaled.min())
    bbox_ed_bx = eng.getEdgeBoxes(img_path)
    max_val_ed_bx = -1
    min_val_ed_bx = 2
    bbox_ed_bx_filtered = []
    result_X = []
    for bbox in bbox_ed_bx:
        if bbox[4] > max_val_ed_bx:
            max_val_ed_bx = bbox[4]
        if bbox[4] < min_val_ed_bx:
            min_val_ed_bx = bbox[4]
    true_max = ed_bx_threshold * (max_val_ed_bx - min_val_ed_bx)
    for bboxes in bbox_ed_bx:
        if bboxes[4] >= true_max:
            bbox = [int(x) for x in bboxes[:4]]
            bbox[3] += bbox[1]
            bbox[2] += bbox[0]
            bbox_ed_bx_filtered.append(bbox)
            cropped_img = img_scaled[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cropped_img = cv2.resize(cropped_img, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_CUBIC)
            result_X.append(cropped_img)


    print("No. of targets by edge box:", len(result_X))
    X = np.asarray(result_X)
    X = X.reshape(X.shape[0], img_shape[0], img_shape[1], 1)
    predictions = model.predict(X)
    selected = 0
    print("Predicted Labels\n", predictions)
    selected_bbox = []
    for i in range(len(predictions)):
    	if predictions[i][1] >= predict_threshold:
    	    selected += 1
    	    selected_bbox.append(np.asarray(bbox_ed_bx_filtered[i]))
    bbox_nms = non_max_suppression_fast(np.asarray(selected_bbox), nms_threshold)
    # img_scaled *= 65535
    # img_scaled = img_scaled.astype(np.uint16)
    for bbox in bbox_nms:
	print ("bbox", bbox)
        cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535))

    return img_scaled


def loadModel(weightsPath, image_shape):
    batch_size = 64
    nb_classes = 2
    nb_epoch = 55

    img_rows, img_cols = image_shape[0], image_shape[1]

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
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('softmax'))

    model.load_weights(weightsPath)
    return model


def getModelScaled(weightsPath, img_shape):
	nb_filters = 72
	nb_classes = 2
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

	model.load_weights(weightsPath)
	
	return model


def getModelNewNegative(weightsPath, img_shape):
	nb_classes = 2
	nb_filters = 96

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
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.75))
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))
	
	model.load_weights(weightsPath)
	
	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	return model


def getModelNewFitNoFitNostalgia(weightsPath, img_shape):
	nb_classes = 2
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
	# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
	# model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
	model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
	model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
								 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.75))
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))
	model.load_weights(weightsPath)
	# model.compile(loss='categorical_crossentropy',
	# 			  optimizer='rmsprop',
	# 			  metrics=['accuracy'])

	return model


def getModelOptimized(weightsPath, img_shape):
    nb_classes = 2
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
    model.load_weights(weightsPath)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def getModelMinimal(weightsPath, img_shape):
        nb_filters = 72
	nb_classes = 2
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
        model.load_weights(weightsPath)
        # model.compile(loss='categorical_crossentropy',
        #                           optimizer='SGD',
        #                           metrics=['accuracy'])

        return model



def getModelPostNostalgia(weightsPath, img_shape):
        nb_filters = 72
	nb_classes = 2
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
	
	model.load_weights(weightsPath)

        model.compile(loss='categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])

        return model


def loadModelTwelve(weightsPath, image_shape):
    batch_size = 64
    nb_classes = 2
    nb_epoch = 66

    img_rows, img_cols = image_shape[0], image_shape[1]

    nb_filters = 72

    pool_size = (2, 2)

    kernel_size = (3, 3)

    input_shape = (img_rows, img_cols, 1)

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

    model.load_weights(weightsPath)

    return model


def show(filelist_path, video, predict_threshold, ed_bx_threshold, nms_threshold, image_enhancement):
    start_time = time.time()
    eng = matlab.engine.start_matlab()
    eng.addpath(r'/home/rpandey/people_detect/edge_boxes/edges', nargout=0)
    print("Loaded MATLAB engine with time ", str(time.time() - start_time), "seconds")

    files = getData(filelist_path)
    img_shape = (64, 128)
    weights_path = '/home/rpandey/people_detect/weights_fit_optimized_part2/weights-improvement-24-0.015.hdf5'

    start_time = time.time()
    model = getModelOptimized(weights_path, img_shape)
    print("Loaded keras model and weights with time taken ", str(time.time() - start_time), "seconds")

    if not video:
        for img_path in files:
            img = cv2.imread(img_path, 2)
            img_detect = processImage(img_path, img=img, model=model, eng=eng, image_enhancement=image_enhancement,
                         		nms_threshold=nms_threshold, predict_threshold=predict_threshold,
                         		ed_bx_threshold=ed_bx_threshold, img_shape=img_shape)
            cv2.imshow("Detected person", img_detect)
	    if cv2.waitKey(100) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
	    	break

    else:
        for video_path in files:
            cap = cv2.VideoCapture(video_path)
            while 1:
                ret, frame = cap.read()
                if not ret:
                    break
		img_path = '/tmp/input_img.jpg'
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = frame.astype(np.uint8)
		
		cv2.imwrite(img_path, frame)
                img_detect = processImage(img_path, img=frame, model=model, eng=eng, image_enhancement=image_enhancement,
                             			nms_threshold=nms_threshold, predict_threshold=predict_threshold,
                             			ed_bx_threshold=ed_bx_threshold, img_shape=img_shape)
      	        cv2.imshow("Detected person", img_detect)
    		if cv2.waitKey(100) & 0xFF == ord('q'):
        		break

    return


class TestData:
    def __init__(self, filelist_path, videos=False, predict_threshold=0.999, nms_threshold=0.5,
                 ed_bx_threshold=0.7, image_enhancement=[False, False]):
        show(filelist_path, video=videos, predict_threshold=predict_threshold, ed_bx_threshold=ed_bx_threshold, nms_threshold=nms_threshold, image_enhancement=image_enhancement)
        return


