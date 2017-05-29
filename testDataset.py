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
        return np.float((dx*dy))/np.float(area_b)
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

def processImageGT(img_path, img, model, eng, image_enhancement, nms_threshold, predict_threshold, ed_bx_threshold, img_shape, annotation_list, max_overlap):
    # im_st_time = time.time()
    if image_enhancement[0]:
        img_scaled = cv2.GaussianBlur(img,(5,5),0)
        # img_scaled = cv2.medianBlur(img, 3)
    if image_enhancement[1]:
        # img_scaled = img.astype(np.float32)
        # img_scaled -= 65535
        # img_scaled *= -1
        # img_scaled = img_scaled.astype(np.uint16)
        hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')
        img_scaled = cdf[img]
        # img_scaled = cv2.equalizeHist(img)
    if (not image_enhancement[0]) and (not image_enhancement[1]):
        img_scaled=img

    img_scaled = img_scaled.astype(np.float32)
    img_scaled -= np.min(img_scaled)
    if (np.max(img_scaled) - np.min(img_scaled)) != 0:
         img_scaled /= (np.max(img_scaled) - np.min(img_scaled))
    img_scaled *= 65535
    img_scaled = img_scaled.astype(np.uint16)
    # print ("max_min", img_scaled.max(), img_scaled.min())
    # bbox_ed_bx = eng.getEdgeBoxes(img_path)
    # max_val_ed_bx = -1
    # min_val_ed_bx = 2
    # bbox_ed_bx_filtered = []
    # result_X = []
    # for bbox in bbox_ed_bx:
    #     if bbox[4] > max_val_ed_bx:
    #         max_val_ed_bx = bbox[4]
    #     if bbox[4] < min_val_ed_bx:
    #         min_val_ed_bx = bbox[4]
    # true_max = ed_bx_threshold * (max_val_ed_bx - min_val_ed_bx)
    # for bboxes in bbox_ed_bx:
    #     if bboxes[4] >= true_max:
    #         bbox = [int(x) for x in bboxes[:4]]
    #         bbox[3] += bbox[1]
    #         bbox[2] += bbox[0]
    #         bbox_ed_bx_filtered.append(bbox)
    #         cropped_img = img_scaled[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #         cropped_img = cv2.resize(cropped_img, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_CUBIC)
    #         result_X.append(cropped_img)
    # print("No. of targets by edge box:", len(result_X))
    result_X = []
    for i in range(len(annotation_list)):
        bbox = annotation_list[i]
        cropped_img = img_scaled[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped_img = cv2.resize(cropped_img, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_CUBIC)
        result_X.append(cropped_img)
    X = np.asarray(result_X)
    X = X.reshape(X.shape[0], img_shape[0], img_shape[1], 1)
    predictions = model.predict(X)
    selected = 0
    # print("Predicted Labels\n", predictions)
    selected_bbox = []
    result_tuple = {"true_positive": [], "false_positive": []}
    for i in range(len(predictions)):
        if predictions[i][1] >= predict_threshold:
            result_tuple["true_positive"].append(annotation_list[i]+[str(predictions[i][1])])
        else:
            result_tuple["false_positive"].append(annotation_list[i]+[str(predictions[i][1])])
            # selected += 1
            # selected_bbox.append(np.asarray(bbox_ed_bx_filtered[i]))
            # cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535), thickness=2)
    # print ("time taken to process %s image was %.6f" % (img_path, float(time.time() - im_st_time)))
    # bbox_nms = non_max_suppression_fast(np.asarray(selected_bbox), nms_threshold)
    # img_scaled *= 65535
    # img_scaled = img_scaled.astype(np.uint16)
    # list_chk = annotation_list
    # true_positive = 0
    # false_positive = 0

    # for bbox in bbox_nms:
    #     overlaps = map(lambda u: area(bbox, u), annotation_list)
    #     print (overlaps)
    #     if not all(x <= max_overlap for x in overlaps):
    #         true_positive += 1
    #         cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535), thickness=2)
    #         list_chk = [y for y in list_chk if y not in [annotation_list[overlaps.index(x)] for x in overlaps if overlaps >= max_overlap]]
    #         # print ("bbox", bbox)
    #     else:
    #         false_positive += 1
    #         cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535))
    # false_negative = len(list_chk)
    # result_tuple = [img_path, true_positive, false_positive, false_negative]
    
    return img_scaled, result_tuple


def processImageKTPHugues(img_path, img, model, eng, bbox_ed_bx_filtered, image_enhancement, nms_threshold, predict_threshold, ed_bx_threshold, img_shape, annotation_list, max_overlap):
    if image_enhancement[0]:
        img_scaled = cv2.GaussianBlur(img,(5,5),0)
        # img_scaled = cv2.medianBlur(img, 3)
    if image_enhancement[1]:
        # img_scaled = img.astype(np.float32)
        # img_scaled -= 65535
        # img_scaled *= -1
        # img_scaled = img_scaled.astype(np.uint16)
        hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')
        img_scaled = cdf[img]
        # img_scaled = cv2.equalizeHist(img)
    if (not image_enhancement[0]) and (not image_enhancement[1]):
        img_scaled=img

    img_scaled = img_scaled.astype(np.float32)
    img_scaled -= np.min(img_scaled)
    if (np.max(img_scaled) - np.min(img_scaled)) != 0:
         img_scaled /= (np.max(img_scaled) - np.min(img_scaled))
    # img_scaled *= 65535
    # img_scaled = img_scaled.astype(np.uint16)
    result_X = []
    for bbox in bbox_ed_bx_filtered:
        cropped_img = img_scaled[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped_img = cv2.resize(cropped_img, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_CUBIC)
        result_X.append(cropped_img)
    # print("No. of targets by edge box:", len(result_X))
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
    list_chk = annotation_list
    true_positive = 0
    false_positive = 0
    img_scaled *= 65535
    img_scaled = img_scaled.astype(np.uint16)
    for bbox in bbox_nms:
        overlaps = map(lambda u: area(bbox, u), annotation_list)
        print (overlaps)
        if not all(x <= max_overlap for x in overlaps):
            true_positive += 1
            cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535), thickness=2)
            list_chk = [y for y in list_chk if y not in [annotation_list[overlaps.index(x)] for x in overlaps if overlaps >= max_overlap]]
            # print ("bbox", bbox)
        else:
            false_positive += 1
            cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535))
    false_negative = len(list_chk)
    result_tuple = [img_path, true_positive, false_positive, false_negative]
    
    return img_scaled, result_tuple


def processImage(img_path, img, model, eng, image_enhancement, nms_threshold, predict_threshold, ed_bx_threshold, img_shape, annotation_list, max_overlap):
    if image_enhancement[0]:
        img_scaled = cv2.GaussianBlur(img,(5,5),0)
        # img_scaled = cv2.medianBlur(img, 3)
    if image_enhancement[1]:
        # img_scaled = img.astype(np.float32)
        # img_scaled -= 65535
        # img_scaled *= -1
        # img_scaled = img_scaled.astype(np.uint16)
        hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')
        img_scaled = cdf[img]
        # img_scaled = cv2.equalizeHist(img)
    if (not image_enhancement[0]) and (not image_enhancement[1]):
        img_scaled=img

    img_scaled = img_scaled.astype(np.float32)
    img_scaled -= np.min(img_scaled)
    if (np.max(img_scaled) - np.min(img_scaled)) != 0:
         img_scaled /= (np.max(img_scaled) - np.min(img_scaled))
    # img_scaled *= 65535
    # img_scaled = img_scaled.astype(np.uint16)
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
    # print("No. of targets by edge box:", len(result_X))
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
    img_scaled *= 65535
    img_scaled = img_scaled.astype(np.uint16)
    list_chk = annotation_list
    true_positive = 0
    false_positive = 0

    for bbox in bbox_nms:
        overlaps = map(lambda u: area(bbox, u), annotation_list)
        print (overlaps)
        if not all(x <= max_overlap for x in overlaps):
            true_positive += 1
            cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535), thickness=2)
            list_chk = [y for y in list_chk if y not in [annotation_list[overlaps.index(x)] for x in overlaps if x >= max_overlap]]
            # print ("bbox", bbox)
        else:
            false_positive += 1
            cv2.rectangle(img_scaled, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (65535, 65535, 65535))
    false_negative = len(list_chk)
    result_tuple = [img_path, true_positive, false_positive, false_negative]
    
    return img_scaled, result_tuple


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


def getModelPostNostalgia1(weightsPath, img_shape):
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


def create_model_ALEXNET(img_shape):
    NB_CLASS = 2         # number of classes
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    ALPHA = 0.0001
    BETA = 0.75
    GAMMA = 0.1
    DROPOUT = 0.5
    WEIGHT_DECAY = 0.0005
    LRN2D_norm = True
    DIM_ORDERING = 'tf'

    def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,border_mode='same', subsample=(1, 1),activation='relu', LRN2D_norm=True,weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
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
        x = Convolution2D(nb_filter, nb_row, nb_col,subsample=subsample,activation=activation,border_mode=border_mode,W_regularizer=W_regularizer,b_regularizer=b_regularizer,bias=False,dim_ordering=dim_ordering)(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
        if LRN2D_norm:
            x = LRN2D(alpha=ALPHA, beta=BETA)(x)
            x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
        return x
    
    
    
    # Define image input layer
    INP_SHAPE = (img_shape[0], img_shape[1], 1)  # 3 - Number of RGB Colours
    img_input = Input(shape=INP_SHAPE)
    CONCAT_AXIS = 3
    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(img_input, 3, 11, 11, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(4, 4), pool_size=(4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    # Channel 1 - Cov Net Layer 6
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    # Channel 1 - Cov Net Layer 8
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    # Final Channel - Cov Net 9
    x = Dense(output_dim=NB_CLASS, activation='softmax')(x)
    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING


def Alexnet_BVLC(weightsPath, img_shape):
    xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model_ALEXNET(img_shape)
    model = Model(input=img_input, output=[xy])
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    # model.summary()
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



def getModelSunday(weightsPath, img_shape):
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
        model.add(Dropout(0.75))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
        model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
        model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
        # model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
        #                                                        gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Dense(nb_classes, init='glorot_normal'))
        model.add(Activation('softmax'))
	model.load_weights(weightsPath)
        # model.compile(loss='categorical_crossentropy',
        #                           optimizer='SGD',
        #                           metrics=['accuracy'])

        return model



def getModelNewFitNegativesNostalgia(weightsPath, img_shape):
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
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_3", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_3"))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name="conv_4", init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_4"))
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

def getModelPostNostalgia2(weightsPath, img_shape):
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
        #                         optimizer='rmsprop',
        #                         metrics=['accuracy'])

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


def getKTPProposal(index):
    proposal_list = []
    base_path = "/home/hthomas/Final_Data/KTP_devkit/HT2016/Proposals"
    for item in open(os.path.join(base_path, "{0:06d}.txt".format(index)), 'rb'):
        item = item.strip()
        item = item.split(" ")
        bbox = [int(x) for x in item]
        asp_rat = (bbox[2]-bbox[0])/float(bbox[3]-bbox[1])
        if asp_rat < 0.6 and asp_rat >= 0.3:
            proposal_list.append(bbox)

    return proposal_list 


def show(filelist_path, video, base_out_path, predict_threshold, ed_bx_threshold,
         nms_threshold, image_enhancement, annotation_dict, max_overlap):
    start_time = time.time()
    eng = matlab.engine.start_matlab()
    eng.addpath(r'/home/rpandey/people_detect/edge_boxes/edges', nargout=0)
    print("Loaded MATLAB engine with time ", str(time.time() - start_time), "seconds")

    if not os.path.exists(base_out_path):
        os.makedirs(os.path.join(base_out_path, "images"))
        os.makedirs(os.path.join(base_out_path, "results"))

    files = getData(filelist_path)
    img_shape = (180, 360)

    result = []
    result_dict = {}

    #weights_path = '/data/stars/user/rpandey/weights_backup/weights_fit_ed_bx_post_nostalgia_2/weights-improvement-58-0.001.hdf5'
    weights_path = '/data/stars/user/rpandey/weights_backup/weights_fit_alexnet_big/weights-improvement-01-0.1269.hdf5'
    start_time = time.time()
    model = Alexnet_BVLC(weights_path, img_shape)
    print("Loaded keras model and weights with time taken ", str(time.time() - start_time), "seconds")
    im_st_time = time.time()
    full_time = 0.0000
    index_ktp = 0
    if not video:
        for img_path in files:
	    index_ktp += 1
	    bbox_ed_bx_filtered = getKTPProposal(index_ktp)
            img = cv2.imread(img_path, 2)
            if not img_path in annotation_dict.keys():
                continue
            img_detect, result_data = processImage(img_path, img=img, model=model, eng=eng, # bbox_ed_bx_filtered=bbox_ed_bx_filtered, 
                                                   image_enhancement=image_enhancement, nms_threshold=nms_threshold,
                                                   predict_threshold=predict_threshold, ed_bx_threshold=ed_bx_threshold,
                                                   img_shape=img_shape, max_overlap=max_overlap,
                                                   annotation_list=annotation_dict[img_path])
	    tm_tk = float(time.time() - im_st_time)
	    full_time += tm_tk
	    print ("time taken to process %s image was %.6f" % (img_path, tm_tk))
	    im_st_time = time.time()
            if  type(result_data) is list:
                result.append(result_data)
                # img_file_name = os.path.join(base_out_path, "images", img_path.split("/")[-1])
                # cv2.imwrite(img_file_name, img_detect)
                print ("Image path: %s True Positives: %d False positives: %d False Negatives %d" % (result_data[0],
                                                                                                     result_data[1],
                                                                                                     result_data[2],
                                                                                                     result_data[3]))
            else:
                result_dict[img_path] = result_data
            # cv2.imshow("Detected person", img_detect)
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
        if not result_dict:
            with open(os.path.join(base_out_path, "results", "result_low_thresh.csv"), 'w') as out_file:
                for item in result:
                    itemx = [str(x) for x in item]
                    data = ",".join(itemx)
                    out_file.write("%s\n" % data)
        else:
            with open(os.path.join(base_out_path, "results", "result.json"), 'w') as out_file:
                json.dump(result_dict, out_file)

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
    print ("chala hun tum tak chalunga tum tk %.6f seconds" % full_time)
    return


class TestData:
    def __init__(self, filelist_path, annotation_dict, base_out_path, videos=False,
                 predict_threshold=0.999, nms_threshold=0.45, ed_bx_threshold=0.7, max_overlap=0.55,
                 image_enhancement=[False, False]):
        show(filelist_path, video=videos, base_out_path=base_out_path, predict_threshold=predict_threshold,
             ed_bx_threshold=ed_bx_threshold, nms_threshold=nms_threshold, image_enhancement=image_enhancement,
             annotation_dict=annotation_dict, max_overlap=max_overlap)
        return


