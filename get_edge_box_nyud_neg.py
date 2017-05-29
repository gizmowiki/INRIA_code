import matlab.engine
import numpy as np
import os
import time
import cv2



def area(bboxa, bbbob):  # returns None if rectangles don't intersect
    dx = min(bboxa[2], bbbob[2]) - max(bboxa[0], bbbob[0])
    dy = min(bboxa[3], bbbob[3]) - max(bboxa[1], bbbob[1])
    if (dx >= 0) and (dy >= 0):
        area_a = (bboxa[3] - bboxa[1] + 1) * (bboxa[2] - bboxa[0] + 1)
        return np.float((dx*dy))/np.float(area_a)
    else:
        return 0

start_time = time.time()
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/rpandey/people_detect/edge_boxes/edges', nargout=0)
print("Loaded MATLAB engine with time ", str(time.time() - start_time), "seconds")
base_out_path = "/local/people_depth"


i_neg = 128163
min_width = 40
ed_bx_threshold = 0.8
nyud_filelist_path = "/data/stars/share/people_depth/people-depth/files_nyud.txt"

for imgfiles in open(nyud_filelist_path):
	imgfiles = imgfiles.strip()
	try:
		img = cv2.imread(imgfiles, 2)
		img = img.astype(np.float32)
	except:
		continue
	img -= img.min()
	if (img.max() - img.min())!= 0:
		img /= (img.max() - img.min())
	img *= 65535
	img = img.astype(np.uint16)
	img_neg = 65535 - img
	bbox_ed_bx = eng.getEdgeBoxes(imgfiles)
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
	print ("\n for image %s" % imgfiles)
	for bboxes in bbox_ed_bx:
		if bboxes[2] >= min_width and bboxes[4]>=true_max:
			i_neg += 1
			bbox = [int(x) for x in bboxes[:4]]
			bbox[3] += bbox[1]
			bbox[2] += bbox[0]
			cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
			cropped_img_flip = np.fliplr(cropped_img)
			neg_crooped_img = img_neg[bbox[1]:bbox[3], bbox[0]:bbox[2]]
			neg_crooped_img_flip = np.fliplr(neg_crooped_img)
			filename = os.path.join(base_out_path, "negatives_nyud", "edge_{0:09d}.png".format(i_neg))
			cv2.imwrite(filename, cropped_img)
			filename_flip = os.path.join(base_out_path, "negatives_nyud", "edge_flip_{0:09d}.png".format(i_neg))
			cv2.imwrite(filename_flip, cropped_img_flip)
			neg_filename = os.path.join(base_out_path, "negatives_nyud", "edge_neg_{0:09d}.png".format(i_neg))
			cv2.imwrite(neg_filename, neg_crooped_img)
			neg_filename_flip = os.path.join(base_out_path, "negatives_nyud", "edge_neg_flip_{0:09d}.png".format(i_neg))
			cv2.imwrite(neg_filename_flip, neg_crooped_img_flip)
			print (i_neg),


