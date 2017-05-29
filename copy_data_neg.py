import cv2
import numpy as np
import os

base_path = "/local/people_depth/negatives_upd"

for item in open('/local/people_depth/files_nyud.txt'):
    item = item.strip()
    img = cv2.imread(item, 2)
    img = img.astype(np.float32)
    img -= 65535
    img *= -1
    img = img.astype(np.uint16)
    filename = os.path.join(base_path, "neg_"+item.split('/')[-1])
    img_flip = np.fliplr(img)
    filename_flip = os.path.join(base_path, "flip_" + item.split('/')[-1])
    cv2.imwrite(filename, img)
    cv2.imwrite(filename_flip, img_flip)
    print ("Completed image ", item)

