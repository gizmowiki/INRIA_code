import numpy as np
import cv2
import os
import sys

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
filepath = os.path.join("/data/stars/user/rpandey/", sys.argv[1].split("/")[3]+".avi")

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(filepath,fourcc, 20.0, (640,480))

for item in sorted(os.listdir(sys.argv[1])):
    if os.path.isdir(os.path.join(sys.argv[1], item)):
        continue
    frame = cv2.imread(os.path.join(sys.argv[1], item))
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release everything if job is finished

out.release()
print ("Video saved in %s" % filepath)
