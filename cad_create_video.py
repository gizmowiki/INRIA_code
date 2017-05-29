import numpy as np
import cv2
import os
import sys

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
filepath = os.path.join("/home/rpandey/", "updated_"+sys.argv[1].split("/")[-3]+"_"+sys.argv[1].split("/")[-2]+".avi")

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(filepath,fourcc, 20.0, (320,240))

for i in range(2000):
    item = "Depth_"+str(i)+".png"
    if not os.path.exists(os.path.join(sys.argv[1], item)):
        continue
    frame = cv2.imread(os.path.join(sys.argv[1], item))
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release everything if job is finished

out.release()
print ("Video saved in %s" % filepath)
