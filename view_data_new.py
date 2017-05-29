import os
import multiprocessing
import cv2
import random
import time
import numpy as np


filelist = []
# base_path = "/dev/shm/people_detect"
# mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'), 0)
def load():
    global filelist
    base_path = "/local/people_depth"
    base_path_neg = "/local/people_depth/"
    # mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'))
    for lines in open(os.path.join(base_path, 'files_positive.txt')):
        lines = lines.strip()
        filelist.append(lines)
    for lines in open(os.path.join(base_path_neg, 'files_negative.txt')):
        lines = lines.strip()
        filelist.append(lines)
    filelist = filelist[0:160000]
    random.shuffle(filelist)
    random.shuffle(filelist)

load()
chunk_index = -512
chunk_size = 64

maxproc = 6
processes = []
samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
max_q_size = int(samples_per_epoch/chunk_size) * maxproc
chunks = range(0, samples_per_epoch, chunk_size)
print chunks
backup_chunks = chunks
j=0
def load_train_my_generator():
    # batch_index = -64
    # batch_size = 64
    # max_q_size = 20
    # maxproc = 8

    # samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
    try:
        queue = multiprocessing.Queue(maxsize=max_q_size)

        def producer(chunk_index):
	    im_l = []
            for i in range(chunk_size):
		im_l.append(filelist[chunk_index + i])
            queue.put(im_l)

        def start_process():
            global processes
	    global chunks
	    global j
            for i in range(len(processes), maxproc):
		j = (j+1)% len(chunks)
		if not chunks:
			print "oo halo"
			chunks = backup_chunks
                thread = multiprocessing.Process(target=producer(chunks[j]))
                time.sleep(0.01)
                thread.start()
            processes.append(thread)

        while True:
            processes = [p for p in processes if p.is_alive()]
            if len(processes) < maxproc:
                start_process()
            yield queue.get()
    except:
        print("Finishing")
        global processes
        for th in processes:
            th.terminate()
            queue.close()
        raise

name = []
k = 0
jj = 0
for Z in load_train_my_generator():
    if k == samples_per_epoch:
	break
    for i in range(len(Z)):
	jj += 1
	# if k == samples_per_epoch:
	# 	break
	if Z[i] not in name:
		print Z[i], k, jj
		name.append(Z[i])
	else:
		print "ye same hai", Z[i] 
    k += chunk_size
