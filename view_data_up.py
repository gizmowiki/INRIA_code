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

    random.shuffle(filelist)
    random.shuffle(filelist)

load()
chunk_index = -64000
chunk_size = 64000
max_q_size = 20
maxproc = 2
processes = []
samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
chunks = range(0, samples_per_epoch, chunk_size)
chunks.reverse()
backup_chunks = chunks
def load_train_my_generator():
    # batch_index = -64
    # batch_size = 64
    # max_q_size = 20
    # maxproc = 8

    # samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
    try:
        queue = multiprocessing.Queue(maxsize=max_q_size)

        def producer(chunk_index):
            result_X = []
            result_Y = []
            # global chunk_index
            # chunk_index += chunk_size
            jj = 0
	    im_l = []
	    print "chunk_index", chunk_index
            for i in range(chunk_size):
                img_file_name = filelist[chunk_index + i]
                im_l.append(img_file_name)
		img = cv2.imread(img_file_name, 2)
                if 'chalearn' in img_file_name:
                    img = img.astype(np.float32)
                    img /= 255
                    img *= (((float(img_file_name.split('_')[-2]) + 1) * 65536 / 4096) - 1)
                    img = img.astype(np.uint16)
                if 'positive' in img_file_name:
                    result_Y.append(1)
                else:
                    jj += 1
                    if jj % 40 == 0:
                        try:
                            gauss = np.random.normal(0.8, 0.3 ** 0.5, img.shape)
                            gauss = gauss.astype(np.uint16)
                            img += gauss
                        except:
                            print ("nhi hua")
                    result_Y.append(0)
                img = cv2.resize(img, (128, 256))
                result_X.append(img)
            # print ("from producer", len(result_Y)
            x_train = result_X
            y_train = np.asarray(result_Y)
            # x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
            queue.put((x_train, y_train, im_l))

        def start_process():
            global processes
	    global chunks
            for i in range(len(processes), maxproc):
		if not chunks:
			chunks = backup_chunks
                thread = multiprocessing.Process(target=producer(chunks.pop()))
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
for X, Y, Z in load_train_my_generator():
    k = 0
    for i in range(len(X)):
	if Z[i] not in name:
		print Z[i]
		name.append(Z[i])
	else:
		print "ye nai hai", Z[i]
        if Y[i] == 1:
		img = X[i]
        	cv2.putText(img, str(Y[i]), (2, 2), cv2.FONT_HERSHEY_SIMPLEX,
        	            fontScale=0.5,
        	            color=(65535, 65535, 65535))
        	cv2.imshow("data", img)
        	if cv2.waitKey(500) & 0xFF == ord('q'):
		    k = 1
        	    cv2.destroyAllWindows()
        	    break
    if k == 1:
        break
