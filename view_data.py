import cv2
import os
import multiprocessing
import cv2
import numpy as np
import random
import time


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
chunk_index = -32
chunk_size = 32
max_q_size = 2
maxproc = 22
processes = []
samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
chunks = range(0, samples_per_epoch, chunk_size)
backup_chunks = chunks
chunks.reverse()
i = 2
def load_train_my_generator():
    # batch_index = -64
    # batch_size = 64
    # max_q_size = 20
    # maxproc = 8
    # samples_per_epoch = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
    try:
        queue = multiprocessing.Queue(maxsize=max_q_size)

        def producer():
            result_X = []
            result_Y = []
	    img_file_list = []
	    # chunk_index = chunks[random.randint(0, len(chunks)-2)]
	    print ("len", len(chunks))
	    # if not chunks:
		# print "hulle hulare"
		# chunks = backup_chunks
	    global chunks
	    if chunks:
	     	chunk_index = chunks.pop()
            jj = 0
	    print ("chunk", chunk_index)
            for i in range(chunk_size):
                img_file_name = filelist[chunk_index + i]
		# print img_file_name
		img_file_list.append(img_file_name)
                img = cv2.imread(img_file_name, 2)
                if 'chalearn' in img_file_name:
                    img = img.astype(np.float32)
                    img /= 255
                    img *= (((float(img_file_name.split('_')[-2]) + 1) * 65536 / 4096) - 1)
                    img = img.astype(np.uint16)
		img = img.astype(np.float32)
		img -= img.min()
		if (img.max() - img.min()) != 0:
			img /= (img.max() - img.min())
		img *= 65535
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
	    result_X = []
	    result_Y = []
            # x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
            queue.put((x_train, y_train, img_file_list))
	    img_file_list = []

        def start_process():
            global processes
	    global chunks
            for i in range(len(processes), maxproc):
                if not chunks:
			chunks = backup_chunks
		thread = multiprocessing.Process(target=producer())
                time.sleep(0.01)
                thread.start()
		# global chunks
		# if not chunks:
		# 	print ("hulle hulare")
		# 	chunks = backup_chunks
		# chunks.pop()
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

kk = 0
f_name = []
for X, Y, Z in load_train_my_generator():
    print "yha tk", len(X)
    for i in range(len(X)):
	name = Z[i].split('/')[-1]
	if name not in f_name:
		print "hulle", name
		f_name.append(name)
	else:
		print "ye same h", name
        img = X[i]
	if Y[i] ==1:
        	cv2.putText(img, str(Y[i]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
        	            fontScale=0.5,
        	            color=(0, 0, 0))
        	cv2.imshow("data", img)
        	if cv2.waitKey(500) & 0xFF == ord('q'):
		    kk = 1
        	    cv2.destroyAllWindows()
        	    break
    if kk == 1:
	break
