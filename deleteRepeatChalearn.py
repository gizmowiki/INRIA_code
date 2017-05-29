import os
import random

base_path = '/local/people_orig_depth'
i = 0
for files in open('/local/filelist_chalearn.txt', 'rb'):
	i += 1
	if (i%3) == 0:
		continue
	files = files.strip()
	print("Now deleting positives ... ", files)
	os.remove(files)
