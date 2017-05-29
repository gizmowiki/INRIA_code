import os
from shutil import copyfile

files = ['/data/stars/share/people_depth/people-depth/fulldata/files_positive.txt']
for item in files:
	if 'negative' in item:
		base_path = '/local/people_detect_tmp/'
	else:
		base_path = '/local/people_detect/'
		for paths in open(item, 'r'):
			paths = paths.strip()
			lines = paths.split('/')
			if not os.path.exists(os.path.join(base_path, lines[-2])):
				os.makedirs(os.path.join(base_path, lines[-2]))
			print (paths, " ---> ", os.path.join(base_path, lines[-2], lines[-1]))
			copyfile(paths, os.path.join(base_path, lines[-2], lines[-1]))
