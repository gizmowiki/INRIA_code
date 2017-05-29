import os
import random

base_path = '/local/people_orig_depth'
for i in range(59598, 1785310, 3):
	# del_id = i + random.randint(0,3)
	pos_del_path = os.path.join(base_path, 'positives', 'chalearn_*{0:08d}.png'.format(i))
	pos_del_path_2 = os.path.join(base_path, 'positives', 'chalearn_*{0:08d}.png'.format(i+2))
	print("Now deleting positives ... ", pos_del_path, pos_del_path_2)
	os.remove(pos_del_path)
	os.remove(pos_del_path_2)
	
