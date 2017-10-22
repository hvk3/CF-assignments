import numpy as np
import os
import sys
import random

import aux

def get_gt(file, delim = '::'):
	ratings = []
	with open(file, 'r') as f:
		for line in f.readlines():
			line = map(int, line.strip().split(delim))
			ratings.append(aux.rating(*line))
	return ratings

if __name__ == '__main__':
	root_dir = sys.argv[1]
	ratings = get_gt(os.path.join(root_dir, 'ratings.dat'))
	random.shuffle(ratings)
	# import pdb;pdb.set_trace()
	for l in xrange(5):
		base_file = os.path.join(root_dir, 'u' + str(l + 1) + '.base')
		test_file = os.path.join(root_dir, 'u' + str(l + 1) + '.test')
		test_set = ratings[l * len(ratings) / 5 : (l + 1) * len(ratings) / 5]
		base_set = ratings[: l * len(ratings) / 5] + ratings[(l + 1) * len(ratings) / 5 : ]
		with open(base_file, 'a') as f:
			for sample in base_set:
				f.write(str(sample.user_id) + '\t' + str(sample.item_id) + '\t' + str(sample.rating) + '\t' + str(sample.timestamp) + '\n')
		with open(test_file, 'a') as f:
			for sample in test_set:
				f.write(str(sample.user_id) + '\t' + str(sample.item_id) + '\t' + str(sample.rating) + '\t' + str(sample.timestamp) + '\n')
