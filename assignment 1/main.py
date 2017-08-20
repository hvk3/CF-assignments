import sup
import os
import sys
import time
import numpy as np

if (__name__ == '__main__'):
	for i in xrange(1, 5):
		print 'Starting fold ' + str(i + 1)
		ith_fold_user_item_matrix, ith_fold_ratings = sup.get_ith_training_test_data(i + 1)
		cache = -10 * np.ones((6500, 6500))
		sup.get_average_ratings(ith_fold_user_item_matrix)
		sup.get_variance_weights(ith_fold_user_item_matrix)

		predicted_ratings = []
		actual_ratings = []
		num_co_rated_items = int(sys.argv[1])
		# similarity_threshold = float(sys.argv[1])
		# k = int(sys.argv[1])

		j = 0
		init = time.time()
		for rating in ith_fold_ratings:
			start = time.time()
			j += 1
			predicted_rating = sup.get_user_based_explicit_ratings_prediction(ith_fold_user_item_matrix\
				, rating.user_id - 1, rating.item_id - 1, cache = cache, apply_similarity_weighting = True, num_co_rated_items = num_co_rated_items)
			predicted_ratings.append(predicted_rating)
			actual_ratings.append(rating.rating)
			print 'Rating ' + str(j)
			print 'Predicted rating:', predicted_rating
			print 'Actual rating:', rating.rating, '\n'
			print 'Time for predictions:', time.time() - start, 'seconds'
		print 'Total time: ', time.time() - init, 'seconds'
		np.save(os.path.join(sup.dataset_root, str(i + 1) + '_user_similarity_' + str(num_co_rated_items)), cache)
		predicted_ratings = np.array(predicted_ratings)
		actual_ratings = np.array(actual_ratings)
		user_explicit_NMAE = sup.get_normalized_mean_absolute_error(predicted_ratings, actual_ratings, 1, 5)
		print 'NMAE for u' + str(i + 1) + ':', user_explicit_NMAE

		with open(os.path.join(sup.dataset_root, 'user_similarity_nmae_' + str(num_co_rated_items) + '_results'), 'a') as f:
			f.write(str(i) + ': ' + str(user_explicit_NMAE) + '\n')
			if (i == 4):
				f.write('\n')
	