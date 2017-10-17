import aux
import sup
import os
import sys
import tqdm
import numpy as np

if (__name__ == '__main__'):
	sup.dataset_root = sys.argv[1]
	use_explicit = (sys.argv[2] == 'explicit')
	use_item_item = (sys.argv[3] == 'item_item')
	co_rated_items = 0
	similarity_threshold = 0.0
	k = 10000000000000
	try:
		apply_variance_weighting = (sys.argv[4] == 'apply_variance')
	except:
		apply_variance_weighting = False
	try:
		apply_significance_weighting = (sys.argv[4] == 'apply_significance')
		if (apply_significance_weighting):
			try:
				co_rated_items = int(sys.argv[5])
			except:
				pass
	except:
		apply_significance_weighting = False
	try:
		apply_threshold = (sys.argv[4] == 'apply_threshold')
		if (apply_threshold):
			try:
				similarity_threshold = float(sys.argv[5])
			except:
				pass
	except:
		apply_threshold = False
	try:
		apply_knn = (sys.argv[4] == 'apply_knn')
		if (apply_knn):
			try:
				k = int(sys.argv[5])
			except:
				pass
	except:
		apply_knn = False
	num_users = 10000
	num_items = 10000
	k_folds = 5
	if ('1m' in sup.dataset_root):
		sup.users_filename = 'users.dat'
		sup.movies_filename = 'movies.dat'
		sup.ratings_filename = 'ratings.dat'
	else:
		sup.users_filename = 'u.user'
		sup.movies_filename = 'u.item'
		sup.ratings_filename = 'u'
	users_filename = os.path.join(sup.dataset_root, sup.users_filename)
	movies_filename = os.path.join(sup.dataset_root, sup.movies_filename)
	ratings_filename = os.path.join(sup.dataset_root, sup.ratings_filename)
	for i in xrange(k_folds):
		ith_fold_user_item_matrix, ith_fold_ratings = sup.get_ith_training_test_data(i + 1)
		user_implicit_matrix = sup.get_user_implicit_matrix(aux.get_user_category_matrix(users_filename, movies_filename, ratings_filename + str(i + 1) + '.base'))
		item_implicit_matrix = sup.get_item_implicit_matrix(sup.dataset_root, sup.movies_filename)
		cache = -10 * np.ones((num_users, num_items))
		if (use_explicit):
			sup.get_average_ratings(ith_fold_user_item_matrix)
			sup.get_variance_weights(ith_fold_user_item_matrix)
		else:
			if (use_item_item):
				sup.get_implicit_item_item_average_ratings(item_implicit_matrix)
			else:
				sup.get_average_ratings(user_implicit_matrix)
				sup.get_variance_weights(user_implicit_matrix)
		predicted_ratings = []
		actual_ratings = []
		for rating_num in tqdm.tqdm(range(len(ith_fold_ratings))):
		# for rating_num in tqdm.tqdm(range(10)):
			rating = ith_fold_ratings[rating_num]
			if (use_explicit):
				if (use_item_item):
					predicted_rating = sup.get_item_based_prediction(ith_fold_user_item_matrix, rating.user_id - 1, rating.item_id - 1,\
					 apply_variance_weighting, cache, False)
				else:
					predicted_rating = sup.get_user_based_prediction(ith_fold_user_item_matrix, rating.user_id - 1, rating.item_id - 1,\
					 apply_variance_weighting, apply_significance_weighting, apply_threshold, apply_knn, co_rated_items, similarity_threshold, k, cache)
			else:
				if (use_item_item):
					predicted_rating = sup.get_item_based_prediction(item_implicit_matrix, rating.user_id - 1, rating.item_id - 1,\
					 apply_variance_weighting, cache, True)
				else:
					predicted_rating = sup.get_user_based_prediction(user_implicit_matrix, rating.user_id - 1, rating.item_id - 1,\
					 apply_variance_weighting, apply_significance_weighting, apply_threshold, apply_knn, co_rated_items, similarity_threshold, k, cache)
			predicted_ratings.append(predicted_rating)
			actual_ratings.append(rating.rating)
		filename = sys.argv[2] + '_' + sys.argv[3]
		if (apply_variance_weighting):
			filename += '_variance'
		if (apply_significance_weighting):
			filename += '_significance_' + str(co_rated_items)
		if (apply_threshold):
			filename += '_threshold_' + str(similarity_threshold).replace('.', '_')
		if (apply_knn):
			filename += '_knn_' + str(k)
		# np.save(os.path.join(sup.dataset_root, str(i + 1) + '_' + filename), cache)
		predicted_ratings = np.array(predicted_ratings)
		actual_ratings = np.array(actual_ratings)
		user_explicit_NMAE = sup.get_normalized_mean_absolute_error(predicted_ratings, actual_ratings, 1, 5)
		print 'NMAE for u' + str(i + 1) + ':', user_explicit_NMAE

		# with open(os.path.join(sup.dataset_root, filename + '_nmae_results'), 'a') as f:
		# 	f.write(str(i) + ': ' + str(user_explicit_NMAE) + '\n')
		# 	if (i == 4):
		# 		f.write('\n')
