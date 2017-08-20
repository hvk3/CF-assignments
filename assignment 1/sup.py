import aux
import numpy as np
import os

dataset_root = 'ml-100k'
users_filename = 'u.user'
movies_filename = 'u.item'

# dataset_root = 'ml-1m'
# users_filename = 'users.dat'
# movies_filename = 'movies.dat'
# ratings_filename = 'ratings.dat'

global_user_average_ratings = np.zeros((1000000))
global_item_average_ratings = np.zeros((1000000))
global_user_variance_weights = np.zeros((1000000))
global_item_variance_weights = np.zeros((1000000))

user_category_matrix = aux.get_user_category_matrix(users_filename, movies_filename, ratings_filename)
item_category_matrix = aux.get_item_category_matrix(movies_filename)

def get_average_ratings(matrix):	# Computes the average ratings for all users and items
	num_users = matrix.shape[0]
	num_items = matrix.shape[1]
	global global_user_average_ratings, global_item_average_ratings
	global_user_average_ratings = global_user_average_ratings[:num_users]
	global_item_average_ratings = global_item_average_ratings[:num_items]
	for user in xrange(num_users):
		if (np.count_nonzero(matrix[user, :]) > 0):
			global_user_average_ratings[user] = np.sum(matrix[user, :]) / np.count_nonzero(matrix[user, :])
	for item in xrange(num_items):
		if (np.count_nonzero(matrix[:, item]) > 0):
			global_item_average_ratings[item] = np.sum(matrix[:, item]) / np.count_nonzero(matrix[:, item])

def get_variance_weights(matrix):	# Computes the variance weights for all users and items
	num_users = matrix.shape[0]
	num_items = matrix.shape[1]
	global global_item_variance_weights, global_user_variance_weights
	global_user_variance_weights = global_user_variance_weights[:num_users]
	global_item_variance_weights = global_item_variance_weights[:num_items]
	for item in xrange(num_items):
		item_average_rating = global_item_average_ratings[item]
		if (np.count_nonzero(matrix[:, item]) > 0):
			global_item_variance_weights[item] = np.sum((matrix[:, item] - item_average_rating) ** 2) * 1.0 / \
				np.count_nonzero(matrix[:, item])
	max_variance_weight, min_variance_weight = np.max(global_item_variance_weights), np.min(global_item_variance_weights)
	if (max_variance_weight > min_variance_weight):
		global_item_variance_weights = (global_item_variance_weights - min_variance_weight) * 1.0 / max_variance_weight

	for user in xrange(num_users):
		user_average_rating = global_user_average_ratings[user]
		if (np.count_nonzero(matrix[user, :]) > 0):
			global_user_variance_weights[user] = np.sum((matrix[user, :] - user_average_rating) ** 2) * 1.0 / \
				np.count_nonzero(matrix[user, :])
	max_variance_weight, min_variance_weight = np.max(global_user_variance_weights), np.min(global_user_variance_weights)
	if (max_variance_weight > min_variance_weight):
		global_user_variance_weights = (global_user_variance_weights - min_variance_weight) * 1.0 / max_variance_weight

def get_Pearson_coefficient(matrix, user_1 = None, user_2 = None, item_1 = None, item_2 = None, \
		apply_variance_weighting = False, apply_similarity_weighting = False, num_co_rated_item = 0, cache = None):	# Computes the Pearson coefficient according to the various strategies outlined
	# Cache maintains user-user or item-item similarity
	Pearson_coefficient_numerator = 0.0
	Pearson_coefficient_partial_denominator_1 = 0.0
	Pearson_coefficient_partial_denominator_2 = 0.0
	common_items_rated = 0
	common_users_rating = 0
	num_users = matrix.shape[0]
	num_items = matrix.shape[1]
	if (user_1 is not None):
		if (cache[user_1][user_2] > -10):
			return cache[user_1][user_2]
		multiplying_factor = 1.0
		average_rating_user_1 = global_user_average_ratings[user_1]
		average_rating_user_2 = global_user_average_ratings[user_2]
		for item in xrange(num_items):
			if (matrix[user_1][item] > 0 and matrix[user_2][item] > 0):
				if (apply_variance_weighting):
					multiplying_factor = global_item_variance_weights[item]
				common_items_rated += 1
				Pearson_coefficient_numerator += (matrix[user_1][item] - average_rating_user_1) * \
					(matrix[user_2][item] - average_rating_user_2) * multiplying_factor
				Pearson_coefficient_partial_denominator_1 += (multiplying_factor * (matrix[user_1][item] - average_rating_user_1) ** 2)
				Pearson_coefficient_partial_denominator_2 += (multiplying_factor * (matrix[user_2][item] - average_rating_user_2) ** 2)
	
	if (item_1 is not None):
		if (cache[item_1][item_2] > -10):
			return cache[item_1][item_2]
		multiplying_factor = 1.0
		average_rating_item_1 = global_item_average_ratings[item_1]
		average_rating_item_2 = global_item_average_ratings[item_2]
		for user in xrange(num_users):
			if (matrix[user][item_1] > 0 and matrix[user][item_2] > 0):
				if (apply_variance_weighting):
					multiplying_factor = global_user_variance_weights[user]
				common_users_rating += 1
				Pearson_coefficient_numerator += (matrix[user][item_1] - average_rating_item_1) * \
					(matrix[user][item_2] - average_rating_item_2) * multiplying_factor
				Pearson_coefficient_partial_denominator_1 += (multiplying_factor * (matrix[user][item_1] - average_rating_item_1) ** 2)
				Pearson_coefficient_partial_denominator_2 += (multiplying_factor * (matrix[user][item_2] - average_rating_item_2) ** 2)

	if (apply_similarity_weighting):
		if (num_co_rated_item > common_users_rating + common_items_rated):
			Pearson_coefficient_numerator *= (common_items_rated + common_users_rating) * 1.0 / num_co_rated_item
	if ((Pearson_coefficient_partial_denominator_1 * Pearson_coefficient_partial_denominator_2) ** 0.5 == 0):
		PC = 0
	else:
		PC = Pearson_coefficient_numerator * 1.0 / \
			(Pearson_coefficient_partial_denominator_1 * Pearson_coefficient_partial_denominator_2) ** 0.5
	if (user_1 is not None):
		cache[user_1][user_2] = PC
	else:
		cache[item_1][item_2] = PC
	return PC

# def get_Pearson_coefficient_implicit_user_based(matrix, user_1, user_2, cache = None):
# 	if (cache[user_1][user_2] > -10):
		

def get_Pearson_coefficient_explicit_user_based(matrix, user_1, user_2, apply_variance_weighting = False, apply_similarity_weighting = False, cache = None, num_co_rated_items = 0):
	return get_Pearson_coefficient(matrix, user_1, user_2, None, None, apply_variance_weighting, apply_similarity_weighting, num_co_rated_items, cache)

def get_Pearson_coefficient_explicit_item_based(matrix, item_1, item_2, apply_variance_weighting = False, cache = None):
	return get_Pearson_coefficient(matrix, None, None, item_1, item_2, apply_variance_weighting, cache = cache)

def get_random_prediction(user, item):
	return random.uniform(0, 5)

def get_user_based_explicit_ratings_prediction(matrix, user, item, apply_variance_weighting = False,\
		apply_similarity_weighting = False, apply_threshold = False, apply_knn = False, num_co_rated_items = 0, similarity_threshold = 0.0, k = 0, cache = None):
	average_user_rating = global_user_average_ratings[user]
	adjustment = 0.0
	num_users = matrix.shape[0]

	num, denom = 0, 0
	best_k_correlations = []
	best_k_correlation_values = []
	for other_user in xrange(num_users):
		if (matrix[other_user][item] != 0):
			similarity = get_Pearson_coefficient_explicit_user_based(matrix, user, other_user, apply_variance_weighting, apply_similarity_weighting, cache = cache, num_co_rated_items = num_co_rated_items)
			if (apply_threshold):
				if (np.absolute(similarity) <= similarity_threshold):
					continue
			if (apply_knn):
				best_k_correlations.append(other_user)
				best_k_correlation_values.append(similarity)
			else:
				denom += np.absolute(similarity)
				num += similarity * (matrix[other_user][item] - global_user_average_ratings[other_user])
	if (apply_knn):
		num, denom = 0, 0
		if (len(best_k_correlations) > 0) and (len(best_k_correlation_values) > 0):
			best_k_correlations, best_k_correlation_values = (list(x) for x in zip(*sorted(zip(best_k_correlations, \
				best_k_correlation_values))))
			best_k_correlations = list(reversed(best_k_correlations))
			best_k_correlation_values = list(reversed(best_k_correlation_values))
			for i in xrange(min(k, len(best_k_correlations))):
				num += best_k_correlation_values[i] * (matrix[best_k_correlations[i]][item] - global_user_average_ratings[best_k_correlations[i]])
				denom += np.absolute(best_k_correlation_values[i])
	if (denom != 0):
		adjustment = num * 1.0 / denom
	return adjustment + average_user_rating

def get_item_based_explicit_ratings_prediction(matrix, user, item, apply_variance_weighting = False, cache = None):
	average_item_rating = global_item_average_ratings[item]
	adjustment = 0.0
	num_items = matrix.shape[1]

	num, denom = 0, 0
	for other_item in xrange(num_items):
		if (matrix[user][other_item] != 0):
			similarity = get_Pearson_coefficient_explicit_item_based(matrix, item, other_item, apply_variance_weighting, cache)
			average_other_item_rating = global_item_average_ratings[other_item]
			denom += np.absolute(similarity)
			num += similarity * (matrix[user][other_item] - average_other_item_rating)
	if (denom != 0):
		adjustment = num * 1.0 / denom
	return adjustment + average_item_rating

def get_normalized_mean_absolute_error(predicted_ratings, actual_ratings, min_rating, max_rating):
	n = len(predicted_ratings)
	assert(n == len(actual_ratings))
	return np.sum(np.absolute(predicted_ratings - actual_ratings)) * 1.0 / (n * (max_rating - min_rating))

def get_ith_training_test_data(i):
	if ('1m' in dataset_root):
		ratings = aux.get_ratings(os.path.join(dataset_root, ratings_filename))
		num_ratings = len(ratings)
		ith_dataset = ratings[(i - 1) * num_ratings / 5 : i * num_ratings / 5 - 1]
		ith_training_set = ith_dataset[: 4 * len(ith_dataset) / 5]
		ith_fold_user_item_matrix = np.zeros((6500, 4000))
		for rating in ratings:
			ith_fold_user_item_matrix[rating.user_id - 1][rating.item_id - 1] = rating.rating
		ith_fold_ratings = ith_dataset[4 * len(ith_dataset) / 5 :]
	else:
		ith_fold_user_item_matrix = aux.get_user_item_matrix(os.path.join(dataset_root, 'u' + str(i) + '.base'))
		ith_fold_ratings = aux.get_ratings(os.path.join(dataset_root, 'u' + str(i) + '.test'))
	return ith_fold_user_item_matrix, ith_fold_ratings
