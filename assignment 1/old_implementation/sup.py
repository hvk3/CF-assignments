import aux
import numpy as np
import os

global_user_average_ratings = np.zeros((10000))
global_item_average_ratings = np.zeros((10000))
global_user_variance_weights = np.zeros((10000))
global_item_variance_weights = np.zeros((10000))

def get_average_ratings(matrix):	# Computes the average ratings for all users and items
	num_users = matrix.shape[0]
	num_items = matrix.shape[1]
	global global_user_average_ratings, global_item_average_ratings

	temp_user_ratings = [np.sum(matrix[user, :]) for user in xrange(num_users)]
	temp_user_rating_counts = [np.count_nonzero(matrix[user, :]) for user in xrange(num_users)]
	global_user_average_ratings = [0 if temp_user_rating_counts[i] == 0 else\
		temp_user_ratings[i] * 1.0 / temp_user_rating_counts[i] for i in xrange(num_users)]

	temp_item_ratings = [np.sum(matrix[:, item]) for item in xrange(num_items)]
	temp_item_rating_counts = [np.count_nonzero(matrix[:, item]) for item in xrange(num_items)]
	global_item_average_ratings = [0 if temp_item_rating_counts[i] == 0 else\
		temp_item_ratings[i] * 1.0 / temp_item_rating_counts[i] for i in xrange(num_items)]

def get_implicit_item_item_average_ratings(matrix):
	num_items = matrix.shape[0]
	temp_item_ratings = [np.sum(matrix[item, :]) for item in xrange(num_items)]
	temp_item_rating_counts = [np.count_nonzero(matrix[item, :]) for item in xrange(num_items)]
	global_item_average_ratings = [0 if temp_item_rating_counts[i] == 0 else\
		temp_item_ratings[i] * 1.0 / temp_item_rating_counts[i] for i in xrange(num_items)]

def get_variance_weights(matrix):	# Computes the variance weights for all users and items
	num_users = matrix.shape[0]
	num_items = matrix.shape[1]
	global global_item_variance_weights, global_user_variance_weights

	temp_item_variances = np.array([np.sum((matrix[:, item] - global_item_average_ratings[item]) ** 2) for item in xrange(num_items)]).astype('float64')
	max_variance, min_variance = np.max(temp_item_variances), np.min(temp_item_variances)
	if (max_variance > min_variance):
		global_item_variance_weights = (temp_item_variances - min_variance) * 1.0 / max_variance

	temp_user_variances = np.array([np.sum((matrix[user, :] - global_user_average_ratings[user]) ** 2) for user in xrange(num_users)]).astype('float64')
	max_variance, min_variance = np.max(temp_user_variances), np.min(temp_user_variances)
	if (max_variance > min_variance):
		global_user_variance_weights = (temp_user_variances - min_variance) * 1.0 / max_variance

def get_user_implicit_matrix(matrix):
	implicit_matrix = np.zeros((6500, 6500))
	for i in xrange(matrix.shape[0]):
		for j in xrange(matrix.shape[1]):
			if (matrix[i][j][0] + matrix[i][j][1]) != 0:
				implicit_matrix[i][j] = matrix[i][j][0] * 5.0 / (matrix[i][j][0] + matrix[i][j][1])
	return implicit_matrix

def get_item_implicit_matrix(dataset_root, movies_filename):
	return aux.get_item_category_matrix(os.path.join(dataset_root, movies_filename))

def get_Pearson_coefficient_user_user(matrix, user_1, user_2, apply_variance_weighting = False, apply_significance_weighting = False, co_rated_items = 0, cache = None):
	if (cache[user_1][user_2] > -10):
		return cache[user_1][user_2]
	num, denom = 0, 0
	ratings_user_1 = matrix[user_1, :]
	ratings_user_2 = matrix[user_2, :]
	ratings_users = zip(ratings_user_1, ratings_user_2)
	indices = [i for i in xrange(len(ratings_users)) if ratings_users[i][0] > 0 and ratings_users[i][1] > 0]
	required_variances = np.array([global_item_variance_weights[index] for index in indices])
	co_rated_filtered_items = np.array(filter(lambda x: x[0] > 0 and x[1] > 0, ratings_users))
	if (len(co_rated_filtered_items) == 0):
		cache[user_1][user_2] = 0
		return cache[user_1][user_2]
	co_rated_filtered_items[:, 0] -= global_user_average_ratings[user_1]
	co_rated_filtered_items[:, 1] -= global_user_average_ratings[user_2]
	if (not apply_variance_weighting):
		num = np.sum(np.multiply(co_rated_filtered_items[:, 0], co_rated_filtered_items[:, 1]))
		denom = (np.sum(co_rated_filtered_items[:, 0] ** 2) * np.sum(co_rated_filtered_items[:, 1] ** 2)) ** 0.5
		if (apply_significance_weighting):
			if (co_rated_items > len(co_rated_filtered_items)):
				num *= len(co_rated_filtered_items) * 1.0 / co_rated_items
	else:
		if (len(np.unique(required_variances)) > 1):
			num = np.sum(np.multiply(required_variances, np.multiply(co_rated_filtered_items[:, 0], co_rated_filtered_items[:, 1])))
			denom = (np.sum(co_rated_filtered_items[:, 0] ** 2) * np.sum(co_rated_filtered_items[:, 1] ** 2)) ** 0.5
		else:
			num = np.sum(np.multiply(co_rated_filtered_items[:, 0], co_rated_filtered_items[:, 1]))
			denom = (np.sum(co_rated_filtered_items[:, 0] ** 2) * np.sum(co_rated_filtered_items[:, 1] ** 2)) ** 0.5
	if (denom == 0):
		denom = 1
		num = 0
	cache[user_1][user_2] = num * 1.0 / denom
	return cache[user_1][user_2]

def get_Pearson_coefficient_item_item(matrix, item_1, item_2, apply_variance_weighting = False, cache = None, implicit = False):
	if (cache[item_1][item_2] > -10):
		return cache[item_1][item_2]
	if (not implicit):
		ratings_item_1 = matrix[:, item_1]
		ratings_item_2 = matrix[:, item_2]
	else:
		ratings_item_1 = matrix[item_1, :]
		ratings_item_2 = matrix[item_2, :]
	ratings_items = zip(ratings_item_1, ratings_item_2)
	indices = [i for i in xrange(len(ratings_items)) if ratings_items[i][0] > 0 and ratings_items[i][1] > 0]
	required_variances = np.array([global_item_variance_weights[index] for index in indices])
	co_rated_filtered_items = np.array(filter(lambda x: x[0] > 0 and x[1] > 0, ratings_items))
	if (len(co_rated_filtered_items) == 0):
		cache[item_1][item_2] = 0
		return cache[item_1][item_2]
	co_rated_filtered_items[:, 0] -= global_item_average_ratings[item_1]
	co_rated_filtered_items[:, 1] -= global_item_average_ratings[item_2]
	if (apply_variance_weighting):
		num = np.sum(np.multiply(required_variances, \
			np.multiply(co_rated_filtered_items[:, 0], co_rated_filtered_items[:, 1])))
		denom = (np.sum(np.multiply(required_variances, co_rated_filtered_items[:, 0] ** 2)) * \
			np.sum(np.multiply(required_variances, co_rated_filtered_items[:, 1] ** 2))) ** 0.5
	else:
		num = np.sum(np.multiply(co_rated_filtered_items[:, 0], co_rated_filtered_items[:, 1]))
		denom = (np.sum(co_rated_filtered_items[:, 0] ** 2) * np.sum(co_rated_filtered_items[:, 1] ** 2)) ** 0.5
	if (denom == 0):
		denom = 1
		num = 0
	cache[item_1][item_2] = num * 1.0 / denom
	return cache[item_1][item_2]

def get_random_prediction(user, item):
	return random.uniform(0, 5)

def get_user_based_prediction(matrix, user, item, apply_variance_weighting = False, apply_significance_weighting = False, apply_threshold = False, apply_knn = False, co_rated_items = 0, similarity_threshold = 0.0, k = 0, cache = None):
	predicted_rating = 0.0
	num_users = matrix.shape[0]
	other_users = [other_user for other_user in xrange(num_users) if matrix[other_user][item] != 0]
	averages_other_users = [matrix[other_user][item] - global_user_average_ratings[other_user] for other_user in other_users]
	similarities = [get_Pearson_coefficient_user_user(matrix, user, other_user, apply_variance_weighting, apply_significance_weighting, co_rated_items, cache)\
		for other_user in other_users]
	if (apply_threshold):
		averages_other_users, similarities = (list(t) for t in zip(*filter(lambda x: x[1] > similarity_threshold, zip(averages_other_users, similarities))))
	if (apply_knn):
		averages_other_users, similarities = (list(t) for t in zip(*sorted(zip(averages_other_users, similarities), key = lambda x: x[1], reverse = True)))
		averages_other_users, similarities = averages_other_users[:min(k, len(averages_other_users))], similarities[:min(k, len(similarities))]
	num = np.sum(np.multiply(averages_other_users, similarities))
	denom = np.sum(np.absolute(similarities))
	if (denom == 0):
		return global_user_average_ratings[user]
	else:
		return global_user_average_ratings[user] + num * 1.0 / denom

def get_item_based_prediction(matrix, user, item, apply_variance_weighting = False, cache = None, implicit = False):
	predicted_rating = 0.0
	num_items = matrix.shape[1]
	other_items = [other_item for other_item in xrange(num_items) if matrix[user][other_item] != 0]
	averages_other_items = [matrix[user][other_item] - global_item_average_ratings[other_item] for other_item in other_items]
	similarities = [get_Pearson_coefficient_item_item(matrix, item, other_item, apply_variance_weighting, cache, implicit)\
		for other_item in other_items]
	num = np.sum(np.multiply(averages_other_items, similarities))
	denom = np.sum(np.absolute(similarities))
	if (denom == 0):
		return global_item_average_ratings[item]
	else:
		return global_item_average_ratings[item] + num * 1.0 / denom

def get_normalized_mean_absolute_error(predicted_ratings, actual_ratings, min_rating, max_rating):
	return np.sum(np.absolute(predicted_ratings - actual_ratings)) * 1.0 / (len(predicted_ratings) * (max_rating - min_rating))

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
		unnecessary_matrix = aux.get_user_item_matrix(os.path.join(dataset_root, 'u' + str(i) + '.base'))
		ith_fold_user_item_matrix = -10 * np.ones(unnecessary_matrix.shape)
		for j in xrange(5):
			if (i != j + 1):
				curr_matrix = aux.get_user_item_matrix(os.path.join(dataset_root, 'u' + str(j + 1) + '.base'))
				for k in xrange(curr_matrix.shape[0]):
					for l in xrange(curr_matrix.shape[1]):
						if (ith_fold_user_item_matrix[k][l] == -10):
							ith_fold_user_item_matrix[k][l] = curr_matrix[k][l]
		ith_fold_ratings = aux.get_ratings(os.path.join(dataset_root, 'u' + str(i) + '.test'))
	return ith_fold_user_item_matrix, ith_fold_ratings
