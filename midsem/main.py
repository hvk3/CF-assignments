import sys

import helpers

ratings_filename = 'ml-100k/u.data'
users_filename = 'ml-100k/u.user'
items_filename = 'ml-100k/u.item'

if __name__ == '__main__':
	file_delimiter = {'data' : ('\t', helpers.rating), 'item' : ('|', helpers.item),\
		'user' : ('|', helpers.user), 'base' : ('\t', helpers.rating), 'test' : ('\t', helpers.rating)}
	num_users = int(sys.argv[1])
	num_items = int(sys.argv[2])
	num_ratings = int(sys.argv[3])
	num_latent_factors = [5, 7, 9, 11, 13]
	regularizers = [0.05, 0.07, 0.09]
	
	items = helpers.read_file(items_filename, *helpers.get_delimiter_class(items_filename, file_delimiter))
	users = helpers.read_file(users_filename, *helpers.get_delimiter_class(users_filename, file_delimiter))
	
	for l in xrange(5):
		ratings_training_filename = 'ml-100k/u' + str(l + 1) + '.base'
		ratings_test_filename = 'ml-100k/u' + str(l + 1) + '.test'
		ratings = helpers.read_file(ratings_training_filename,\
			*helpers.get_delimiter_class(ratings_training_filename, file_delimiter))
		for i in num_latent_factors:
			for j in regularizers:
				print 'Number of latent factors:', i
				print 'Regularizer:', j
				U0, V0 = helpers.init_matrices(num_users, num_items, i)
				U, V = helpers.solve(helpers.generate_user_item_matrix(ratings, num_users, num_items).T, U0, V0, 100, j)
				predicted_ratings_matrix = helpers.np.dot(V, U).T
				predicted_ratings_matrix[predicted_ratings_matrix > 5] = 5
				predicted_ratings_matrix[predicted_ratings_matrix < 0] = 1
				test_ratings = helpers.read_file(ratings_test_filename,\
					*helpers.get_delimiter_class(ratings_test_filename, file_delimiter))
				mae= 0.0
				for test_rating in test_ratings:
					mae += helpers.np.absolute(predicted_ratings_matrix[test_rating.user_id - 1][test_rating.item_id - 1] - test_rating.rating)
				print 'NMAE for u' + str(l + 1) + '.base :', mae / (4.0 * len(test_ratings))
