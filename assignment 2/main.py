import os
import sys

import aux

if __name__ == '__main__':
	file_delimiter = {'data' : ('\t', aux.rating), 'item' : ('|', aux.item),\
		'user' : ('|', aux.user), 'base' : ('\t', aux.rating), 'test' : ('\t', aux.rating)}
	num_users = int(sys.argv[1])
	num_items = int(sys.argv[2])
	num_ratings = int(sys.argv[3])
	user_AE = (sys.argv[4] == 'user')
	k = [10, 20, 40, 80, 100, 200, 300, 400, 500][::-1]
	regularizer = [0.01][::-1]
	dataset_to_use = sys.argv[5]
	train = (len(sys.argv) > 6)
	if (num_ratings == 1000000):
		name = '_1m'
	else:
		name = '_100k'
	num_epochs = 10

	for regularizer_ in regularizer:
		for k_ in k:
			for l in xrange(5):
				ratings_training_filename = os.path.join(dataset_to_use, 'u' + str(l + 1) + '.base')
				ratings_test_filename = os.path.join(dataset_to_use, 'u' + str(l + 1) + '.test')
				train_ratings = aux.read_file(ratings_training_filename, *aux.get_delimiter_class(ratings_training_filename, file_delimiter))
				test_ratings = aux.read_file(ratings_test_filename,	*aux.get_delimiter_class(ratings_test_filename, file_delimiter))
				user_item_matrix = aux.generate_user_item_matrix(train_ratings, num_users, num_items)
				if ('_1m' in name):
					user_item_matrix[user_item_matrix == 0] = 3
				item_AE_shape = user_item_matrix[:,0].shape[0]
				user_AE_shape = user_item_matrix[0,:].shape[0]
				if (user_AE):
					net = aux.Net(k_, user_AE_shape)
					train_data = [user_item_matrix[i, :] for i in xrange(item_AE_shape)]
				else:
					net = aux.Net(k_, item_AE_shape)
					train_data = [user_item_matrix[:, i] for i in xrange(user_AE_shape)]
				if (train):
					print 'Training ' + sys.argv[4] + ' autoencoder for u' + str(l + 1) + '.base with hidden dimensions ' + str(k_)\
						+ ' and regularizer ' + str(regularizer_)
					for epoch in xrange(num_epochs):
						aux.train_net(net, train_data, regularizer_)
					if (user_AE):
						aux.torch.save(net, 'userAE/userAE_' + str(k_) + '_' + str(l + 1) + name + '.pt')
					else:
						aux.torch.save(net, 'itemAE/itemAE_' + str(k_) + '_' + str(l + 1) + name + '.pt')

			net_mae = 0.0
			for l in xrange(5):
				if (user_AE):
					net = aux.torch.load('userAE/userAE_' + str(k_) + '_' + str(l + 1) + name + '.pt')
				else:
					net = aux.torch.load('itemAE/itemAE_' + str(k_) + '_' + str(l + 1) + name + '.pt')
				ratings_training_filename = os.path.join(dataset_to_use, 'u' + str(l + 1) + '.base')
				ratings_test_filename = os.path.join(dataset_to_use, 'u' + str(l + 1) + '.test')
				train_ratings = aux.read_file(ratings_training_filename, *aux.get_delimiter_class(ratings_training_filename, file_delimiter))
				test_ratings = aux.read_file(ratings_test_filename,	*aux.get_delimiter_class(ratings_test_filename, file_delimiter))
				user_item_matrix = aux.generate_user_item_matrix(train_ratings, num_users, num_items)
				item_AE_shape = user_item_matrix[:,0].shape[0]
				user_AE_shape = user_item_matrix[0,:].shape[0]
				if (user_AE):
					train_data = [user_item_matrix[i, :] for i in xrange(item_AE_shape)]
				else:
					train_data = [user_item_matrix[:, i] for i in xrange(user_AE_shape)]
				mae = aux.test_net(net, train_data, test_ratings, user_AE)
				print 'NMAE for u' + str(l + 1) + '.test, hidden dimensions = ' + str(k_) + ', regularizer = ' + str(regularizer_) + ' :', mae / 4.
				print 'MAE for u' + str(l + 1) + '.test, hidden dimensions = ' + str(k_) + ', regularizer = ' + str(regularizer_) + ' :', mae
				net_mae += mae
			print 'Average MAE over 5 folds, hidden dimensions = ' + str(k_) + ', regularizer = ' + str(regularizer_) + ' :', net_mae / 5.
			with open(sys.argv[4] + name + '_results', 'a') as f:
				f.write('Average MAE over 5 folds, hidden dimensions = ' + str(k_) + ', regularizer = ' + str(regularizer_) + ' : ' +  str(net_mae / 5.) + '\n')
