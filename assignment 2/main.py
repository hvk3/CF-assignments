import sys
import aux

ratings_filename = 'ml-100k/u.data'
users_filename = 'ml-100k/u.user'
items_filename = 'ml-100k/u.item'

if __name__ == '__main__':
	file_delimiter = {'data' : ('\t', aux.rating), 'item' : ('|', aux.item),\
		'user' : ('|', aux.user), 'base' : ('\t', aux.rating), 'test' : ('\t', aux.rating)}
	num_users = int(sys.argv[1])
	num_items = int(sys.argv[2])
	num_ratings = int(sys.argv[3])
	user_AE = (sys.argv[4] == 'user')
	k = int(sys.argv[5])
	num_epochs = 100

	items = aux.read_file(items_filename, *aux.get_delimiter_class(items_filename, file_delimiter))
	users = aux.read_file(users_filename, *aux.get_delimiter_class(users_filename, file_delimiter))

	for l in xrange(5):
		ratings_training_filename = 'ml-100k/u' + str(l + 1) + '.base'
		ratings_test_filename = 'ml-100k/u' + str(l + 1) + '.test'
		train_ratings = aux.read_file(ratings_training_filename, *aux.get_delimiter_class(ratings_training_filename, file_delimiter))
		test_ratings = aux.read_file(ratings_test_filename,	*aux.get_delimiter_class(ratings_test_filename, file_delimiter))
		user_item_matrix = aux.generate_user_item_matrix(train_ratings, num_users, num_items)
		item_AE_shape = user_item_matrix[:,0].shape[0]
		user_AE_shape = user_item_matrix[0,:].shape[0]
		if (user_AE):
			net = aux.Net(k, user_AE_shape)
			train_data = [user_item_matrix[i, :] for i in xrange(item_AE_shape)]
		else:
			net = aux.Net(k, item_AE_shape)
			train_data = [user_item_matrix[:, i] for i in xrange(user_AE_shape)]
		for epoch in xrange(num_epochs):
			print 'Epoch ' + str(epoch + 1) + ' ->',
			aux.train_net(net, train_data)
		import pdb;pdb.set_trace()
		if (user_AE):
			model.save_state_dict('userAE_' + str(l + 1) + '.pt')
		else:
			model.save_state_dict('itemAE_' + str(l + 1) + '.pt')
		print 'NMAE for u' + str(l + 1) + '.base :', aux.test_net(net, test_ratings, user_AE)
