import sys

import helpers

ratings_filename = '../ml-100k/u.data'
users_filename = '../ml-100k/u.user'
items_filename = '../ml-100k/u.item'

if __name__ == '__main__':
	file_delimiter = {'data' : ('\t', helpers.rating), 'item' : ('|', helpers.item),\
		'user' : ('|', helpers.user), 'base' : ('|', helpers.rating), 'test' : ('|', helpers.rating)}
	num_users = int(sys.argv[1])
	num_items = int(sys.argv[2])
	num_ratings = int(sys.argv[3])
	
	items = helpers.read_file(items_filename, *helpers.get_delimiter_class(items_filename, file_delimiter))
	ratings = helpers.read_file(ratings_filename, *helpers.get_delimiter_class(ratings_filename, file_delimiter))
	users = helpers.read_file(users_filename, *helpers.get_delimiter_class(users_filename, file_delimiter))

	import pdb;pdb.set_trace()
	user_item_matrix = helpers.generate_user_item_matrix(ratings, num_users, num_items)
	