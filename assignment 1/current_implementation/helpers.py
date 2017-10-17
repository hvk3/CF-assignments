import numpy as np

class info:
	def __init__(self, num_objects, object):
		self.num_objects = num_objects
		self.object = object

class item:
	def __init__(self, movie_id, movie_title, release_date, video_release_date, IMDb_URL, categories):
		self.movie_id = int(movie_id)
		self.movie_title = movie_title
		self.release_date = release_date
		self.video_release_date = video_release_date
		self.IMDb_URL = IMDb_URL
		self.categories = categories

class rating:
	def __init__(self, user_id, item_id, rating, timestamp):
		self.user_id = int(user_id)
		self.item_id = int(item_id)
		self.rating = int(rating)
		self.timestamp = int(timestamp)

class user:
	def __init__(self, user_id, age, gender, occupation, zip_code):
		self.user_id = int(user_id)
		self.age = int(age)
		self.gender = gender
		self.occupation = occupation
		self.zip_code = zip_code

def read_file(filename, delimiter, class_type):
	inputs = []
	with open(filename, 'r') as f:
		for row in f.readlines():
			row = row.strip().split(delimiter)
			try:
				compressed_row = row[:5] + [row[5:]]
				inputs.append(class_type(*compressed_row))
			except:
				inputs.append(class_type(*row))
	return inputs

def get_delimiter_class(filename, dict):
	return dict[filename.strip().split('.')[-1]]

def generate_user_item_matrix(ratings, num_users, num_items):
	user_item_matrix = np.zeros((num_users, num_items))
	for rating in ratings:
		user_item_matrix[rating.user_id - 1][rating.item_id - 1] = rating.rating
	return user_item_matrix