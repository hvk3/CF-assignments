import numpy as np

class movie:
	def __init__(self, movie_id, movie_title, release_date, video_release_date, IMDb_URL, genres):
		self.movie_id = movie_id
		self.movie_title = movie_title
		self.release_date = release_date
		self.video_release_date = video_release_date
		self.IMDb_URL = IMDb_URL
		self.genres = genres

class user:
	def __init__(self, user_id, age, gender, occupation, zip_code):
		self.user_id = user_id
		self.age = age
		self.gender = gender
		self.occupation = occupation
		self.zip_code = zip_code

class rating:
	def __init__(self, user_id, item_id, rating, timestamp):
		self.user_id = user_id
		self.item_id = item_id
		self.rating = rating
		self.timestamp = timestamp

class genre:
	def __init__(self, genres):
		self.genres = genres

def get_movies(filename):
	movies = []
	if ('1m' in filename):
		delimiter = '::'
	else:
		delimiter = '|'
	i = 1
	with open(filename, 'r') as f:
		for row in f.readlines():
			row = row.strip().split(delimiter)
			movies.append(movie(i, row[1], row[2], row[3], row[4], map(int, row[5:])))
			i += 1
	return movies

def get_users(filename):
	users = []
	if ('1m' in filename):
		delimiter = '::'
	else:
		delimiter = '|'
	i = 1
	with open(filename, 'r') as f:
		for row in f.readlines():
			row = row.strip().split(delimiter)
			users.append(user(i, row[1], row[2], row[3], row[4]))
			i += 1
	return users

def get_ratings(filename):
	ratings = []
	if ('1m' in filename):
		delimiter = '::'
	else:
		delimiter = '\t'
	with open(filename, 'r') as f:
		for row in f.readlines():
			row = row.strip().split(delimiter)
			ratings.append(rating(int(row[0]), int(row[1]), int(row[2]), int(row[3])))
	return ratings

def get_dataset_info(filename = 'ml-100k/u.info'):
	num_users, num_movies = 0, 0
	with open(filename, 'r') as f:
		for row in f.readlines():
			row = row.strip().split()
			if ('users' in row):
				num_users = int(row[0])
			if ('items' in row):
				num_movies = int(row[0])
	return num_users, num_movies

def get_user_item_matrix(ratings_filename):
	ratings = get_ratings(ratings_filename)
	if ('1m' in ratings_filename):
		num_users, num_movies = 6500, 4000
	else:
		num_users, num_movies = get_dataset_info()

	user_item_matrix = np.zeros((num_users, num_movies))
	for rating in ratings:
		user_item_matrix[rating.user_id - 1][rating.item_id - 1] = rating.rating
	return user_item_matrix

def get_item_category_matrix(movies_filename):
	movies = get_movies(movies_filename)
	num_movies = len(movies)
	num_categories = len(movies[0].genres)

	item_category_matrix = np.zeros((num_movies, num_categories))
	for movie in movies:
		for category in xrange(num_categories):
			item_category_matrix[movie.movie_id - 1][category] = movie.genres[category]
	return item_category_matrix

def get_user_category_matrix(users_filename, movies_filename, ratings_filename):
	movies = get_movies(movies_filename)
	users = get_users(users_filename)
	ratings = get_ratings(ratings_filename)
	num_users = len(users)
	num_categories = len(movies[0].genres)

	user_category_matrix = np.zeros((num_users, num_categories, 2))
	for i in xrange(num_users):
		for j in xrange(num_categories):
			for k in xrange(2):
				user_category_matrix[i][j][k] = 0
	for rating in ratings:
		for category in xrange(num_categories):
			if (movies[rating.item_id - 1].genres[category] > 0):
				if (rating.rating >= 3):
					user_category_matrix[rating.user_id - 1][category][0] += 1
				else:
					user_category_matrix[rating.user_id - 1][category][1] += 1

	return user_category_matrix

