import numpy as np
import scipy.linalg

np.random.seed(0)

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

def remove_bias(ratings_matrix):
	b_u, b_i = 0.0, 0.0
	global_mean = np.sum(ratings_matrix) / np.count_nonzero(ratings_matrix)
	user_averages = np.sum(ratings_matrix, axis = 0) / ratings_matrix.shape[1]
	item_averages = np.sum(ratings_matrix, axis = 1) / ratings_matrix.shape[0]
	user_biases = user_averages - global_mean
	item_biases = item_averages - global_mean
	return user_biases, item_biases, global_mean

def get_largest_eigenvalue(matrix):
	return np.max(np.linalg.eigvals(np.dot(matrix.T, matrix)))

def init_matrices(num_users, num_items, num_latent_factors):
	U = np.random.randint(0, 5, (num_users, num_latent_factors)).T
	V = np.random.randint(0, 5, (num_items, num_latent_factors))
	return U, V

def least_squares(A, b):
	return np.linalg.lstsq(A.T, b.T)[0]

def soft(T, regularizer):
	return np.sign(T) * np.maximum(np.zeros(T.shape), np.abs(T) - regularizer)

def solve(ratings_matrix, U0, V0, max_iterations, regularizer):
	iteration, beta = 0, 1
	previous_cost, current_cost = 0.0, 0.0
	A, U, V = (ratings_matrix > 0), np.copy(U0), np.copy(V0)
	for _ in xrange(max_iterations):
		Z = np.dot(V, U) + (ratings_matrix - (A * np.dot(V, U))) / beta
		P = np.dot(U, U.T) + regularizer * np.identity(U.shape[0])
		Q = np.dot(Z, U.T)
		V = least_squares(Q, P)
		W = np.dot(V, U) + (ratings_matrix - (A * np.dot(V, U))) / beta
		alpha = 1.01 * get_largest_eigenvalue(V)
		U = soft(U + 1.0 / alpha * np.dot(V.T, W - np.dot(V, U)), regularizer / (2.0 * alpha))
	return U, V
