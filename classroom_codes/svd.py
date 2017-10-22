import numpy as np

def iterative_PCA(ratings_matrix, k, max_iterations):
	N, M = ratings_matrix.shape
	user_ratings_matrix = np.copy(ratings_matrix)
	for i in xrange(N):
		for j in xrange(M):
			if (ratings_matrix[i, j] == 0):
				user_ratings_matrix[i, j] = np.sum(ratings_matrix[i, :]) / np.count_nonzero(ratings_matrix[i, :])
				user_ratings_matrix[i, j] += np.sum(ratings_matrix[:, j]) / np.count_nonzero(ratings_matrix[:, j])
				user_ratings_matrix[i, j] /= 2
	U, s, V = None, None, None
	for iteration in xrange(max_iterations):
		for j in xrange(M):
			user_ratings_matrix[:, j] -= np.mean(user_ratings_matrix[:, j])
		U, s, V = np.linalg.svd(user_ratings_matrix)
		U = U[:, :k]
		s = np.diag(s)[:k, :k]
		V = V[:k, :]
		new_user_ratings_matrix = np.around(np.dot(np.dot(U, s), V))
		for j in xrange(M):
			user_ratings_matrix[:, j] += np.mean(user_ratings_matrix[:, j])
		for i in xrange(N):
			for j in xrange(M):
				if (user_ratings_matrix[i, j] == 0):
					user_ratings_matrix[i, j] = new_user_ratings_matrix[i, j]
	return np.dot(U, s), V

if __name__ == '__main__':
	N, M, k = 20, 30, 10
	ratings_matrix = np.random.randint(0, 5, (N, M)).astype('float64')
	U, V = iterative_PCA(ratings_matrix, k, 1000)
	# print ((np.dot(U, V) - ratings_matrix) ** 2).mean(axis = None)
	print np.dot(U, V), ratings_matrix
