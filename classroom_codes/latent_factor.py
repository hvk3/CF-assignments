import numpy as np
from sklearn.metrics import mean_squared_error

def solve(A, b):
	return np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

def UV_factorization_solver(R, k, num_iters = 10):
	U = np.random.randint(1, 5, (R.shape[0], k))
	V = np.random.randint(1, 5, (R.shape[0], k)).T

	for i in xrange(num_iters):
		U = solve(V, R)
		V = solve(U, R).T
		print U.shape, V.shape
	print U, V

def UV_factorization(R, k):
	U = np.random.randint(1, 5, (R.shape[0], k))
	V = np.random.randint(1, 5, (R.shape[0], k)).T
	regularizer = 0.9
	for i in xrange(10):
		if (i % 2 == 0):
			temp = np.dot(U.T, U) + regularizer * np.identity(np.dot(U.T, U).shape[0])
			temp_inv = np.linalg.inv(temp)
			V = np.dot(np.dot(temp_inv, U.T), R)
		else:
			temp = np.dot(V.T, V) + regularizer * np.identity(np.dot(V.T, V).shape[0])
			temp_inv = np.linalg.inv(temp)
			U = np.dot(np.dot(temp_inv, V.T).T, R).T
			for i in xrange(U.shape[0]):
				U[i, :] /= np.linalg.norm(U[i, :])
	return U, V

if (__name__ == '__main__'):
	k = 10
	m = 50
	U = np.random.randint(1, 5, (m, k))
	V = np.random.randint(1, 5, (m, k)).T
	R = np.dot(U, V)
	R %= 5

	for i in xrange(R.shape[0]):
		for j in xrange(R.shape[1]):
			if (R[i][j] == 0):
				R[i][j] = np.random.randint(1, 5)

	U_prime, V_prime = UV_factorization(R, k)
	U_prime_prime, V_prime_prime = UV_factorization_solver(R, k)
	print mean_squared_error(R, np.dot(U_prime, V_prime))
	print mean_squared_error(R, np.dot(U_prime_prime, V_prime_prime))