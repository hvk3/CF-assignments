import numpy as np
import scipy as scp

def largest_eigenvalue(A):
	return np.max(np.diag(np.linalg.eigvals(A)))

def perform_L1_minimization(B, U, regularizer, max_iterations):
	V = np.zeros((U.shape[1], B.shape[1]))
	alpha = largest_eigenvalue(np.dot(U.T, U))
	for max_iteration in xrange(max_iterations):
		T = V + 1.0 / alpha * np.dot(U.T, (B - np.dot(U, V)))
		V = np.sign(T) * np.maximum(np.zeros(T.shape), np.abs(T) - regularizer / 2.0 / alpha)
	return V

if __name__ == '__main__':
	U = np.random.randint(0, 11, (100, 50))
	V = np.random.choice(11, (50, 20), p = [0.5] + [0.05] * 10)
	B = np.dot(U, V)
	V_prime = perform_L1_minimization(B, U, 0.0, 5000)
	print ((V - V_prime) ** 2).mean(axis = None)