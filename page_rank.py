import numpy as np

def Laplacian(L, d):
	N = L.shape[0]
	e = np.ones((N, 1))
	c = np.sum(L, axis = 0)
	A = (1 - d) * (e * e.transpose()) / N + d * np.matmul(L, np.diag(1.0 / c))
	return A

def Page_Rank_closed_form(L, d):
	A = Laplacian(L, d)
	N = L.shape[0]
	w, v = np.linalg.eig(A)
	p = v[:, 0].real
	factor = A.shape[0] / np.sum(p)
	p = factor * p
	return p 

def Page_Rank_power_method(L, d):
	A = Laplacian(L, d)
	N = L.shape[0]
	e = np.ones((N, 1))
	p = np.zeros((N, 1))
	p[:, :] = 1 - d
	threshold = 1e-6
	num_iters = 0
	while True:
		p_new = np.matmul(A, p)
		p_new = N * p_new * (1.0 / np.sum(np.matmul(e.transpose(), p_new)))
		diff = np.sum(np.abs(p - p_new))
		if diff < threshold:
			break
		p = p_new
		num_iters += 1
	return p, num_iters

L = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]])
d = 0.85

p_1 = Page_Rank_closed_form(L, d)
p_1 = np.reshape(p_1, (L.shape[0], 1))
print("Page rank (closed form):")
print(p_1)

p_2, num_iters = Page_Rank_power_method(L, d)
print("Page rank (closed form):")
print(p_2)
print("Number of iterations:", num_iters)

diff = np.sum(np.abs(p_1 - p_2))
print("Total difference between two methods:", diff)