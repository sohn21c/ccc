import numpy as np
from layers import BN

def test_bn_forward():
	'''
	This method tests if mean-only batchnorm layer's forward pass is correctly implemented by checking the normalized values being close to zero
	'''
	np.random.seed(32)

	N, D = 1000, 10
	X = np.random.randn(N,D)*np.random.randint(5) + np.random.randint(10)
	mean_before = np.mean(X, axis=0)
	bn_layer = BN(D)
	out = bn_layer.forward(X)
	mean_after = np.mean(out, axis=0)

	assert np.all(np.abs(mean_after) < 1e-9), 'forward pass failed'

def test_bn_backward():
	'''
	This method tests if mean-only batchnorm layer's backward pass is correctly implemented with numerical gradient checking
	'''
	np.random.seed(32)

	N, D = 10, 128
	X = np.random.randn(N, D)*np.random.randint(5) + np.random.randint(10)
	bn_layer = BN(D)
	out = bn_layer.forward(X)
	dY = np.random.randn(N, D)
	fX = lambda x: bn_layer.forward(x)
	approx_dX = get_gradient(fX, X, dY)
	dX, *_ = bn_layer.backward(dY)

	norm_approx_dX = np.linalg.norm(approx_dX)
	norm_dX = np.linalg.norm(dX)
	norm_diff = np.linalg.norm(dX - approx_dX)

	assert norm_diff / (norm_dX + norm_approx_dX) < 1e-9, 'backward pass failed'

# helper function
def get_gradient(fx, x, dy, e=1e-4):
	'''
	returns a numpy array grad with centered difference approximation
	'''
	grad = np.zeros(x.shape)
	iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

	while not iterator.finished:
		index = iterator.multi_index
		orig = x[index]
		x[index] = orig + e 
		fx_plus = fx(x)
		x[index] = orig - e
		fx_minus = fx(x)
		x[index] = orig
		grad[index] = np.sum((fx_plus - fx_minus) / (2*e) * dy)
		iterator.iternext()

	return grad
