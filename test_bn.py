import numpy as np
from layers import BN

def test_bn_forward():
	'''
	This method tests if mean-only batchnorm layer's forward pass is correctly implemented by checking the normalized values are close to zero
	'''
	np.random.seed(32)

	N, D = 1000, 10
	X = np.random.randn(N,D)*np.random.randint(5) + np.random.randint(10)
	mean_before = np.mean(X, axis=0)
	bn_layer = BN(D)
	out = bn_layer.forward(X)
	mean_after = np.mean(out, axis=0)

	assert np.all(np.abs(mean_after) < 1e-9), 'forward pass failed'