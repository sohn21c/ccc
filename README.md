# mnist-numpy
A basic fully connected network implemented purely in NumPy and trained on the MNIST dataset.

## Experiments
The MNIST dataset is split into 50000 train, 10000 validation and 10000 test samples. All splits are normalized using the statistics of the training split (using the global mean and standard deviation, not per pixel).

The network has 2 fully connected layers with ReLU activations. The first hidden layer has 256 units and the second 128 units. The network is initialized with Xavier-He initialization.

The network is trained for 250 epochs with vanilla minibatch SGD and learning rate 1e-3. The final accuracy on the test set is about 0.97.


## Code structure:
### layers.py
Contains classes that represent layers for different transformations. Each class has a forward and a backward method that define a transformation and its gradient. The class keeps track of the variables defining the transformation and the variables needed to calculate the gradient. The file also contains a class that defines the softmax cross entropy loss.

### network.py
Defines Network, a configurable class representing a sequential neural network with any combination of layers. Network has a train function that performs minibatch SGD.

### main.py
Data loading, training and validation scripts. Running it trains the networks described in experiments. For loading the data it expects two files "data/mnist_train.csv" and "data/mnist_test.csv". These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/. To run use "python3 main.py".

## Added by James Sohn
### Task 1
- Objective  
	- To implement mean-only batch normalization with scaling `x_hat = x - mu` and output `y = x_hat + beta`.  

- Mean-only batchnorm layer  
	- Code can be found [here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/layers.py#L84)

- Forward pass  
	- Code can be found [here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/layers.py#L103)
	- Forward pass takes the mean of the minibatch and normalize the layer's input to the scale and output as described in objective above with the learanable parameter bias initialized as `zero_vector` of shape `(1, input_dim)`.  
	- The code divides the cases into `train` and `test`. During `train`, exponentially weighted average of population is tracked and later used for normalization at `test` time. Knowing that the `train / val / test` data was normalized the same at `loading`(as shown [here](https://github.com/sohn21c/ccc/blob/6e6bdd4243ef99cc962f04d6909d8d9b4956d071/main.py#L56)) and the batch size at `test` time is big enough(`128`, as shown [here](https://github.com/sohn21c/ccc/blob/6e6bdd4243ef99cc962f04d6909d8d9b4956d071/main.py#L31)), it would have worked without the `train / test` case separation. However, having written the section as such, one can choose to use smaller batch size at `test` time without an issue.  
	- Momentum for exponentially weighted average is set at `0.1` with the calculation `pop_mean = pop_mean * (1-momentum) + batch_mean * momentum`. With the current dataset size, weight momentum can be closer to zero if necessary.    

- Backward pass  
	- Code can be found [here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/layers.py#L112)  
	- Computation graph  
		1. From `y = x_hat + beta`, one can derive `dbeta = np.sum(dY, axis=0)` as `beta` is 1-D vector.  
		2. From `y = x_hat + beta`, one can derive `dx_hat = dY`.  
		3. From `x_hat = x - mu`, one can derive `dmu = -np.sum(dx_hat, axis=0)` as `mu` is 1-D vector.  
		4. From `x_hat = x - mu`, one can derive `dx1 = dx_hat`.  
		5. From `mu = sum(x2) / N`, one can derive `dx2 = dmu / dY.shape[0] * np.ones(dY.shape)` i.e. `dx2 = dmu / N * np.ones(N x D)`.  
		6. Finally, `dx = dx1 + dx2`  
	- The implemented backward pass without standard deviation and learnable parameter gamma for scale does not need cached value from forward pass, however the code still carries it and use it as a flow control to make sure forward pass happened correctly.

- main()
	- Added training network can be found [here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/main.py#L19)  
	- Per the original desciption in the `main()`([here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/main.py#L11)), the network with batchnorm is placed before the network without the batchnorm.  
	- Batchnorm layer has been experimented with being added before or after the activation layer. The network implemented here has layer added before the ReLU activation per original paper.  

### Task 2
- Objective 
	- To write a unit test to do a numerical gradient check on normalization layer, using the pytest framework.  

- File  
 	- `test_bn.py`

- Forward Pass  
	- Code can be found [here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/test_bn.py#L4)  
	- Although the task didn't ask for forward pass testing, it was added for sanity check. The test generates random numpy array of shape (1000, 10) with some perturbation of random scale. It uses the forward pass of batchnorm layer to normalize the input and checks if all elements after the process are close to zero.  

- Numerical gradient check  
	- Code can be found [here](https://github.com/sohn21c/ccc/blob/01c54f5113408ed2f6e30e943f9fff8bb446cfb4/test_bn.py#L19)  
	- Numerical gradient check was done by comparing the gradients, respectively calculated in the backward pass of batchnorm layer and centered difference approximation. The centered difference approximation is used over single sided approximation for accuracy.  
	- Epsilon for centered diff. approximation is set at `1e-4` and criteria for the test if both gradients are close to each other is set at `1e-9`.  

- Run  
	- `py.test` will run both tests.  

### Result
With the mean-only batchnorm implemented, the model converges faster and handles the higher learning well. Shown below are some of the comparsion runs. 

1. learning rate: 1e-3  
	- With BN  
		at 200 epochs,  
		Test loss: 0.09773185980449856  
		Test accuracy: 0.9711  
	- WIthout BN  
		at 250 epochs,  
		Test loss: 0.09541730474917517  
		Test accuracy: 0.9707  

2. learning_rate: 1e-2  
	- With BN  
		at 100 epochs,  
		Test loss: 0.0714296053143353  
		Test accuracy: 0.9779  
	- Without BN  
		at 150 epochs,  
		Test loss: 0.0885770614688055  
		Test accuracy: 0.9783  

3. learning_rate: 3e-2  
	- With BN  
		at 75 epochs,  
		Test loss: 0.07352606911902576  
		Test accuracy: 0.979   
	- Without BN  
		at 100 epochs,  
		Test loss: 0.08451743866894222  
		Test accuracy: 0.9791  

4. learning_rate: 1e-1  
	- With BN  
		at 50 epochs,  
		Test loss: 0.07718941271657326  
		Test accuracy: 0.9816  
	- Without BN  
		at 50 epochs,  
		Test loss: 0.08209482968920581  
		Test accuracy: 0.9799  





























