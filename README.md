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
	To implement mean-only batch normalization with scaling `x_hat = x - mu` and output `y = x_hat + beta`.  
- Forward pass  
	- Code can be found [here](https://github.com/sohn21c/ccc/commit/07e82fceca8db3ced5d43eb63b28a94637c73069)
	- Forward pass takes the mean of the minibatch and normalize the layer's input as described in objective above with the learanable parameter bias initialized at `0`.  
	- The code divides the cases into `train` and `test`. During `train`, exponentially weighted average of population is tracked and later used for normalization at `test` time. Knowing that the `train + val + test` data was normalized the same at `loading`(as shown [here](https://github.com/sohn21c/ccc/blob/6e6bdd4243ef99cc962f04d6909d8d9b4956d071/main.py#L56)) and the batch size at `test` time is big enough(`128`, as shown [here](https://github.com/sohn21c/ccc/blob/6e6bdd4243ef99cc962f04d6909d8d9b4956d071/main.py#L31)), it would have worked without the `train`/`test` separation. However, having written the section as such, one can choose to use smaller batch size at `test` time without an issue.  
	- Momentum for exponentially weighted average is set at `0.1` with the calculation `pop_mean = pop_mean * (1-momentum) + mean * momentum`    

- Backward pass  
### Task 2
### Reesult
### Comment