# MNIST Neural Net

A fully connected feedforward multi-layer neural network using backpropogation to learn the MNIST dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
##### C++17
You will need a compiler which supports C++17.

##### cmake >= 3.9
For this you may need to install pip. See [here](https://packaging.python.org/guides/installing-using-linux-tools/).
Then install virtualenv, create a virtualenv and activate the virtualenv. See [here](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).
Finally, install cmake locally, inside the virtualenv:
```
pip install cmake
```
##### Python dev libraries
```
sudo apt install py-dev
```

### Build

Once you've cloned the source, cd into the build directory and run the following to build the source code:

```
cmake ..
make
```

### Configure the network
Most configuration options are set in the train.cpp, upon instantantion of the Net class in train.cpp, which has fingerprint:
```
Net(const std::vector<unsigned> topology,
		bool softmax,
		bool plot_graphs,
		float learning_rate,
		bool regularization);
```

The number of epochs of stochastic gradient descent is set as a parameter of the `Net:sgd(int num_iters)` function, called in `Net::train`.



### Train the network

To train the neural network just run:

```
./train
```
