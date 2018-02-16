# MNIST Neural Net

A fully connected feedforward multi-layer neural network using backpropogation to learn the MNIST dataset.
The error metric 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
##### C++17
You will need a compiler which supports C++17.

##### cmake >= 3.9
Unfortunately, many of the unix distributions do not include this version in their repositories. The easiest method to get the latest version is through pip. For this you may need to install pip. See [here](https://packagingpython.org/guides/installing-using-linux-tools/).
Then install virtualenv, create a virtualenv and activate the virtualenv. See [here](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).
Finally, install cmake locally, inside the virtualenv:
```
pip install cmake
```
##### Python dev libraries, numpy and matplotlib
```
sudo apt-get install python-matplotlib python-numpy python2.7-dev
```

### Build

Once you've cloned the source, cd into the build directory and run the following to build the source code:

```
cmake ..
make
```

### Construct, configure and train the network

Getting started is simple, an example is given in train.cpp file. The basic idea is as follows:

##### 1. Define the network's topology, each entry being the number of layers in that layer
For example, this would define a network with 784 nodes in the input layer, 25 nodes in the hidden layer and 10 nodes in the output layer:
```
std::vector<unsigned> topology = {784, 25, 10};
```
##### 2. Initialise the network with this topology
```
Net net(topology);
```
##### 3. Load the mnist_data_set:
```
auto data = net.mnist_data_set();
```
##### 4. Randomly initialise parameters with a uniform distribution between [-1, 1] for all layers in the network
```
auto params = net.init_params();
```
##### 5. Define other network options
```
int epochs = 10;
int batch_size = 128;
float learning_rate = 0.5;
```
##### 6. Run Stochastic Gradient Descent on the network
```
sgd_t sgd_res = net.sgd(training_data, test_data, epochs, batch_size, learning_rate);
```