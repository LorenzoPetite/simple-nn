# C++ Neural Net

A fully connected feedforward multi-layer neural network using the backpropogation algorithm to learn, for example, to recognise handwritten digits to an accuracy of 98%.

The Net::sgd() function runs stochastic gradient descent to minimise the Mean Squared Error cost function:

![alt text](https://latex.codecogs.com/gif.latex?\frac{1}{m}\sum&space;(t-p)^{2})

where ![alt text](http://latex.codecogs.com/gif.latex?t) is the target output and ![alt text](http://latex.codecogs.com/gif.latex?p) is the predicted output.

The activation function for all node's is the sigmoid function:

![alt text](https://latex.codecogs.com/gif.latex?a&space;=&space;\frac{1}{1&plus;e^{-z}})

where
![alt text](https://latex.codecogs.com/gif.latex?z^{i&plus;1}=\theta^{i}&space;\cdot&space;a^{i}&plus;b^{i})
 such that ![alt text](https://latex.codecogs.com/gif.latex?\theta&space;^{i}) and ![alt text](https://latex.codecogs.com/gif.latex?b^{i}) are the weights and bias, respectively, which connect layer i to layer i+1.
 
Every epoch, the sgd() function performs cross-validation by evaluating the network's performance on a test set.  

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
bool cross_validation = true;
```
##### 6. Run Stochastic Gradient Descent on the network
```
sgd_t sgd_res = net.sgd(data, epochs, batch_size, learning_rate, cross_validation);
```
