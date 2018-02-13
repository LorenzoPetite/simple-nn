# MNIST Neural Net

A fully connected feedforward multi-layer neural network using backpropogation to learn the MNIST dataset.

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

An example is given in train.cpp file.