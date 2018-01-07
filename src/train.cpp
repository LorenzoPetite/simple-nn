#include "Net.h"
#include <vector>
#include <iostream>
#include <istream>
#include <string>

int main()
{
	std::string line = "";

	// Ask user to set topology of neural network.
	std::cout << "Set topology: " << std::endl;
	std::cout << "\teg. <784, 2000, 10>" << std::endl;
	std::cout << "\tfor 784 features," << std::endl;
	std::cout << "\t10 classes," << std::endl;
	std::cout << "\tone hidden layer with 2000 nodes." << std::endl;
	std::vector<unsigned> topology;
	std:: getline(std::cin, line);
	if(!line.empty())
	{
		std::istringstream iss(line);
		unsigned x;
		while (iss >> x)
		{
			topology.push_back(x);
		}
	}
	if (line.empty() || topology.size() < 2){
		//std::cout << "You must enter a valid topology.";
		topology.push_back(4);
		topology.push_back(10);
		topology.push_back(3);
		//return 1;
	}


	// Ask user to choose activation function from list, or use default.
	std::cout << "Which activation_function would you like (default): " << std::endl;
	std::cout << "1: Mean squared error (default)" << std::endl;
	int activation_function = 1;
	std::getline(std::cin, line);
	if (!line.empty())
	{
		std::istringstream iss(line);
		iss >> activation_function;
	}

	// Ask user whether to plot graphs, or use default.
	bool plot_graphs = 1;
	std::cout << "Plot error graphs? (Y/n): ";
	std::getline(std::cin, line);
	if (!line.empty())
	{
		std::string input;
		std::istringstream iss(line);
		iss >> input;
		if (input == "N" || input == "n")
			plot_graphs = 0;
	}
	std::cout << std::endl;

	// Ask user to set learning rate, or use default.
	float learning_rate = 0.1;
	std::cout << "Set learning rate (0.1): ";
	std::getline(std::cin, line);
	if (!line.empty())
	{
		std::istringstream iss(line);
		iss >> learning_rate;
	}
	std::cout << std::endl;

	// regularisation not supported at the moment.
	bool regularization = 0;

	// Construct network
	Net net(topology, activation_function, plot_graphs, learning_rate, regularization);

	net.train();

	return 0;
}
