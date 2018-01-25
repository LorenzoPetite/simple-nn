#include "Net.h"
#include <vector>
#include <iostream>
#include <istream>
#include <string>

int main()
{
	std::vector<unsigned> topology = {784, 800, 10};

	bool softmax = true;

	std::string line;
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
	std::cout << "Set learning rate (" << learning_rate << "): ";
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
	Net net(topology, softmax, plot_graphs, learning_rate, regularization);

	net.train();

	return 0;
}
