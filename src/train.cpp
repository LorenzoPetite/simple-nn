#include "Net.h"
#include <vector>
#include <iostream>
#include <istream>
#include <string>

int main()
{
	std::vector<unsigned> topology = {784, 25, 10};
	
	// Construct network
	Net net(topology);
	
	std::srand(0);
	auto data = net.mnist_data_set();
	for (unsigned i = 0; i < net.m_topology.size() - 1; i++)
		net.m_params.push_back( net.init_params(net.m_topology[i + 1], net.m_topology[i], 0.1) );
	//auto output = feedforward(data.first.first.topRows(100));
	//std::cout << output << std::endl;
	net.sgd(data.first, data.second, 100, 128, 0.5);

	return 0;
}
