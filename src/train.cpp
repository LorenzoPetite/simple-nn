#include "Net.h"
#include <vector>
#include <iostream>
#include <istream>
#include <string>

int main()
{
	std::srand(0);
	
	std::vector<unsigned> topology = {784, 25, 10};
	
	Net net(topology);
	
	auto data = net.mnist_data_set();
	
	auto params = net.init_params();
	
	auto& training_data = data.first;
	auto& test_data = data.second;
	int epochs = 10;
	int batch_size = 128;
	double learning_rate = 0.5;
	
	sgd_t sgd_res = net.sgd(training_data, test_data, epochs, batch_size, learning_rate);
	
	net.plot_graphs(sgd_res.report);
	
//	report_t& report = sgd_res.report;
//	for (report_t::iterator itr = report.begin(); itr != report.end(); ++itr)	
//		ofstream report_file(itr->first);
//		int size = itr->second.size();
//		for (int i = 0; i < size; i++)
//		    report_file << itr->second[i] << "\n";
//	}
	
	return 0;
}
