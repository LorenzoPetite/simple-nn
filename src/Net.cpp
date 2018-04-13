#include "Net.h"
#include <utility>
#include <memory>
#include "matplotlibcpp.h"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include <vector>
#include <istream>
#include <fstream>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

Net::Net(const std::vector<unsigned> topology) :
			m_topology(topology)
{
}

template <class T>
mat_f_t Net::class_to_output(T vec_class)
{
	mat_f_t result = mat_f_t::Zero(m_topology.back(), vec_class.size());
	for (int i = 0; i < vec_class.size(); i++)
		result(vec_class[i], i) = 1.0; // set result(i,j) = 1 (where i = class number)
	return result;
}

data_t Net::mnist_data_set()
{
	// Load MNIST data
	mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);
	
	auto& v = dataset.training_images;

	normalize_dataset(dataset);
	
	single_data_t training_data;
	single_data_t test_data;
	
	training_data.output = class_to_output(dataset.training_labels);
	test_data.output = class_to_output(dataset.test_labels);
	
	v = dataset.training_images;
	training_data.input.resize(v[0].size(), v.size());
	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++)
			training_data.input(j, i) = v[i][j];

	v = dataset.test_images;
	test_data.input.resize(v[0].size(), v.size());
	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++)
			test_data.input(j, i) = v[i][j];
	data_t mnist_dataset {training_data, test_data};
	return mnist_dataset;
	
}

// using a uniform distribution in [-1. 1] randomly initialise paramaters for a single layer
layer_param_t Net::init_layer_params(long rows, long cols, float bound)
{
	//std::srand((int)time(0));
	//std::srand(0);
	mat_f_t weights(rows, cols);
	weights = ( mat_f_t::Random(rows, cols).array() ) * bound;
	mat_f_t bias(1, cols);
	bias = mat_f_t::Random(rows, 1).array();
	return {weights, bias};
}

// using a uniform distribution in [-1. 1] randomly initialise paramaters for all layers of network
param_t Net::init_params()
{
	for (unsigned i = 0; i < m_topology.size() - 1; i++)
		m_params.push_back( init_layer_params(m_topology[i + 1], m_topology[i], 0.1) );
	return m_params;
}

void Net::write_matrix_to_csv(mat_f_t matrix, std::string file_name)
{
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
	std::ofstream file(file_name);
	file << matrix.format(CSVFormat);
}

void Net::save_params_to_csv(param_t params, std::string path)
{
	std::ofstream file(path + "params.list");
	for (int i = 0; i < params.size(); i++)
	{
		write_matrix_to_csv(params[i].weights, path + "/" + std::to_string(i) + "_weights.mat" );
		write_matrix_to_csv(params[i].bias, path + "/" + std::to_string(i) + "_bias.mat" );
		file << path + "/" + std::to_string(i) + "_weights.mat\n"
		     << path + "/" + std::to_string(i) + "_bias.mat\n";
	}
	
}

param_t Net::load_matrix_from_csv(std::string file)
{
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
	std::ofstream file(file_name);
	matrix.format(CSVFormat) << file;
}

param_t Net::load_params_from_csv(std::string path)
{
	
}

single_data_t Net::shuffle_training_data(single_data_t data)
{
	int num_samples = data.input.cols();
	// choose random sample of imputs
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(num_samples);
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
	return {(data.input * perm), (data.output * perm)};
}

mat_f_t Net::sigmoid(mat_f_t z)
{
	return (1.0 / (1.0 + ((-1.0)*z.array()).exp()) ).matrix();
}

mat_f_t Net::sigmoid_prime(mat_f_t z)
{
    mat_f_t sp;
    mat_f_t x = sigmoid(z);
    sp = x.array()*(1-x.array());
    return sp;
}

mat_f_t Net::rowwise_sum(mat_f_t mat, mat_f_t mat_vec)
{
	Eigen::VectorXf vec(Eigen::Map<Eigen::VectorXf>(mat_vec.data(), mat_vec.rows()*mat_vec.cols()));
	return mat.rowwise() + vec.transpose();
}

mat_f_t Net::colwise_sum(mat_f_t mat, mat_f_t mat_vec)
{
	Eigen::VectorXf vec(Eigen::Map<Eigen::VectorXf>(mat_vec.data(), mat_vec.rows()*mat_vec.cols()));
	return mat.colwise() + vec;
}

mat_f_t Net::feedforward(mat_f_t activation)
{
	for (unsigned i = 0; i < m_params.size(); i++)
	{
		auto weights = m_params[i].weights;
		auto bias = m_params[i].bias;
		auto z = colwise_sum( weights * activation,  bias);
		activation = sigmoid(z);
	}
	return activation;
}

// mean squared error
float Net::cost(mat_f_t target_output, mat_f_t output)
{
	return (float)((target_output - output).array().square().sum()) / (float)output.cols();
}

// mean squared error derivative
mat_f_t Net::cost_derivative(mat_f_t target_output, mat_f_t output)
{
	return output - target_output;
}

// finds max coeff in each column and returns row vector of their positions
vec_i_t Net::output_to_class(mat_f_t o)
{
	// o size = (num_classes x num_samples)
	int num_samples = o.cols();
	vec_i_t c(num_samples);
	vec_i_t::Index maxIndex[num_samples];
	vec_f_t maxVal(num_samples);
	for(int i=0; i < num_samples; ++i) {
		maxVal(i) = o.col(i).maxCoeff( &maxIndex[i] );
		c(i) = (int)maxIndex[i];
	}
	return c;
}

// returns proportion of rows of actual_classes != rows of predicted_classes
float Net::class_error(vec_i_t actual_classes, vec_i_t predicted_classes)
{	
	int rows = predicted_classes.rows(); int cols = predicted_classes.cols();
	int sum = 0;
	for (int i = 0; i < rows; i++)
		if ( predicted_classes.row(i) != actual_classes.row(i) )
			sum++;
	return (float)sum / (float)rows;
}

// runs feedforward on a dataset and calculates cost and accuracy
eval_t Net::evaluate(single_data_t data)
{
	auto& x = data.input;
	auto& y = data.output;
	mat_f_t prediction = feedforward(x);
	vec_i_t predicted_classes = output_to_class(prediction);
	vec_i_t actual_classes = output_to_class(y);
	float cos = cost(y, prediction);
	float error = class_error(actual_classes, predicted_classes);
	float accuracy = (1.0 - error) * 100;
	return {cos, accuracy, error};
}

// calculate partial derivatives of cost function with respect to weights and bias for all layers
// of the network using backpropogation algorithm.
param_t Net::backprop(single_data_t data)
{
	param_t nabla;
	int num_layers = m_params.size() + 1;
	for (int i = 0; i < num_layers - 1; i++)
	{
		auto& w = m_params[i].weights;
		auto& b = m_params[i].bias;
		layer_param_t layer_nabla_zero {
			mat_f_t::Zero(w.rows(), w.cols()),
			mat_f_t::Zero(b.rows(), b.cols())
		};
		nabla.push_back(layer_nabla_zero);
	}
	mat_f_t activation = data.input;
	std::vector<mat_f_t> activations;
	activations.push_back(activation);
	std::vector<mat_f_t> vec_z;
	for (int i = 0; i < num_layers - 1; i++)
	{
		auto& weights = m_params[i].weights;
		auto& bias = m_params[i].bias;
		mat_f_t z = colwise_sum((weights * activation), bias);
		vec_z.push_back(z);
		activation = sigmoid(z);
		activations.push_back(activation);
	}
	mat_f_t	delta = cost_derivative(data.output, activations.back());
	nabla.back().weights = delta;
	nabla.back().weights = delta * activations[activations.size() - 2].transpose();
	for (int i = num_layers - 3; i >= 0; i--)
 	{
		auto& weights = m_params[i+1].weights;
		mat_f_t z = vec_z[i];
		mat_f_t sp = sigmoid_prime(z);
		delta = (weights.transpose() * delta).array() * sp.array();
		nabla[i].weights = delta * activations[i].transpose();
		nabla[i].bias = delta;
	}
	return nabla;
}

// runs backpropogation to calculate gradients and uses result to update the
// parameters of the network
param_t Net::update_mini_batch(single_data_t mini_batch, float eta)
{
	int num_layers = m_params.size() + 1;
	int batch_size = mini_batch.input.cols();
	
	param_t nabla;
	for (int i = 0; i < num_layers - 1; i++)
	{
		auto& w = m_params[i].weights;
		auto& b = m_params[i].bias;
		layer_param_t layer_nabla_zero {
			mat_f_t::Zero(w.rows(), w.cols()),
			mat_f_t::Zero(b.rows(), b.cols())
		};
		nabla.push_back(layer_nabla_zero);
	}
	
	for (int i = 0; i < batch_size; i++)
	{
		single_data_t mini_batch_i {mini_batch.input.col(i), mini_batch.output.col(i)};
		param_t delta_nabla = backprop(mini_batch_i);
		for (int j = 0; j < num_layers - 1; j++)
		{
			nabla[j].weights += delta_nabla[j].weights;
			nabla[j].bias += delta_nabla[j].bias;
		}
	}	
	
	for (int i = 0; i < num_layers - 1; i++)
	{
		m_params[i].weights = m_params[i].weights.array() - ( eta / (float)batch_size ) * nabla[i].weights.array();
		m_params[i].bias = m_params[i].bias.array() - ( eta / (float)batch_size ) * nabla[i].bias.array();
	}
	return m_params;
}


// runs stochastic gradient descent (sgd) algorithm
sgd_t Net::sgd(data_t data, int num_iters, int batch_size, float eta, bool cross_validation)
{
	std::vector<float> empty_vec = {};
	report_t reports;
	reports.emplace("train error", empty_vec);
	reports.emplace("train cost", empty_vec);
	reports.emplace("test error", empty_vec);
	reports.emplace("test cost", empty_vec);
	
	param_t params;
	
	int i = 0;
	while (i < num_iters)
	{
		std::cout << "Begin epoch: " << i+1 << std::endl;
		
		data.train = shuffle_training_data(data.train);
		//data.test = shuffle_training_data(data.test);
		
		int num_samples = data.train.input.cols();
		
		for (int j = 0; j < ( num_samples - batch_size ); j += batch_size)
		{	
			auto x = data.train.input.block( 0, j, data.train.input.rows(), batch_size );
			auto y = data.train.output.block( 0, j, data.train.output.rows(), batch_size );
			single_data_t mini_batch {x, y};
			params = update_mini_batch(mini_batch, eta);
		}
		
		eval_t train_eval = evaluate(data.train);
		reports["train error"].push_back(train_eval.error);
		reports["train cost"].push_back(train_eval.cost);
		
		if (cross_validation)
		{
			eval_t test_eval = evaluate(data.test);
			reports["test error"].push_back(test_eval.error);
			reports["test cost"].push_back(test_eval.cost);
			std::cout << "Finished epoch: " << i+1
					  << " accuracy: " << train_eval.accuracy << "% (" << test_eval.accuracy << "%)"
					  << " cost: " << (float)train_eval.cost << " (" << test_eval.cost << ")" << std::endl;
		} else
		{
			std::cout << "Finished epoch: " << i+1
					  << " accuracy: " << train_eval.accuracy << "% "
					  << " cost: " << (float)train_eval.cost << std::endl;
		}
		++i;
	}
	return {params, reports};
}

void Net::plot_graphs(report_t cost_data)
{
	namespace plt = matplotlibcpp;
	int max_epoch_size = 5;
	int total_max_error = 1;
	for (auto& cost_pair : cost_data)
	{
		plt::named_plot(cost_pair.first, cost_pair.second);
		int epoch_size = cost_pair.second.size();
		if (epoch_size > max_epoch_size) { max_epoch_size = epoch_size; }
		std::vector<float>::iterator max_error_position;
		max_error_position = std::max_element(cost_pair.second.begin(), cost_pair.second.end());
		int max_error = *max_error_position;
		if (max_error > total_max_error) { total_max_error = max_error; }
	}
	plt::xlim(0, max_epoch_size);
	plt::ylim(0, total_max_error);
	plt::xlabel("Epoch");
	plt::ylabel("Error");
	plt::legend();
	plt::show();
}
