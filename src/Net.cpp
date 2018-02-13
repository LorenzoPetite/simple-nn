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

std::pair<data_t, data_t> Net::mnist_data_set()
{
	// Load MNIST data
	mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);
	
	mat_f_t raw_images;
	auto& v = dataset.training_images;
//	v = dataset.training_images;
//	raw_images.resize(v[0].size(), v.size());
//	for (int i = 0; i < v.size(); i++)
//		for (int j = 0; j < v[i].size(); j++)
//			raw_images(j, i) = v[i][j];
//	
//	std::cout << raw_images.array().mean() << std::endl;
//	std::cout << raw_images.maxCoeff() << std::endl;

	normalize_dataset(dataset);
	
	data_t training_data;
	data_t test_data;
	
	training_data.second = class_to_output(dataset.training_labels);
	test_data.second = class_to_output(dataset.test_labels);
	
	v = dataset.training_images;
	training_data.first.resize(v[0].size(), v.size());
	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++)
			training_data.first(j, i) = v[i][j];
	
	std::cout << training_data.first.array().mean() << std::endl;
	std::cout << training_data.first.maxCoeff() << std::endl;


	v = dataset.test_images;
	test_data.first.resize(v[0].size(), v.size());
	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++)
			test_data.first(j, i) = v[i][j];
	std::pair<data_t, data_t> mnist_dataset (training_data, test_data);
	return mnist_dataset;
	
}

lyr_param_t Net::init_lyr_params(long rows, long cols, float bound)
{
	//std::srand((int)time(0));
	//std::srand(0);
	mat_f_t weights(rows, cols);
	weights = ( mat_f_t::Random(rows, cols).array() ) * bound;
	mat_f_t bias(1, cols);
	bias = mat_f_t::Random(rows, 1).array();
	return {weights, bias};
}

param_t Net::init_params()
{
	for (unsigned i = 0; i < m_topology.size() - 1; i++)
		m_params.push_back( init_lyr_params(m_topology[i + 1], m_topology[i], 0.1) );
	return m_params;
}

data_t Net::shuffle_training_data(data_t data)
{
	int num_samples = data.first.cols();
	// choose random sample of imputs
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(num_samples);
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
	data_t res;
	res.first = (data.first * perm);
	res.second = (data.second * perm);
	return res;
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
		auto z_nobias = weights * activation;
		auto z = colwise_sum( z_nobias,  bias);
		activation = sigmoid(z);
	}
	return activation;
}

float Net::cost(mat_f_t target_output, mat_f_t output)
{
	return (float)((target_output - output).array().square().sum()) / (float)output.cols();
}

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

float Net::class_error(vec_i_t actual_classes, vec_i_t predicted_classes)
{	
	int rows = predicted_classes.rows(); int cols = predicted_classes.cols();
	int sum = 0;
	// compares column by column
	for (int i = 0; i < rows; i++)
		if ( predicted_classes.row(i) != actual_classes.row(i) )
			sum++;
	return (float)sum / (float)rows;
}

eval_t Net::evaluate(data_t data)
{
	auto& x = data.first;
	auto& y = data.second;
	mat_f_t prediction = feedforward(x);
	vec_i_t predicted_classes = output_to_class(prediction);
	vec_i_t actual_classes = output_to_class(y);
	float cos = cost(y, prediction);
	float err = class_error(actual_classes, predicted_classes);
	float acc = (1.0 - err) * 100;
	return {cos, acc};
}

nabla_t Net::backprop(data_t data)
{
	//std::cout << "***** backprop *****" << std::endl;
	std::vector<mat_f_t> nabla_w;
	std::vector<mat_f_t> nabla_b;
	int num_layers = m_params.size() + 1;
	for (int i = 0; i < num_layers - 1; i++)
	{
		auto& weights = m_params[i].weights;
		auto& bias = m_params[i].bias;
		nabla_w.push_back(mat_f_t::Zero(weights.rows(), weights.cols()));
		nabla_b.push_back(mat_f_t::Zero(bias.rows(), bias.cols()));
	}
	mat_f_t activation = data.first;
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
	mat_f_t delta;
	delta.resizeLike(vec_z.back());
	delta = cost_derivative(data.second, activations.back());
	nabla_b.back() = delta;
	nabla_w.back() = delta * activations[activations.size() - 2].transpose();
	for (int i = num_layers - 3; i >= 0; i--)
 	{
		auto& weights = m_params[i+1].weights;
		mat_f_t z = vec_z[i];
		mat_f_t sp = sigmoid_prime(z);
		delta = (weights.transpose() * delta).array() * sp.array();
		nabla_w[i] = delta * activations[i].transpose();
		nabla_b[i] = delta;
	}
	nabla_t nablas (nabla_w, nabla_b);
	return nablas;
}

param_t Net::update_mini_batch(data_t mini_batch, double eta)
{
	int num_layers = m_params.size() + 1;
	int batch_size = mini_batch.first.cols();
	
	std::vector<mat_f_t> nabla_w;
	std::vector<mat_f_t> nabla_b;
	for (int i = 0; i < num_layers - 1; i++)
	{
		auto& weights = m_params[i].weights;
		auto& bias = m_params[i].bias;
		nabla_w.push_back(mat_f_t::Zero(weights.rows(), weights.cols()));
		nabla_b.push_back(mat_f_t::Zero(bias.rows(), bias.cols()));
	}
	
	float delta_mean_sum = 0;
	for (int i = 0; i < batch_size; i++)
	{
		data_t mini_batch_i (mini_batch.first.col(i), mini_batch.second.col(i));
		nabla_t delta_nablas = backprop(mini_batch_i);
		delta_mean_sum += delta_nablas.first[0].array().mean();
		for (int j = 0; j < num_layers - 1; j++)
		{
			mat_f_t delta_nabla_w = delta_nablas.first[j];
			mat_f_t delta_nabla_b = delta_nablas.second[j];
			nabla_w[j] += delta_nabla_w;
			nabla_b[j] += delta_nabla_b;
		}
	}	
	
	for (int i = 0; i < num_layers - 1; i++)
	{
		m_params[i].weights = m_params[i].weights.array() - ( eta / (float)batch_size ) * nabla_w[i].array();
		m_params[i].bias = m_params[i].bias.array() - ( eta / (float)batch_size ) * nabla_b[i].array();
	}
	return m_params;
}

sgd_t Net::sgd(data_t training_data, data_t test_data, int num_iters, int batch_size, double eta)
{
	std::vector<float> empty_vec = {};
	report_t reports;
	reports.emplace("train acc", empty_vec);
	reports.emplace("train cost", empty_vec);
	reports.emplace("test acc", empty_vec);
	reports.emplace("test cost", empty_vec);
	
	param_t params;
	
	int i = 0;
	while (i < num_iters)
	{
		std::cout << "Begin epoch: " << i+1 << std::endl;
		
		training_data = shuffle_training_data(training_data);
		//test_data = shuffle_training_data(test_data);
		
		auto& input = training_data.first;
		auto& target_output = training_data.second;
		int num_samples = input.cols();
		
		for (int j = 0; j < ( num_samples - batch_size ); j += batch_size)
		{	
			auto x = input.block( 0, j, input.rows(), batch_size );
			auto y = target_output.block( 0, j, target_output.rows(), batch_size );
			data_t mini_batch (x, y);
			params = update_mini_batch(mini_batch, eta);
		}
		
		eval_t train_eval = evaluate(training_data);
		eval_t test_eval = evaluate(test_data);
		reports["train acc"].push_back(train_eval.acc);
		reports["train cost"].push_back(train_eval.cost);
		reports["test acc"].push_back(test_eval.acc);
		reports["test cost"].push_back(test_eval.cost);
		
		std::cout << "Finished epoch: " << i+1
					  << " acc: " << train_eval.acc << "% (" << test_eval.acc << "%)"
					  << " cost: " << (float)train_eval.cost << " (" << test_eval.cost << ")" << std::endl;
		++i;
	}
	return {params, reports};
}

void Net::plot_graphs(report_t cost_data)
{
	namespace plt = matplotlibcpp;
	for (auto& cost_pair : cost_data)
	{
		plt::named_plot(cost_pair.first, cost_pair.second);
	}
	plt::xlim(0, 150);
	plt::ylim(0,5);
	plt::xlabel("Epoch");
	plt::ylabel("Error");
	plt::legend();
	plt::show();
}
