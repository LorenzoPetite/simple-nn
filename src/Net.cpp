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


std::string func_mat_info(const std::vector<mat_nn_t> &, const std::string);
std::string func_mat_info(const mat_nn_t&, const std::string);

std::string func_mat_info(const std::vector<mat_nn_t> & mat, const std::string mat_name)
{
	std::string result = mat_name + ": ";
	for (int i = 0; i < mat.size(); ++i){
		result += i + ": " + std::to_string(mat[i].rows()) + "x" + std::to_string(mat[i].cols());
	}
	return result;
}
std::string func_mat_info(const mat_nn_t& mat, const std::string mat_name)
{
	std::string result = mat_name + ": ";
	result += std::to_string(mat.rows()) + "x" + std::to_string(mat.cols());
	return result;
}
// calls func_mat_info with arguments automatically as (matrix_value, matrix_name)
#define mat_info(mat) func_mat_info(mat, #mat)

Net::Net(const std::vector<unsigned> topology,
		int activation_function,
		bool plot_graphs,
		float learning_rate,
		bool regularization) :
			m_topology(topology),
			m_activation_function(activation_function),
			m_plot_graphs(plot_graphs),
			m_learning_rate(learning_rate),
			m_regularization(regularization)
{

	// Construct layers
	for (unsigned i = 0; i < m_topology.size(); i++)
	{
		std::unique_ptr<Layer> l(new Layer);
		m_v_layer.push_back(std::move(l));
	}

}

void Layer::init_weights(long rows, long cols, float bound)
{
	//std::srand((int)time(0));
	m_weights.resize(rows,cols);
	m_weights = ( mat_nn_t::Random(rows, cols).array() ) * bound;
	m_bias.resize(1, cols);
	m_bias = mat_nn_t::Random(1, cols).array() ;
	std::cout << "m_bias: " << std::endl << m_bias << std::endl;
}

void Layer::hyperbolic_tangent()
{
	m_output.resize(m_z.rows(), m_z.cols());
	m_output = ((m_z.array().tanh() + 1) / 2).matrix();
}

void Layer::softmax()
{
	std::cout << "softmax()" << std::endl;
	mat_nn_t t;
	t.resizeLike(m_z);
	t = m_z.array().exp();
	std::cout << "m_z.row(0): " << std::endl;
	std::cout << m_z.row(0) << std::endl;
	std::cout << "t.row(0): " << std::endl;
	std::cout << t.row(0) << std::endl;
	Eigen::VectorXf t_total = t.rowwise().sum();
	std::cout << "t_total(0): " << std::endl;
	std::cout << t_total(0) << std::endl;
	mat_nn_t g;
	g.resizeLike(m_z);
	for (int i = 0; i < t_total.rows(); i++)
		g.row(i) = t.row(i).array() / t_total(i);
	m_output = g;
	std::cout << "m_output.row(0): ";
	std::cout << m_output.row(0) << std::endl;
	std::cout << "finished softmax()";
}

void Layer::sigmoid()
{
	m_output.resizeLike(m_z);
	m_output = (1.0 / (1.0 + ((-1.0)*m_z.array()).exp()) ).matrix();
}

void Layer::hyperbolic_tangent_diff()
{
	m_output_diff.resize(m_z.rows(), m_z.cols());
	m_output_diff = ((1 - (m_z).array().tanh().square()) / 2).matrix();
}

void Layer::activate(int activation_function)
{
	switch(activation_function) {
		case 1: hyperbolic_tangent();
				break;
		case 2: softmax();
				break;
		case 3: sigmoid();
	}
}

void Layer::activate_diff(int activation_function)
{
	switch(activation_function) {
		case 1: hyperbolic_tangent_diff();
				break;
	}
}

void Net::feedforward()
{
	for (unsigned i = 1; i < m_topology.size(); i++)
	{
		m_v_layer[i]->m_z.resize(m_v_layer[i-1]->m_output.rows(),
							 m_v_layer[i-1]->m_weights.cols());
		auto& bias = m_v_layer[i-1]->m_bias;
		Eigen::VectorXf vec_bias(Eigen::Map<Eigen::VectorXf>(bias.data(), bias.rows()*bias.cols()));
		std::cout << mat_info(vec_bias) << std:: endl;
		std::cout << mat_info(m_v_layer[i-1]->m_weights) << std::endl;
		std::cout << mat_info(m_v_layer[i-1]->m_output) << std::endl;
		std::cout << mat_info(m_v_layer[i]->m_z) << std::endl;
		m_v_layer[i]->m_z_nobias = (m_v_layer[i-1]->m_output * m_v_layer[i-1]->m_weights);
		m_v_layer[i]->m_z = m_v_layer[i]->m_z_nobias.rowwise() + vec_bias.transpose();
		if (i == m_topology.size() - 1)
			m_v_layer[i]->activate(2); // output layer ~> softmax
		else
			m_v_layer[i]->activate(3); // hidden layer ~> sigmoid
	}
}

void Net::sgd()
{
	init_random_batch(60);

	std::cout << "initialised random batch" << std::endl;

	int i = 0;
	while (i < 500)
	{
		std::cout << "Begin epoch: " << i+1 << std::endl;
		feedforward();
		std::cout << "Completed ff" << std::endl;
		cost();
		class_error();
		update_batch();
		std::cout << "Finished epoch: " << i+1 << std::endl;
		++i;
	}
}

void Net::update_batch()
{
	std::cout << "update_batch()" << std::endl;

	auto & out_l = m_v_layer.back();
	out_l->m_delta_w.resizeLike(out_l->m_output);
	out_l->m_delta_w = ( out_l->m_output.array() - out_l->m_target_output.array() ).matrix();

	std::cout << mat_info(out_l->m_delta_w) << std::endl;
	std::cout << out_l->m_delta_w.row(0) << std::endl;

	for (auto l = std::prev(m_v_layer.end(), 2); l != m_v_layer.begin(); --l)
	{
		mat_nn_t weighted_error;
		weighted_error.resize((*(l+1))->m_delta_w.rows(), (*l)->m_weights.rows());
		weighted_error = ( (*(l+1))->m_delta_w * (*l)->m_weights.transpose() ).array();
		std::cout << mat_info(weighted_error) << std::endl;
		mat_nn_t sigmoid_diff;
		sigmoid_diff.resizeLike((*l)->m_output);
		sigmoid_diff = ( (*l)->m_output.array() * ( 1.0 - (*l)->m_output.array() ) );
		std::cout << mat_info(sigmoid_diff) << std::endl;
		mat_nn_t res;
		res.resizeLike(weighted_error);
		res = weighted_error.array() * sigmoid_diff.array();
		(*l)->m_delta_w = res;
		std::cout << mat_info((*l)->m_delta_w) << std::endl;
		std::cout << (*l)->m_delta_w.row(0) << std::endl;
	}
	std::cout << "calculated deltas\n" << std::endl;
	backprop();
}

void Net::backprop()
{
	std::cout << "begin backprop" << std::endl;

	for (auto l = m_v_layer.begin();
			  l != std::prev(m_v_layer.end(), 1); // first lyr to 2nd-to-last lyr (not incl output lyr)
			  ++l)
	{

		// weights gradient
		mat_nn_t nabla_w =
			( (*l)->m_output.transpose() ) * (*(l+1))->m_delta_w;
		nabla_w = nabla_w / (*l)->m_output.rows(); // divide by batch size

		// bias gradient
		mat_nn_t delta_nabla_b = (*(l+1))->m_delta_w;

		std::cout << "calculated gradient" << std::endl;

		//mat_nn_t nabla_w = delta_nabla_w.colwise().mean();
		mat_nn_t nabla_b = delta_nabla_b.colwise().mean();
		//mat_nn_t nabla_b = delta_nabla_b.transpose();

		std::cout << "calculated weights_correction" << std::endl;

		std::cout << mat_info(nabla_w) << std::endl;
		std::cout << nabla_w.row(0) << std::endl;
		std::cout << mat_info(nabla_b) << std::endl;
		std::cout << nabla_b.row(0) << std::endl;

		std::cout << mat_info((*l)->m_weights) << std::endl;
		std::cout << mat_info((*l)->m_bias) << std::endl;

		(*l)->m_weights -= m_learning_rate * nabla_w;
		(*l)->m_bias -= m_learning_rate * nabla_b;
	}

	std::cout << "finished updating weights" << std::endl;
}

float Net::cross_entropy(float p, float y)
{
	// -( (y)ln(p) + (1-y)ln(1-p) )
	//std::cout << "p, y: " << p << ", " << y << std::endl;
	float cel = 0;
	if (y > 0.5)
		cel = -log(p);
	//else if (y < 0.5)
	//	cel = -log(1.0 - p);
	//std::cout << "cross-entropy loss: " << cel << std::endl;
	return cel;
}

void Net::cost()
{
	std::cout << "cost()\n";
	auto& y = m_v_layer.back()->m_target_output;
	auto& p = m_v_layer.back()->m_output;
	float cost = 0;
	for (int i = 0; i < p.rows(); i ++)
		for (int j = 0; j < p.cols(); j++)
			cost += cross_entropy(p(i,j), y(i,j));
	cost = cost / (float)p.rows();
	m_cost.push_back( cost );
	std::cout << cost << std::endl;
	std::cout << "finished cost()\n";
}

mat_nn_t Net::output_to_class()
{
	auto& o = m_v_layer.back()->m_output;
	int num_rows = o.rows();
	Eigen::VectorXf c(num_rows);
	Eigen::MatrixXf::Index maxIndex[num_rows];
	Eigen::VectorXf maxVal(num_rows);
	for(int i=0; i < num_rows; ++i) {
	    maxVal(i) = o.row(i).maxCoeff( &maxIndex[i] );
		c(i) = maxIndex[i];
	}
	return c;
}

mat_nn_t Net::class_to_output(std::vector<uint8_t> vec_class)
{
	mat_nn_t output = mat_nn_t::Zero(vec_class.size(), m_topology.back());
	for (int i = 0; i < vec_class.size(); i++)
		if (vec_class[i] < output.cols())
			output(i, vec_class[i]) = 1.0;
	return output;
}

mat_nn_t Net::class_to_output(mat_nn_t vec_class)
{
	std::cout << "class_to_output()" << std::endl;
	mat_nn_t output = mat_nn_t::Zero(vec_class.size(), m_topology.back());
	for (int i = 0; i < vec_class.size(); i++)
		if (vec_class(i, 0) < output.cols())
		{
			output(i, vec_class(i, 0)) = 1.0;
		}
	std::cout << "finished class_to_output()" << std::endl;
	return output;
}

void Net::class_error()
{
	std::cout << "class_error()" << std::endl;
	mat_nn_t classes = output_to_class();
	mat_nn_t target_output = m_v_layer.back()->m_target_output;
	mat_nn_t prediction = class_to_output(classes);
	int rows = prediction.rows();
	int cols = prediction.cols();
	int sum = 0;
	for (int i = 0; i < rows; i++)
		if ( prediction.block(i, 0, 1, cols) != target_output.block(i, 0, 1, cols) )
			sum++;
	m_class_error.push_back((float)sum / (float)rows);
	std::cout << "finished class_error()" << std::endl;
}

void Net::plot_graphs()
{
	namespace plt = matplotlibcpp;
	plt::named_plot("Training cost", m_cost);
	plt::named_plot("Training class", m_class_error);
	plt::xlim(0, (int)m_cost.size());
	plt::ylim(0,5);
	plt::xlabel("Epoch");
	plt::ylabel("Error");
	plt::legend();
	plt::show();
}

void Net::load_iris_data_set(const std::string data_path)
{
	std::ifstream file(data_path);

	unsigned num_features = m_topology.front();
	unsigned num_classes = m_topology.back();

	int num_samples = 0;
	float input;
	std::vector<float> v_input_data;
	std::vector<float> v_output_data;
	std::string sample;

	while (std::getline(file, sample))
	{
		std::istringstream iss(sample);
		for (int cols = 0; cols < num_features; cols++)
		{
			iss >> input;
			v_input_data.push_back(input);
		}
		for (int cols = num_features; cols < num_features + num_classes; cols++)
		{
			iss >> input;
			v_output_data.push_back(input);
		}
		num_samples++;
	}

	m_input.resize(num_samples, num_features);
	m_target_output.resize(num_samples, num_classes);
	for (int i = 0; i < num_samples; i++)
	{
		for (int j = 0; j < num_features; j++)
			m_input(i,j) = v_input_data[i+j];
		for (int j= 0; j < num_classes; j++)
			m_target_output(i,j) = v_output_data[i+j];
	}
}

void Net::load_mnist_data_set()
{
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
	// Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	normalize_dataset(dataset);
	m_target_output = class_to_output(dataset.training_labels);
	auto& v = dataset.training_images;
	m_input.resize(v.size(), 784);
	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++)
			m_input(i, j) = v[i][j];
	std::cout << "max: " << m_input.maxCoeff() << std::endl;
}

void Net::load_data_set()
{
	load_mnist_data_set();
}

void Net::init_random_batch(unsigned batch_size)
{
	// choose random sample of imputs
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_input.rows());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
	mat_nn_t rand_inputs = (perm * m_input).topRows(batch_size);
	mat_nn_t rand_outputs = (perm * m_target_output).topRows(batch_size);
	// place sample inputs into output of input layer
	m_v_layer.front()->m_output.resize(rand_inputs.rows(), rand_inputs.cols());
	m_v_layer.front()->m_output = rand_inputs;
	//place sample outputs into target_output vector of output layer
	m_v_layer.back()->m_target_output.resize(rand_outputs.rows(), rand_outputs.cols());
	m_v_layer.back()->m_target_output = rand_outputs;
}

void Net::train()
{
	load_data_set();

	std::cout << "finished loading data set" << std::endl;

	std::cout << m_v_layer.size() << std::endl;

	for (unsigned i = 0; i < m_topology.size() - 1; i++)
	{
		m_v_layer[i]->init_weights(m_topology[i], m_topology[i+1], 0.5);
		std::cout << "m_weights[" << i << "]: " << m_v_layer[i]->m_weights.rows() << "x" << m_v_layer[i]->m_weights.cols() << std::endl;
	}

	std::cout << "finished initialising weights" << std::endl;

	std::cout << "train()";
	std::cout << mat_info(m_input) << std::endl;
	std::cout << m_input.topRows(1) << std::endl;
	std::cout << mat_info(m_target_output) << std::endl;
	std::cout << m_target_output.topRows(1) << std::endl;

	sgd();



	plot_graphs();
}
