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

Net::Net(const std::vector<unsigned> topology,
		bool softmax,
		bool plot_graphs,
		float learning_rate,
		bool regularization) :
			m_topology(topology),
			m_softmax(softmax),
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

void Net::load_mnist_data_set()
{
	// Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	normalize_dataset(dataset);
	m_target_output = class_to_output(dataset.training_labels);
	m_test_target_output = class_to_output(dataset.test_labels);
	auto& v = dataset.training_images;
	m_input.resize(v.size(), 784);
	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++)
			m_input(i, j) = v[i][j];

	auto& w = dataset.test_images;
	m_test_input.resize(w.size(), 784);
	for (int i = 0; i < w.size(); i++)
		for (int j = 0; j < w[i].size(); j++)
			m_test_input(i, j) = v[i][j];
}

void Net::load_data_set()
{
	load_mnist_data_set();
}

void Layer::init_weights(long rows, long cols, float bound)
{
	//std::srand((int)time(0));
	m_weights.resize(rows,cols);
	m_weights = ( mat_f_t::Random(rows, cols).array() ) * bound;
	m_bias.resize(1, cols);
	m_bias = mat_f_t::Random(1, cols).array() ;
}

void Net::shuffle_training_data()
{
	// choose random sample of imputs
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_input.rows());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
	m_input = (perm * m_input);
	m_target_output = (perm * m_target_output);
}

void Layer::softmax()
{
	mat_f_t t;
	t.resizeLike(m_z);
	t = m_z.array().exp();
	Eigen::VectorXf t_total = t.rowwise().sum();
	mat_f_t g;
	m_output.resizeLike(m_z);
	for (int i = 0; i < t_total.rows(); i++)
		m_output.row(i) = t.row(i).array() / t_total(i);
}

void Layer::sigmoid()
{
	m_output.resizeLike(m_z);
	m_output = (1.0 / (1.0 + ((-1.0)*m_z.array()).exp()) ).matrix();
}

void Layer::hyperbolic_tangent()
{
	m_output.resize(m_z.rows(), m_z.cols());
	m_output = ((m_z.array().tanh() + 1) / 2).matrix();
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

void Net::feedforward()
{
	for (unsigned i = 1; i < m_topology.size(); i++)
	{
		m_v_layer[i]->m_z.resize(m_v_layer[i-1]->m_output.rows(),
							 m_v_layer[i-1]->m_weights.cols());
		auto& bias = m_v_layer[i-1]->m_bias;
		Eigen::VectorXf vec_bias(Eigen::Map<Eigen::VectorXf>(bias.data(), bias.rows()*bias.cols()));
		m_v_layer[i]->m_z_nobias = (m_v_layer[i-1]->m_output * m_v_layer[i-1]->m_weights);
		m_v_layer[i]->m_z = m_v_layer[i]->m_z_nobias.rowwise() + vec_bias.transpose();
		if (i == m_topology.size() - 1)
			m_v_layer[i]->activate(2); // output layer ~> softmax
		else
			m_v_layer[i]->activate(3); // hidden layer ~> sigmoid
	}
}

float Net::cross_entropy(float p, float y)
{
	// -( (y)ln(p) + (1-y)ln(1-p) )
	float cel = 0;
	if (y > 0.5)
		cel = -log(p);
	return cel;
}

float Net::cost()
{
	auto& y = m_v_layer.back()->m_target_output;
	auto& p = m_v_layer.back()->m_output;
	float cost = 0;
	for (int i = 0; i < p.rows(); i ++)
		for (int j = 0; j < p.cols(); j++)
			cost += cross_entropy(p(i,j), y(i,j));
	cost = cost / (float)p.rows();
	return cost;
}

vec_i_t Net::output_to_class()
{
	auto& o = m_v_layer.back()->m_output;
	int num_rows = o.rows();
	vec_i_t c(num_rows);
	vec_i_t::Index maxIndex[num_rows];
	vec_f_t maxVal(num_rows);
	for(int i=0; i < num_rows; ++i) {
	    maxVal(i) = o.row(i).maxCoeff( &maxIndex[i] );
		c(i) = (int)maxIndex[i];
	}
	return c;
}

template <class T>
mat_f_t Net::class_to_output(T vec_class)
{
	mat_f_t result = mat_f_t::Zero(vec_class.size(), m_topology.back());
	for (int i = 0; i < vec_class.size(); i++)
		result(i, vec_class[i]) = 1.0; // set result(i,j) = 1 (where j = class number)
	return result;
}

float Net::class_error()
{
	vec_i_t classes = output_to_class();
	mat_f_t prediction = class_to_output(classes);
	mat_f_t target_output = m_v_layer.back()->m_target_output;
	int rows = prediction.rows();
	int cols = prediction.cols();
	int sum = 0;
	for (int i = 0; i < rows; i++)
		if ( prediction.block(i, 0, 1, cols) != target_output.block(i, 0, 1, cols) )
			sum++;
	return (float)sum / (float)rows;
}

void Layer::hyperbolic_tangent_diff()
{
	m_output_diff.resize(m_z.rows(), m_z.cols());
	m_output_diff = ((1 - (m_z).array().tanh().square()) / 2).matrix();
}

void Layer::activate_diff(int activation_function)
{
	switch(activation_function) {
		case 1: hyperbolic_tangent_diff();
				break;
	}
}

void Net::sgd()
{
	//shuffle_training_data();

	int i = 0;
	while (i < 50)
	{
		std::cout << "Begin epoch: " << i+1 << std::endl;

		m_v_layer.front()->m_output = m_test_input.topRows(100);
		m_v_layer.back()->m_target_output = m_test_target_output.topRows(100);
		feedforward();
		m_test_cost.push_back( cost() );
		m_test_class_error.push_back( class_error() );

		int btch_sz = 32;
		// place sample inputs into output of input layer
		m_v_layer.front()->m_output = m_input.block( (i*btch_sz) % m_input.rows(), 0, btch_sz, m_input.cols() );
		//place sample outputs into target_output vector of output layer
		m_v_layer.back()->m_target_output = m_target_output.block( (i*btch_sz) % m_target_output.rows(), 0, btch_sz, m_target_output.cols() );
		feedforward();
		m_cost.push_back( cost() );
		m_class_error.push_back( class_error() );

		// weight decay
		if (i < 2)
			m_learning_rate = 0.0005;
		else if ( i < 4 )
			m_learning_rate = 0.0002;
		else if ( i < 7 )
			m_learning_rate = 0.0001;
		else if ( i < 11)
			m_learning_rate = 0.00005;
		else
			m_learning_rate = 0.00001;

		update_batch();
		std::cout << "Finished epoch: " << i+1 << std::endl;
		++i;
	}
}

void Net::update_batch()
{
	auto & out_l = m_v_layer.back();
	out_l->m_delta_w.resizeLike(out_l->m_output);
	out_l->m_delta_w = ( out_l->m_output.array() - out_l->m_target_output.array() ).matrix();

	for (auto l = std::prev(m_v_layer.end(), 2); l != m_v_layer.begin(); --l)
	{
		mat_f_t weighted_error;
		weighted_error.resize((*(l+1))->m_delta_w.rows(), (*l)->m_weights.rows());
		weighted_error = ( (*(l+1))->m_delta_w * (*l)->m_weights.transpose() ).array();
		mat_f_t sigmoid_diff;
		sigmoid_diff.resizeLike((*l)->m_output);
		sigmoid_diff = ( (*l)->m_output.array() * ( 1.0 - (*l)->m_output.array() ) );
		mat_f_t res;
		res.resizeLike(weighted_error);
		res = weighted_error.array() * sigmoid_diff.array();
		(*l)->m_delta_w = res;
	}
	backprop();
}

void Net::backprop()
{
	for (auto l = m_v_layer.begin(); // first layer
			  l != std::prev(m_v_layer.end(), 1); // second to last lyr (i.e. not incl output lyr)
			  ++l)
	{
		// weights gradient
		mat_f_t nabla_w =
			( (*l)->m_output.transpose() ) * (*(l+1))->m_delta_w;
		nabla_w = nabla_w / (*l)->m_output.rows(); // divide by batch size

		// bias gradient
		mat_f_t delta_nabla_b = (*(l+1))->m_delta_w;

		mat_f_t nabla_b = delta_nabla_b.colwise().mean();

		(*l)->m_weights -= m_learning_rate * nabla_w;
		(*l)->m_bias -= m_learning_rate * nabla_b;
	}
}

void Net::plot_graphs()
{
	namespace plt = matplotlibcpp;
	plt::named_plot("Training cost", m_cost);
	plt::named_plot("Test cost", m_test_cost);
	plt::named_plot("Training class", m_class_error);
	plt::named_plot("Test class", m_test_class_error);
	plt::xlim(0, (int)m_cost.size());
	plt::ylim(0,5);
	plt::xlabel("Epoch");
	plt::ylabel("Error");
	plt::legend();
	plt::show();
}

void Net::train()
{
	load_data_set();
	for (unsigned i = 0; i < m_topology.size() - 1; i++)
		m_v_layer[i]->init_weights(m_topology[i], m_topology[i+1], 0.5);
	m_v_layer.front()->m_output = m_test_input;
	m_v_layer.back()->m_target_output = m_test_target_output;
	sgd();
	plot_graphs();
}
