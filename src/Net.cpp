#include "Net.h"
#include <utility>
#include <memory>
#include "matplotlibcpp.h"
#include <vector>
#include <istream>
#include <fstream>
#include <Eigen/Dense>


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
	m_weights.resize(rows,cols);
	m_weights = mat_nn_t::Random(rows, cols) * bound;
	m_bias.resize(1, cols);
	m_bias = mat_nn_t::Random(1, cols) * bound;
}

void Layer::hyperbolic_tangent()
{
	m_output.resize(m_z.rows(), m_z.cols());
	m_output = ((m_z.array().tanh() + 1) / 2).matrix();
}

void Layer::hyperbolic_tangent_diff()
{
	m_output_diff.resize(m_z.rows(), m_z.cols());
	m_output_diff = ((1 - m_z.array().tanh().square()) / 2).matrix();

}

void Layer::activate(int activation_function)
{
	switch(activation_function) {
		case 1: hyperbolic_tangent();
				break;
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
		std::cout << mat_info(m_v_layer[i]->m_z) << std::endl;
		std::cout << mat_info(m_v_layer[i-1]->m_output) << std::endl;
		m_v_layer[i]->m_z = (m_v_layer[i-1]->m_output * m_v_layer[i-1]->m_weights).rowwise()
							+ vec_bias.transpose();
		m_v_layer[i]->activate(m_activation_function);
	}
}

void Net::sgd()
{

	init_random_batch(60);


	std::cout << "initialised random batch" << std::endl;

	int i = 0;
	while (i < 500)
	{
		std::cout << "Begin epoch: " << i << std::endl;
		feedforward();
		std::cout << "Completed ff" << std::endl;
		cost();
		class_error();
		update_batch();
		std::cout << "Finished epoch: " << i << std::endl;
		++i;
	}
}

void Net::update_batch()
{
	std::cout << "update_batch()" << std::endl;
	for (auto& l : m_v_layer)
	{
		if (l == m_v_layer.front()) continue;
		l->activate_diff(m_activation_function);
	}
	std::cout << "calculated activation differentials" << std::endl;

	auto & out_l = m_v_layer.back();
	// mean squared error derivative:
	out_l->m_delta = out_l->m_output_diff.array()
					* ( out_l->m_output - out_l->m_target_output ).array();

	for (auto l = std::prev(m_v_layer.end(), 2); l != m_v_layer.begin(); --l)
	{
		(*l)->m_delta = (*l)->m_output_diff.array()
						* ( (*(l+1))->m_delta * (*l)->m_weights.transpose() ).array();
	}
	backprop();
}

void Net::backprop()
{
	std::cout << "begin backprop" << std::endl;

	for (auto layer = m_v_layer.begin();
			layer != std::prev(m_v_layer.end(), 1);
			++layer)
	{

		// weights gradient
		mat_nn_t nabla_w = (*layer)->m_output.transpose()
							* (*(layer+1))->m_delta;
		nabla_w = nabla_w / (*layer)->m_output.rows();

		// bias gradient
		mat_nn_t delta_nabla_b = (*(layer+1))->m_delta;

		std::cout << "calculated gradient" << std::endl;

		//mat_nn_t nabla_w = delta_nabla_w.colwise().mean();
		mat_nn_t nabla_b = delta_nabla_b.colwise().mean();

		std::cout << "calculated weights_correction" << std::endl;

		(*layer)->m_weights -= m_learning_rate * nabla_w;
		(*layer)->m_bias -= m_learning_rate * nabla_b;
	}

	std::cout << "finished updating weights" << std::endl;
}

void Net::cost()
{
	auto& target_output = m_v_layer.back()->m_target_output;
	auto& prediction = m_v_layer.back()->m_output;
	m_cost.push_back(
			((target_output.array() - prediction.array())
			.square()
			.sum())
			/ (prediction.rows() * prediction.cols())
		);
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

void Net::class_error()
{
	mat_nn_t classes = output_to_class();
	int rows = classes.rows();
	int cols = classes.cols();
	int sum = 0;
	for (int i = 0; i < rows; i++)
	{
		if ( classes.block(i, 0, 1, cols) != m_target_output.block(i, 0, 1, cols) )
			sum++;
	}
	m_class_error.push_back(sum / rows);
}

void Net::plot_graphs()
{
	namespace plt = matplotlibcpp;
	plt::named_plot("Training cost", m_cost);
	plt::named_plot("Training class", m_class_error);
	plt::xlim(0,500);
	plt::ylim(0,1);
	plt::xlabel("Epoch");
	plt::ylabel("Error");
	plt::legend();
	plt::show();
}

void Net::load_data_set(const std::string data_path)
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
	std::string data_path = "/home/laurence/university/programmingI/final_assignment/oo_neural_net/data/iris_training.dat";
	load_data_set(data_path);

	std::cout << "finished loading data set" << std::endl;

	std::cout << m_v_layer.size() << std::endl;

	for (unsigned i = 0; i < m_topology.size() - 1; i++)
	{
		m_v_layer[i]->init_weights(m_topology[i], m_topology[i+1], 0.5);
		std::cout << "m_weights[" << i << "]" << m_v_layer[i]->m_weights.rows() << "x" << m_v_layer[i]->m_weights.cols() << std::endl;
	}

	std::cout << "finished initialising weights" << std::endl;

	sgd();

	plot_graphs();
}
