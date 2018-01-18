#ifndef PETITE_Net
#define PETITE_Net

// Net.h
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

typedef Eigen::MatrixXf mat_nn_t;

// ****************  Layer *****
//
struct Layer
{
	long m_lyr_sz;
	long m_nxt_lyr_sz;
	mat_nn_t m_output;
	mat_nn_t m_target_output;
	mat_nn_t m_weights;
	mat_nn_t m_bias;
	mat_nn_t m_z_nobias;
	mat_nn_t m_z;
	mat_nn_t m_output_diff;
	mat_nn_t m_bias_diff;
	mat_nn_t m_delta_w;
	mat_nn_t m_delta_b;
	void activate(int);
	void activate_diff(int);
	void hyperbolic_tangent();
	void hyperbolic_tangent_diff();
	void softmax();
	void sigmoid();
	void init_weights(long, long, float);

};


// **************** class Net ****************
//
class Net
{
	private:
		const std::vector<unsigned> m_topology;
		const int m_activation_function;
		const bool m_plot_graphs;
		const float m_learning_rate;
		const bool m_regularization;
		std::vector<std::unique_ptr<Layer>> m_v_layer;
		std::vector<float> m_cost;
		std::vector<float> m_class_error;
		mat_nn_t m_input;
		mat_nn_t m_target_output;
		void gradient_descent();
		void mean_sqrd_error();
		float cross_entropy(float, float);
		void init_random_batch(unsigned);
		void sgd();
		void update_batch();
		void backprop();
		void cost();
		void class_error();
		void plot_graphs();
		void load_mnist_data_set();
		void load_iris_data_set(const std::string);

	public:
		Net(const std::vector<unsigned> topology,
			int activation_function,
			bool plot_graphs,
			float learning_rate,
			bool regularization);
		void feedforward();
		mat_nn_t output_to_class();
		mat_nn_t class_to_output(std::vector<uint8_t>);
		mat_nn_t class_to_output(mat_nn_t);
		void set_weights();
		void load_data_set();
		void train();
};
#endif

