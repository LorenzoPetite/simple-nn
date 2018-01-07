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
	mat_nn_t m_z;
	mat_nn_t m_output_diff;
	mat_nn_t m_delta;
	void activate(int);
	void activate_diff(int);
	void hyperbolic_tangent();
	void hyperbolic_tangent_diff();
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
		void init_random_batch(unsigned);
		void sgd();
		void update_batch();
		void backprop();
		void cost();
		void class_error();
		void plot_graphs();

	public:
		Net(const std::vector<unsigned> topology,
			int activation_function,
			bool plot_graphs,
			float learning_rate,
			bool regularization);
		void feedforward();
		mat_nn_t output_to_class();
		void set_weights();
		void load_data_set(const std::string path);
		void train();
};
#endif

