#ifndef PETITE_Net
#define PETITE_Net

// Net.h
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

typedef Eigen::MatrixXf mat_f_t;
typedef Eigen::VectorXf vec_f_t;
typedef Eigen::VectorXi	vec_i_t;

// ****************  Layer *****
//
struct Layer
{
	long m_lyr_sz;
	long m_nxt_lyr_sz;
	mat_f_t m_output;
	mat_f_t m_target_output;
	mat_f_t m_weights;
	mat_f_t m_bias;
	mat_f_t m_z_nobias;
	mat_f_t m_z;
	mat_f_t m_output_diff;
	mat_f_t m_bias_diff;
	mat_f_t m_delta_w;
	mat_f_t m_delta_b;
	void activate(int);
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
		const bool m_softmax;
		const bool m_plot_graphs;
		float m_learning_rate;
		const bool m_regularization;
		std::vector<std::unique_ptr<Layer>> m_v_layer;
		std::vector<float> m_cost;
		std::vector<float> m_test_cost;
		std::vector<float> m_test_class_error;
		std::vector<float> m_class_error;
		mat_f_t m_input;
		mat_f_t m_target_output;
		mat_f_t m_test_input;
		mat_f_t m_test_target_output;
		void gradient_descent();
		void mean_sqrd_error();
		float cross_entropy(float, float);
		void shuffle_training_data();
		void sgd();
		void update_batch();
		void backprop();
		float cost();
		float class_error();
		void plot_graphs();
		void load_mnist_data_set();
		void load_iris_data_set(const std::string);

	public:
		Net(const std::vector<unsigned> topology,
			bool softmax,
			bool plot_graphs,
			float learning_rate,
			bool regularization);
		void feedforward();
		vec_i_t output_to_class();
		template<class T> mat_f_t class_to_output(T);
		void set_weights();
		void load_data_set();
		void train();
};


// Helper functions
std::string func_mat_info(const std::vector<mat_f_t> &, const std::string);
std::string func_mat_info(const mat_f_t&, const std::string);

std::string func_mat_info(const std::vector<mat_f_t> & mat, const std::string mat_name)
{
	std::string result = mat_name + ": ";
	for (int i = 0; i < mat.size(); ++i){
		result += i + ": " + std::to_string(mat[i].rows()) + "x" + std::to_string(mat[i].cols());
	}
	return result;
}
std::string func_mat_info(const mat_f_t& mat, const std::string mat_name)
{
	std::string result = mat_name + ": ";
	result += std::to_string(mat.rows()) + "x" + std::to_string(mat.cols());
	return result;
}
// calls func_mat_info with arguments automatically as (matrix_value, matrix_name)
#define mat_info(mat) func_mat_info(mat, #mat)
#endif

