#ifndef PETITE_Net
#define PETITE_Net

// Net.h
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>
#include <map>

typedef Eigen::MatrixXf mat_f_t;
typedef Eigen::VectorXf vec_f_t;
typedef Eigen::VectorXi	vec_i_t;
typedef std::pair<mat_f_t, mat_f_t> data_t;
typedef data_t param_t;
typedef std::pair<std::vector<mat_f_t>, std::vector<mat_f_t>> nabla_t;

// **************** class Net ****************
//
class Net
{
	public:
                Net(const std::vector<unsigned> topology);
		const std::vector<unsigned> m_topology;
                std::vector<param_t> m_params;
                
                std::pair<data_t, data_t> mnist_data_set();
                mat_f_t sigmoid(mat_f_t);
                mat_f_t rowwise_sum(mat_f_t, mat_f_t);
                mat_f_t colwise_sum(mat_f_t, mat_f_t);
                mat_f_t feedforward(mat_f_t);
                vec_i_t output_to_class();
		template<class T> mat_f_t class_to_output(T);
                //void set_weights();
		void train();
                
                param_t init_params(long, long, float);
                data_t shuffle_training_data(data_t);
		float cross_entropy(float, float);
                mat_f_t sigmoid_prime(mat_f_t);
                mat_f_t cost_derivative(mat_f_t, mat_f_t);
		float class_error();
		void plot_graphs(std::map<std::string, std::vector<float>>);
		void sgd(data_t, data_t, int, int, double);
                vec_i_t output_to_class(mat_f_t);
                float class_error(vec_i_t, vec_i_t);
                nabla_t backprop(data_t);
		void update_mini_batch(data_t, double);
		float cost(mat_f_t, mat_f_t);
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

