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
typedef std::map<std::string, std::vector<float>> report_t;

struct single_data_t
{
    mat_f_t input;
    mat_f_t output;
};

struct data_t
{
    single_data_t train;
    single_data_t test;
};

struct layer_param_t
{
    mat_f_t weights;
    mat_f_t bias;
};

typedef std::vector<layer_param_t> param_t;

struct eval_t
{
    float cost;
    float accuracy;
    float error;
};

struct sgd_t
{
    param_t params;
    report_t report;
};

// **************** class Net ****************
//
class Net
{
	public:
                Net(const std::vector<unsigned> topology);
		const std::vector<unsigned> m_topology;
                param_t m_params;
                
                data_t mnist_data_set();
                mat_f_t sigmoid(mat_f_t);
                mat_f_t rowwise_sum(mat_f_t, mat_f_t);
                mat_f_t colwise_sum(mat_f_t, mat_f_t);
                mat_f_t feedforward(mat_f_t);
		template<class T> mat_f_t class_to_output(T);
                vec_i_t output_to_class(mat_f_t);
                //void set_weights();
		void train();
                
                layer_param_t init_layer_params(long, long, float);
                param_t init_params();
                single_data_t shuffle_training_data(single_data_t);
		float cross_entropy(float, float);
                mat_f_t sigmoid_prime(mat_f_t);
                mat_f_t cost_derivative(mat_f_t, mat_f_t);
                eval_t evaluate(single_data_t);
                float cost(mat_f_t, mat_f_t);
                float class_error(vec_i_t, vec_i_t);
		void plot_graphs(std::map<std::string, std::vector<float>>);
                param_t backprop(single_data_t);
		param_t update_mini_batch(single_data_t, float);
                sgd_t sgd(data_t, int, int, float, bool);
                
//                void write_matrix_to_csv(mat_f_t matrix, std::string file);
//                void save_params_to_csv(param_t params, std::string path);
//                param_t load_matrix_from_csv(std::string);
//                param_t load_params_from_csv(std::string path);
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

