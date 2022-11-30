#include <layer_in_c/nn/nn.h>
namespace lic = layer_in_c;

typedef float DTYPE;
constexpr size_t INPUT_DIM = 17;
constexpr size_t LAYER_1_DIM = 50;
constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN =  lic::nn::activation_functions::RELU;
constexpr size_t LAYER_2_DIM = 50;
constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = lic::nn::activation_functions::RELU;
constexpr size_t OUTPUT_DIM = 13;
constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_FN = lic::nn::activation_functions::LINEAR;

typedef lic::nn_models::three_layer_fc::StructureSpecification<DTYPE, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_FN> NETWORK_STRUCTURE_SPEC;
typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, NETWORK_STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParameters<DTYPE>> NETWORK_SPEC;
typedef lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, NETWORK_SPEC> NetworkType;

template <typename T, typename DEVICE, typename SPEC>
T abs_diff(lic::nn::layers::dense::Layer<DEVICE, SPEC> l1, lic::nn::layers::dense::Layer<DEVICE, SPEC> l2) {
    T acc = 0;
    acc += abs_diff_matrix<DTYPE, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(l1.weights, l2.weights);
    acc += abs_diff_vector<DTYPE, SPEC::OUTPUT_DIM>(l1.biases, l2.biases);
    return acc;
}
template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff(lic::nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC> n1, lic::nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC> n2) {
    typename SPEC::T acc = 0;
    acc += abs_diff<DTYPE, DEVICE, typename SPEC::LAYER_1::SPEC>(n1.layer_1, n2.layer_1);
    acc += abs_diff<DTYPE, DEVICE, typename SPEC::LAYER_2::SPEC>(n1.layer_2, n2.layer_2);
    acc += abs_diff<DTYPE, DEVICE, typename SPEC::OUTPUT_LAYER::SPEC>(n1.output_layer, n2.output_layer);
    return acc;
}

class NeuralNetworkTest : public ::testing::Test {
protected:
    const std::string DATA_FILE_PATH = "../model-learning/data.hdf5";
    std::string model_name = "model_1";
    void SetUp() override {
        auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
        data_file.getDataSet("data/X_train").read(X_train);
        data_file.getDataSet("data/Y_train").read(Y_train);
        data_file.getDataSet("data/X_val").read(X_val);
        data_file.getDataSet("data/Y_val").read(Y_val);
        data_file.getDataSet("data/X_mean").read(X_mean);
        data_file.getDataSet("data/X_std").read(X_std);
        data_file.getDataSet("data/Y_mean").read(Y_mean);
        data_file.getDataSet("data/Y_std").read(Y_std);
    }

    std::vector<std::vector<DTYPE>> X_train;
    std::vector<std::vector<DTYPE>> Y_train;
    std::vector<std::vector<DTYPE>> X_val;
    std::vector<std::vector<DTYPE>> Y_val;
    std::vector<DTYPE> X_mean;
    std::vector<DTYPE> X_std;
    std::vector<DTYPE> Y_mean;
    std::vector<DTYPE> Y_std;
};

