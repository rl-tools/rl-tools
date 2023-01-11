#include <layer_in_c/nn/nn.h>
namespace lic = layer_in_c;

using DTYPE = double;


using NN_DEVICE = lic::devices::DefaultCPU;
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, 17, 13, 3, 50, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY>;

using STRUCTURE_SPEC = StructureSpecification;
using NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>>;
using NetworkType = lic::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC>;

using NETWORK_SPEC_BACKWARD_ONLY = lic::nn_models::mlp::InferenceBackwardSpecification<StructureSpecification>;
using NetworkTypeBackwardOnly = lic::nn_models::mlp::NeuralNetworkBackward<NETWORK_SPEC_BACKWARD_ONLY>;

constexpr typename NN_DEVICE::index_t INPUT_DIM = STRUCTURE_SPEC::INPUT_DIM;
constexpr typename NN_DEVICE::index_t LAYER_1_DIM = STRUCTURE_SPEC::HIDDEN_DIM;
constexpr typename NN_DEVICE::index_t LAYER_2_DIM = STRUCTURE_SPEC::HIDDEN_DIM;
constexpr typename NN_DEVICE::index_t OUTPUT_DIM = STRUCTURE_SPEC::OUTPUT_DIM;

class NeuralNetworkTest : public ::testing::Test {
protected:
    std::string DATA_FILE_PATH = "../model-learning/data.hdf5";
    std::string model_name = "model_1";
    void SetUp() override {
        const char* data_file_path = std::getenv("LAYER_IN_C_TEST_NN_DATA_FILE");
        if (data_file_path != NULL){
            DATA_FILE_PATH = std::string(data_file_path);
//            std::runtime_error("Environment variable LAYER_IN_C_TEST_DATA_DIR not set. Skipping test.");
        }

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

using StructureSpecification_3 = lic::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, 17, 13, 3, 50, lic::nn::activation_functions::GELU, lic::nn::activation_functions::IDENTITY>;

using NETWORK_SPEC_3 = lic::nn_models::mlp::AdamSpecification<StructureSpecification_3, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>>;
using NetworkType_3 = lic::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC_3>;
