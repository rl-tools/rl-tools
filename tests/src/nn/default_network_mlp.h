#include <backprop_tools/nn/nn.h>
#include "../utils/utils.h"


namespace bpt = backprop_tools;

using DTYPE = double;


using NN_DEVICE = bpt::devices::DefaultCPU;
using StructureSpecification = bpt::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, 17, 13, 3, 50, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY>;

using STRUCTURE_SPEC = StructureSpecification;
using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTF<DTYPE, typename NN_DEVICE::index_t>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<STRUCTURE_SPEC>;
using NetworkType = bpt::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC>;

using NETWORK_SPEC_BACKWARD_ONLY = bpt::nn_models::mlp::InferenceBackwardSpecification<StructureSpecification>;
using NetworkTypeBackwardOnly = bpt::nn_models::mlp::NeuralNetworkBackward<NETWORK_SPEC_BACKWARD_ONLY>;

constexpr typename NN_DEVICE::index_t INPUT_DIM = STRUCTURE_SPEC::INPUT_DIM;
constexpr typename NN_DEVICE::index_t LAYER_1_DIM = STRUCTURE_SPEC::HIDDEN_DIM;
constexpr typename NN_DEVICE::index_t LAYER_2_DIM = STRUCTURE_SPEC::HIDDEN_DIM;
constexpr typename NN_DEVICE::index_t OUTPUT_DIM = STRUCTURE_SPEC::OUTPUT_DIM;

class NeuralNetworkTest : public ::testing::Test {
protected:
    std::string DATA_FILE_PATH;;
    std::string model_name = "model_1";
    NeuralNetworkTest(){
        std::string DATA_FILE_NAME = "mlp_data.hdf5";
        const char *data_path_stub = BACKPROP_TOOLS_MACRO_TO_STR(BACKPROP_TOOLS_TESTS_DATA_PATH);
        this->DATA_FILE_PATH = std::string(data_path_stub) + "/" + DATA_FILE_NAME;

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

using StructureSpecification_3 = bpt::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, 17, 13, 3, 50, bpt::nn::activation_functions::GELU, bpt::nn::activation_functions::IDENTITY>;

using NETWORK_SPEC_3 = bpt::nn_models::mlp::AdamSpecification<StructureSpecification_3>;
using NetworkType_3 = bpt::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC_3>;
