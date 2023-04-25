#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn_models/models.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>

namespace bpt = backprop_tools;

#include <random>
#include <chrono>
#include <highfive/H5File.hpp>

using T = float;
using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

constexpr TI BATCH_SIZE = 32;
constexpr TI NUM_EPOCHS = 10;
constexpr TI INPUT_DIM = 28 * 28;
constexpr TI OUTPUT_DIM = 28 * 28;
constexpr TI NUM_LAYERS = 3;
constexpr TI HIDDEN_DIM = 50;
constexpr TI DATASET_SIZE_TRAIN = 60000;
constexpr TI DATASET_SIZE_VAL = 10000;
using StructureSpecification = bpt::nn_models::mlp::StructureSpecification<T, DEVICE::index_t, INPUT_DIM, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, 1>;

using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTF<T>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<StructureSpecification>;
using NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC>;

int main(){
    std::string dataset_path = "examples/docker/00_basic_mnist/mnist.hdf5";
    const char* dataset_path_env = std::getenv("BACKPROP_TOOLS_NN_MNIST_DATA_FILE");
    if (dataset_path_env != NULL){
        dataset_path = std::string(dataset_path_env);
    }


    auto data_file = HighFive::File(dataset_path, HighFive::File::ReadOnly);

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    OPTIMIZER optimizer;
    device.logger = &logger;
    NETWORK_TYPE network;
    typename NETWORK_TYPE::Buffers<1> buffers;

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, DATASET_SIZE_TRAIN, INPUT_DIM>> x_train;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, DATASET_SIZE_VAL, INPUT_DIM>> x_val;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, DATASET_SIZE_TRAIN, 1>> y_train;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, DATASET_SIZE_VAL, 1>> y_val;

    bpt::MatrixDynamic<bpt::matrix::Specification<T, DEVICE::index_t, 1, OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> d_loss_d_output_matrix;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, DEVICE::index_t, 1, INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> d_input_matrix;

    bpt::malloc(device, network);
    bpt::malloc(device, buffers);
    bpt::malloc(device, x_train);
    bpt::malloc(device, y_train);
    bpt::malloc(device, x_val);
    bpt::malloc(device, y_val);
    bpt::malloc(device, d_loss_d_output_matrix);
    bpt::malloc(device, d_input_matrix);

    bpt::reset_optimizer_state(device, network, optimizer);
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 2);
    bpt::init_weights(device, network, rng);

    constexpr TI NUM_BATCHES = DATASET_SIZE_TRAIN / BATCH_SIZE;
    for(int epoch_i=0; epoch_i < NUM_EPOCHS; epoch_i++){
        T epoch_loss = 0;
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        for (int batch_i=0; batch_i < NUM_BATCHES; batch_i++){
            T loss = 0;
            bpt::zero_gradient(device, network);
            for (int sample_i=0; sample_i < BATCH_SIZE; sample_i++){
                auto input = bpt::row(device, x_train, batch_i * BATCH_SIZE + sample_i);
                bpt::forward(device, network, input);
                T d_loss_d_output[OUTPUT_DIM];
                d_loss_d_output_matrix._data = d_loss_d_output;
//                bpt::nn::loss_functions::d_mse_d_x(device, network.output_layer.output, output, d_loss_d_output_matrix, T(1)/((T)BATCH_SIZE));
//                loss += bpt::nn::loss_functions::mse(device, network.output_layer.output, output, T(1)/((T)BATCH_SIZE));

                T d_input[INPUT_DIM];
                d_input_matrix._data = d_input;
                bpt::backward(device, network, input, d_loss_d_output_matrix, d_input_matrix, buffers);
            }
            loss /= BATCH_SIZE;
            epoch_loss += loss;

            bpt::update(device, network, optimizer);
            if(batch_i % 1000 == 0){
                std::cout << "epoch_i " << epoch_i << " batch_i " << batch_i << " loss: " << loss << std::endl;
            }
        }
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> epoch_duration = epoch_end_time - epoch_start_time;

        epoch_loss /= NUM_BATCHES;

        T val_loss = 0;
        for (int sample_i=0; sample_i < DATASET_SIZE_VAL; sample_i++){
            auto input = bpt::row(device, x_val, sample_i);
            bpt::MatrixDynamic<bpt::matrix::Specification<T, DEVICE::index_t, 1, OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> output_matrix;
            bpt::forward(device, network, input);
//            val_loss += bpt::nn::loss_functions::mse(device, network.output_layer.output, output_matrix, T(1)/BATCH_SIZE);
        }
        val_loss /= DATASET_SIZE_VAL;
    }
    return 0;
}
