#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn_models/models.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/containers/persist.h>

namespace bpt = backprop_tools;

#include <random>
#include <chrono>
#include <highfive/H5File.hpp>

using T = float;
using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

constexpr TI BATCH_SIZE = 32;
constexpr TI NUM_EPOCHS = 1;
constexpr TI INPUT_DIM = 28 * 28;
constexpr TI OUTPUT_DIM = 10;
constexpr TI NUM_LAYERS = 3;
constexpr TI HIDDEN_DIM = 50;
constexpr TI DATASET_SIZE_TRAIN = 60000;
constexpr TI DATASET_SIZE_VAL = 10000;
constexpr TI VALIDATION_LIMIT = 50;


using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTF<T, TI>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;

namespace mnist_model{ // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
    using namespace backprop_tools::nn_models::sequential::interface;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, INPUT_DIM, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
}

using NETWORK_TYPE = mnist_model::MODEL;

int main(){
    std::string dataset_path = "examples/docker/00_basic_mnist/mnist.hdf5";
    const char* dataset_path_env = std::getenv("BACKPROP_TOOLS_NN_MNIST_DATA_FILE");
    if (dataset_path_env != NULL){
        dataset_path = std::string(dataset_path_env);
    }

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    OPTIMIZER optimizer;
    device.logger = &logger;
    NETWORK_TYPE network;
    typename NETWORK_TYPE::DoubleBuffer<1> buffers;

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, DATASET_SIZE_TRAIN, INPUT_DIM>> x_train;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, DATASET_SIZE_VAL, INPUT_DIM>> x_val;
    bpt::MatrixDynamic<bpt::matrix::Specification<TI, TI, DATASET_SIZE_TRAIN, 1>> y_train;
    bpt::MatrixDynamic<bpt::matrix::Specification<TI, TI, DATASET_SIZE_VAL, 1>> y_val;

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



    {
        auto data_file = HighFive::File(dataset_path, HighFive::File::ReadOnly);
        bpt::load(device, x_train, data_file.getGroup("train"), "inputs");
        bpt::load(device, y_train, data_file.getGroup("train"), "labels");
        bpt::load(device, x_val, data_file.getGroup("test"), "inputs");
        bpt::load(device, y_val, data_file.getGroup("test"), "labels");
    }


    bpt::reset_optimizer_state(device, optimizer, network);
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
                auto output = bpt::row(device, y_train, batch_i * BATCH_SIZE + sample_i);
                auto prediction = bpt::row(device, bpt::output(network), 0);
                bpt::forward(device, network, input);
                bpt::nn::loss_functions::categorical_cross_entropy::gradient(device, prediction, output, d_loss_d_output_matrix, T(1)/((T)BATCH_SIZE));
                loss += bpt::nn::loss_functions::categorical_cross_entropy::evaluate(device, prediction, output, T(1)/((T)BATCH_SIZE));

                T d_input[INPUT_DIM];
                d_input_matrix._data = d_input;
                bpt::backward_full(device, network, input, d_loss_d_output_matrix, d_input_matrix, buffers);
            }
            loss /= BATCH_SIZE;
            epoch_loss += loss;

            bpt::step(device, optimizer, network);
            if(batch_i % 1000 == 0){
                std::cout << "epoch_i " << epoch_i << " batch_i " << batch_i << " loss: " << loss << std::endl;
            }
        }
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> epoch_duration = epoch_end_time - epoch_start_time;

        epoch_loss /= NUM_BATCHES;

        T val_loss = 0;
        T accuracy = 0;
        for (int sample_i=0; sample_i < VALIDATION_LIMIT; sample_i++){
            auto input = bpt::row(device, x_val, sample_i);
            auto output = bpt::row(device, y_val, sample_i);

            bpt::forward(device, network, input);
            val_loss += bpt::nn::loss_functions::categorical_cross_entropy::evaluate(device, bpt::output(network), output, T(1)/BATCH_SIZE);
            TI predicted_label = bpt::argmax_row(device, bpt::output(network));
            for(TI row_i = 0; row_i < 28; row_i++){
                for(TI col_i = 0; col_i < 28; col_i++){
                    T val = bpt::get(input, 0, row_i * 28 + col_i);
                    std::cout << (val > 0.5 ? (std::string(" ") + std::to_string(predicted_label)) : std::string("  "));
                }
                std::cout << std::endl;
            }
            accuracy += predicted_label == bpt::get(output, 0, 0);
        }
        val_loss /= DATASET_SIZE_VAL;
        accuracy /= VALIDATION_LIMIT;
        bpt::logging::text(device, device.logger, "Validation accuracy: ", accuracy * 100, "%", "");
    }
    return 0;
}
