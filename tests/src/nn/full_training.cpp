#include <gtest/gtest.h>

#include <random>
#include <highfive/H5File.hpp>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/utils/rng_std.h>

#include "../utils/utils.h"

namespace lic = layer_in_c;

typedef float T;
constexpr size_t INPUT_DIM = 17;
constexpr size_t LAYER_1_DIM = 50;
constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN =  lic::nn::activation_functions::TANH;
constexpr size_t LAYER_2_DIM = 50;
constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = lic::nn::activation_functions::TANH;
constexpr size_t OUTPUT_DIM = 13;
constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_FN = lic::nn::activation_functions::IDENTITY;

typedef lic::nn_models::three_layer_fc::StructureSpecification<T, INPUT_DIM, 50, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_FN> NETWORK_STRUCTURE_SPEC;
typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, NETWORK_STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParametersTF<T>> NETWORK_SPEC;
typedef lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, NETWORK_SPEC> NetworkType;

std::vector<std::vector<T>> X_train;
std::vector<std::vector<T>> Y_train;
std::vector<std::vector<T>> X_val;
std::vector<std::vector<T>> Y_val;
std::vector<T> X_mean;
std::vector<T> X_std;
std::vector<T> Y_mean;
std::vector<T> Y_std;

TEST(NEURAL_NETWORK_FULL_TRAINING, FULL_TRAINING) {
    // loading data
    std::string DATA_FILE_PATH = "../model-learning/data.hdf5";
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

    NetworkType network;
    std::vector<T> losses;
    std::vector<T> val_losses;
    constexpr int n_epochs = 3;
    //    this->reset();
    lic::reset_optimizer_state(network);
    std::mt19937 rng(2);
    lic::init_weights<NETWORK_SPEC, lic::utils::random::stdlib::uniform<T, typeof(rng)>, typeof(rng)>(network, rng);

    constexpr int batch_size = 32;
    int n_iter = X_train.size() / batch_size;

    for(int epoch_i=0; epoch_i < n_epochs; epoch_i++){
        T epoch_loss = 0;
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            T loss = 0;
            lic::zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                T input[INPUT_DIM];
                T output[OUTPUT_DIM];
                standardise<T,  INPUT_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<T, OUTPUT_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                lic::forward(network, input);
                T d_loss_d_output[OUTPUT_DIM];
                lic::nn::loss_functions::d_mse_d_x<T, OUTPUT_DIM, batch_size>(network.output_layer.output, output, d_loss_d_output);
                loss += lic::nn::loss_functions::mse<T, OUTPUT_DIM, batch_size>(network.output_layer.output, output);

                T d_input[INPUT_DIM];
                lic::backward(network, input, d_loss_d_output, d_input);
            }
            loss /= batch_size;
            epoch_loss += loss;

            //            std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

            lic::update(network);
            std::cout << "epoch_i " << epoch_i << " batch_i " << batch_i << " loss: " << loss << std::endl;
        }
        epoch_loss /= n_iter;
        losses.push_back(epoch_loss);

        T val_loss = 0;
        for (int sample_i=0; sample_i < X_val.size(); sample_i++){
        T input[INPUT_DIM];
        T output[OUTPUT_DIM];
        standardise<T,  INPUT_DIM>(X_val[sample_i].data(), X_mean.data(), X_std.data(), input);
        standardise<T, OUTPUT_DIM>(Y_val[sample_i].data(), Y_mean.data(), Y_std.data(), output);
        lic::forward(network, input);
        val_loss += lic::nn::loss_functions::mse<T, OUTPUT_DIM, batch_size>(network.output_layer.output, output);
        }
        val_loss /= X_val.size();
        val_losses.push_back(val_loss);
    }


    for (int i=0; i < losses.size(); i++){
    std::cout << "epoch_i " << i << " loss: train:" << losses[i] << " val: " << val_losses[i] << std::endl;
    }

    ASSERT_LT(losses[0], 0.005);
    ASSERT_LT(losses[1], 0.002);
    ASSERT_LT(losses[2], 0.002);
    ASSERT_LT(val_losses[0], 0.002);
    ASSERT_LT(val_losses[1], 0.001);
    ASSERT_LT(val_losses[2], 0.001);

    // GELU PyTorch [0.00456139 0.00306715 0.00215886]


//    TANH
//    epoch_i 0 loss: train:0.00265808 val: 0.00041066
//    epoch_i 1 loss: train:0.000322035 val: 0.000304242
//    epoch_i 2 loss: train:0.000230863 val: 0.000214781
//    RELU
//    epoch_i 0 loss: train:0.00178393 val: 0.000915656
//    epoch_i 1 loss: train:0.000778314 val: 0.000739406
//    epoch_i 2 loss: train:0.000696363 val: 0.000696439
}
