#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include "../../include/nn_models/three_layer_fc.h"
#include "../../include/nn/nn.h"
#include "../utils/utils.h"
#include <highfive/H5File.hpp>
#include <sstream>
#define DTYPE float

using namespace layer_in_c;
template<typename T>
class AdamParameters{
public:
    static constexpr T ALPHA   = 0.001;
    static constexpr T BETA_1  = 0.9;
    static constexpr T BETA_2  = 0.999;
    static constexpr T EPSILON = 1e-7;

};

#define INPUT_DIM 17
#define LAYER_1_DIM 50
#define LAYER_1_FN nn::activation_functions::RELU
#define LAYER_2_DIM 50
#define LAYER_2_FN nn::activation_functions::RELU
#define OUTPUT_DIM 13
#define OUTPUT_FN nn::activation_functions::LINEAR
#define SKIP_TESTS
#define SKIP_BACKPROP_TESTS
#define SKIP_ADAM_TESTS
#define SKIP_OVERFITTING_TESTS
//#define SKIP_TRAINING_TESTS

const std::string DATA_FILE_PATH = "../model_learning/data.hdf5";

constexpr uint32_t N_WEIGHTS = ((INPUT_DIM + 1) * LAYER_1_DIM + (LAYER_1_DIM + 1) * LAYER_2_DIM + (LAYER_2_DIM + 1) * OUTPUT_DIM);


typedef nn_models::ThreeLayerNeuralNetworkTraining<DTYPE, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_FN, AdamParameters<DTYPE>> NetworkType_1;

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("layer_1/weight").read(weights);
    acc += abs_diff_matrix<T, LAYER_1_DIM, INPUT_DIM>(network.layer_1.weights, weights);
    return acc;
}



template <typename NETWORK_TYPE>
class NeuralNetworkTest : public ::testing::Test {
protected:
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
        data_file.getDataSet(model_name + "/init/layer_1/weight").read(layer_1_weights);
        data_file.getDataSet(model_name + "/init/layer_1/bias").read(layer_1_biases);
        data_file.getDataSet(model_name + "/init/layer_2/weight").read(layer_2_weights);
        data_file.getDataSet(model_name + "/init/layer_2/bias").read(layer_2_biases);
        data_file.getDataSet(model_name + "/init/output_layer/weight").read(output_layer_weights);
        data_file.getDataSet(model_name + "/init/output_layer/bias").read(output_layer_biases);
        this->reset();
    }

    void reset(){
        assign<DTYPE, LAYER_1_DIM, INPUT_DIM>(network.layer_1.weights     , layer_1_weights);
        memcpy(network.layer_1.biases, layer_1_biases.data(), sizeof(DTYPE) * LAYER_1_DIM);
        assign<DTYPE, LAYER_2_DIM, LAYER_1_DIM>(network.layer_2.weights     , layer_2_weights);
        memcpy(network.layer_2.biases, layer_2_biases.data(), sizeof(DTYPE) * LAYER_2_DIM);
        assign<DTYPE, OUTPUT_DIM, LAYER_2_DIM>(network.output_layer.weights, output_layer_weights);
        memcpy(network.output_layer.biases, output_layer_biases.data(), sizeof(DTYPE) * OUTPUT_DIM);
    }

    NETWORK_TYPE network;
    std::vector<std::vector<DTYPE>> X_train;
    std::vector<std::vector<DTYPE>> Y_train;
    std::vector<std::vector<DTYPE>> X_val;
    std::vector<std::vector<DTYPE>> Y_val;
    std::vector<DTYPE> X_mean;
    std::vector<DTYPE> X_std;
    std::vector<DTYPE> Y_mean;
    std::vector<DTYPE> Y_std;
    std::vector<std::vector<DTYPE>> layer_1_weights;
    std::vector<DTYPE> layer_1_biases;
    std::vector<std::vector<DTYPE>> layer_2_weights;
    std::vector<DTYPE> layer_2_biases;
    std::vector<std::vector<DTYPE>> output_layer_weights;
    std::vector<DTYPE> output_layer_biases;
};

class NeuralNetworkTestBackwardPass : public NeuralNetworkTest<NetworkType_1> {
public:
    NeuralNetworkTestBackwardPass() : NeuralNetworkTest(){
        model_name = "model_1";
    }
protected:
    void SetUp() override {
        NeuralNetworkTest::SetUp();
        auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
        data_file.getDataSet("model_1/gradients/0/layer_1/weight").read(batch_0_layer_1_weights_grad);
        data_file.getDataSet("model_1/gradients/0/layer_1/bias").read(batch_0_layer_1_biases_grad);
        data_file.getDataSet("model_1/gradients/0/layer_2/weight").read(batch_0_layer_2_weights_grad);
        data_file.getDataSet("model_1/gradients/0/layer_2/bias").read(batch_0_layer_2_biases_grad);
        data_file.getDataSet("model_1/gradients/0/output_layer/weight").read(batch_0_output_layer_weights_grad);
        data_file.getDataSet("model_1/gradients/0/output_layer/bias").read(batch_0_output_layer_biases_grad);
        this->reset();
        DTYPE input[INPUT_DIM];
        DTYPE output[OUTPUT_DIM];
        standardise<DTYPE, INPUT_DIM>(X_train[0].data(), X_mean.data(), X_std.data(), input);
        standardise<DTYPE, OUTPUT_DIM>(Y_train[0].data(), Y_mean.data(), Y_std.data(), output);
        forward(network, input);
        DTYPE d_loss_d_output[OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
        DTYPE d_input[INPUT_DIM];
        zero_gradient(network);
        backward(network, input, d_loss_d_output, d_input);
    }
    std::vector<std::vector<DTYPE>> batch_0_layer_1_weights_grad;
    std::vector<DTYPE> batch_0_layer_1_biases_grad;
    std::vector<std::vector<DTYPE>> batch_0_layer_2_weights_grad;
    std::vector<DTYPE> batch_0_layer_2_biases_grad;
    std::vector<std::vector<DTYPE>> batch_0_output_layer_weights_grad;
    std::vector<DTYPE> batch_0_output_layer_biases_grad;
};

constexpr DTYPE BACKWARD_PASS_GRADIENT_TOLERANCE (1e-8);
#ifndef SKIP_BACKPROP_TESTS
#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestBackwardPass, layer_1_weights) {
    DTYPE out = abs_diff_matrix<
            DTYPE, LAYER_1_DIM, INPUT_DIM
    >(
            network.layer_1.d_weights,
            batch_0_layer_1_weights_grad
    );
    std::cout << "layer_1_weights diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_1_DIM * INPUT_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestBackwardPass, layer_1_biases) {
    DTYPE out = abs_diff<
            DTYPE, LAYER_1_DIM
    >(
            network.layer_1.d_biases,
            batch_0_layer_1_biases_grad.data()
    );
    std::cout << "layer_1_biases diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_1_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestBackwardPass, layer_2_weights) {
    DTYPE out = abs_diff_matrix<
            DTYPE, LAYER_2_DIM, LAYER_1_DIM
    >(
            network.layer_2.d_weights,
            batch_0_layer_2_weights_grad
    );
    std::cout << "layer_2_weights diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_2_DIM * LAYER_1_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestBackwardPass, layer_2_biases) {
    DTYPE out = abs_diff<
            DTYPE, LAYER_2_DIM
    >(
            network.layer_2.d_biases,
            batch_0_layer_2_biases_grad.data()
    );
    std::cout << "layer_2_biases diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_2_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestBackwardPass, output_layer_weights) {
    DTYPE out = abs_diff_matrix<
            DTYPE, OUTPUT_DIM, LAYER_2_DIM
    >(
            network.output_layer.d_weights,
            batch_0_output_layer_weights_grad
    );
    std::cout << "output_layer_weights diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * OUTPUT_DIM * LAYER_2_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestBackwardPass, output_layer_biases) {
    DTYPE out = abs_diff<
            DTYPE, OUTPUT_DIM
    >(
            network.output_layer.d_biases,
            batch_0_output_layer_biases_grad.data()
    );
    std::cout << "output_layer_biases diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * OUTPUT_DIM);
}
#endif
#endif


#ifndef SKIP_ADAM_TESTS
typedef NeuralNetworkTest<NetworkType_1> NeuralNetworkTestAdamUpdate;
#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestAdamUpdate, AdamUpdate) {
    this->reset();

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    std::vector<std::vector<DTYPE>> batch_0_layer_1_weights;
    std::vector<DTYPE> batch_0_layer_1_biases;
    std::vector<std::vector<DTYPE>> batch_0_layer_2_weights;
    std::vector<DTYPE> batch_0_layer_2_biases;
    std::vector<std::vector<DTYPE>> batch_0_output_layer_weights;
    std::vector<DTYPE> batch_0_output_layer_biases;
    data_file.getDataSet("model_1/weights/0/layer_1/weight").read(batch_0_layer_1_weights);
    data_file.getDataSet("model_1/weights/0/layer_1/bias").read(batch_0_layer_1_biases);
    data_file.getDataSet("model_1/weights/0/layer_2/weight").read(batch_0_layer_2_weights);
    data_file.getDataSet("model_1/weights/0/layer_2/bias").read(batch_0_layer_2_biases);
    data_file.getDataSet("model_1/weights/0/output_layer/weight").read(batch_0_output_layer_weights);
    data_file.getDataSet("model_1/weights/0/output_layer/bias").read(batch_0_output_layer_biases);
    DTYPE input[INPUT_DIM];
    DTYPE output[OUTPUT_DIM];
    standardise<DTYPE, INPUT_DIM>(&X_train[0][0], &X_mean[0], &X_std[0], input);
    standardise<DTYPE, OUTPUT_DIM>(&Y_train[0][0], &Y_mean[0], &Y_std[0], output);
    forward(network, input);
    DTYPE d_loss_d_output[OUTPUT_DIM];
    d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
    DTYPE d_input[INPUT_DIM];
    zero_gradient(network);
    backward(network, input, d_loss_d_output, d_input);
    reset_optimizer_state(network);
    update(network, 1, 1);

    DTYPE out = abs_diff_matrix<
            DTYPE,
            LAYER_1_DIM,
            INPUT_DIM
    >(
            network.layer_1.weights,
            batch_0_layer_1_weights
    );
    ASSERT_LT(out, 1.5e-7);
}
#endif
#endif

//#ifdef SKIP_TESTS
//TEST_F(NeuralNetworkTest, OverfitSample) {
//    this->reset();
//
//    DTYPE input[INPUT_DIM];
//    DTYPE output[OUTPUT_DIM];
//    standardise<DTYPE, INPUT_DIM>(X_train[1].data(), X_mean.data(), X_std.data(), input);
//    standardise<DTYPE, OUTPUT_DIM>(Y_train[1].data(), Y_mean.data(), Y_std.data(), output);
//    constexpr int n_iter = 1000;
//    DTYPE loss = 0;
//    reset_optimizer_state(network);
//    for (int batch_i = 0; batch_i < n_iter; batch_i++){
//        forward(network, input);
//        DTYPE d_loss_d_output[OUTPUT_DIM];
//        d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
//        loss = mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);
//        std::cout << "batch_i: " << batch_i << " loss: " << loss << std::endl;
//
//        zero_gradient(network);
//        DTYPE d_input[INPUT_DIM];
//        backward(network, input, d_loss_d_output, d_input);
//
//        update(network, batch_i + 1, 1);
//    }
//    ASSERT_LT(loss, 5e-10);
//
//
//}
//#endif

#ifndef SKIP_OVERFITTING_TESTS
class NeuralNetworkTestOverfitBatch : public NeuralNetworkTest<NetworkType_1> {
public:
    NeuralNetworkTestOverfitBatch() : NeuralNetworkTest(){
        model_name = "model_2";
    }
protected:

    void SetUp() override {
        NeuralNetworkTest::SetUp();
        this->reset();
    }
};
#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestOverfitBatch, OverfitBatch) {
    this->reset();

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    HighFive::Group g = data_file.getGroup("model_2/overfit_small_batch");

    constexpr int n_iter = 1000;
    constexpr int batch_size = 32;
    DTYPE loss = 0;
    reset_optimizer_state(network);
    {
        DTYPE diff = abs_diff_network<DTYPE>(network, data_file.getGroup(model_name+"/init"));
        std::cout << "initial diff: " << diff << std::endl;
        ASSERT_EQ(diff, 0);
    }
    for (int batch_i=0; batch_i < n_iter; batch_i++){
        uint32_t batch_i_real = 0;
        loss = 0;
        zero_gradient(network);
        for (int sample_i=0; sample_i < batch_size; sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_train[batch_i_real * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i_real * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
            forward(network, input);
            DTYPE d_loss_d_output[OUTPUT_DIM];
            d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
            loss += mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);

            DTYPE d_input[INPUT_DIM];
            backward(network, input, d_loss_d_output, d_input);
        }
        loss /= batch_size;

        std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

        update(network, batch_i + 1, batch_size);
//        constexpr int comp_batch = 100;
//        if(batch_i == comp_batch){
        std::stringstream ss;
        ss << "model_2/overfit_small_batch/" << batch_i;
        DTYPE diff = abs_diff_network<DTYPE>(network, data_file.getGroup(ss.str()));
        std::cout << "batch_i: " << batch_i << " diff: " << diff << std::endl;
        if(batch_i == 10){
            ASSERT_LT(diff, 2.5e-7 * N_WEIGHTS);
        } else {
            if(batch_i == 100){
                ASSERT_LT(diff, 1.0e-4 * N_WEIGHTS);
            }
        }
    }
    ASSERT_LT(loss, 1e-10);
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestOverfitBatch, OverfitBatches) {
    std::vector<DTYPE> losses;
    constexpr int n_batches = 10;
    for(int batch_i_real=0; batch_i_real < n_batches; batch_i_real++){
        this->reset();

        constexpr int n_iter = 1000;
        constexpr int batch_size = 32;
        DTYPE loss = 0;
        reset_optimizer_state(network);
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            loss = 0;
            zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                DTYPE input[INPUT_DIM];
                DTYPE output[OUTPUT_DIM];
                standardise<DTYPE,  INPUT_DIM>(X_train[batch_i_real * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i_real * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                forward(network, input);
                DTYPE d_loss_d_output[OUTPUT_DIM];
                d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
                loss += mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);

                DTYPE d_input[INPUT_DIM];
                backward(network, input, d_loss_d_output, d_input);
            }
            loss /= batch_size;

//            std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

            update(network, batch_i + 1, batch_size);
        }
        std::cout << "batch_i_real " << batch_i_real << " loss: " << loss << std::endl;
        losses.push_back(loss);
    }
    DTYPE mean_loss = accumulate(losses.begin(), losses.end(), (DTYPE)0) / n_batches;
    DTYPE std_loss = 0;
    for(auto loss : losses){
        std_loss += (loss - mean_loss) * (loss - mean_loss);
    }
    std_loss = sqrt(std_loss / n_batches);
    DTYPE min_loss = *min_element(losses.begin(), losses.end());
    DTYPE max_loss = *max_element(losses.begin(), losses.end());
    std::sort(losses.begin(), losses.end());
    DTYPE perc1_loss = losses[(int)(n_batches * 0.01)];
    DTYPE perc5_loss = losses[(int)(n_batches * 0.05)];
    DTYPE perc95_loss = losses[(int)(n_batches * 0.95)];
    DTYPE perc99_loss = losses[(int)(n_batches * 0.99)];

    constexpr DTYPE mean_loss_target   = 3.307228189225464e-12;
    constexpr DTYPE std_loss_target    = 1.363137554222238e-11;
    constexpr DTYPE min_loss_target    = 1.540076829357074e-14;
    constexpr DTYPE max_loss_target    = 1.1320114290391814e-10;
    constexpr DTYPE perc1_loss_target  = 2.662835865368629e-14;
    constexpr DTYPE perc5_loss_target  = 3.350572399960167e-14;
    constexpr DTYPE prec95_loss_target = 1.266289100486372e-11;
    constexpr DTYPE prec99_loss_target = 7.271057700375415e-11;
    constexpr DTYPE assertion_approx_factor = 2;
    std::cout << "mean_loss "   << mean_loss   << " mean_loss_target: " << mean_loss_target << std::endl;
    std::cout << "std_loss "    << std_loss    << " std_loss_target: " << std_loss_target << std::endl;
    std::cout << "min_loss "    << min_loss    << " min_loss_target: " << min_loss_target << std::endl;
    std::cout << "max_loss "    << max_loss    << " max_loss_target: " << max_loss_target << std::endl;
    std::cout << "perc1_loss "  << perc1_loss  << " perc1_loss_target: " << perc1_loss_target << std::endl;
    std::cout << "perc5_loss "  << perc5_loss  << " perc5_loss_target: " << perc5_loss_target << std::endl;
    std::cout << "perc95_loss " << perc95_loss << " prec95_loss_target: " << prec95_loss_target << std::endl;
    std::cout << "perc99_loss " << perc99_loss << " prec99_loss_target: " << prec99_loss_target << std::endl;

    ASSERT_LT(mean_loss_target, 3.307228189225464e-12 * assertion_approx_factor);
    ASSERT_LT(std_loss_target, 1.363137554222238e-11 * assertion_approx_factor);
    ASSERT_LT(min_loss_target, 1.540076829357074e-14 * assertion_approx_factor);
    ASSERT_LT(max_loss_target, 1.1320114290391814e-10 * assertion_approx_factor);
    ASSERT_LT(perc1_loss_target, 2.662835865368629e-14 * assertion_approx_factor);
    ASSERT_LT(perc5_loss_target, 3.350572399960167e-14 * assertion_approx_factor);
    ASSERT_LT(prec95_loss_target, 1.266289100486372e-11 * assertion_approx_factor);
    ASSERT_LT(prec99_loss_target, 7.271057700375415e-11 * assertion_approx_factor);
}
#endif
#endif

typedef nn_models::ThreeLayerNeuralNetworkTraining<DTYPE, INPUT_DIM, LAYER_1_DIM, nn::activation_functions::GELU_SQUARE, LAYER_2_DIM, nn::activation_functions::GELU_SQUARE, OUTPUT_DIM, OUTPUT_FN, AdamParameters<DTYPE>> NetworkType_3;
class NeuralNetworkTestTrainModel : public NeuralNetworkTest<NetworkType_3> {
public:
    NeuralNetworkTestTrainModel() : NeuralNetworkTest<NetworkType_3>(){
        model_name = "model_3";
    }
protected:

    void SetUp() override {
        NeuralNetworkTest::SetUp();
        this->reset();
    }
};
#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestTrainModel, TrainModel) {
    std::vector<DTYPE> losses;
    std::vector<DTYPE> val_losses;
    constexpr int n_epochs = 3;
    this->reset();
    reset_optimizer_state(network);
    constexpr int batch_size = 32;
    int n_iter = X_train.size() / batch_size;

    for(int epoch_i=0; epoch_i < n_epochs; epoch_i++){
        DTYPE epoch_loss = 0;
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            DTYPE loss = 0;
            zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                DTYPE input[INPUT_DIM];
                DTYPE output[OUTPUT_DIM];
                standardise<DTYPE,  INPUT_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                forward(network, input);
                DTYPE d_loss_d_output[OUTPUT_DIM];
                d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
                loss += mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);

                DTYPE d_input[INPUT_DIM];
                backward(network, input, d_loss_d_output, d_input);
            }
            loss /= batch_size;
            epoch_loss += loss;

//            std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

            update(network, batch_i + 1, batch_size);
            std::cout << "epoch_i " << epoch_i << " batch_i " << batch_i << " loss: " << loss << std::endl;
        }
        epoch_loss /= n_iter;
        losses.push_back(epoch_loss);

        DTYPE val_loss = 0;
        for (int sample_i=0; sample_i < X_val.size(); sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_val[sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_val[sample_i].data(), Y_mean.data(), Y_std.data(), output);
            forward(network, input);
            val_loss += mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);
        }
        val_loss /= X_val.size();
        val_losses.push_back(val_loss);
    }


    for (int i=0; i < losses.size(); i++){
        std::cout << "epoch_i " << i << " loss: train:" << losses[i] << " val: " << val_losses[i] << std::endl;
    }
    // loss
    // array([0.05651794, 0.02564381, 0.02268129, 0.02161846, 0.02045725,
    //       0.01928116, 0.01860152, 0.01789362, 0.01730141, 0.01681832],
    //      dtype=float32)

    // val_loss
    // array([0.02865824, 0.02282167, 0.02195382, 0.02137529, 0.02023922,
    //       0.0191351 , 0.01818279, 0.01745798, 0.01671058, 0.01628938],
    //      dtype=float32)

    ASSERT_LT(losses[0], 0.06);
    ASSERT_LT(losses[1], 0.03);
    ASSERT_LT(losses[2], 0.025);
//    ASSERT_LT(losses[9], 0.02);
    ASSERT_LT(val_losses[0], 0.04);
    ASSERT_LT(val_losses[1], 0.03);
    ASSERT_LT(val_losses[2], 0.025);
//    ASSERT_LT(val_losses[9], 0.02);

// GELU PyTorch [0.00456139 0.00306715 0.00215886]
}
#endif

#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestTrainModel, ModelInitTrain) {
    std::vector<DTYPE> losses;
    std::vector<DTYPE> val_losses;
    constexpr int n_epochs = 3;
//    this->reset();
    reset_optimizer_state(network);
    std::mt19937 rng(0);
    init_weights(network, rng);

    constexpr int batch_size = 32;
    int n_iter = X_train.size() / batch_size;

    for(int epoch_i=0; epoch_i < n_epochs; epoch_i++){
        DTYPE epoch_loss = 0;
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            DTYPE loss = 0;
            zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                DTYPE input[INPUT_DIM];
                DTYPE output[OUTPUT_DIM];
                standardise<DTYPE,  INPUT_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                forward(network, input);
                DTYPE d_loss_d_output[OUTPUT_DIM];
                nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
                loss += nn::loss_functions::mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);

                DTYPE d_input[INPUT_DIM];
                backward(network, input, d_loss_d_output, d_input);
            }
            loss /= batch_size;
            epoch_loss += loss;

//            std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

            update(network, batch_i + 1, batch_size);
            std::cout << "epoch_i " << epoch_i << " batch_i " << batch_i << " loss: " << loss << std::endl;
        }
        epoch_loss /= n_iter;
        losses.push_back(epoch_loss);

        DTYPE val_loss = 0;
        for (int sample_i=0; sample_i < X_val.size(); sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_val[sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_val[sample_i].data(), Y_mean.data(), Y_std.data(), output);
            forward(network, input);
            val_loss += nn::loss_functions::mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);
        }
        val_loss /= X_val.size();
        val_losses.push_back(val_loss);
    }


    for (int i=0; i < losses.size(); i++){
        std::cout << "epoch_i " << i << " loss: train:" << losses[i] << " val: " << val_losses[i] << std::endl;
    }
    // loss
    // array([0.05651794, 0.02564381, 0.02268129, 0.02161846, 0.02045725,
    //       0.01928116, 0.01860152, 0.01789362, 0.01730141, 0.01681832],
    //      dtype=float32)

    // val_loss
    // array([0.02865824, 0.02282167, 0.02195382, 0.02137529, 0.02023922,
    //       0.0191351 , 0.01818279, 0.01745798, 0.01671058, 0.01628938],
    //      dtype=float32)

    ASSERT_LT(losses[0], 0.06);
    ASSERT_LT(losses[1], 0.03);
    ASSERT_LT(losses[2], 0.025);
//    ASSERT_LT(losses[9], 0.02);
    ASSERT_LT(val_losses[0], 0.04);
    ASSERT_LT(val_losses[1], 0.03);
    ASSERT_LT(val_losses[2], 0.025);
//    ASSERT_LT(val_losses[9], 0.02);

// GELU PyTorch [0.00456139 0.00306715 0.00215886]
}
#endif
