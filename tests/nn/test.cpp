#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/nn/nn.h>
#include <layer_in_c/utils/rng_std.h>
#include "../utils/utils.h"
#include <sstream>
#include "default_network.h"
//#define SKIP_TESTS
//#define SKIP_BACKPROP_TESTS
//#define SKIP_ADAM_TESTS
//#define SKIP_OVERFITTING_TESTS
//#define SKIP_TRAINING_TESTS


constexpr uint32_t N_WEIGHTS = ((INPUT_DIM + 1) * LAYER_1_DIM + (LAYER_1_DIM + 1) * LAYER_2_DIM + (LAYER_2_DIM + 1) * OUTPUT_DIM);


typedef lic::nn_models::three_layer_fc::StructureSpecification<DTYPE, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_FN> NETWORK_STRUCTURE_SPEC;
typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, NETWORK_STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>> NETWORK_SPEC;
typedef lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, NETWORK_SPEC> NetworkType_1;

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("layer_1/weight").read(weights);
    acc += abs_diff_matrix<T, LAYER_1_DIM, INPUT_DIM>(network.layer_1.weights, weights);
    return acc;
}

template <typename NetworkType>
class NeuralNetworkTestLoadWeights : public NeuralNetworkTest {
public:
    NeuralNetworkTestLoadWeights() : NeuralNetworkTest(){
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
        lic::forward(network, input);
        DTYPE d_loss_d_output[OUTPUT_DIM];
        lic::nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM, 1>(network.output_layer.output, output, d_loss_d_output);
        DTYPE d_input[INPUT_DIM];
        lic::zero_gradient(network);
        lic::backward(network, input, d_loss_d_output, d_input);
    }
    void reset(){
        auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
        data_file.getDataSet(model_name + "/init/layer_1/weight").read(layer_1_weights);
        data_file.getDataSet(model_name + "/init/layer_1/bias").read(layer_1_biases);
        data_file.getDataSet(model_name + "/init/layer_2/weight").read(layer_2_weights);
        data_file.getDataSet(model_name + "/init/layer_2/bias").read(layer_2_biases);
        data_file.getDataSet(model_name + "/init/output_layer/weight").read(output_layer_weights);
        data_file.getDataSet(model_name + "/init/output_layer/bias").read(output_layer_biases);
        assign<DTYPE, LAYER_1_DIM, INPUT_DIM>(network.layer_1.weights     , layer_1_weights);
        memcpy(network.layer_1.biases, layer_1_biases.data(), sizeof(DTYPE) * LAYER_1_DIM);
        assign<DTYPE, LAYER_2_DIM, LAYER_1_DIM>(network.layer_2.weights     , layer_2_weights);
        memcpy(network.layer_2.biases, layer_2_biases.data(), sizeof(DTYPE) * LAYER_2_DIM);
        assign<DTYPE, OUTPUT_DIM, LAYER_2_DIM>(network.output_layer.weights, output_layer_weights);
        memcpy(network.output_layer.biases, output_layer_biases.data(), sizeof(DTYPE) * OUTPUT_DIM);
    }

    NetworkType network;
    std::vector<std::vector<DTYPE>> layer_1_weights;
    std::vector<DTYPE> layer_1_biases;
    std::vector<std::vector<DTYPE>> layer_2_weights;
    std::vector<DTYPE> layer_2_biases;
    std::vector<std::vector<DTYPE>> output_layer_weights;
    std::vector<DTYPE> output_layer_biases;
    std::vector<std::vector<DTYPE>> batch_0_layer_1_weights_grad;
    std::vector<DTYPE> batch_0_layer_1_biases_grad;
    std::vector<std::vector<DTYPE>> batch_0_layer_2_weights_grad;
    std::vector<DTYPE> batch_0_layer_2_biases_grad;
    std::vector<std::vector<DTYPE>> batch_0_output_layer_weights_grad;
    std::vector<DTYPE> batch_0_output_layer_biases_grad;
};

constexpr DTYPE BACKWARD_PASS_GRADIENT_TOLERANCE (1e-8);
#ifndef SKIP_BACKPROP_TESTS
typedef NeuralNetworkTestLoadWeights<NetworkType_1> NeuralNetworkTestBackwardPass;
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
typedef NeuralNetworkTestBackwardPass NeuralNetworkTestAdamUpdate;
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
    lic::forward(network, input);
    DTYPE d_loss_d_output[OUTPUT_DIM];
    lic::nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM, 1>(network.output_layer.output, output, d_loss_d_output);
    DTYPE d_input[INPUT_DIM];
    lic::zero_gradient(network);
    lic::backward(network, input, d_loss_d_output, d_input);
    lic::reset_optimizer_state(network);
    lic::update(network);

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
class NeuralNetworkTestOverfitBatch : public NeuralNetworkTestBackwardPass {
public:
    NeuralNetworkTestOverfitBatch() : NeuralNetworkTestBackwardPass(){
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
    lic::reset_optimizer_state(network);
    {
        DTYPE diff = abs_diff_network<DTYPE>(network, data_file.getGroup(model_name+"/init"));
        std::cout << "initial diff: " << diff << std::endl;
        ASSERT_EQ(diff, 0);
    }
    for (int batch_i=0; batch_i < n_iter; batch_i++){
        uint32_t batch_i_real = 0;
        loss = 0;
        lic::zero_gradient(network);
        for (int sample_i=0; sample_i < batch_size; sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_train[batch_i_real * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i_real * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
            lic::forward(network, input);
            DTYPE d_loss_d_output[OUTPUT_DIM];
            lic::nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output, d_loss_d_output);
            loss += lic::nn::loss_functions::mse<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output);

            DTYPE d_input[INPUT_DIM];
            lic::backward(network, input, d_loss_d_output, d_input);
        }
        loss /= batch_size;

        std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

        lic::update(network);
//        constexpr int comp_batch = 100;
//        if(batch_i == comp_batch){
        std::stringstream ss;
        ss << "model_2/overfit_small_batch/" << batch_i;
        DTYPE diff = abs_diff_network<DTYPE>(network, data_file.getGroup(ss.str()));
        std::cout << "batch_i: " << batch_i << " diff: " << diff << std::endl;
        if(batch_i == 10){
            ASSERT_LT(diff, 2.5e-7 * 3 * N_WEIGHTS);
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
        lic::reset_optimizer_state(network);
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            loss = 0;
            lic::zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                DTYPE input[INPUT_DIM];
                DTYPE output[OUTPUT_DIM];
                standardise<DTYPE,  INPUT_DIM>(X_train[batch_i_real * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i_real * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                lic::forward(network, input);
                DTYPE d_loss_d_output[OUTPUT_DIM];
                lic::nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output, d_loss_d_output);
                loss += lic::nn::loss_functions::mse<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output);

                DTYPE d_input[INPUT_DIM];
                lic::backward(network, input, d_loss_d_output, d_input);
            }
            loss /= batch_size;

//            std::cout << "batch_i " << batch_i << " loss: " << loss << std::endl;

            lic::update(network);
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

constexpr auto MODEL_TRAINING_ACTIVATION_FN = lic::nn::activation_functions::GELU_SQUARE;

typedef lic::nn_models::three_layer_fc::StructureSpecification<DTYPE, INPUT_DIM, LAYER_1_DIM, MODEL_TRAINING_ACTIVATION_FN, LAYER_2_DIM, MODEL_TRAINING_ACTIVATION_FN, OUTPUT_DIM, OUTPUT_FN> NETWORK_STRUCTURE_SPEC_3;
typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, NETWORK_STRUCTURE_SPEC_3, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>> NETWORK_SPEC_3;
typedef lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, NETWORK_SPEC_3> NetworkType_3;
//typedef nn_models::three_layer_fc::SGDSpecification<DTYPE, INPUT_DIM, LAYER_1_DIM, MODEL_TRAINING_ACTIVATION_FN, LAYER_2_DIM, MODEL_TRAINING_ACTIVATION_FN, OUTPUT_DIM, OUTPUT_FN, nn::layers::DefaultSGDParameters<DTYPE>> NETWORK_SPEC_3;
//typedef nn_models::three_layer_fc::NeuralNetworkSGD<NETWORK_SPEC_3> NetworkType_3;
class NeuralNetworkTestTrainModel : public NeuralNetworkTestLoadWeights<NetworkType_3> {
public:
    typedef NetworkType_3 NETWORK_TYPE;
    NeuralNetworkTestTrainModel() : NeuralNetworkTestLoadWeights<NetworkType_3>(){
        model_name = "model_3";
    }
protected:

    void SetUp() override {
        NeuralNetworkTest::SetUp();
        this->reset();
    }
};
#ifndef SKIP_TRAINING_TESTS
#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestTrainModel, TrainModel) {
    std::vector<DTYPE> losses;
    std::vector<DTYPE> val_losses;
    constexpr int n_epochs = 3;
    this->reset();
    lic::reset_optimizer_state(network);
    constexpr int batch_size = 32;
    int n_iter = X_train.size() / batch_size;

    for(int epoch_i=0; epoch_i < n_epochs; epoch_i++){
        DTYPE epoch_loss = 0;
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            DTYPE loss = 0;
            lic::zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                DTYPE input[INPUT_DIM];
                DTYPE output[OUTPUT_DIM];
                standardise<DTYPE,  INPUT_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                lic::forward(network, input);
                DTYPE d_loss_d_output[OUTPUT_DIM];
                lic::nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output, d_loss_d_output);
                loss += lic::nn::loss_functions::mse<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output);

                DTYPE d_input[INPUT_DIM];
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

        DTYPE val_loss = 0;
        for (int sample_i=0; sample_i < X_val.size(); sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_val[sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_val[sample_i].data(), Y_mean.data(), Y_std.data(), output);
            lic::forward(network, input);
            val_loss += lic::nn::loss_functions::mse<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output);
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

//#ifndef SKIP_TESTS
TEST_F(NeuralNetworkTestTrainModel, ModelInitTrain) {
//    assert((std::is_same_v<typeof(network), NETWORK_TYPE>));
    typedef lic::nn_models::three_layer_fc::StructureSpecification<DTYPE, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_FN> NETWORK_STRUCTURE_SPEC;
    typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, NETWORK_STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>> NETWORK_SPEC;
    typedef lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, NETWORK_SPEC> NetworkType;
    NetworkType network;
    std::vector<DTYPE> losses;
    std::vector<DTYPE> val_losses;
    constexpr int n_epochs = 3;
//    this->reset();
    lic::reset_optimizer_state(network);
    std::mt19937 rng(2);
    lic::init_weights<NETWORK_SPEC, lic::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(network, rng);

    constexpr int batch_size = 32;
    int n_iter = X_train.size() / batch_size;

    for(int epoch_i=0; epoch_i < n_epochs; epoch_i++){
        DTYPE epoch_loss = 0;
        for (int batch_i=0; batch_i < n_iter; batch_i++){
            DTYPE loss = 0;
            lic::zero_gradient(network);
            for (int sample_i=0; sample_i < batch_size; sample_i++){
                DTYPE input[INPUT_DIM];
                DTYPE output[OUTPUT_DIM];
                standardise<DTYPE,  INPUT_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
                standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
                lic::forward(network, input);
                DTYPE d_loss_d_output[OUTPUT_DIM];
                lic::nn::loss_functions::d_mse_d_x<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output, d_loss_d_output);
                loss += lic::nn::loss_functions::mse<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output);

                DTYPE d_input[INPUT_DIM];
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

        DTYPE val_loss = 0;
        for (int sample_i=0; sample_i < X_val.size(); sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_val[sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_val[sample_i].data(), Y_mean.data(), Y_std.data(), output);
            lic::forward(network, input);
            val_loss += lic::nn::loss_functions::mse<DTYPE, OUTPUT_DIM, batch_size>(network.output_layer.output, output);
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
//#endif
#endif
