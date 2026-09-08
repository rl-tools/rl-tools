#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <gtest/gtest.h>

namespace {
    using namespace rl_tools;

    template<nn::activation_functions::ActivationFunction ACTIVATION, nn::layers::conv2d::Normalization NORMALIZATION, typename MODE>
    void compare_activation(const Mode<MODE>& mode){
        using CPU = devices::DEVICE_FACTORY<>;
        using GPU = devices::DefaultCUDA;
        using TI = GPU::index_t;
        using POLICY = numeric_types::Policy<float>;
        using CONFIG = nn::layers::conv2d::Configuration<POLICY, TI, 8, 3, 3, 2, 2, 1, 1, ACTIVATION, NORMALIZATION>;
        using INPUT_SHAPE = tensor::Shape<TI, 2, 8, 9, 3>;
        using CAPABILITY = nn::capability::Gradient<nn::parameters::Gradient>;
        using LAYER = nn::layers::conv2d::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
        using INPUT = Tensor<tensor::Specification<float, TI, INPUT_SHAPE>>;
        using OUTPUT = Tensor<tensor::Specification<float, TI, typename LAYER::OUTPUT_SHAPE>>;
        CPU cpu;
        GPU gpu;
        init(gpu);
        typename CPU::SPEC::RANDOM::template ENGINE<> rng;
        malloc(cpu, rng);
        init(cpu, rng, 7);
        LAYER reference, actual, copied;
        typename LAYER::template Buffer<true> cpu_buffer, gpu_buffer;
        INPUT input, gpu_input, d_input, gpu_d_input, copied_d_input;
        OUTPUT output, gpu_output, copied_output, d_output, gpu_d_output, copied_d_output;
        malloc(cpu, reference);
        malloc(gpu, actual);
        malloc(cpu, copied);
        malloc(cpu, cpu_buffer);
        malloc(gpu, gpu_buffer);
        malloc(cpu, input);
        malloc(gpu, gpu_input);
        malloc(cpu, d_input);
        malloc(gpu, gpu_d_input);
        malloc(cpu, copied_d_input);
        malloc(cpu, output);
        malloc(gpu, gpu_output);
        malloc(cpu, copied_output);
        malloc(cpu, d_output);
        malloc(gpu, gpu_d_output);
        malloc(cpu, copied_d_output);
        init_weights(cpu, reference, rng);
        if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            for(TI channel = 0; channel < CONFIG::OUTPUT_CHANNELS; channel++){
                set(cpu, reference.norm.gamma.parameters, 0.7f + 0.1f * channel, channel);
                set(cpu, reference.norm.beta.parameters, -0.3f + 0.1f * channel, channel);
                set(cpu, reference.norm.running_mean, 0.05f * channel, channel);
                set(cpu, reference.norm.running_var, 1.0f + 0.1f * channel, channel);
            }
        }
        randn(cpu, input, rng);
        randn(cpu, d_output, rng);
        copy(cpu, gpu, reference, actual);
        copy(cpu, gpu, input, gpu_input);
        copy(cpu, gpu, d_output, gpu_d_output);
        auto close = [&](const auto& expected, const auto& observed){
            EXPECT_LT(abs_diff(cpu, expected, observed) / (1.0f + abs_sum(cpu, expected)), 2e-4f);
        };
        evaluate(cpu, reference, input, output, cpu_buffer, rng);
        evaluate(gpu, actual, gpu_input, gpu_output, gpu_buffer, rng);
        copy(gpu, cpu, gpu_output, copied_output);
        close(output, copied_output);
        zero_gradient(cpu, reference);
        zero_gradient(gpu, actual);
        for(TI pass = 0; pass < 3; pass++){
            forward(cpu, reference, input, cpu_buffer, rng, mode);
            forward(gpu, actual, gpu_input, gpu_buffer, rng, mode);
            copy(gpu, cpu, actual.output, copied_output);
            close(reference.output, copied_output);
            backward_full(cpu, reference, input, d_output, d_input, cpu_buffer, mode);
            if(pass < 2){
                backward_full(gpu, actual, gpu_input, gpu_d_output, gpu_d_input, gpu_buffer, mode);
                copy(gpu, cpu, gpu_d_input, copied_d_input);
                close(d_input, copied_d_input);
            }
            else{
                backward(gpu, actual, gpu_input, gpu_d_output, gpu_buffer, mode);
            }
            copy(gpu, cpu, actual, copied);
            copy(gpu, cpu, gpu_d_output, copied_d_output);
            close(reference.weights.gradient, copied.weights.gradient);
            close(reference.biases.gradient, copied.biases.gradient);
            EXPECT_EQ(abs_diff(cpu, d_output, copied_d_output), 0);
            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
                close(reference.norm.gamma.gradient, copied.norm.gamma.gradient);
                close(reference.norm.beta.gradient, copied.norm.beta.gradient);
            }
        }
        free(cpu, copied_d_output);
        free(gpu, gpu_d_output);
        free(cpu, d_output);
        free(cpu, copied_output);
        free(gpu, gpu_output);
        free(cpu, output);
        free(cpu, copied_d_input);
        free(gpu, gpu_d_input);
        free(cpu, d_input);
        free(gpu, gpu_input);
        free(cpu, input);
        free(gpu, gpu_buffer);
        free(cpu, cpu_buffer);
        free(cpu, copied);
        free(gpu, actual);
        free(cpu, reference);
        free(cpu, rng);
        free(gpu);
    }

    template<nn::activation_functions::ActivationFunction ACTIVATION>
    void compare_modes(){
        using N = nn::layers::conv2d::Normalization;
        compare_activation<ACTIVATION, N::NONE>(Mode<mode::Default<>>{});
        compare_activation<ACTIVATION, N::BATCH_NORM>(Mode<mode::Default<>>{});
        compare_activation<ACTIVATION, N::BATCH_NORM>(Mode<mode::Evaluation<>>{});
    }
}

TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, GELU){
    compare_modes<nn::activation_functions::GELU>();
}
TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, TANH){
    compare_modes<nn::activation_functions::TANH>();
}
TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, FAST_TANH){
    compare_modes<nn::activation_functions::FAST_TANH>();
}
TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, SIGMOID){
    compare_modes<nn::activation_functions::SIGMOID>();
}
TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, RELU){
    compare_modes<nn::activation_functions::RELU>();
}
TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, IDENTITY){
    compare_modes<nn::activation_functions::IDENTITY>();
}

TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, DETERMINISTIC_BACKWARD){
    using namespace rl_tools;
    using CPU = devices::DEVICE_FACTORY<>;
    using GPU = devices::DefaultCUDA;
    using TI = GPU::index_t;
    using POLICY = numeric_types::Policy<float>;
    using CONFIG = nn::layers::conv2d::Configuration<POLICY, TI, 64, 4, 8, 4, 10, 0, 0, nn::activation_functions::IDENTITY>;
    using INPUT_SHAPE = tensor::Shape<TI, 512, 4, 18, 50>;
    using LAYER = nn::layers::conv2d::Layer<CONFIG, nn::capability::Gradient<nn::parameters::Gradient>, INPUT_SHAPE>;
    using INPUT = Tensor<tensor::Specification<float, TI, INPUT_SHAPE>>;
    using OUTPUT = Tensor<tensor::Specification<float, TI, typename LAYER::OUTPUT_SHAPE>>;
    CPU cpu;
    GPU gpu;
    init(gpu);
    typename CPU::SPEC::RANDOM::template ENGINE<> rng;
    malloc(cpu, rng);
    init(cpu, rng, 19);
    LAYER initial, actual, baseline, repeated;
    typename LAYER::template Buffer<true> buffer;
    INPUT input, gpu_input, gpu_d_input, baseline_d_input, repeated_d_input;
    OUTPUT d_output, gpu_d_output;
    malloc(cpu, initial);
    malloc(gpu, actual);
    malloc(cpu, baseline);
    malloc(cpu, repeated);
    malloc(gpu, buffer);
    malloc(cpu, input);
    malloc(gpu, gpu_input);
    malloc(gpu, gpu_d_input);
    malloc(cpu, baseline_d_input);
    malloc(cpu, repeated_d_input);
    malloc(cpu, d_output);
    malloc(gpu, gpu_d_output);
    init_weights(cpu, initial, rng);
    randn(cpu, input, rng);
    randn(cpu, d_output, rng);
    copy(cpu, gpu, initial, actual);
    copy(cpu, gpu, input, gpu_input);
    copy(cpu, gpu, d_output, gpu_d_output);
    for(TI pass = 0; pass < 12; pass++){
        zero_gradient(gpu, actual);
        forward(gpu, actual, gpu_input, buffer, rng);
        backward_full(gpu, actual, gpu_input, gpu_d_output, gpu_d_input, buffer);
        copy(gpu, cpu, actual, repeated);
        copy(gpu, cpu, gpu_d_input, repeated_d_input);
        if(pass == 0){
            copy(cpu, cpu, repeated, baseline);
            copy(cpu, cpu, repeated_d_input, baseline_d_input);
        }
        else{
            EXPECT_EQ(abs_diff(cpu, baseline.weights.gradient, repeated.weights.gradient), 0);
            EXPECT_EQ(abs_diff(cpu, baseline.biases.gradient, repeated.biases.gradient), 0);
            EXPECT_EQ(abs_diff(cpu, baseline_d_input, repeated_d_input), 0);
        }
        backward_input(gpu, actual, gpu_d_output, gpu_d_input, buffer);
        copy(gpu, cpu, gpu_d_input, repeated_d_input);
        EXPECT_EQ(abs_diff(cpu, baseline_d_input, repeated_d_input), 0);
    }
    free(gpu, gpu_d_output);
    free(cpu, d_output);
    free(cpu, repeated_d_input);
    free(cpu, baseline_d_input);
    free(gpu, gpu_d_input);
    free(gpu, gpu_input);
    free(cpu, input);
    free(gpu, buffer);
    free(cpu, repeated);
    free(cpu, baseline);
    free(gpu, actual);
    free(cpu, initial);
    free(cpu, rng);
    free(gpu);
}

namespace {
    template<nn::activation_functions::ActivationFunction ACTIVATION>
    void compare_cnn_feature_gradients(){
        using CPU = devices::DEVICE_FACTORY<>;
        using GPU = devices::DefaultCUDA;
        using TI = GPU::index_t;
        using POLICY = numeric_types::Policy<float>;
        using IMAGE_SHAPE = tensor::Shape<TI, 1, 4, 8, 10, 3>;
        using STATE_SHAPE = tensor::Shape<TI, 1, 4, 7>;
        using CONV_1 = nn::layers::conv2d::BindConfiguration<nn::layers::conv2d::Configuration<POLICY, TI, 8, 3, 3, 2, 2, 1, 1, ACTIVATION>>;
        using CONV_2 = nn::layers::conv2d::BindConfiguration<nn::layers::conv2d::Configuration<POLICY, TI, 16, 3, 3, 2, 2, 1, 1, ACTIVATION>>;
        using FLATTEN = nn::layers::flatten::BindConfiguration<nn::layers::flatten::Configuration<POLICY, TI>>;
        using EMBED = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<POLICY, TI, 6, nn::activation_functions::IDENTITY>>;
        using CNN = nn_models::sequential::Module<CONV_1, CONV_2, FLATTEN, EMBED>;
        using STATE = nn_models::sequential::Module<EMBED>;
        using HEAD = nn_models::mlp::BindConfiguration<nn_models::mlp::Configuration<POLICY, TI, 4, 3, 16, nn::activation_functions::RELU, nn::activation_functions::IDENTITY>>;
        using CAPABILITY = nn::capability::Gradient<nn::parameters::Gradient>;
        using MODEL = nn_models::parallel::Build<CAPABILITY, HEAD, nn_models::parallel::Branch<CNN, IMAGE_SHAPE>, nn_models::parallel::Branch<STATE, STATE_SHAPE>>;
        CPU cpu;
        GPU gpu;
        init(gpu);
        typename CPU::SPEC::RANDOM::template ENGINE<> rng;
        malloc(cpu, rng);
        init(cpu, rng, 19);
        MODEL reference, actual, copied;
        typename MODEL::template Buffer<true> cpu_buffer, gpu_buffer;
        Tensor<tensor::Specification<float, TI, IMAGE_SHAPE>> image, gpu_image;
        Tensor<tensor::Specification<float, TI, STATE_SHAPE>> state, gpu_state;
        Tensor<tensor::Specification<float, TI, typename MODEL::OUTPUT_SHAPE>> d_output, gpu_d_output;
        Tensor<tensor::Specification<float, TI, tensor::Shape<TI, 1, 4, 12>>> d_features, gpu_d_features;
        malloc(cpu, reference);
        malloc(gpu, actual);
        malloc(cpu, copied);
        malloc(cpu, cpu_buffer);
        malloc(gpu, gpu_buffer);
        malloc(cpu, image);
        malloc(gpu, gpu_image);
        malloc(cpu, state);
        malloc(gpu, gpu_state);
        malloc(cpu, d_output);
        malloc(gpu, gpu_d_output);
        malloc(cpu, d_features);
        malloc(gpu, gpu_d_features);
        init_weights(cpu, reference, rng);
        randn(cpu, image, rng);
        randn(cpu, state, rng);
        randn(cpu, d_output, rng);
        randn(cpu, d_features, rng);
        copy(cpu, gpu, reference, actual);
        copy(cpu, gpu, image, gpu_image);
        copy(cpu, gpu, state, gpu_state);
        copy(cpu, gpu, d_output, gpu_d_output);
        copy(cpu, gpu, d_features, gpu_d_features);
        auto inputs = nn_models::parallel::pack_inputs(image, state);
        auto gpu_inputs = nn_models::parallel::pack_inputs(gpu_image, gpu_state);
        zero_gradient(cpu, reference);
        zero_gradient(gpu, actual);
        forward(cpu, reference, inputs, cpu_buffer, rng);
        forward(gpu, actual, gpu_inputs, gpu_buffer, rng);
        backward(cpu, reference, inputs, d_output, d_features, cpu_buffer);
        backward(gpu, actual, gpu_inputs, gpu_d_output, gpu_d_features, gpu_buffer);
        copy(gpu, cpu, actual, copied);
        auto close = [&](const auto& expected, const auto& observed){
            EXPECT_LT(abs_diff(cpu, expected.weights.gradient, observed.weights.gradient) / (1.0f + abs_sum(cpu, expected.weights.gradient)), 3e-4f);
            EXPECT_LT(abs_diff(cpu, expected.biases.gradient, observed.biases.gradient) / (1.0f + abs_sum(cpu, expected.biases.gradient)), 3e-4f);
        };
        auto& reference_cnn = get<0>(reference.pipelines);
        auto& copied_cnn = get<0>(copied.pipelines);
        close(nn_models::sequential::layer<0>(reference_cnn), nn_models::sequential::layer<0>(copied_cnn));
        close(nn_models::sequential::layer<1>(reference_cnn), nn_models::sequential::layer<1>(copied_cnn));
        close(nn_models::sequential::layer<3>(reference_cnn), nn_models::sequential::layer<3>(copied_cnn));
        close(nn_models::sequential::layer<0>(get<1>(reference.pipelines)), nn_models::sequential::layer<0>(get<1>(copied.pipelines)));
        close(reference.head.input_layer, copied.head.input_layer);
        close(reference.head.hidden_layers[0], copied.head.hidden_layers[0]);
        close(reference.head.output_layer, copied.head.output_layer);
        free(gpu, gpu_d_features);
        free(cpu, d_features);
        free(gpu, gpu_d_output);
        free(cpu, d_output);
        free(gpu, gpu_state);
        free(cpu, state);
        free(gpu, gpu_image);
        free(cpu, image);
        free(gpu, gpu_buffer);
        free(cpu, cpu_buffer);
        free(cpu, copied);
        free(gpu, actual);
        free(cpu, reference);
        free(cpu, rng);
        free(gpu);
    }
}

TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, CNN_FEATURE_GRADIENT_RELU){
    compare_cnn_feature_gradients<nn::activation_functions::RELU>();
}
TEST(NN_LAYERS_CONV2D_CUDA_ACTIVATION, CNN_FEATURE_GRADIENT_GELU){
    compare_cnn_feature_gradients<nn::activation_functions::GELU>();
}
