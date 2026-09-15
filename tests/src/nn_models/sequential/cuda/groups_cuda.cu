#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/nn/optimizers/sgd/operations_cuda.h>
#include <metra/metra.h>
#include <gtest/gtest.h>

namespace rl_tools::tests::groups_cuda{
    struct Fixed{};
    struct Learned{};
    using CPU = devices::DEVICE_FACTORY<>;
    using GPU = devices::DefaultCUDA;
    using TI = GPU::index_t;
    using POLICY = numeric_types::Policy<float>;
    using DENSE = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<POLICY, TI, 3, nn::activation_functions::TANH>>;
    using CHAIN = nn_models::sequential::Module<nn_models::Group<Fixed, DENSE>, nn_models::Group<Learned, DENSE>>;
    using INPUT_SHAPE = tensor::Shape<TI, 1, 2, 3>;
    using WEIGHTS = Mode<nn::optimizers::mode::Update<mode::Default<>, nn::group::Every<nn::group::Only<Learned>, nn::group::Category<nn::parameters::categories::Weights>>>>;
    using FREEZE = Mode<nn::optimizers::mode::Freeze<mode::Default<>, nn::group::Any<nn::group::Only<Fixed>, nn::group::Category<nn::parameters::categories::Biases>>>>;

    template<typename PARAMETER, typename OPTIMIZER>
    void check_selection(){
        using MODEL = nn_models::sequential::Build<nn::capability::Gradient<PARAMETER>, CHAIN, INPUT_SHAPE>;
        CPU cpu;
        GPU gpu;
        init(gpu);
        CPU::SPEC::RANDOM::ENGINE<> rng;
        malloc(cpu, rng);
        init(cpu, rng, 31);
        MODEL reference, actual, copied, snapshot;
        typename MODEL::template Buffer<> buffer;
        Tensor<tensor::Specification<float, TI, INPUT_SHAPE>> input, d_output;
        OPTIMIZER cpu_optimizer, gpu_optimizer;
        malloc(cpu, reference);
        malloc(gpu, actual);
        malloc(cpu, copied);
        malloc(cpu, snapshot);
        malloc(cpu, buffer);
        malloc(cpu, input);
        malloc(cpu, d_output);
        malloc(cpu, cpu_optimizer);
        malloc(gpu, gpu_optimizer);
        init(cpu, cpu_optimizer);
        init(gpu, gpu_optimizer);
        init_weights(cpu, reference, rng);
        reset_optimizer_state(cpu, cpu_optimizer, reference);
        reset_optimizer_state(gpu, gpu_optimizer, actual);
        zero_gradient(cpu, reference);
        set_all(cpu, input, 0.4f);
        set_all(cpu, d_output, 0.7f);
        forward(cpu, reference, input, buffer, rng);
        backward(cpu, reference, input, d_output, buffer);
        copy(cpu, gpu, reference, actual);
        step(cpu, cpu_optimizer, reference);
        step(gpu, gpu_optimizer, actual);
        copy(gpu, cpu, actual, snapshot);
        step(cpu, cpu_optimizer, reference, WEIGHTS{});
        step(gpu, gpu_optimizer, actual, FREEZE{});
        copy(gpu, cpu, actual, copied);
        EXPECT_EQ(abs_diff(cpu, get_layer<0>(copied).weights, get_layer<0>(snapshot).weights), 0);
        EXPECT_EQ(abs_diff(cpu, get_layer<1>(copied).biases, get_layer<1>(snapshot).biases), 0);
        EXPECT_GT(abs_diff(cpu, get_layer<1>(copied).weights.parameters, get_layer<1>(snapshot).weights.parameters), 0);
        EXPECT_LT(abs_diff(cpu, reference, copied), 1e-4f);
        metra::log("nn_groups/cuda_update_error", abs_diff(cpu, reference, copied));
        step(cpu, cpu_optimizer, reference);
        step(gpu, gpu_optimizer, actual);
        copy(gpu, cpu, actual, copied);
        EXPECT_GT(abs_diff(cpu, get_layer<0>(copied).weights.parameters, get_layer<0>(snapshot).weights.parameters), 0);
        EXPECT_LT(abs_diff(cpu, reference, copied), 1e-4f);
        free(gpu, gpu_optimizer);
        free(cpu, cpu_optimizer);
        free(cpu, d_output);
        free(cpu, input);
        free(cpu, buffer);
        free(cpu, snapshot);
        free(cpu, copied);
        free(gpu, actual);
        free(cpu, reference);
        free(cpu, rng);
    }

    TEST(NN_GROUPS_CUDA, AdamFreezeUnfreeze){
        check_selection<nn::parameters::Adam, nn::optimizers::Adam<nn::optimizers::adam::Specification<POLICY, TI>>>();
    }
    TEST(NN_GROUPS_CUDA, SgdFreezeUnfreeze){
        check_selection<nn::parameters::SGD, nn::optimizers::SGD<nn::optimizers::sgd::Specification<POLICY, TI>>>();
    }

    template<typename LAYER, typename SHAPE, bool INPUT_ONLY_SUPPORTED>
    void check_pruned_backward(){
        using NETWORK = nn_models::sequential::Build<nn::capability::Gradient<nn::parameters::Adam>, nn_models::sequential::Module<nn_models::Group<Fixed, LAYER>, nn_models::Group<Learned, LAYER>, nn_models::Group<Fixed, LAYER>, nn_models::Group<Learned, LAYER>>, SHAPE>;
        using TRAINING_MODE = Mode<nn_models::backpropagation::mode::SkipParameterGradients<mode::Default<>, nn::group::Only<Fixed>>>;
        using TENSOR = Tensor<tensor::Specification<float, TI, SHAPE>>;
        CPU cpu;
        GPU gpu;
        init(gpu);
        CPU::SPEC::RANDOM::ENGINE<> rng;
        malloc(cpu, rng);
        init(cpu, rng, 47);
        NETWORK actual, expected, actual_cpu, expected_cpu;
        typename NETWORK::template Buffer<> actual_buffer, expected_buffer;
        TENSOR input, gpu_input, d_output, actual_d_output, expected_d_output, actual_d_input, expected_d_input, actual_d_input_cpu, expected_d_input_cpu;
        malloc(gpu, actual);
        malloc(gpu, expected);
        malloc(cpu, actual_cpu);
        malloc(cpu, expected_cpu);
        malloc(gpu, actual_buffer);
        malloc(gpu, expected_buffer);
        malloc(cpu, input);
        malloc(gpu, gpu_input);
        malloc(cpu, d_output);
        malloc(gpu, actual_d_output);
        malloc(gpu, expected_d_output);
        malloc(gpu, actual_d_input);
        malloc(gpu, expected_d_input);
        malloc(cpu, actual_d_input_cpu);
        malloc(cpu, expected_d_input_cpu);
        init_weights(cpu, actual_cpu, rng);
        zero_gradient(cpu, actual_cpu);
        copy(cpu, gpu, actual_cpu, actual);
        copy(cpu, gpu, actual_cpu, expected);
        randn(cpu, input, rng);
        randn(cpu, d_output, rng);
        copy(cpu, gpu, input, gpu_input);
        for(bool input_gradients: {false, true}){
            zero_gradient(gpu, actual);
            zero_gradient(gpu, expected);
            copy(cpu, gpu, d_output, actual_d_output);
            copy(cpu, gpu, d_output, expected_d_output);
            forward(gpu, actual, gpu_input, actual_buffer, rng);
            forward(gpu, expected, gpu_input, expected_buffer, rng);
            if(input_gradients){
                backward_full(gpu, actual, gpu_input, actual_d_output, actual_d_input, actual_buffer, TRAINING_MODE{});
                backward_full(gpu, expected, gpu_input, expected_d_output, expected_d_input, expected_buffer);
                copy(gpu, cpu, actual_d_input, actual_d_input_cpu);
                copy(gpu, cpu, expected_d_input, expected_d_input_cpu);
                EXPECT_LT(abs_diff(cpu, actual_d_input_cpu, expected_d_input_cpu), 1e-5f);
                EXPECT_GT(squared_sum(cpu, actual_d_input_cpu), 0);
            }
            else{
                backward(gpu, actual, gpu_input, actual_d_output, actual_buffer, TRAINING_MODE{});
                backward(gpu, expected, gpu_input, expected_d_output, expected_buffer);
            }
            copy(gpu, cpu, actual, actual_cpu);
            copy(gpu, cpu, expected, expected_cpu);
            EXPECT_LT(abs_diff(cpu, get_layer<1>(actual_cpu).weights.gradient, get_layer<1>(expected_cpu).weights.gradient), 1e-5f);
            EXPECT_GT(squared_sum(cpu, get_layer<1>(actual_cpu).weights.gradient), 0);
            EXPECT_LT(abs_diff(cpu, get_layer<3>(actual_cpu).weights.gradient, get_layer<3>(expected_cpu).weights.gradient), 1e-5f);
            if constexpr(INPUT_ONLY_SUPPORTED){
                EXPECT_EQ(gradient_norm(cpu, get_layer<0>(actual_cpu)), 0);
                EXPECT_EQ(gradient_norm(cpu, get_layer<2>(actual_cpu)), 0);
            }
            else{
                EXPECT_LT(abs_diff(cpu, get_layer<2>(actual_cpu).weights.gradient, get_layer<2>(expected_cpu).weights.gradient), 1e-5f);
                EXPECT_GT(squared_sum(cpu, get_layer<2>(actual_cpu).weights.gradient), 0);
                if(!input_gradients){
                    EXPECT_EQ(squared_sum(cpu, get_layer<0>(actual_cpu).weights.gradient), 0);
                }
            }
        }
        free(cpu, expected_d_input_cpu);
        free(cpu, actual_d_input_cpu);
        free(gpu, expected_d_input);
        free(gpu, actual_d_input);
        free(gpu, expected_d_output);
        free(gpu, actual_d_output);
        free(cpu, d_output);
        free(gpu, gpu_input);
        free(cpu, input);
        free(gpu, expected_buffer);
        free(gpu, actual_buffer);
        free(cpu, expected_cpu);
        free(cpu, actual_cpu);
        free(gpu, expected);
        free(gpu, actual);
        free(cpu, rng);
    }

    TEST(NN_GROUPS_CUDA, PrunedNonlinearDenseBackward){
        check_pruned_backward<DENSE, INPUT_SHAPE, true>();
    }
    TEST(NN_GROUPS_CUDA, FrozenNonlinearConvolutionFallback){
        using SHAPE = tensor::Shape<TI, 1, 2, 4, 4, 2>;
        using CONV = nn::layers::conv2d::BindConfiguration<nn::layers::conv2d::Configuration<POLICY, TI, 2, 1, 1, 1, 1, 0, 0, nn::activation_functions::GELU, nn::layers::conv2d::Normalization::BATCH_NORM>>;
        check_pruned_backward<CONV, SHAPE, false>();
    }
}
