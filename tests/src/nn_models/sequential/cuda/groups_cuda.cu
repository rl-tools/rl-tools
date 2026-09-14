#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
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
}
