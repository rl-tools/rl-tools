#define RL_TOOLS_NAMESPACE base
#include "rl_tools/base/include/rl_tools/operations/cpu.h"
#include "rl_tools/base/include/rl_tools/nn/layers/gru/operations_generic.h"
#include "rl_tools/base/include/rl_tools/nn/layers/dense/operations_generic.h"
#include "rl_tools/base/include/rl_tools/nn_models/sequential/operations_generic.h"

#include "rl_tools/base/include/rl_tools/nn/layers/gru/persist.h"
#include "rl_tools/base/include/rl_tools/nn/layers/dense/persist.h"
#include "rl_tools/base/include/rl_tools/nn_models/sequential/persist.h"

using T = float;
constexpr unsigned SEQUENCE_LENGTH = 5;
constexpr unsigned BATCH_SIZE = 6;
constexpr unsigned INPUT_DIM = 22;
constexpr unsigned HIDDEN_DIM = 16;
constexpr unsigned ACTION_DIM = 4;
constexpr unsigned SEED = 0;
constexpr bool DYNAMIC_ALLOCATION = true;
namespace base{
    namespace rlt = rl_tools;
    using DEVICE = rl_tools::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;
    template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
    using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

    using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Input>;
    using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
    using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM, rlt::nn::parameters::groups::Normal>;
    using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
    using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Output>;
    using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
    using MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU, Module<OUTPUT_LAYER>>>;
    using CAPABILITY = rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
    using POLICY = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
}

#undef RL_TOOLS_NAMESPACE
#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

namespace target{
    namespace rlt = rl_tools;
    using DEVICE = rl_tools::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
    using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

    // using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::FAST_TANH, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    // using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;
    // using MODULE_CHAIN = Module<MLP>;

    using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Input>;
    using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
    using GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, rlt::nn::parameters::groups::Normal>;
    using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
    using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Output>;
    using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
    using MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU, Module<OUTPUT_LAYER>>>;
    using CAPABILITY = rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
    using POLICY = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
}

#undef RL_TOOLS_NAMESPACE




int main(){
    base::DEVICE base_device;
    base::DEVICE::SPEC::RANDOM::ENGINE<> rng;
    base::POLICY base_policy;

    base::rlt::malloc(base_device);
    base::rlt::malloc(base_device, rng);
    base::rlt::init(base_device);
    base::rlt::init(base_device, rng, SEED);
    base::rlt::malloc(base_device, base_policy);
    base::rlt::init_weights(base_device, base_policy, rng);


}