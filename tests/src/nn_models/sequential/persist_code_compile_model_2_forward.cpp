#include <rl_tools/operations/cpu.h>

#include "../../../data/nn_models_sequential_persist_code_model_2_forward.h"

#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev//operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>


namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using T = double;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_COMPILE, MODEL_2_FORWARD){
    DEVICE device;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::OUTPUT_DIM>> output;
    rl_tools_export::model::MODEL::Buffer<1> buffer;

    rlt::malloc(device, output);
    rlt::malloc(device, buffer);

    bool rng = false;

    rlt::evaluate(device, rl_tools_export::model::module, rl_tools_export::input::container, output, buffer, rng);

    auto abs_diff = rlt::abs_diff(device, output, rl_tools_export::output::container);

    std::cout << "Oiriginal output:" << std::endl;
    rlt::print(device, rl_tools_export::output::container);
    std::cout << "Loaded output:" << std::endl;
    rlt::print(device, output);

    std::cout << "abs_diff: " << abs_diff << std::endl;
    ASSERT_LT(abs_diff, 1e-5);

    {
        auto& first_layer = rl_tools_export::model::module.content;
        for(TI input_i=0; input_i < rl_tools_export::model::MODEL::INPUT_DIM; input_i++){
            T mean = rlt::get(first_layer.mean.parameters, 0, input_i);
            ASSERT_EQ(mean, input_i);
            T precision = rlt::get(first_layer.precision.parameters, 0, input_i);
            ASSERT_EQ(precision, input_i*2);
        }
    }

    {
        auto& last_layer = rlt::get_last_layer(rl_tools_export::model::module);
        for(TI output_i=0; output_i < rl_tools_export::model::MODEL::OUTPUT_DIM; output_i++){
            T p = rlt::get(last_layer.log_std.parameters, 0, output_i);
            ASSERT_EQ(p, output_i);
        }
    }
}

