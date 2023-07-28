#include <backprop_tools/operations/cpu.h>

#include "../../../../data/nn_models_sequential_persist_code.h"

#include <backprop_tools/nn/operations_generic.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>


namespace bpt = backprop_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using T = float;
using DEVICE = bpt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

TEST(BACKPROP_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_COMPILE, COMPILE) {
    DEVICE device;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, backprop_tools_export::model::MODEL::OUTPUT_DIM>> output;
    backprop_tools_export::model::MODEL::DoubleBuffer<1> buffer;

    bpt::malloc(device, output);
    bpt::malloc(device, buffer);

    bpt::evaluate(device, backprop_tools_export::model::model, backprop_tools_export::input::container, output, buffer);

    auto abs_diff = bpt::abs_diff(device, output, backprop_tools_export::output::container);

    std::cout << "Oiriginal output:" << std::endl;
    bpt::print(device, backprop_tools_export::output::container);
    std::cout << "Loaded output:" << std::endl;
    bpt::print(device, output);

    std::cout << "abs_diff: " << abs_diff << std::endl;
    ASSERT_LT(abs_diff, 1e-5);
}

