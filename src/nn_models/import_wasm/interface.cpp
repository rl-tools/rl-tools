#include <rl_tools/operations/dummy.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include "/Users/jonas/rl_tools/experiments/2024-07-02_17-00-10/4f39b6e_zoo_algorithm_environment/td3_l2f/0000/steps/000000003000000/checkpoint.h"

namespace rlt = rl_tools;

#include <stdint.h>

using DEVICE = rlt::devices::DefaultDummy;
using T = rlt::checkpoint::actor::MODEL::T;
using TI = DEVICE::index_t;
constexpr TI BATCH_SIZE = 1;
rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE, rl_tools::checkpoint::actor::MODEL::OUTPUT_DIM>> output;
rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE, rl_tools::checkpoint::actor::MODEL::INPUT_DIM>> input;
rlt::checkpoint::actor::MODEL::Buffer<BATCH_SIZE, rlt::MatrixStaticTag> buffer;

extern "C" {
    int32_t batch_size();
    int32_t input_dim();
    int32_t output_dim();
    void set_input(int32_t row, int32_t col, T value);
    T get_output(int32_t row, int32_t col);
    void evaluate();

    // example
    int32_t get_example_batch_size();
    T get_example_input(int32_t row, int32_t col);
    T get_example_output(int32_t row, int32_t col);
}
int32_t batch_size(){
    return BATCH_SIZE;
}
int32_t input_dim(){
    return rl_tools::checkpoint::actor::MODEL::INPUT_DIM;
}
int32_t output_dim(){
    return rl_tools::checkpoint::actor::MODEL::OUTPUT_DIM;
}

int32_t get_example_batch_size(){
    return rlt::checkpoint::example::input::CONTAINER_TYPE::COLS;
}
T get_example_input(int32_t row, int32_t col){
    return rlt::get(rlt::checkpoint::example::input::container, row, col);
}
T get_example_output(int32_t row, int32_t col){
    return rlt::get(rlt::checkpoint::example::output::container, row, col);
}

void set_input(int32_t row, int32_t col, T value){
    rlt::set(input, row, col, value);
}

T get_output(int32_t row, int32_t col){
    return rlt::get(output, row, col);
}

void evaluate(){
    DEVICE device;
    bool rng;
    rlt::evaluate(device, rlt::checkpoint::actor::module, input, output, buffer, rng);
}

