#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>
namespace bpt = backprop_tools;

#include "/home/jonas/phd/projects/rl_for_control/backprop_tools_multirotor/checkpoints/multirotor_td3/2023-05-24T13_11_21-0400_ppo_ant_101/actor_000000000000000.h"

#include <iostream>

using T = typename bpt::checkpoint::action::CONTAINER_SPEC::T;
using TI = typename bpt::checkpoint::action::CONTAINER_SPEC::TI;
using ACTOR_TYPE = decltype(bpt::checkpoint::actor::mlp);
using DEVICE = bpt::devices::DefaultCPU;

int main(){
    DEVICE device;
    for(int i = 0; i < backprop_tools::checkpoint::action::CONTAINER_SPEC::COLS; i++){
        std::cout << i << ": " << get(backprop_tools::checkpoint::action::container, 0, i) << std::endl;
    }
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output;
    ACTOR_TYPE::Buffers<1> buffers;
    bpt::malloc(device, output);
    bpt::malloc(device, buffers);
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, bpt::checkpoint::observation::container, output, buffers);
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(bpt::get(output, 0, i) - bpt::get(bpt::checkpoint::action::container, 0, i));
    }
    return 0;
}