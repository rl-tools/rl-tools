// #include "rl_tools_adapter_new.h"
#include <iostream>
#include <random>
#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#ifndef RL_TOOLS_WASM
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
// #include <rl_tools/nn/layers/dense/operations_generic.h>
#else
#include <rl_tools/nn/layers/dense/operations_generic.h>
#endif
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/inference/executor/executor.h>
#include <rl_tools/inference/executor/operations_generic.h>
#include <rl_tools/inference/applications/l2f/l2f.h>
#include <rl_tools/inference/applications/l2f/operations_generic.h>

#include "../../../../../logs/2025-04-11_12-54-35/checkpoint.h"

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
using TIMESTAMP = uint64_t;

using POLICY = rlt::checkpoint::actor::TYPE;
static constexpr TI ACTION_HISTORY_LENGTH = 1;
static constexpr TI OUTPUT_DIM = POLICY::OUTPUT_SHAPE::LAST;
static constexpr TIMESTAMP CONTROL_INTERVAL_INFERENCE_NS = 2500 * 1000;
static constexpr TIMESTAMP CONTROL_INTERVAL_TRAINING_NS = 10000 * 1000;
static constexpr bool DYNAMIC_ALLOCATION = false;
using SPEC = rlt::inference::applications::l2f::Specification<T, TI, TIMESTAMP, ACTION_HISTORY_LENGTH, OUTPUT_DIM, CONTROL_INTERVAL_INFERENCE_NS, CONTROL_INTERVAL_TRAINING_NS, POLICY, DYNAMIC_ALLOCATION>;

// static constexpr uint OUTPUT_DIM = 4;
// int main(){
// }

#include <gtest/gtest.h>

TEST(RL_TOOLS_INFERENCE_EXECUTOR, MAIN){
    auto& policy = rlt::checkpoint::actor::module;
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = 0;
    rlt::init(device, rng, seed);

    rlt::inference::applications::L2F<SPEC> executor;
    rlt::malloc(device, executor);
    rlt::reset(device, executor, policy, rng);

    // RLtoolsAction action;
    // float test_result = rl_tools_test(&action);
    // std::cout << "test: " << test_result << std::endl;
    // for(uint i = 0; i < OUTPUT_DIM; i++){
    //     std::cout << "action[" << i << "] = " << action.action[i] << std::endl;
    // }
    // RLtoolsObservation observation;
    // observation.position[0] = 0.0f;
    // observation.position[1] = 0.0f;
    // observation.position[2] = 0.0f;
    // observation.orientation[0] = 1.0f;
    // observation.orientation[1] = 0.0f;
    // observation.orientation[2] = 0.0f;
    // observation.orientation[3] = 0.0f;
    // observation.linear_velocity[0] = 0.0f;
    // observation.linear_velocity[1] = 0.0f;
    // observation.linear_velocity[2] = 0.0f;
    // observation.angular_velocity[0] = 0.0f;
    // observation.angular_velocity[1] = 0.0f;
    // observation.angular_velocity[2] = 0.0f;
    // for(uint j = 0; j < OUTPUT_DIM; j++){
    //     observation.previous_action[j] = 0.0f;
    // }
    // std::default_random_engine rng;
    // RLtoolsStatus status;
    // for(uint64_t timestamp=0; timestamp < 10000000;){
    //     status = rl_tools_control(timestamp, &observation, &action);
    //     if(status != RL_TOOLS_STATUS_OK){
    //
    //     }
    //     std::cout << timestamp << " status: " << rl_tools_get_status_message(status) << std::endl;
    //     for(uint i = 0; i < OUTPUT_DIM; i++){
    //         std::cout << "action[" << i << "] = " << action.action[i] << std::endl;
    //     }
    //     timestamp += 2000; //std::uniform_int_distribution<uint>(500, 5000)(rng);
    // }
    // return 0;
}
