#ifndef RL_TOOLS_WASM
#include <rl_tools/operations/arm.h>
#else
#include <rl_tools/operations/wasm32.h>
#endif

// #define RL_TOOLS_DISABLE_TEST
#include "c_interface.h"
#include "../../executor/c_backend.h"


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

#include "operations_generic.h"

#include "../../../../../tests/data/test_inference_executor_policy.h"

#ifdef RL_TOOLS_ENABLE_INFORMATIVE_STATUS_MESSAGES
#include <cstdio>
#endif

namespace rlt = rl_tools;

#ifndef RL_TOOLS_WASM
using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
#else
using DEVICE = rlt::devices::DefaultWASM32;
#endif


using TI = typename DEVICE::index_t;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
static constexpr TI TEST_SEQUENCE_LENGTH = rlt::checkpoint::example::input::SHAPE::template GET<0>;
static constexpr TI TEST_BATCH_SIZE = rlt::checkpoint::example::input::SHAPE::template GET<1>;
static constexpr TI TEST_SEQUENCE_LENGTH_ACTUAL = 5;
static constexpr TI TEST_BATCH_SIZE_ACTUAL = 2;
static_assert(TEST_BATCH_SIZE_ACTUAL <= TEST_BATCH_SIZE);
static_assert(TEST_SEQUENCE_LENGTH_ACTUAL <= TEST_SEQUENCE_LENGTH);
using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
using ACTOR_TYPE_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
using ACTOR_TYPE = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI ACTION_HISTORY_LENGTH = 1; //rlt::checkpoint::environment::ACTION_HISTORY_LENGTH
constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 1 * 1000 * 1000; // Inference is at 500hz
constexpr TI CONTROL_INTERVAL_NATIVE_NS = 10 * 1000 * 1000; // Training is 100hz
static constexpr TI INPUT_DIM = rlt::get_last(ACTOR_TYPE::INPUT_SHAPE{});
static constexpr TI OUTPUT_DIM = rlt::get_last(ACTOR_TYPE::OUTPUT_SHAPE{});
static_assert(OUTPUT_DIM == 4);
static_assert(INPUT_DIM == (18 + ACTION_HISTORY_LENGTH * OUTPUT_DIM));
constexpr TI TIMING_STATS_NUM_STEPS = 100;
static constexpr bool FORCE_SYNC_INTERMEDIATE = true;
static constexpr TI FORCE_SYNC_NATIVE = 0;
static constexpr bool DYNAMIC_ALLOCATION = false;
using SPEC = rlt::inference::applications::l2f::Specification<T, TI, RLtoolsInferenceTimestamp, ACTION_HISTORY_LENGTH, OUTPUT_DIM, ACTOR_TYPE, CONTROL_INTERVAL_INTERMEDIATE_NS, CONTROL_INTERVAL_NATIVE_NS, FORCE_SYNC_INTERMEDIATE, FORCE_SYNC_NATIVE, DYNAMIC_ALLOCATION>;

auto& policy = rlt::checkpoint::actor::module;

DEVICE device;
RNG rng;
static rlt::inference::applications::L2F<SPEC> executor;

// Buffers
static ACTOR_TYPE_TEST::template Buffer<false> buffers_test;
static ACTOR_TYPE_TEST::State<false> policy_state_test;
static ACTOR_TYPE::State<false> policy_state_buffer;
static ACTOR_TYPE::template Buffer<false> buffers;
static rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, INPUT_DIM>, false>> input;
static rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, OUTPUT_DIM>, false>> output;


// Main functions (possibly with side effects)
void rl_tools_inference_applications_l2f_reset(){
    rlt::reset(device, executor, policy, rng);
}
void rl_tools_inference_applications_l2f_init(){
    TI seed = 0;
    rlt::init(device, rng, seed);
    rl_tools_inference_applications_l2f_reset();
}


const char* rl_tools_inference_applications_l2f_checkpoint_name(){
    return (char*)rlt::checkpoint::meta::name;
}

float rl_tools_inference_applications_l2f_test(RLtoolsInferenceApplicationsL2FAction* p_output){
#ifndef RL_TOOLS_DISABLE_TEST
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    float acc = 0;
    uint64_t num_values = 0;
    for(TI batch_i = 0; batch_i < TEST_BATCH_SIZE_ACTUAL; batch_i++){
        rlt::reset(device, rlt::checkpoint::actor::module, policy_state_test, rng);
        for(TI step_i = 0; step_i < TEST_SEQUENCE_LENGTH_ACTUAL; step_i++){
            const auto step_input = rlt::view(device, rlt::checkpoint::example::input::container, step_i);
            const auto batch_input = rlt::view_range(device, step_input, batch_i, rlt::tensor::ViewSpec<0, 1>{});
            rlt::utils::assert_exit(device, !rlt::is_nan(device, batch_input), "input is nan");
            rlt::utils::assert_exit(device, !rlt::is_nan(device, policy_state_test.content_state.next_content_state.state.state), "state is nan");
            rlt::evaluate_step(device, rlt::checkpoint::actor::module, batch_input, policy_state_test, output, buffers_test, rng, mode);
            rlt::utils::assert_exit(device, !rlt::is_nan(device, output), "output is nan");
            for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
                acc += rlt::math::abs(device.math, rlt::get(device, output, 0, action_i) - rlt::get(device, rlt::checkpoint::example::output::container, step_i, batch_i, action_i));
                num_values += 1;
                rlt::utils::assert_exit(device, !rlt::math::is_nan(device.math, acc), "output is nan");
                if(batch_i == 0 && step_i == TEST_SEQUENCE_LENGTH-1){
                    p_output->action[action_i] = rlt::get(device, output, 0, action_i);
                }
            }
        }
    }
    return acc / num_values;
#else
    return 0;
#endif
}

RLtoolsInferenceExecutorStatus rl_tools_inference_applications_l2f_control(RLtoolsInferenceTimestamp nanoseconds, RLtoolsInferenceApplicationsL2FObservation* c_observation, RLtoolsInferenceApplicationsL2FAction* c_action){
    rlt::inference::applications::l2f::Observation<SPEC> observation;
    for (TI dim_i = 0; dim_i < 3; dim_i++){
        observation.position[dim_i] = c_observation->position[dim_i];
        observation.orientation[dim_i] = c_observation->orientation[dim_i];
        observation.linear_velocity[dim_i] = c_observation->linear_velocity[dim_i];
        observation.angular_velocity[dim_i] = c_observation->angular_velocity[dim_i];
    }
    observation.orientation[3] = c_observation->orientation[3];
    rlt::inference::applications::l2f::Action<SPEC> action;
    static_assert(RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM == OUTPUT_DIM);
    auto status = rlt::control(device, executor, nanoseconds, policy, observation, action, rng);
    for (TI action_i=0; action_i < OUTPUT_DIM; action_i++){
        c_action->action[action_i] = action.action[action_i];
    }
    return rlt::convert(status);
}
