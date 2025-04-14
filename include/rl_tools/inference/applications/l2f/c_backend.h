#include "../../executor/operations_generic.h"
#include "c_interface.h"
#include "../../executor/c_backend.h"
#include "operations_generic.h"

using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI ACTION_HISTORY_LENGTH = 1; //rl_tools::checkpoint::environment::ACTION_HISTORY_LENGTH
constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 1 * 1000 * 1000; // Inference is at 500hz
constexpr TI CONTROL_INTERVAL_NATIVE_NS = 10 * 1000 * 1000; // Training is 100hz
static constexpr TI INPUT_DIM = rl_tools::get_last(ACTOR_TYPE::INPUT_SHAPE{});
static constexpr TI OUTPUT_DIM = rl_tools::get_last(ACTOR_TYPE::OUTPUT_SHAPE{});
static_assert(OUTPUT_DIM == 4);
static_assert(INPUT_DIM == (18 + ACTION_HISTORY_LENGTH * OUTPUT_DIM));
constexpr TI TIMING_STATS_NUM_STEPS = 100;
static constexpr bool FORCE_SYNC_INTERMEDIATE = true;
static constexpr TI FORCE_SYNC_NATIVE = 0;
static constexpr bool DYNAMIC_ALLOCATION = false;
using SPEC = rl_tools::inference::applications::l2f::Specification<T, TI, RLtoolsInferenceTimestamp, ACTION_HISTORY_LENGTH, OUTPUT_DIM, ACTOR_TYPE, CONTROL_INTERVAL_INTERMEDIATE_NS, CONTROL_INTERVAL_NATIVE_NS, FORCE_SYNC_INTERMEDIATE, FORCE_SYNC_NATIVE, DYNAMIC_ALLOCATION>;


DEVICE device;
RNG rng;
static rl_tools::inference::applications::L2F<SPEC> executor;

// Buffers
static ACTOR_TYPE_TEST::template Buffer<false> buffers_test;
static ACTOR_TYPE_TEST::State<false> policy_state_test;
static ACTOR_TYPE::State<false> policy_state_buffer;
static ACTOR_TYPE::template Buffer<false> buffers;
static rl_tools::Tensor<rl_tools::tensor::Specification<T, TI, rl_tools::tensor::Shape<TI, 1, INPUT_DIM>, false>> input;
static rl_tools::Tensor<rl_tools::tensor::Specification<T, TI, rl_tools::tensor::Shape<TI, 1, OUTPUT_DIM>, false>> output;


// Main functions (possibly with side effects)
void rl_tools_inference_applications_l2f_reset(){
    rl_tools::reset(device, executor, rl_tools_inference_applications_l2f_policy, rng);
}
void rl_tools_inference_applications_l2f_init(){
    TI seed = 0;
    rl_tools::init(device, rng, seed);
    rl_tools_inference_applications_l2f_reset();
}


const char* rl_tools_inference_applications_l2f_checkpoint_name(){
    return (char*)rl_tools::checkpoint::meta::name;
}

float rl_tools_inference_applications_l2f_test(RLtoolsInferenceApplicationsL2FAction* p_output){
#ifndef RL_TOOLS_DISABLE_TEST
    rl_tools::Mode<rl_tools::mode::Evaluation<>> mode;
    float acc = 0;
    uint64_t num_values = 0;
    for(TI batch_i = 0; batch_i < TEST_BATCH_SIZE_ACTUAL; batch_i++){
        rl_tools::reset(device, rl_tools::checkpoint::actor::module, policy_state_test, rng);
        for(TI step_i = 0; step_i < TEST_SEQUENCE_LENGTH_ACTUAL; step_i++){
            const auto step_input = rl_tools::view(device, rl_tools::checkpoint::example::input::container, step_i);
            const auto batch_input = rl_tools::view_range(device, step_input, batch_i, rl_tools::tensor::ViewSpec<0, 1>{});
            rl_tools::utils::assert_exit(device, !rl_tools::is_nan(device, batch_input), "input is nan");
            rl_tools::utils::assert_exit(device, !rl_tools::is_nan(device, policy_state_test.content_state.next_content_state.state.state), "state is nan");
            rl_tools::evaluate_step(device, rl_tools::checkpoint::actor::module, batch_input, policy_state_test, output, buffers_test, rng, mode);
            rl_tools::utils::assert_exit(device, !rl_tools::is_nan(device, output), "output is nan");
            for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
                acc += rl_tools::math::abs(device.math, rl_tools::get(device, output, 0, action_i) - rl_tools::get(device, rl_tools::checkpoint::example::output::container, step_i, batch_i, action_i));
                num_values += 1;
                rl_tools::utils::assert_exit(device, !rl_tools::math::is_nan(device.math, acc), "output is nan");
                if(batch_i == 0 && step_i == TEST_SEQUENCE_LENGTH_ACTUAL-1){
                    p_output->action[action_i] = rl_tools::get(device, output, 0, action_i);
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
    static_assert(RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM == OUTPUT_DIM);
    rl_tools::inference::applications::l2f::Observation<SPEC> observation;
    for (TI dim_i = 0; dim_i < 3; dim_i++){
        observation.position[dim_i] = c_observation->position[dim_i];
        observation.orientation[dim_i] = c_observation->orientation[dim_i];
        observation.linear_velocity[dim_i] = c_observation->linear_velocity[dim_i];
        observation.angular_velocity[dim_i] = c_observation->angular_velocity[dim_i];
    }
    observation.orientation[3] = c_observation->orientation[3];
    for (TI action_i=0; action_i < OUTPUT_DIM; action_i++){
        observation.previous_action[action_i] = c_observation->previous_action[action_i];
    }
    rl_tools::inference::applications::l2f::Action<SPEC> action;
    auto status = rl_tools::control(device, executor, nanoseconds, rl_tools_inference_applications_l2f_policy, observation, action, rng);
    for (TI action_i=0; action_i < OUTPUT_DIM; action_i++){
        c_action->action[action_i] = action.action[action_i];
    }
    return rl_tools::convert(status);
}
