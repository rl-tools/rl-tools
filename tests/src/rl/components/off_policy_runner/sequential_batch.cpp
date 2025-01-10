#include <rl_tools/operations/cpu.h>
#include "dummy_environment.h"
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
namespace rlt = rl_tools;

#include <gtest/gtest.h>

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = DEVICE::index_t;

using ENVIRONMENT_SPEC = rlt::rl::environments::dummy::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::Dummy<ENVIRONMENT_SPEC>;
using EXPLORATION_POLICY_SPEC = rlt::nn_models::random_uniform::Specification<T, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, rlt::nn_models::random_uniform::Range::MINUS_ONE_TO_ONE>;
using EXPLORATION_POLICY = rlt::nn_models::RandomUniform<EXPLORATION_POLICY_SPEC>;
using POLICIES = rl_tools::utils::Tuple<TI, EXPLORATION_POLICY>;
struct OFF_POLICY_RUNNER_PARAMETERS{
    static constexpr TI N_ENVIRONMENTS = 2;
    static constexpr bool ASYMMETRIC_OBSERVATIONS = !rl_tools::utils::typing::is_same_v<typename ENVIRONMENT::Observation, typename ENVIRONMENT::ObservationPrivileged>;
    static constexpr TI REPLAY_BUFFER_CAPACITY = 1000;
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
    static constexpr bool COLLECT_EPISODE_STATS = false;
    static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
    static constexpr bool SAMPLE_PARAMETERS = true;
};
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, POLICIES, OFF_POLICY_RUNNER_PARAMETERS>;
using OFF_POLICY_RUNNER = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;

constexpr TI SEQUENCE_LENGTH = 10;
constexpr TI BATCH_SIZE = 1;
struct SEQUENTIAL_BATCH_PARAMETERS_INCLUSIVE{
    static constexpr bool INCLUDE_FIRST_STEP_IN_TARGETS = true;
    static constexpr bool ALWAYS_SAMPLE_FROM_INITIAL_STATE = false;
    static constexpr bool RANDOM_SEQ_LENGTH = true;
    static constexpr bool ENABLE_NOMINAL_SEQUENCE_LENGTH_PROBABILITY = true;
    static constexpr T NOMINAL_SEQUENCE_LENGTH_PROBABILITY = 0.5;
};
struct SEQUENTIAL_BATCH_PARAMETERS_EXCLUSIVE: SEQUENTIAL_BATCH_PARAMETERS_INCLUSIVE{
    static constexpr bool INCLUDE_FIRST_STEP_IN_TARGETS = false;
};
using SEQUENTIAL_BATCH_SPEC_INCLUSIVE = rlt::rl::components::off_policy_runner::SequentialBatchSpecification<OFF_POLICY_RUNNER_SPEC, 10, BATCH_SIZE, SEQUENTIAL_BATCH_PARAMETERS_INCLUSIVE>;
using SEQUENTIAL_BATCH_SPEC_EXCLUSIVE = rlt::rl::components::off_policy_runner::SequentialBatchSpecification<OFF_POLICY_RUNNER_SPEC, 10, BATCH_SIZE, SEQUENTIAL_BATCH_PARAMETERS_EXCLUSIVE>;
using SEQUENTIAL_BATCH_INCLUSIVE = rlt::rl::components::off_policy_runner::SequentialBatch<SEQUENTIAL_BATCH_SPEC_INCLUSIVE>;
using SEQUENTIAL_BATCH_EXCLUSIVE = rlt::rl::components::off_policy_runner::SequentialBatch<SEQUENTIAL_BATCH_SPEC_EXCLUSIVE>;

enum class SequenceState{
    FIRST_STEP,
    IN_SEQUENCE,
    PADDING,
    DEAD
};

template <typename DEVICE, typename BATCH>
void check_batch(DEVICE& device, BATCH& batch){
    constexpr bool EXCLUSIVE = !BATCH::SPEC::PARAMETERS::INCLUDE_FIRST_STEP_IN_TARGETS;
    for (TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
        SequenceState state = SequenceState::FIRST_STEP;
        for (TI seq_step_i = 0; seq_step_i < SEQUENCE_LENGTH + 1; seq_step_i++){
            T observation = rlt::get(device, batch.observations_current, seq_step_i, batch_step_i, 0);
            bool reset = seq_step_i < SEQUENCE_LENGTH ? rlt::get(device, batch.reset, seq_step_i, batch_step_i, 0) : true;
            bool next_reset = rlt::get(device, batch.next_reset_base, seq_step_i, batch_step_i, 0);
            bool final_step_mask = seq_step_i < SEQUENCE_LENGTH ? rlt::get(device, batch.final_step_mask, seq_step_i, batch_step_i, 0) : false;
            bool next_final_step_mask = rlt::get(device, batch.next_final_step_mask_base, seq_step_i, batch_step_i, 0);
            switch(state){
            case SequenceState::FIRST_STEP:
                    rlt::utils::assert_exit(device, reset, "reset");
                    rlt::utils::assert_exit(device, next_reset, "next_reset");
                    rlt::utils::assert_exit(device, !next_final_step_mask, "!next_final_step_mask");
                    break;
                case SequenceState::IN_SEQUENCE:
                    rlt::utils::assert_exit(device, !reset, "!reset");
                    rlt::utils::assert_exit(device, (EXCLUSIVE && (seq_step_i == 1 && next_reset)) || !next_reset, "!next_reset");
                    rlt::utils::assert_exit(device, !next_final_step_mask, "!next_final_step_mask");
                    break;
                case SequenceState::PADDING:
                    rlt::utils::assert_exit(device, reset, "reset");
                    rlt::utils::assert_exit(device, (EXCLUSIVE && (seq_step_i == 1 && next_reset)) || !next_reset, "!next_reset");
                    rlt::utils::assert_exit(device, next_final_step_mask, "next_final_step_mask");
                    break;
            case SequenceState::DEAD:
                    std::cout << "Reached DEAD state" << std::endl;
                    rlt::utils::assert_exit(device, reset, "reset");
                    rlt::utils::assert_exit(device, next_reset, "next_reset");
                    rlt::utils::assert_exit(device, !final_step_mask, "!final_step_mask");
                    rlt::utils::assert_exit(device, !next_final_step_mask, "!next_final_step_mask");
                    break;
            }
            SequenceState next_state;
            switch(state){
                case SequenceState::FIRST_STEP:
                case SequenceState::IN_SEQUENCE:
                    if(seq_step_i == SEQUENCE_LENGTH-1){
                        next_state = SequenceState::PADDING;
                    }
                    else{
                        next_state = final_step_mask ? SequenceState::PADDING : SequenceState::IN_SEQUENCE;
                    }
                    break;
            case SequenceState::PADDING:
                    if(seq_step_i == SEQUENCE_LENGTH - 1){
                        next_state = SequenceState::DEAD;
                    }
                    else{
                        next_state = SequenceState::FIRST_STEP;
                    }
                    break;
                case SequenceState::DEAD:
                    break;
            }
            state = next_state;
        }
    }

}

TEST(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_SQUENTIAL_BATCH, TEST){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    OFF_POLICY_RUNNER off_policy_runner;
    SEQUENTIAL_BATCH_INCLUSIVE inclusive_batch;
    SEQUENTIAL_BATCH_EXCLUSIVE exclusive_batch;
    EXPLORATION_POLICY policy;
    EXPLORATION_POLICY::Buffer<> policy_buffer;
    rlt::malloc(device, off_policy_runner);
    rlt::malloc(device, inclusive_batch);
    rlt::malloc(device, exclusive_batch);
    rlt::malloc(device, policy_buffer);
    rlt::init(device, off_policy_runner);
    for(TI step_i = 0; step_i < 1000; step_i++){
        rlt::step<0>(device, off_policy_runner, policy, policy_buffer, rng);
    }
    for(TI batch_i = 0; batch_i < 100000; batch_i++){
        rlt::gather_batch(device, off_policy_runner, inclusive_batch, rng);
        check_batch(device, inclusive_batch);
    }
    for(TI batch_i = 0; batch_i < 100000; batch_i++){
        rlt::gather_batch(device, off_policy_runner, exclusive_batch, rng);
        check_batch(device, exclusive_batch);
    }
}
