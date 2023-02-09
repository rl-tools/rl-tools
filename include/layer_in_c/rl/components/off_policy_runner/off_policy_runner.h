#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H

// Please include the file containing the environments operations before including this file
#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>

namespace lic = layer_in_c;

namespace layer_in_c::rl::components::off_policy_runner {
    template<typename T>
    struct DefaultParameters{
        static constexpr T EXPLORATION_NOISE = 0.1;
    };
    template<typename T_T, typename T_TI, typename T_ENVIRONMENT, typename T_POLICY, T_TI T_N_ENVIRONMENTS, T_TI T_REPLAY_BUFFER_CAPACITY, T_TI T_STEP_LIMIT, typename T_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT =  T_ENVIRONMENT;
        using POLICY = T_POLICY;
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        static constexpr TI REPLAY_BUFFER_CAPACITY = T_REPLAY_BUFFER_CAPACITY;
        static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
        typedef T_PARAMETERS PARAMETERS;
    };

    template<typename SPEC>
    struct Buffers{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> observations;
        Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>> actions;
        Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> next_observations;
    };

    template<typename SPEC, typename SPEC::TI T_BATCH_SIZE>
    struct Batch{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        Matrix<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> observations;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> actions;
        Matrix<matrix::Specification<T, TI, 1, BATCH_SIZE>> rewards;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> next_observations;
        Matrix<matrix::Specification<bool, TI, 1, BATCH_SIZE>> terminated;
        Matrix<matrix::Specification<bool, TI, 1, BATCH_SIZE>> truncated;
    };

}

namespace layer_in_c::rl::components{
    template<typename T_SPEC>
    struct OffPolicyRunner {
        using SPEC = T_SPEC;
        using ReplayBufferSpec = replay_buffer::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ENVIRONMENT::OBSERVATION_DIM, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::REPLAY_BUFFER_CAPACITY>;
        using POLICY = typename SPEC::POLICY;
        static constexpr typename SPEC::TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
//        using POLICY_EVAL_BUFFERS = typename POLICY::template Buffers<N_ENVIRONMENTS>;

        off_policy_runner::Buffers<SPEC> buffers;
        typename POLICY::template Buffers<N_ENVIRONMENTS> policy_eval_buffers;

        struct State{
            typename SPEC::ENVIRONMENT env;
            ReplayBuffer<ReplayBufferSpec> replay_buffer;
            typename SPEC::ENVIRONMENT::State state;
            bool truncated = true;
            typename SPEC::TI episode_step = 0;
            typename SPEC::T episode_return = 0;
        };
        State states[N_ENVIRONMENTS];
        typename SPEC::TI step = 0;
    };
}

#endif
