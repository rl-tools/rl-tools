#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H

// Please include the file containing the environments operations before including this file
#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>

namespace lic = layer_in_c;


/* requirements
- Multiple environments
- Batched action inference


*/

namespace layer_in_c::rl::components::off_policy_runner {
    template<typename T>
    struct DefaultParameters{
        static constexpr T EXPLORATION_NOISE = 0.1;
    };
    template<typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_N_ENVIRONMENTS, T_TI T_REPLAY_BUFFER_CAPACITY, T_TI T_STEP_LIMIT, typename T_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT =  T_ENVIRONMENT;
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


    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE>
    struct BatchSpecification {
        using SPEC = T_SPEC;
        static constexpr typename SPEC::TI BATCH_SIZE = T_BATCH_SIZE;
    };

    template <typename T_SPEC>
    struct Batch{
        using SPEC = typename T_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        static constexpr TI BATCH_SIZE = T_SPEC::BATCH_SIZE;
        static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        Matrix<matrix::Specification<T, TI, BATCH_SIZE, 2*OBSERVATION_DIM + ACTION_DIM>> observations_action_next_observations;

        template<typename SPEC::TI DIM>
        using OANO_VIEW = typename decltype(observations_action_next_observations)::template VIEW<BATCH_SIZE, DIM>;

        OANO_VIEW<OBSERVATION_DIM> observations;
        OANO_VIEW<ACTION_DIM> actions;
        OANO_VIEW<OBSERVATION_DIM + ACTION_DIM> observations_and_actions;
        OANO_VIEW<OBSERVATION_DIM> next_observations;

        Matrix<matrix::Specification<T, TI, 1, BATCH_SIZE>> rewards;
        Matrix<matrix::Specification<bool, TI, 1, BATCH_SIZE>> terminated;
        Matrix<matrix::Specification<bool, TI, 1, BATCH_SIZE>> truncated;
    };

}

namespace layer_in_c::rl::components{
    template<typename T_SPEC>
    struct OffPolicyRunner {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using REPLAY_BUFFER_SPEC = replay_buffer::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ENVIRONMENT::OBSERVATION_DIM, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::REPLAY_BUFFER_CAPACITY>;
        using REPLAY_BUFFER_TYPE = ReplayBuffer<REPLAY_BUFFER_SPEC>;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        static constexpr TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
//        using POLICY_EVAL_BUFFERS = typename POLICY::template Buffers<N_ENVIRONMENTS>;

        off_policy_runner::Buffers<SPEC> buffers;

        ENVIRONMENT envs[N_ENVIRONMENTS];
        REPLAY_BUFFER_TYPE replay_buffers[N_ENVIRONMENTS];
        Matrix<matrix::Specification<typename SPEC::ENVIRONMENT::State, TI, 1, N_ENVIRONMENTS>> states;
        Matrix<matrix::Specification<T, TI, 1, N_ENVIRONMENTS>> episode_return;
        Matrix<matrix::Specification<TI, TI, 1, N_ENVIRONMENTS>> episode_step;
        Matrix<matrix::Specification<bool, TI, 1, N_ENVIRONMENTS>> truncated; // init to true !!
        TI step = 0;
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}

#endif
