namespace layer_in_c::rl::algorithms{
    namespace ppo{
        template<typename T, typename TI>
        struct DefaultParameters {
            static constexpr T GAMMA = 0.9;
            static constexpr T LAMBDA = 0.95;
            static constexpr T EPSILON_CLIP = 0.2;
            static constexpr T INITIAL_ACTION_STD = 0.5;
            static constexpr bool LEARN_ACTION_STD = true;
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
            static constexpr T ADVANTAGE_EPSILON = 1e-8;
            static constexpr bool NORMALIZE_ADVANTAGE = true;
            static constexpr TI N_WARMUP_STEPS_CRITIC = 0;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 0;
            static constexpr TI N_EPOCHS = 20;
            static constexpr bool IGNORE_TERMINATION = false; // ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
        };

        template<
                typename T_T,
                typename T_TI,
                typename T_ENVIRONMENT,
                typename T_ACTOR_NETWORK_TYPE,
                typename T_CRITIC_NETWORK_TYPE,
                typename T_PARAMETERS = DefaultParameters<T_T, T_TI>
        >
        struct Specification {
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            using ACTOR_NETWORK_TYPE = T_ACTOR_NETWORK_TYPE;
            using CRITIC_NETWORK_TYPE = T_CRITIC_NETWORK_TYPE;
            static constexpr TI BATCH_SIZE = ACTOR_NETWORK_TYPE::SPEC::BATCH_SIZE;
            using PARAMETERS = T_PARAMETERS;

            static_assert(ACTOR_NETWORK_TYPE::SPEC::BATCH_SIZE == CRITIC_NETWORK_TYPE::SPEC::BATCH_SIZE);
            static_assert(ACTOR_NETWORK_TYPE::INPUT_DIM == ENVIRONMENT::OBSERVATION_DIM);
            static_assert(CRITIC_NETWORK_TYPE::INPUT_DIM == ENVIRONMENT::OBSERVATION_DIM);
            static_assert(ACTOR_NETWORK_TYPE::OUTPUT_DIM == ENVIRONMENT::ACTION_DIM);
            static_assert(CRITIC_NETWORK_TYPE::OUTPUT_DIM == 1);
        };
    }

    template<typename T_SPEC>
    struct PPO {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        typename SPEC::ACTOR_NETWORK_TYPE actor;
        typename SPEC::CRITIC_NETWORK_TYPE critic;
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        bool initialized = false;
#endif

    };
}
