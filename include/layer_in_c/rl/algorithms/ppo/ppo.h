namespace layer_in_c::rl::algorithms{
    namespace ppo{
        template<typename T, typename TI>
        struct DefaultParameters {
            static constexpr T GAMMA = 0.99;
            static constexpr TI ACTOR_BATCH_SIZE = 32;
            static constexpr TI CRITIC_BATCH_SIZE = 32;
            static constexpr TI N_WARMUP_STEPS_CRITIC = 0;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 0;
            static constexpr bool IGNORE_TERMINATION = false; // ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
        };

        template<
                typename T_T,
                typename T_TI,
                typename T_ENVIRONMENT,
                typename T_ACTOR_NETWORK_TYPE,
                typename T_CRITIC_NETWORK_TYPE,
                typename T_PARAMETERS
        >
        struct Specification {
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            using ACTOR_NETWORK_TYPE = T_ACTOR_NETWORK_TYPE;
            using CRITIC_NETWORK_TYPE = T_CRITIC_NETWORK_TYPE;
            using PARAMETERS = T_PARAMETERS;
        };
    }

    template<typename T_SPEC>
    struct PPO {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        typename SPEC::ACTOR_NETWORK_TYPE actor;
        typename SPEC::CRITIC_NETWORK_TYPE critic;

    };
}
