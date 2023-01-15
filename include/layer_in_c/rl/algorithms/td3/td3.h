#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3
#define LAYER_IN_C_RL_ALGORITHMS_TD3


namespace layer_in_c::rl::algorithms::td3 {
    // todo remove namespace assignment
    namespace lic = layer_in_c;
    template<typename T, typename TI>
    struct DefaultParameters {
        static constexpr T GAMMA = 0.99;
        static constexpr TI ACTOR_BATCH_SIZE = 32;
        static constexpr TI CRITIC_BATCH_SIZE = 32;
        static constexpr T ACTOR_POLYAK = 1.0 - 0.005;
        static constexpr T CRITIC_POLYAK = 1.0 - 0.005;
        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
    };

    template<
        typename T_T,
        typename T_ENVIRONMENT,
        typename T_ACTOR_NETWORK_TYPE,
        typename T_ACTOR_TARGET_NETWORK_TYPE,
        typename T_CRITIC_NETWORK_TYPE,
        typename T_CRITIC_TARGET_NETWORK_TYPE,
        typename T_PARAMETERS
    >
    struct Specification {
        using T = T_T;
        using ENVIRONMENT = T_ENVIRONMENT;
        using ACTOR_NETWORK_TYPE = T_ACTOR_NETWORK_TYPE;
        using ACTOR_TARGET_NETWORK_TYPE = T_ACTOR_TARGET_NETWORK_TYPE;
        using CRITIC_NETWORK_TYPE = T_CRITIC_NETWORK_TYPE;
        using CRITIC_TARGET_NETWORK_TYPE = T_CRITIC_TARGET_NETWORK_TYPE;
        using PARAMETERS = T_PARAMETERS;
    };

    template<typename T_SPEC>
    struct ActorCritic {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;

        typename SPEC::ACTOR_NETWORK_TYPE actor;
        typename SPEC::ACTOR_TARGET_NETWORK_TYPE actor_target;

        typename SPEC::CRITIC_NETWORK_TYPE critic_1;
        typename SPEC::CRITIC_NETWORK_TYPE critic_2;
        typename SPEC::CRITIC_TARGET_NETWORK_TYPE critic_target_1;
        typename SPEC::CRITIC_TARGET_NETWORK_TYPE critic_target_2;
    };
}



#endif