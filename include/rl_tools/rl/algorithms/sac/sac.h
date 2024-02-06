#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_SAC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_SAC_H
//#include "../../../nn_models/output_view/model.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::sac {
    template<typename T, typename TI, TI ACTION_DIM=1>
    struct DefaultParameters {
        static constexpr T GAMMA = 0.99;
        static constexpr T ALPHA = 0.5;
        static constexpr TI ACTOR_BATCH_SIZE = 32;
        static constexpr TI CRITIC_BATCH_SIZE = 32;
        static constexpr TI N_WARMUP_STEPS_CRITIC = 0;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 0;
        static constexpr TI CRITIC_TRAINING_INTERVAL = 1;
        static constexpr TI ACTOR_TRAINING_INTERVAL = 1;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1;
        static constexpr T ACTOR_POLYAK = 1.0 - 0.005;
        static constexpr T CRITIC_POLYAK = 1.0 - 0.005;
//        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
//        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
        static constexpr bool IGNORE_TERMINATION = false; // ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
        static constexpr T TARGET_ENTROPY = -((T)ACTION_DIM);
        static constexpr bool ADAPTIVE_ALPHA = true;
    };

    template<
        typename T_T,
        typename T_TI,
        typename T_ENVIRONMENT,
        typename T_ACTOR_NETWORK_TYPE,
        typename T_CRITIC_NETWORK_TYPE,
        typename T_CRITIC_TARGET_NETWORK_TYPE,
        typename T_ALPHA_PARAMETER_TYPE,
        typename T_ACTOR_OPTIMIZER,
        typename T_CRITIC_OPTIMIZER,
        typename T_ALPHA_OPTIMIZER,
        typename T_PARAMETERS,
        typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag
    >
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT = T_ENVIRONMENT;
        using ACTOR_NETWORK_TYPE = T_ACTOR_NETWORK_TYPE;
        using CRITIC_NETWORK_TYPE = T_CRITIC_NETWORK_TYPE;
        using CRITIC_TARGET_NETWORK_TYPE = T_CRITIC_TARGET_NETWORK_TYPE;
        using ALPHA_PARAMETER_TYPE = T_ALPHA_PARAMETER_TYPE;
        using ACTOR_OPTIMIZER = T_ACTOR_OPTIMIZER;
        using CRITIC_OPTIMIZER = T_CRITIC_OPTIMIZER;
        using ALPHA_OPTIMIZER = T_ALPHA_OPTIMIZER;
        using PARAMETERS = T_PARAMETERS;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

    template<typename T_SPEC, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
    struct ActorTrainingBuffers{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr TI BATCH_SIZE = SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
        static constexpr TI ACTOR_INPUT_DIM = SPEC::ACTOR_NETWORK_TYPE::INPUT_DIM;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static constexpr TI CRITIC_OBSERVATION_DIM = SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM - SPEC::ENVIRONMENT::ACTION_DIM;

        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>> state_action_value_input;
        template<typename SPEC::TI DIM>
        using STATE_ACTION_VALUE_VIEW = typename decltype(state_action_value_input)::template VIEW<BATCH_SIZE, DIM>;
        STATE_ACTION_VALUE_VIEW<CRITIC_OBSERVATION_DIM> observations;
        STATE_ACTION_VALUE_VIEW<ACTION_DIM> actions;
//        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> state_action_value;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> d_output;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>> d_critic_1_input, d_critic_2_input;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> d_critic_action_input;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> action_sample, action_noise;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM * 2>> d_actor_output;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTOR_INPUT_DIM>> d_actor_input;

    };
    template<typename T_SPEC, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
    struct CriticTrainingBuffers{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr TI BATCH_SIZE = SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static constexpr TI CRITIC_OBSERVATION_DIM = SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM - ACTION_DIM;


        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM*2>> next_state_action_value_input_full;
        template<typename SPEC::TI DIM>
        using NEXT_STATE_ACTION_VALUE_VIEW = typename decltype(next_state_action_value_input_full)::template VIEW<BATCH_SIZE, DIM>;
        NEXT_STATE_ACTION_VALUE_VIEW<CRITIC_OBSERVATION_DIM + ACTION_DIM> next_state_action_value_input;
        NEXT_STATE_ACTION_VALUE_VIEW<CRITIC_OBSERVATION_DIM> next_observations;
        NEXT_STATE_ACTION_VALUE_VIEW<ACTION_DIM*2> next_actions_distribution;
        NEXT_STATE_ACTION_VALUE_VIEW<ACTION_DIM> next_actions_mean;
        NEXT_STATE_ACTION_VALUE_VIEW<ACTION_DIM> next_actions_log_std;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> action_value;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> target_action_value;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> next_state_action_value_critic_1;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> next_state_action_value_critic_2;

        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>> d_input;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> d_output;
        typename CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> action_log_probs;
    };

    template<typename T_SPEC>
    struct ActorCritic {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

//        T target_next_action_noise_std = SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD;
//        T target_next_action_noise_clip = SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP;

        typename SPEC::ACTOR_NETWORK_TYPE actor;
//        using ACTOR_VIEW = nn_models::output_view::MODEL<nn_models::output_view::MODEL_VIEW_SPEC<TI, typename SPEC::ACTOR_NETWORK_TYPE, 0, SPEC::ENVIRONMENT::ACTION_DIM>>;
//        ACTOR_VIEW actor_view;

        typename SPEC::CRITIC_NETWORK_TYPE critic_1;
        typename SPEC::CRITIC_NETWORK_TYPE critic_2;
        typename SPEC::CRITIC_TARGET_NETWORK_TYPE critic_target_1;
        typename SPEC::CRITIC_TARGET_NETWORK_TYPE critic_target_2;
        using ALPHA_CONTAINER = typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, 1>>;
        using ALPHA_PARAMETER_SPEC = typename SPEC::ALPHA_PARAMETER_TYPE::template spec<ALPHA_CONTAINER, nn::parameters::categories::Biases, nn::parameters::groups::Normal>;
        typename SPEC::ALPHA_PARAMETER_TYPE::template instance<ALPHA_PARAMETER_SPEC> log_alpha;


        typename SPEC::ACTOR_OPTIMIZER actor_optimizer;
        typename SPEC::CRITIC_OPTIMIZER critic_optimizers[2];
        typename SPEC::ALPHA_OPTIMIZER alpha_optimizer;
//        ActorCritic(): actor_view(actor){};
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END



#endif