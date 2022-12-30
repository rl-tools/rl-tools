#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3
#define LAYER_IN_C_RL_ALGORITHMS_TD3

namespace lic = layer_in_c;

namespace layer_in_c::rl::algorithms::td3 {
    template<typename T>
    struct DefaultTD3Parameters {
        static constexpr T GAMMA = 0.99;
        static constexpr uint32_t ACTOR_BATCH_SIZE = 32;
        static constexpr uint32_t CRITIC_BATCH_SIZE = 32;
        static constexpr T ACTOR_POLYAK = 1.0 - 0.005;
        static constexpr T CRITIC_POLYAK = 1.0 - 0.005;
        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
    };

    template<typename T, int T_LAYER_1_DIM, int T_LAYER_2_DIM, lic::nn::activation_functions::ActivationFunction FN, typename T_OPTIMIZER_PARAMETERS>
    struct ActorNetworkSpecification {
        static constexpr int LAYER_1_DIM = T_LAYER_1_DIM;
        static constexpr int LAYER_2_DIM = T_LAYER_2_DIM;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN = FN;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = FN;
        typedef T_OPTIMIZER_PARAMETERS OPTIMIZER_PARAMETERS;
    };

    template<typename T, int T_LAYER_1_DIM, int T_LAYER_2_DIM, lic::nn::activation_functions::ActivationFunction FN, typename T_OPTIMIZER_PARAMETERS>
    struct CriticNetworkSpecification {
        static constexpr int LAYER_1_DIM = T_LAYER_1_DIM;
        static constexpr int LAYER_2_DIM = T_LAYER_2_DIM;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN = FN;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = FN;
        typedef T_OPTIMIZER_PARAMETERS OPTIMIZER_PARAMETERS;
    };


    template<
            typename T_T,
            typename T_ENVIRONMENT,
            typename T_ACTOR_SPEC,
            typename T_CRITIC_SPEC,
            typename T_PARAMETERS
    >
    struct ActorCriticSpecification {
        typedef T_T T;
        typedef T_ENVIRONMENT ENVIRONMENT;
        typedef T_ACTOR_SPEC ACTOR_SPEC;
        typedef T_CRITIC_SPEC CRITIC_SPEC;
        typedef T_PARAMETERS PARAMETERS;
    };

    template<typename DEVICE, typename T_SPEC>
    struct ActorCritic {
        typedef T_SPEC SPEC;
        typedef typename SPEC::T T;
        static constexpr lic::nn::activation_functions::ActivationFunction ACTOR_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
//        static constexpr lic::nn::activation_functions::ActivationFunction ACTOR_ACTIVATION_FUNCTION = lic::nn::activation_functions::SIGMOID_STRETCHED;
//        static constexpr lic::nn::activation_functions::ActivationFunction ACTOR_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;

        typedef lic::nn_models::three_layer_fc::StructureSpecification<
                typename SPEC::T,
                SPEC::ENVIRONMENT::OBSERVATION_DIM,
                SPEC::ACTOR_SPEC::LAYER_1_DIM, SPEC::ACTOR_SPEC::LAYER_1_FN,
                SPEC::ACTOR_SPEC::LAYER_2_DIM, SPEC::ACTOR_SPEC::LAYER_2_FN,
                SPEC::ENVIRONMENT::ACTION_DIM, ACTOR_ACTIVATION_FUNCTION> ACTOR_NETWORK_STRUCTURE_SPEC;

        typedef lic::nn_models::three_layer_fc::AdamSpecification<DEVICE, ACTOR_NETWORK_STRUCTURE_SPEC, typename SPEC::ACTOR_SPEC::OPTIMIZER_PARAMETERS> ACTOR_NETWORK_SPEC;
        typedef layer_in_c::nn_models::three_layer_fc::NeuralNetworkAdam<DEVICE, ACTOR_NETWORK_SPEC> ACTOR_NETWORK_TYPE;

        typedef lic::nn_models::three_layer_fc::InferenceSpecification<DEVICE, ACTOR_NETWORK_STRUCTURE_SPEC> ACTOR_TARGET_NETWORK_SPEC;
        typedef layer_in_c::nn_models::three_layer_fc::NeuralNetwork<DEVICE, ACTOR_TARGET_NETWORK_SPEC> ACTOR_TARGET_NETWORK_TYPE;

        static constexpr int CRITIC_INPUT_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM + SPEC::ENVIRONMENT::ACTION_DIM;
        typedef layer_in_c::nn_models::three_layer_fc::StructureSpecification<T, CRITIC_INPUT_DIM,
                SPEC::CRITIC_SPEC::LAYER_1_DIM, SPEC::CRITIC_SPEC::LAYER_1_FN,
                SPEC::CRITIC_SPEC::LAYER_2_DIM, SPEC::CRITIC_SPEC::LAYER_2_FN,
                1, layer_in_c::nn::activation_functions::IDENTITY> CRITIC_NETWORK_STRUCTURE_SPEC;

        typedef lic::nn_models::three_layer_fc::AdamSpecification<DEVICE, CRITIC_NETWORK_STRUCTURE_SPEC, typename SPEC::CRITIC_SPEC::OPTIMIZER_PARAMETERS> CRITIC_NETWORK_SPEC;
        typedef layer_in_c::nn_models::three_layer_fc::NeuralNetworkAdam<DEVICE, CRITIC_NETWORK_SPEC> CRITIC_NETWORK_TYPE;

        typedef layer_in_c::nn_models::three_layer_fc::InferenceSpecification<DEVICE, CRITIC_NETWORK_STRUCTURE_SPEC> CRITIC_TARGET_NETWORK_SPEC;
        typedef layer_in_c::nn_models::three_layer_fc::NeuralNetwork<DEVICE, CRITIC_TARGET_NETWORK_SPEC> CRITIC_TARGET_NETWORK_TYPE;

        ACTOR_NETWORK_TYPE actor;
        ACTOR_TARGET_NETWORK_TYPE actor_target;

        CRITIC_NETWORK_TYPE critic_1;
        CRITIC_NETWORK_TYPE critic_2;
        CRITIC_TARGET_NETWORK_TYPE critic_target_1;
        CRITIC_TARGET_NETWORK_TYPE critic_target_2;
    };
}



#endif