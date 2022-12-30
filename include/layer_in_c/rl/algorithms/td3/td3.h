#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3
#define LAYER_IN_C_RL_ALGORITHMS_TD3


namespace layer_in_c::rl::algorithms::td3 {
    // todo remove namespace assignment
    namespace lic = layer_in_c;
    template<typename T>
    struct DefaultTD3Parameters {
        static constexpr T GAMMA = 0.99;
        static constexpr size_t ACTOR_BATCH_SIZE = 32;
        static constexpr size_t CRITIC_BATCH_SIZE = 32;
        static constexpr T ACTOR_POLYAK = 1.0 - 0.005;
        static constexpr T CRITIC_POLYAK = 1.0 - 0.005;
        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
    };

    template<typename T, size_t T_LAYER_1_DIM, size_t T_LAYER_2_DIM, lic::nn::activation_functions::ActivationFunction FN, typename T_OPTIMIZER_PARAMETERS>
    struct ActorNetworkSpecification {
        static constexpr size_t LAYER_1_DIM = T_LAYER_1_DIM;
        static constexpr size_t LAYER_2_DIM = T_LAYER_2_DIM;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN = FN;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = FN;
        typedef T_OPTIMIZER_PARAMETERS OPTIMIZER_PARAMETERS;
    };

    template<typename T, size_t T_LAYER_1_DIM, size_t T_LAYER_2_DIM, lic::nn::activation_functions::ActivationFunction FN, typename T_OPTIMIZER_PARAMETERS>
    struct CriticNetworkSpecification {
        static constexpr size_t LAYER_1_DIM = T_LAYER_1_DIM;
        static constexpr size_t LAYER_2_DIM = T_LAYER_2_DIM;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN = FN;
        static constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = FN;
        typedef T_OPTIMIZER_PARAMETERS OPTIMIZER_PARAMETERS;
    };


    template<
            typename T_NN_DEVICE,
            typename T_T,
            typename T_ENVIRONMENT,
            typename T_ACTOR_SPEC,
            typename T_CRITIC_SPEC,
            typename T_PARAMETERS
    >
    struct ActorCriticSpecification {
        typedef T_NN_DEVICE NN_DEVICE;
        typedef T_T T;
        typedef T_ENVIRONMENT ENVIRONMENT;
        typedef T_ACTOR_SPEC ACTOR_SPEC;
        typedef T_CRITIC_SPEC CRITIC_SPEC;
        typedef T_PARAMETERS PARAMETERS;
    };

    template<typename T_DEVICE, typename T_SPEC>
    struct ActorCritic {
        using DEVICE = T_DEVICE;
        using SPEC = T_SPEC;
        using NN_DEVICE = typename SPEC::NN_DEVICE;
        typedef typename SPEC::T T;
        static constexpr lic::nn::activation_functions::ActivationFunction ACTOR_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
//        static constexpr lic::nn::activation_functions::ActivationFunction ACTOR_ACTIVATION_FUNCTION = lic::nn::activation_functions::SIGMOID_STRETCHED;
//        static constexpr lic::nn::activation_functions::ActivationFunction ACTOR_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;

        using ACTOR_NETWORK_STRUCTURE_SPEC = lic::nn_models::three_layer_fc::StructureSpecification<
                typename SPEC::T,
                SPEC::ENVIRONMENT::OBSERVATION_DIM,
                SPEC::ACTOR_SPEC::LAYER_1_DIM, SPEC::ACTOR_SPEC::LAYER_1_FN,
                SPEC::ACTOR_SPEC::LAYER_2_DIM, SPEC::ACTOR_SPEC::LAYER_2_FN,
                SPEC::ENVIRONMENT::ACTION_DIM, ACTOR_ACTIVATION_FUNCTION>;

        using ACTOR_NETWORK_SPEC = lic::nn_models::three_layer_fc::AdamSpecification<NN_DEVICE, ACTOR_NETWORK_STRUCTURE_SPEC, typename SPEC::ACTOR_SPEC::OPTIMIZER_PARAMETERS>;
        using ACTOR_NETWORK_TYPE = lic::nn_models::three_layer_fc::NeuralNetworkAdam<NN_DEVICE, ACTOR_NETWORK_SPEC>;

        using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::three_layer_fc::InferenceSpecification<NN_DEVICE, ACTOR_NETWORK_STRUCTURE_SPEC>;
        using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::three_layer_fc::NeuralNetwork<NN_DEVICE , ACTOR_TARGET_NETWORK_SPEC>;

        static constexpr size_t CRITIC_INPUT_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM + SPEC::ENVIRONMENT::ACTION_DIM;
        using CRITIC_NETWORK_STRUCTURE_SPEC = layer_in_c::nn_models::three_layer_fc::StructureSpecification<T, CRITIC_INPUT_DIM,
                SPEC::CRITIC_SPEC::LAYER_1_DIM, SPEC::CRITIC_SPEC::LAYER_1_FN,
                SPEC::CRITIC_SPEC::LAYER_2_DIM, SPEC::CRITIC_SPEC::LAYER_2_FN,
                1, layer_in_c::nn::activation_functions::IDENTITY>;

        using CRITIC_NETWORK_SPEC = lic::nn_models::three_layer_fc::AdamSpecification<NN_DEVICE, CRITIC_NETWORK_STRUCTURE_SPEC, typename SPEC::CRITIC_SPEC::OPTIMIZER_PARAMETERS>;
        using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::three_layer_fc::NeuralNetworkAdam<NN_DEVICE, CRITIC_NETWORK_SPEC>;

        using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::three_layer_fc::InferenceSpecification<NN_DEVICE, CRITIC_NETWORK_STRUCTURE_SPEC>;
        using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::three_layer_fc::NeuralNetwork<NN_DEVICE, CRITIC_TARGET_NETWORK_SPEC>;

        ACTOR_NETWORK_TYPE actor;
        ACTOR_TARGET_NETWORK_TYPE actor_target;

        CRITIC_NETWORK_TYPE critic_1;
        CRITIC_NETWORK_TYPE critic_2;
        CRITIC_TARGET_NETWORK_TYPE critic_target_1;
        CRITIC_TARGET_NETWORK_TYPE critic_target_2;
    };
}



#endif