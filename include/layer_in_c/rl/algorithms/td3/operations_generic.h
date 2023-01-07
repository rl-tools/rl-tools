#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ALGORITHMS_TD3_OPERATIONS_GENERIC_H

#include "td3.h"

#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>
#include <layer_in_c/nn/nn.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/utils/generic/polyak.h>
#include <layer_in_c/math/operations_generic.h>

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void update_target_layer(nn::layers::dense::Layer<DEVICE, SPEC>& target, const nn::layers::dense::Layer<DEVICE, SPEC>& source, typename SPEC::T polyak) {
        utils::polyak::update_matrix<DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(DEVICE(), target.weights, source.weights, polyak);
        utils::polyak::update       <DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM                 >(DEVICE(), target.biases , source.biases , polyak);
    }
    template<typename T, typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void update_target_network(nn_models::mlp::NeuralNetwork<DEVICE, TARGET_SPEC>& target, const nn_models::mlp::NeuralNetwork<DEVICE, SOURCE_SPEC>& source, T polyak) {
        using TargetNetworkType = typename utils::typing::remove_reference<decltype(target)>::type;
        update_target_layer(target.input_layer, source.input_layer, polyak);
        for(index_t layer_i=0; layer_i < TargetNetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_target_layer(target.hidden_layers[layer_i], source.hidden_layers[layer_i], polyak);
        }
        update_target_layer(target.output_layer, source.output_layer, polyak);
    }

    template <typename DEVICE, typename SPEC>
    void update_targets(rl::algorithms::td3::ActorCritic<DEVICE, SPEC>& actor_critic) {
        update_target_network(actor_critic.actor_target   , actor_critic.   actor, SPEC::PARAMETERS::ACTOR_POLYAK);
        update_target_network(actor_critic.critic_target_1, actor_critic.critic_1, SPEC::PARAMETERS::CRITIC_POLYAK);
        update_target_network(actor_critic.critic_target_2, actor_critic.critic_2, SPEC::PARAMETERS::CRITIC_POLYAK);

    }


}

#endif
