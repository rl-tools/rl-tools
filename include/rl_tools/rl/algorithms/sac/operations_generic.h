#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_OPERATIONS_GENERIC_H

#include "sac.h"

#include "../../../rl/components/replay_buffer/replay_buffer.h"
#include "../../../rl/components/off_policy_runner/off_policy_runner.h"
#include "../../../nn/nn.h"
#include "../../../nn_models/operations_generic.h"
#include "../../../utils/polyak/operations_generic.h"
#include "../../../math/operations_generic.h"
#include "../../../utils/generic/memcpy.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic){
        malloc(device, actor_critic.actor);
        malloc(device, actor_critic.critic_1);
        malloc(device, actor_critic.critic_2);
        malloc(device, actor_critic.critic_target_1);
        malloc(device, actor_critic.critic_target_2);
        malloc(device, actor_critic.log_alpha);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic){
        free(device, actor_critic.actor);
        free(device, actor_critic.critic_1);
        free(device, actor_critic.critic_2);
        free(device, actor_critic.critic_target_1);
        free(device, actor_critic.critic_target_2);
        free(device, actor_critic.log_alpha);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        malloc(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations = view(device, actor_training_buffers.state_action_value_input, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::CRITIC_OBSERVATION_DIM>{}, 0, 0);
        actor_training_buffers.actions      = view(device, actor_training_buffers.state_action_value_input, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM);
//        malloc(device, actor_training_buffers.state_action_value);
        malloc(device, actor_training_buffers.d_output);
        malloc(device, actor_training_buffers.d_critic_1_input);
        malloc(device, actor_training_buffers.d_critic_2_input);
        malloc(device, actor_training_buffers.d_critic_action_input);
        malloc(device, actor_training_buffers.action_sample);
        malloc(device, actor_training_buffers.action_noise);
        malloc(device, actor_training_buffers.d_actor_output);
        malloc(device, actor_training_buffers.d_actor_input);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        free(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations._data = nullptr;
        actor_training_buffers.actions._data      = nullptr;
//        free(device, actor_training_buffers.state_action_value);
        free(device, actor_training_buffers.d_output);
        free(device, actor_training_buffers.d_critic_1_input);
        free(device, actor_training_buffers.d_critic_2_input);
        free(device, actor_training_buffers.d_critic_action_input);
        free(device, actor_training_buffers.action_sample);
        free(device, actor_training_buffers.action_noise);
        free(device, actor_training_buffers.d_actor_output);
        free(device, actor_training_buffers.d_actor_input);
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        malloc(device, critic_training_buffers.next_state_action_value_input_full);
        critic_training_buffers.next_state_action_value_input = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::CRITIC_OBSERVATION_DIM + BUFFERS::ACTION_DIM>{}, 0, 0);
        critic_training_buffers.next_observations             = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::CRITIC_OBSERVATION_DIM>{}, 0, 0);
        critic_training_buffers.next_actions_distribution     = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM*2>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM);
        critic_training_buffers.next_actions_mean             = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM);
        critic_training_buffers.next_actions_log_std          = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM + BUFFERS::ACTION_DIM);
        malloc(device, critic_training_buffers.action_value);
        malloc(device, critic_training_buffers.target_action_value);
        malloc(device, critic_training_buffers.next_state_action_value_critic_1);
        malloc(device, critic_training_buffers.next_state_action_value_critic_2);
        malloc(device, critic_training_buffers.d_output);
        malloc(device, critic_training_buffers.d_input);
        malloc(device, critic_training_buffers.action_log_probs);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        free(device, critic_training_buffers.next_state_action_value_input_full);
        critic_training_buffers.next_state_action_value_input._data = nullptr;
        critic_training_buffers.next_observations._data = nullptr;
        critic_training_buffers.next_actions_distribution._data = nullptr;
        critic_training_buffers.next_actions_mean._data = nullptr;
        critic_training_buffers.next_actions_log_std._data = nullptr;
        free(device, critic_training_buffers.action_value);
        free(device, critic_training_buffers.target_action_value);
        free(device, critic_training_buffers.next_state_action_value_critic_1);
        free(device, critic_training_buffers.next_state_action_value_critic_2);
        free(device, critic_training_buffers.d_output);
        free(device, critic_training_buffers.d_input);
        free(device, critic_training_buffers.action_log_probs);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void init(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, RNG& rng){
        init_weights(device, actor_critic.actor   , rng);
        init_weights(device, actor_critic.critic_1, rng);
        init_weights(device, actor_critic.critic_2, rng);
        zero_gradient(device, actor_critic.actor);
        zero_gradient(device, actor_critic.critic_1);
        zero_gradient(device, actor_critic.critic_2);
        zero_gradient(device, actor_critic.log_alpha);
        reset_optimizer_state(device, actor_critic.actor_optimizer, actor_critic.actor);
        reset_optimizer_state(device, actor_critic.critic_optimizers[0], actor_critic.critic_1);
        reset_optimizer_state(device, actor_critic.critic_optimizers[1], actor_critic.critic_2);
        reset_optimizer_state(device, actor_critic.alpha_optimizer, actor_critic.log_alpha);
        set(actor_critic.log_alpha.parameters, 0, 0, math::log(typename DEVICE::SPEC::MATH{}, SPEC::PARAMETERS::ALPHA));


        copy(device, device, actor_critic.critic_1, actor_critic.critic_target_1);
        copy(device, device, actor_critic.critic_2, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename SPEC>
    void target_actions(DEVICE& device, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, typename SPEC::T alpha) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        constexpr auto OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            T min_next_state_action_value = math::min(device.math,
                    get(training_buffers.next_state_action_value_critic_1, batch_step_i, 0),
                    get(training_buffers.next_state_action_value_critic_2, batch_step_i, 0)
            );
            T reward = get(batch.rewards, 0, batch_step_i);
            bool terminated = get(batch.terminated, 0, batch_step_i);
            T future_value = SPEC::PARAMETERS::IGNORE_TERMINATION || !terminated ? SPEC::PARAMETERS::GAMMA * min_next_state_action_value : 0;
            T entropy_bonus = -alpha * get(training_buffers.action_log_probs, batch_step_i, 0);
            T current_target_action_value = reward + future_value + entropy_bonus;
            set(training_buffers.target_action_value, batch_step_i, 0, current_target_action_value); // todo: improve pitch of target action values etc. (by transformig it into row vectors instead of column vectors)
        }
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename RNG>
    void train_critic(DEVICE& device, const rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(BATCH_SIZE == CRITIC_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);

        zero_gradient(device, critic);

        evaluate(device, actor_critic.actor, batch.next_observations, training_buffers.next_actions_distribution, actor_buffers);
        for(TI sample_i = 0; sample_i < BATCH_SIZE; sample_i++){
            T action_log_prob = 0;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T mean = get(training_buffers.next_actions_mean, sample_i, action_i);
                T log_std = get(training_buffers.next_actions_log_std, sample_i, action_i);
                T action_sampled = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, mean, math::exp(typename DEVICE::SPEC::MATH{}, log_std), rng);
                set(training_buffers.next_actions_mean, sample_i, action_i, action_sampled);
                action_log_prob += random::normal_distribution::log_prob<DEVICE, T>(typename DEVICE::SPEC::RANDOM{}, mean, log_std, action_sampled);
            }
            set(training_buffers.action_log_probs, sample_i, 0, action_log_prob);
        }
        copy(device, device, batch.next_observations_privileged, training_buffers.next_observations);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_buffers);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_buffers);

        T log_alpha = get(actor_critic.log_alpha.parameters, 0, 0);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, log_alpha);
        target_actions(device, batch, training_buffers, alpha);
        forward(device, critic, batch.observations_and_actions);
        nn::loss_functions::mse::gradient(device, output(critic), training_buffers.target_action_value, training_buffers.d_output);
        backward(device, critic, batch.observations_and_actions, training_buffers.d_output, critic_buffers);
        step(device, optimizer, critic);
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE>
    typename SPEC::T critic_loss(DEVICE& device, const rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, typename SPEC::ACTOR_NETWORK_TYPE::template Buffers<BATCH_SIZE>& actor_buffers, typename CRITIC_TYPE::template Buffers<BATCH_SIZE>& critic_buffers, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

        evaluate(device, actor_critic.actor, batch.next_observations, training_buffers.next_actions_distribution, actor_buffers);
        copy(device, device, batch.next_observations_privileged, training_buffers.next_observations);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_buffers);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_buffers);

        T log_alpha = get(actor_critic.log_alpha, 0, 0);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, log_alpha);
        target_actions(device, batch, training_buffers, alpha);
        evaluate(device, critic, batch.observations_and_actions, training_buffers.action_value, critic_buffers);
        return nn::loss_functions::mse::evaluate(device, training_buffers.action_value, training_buffers.target_action_value);
    }
    template <typename DEVICE, typename SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename RNG>
    void train_actor(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
        static_assert(BATCH_SIZE == CRITIC_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static_assert(SPEC::ACTOR_NETWORK_TYPE::OUTPUT_DIM == ACTION_DIM*2);

        zero_gradient(device, actor_critic.actor);
        forward(device, actor_critic.actor, batch.observations);
//        auto actions_view = view(device, output(actor_critic.actor), matrix::ViewSpec<BATCH_SIZE, ACTION_DIM>{}, 0, 0);
//        copy(device, device, actions_view, training_buffers.actions)
        auto actions_full = output(actor_critic.actor);
        for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T std = math::exp(typename DEVICE::SPEC::MATH{}, get(actions_full, batch_i, action_i + ACTION_DIM));
                // action_sample = noise * std + mean
                T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, (T)1, rng);
                set(training_buffers.action_noise, batch_i, action_i, noise);
                T action_sample = get(actions_full, batch_i, action_i) + std * noise;
                set(training_buffers.action_sample, batch_i, action_i, action_sample);
                T action = math::tanh(typename DEVICE::SPEC::MATH{}, action_sample);
                set(training_buffers.actions, batch_i, action_i, action);
            }
        }
        copy(device, device, batch.observations_privileged, training_buffers.observations);
//        auto& critic = actor_critic.critic_1;
        forward(device, actor_critic.critic_1, training_buffers.state_action_value_input);
        forward(device, actor_critic.critic_2, training_buffers.state_action_value_input);
        set_all(device, training_buffers.d_output, (T)-1/BATCH_SIZE);
        backward_input(device, actor_critic.critic_1, training_buffers.d_output, training_buffers.d_critic_1_input, critic_buffers);
        backward_input(device, actor_critic.critic_2, training_buffers.d_output, training_buffers.d_critic_2_input, critic_buffers);
/*
        Gradient of the loss function:
        mu, std = policy(observation)
        action_sample = gaussian::sample(mu, std)
        action = tanh(action_sample)
        action_prob = gaussian::prob(mu, std, action_sample) * | d/d_action tanh^{-1}(action) |
                    = gaussian::prob(mu, std, action_sample) * | (d/d_action_sample tanh(action_sample))^{-1} |
                    = gaussian::prob(mu, std, action_sample) * | (d/d_action_sample tanh(action_sample))|^{-1}
                    = gaussian::prob(mu, std, action_sample) * ((1-tanh(action_sample)^2))^{-1}
        action_log_prob = gaussian::log_prob(mu, std, action_sample) - log(1-tanh(action_sample)^2))
        actor_loss = alpha  * action_log_prob - min(Q_1, Q_2);
        d/d_mu _actor_loss = alpha * d/d_mu action_log_prob - d/d_mu min(Q_1, Q_2)
        d/d_mu action_log_prob = d/d_mu gaussian::log_prob(mu, std, action_sample) + d/d_action_sample gaussian::log_prob(mu, std, action_sample) * d/d_mu action_sample + 1/(1-tanh(action_sample)^2) * 2*tanh(action_sample) * d/d_mu action_sample
        d/d_mu action_sample = 1
        d/d_std action_sample = noise
        d/d_mu min(Q_1, Q_2) = d/d_action min(Q_1, Q_2) * d/d_mu action
        d/d_mu action = d/d_action_sample tanh(action_sample) * d/d_mu action_sample
*/
        T d_alpha = 0;
        T log_alpha = get(actor_critic.log_alpha.parameters, 0, 0);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, log_alpha);
        T mean_entropy = 0;
        for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
            bool critic_1_value = get(output(actor_critic.critic_1), batch_i, 0) < get(output(actor_critic.critic_2), batch_i, 0);
            T entropy = 0;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++) {
                T action = get(training_buffers.actions, batch_i, action_i);
                T d_mu = 0;
                T d_std = 0;
                {
                    T d_input = 0;
                    if(critic_1_value) {
                        d_input = get(training_buffers.d_critic_1_input, batch_i, SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM - ACTION_DIM + action_i);
                    }
                    else{
                        d_input = get(training_buffers.d_critic_2_input, batch_i, SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM - ACTION_DIM + action_i);
                    }
                    d_mu  = d_input * (1-action*action);
                    d_std = d_input * (1-action*action) * get(training_buffers.action_noise, batch_i, action_i);
                }
                T log_std = get(actions_full, batch_i, action_i + ACTION_DIM);
                T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);

                T d_log_std = std * d_std;

                T mu = get(actions_full, batch_i, action_i);
                T action_sample = get(training_buffers.action_sample, batch_i, action_i);
                T eps = 1e-6;
                T one_minus_action_square_plus_eps = (1-action*action + eps);
                T one_over_one_minus_action_square_plus_eps = 1/one_minus_action_square_plus_eps;
                d_mu += alpha/BATCH_SIZE * (random::normal_distribution::d_log_prob_d_mean<DEVICE, T>(typename DEVICE::SPEC::RANDOM{}, mu, log_std, action_sample) + random::normal_distribution::d_log_prob_d_sample<DEVICE, T>(typename DEVICE::SPEC::RANDOM{}, mu, log_std, action_sample) + one_over_one_minus_action_square_plus_eps * 2*action);

                T noise = get(training_buffers.action_noise, batch_i, action_i);
                d_log_std += alpha/BATCH_SIZE * (random::normal_distribution::d_log_prob_d_log_std<DEVICE, T>(typename DEVICE::SPEC::RANDOM{}, mu, log_std, action_sample) + random::normal_distribution::d_log_prob_d_sample<DEVICE, T>(typename DEVICE::SPEC::RANDOM{}, mu, log_std, action_sample) * noise * std + one_over_one_minus_action_square_plus_eps * 2*action * noise * std);

                set(training_buffers.d_actor_output, batch_i, action_i, d_mu);
                set(training_buffers.d_actor_output, batch_i, action_i + ACTION_DIM, d_log_std);

                T action_log_prob = random::normal_distribution::log_prob<DEVICE, T>(typename DEVICE::SPEC::RANDOM{}, mu, log_std, action_sample) - math::log(typename DEVICE::SPEC::MATH{}, one_minus_action_square_plus_eps);
                entropy += -action_log_prob;
            }
            d_alpha += entropy - SPEC::PARAMETERS::TARGET_ENTROPY;
            mean_entropy += entropy;
        }
        backward(device, actor_critic.actor, batch.observations, training_buffers.d_actor_output, actor_buffers);
        step(device, optimizer, actor_critic.actor);
        // adapting alpha
        if constexpr(SPEC::PARAMETERS::ADAPTIVE_ALPHA){
            d_alpha /= BATCH_SIZE;
            mean_entropy /= BATCH_SIZE;
//            T d_log_alpha = 1/alpha * d_alpha; // This and the former optimize the same objective but the latter seems to work better
            T d_log_alpha = d_alpha;
            set(actor_critic.log_alpha.gradient, 0, 0, d_log_alpha);
            step(device, actor_critic.alpha_optimizer, actor_critic.log_alpha);
        }
    }

    template <typename DEVICE, typename SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE>
    typename SPEC::T actor_value(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, typename SPEC::ACTOR_NETWORK_TYPE::template Buffers<BATCH_SIZE>& actor_buffers, typename SPEC::CRITIC_NETWORK_TYPE::template Buffers<BATCH_SIZE>& critic_buffers, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);

        evaluate(device, actor_critic.actor, batch.observations, training_buffers.actions, actor_buffers);
        copy(device, device, batch.observations, training_buffers.observations);
        auto& critic = actor_critic.critic_1;
        evaluate(device, critic, training_buffers.state_action_value_input, training_buffers.state_action_value, critic_buffers);
        return mean(device, training_buffers.state_action_value);
    }

    namespace rl::algorithms::sac{
        template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        void update_target_layer(DEVICE& device, const  nn::layers::dense::Layer<SOURCE_SPEC>& source, nn::layers::dense::Layer<TARGET_SPEC>& target, typename SOURCE_SPEC::T polyak) {
            rl_tools::utils::polyak::update(device, source.weights.parameters, target.weights.parameters, polyak);
            rl_tools::utils::polyak::update(device, source.biases.parameters , target.biases.parameters , polyak);
        }
        template<typename T, typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        void update_target_network(DEVICE& device, const  nn_models::mlp::NeuralNetwork<SOURCE_SPEC>& source, nn_models::mlp::NeuralNetwork<TARGET_SPEC>& target, T polyak) {
            using TargetNetworkType = nn_models::mlp::NeuralNetwork<TARGET_SPEC>;
            update_target_layer(device, source.input_layer, target.input_layer, polyak);
            for(typename DEVICE::index_t layer_i=0; layer_i < TargetNetworkType::NUM_HIDDEN_LAYERS; layer_i++){
                update_target_layer(device, source.hidden_layers[layer_i], target.hidden_layers[layer_i], polyak);
            }
            update_target_layer(device, source.output_layer, target.output_layer, polyak);
        }
        template<typename T, typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        void update_target_network(DEVICE& device, const  nn_models::sequential::Module<SOURCE_SPEC>& source, nn_models::sequential::Module<TARGET_SPEC>& target, T polyak) {
            update_target_layer(device, source.content, target.content, polyak);
            if constexpr(!rl_tools::utils::typing::is_same_v<typename SOURCE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
                update_target_network(device, source.next_module, target.next_module, polyak);
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    void update_critic_targets(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic) {
        rl::algorithms::sac::update_target_network(device, actor_critic.critic_1, actor_critic.critic_target_1, SPEC::PARAMETERS::CRITIC_POLYAK);
        rl::algorithms::sac::update_target_network(device, actor_critic.critic_2, actor_critic.critic_target_2, SPEC::PARAMETERS::CRITIC_POLYAK);
    }

    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& ac) {
        bool found_nan = false;
        found_nan = found_nan || is_nan(device, ac.actor);
        found_nan = found_nan || is_nan(device, ac.critic_1);
        found_nan = found_nan || is_nan(device, ac.critic_2);
        found_nan = found_nan || is_nan(device, ac.critic_target_1);
        found_nan = found_nan || is_nan(device, ac.critic_target_2);
        return found_nan;
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::sac::ActorCritic<SOURCE_SPEC>& source, rl::algorithms::sac::ActorCritic<TARGET_SPEC>& target){
        copy(source_device, target_device, source.actor   , target.actor);
        copy(source_device, target_device, source.critic_1, target.critic_1);
        copy(source_device, target_device, source.critic_2, target.critic_2);

        copy(source_device, target_device, source.critic_target_1, target.critic_target_1);
        copy(source_device, target_device, source.critic_target_2, target.critic_target_2);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::sac::CriticTrainingBuffers<SOURCE_SPEC>& source, rl::algorithms::sac::CriticTrainingBuffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.next_state_action_value_input_full, target.next_state_action_value_input_full);
        copy(source_device, target_device, source.target_action_value, target.target_action_value);
        copy(source_device, target_device, source.next_state_action_value_critic_1, target.next_state_action_value_critic_1);
        copy(source_device, target_device, source.next_state_action_value_critic_2, target.next_state_action_value_critic_2);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
