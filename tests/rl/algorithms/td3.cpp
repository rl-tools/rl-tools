#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/persist.h>

#include <layer_in_c/rl/environments/pendulum.h>
#include <layer_in_c/rl/algorithms/td3/off_policy_runner.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>
#include <layer_in_c/utils/rng_std.h>
#include "../../utils/utils.h"
#include "../../utils/nn_comparison.h"

namespace lic = layer_in_c;
#define DATA_FILE_PATH "../multirotor-torch/model.hdf5"
#define DTYPE float

const DTYPE STATE_TOLERANCE = 0.00001;

typedef Pendulum<DTYPE, DefaultPendulumParams<DTYPE>> ENVIRONMENT;


#define N_WARMUP_STEPS 100
#define SKIP_FULL_TRAINING

template <typename T>
struct Dataset{
    Dataset(HighFive::Group g){
        g.getDataSet("states").read(states);
        g.getDataSet("actions").read(actions);
        g.getDataSet("next_states").read(next_states);
        g.getDataSet("rewards").read(rewards);
        g.getDataSet("terminated").read(terminated);
    };
    std::vector<std::vector<DTYPE>> states;
    std::vector<std::vector<DTYPE>> actions;
    std::vector<std::vector<DTYPE>> next_states;
    std::vector<std::vector<DTYPE>> rewards;
    std::vector<std::vector<DTYPE>> terminated;
};

template <typename RB>
void load_dataset(HighFive::Group g, RB& rb){
    g.getDataSet("states").read(rb.observations);
    g.getDataSet("actions").read(rb.actions);
    g.getDataSet("next_states").read(rb.next_observations);
    g.getDataSet("rewards").read(rb.rewards);
    std::vector<typename RB::T> terminated;
    g.getDataSet("terminated").read(terminated);
    for(int i = 0; i < terminated.size(); i++){
        rb.terminated[i] = terminated[i] == 1;
    }
    std::vector<typename RB::T> truncated;
    g.getDataSet("truncated").read(truncated);
    for(int i = 0; i < truncated.size(); i++){
        rb.truncated[i] = truncated[i] == 1;
    }
    rb.position = terminated.size();
}


OffPolicyRunner<DTYPE, ENVIRONMENT, DefaultOffPolicyRunnerParameters<DTYPE>> off_policy_runner;

#ifndef SKIP_FULL_TRAINING
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_FULL_TRAINING) {
    std::mt19937 rng(0);
    ActorCritic<DTYPE, ENVIRONMENT, DefaultActorNetworkDefinition<DTYPE>, DefaultCriticNetworkDefinition<DTYPE>, DefaultTD3Parameters<DTYPE>> actor_critic;
    init(actor_critic, rng);
    for(int step_i = 0; step_i < 10000000; step_i++){
        step(off_policy_runner, actor_critic.actor, rng);
        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                std::cout << "step_i: " << step_i << std::endl;
            }
            DTYPE critic_1_loss = train_critic(actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
            train_critic(actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//        std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                train_actor(actor_critic, off_policy_runner.replay_buffer, rng);
                update_targets(actor_critic);
            }
        }
    }
}
#endif

template <typename DEVICE, typename SPEC>
typename SPEC::T assign(lic::nn::layers::dense::Layer<DEVICE, SPEC>& layer, const HighFive::Group g){
    std::vector<std::vector<typename SPEC::T>> weights;
    std::vector<typename SPEC::T> biases;
    g.getDataSet("weight").read(weights);
    g.getDataSet("bias").read(biases);
    for(int i = 0; i < SPEC::OUTPUT_DIM; i++){
        for(int j = 0; j < SPEC::INPUT_DIM; j++){
            layer.weights[i][j] = weights[i][j];
        }
        layer.biases[i] = biases[i];
    }
}
template <typename NT>
void assign_network(NT& network, const HighFive::Group g){
    assign(network.layer_1, g.getGroup("0"));
    assign(network.layer_2, g.getGroup("1"));
    assign(network.output_layer, g.getGroup("2"));
}

template <typename T>
struct TD3Parameters: public DefaultTD3Parameters<T>{
    constexpr static int CRITIC_BATCH_SIZE = 1;
};
template <typename T>
using TestActorNetworkDefinition = ActorNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU>;

template <typename T>
using TestCriticNetworkDefinition = CriticNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU>;

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("0/weight").read(weights);
    acc += abs_diff_matrix<T, NT::SPEC::LAYER_1::OUTPUT_DIM, NT::SPEC::LAYER_1::INPUT_DIM>(network.layer_1.weights, weights);
    return acc;
}
/*
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_CRITIC_FORWARD) {
    typedef ActorCritic<lic::devices::Generic, ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(
            actor_critic, rng);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));

    DTYPE input[ActorCriticType::CRITIC_INPUT_DIM];
    for (int i = 0; i < batch.states[0].size(); i++) {
        input[i] = batch.states[0][i];
    }
    for (int i = 0; i < batch.actions[0].size(); i++) {
        input[batch.states[0].size() + i] = batch.actions[0][i];
    }

    DTYPE output[1];
    lic::evaluate(actor_critic.critic_1, input, output);
    std::cout << "output: " << output[0] << std::endl;
    ASSERT_LT(abs(output[0] - -0.00560237), 1e-7);

    lic::evaluate(actor_critic.critic_target_1, input, output);
    std::cout << "output: " << output[0] << std::endl;
    ASSERT_LT(abs(output[0] - -0.00560237), 1e-7);
}
 */
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_) {
    typedef ActorCritic<lic::devices::Generic, ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(
            actor_critic, rng);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));

    DTYPE input[ActorCriticType::CRITIC_INPUT_DIM];
    for (int i = 0; i < batch.states[0].size(); i++) {
        input[i] = batch.states[0][i];
    }
    for (int i = 0; i < batch.actions[0].size(); i++) {
        input[batch.states[0].size() + i] = batch.actions[0][i];
    }
    DTYPE target[1] = {1};
    DTYPE output[1];
    lic::evaluate(actor_critic.critic_1, input, output);
    DTYPE loss = lic::nn::loss_functions::mse<DTYPE, 1>(output, target);
    ASSERT_LT(std::abs(loss - 1.0112361), 1e-7);

    lic::zero_gradient(actor_critic.critic_1);
    lic::forward_backward_mse(actor_critic.critic_1, input, target);
    std::cout << "output: " << actor_critic.critic_1.output_layer.output[0] << std::endl;
    ASSERT_LT(std::abs(actor_critic.critic_1.output_layer.output[0] - -0.00560237), 1e-7);

    auto critic_1_after_backward = actor_critic.critic_1;
    lic::load(critic_1_after_backward, data_file.getGroup("critic_1_backward"));
    DTYPE diff_grad_per_weight = abs_diff_grad(actor_critic.critic_1, critic_1_after_backward)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
    ASSERT_LT(diff_grad_per_weight, 1e-8);

    std::cout << "diff_grad_per_weight: " << diff_grad_per_weight << std::endl;
}
/*
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_2) {
    typedef ActorCritic<lic::devices::Generic, ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_2"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_2"));

    ReplayBuffer<DTYPE, 3, 1, 100> replay_buffer;
    load_dataset(data_file.getGroup("batch"), replay_buffer);
    replay_buffer.position = 1;

    auto pre_critic = actor_critic.critic_1;
    auto post_critic = actor_critic.critic_1;
    lic::load(post_critic, data_file.getGroup("critic_training/0"));

    DTYPE critic_1_loss = train_critic(actor_critic, actor_critic.critic_1, replay_buffer, rng);

    DTYPE pre_post_diff = abs_diff(pre_critic, post_critic);
    DTYPE diff_target = abs_diff(post_critic, actor_critic.critic_1);


}

 */