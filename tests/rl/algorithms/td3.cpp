#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <layer_in_c/nn_models/models.h>

#include <layer_in_c/rl/environments/pendulum.h>
#include <layer_in_c/rl/algorithms/td3/off_policy_runner.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>
#include <layer_in_c/utils/rng_std.h>
#include "../../utils/utils.h"
#define DATA_FILE_PATH "../multirotor-torch/model.hdf5"
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

typedef Pendulum<DTYPE, DefaultPendulumParams<DTYPE>> ENVIRONMENT;


using namespace layer_in_c;
using namespace layer_in_c::nn_models;
//using namespace layer_in_c::nn::layers;
#define N_WARMUP_STEPS 100
#define SKIP_FULL_TRAINING

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

template <typename SPEC>
typename SPEC::T assign(Layer<SPEC>& layer, const HighFive::Group g){
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
using TestActorNetworkDefinition = ActorNetworkSpecification<T, 64, 64, ActivationFunction::RELU>;

template <typename T>
using TestCriticNetworkDefinition = CriticNetworkSpecification<T, 64, 64, ActivationFunction::RELU>;

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("0/weight").read(weights);
    acc += abs_diff_matrix<T, NT::SPEC::LAYER_1::OUTPUT_DIM, NT::SPEC::LAYER_1::INPUT_DIM>(network.layer_1.weights, weights);
    return acc;
}
template <typename T, typename NT>
T abs_diff_network(const NT n1, const NT n2){
    constexpr int S = NT::SPEC::LAYER_1::OUTPUT_DIM * NT::SPEC::LAYER_1::INPUT_DIM;
    return abs_diff<T, S>((T*)n1.layer_1.weights, (T*)n2.layer_1.weights);
}

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_1) {
    std::mt19937 rng(0);
    typedef ActorCritic<ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;
    init<ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);
    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);

    std::vector<std::vector<DTYPE>> layer_1_weights;
    assign_network(actor_critic.   actor, data_file.getGroup("actor"));
    assign_network(actor_critic.critic_1, data_file.getGroup("critic_1"));
    assign_network(actor_critic.critic_2, data_file.getGroup("critic_2"));
    assign_network(actor_critic.   actor_target, data_file.getGroup("actor_target"));
    assign_network(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    assign_network(actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    ReplayBuffer<DTYPE, 3, 1, 100> replay_buffer;
    data_file.getDataSet("batch/states"     ).read(replay_buffer.observations);
    data_file.getDataSet("batch/actions"    ).read(replay_buffer.actions);
    data_file.getDataSet("batch/next_states").read(replay_buffer.next_observations);
    data_file.getDataSet("batch/rewards"    ).read(replay_buffer.rewards);
    replay_buffer.position = 1;

    DTYPE init_diff = abs_diff_network<DTYPE>(actor_critic.critic_1, data_file.getGroup("critic_1"));
    DTYPE pre_diff = abs_diff_network<DTYPE>(actor_critic.critic_1, data_file.getGroup("critic_training/0"));
    auto pre_critic = actor_critic.critic_1;
    DTYPE critic_1_loss = train_critic(actor_critic, actor_critic.critic_1, replay_buffer, rng);
    DTYPE post_diff = abs_diff_network<DTYPE>(actor_critic.critic_1, data_file.getGroup("critic_training/0"));
    DTYPE post_diff_update = abs_diff_network<DTYPE>(actor_critic.critic_1, pre_critic);
//    data_file.getDataSet("batch/terminated" ).read(replay_buffer.terminated);

}

