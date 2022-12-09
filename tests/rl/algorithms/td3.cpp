#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/operations_generic.h>
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
typedef lic::rl::environments::pendulum::Spec<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::pendulum::Pendulum<lic::devices::Generic, PENDULUM_SPEC> ENVIRONMENT;
ENVIRONMENT env;

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
struct TD3Parameters: public lic::rl::algorithms::td3::DefaultTD3Parameters<T>{
    constexpr static int CRITIC_BATCH_SIZE = 32;
    constexpr static int ACTOR_BATCH_SIZE = 32;
};
template <typename T>
using TestActorNetworkDefinition = lic::rl::algorithms::td3::ActorNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

template <typename T>
using TestCriticNetworkDefinition = lic::rl::algorithms::td3::CriticNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

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
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::rl::algorithms::td3::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(
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
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_CRITIC_BACKWARD) {
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::rl::algorithms::td3::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);

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
*/
/*
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_CRITIC_TRAINING) {
    typedef lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>> ActorCriticSpec;
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);



    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.actor, data_file.getGroup("actor"));
    lic::load(actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    typedef lic::rl::algorithms::td3::ReplayBuffer<DTYPE, 3, 1, 100> ReplayBufferType;
    ReplayBufferType replay_buffer;
    load_dataset(data_file.getGroup("batch"), replay_buffer);
    static_assert(TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE == TD3Parameters<DTYPE>::CRITIC_BATCH_SIZE, "ACTOR_BATCH_SIZE must be CRITIC_BATCH_SIZE");
    replay_buffer.position = TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE;

    auto pre_critic = actor_critic.critic_1;
    lic::reset_optimizer_state(actor_critic.critic_1);
    DTYPE mean_ratio = 0;
    int num_updates = data_file.getGroup("critic_training").getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        auto post_critic = actor_critic.critic_1;
        std::stringstream ss;
        ss << "critic_training/" << training_step_i;
        lic::load(post_critic, data_file.getGroup(ss.str()));

        DTYPE critic_1_loss = lic::train_critic<lic::devices::Generic, ActorCriticSpec,  ActorCriticType::CRITIC_NETWORK_TYPE, ReplayBufferType::CAPACITY, typeof(rng), true>(actor_critic, actor_critic.critic_1, replay_buffer, rng);

        DTYPE pre_post_diff_per_weight = abs_diff(pre_critic, post_critic)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
        DTYPE diff_target_per_weight = abs_diff(post_critic, actor_critic.critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
        DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

        std::cout << "pre_post_diff_per_weight: " << pre_post_diff_per_weight << std::endl;
        std::cout << "diff_target_per_weight: " << diff_target_per_weight << std::endl;
        std::cout << "pre_post to diff_target: " << diff_ratio << std::endl;

        mean_ratio += diff_ratio;

//        ASSERT_LT(diff_target_per_weight, 1e-7);
//        actor_critic.critic_1 = post_critic;

    }
    mean_ratio /= num_updates;
    std::cout << "mean_ratio: " << mean_ratio << std::endl;
    ASSERT_GT(mean_ratio, 1e6);
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_ACTOR_TRAINING) {
    typedef lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>> ActorCriticSpec;
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);



    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.actor, data_file.getGroup("actor"));
    lic::load(actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    typedef lic::rl::algorithms::td3::ReplayBuffer<DTYPE, 3, 1, 100> ReplayBufferType;
    ReplayBufferType replay_buffer;
    load_dataset(data_file.getGroup("batch"), replay_buffer);
    static_assert(TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE == TD3Parameters<DTYPE>::CRITIC_BATCH_SIZE, "ACTOR_BATCH_SIZE must be CRITIC_BATCH_SIZE");
    replay_buffer.position = TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE;

    auto pre_actor = actor_critic.actor;
    lic::reset_optimizer_state(actor_critic.actor);
    DTYPE mean_ratio = 0;
    int num_updates = data_file.getGroup("actor_training").getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        auto post_actor = actor_critic.actor;
        std::stringstream ss;
        ss << "actor_training/" << training_step_i;
        lic::load(post_actor, data_file.getGroup(ss.str()));

        DTYPE actor_1_loss = lic::train_actor<lic::devices::Generic, ActorCriticSpec,  ReplayBufferType::CAPACITY, typeof(rng), true>(actor_critic, replay_buffer, rng);

        DTYPE pre_post_diff_per_weight = abs_diff(pre_actor, post_actor)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
        DTYPE diff_target_per_weight = abs_diff(post_actor, actor_critic.actor)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
        DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

        std::cout << "pre_post_diff_per_weight: " << pre_post_diff_per_weight << std::endl;
        std::cout << "diff_target_per_weight: " << diff_target_per_weight << std::endl;
        std::cout << "pre_post to diff_target: " << diff_ratio << std::endl;

        mean_ratio += diff_ratio;

//        ASSERT_LT(diff_target_per_weight, 1e-7);
//        actor_critic.critic_1 = post_critic;

    }
    mean_ratio /= num_updates;
    std::cout << "mean_ratio: " << mean_ratio << std::endl;
    ASSERT_GT(mean_ratio, 100000);
}
*/

const DTYPE STATE_TOLERANCE = 0.00001;
#define N_WARMUP_STEPS 100
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_FULL_TRAINING) {
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, lic::rl::algorithms::td3::DefaultTD3Parameters<DTYPE>>> ActorCriticType;
    lic::rl::algorithms::td3::OffPolicyRunner<DTYPE, ENVIRONMENT, lic::rl::algorithms::td3::DefaultOffPolicyRunnerParameters<DTYPE, 50000, 200>> off_policy_runner;
    ActorCriticType actor_critic;
    std::mt19937 rng(0);
    lic::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.actor, data_file.getGroup("actor"));
    lic::load(actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));
    lic::reset_optimizer_state(actor_critic.actor);
    lic::reset_optimizer_state(actor_critic.critic_1);
    lic::reset_optimizer_state(actor_critic.critic_2);

    for(int step_i = 0; step_i < 10000000; step_i++){
        step(off_policy_runner, actor_critic.actor, rng);
        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                std::cout << "step_i: " << step_i << std::endl;
            }
            DTYPE critic_1_loss = lic::train_critic(actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
            lic::train_critic(actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                lic::train_actor(actor_critic, off_policy_runner.replay_buffer, rng);
                lic::update_targets(actor_critic);
            }
        }
    }
}