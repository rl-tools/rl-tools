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
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_CRITIC_FORWARD) {
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(
            actor_critic, rng);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));

    std::vector<std::vector<DTYPE>> outputs;
    data_file.getDataSet("batch_output").read(outputs);

    for(int batch_sample_i = 0; batch_sample_i < batch.states.size(); batch_sample_i++){
        DTYPE input[ActorCriticType::CRITIC_INPUT_DIM];
        for (int i = 0; i < batch.states[batch_sample_i].size(); i++) {
            input[i] = batch.states[batch_sample_i][i];
        }
        for (int i = 0; i < batch.actions[batch_sample_i].size(); i++) {
            input[batch.states[batch_sample_i].size() + i] = batch.actions[batch_sample_i][i];
        }

        DTYPE output[1];
        lic::evaluate(actor_critic.critic_1, input, output);
        std::cout << "output: " << output[0] << std::endl;
        ASSERT_LT(abs(output[0] - outputs[batch_sample_i][0]), 1e-7);

        lic::evaluate(actor_critic.critic_target_1, input, output);
        std::cout << "output: " << output[0] << std::endl;
        ASSERT_LT(abs(output[0] - outputs[batch_sample_i][0]), 1e-7);
    }

}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_CRITIC_BACKWARD) {
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));
    assert(batch.states.size() == 32);

    DTYPE loss = 0;
    lic::zero_gradient(actor_critic.critic_1);
    for(int batch_sample_i = 0; batch_sample_i < batch.states.size(); batch_sample_i++){
        DTYPE input[ActorCriticType::CRITIC_INPUT_DIM];
        for (int i = 0; i < batch.states[batch_sample_i].size(); i++) {
            input[i] = batch.states[batch_sample_i][i];
        }
        for (int i = 0; i < batch.actions[batch_sample_i].size(); i++) {
            input[batch.states[batch_sample_i].size() + i] = batch.actions[batch_sample_i][i];
        }
        DTYPE target[1] = {1};
        DTYPE output[1];
        lic::evaluate(actor_critic.critic_1, input, output);
        loss += lic::nn::loss_functions::mse<DTYPE, 1, 1>(output, target);

        lic::forward_backward_mse<ActorCriticType::CRITIC_NETWORK_TYPE::SPEC, 32>(actor_critic.critic_1, input, target);
        std::cout << "output: " << actor_critic.critic_1.output_layer.output[0] << std::endl;
    }

    auto critic_1_after_backward = actor_critic.critic_1;
    lic::load(critic_1_after_backward, data_file.getGroup("critic_1_backward"));
    DTYPE diff_grad_per_weight = abs_diff_grad(actor_critic.critic_1, critic_1_after_backward)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
    ASSERT_LT(diff_grad_per_weight, 1e-8);

    std::cout << "diff_grad_per_weight: " << diff_grad_per_weight << std::endl;
}
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
    auto critic_training_group = data_file.getGroup("critic_training");
    int num_updates = critic_training_group.getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        auto step_group = critic_training_group.getGroup(std::to_string(training_step_i));

        auto post_critic = actor_critic.critic_1;
        lic::load(post_critic, step_group.getGroup("critic"));

        std::vector<std::vector<DTYPE>> target_next_action_noise_vector;
        step_group.getDataSet("target_next_action_noise").read(target_next_action_noise_vector);


        DTYPE target_next_action_noise[ActorCriticSpec::PARAMETERS::CRITIC_BATCH_SIZE][ActorCriticSpec::ENVIRONMENT::ACTION_DIM];
        for(int i = 0; i < ActorCriticSpec::PARAMETERS::CRITIC_BATCH_SIZE; i++){
            for(int j = 0; j < ActorCriticSpec::ENVIRONMENT::ACTION_DIM; j++){
                target_next_action_noise[i][j] = target_next_action_noise_vector[i][j];
            }
        }

        DTYPE critic_1_loss = lic::train_critic_deterministic(actor_critic, actor_critic.critic_1, replay_buffer, target_next_action_noise, rng);

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
/*

template <typename T>
struct TD3ParametersCopyTraining: public lic::rl::algorithms::td3::DefaultTD3Parameters<T>{
    constexpr static int CRITIC_BATCH_SIZE = 100;
    constexpr static int ACTOR_BATCH_SIZE = 100;
};

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_LOADING_TRAINED_ACTOR) {
    constexpr bool verbose = false;
//    constexpr int BATCH_SIZE = 100;
    typedef lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3ParametersCopyTraining<DTYPE>> ActorCriticSpec;
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    auto step_group = data_file.getGroup("full_training").getGroup("steps").getGroup(std::to_string(10001));
    lic::load(actor_critic.actor, step_group.getGroup("actor"));
    DTYPE mean_return = lic::evaluate<ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, typeof(rng), 200>(actor_critic.actor, rng, 100);
    std::cout << "mean return: " << mean_return << std::endl;
}

typedef lic::rl::algorithms::td3::ReplayBuffer<DTYPE, 3, 1, 1000> ReplayBufferTypeCopyTraining;
constexpr int BATCH_DIM = ENVIRONMENT::OBSERVATION_DIM * 2 + ENVIRONMENT::ACTION_DIM + 2;
template <typename T, typename REPLAY_BUFFER_TYPE>
void load(ReplayBufferTypeCopyTraining& rb, std::vector<std::vector<T>> batch){
    for(int i = 0; i < batch.size(); i++){
        memcpy( rb.     observations[i], &batch[i][0], sizeof(T) * ENVIRONMENT::OBSERVATION_DIM);
        memcpy( rb.          actions[i], &batch[i][ENVIRONMENT::OBSERVATION_DIM], sizeof(T) * ENVIRONMENT::ACTION_DIM);
        memcpy( rb.next_observations[i], &batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM], sizeof(T) * ENVIRONMENT::OBSERVATION_DIM);
        rb.   rewards[i] = batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM    ];
        rb. truncated[i] = batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM + 1] == 1;
        rb.terminated[i] = batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM + 2] == 1;
    }
    rb.position = batch.size();
}

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_COPY_TRAINING) {
    constexpr bool verbose = false;
//    constexpr int BATCH_SIZE = 100;
    typedef lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3ParametersCopyTraining<DTYPE>> ActorCriticSpec;
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

    ReplayBufferTypeCopyTraining replay_buffer;

    lic::reset_optimizer_state(actor_critic.actor);
    lic::reset_optimizer_state(actor_critic.critic_1);
    lic::reset_optimizer_state(actor_critic.critic_2);
    DTYPE mean_ratio_critic = 0;
    DTYPE mean_ratio_actor = 0;
    auto full_training_group = data_file.getGroup("full_training");
    auto steps_group = full_training_group.getGroup("steps");
    int num_steps = steps_group.getNumberObjects();
    auto pre_critic_1 = actor_critic.critic_1;
    auto pre_actor = actor_critic.actor;
    for(int step_i = 0; step_i < num_steps; step_i++){
        if(verbose){
            std::cout << "step_i: " << step_i << std::endl;
        }
        auto step_group = steps_group.getGroup(std::to_string(step_i));
        if(step_group.exist("critics_batch")){
            std::vector<std::vector<DTYPE>> batch;
            step_group.getDataSet("critics_batch").read(batch);
            assert(batch.size() == ActorCriticSpec::PARAMETERS::CRITIC_BATCH_SIZE);

            DTYPE target_next_action_noise[ActorCriticSpec::PARAMETERS::CRITIC_BATCH_SIZE][ActorCriticSpec::ENVIRONMENT::ACTION_DIM];
            step_group.getDataSet("target_next_action_noise").read(target_next_action_noise);

            load<DTYPE, ReplayBufferTypeCopyTraining>(replay_buffer, batch);
            if (step_i == 0 && step_group.exist("pre_critic1")){
                ActorCriticType::CRITIC_NETWORK_TYPE pre_critic_1_step;
                lic::load(pre_critic_1_step, step_group.getGroup("pre_critic1"));
                DTYPE pre_current_diff = abs_diff(pre_critic_1_step, actor_critic.critic_1);
                ASSERT_EQ(pre_current_diff, 0);
            }

            ActorCriticType::CRITIC_NETWORK_TYPE post_critic_1;// = actor_critic.critic_1;
            lic::load(post_critic_1, step_group.getGroup("critic1"));

            DTYPE critic_1_loss = lic::train_critic_deterministic(actor_critic, actor_critic.critic_1, replay_buffer, target_next_action_noise, rng);


            DTYPE pre_post_diff_per_weight = abs_diff(pre_critic_1, post_critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_target_per_weight = abs_diff(post_critic_1, actor_critic.critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

            DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(pre_critic_1, post_critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_target_grad_per_weight = abs_diff_grad(post_critic_1, actor_critic.critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

            DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(pre_critic_1, post_critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_target_adam_per_weight = abs_diff_adam(post_critic_1, actor_critic.critic_1)/ActorCriticType::CRITIC_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_ratio_adam = pre_post_diff_adam_per_weight/diff_target_adam_per_weight;

            if(verbose){
                std:: cout << "    critic update" << std::endl;
//                std::cout << "pre_post_diff_per_weight: " << pre_post_diff_per_weight << std::endl;
//                std::cout << "diff_target_per_weight: " << diff_target_per_weight << std::endl;
                std::cout << "        update ratio     : " << diff_ratio << std::endl;
//                std::cout << "pre_post_diff_grad_per_weight: " << pre_post_diff_grad_per_weight << std::endl;
//                std::cout << "diff_target_grad_per_weight: " << diff_target_grad_per_weight << std::endl;
                std::cout << "        update ratio grad: " << diff_ratio_grad << std::endl;
//                std::cout << "pre_post_diff_adam_per_weight: " << pre_post_diff_adam_per_weight << std::endl;
//                std::cout << "diff_target_adam_per_weight: " << diff_target_adam_per_weight << std::endl;
                std::cout << "        update ratio adam: " << diff_ratio_adam << std::endl;
            }

            switch(step_i){
                case 0: {
                    ASSERT_GT(diff_ratio, 1e6);
                    ASSERT_GT(diff_ratio_grad, 1e6);
                    ASSERT_GT(diff_ratio_adam, 1e6);
                }
                    break;
            }

            mean_ratio_critic += diff_ratio;

            DTYPE critic_2_loss = lic::train_critic_deterministic(actor_critic, actor_critic.critic_2, replay_buffer, target_next_action_noise, rng);
            pre_critic_1 = actor_critic.critic_1;

            if(true){//(step_i % 100 == 0){
                DTYPE diff = 0;
                for(int batch_sample_i = 0; batch_sample_i < ActorCriticSpec::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
                    DTYPE input[ActorCriticSpec::ENVIRONMENT::OBSERVATION_DIM + ActorCriticSpec::ENVIRONMENT::ACTION_DIM];
                    memcpy(input, replay_buffer.observations[batch_sample_i], sizeof(DTYPE) * ActorCriticSpec::ENVIRONMENT::OBSERVATION_DIM);
                    memcpy(&input[ActorCriticSpec::ENVIRONMENT::OBSERVATION_DIM], replay_buffer.actions[batch_sample_i], sizeof(DTYPE) * ActorCriticSpec::ENVIRONMENT::ACTION_DIM);
                    DTYPE current_value = lic::evaluate(actor_critic.critic_1, input);
                    DTYPE desired_value = lic::evaluate(post_critic_1, input);
                    diff += (current_value - desired_value) * (current_value - desired_value) / ActorCriticSpec::PARAMETERS::CRITIC_BATCH_SIZE;
                }
                std::cout << "value mse: " << diff << std::endl;
            }
        }

        if(step_group.exist("actor_batch")){
            std::vector<std::vector<DTYPE>> batch;
            step_group.getDataSet("actor_batch").read(batch);
            assert(batch.size() == ActorCriticSpec::PARAMETERS::ACTOR_BATCH_SIZE);
            load<DTYPE, ReplayBufferTypeCopyTraining>(replay_buffer, batch);

            ActorCriticType::ACTOR_NETWORK_TYPE post_actor;
            lic::load(post_actor, step_group.getGroup("actor"));


            DTYPE actor_loss = lic::train_actor<lic::devices::Generic, ActorCriticSpec, ReplayBufferTypeCopyTraining::CAPACITY, typeof(rng), true>(actor_critic, replay_buffer, rng);

            if(true){//(step_i % 100 == 1){
                DTYPE diff = 0;
                for(int batch_sample_i = 0; batch_sample_i < ActorCriticSpec::PARAMETERS::ACTOR_BATCH_SIZE; batch_sample_i++){
                    DTYPE current_action[ActorCriticSpec::ENVIRONMENT::ACTION_DIM];
                    lic::evaluate(actor_critic.actor, replay_buffer.observations[batch_sample_i], current_action);
                    DTYPE desired_action[ActorCriticSpec::ENVIRONMENT::ACTION_DIM];
                    lic::evaluate(post_actor, replay_buffer.observations[batch_sample_i], desired_action);
                    diff += lic::nn::loss_functions::mse<DTYPE, ActorCriticSpec::ENVIRONMENT::ACTION_DIM, ActorCriticSpec::PARAMETERS::ACTOR_BATCH_SIZE>(current_action, desired_action);
                }
                std::cout << "action mse: " << diff << std::endl;
            }

            DTYPE pre_post_diff_per_weight = abs_diff(pre_actor, post_actor)/ActorCriticType::ACTOR_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_target_per_weight = abs_diff(post_actor, actor_critic.actor)/ActorCriticType::ACTOR_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

            DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(pre_actor, post_actor)/ActorCriticType::ACTOR_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_target_grad_per_weight = abs_diff_grad(post_actor, actor_critic.actor)/ActorCriticType::ACTOR_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

            DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(pre_actor, post_actor)/ActorCriticType::ACTOR_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_target_adam_per_weight = abs_diff_adam(post_actor, actor_critic.actor)/ActorCriticType::ACTOR_NETWORK_STRUCTURE_SPEC::NUM_WEIGHTS;
            DTYPE diff_ratio_adam = pre_post_diff_adam_per_weight/diff_target_adam_per_weight;

            if(verbose){
                std:: cout << "    actor update" << std::endl;
//                std::cout << "        pre_post_diff_per_weight: " << pre_post_diff_per_weight << std::endl;
//                std::cout << "        diff_target_per_weight: " << diff_target_per_weight << std::endl;
                std::cout << "        update ratio     : " << diff_ratio << std::endl;

//                std::cout << "        pre_post_diff_grad_per_weight: " << pre_post_diff_grad_per_weight << std::endl;
//                std::cout << "        diff_target_grad_per_weight: " << diff_target_grad_per_weight << std::endl;
                std::cout << "        update ratio grad: " << diff_ratio_grad << std::endl;

//                std::cout << "        pre_post_diff_adam_per_weight: " << pre_post_diff_adam_per_weight << std::endl;
//                std::cout << "        diff_target_adam_per_weight: " << diff_target_adam_per_weight << std::endl;
                std::cout << "        update ratio adam: " << diff_ratio_adam << std::endl;
            }

            switch(step_i){
                case 0: {
                    ASSERT_GT(diff_ratio, 1e6);
                    ASSERT_GT(diff_ratio_grad, 1e6);
                    ASSERT_GT(diff_ratio_adam, 1e6);
                }
                break;
            }

            mean_ratio_actor += diff_ratio;

            pre_actor = actor_critic.actor;
        }
        if(step_group.exist("update_target_networks")){
            if(verbose){
                std:: cout << "    target update" << std::endl;
            }
            lic::update_targets(actor_critic);
        }
        if(step_i % 100 == 0){
            if(!verbose){
                std::cout << "step_i: " << step_i << std::endl;
            }
            lic::evaluate<ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, typeof(rng), 200>(actor_critic.actor, rng, 100);
        }
    }
    mean_ratio_critic /= num_steps;
    mean_ratio_actor /= num_steps;
    std::cout << "mean_ratio_critic: " << mean_ratio_critic << std::endl;
    std::cout << "mean_ratio_actor: " << mean_ratio_actor << std::endl;
    ASSERT_GT(mean_ratio_critic, 100000);
}

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
        lic::step(off_policy_runner, actor_critic.actor, rng);
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
 */