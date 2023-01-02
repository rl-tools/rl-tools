#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>

#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>

#include "../../../utils/utils.h"
#include "../../../utils/nn_comparison_mlp.h"

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace lic = layer_in_c;
std::string get_data_file_path(){
    std::string DATA_FILE_PATH = "../multirotor-torch/model_first_stage.hdf5";
    const char* data_file_path = std::getenv("LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FIRST_STAGE_DATA_FILE");
    if (data_file_path != NULL){
        DATA_FILE_PATH = std::string(data_file_path);
//            std::runtime_error("Environment variable LAYER_IN_C_TEST_DATA_DIR not set. Skipping test.");
    }
    return DATA_FILE_PATH;
}
#define DTYPE double
typedef lic::rl::environments::pendulum::Specification<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<lic::devices::CPU, PENDULUM_SPEC> ENVIRONMENT;
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
struct TD3Parameters: public lic::rl::algorithms::td3::DefaultParameters<T>{
    constexpr static int CRITIC_BATCH_SIZE = 32;
    constexpr static int ACTOR_BATCH_SIZE = 32;
};
struct ActorStructureSpec{
    using T = DTYPE;
    static constexpr size_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr size_t OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
};

struct CriticStructureSpec{
    using T = DTYPE;
    static constexpr size_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM;
    static constexpr size_t OUTPUT_DIM = 1;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
};

template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T>{
    constexpr static size_t CRITIC_BATCH_SIZE = 32;
    constexpr static size_t ACTOR_BATCH_SIZE = 32;
};

using NN_DEVICE = lic::devices::CPU;
using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<NN_DEVICE, ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<NN_DEVICE, ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<NN_DEVICE, ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<NN_DEVICE , ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<NN_DEVICE, CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<NN_DEVICE, CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<NN_DEVICE, CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<NN_DEVICE, CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<lic::devices::CPU, TD3_SPEC>;

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("0/weight").read(weights);
    acc += abs_diff_matrix<T, NT::SPEC::LAYER_1::OUTPUT_DIM, NT::SPEC::LAYER_1::INPUT_DIM>(network.layer_1.weights, weights);
    return acc;
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_CRITIC_FORWARD) {
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init(actor_critic, rng);
    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));

    std::vector<std::vector<DTYPE>> outputs;
    data_file.getDataSet("batch_output").read(outputs);

    for(int batch_sample_i = 0; batch_sample_i < batch.states.size(); batch_sample_i++){
        DTYPE input[ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM];
        for (int i = 0; i < batch.states[batch_sample_i].size(); i++) {
            input[i] = batch.states[batch_sample_i][i];
        }
        for (int i = 0; i < batch.actions[batch_sample_i].size(); i++) {
            input[batch.states[batch_sample_i].size() + i] = batch.actions[batch_sample_i][i];
        }

        DTYPE output[1];
        lic::evaluate(actor_critic.critic_1, input, output);
        std::cout << "output: " << output[0] << std::endl;
        ASSERT_LT(abs(output[0] - outputs[batch_sample_i][0]), 1e-15);

        lic::evaluate(actor_critic.critic_target_1, input, output);
        std::cout << "output: " << output[0] << std::endl;
        ASSERT_LT(abs(output[0] - outputs[batch_sample_i][0]), 1e-15);
    }

}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_CRITIC_BACKWARD) {
//    using ActorCriticSpec = lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>>;
//    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init(actor_critic, rng);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));
    assert(batch.states.size() == 32);

    DTYPE loss = 0;
    lic::zero_gradient(actor_critic.critic_1);
    for(int batch_sample_i = 0; batch_sample_i < batch.states.size(); batch_sample_i++){
        DTYPE input[ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM];
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

        lic::forward_backward_mse<decltype(actor_critic.critic_1)::DEVICE, decltype(actor_critic.critic_1)::SPEC, 32>(actor_critic.critic_1, input, target);
        std::cout << "output: " << actor_critic.critic_1.output_layer.output[0] << std::endl;
    }

    auto critic_1_after_backward = actor_critic.critic_1;
    lic::load(critic_1_after_backward, data_file.getGroup("critic_1_backward"));
    DTYPE diff_grad_per_weight = abs_diff_grad(actor_critic.critic_1, critic_1_after_backward)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
    ASSERT_LT(diff_grad_per_weight, 1e-17);

    std::cout << "diff_grad_per_weight: " << diff_grad_per_weight << std::endl;
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_CRITIC_TRAINING) {
    constexpr bool verbose = true;
//    typedef lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>> ActorCriticSpec;
//    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init(actor_critic, rng);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(actor_critic.actor, data_file.getGroup("actor"));
    lic::load(actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    using ReplayBufferSpec = lic::rl::components::replay_buffer::Spec<DTYPE, 3, 1, 100>;
    using ReplayBufferType = lic::rl::components::ReplayBuffer<lic::devices::Generic, ReplayBufferSpec>;
    ReplayBufferType replay_buffer;
    load_dataset(data_file.getGroup("batch"), replay_buffer);
    static_assert(TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE == TD3Parameters<DTYPE>::CRITIC_BATCH_SIZE, "ACTOR_BATCH_SIZE must be CRITIC_BATCH_SIZE");
    replay_buffer.position = TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE;

    auto pre_critic_1 = actor_critic.critic_1;
    lic::reset_optimizer_state(actor_critic.critic_1);
    DTYPE mean_ratio = 0;
    DTYPE mean_ratio_grad = 0;
    DTYPE mean_ratio_adam = 0;
    auto critic_training_group = data_file.getGroup("critic_training");
    int num_updates = critic_training_group.getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        auto step_group = critic_training_group.getGroup(std::to_string(training_step_i));

        ActorCriticType::SPEC::CRITIC_NETWORK_TYPE post_critic_1;
        lic::load(post_critic_1, step_group.getGroup("critic"));

        std::vector<std::vector<DTYPE>> target_next_action_noise_vector;
        step_group.getDataSet("target_next_action_noise").read(target_next_action_noise_vector);


        DTYPE target_next_action_noise[ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE][ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
        for(int i = 0; i < ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE; i++){
            for(int j = 0; j < ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM; j++){
                target_next_action_noise[i][j] = target_next_action_noise_vector[i][j];
            }
        }

        DTYPE critic_1_loss = lic::train_critic<
                decltype(actor_critic)::SPEC,
                decltype(actor_critic.critic_1),
                decltype(replay_buffer)::DEVICE,
                decltype(replay_buffer)::CAPACITY,
                decltype(rng),
                true
        >(actor_critic, actor_critic.critic_1, replay_buffer, target_next_action_noise, rng);

        DTYPE pre_post_diff_per_weight = abs_diff(pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_per_weight = abs_diff(post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

        DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_grad_per_weight = abs_diff_grad(post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

        DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_adam_per_weight = abs_diff_adam(post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
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

        mean_ratio += diff_ratio;
        mean_ratio_grad += diff_ratio_grad;
        mean_ratio_adam += diff_ratio_adam;

//        ASSERT_LT(diff_target_per_weight, 1e-7);
//        actor_critic.critic_1 = post_critic;

    }
    mean_ratio /= num_updates;
    mean_ratio_grad /= num_updates;
    mean_ratio_adam /= num_updates;
    std::cout << "mean_ratio: " << mean_ratio << std::endl;
    std::cout << "mean_ratio grad: " << mean_ratio_grad << std::endl;
    std::cout << "mean_ratio adam: " << mean_ratio_adam << std::endl;
    ASSERT_GT(mean_ratio, 1e14);
    ASSERT_GT(mean_ratio_grad, 1e14);
    ASSERT_GT(mean_ratio_adam, 1e14);
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_ACTOR_TRAINING) {
    constexpr bool verbose = true;
//    typedef lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3Parameters<DTYPE>> ActorCriticSpec;
//    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    ActorCriticType actor_critic;

    std::mt19937 rng(0);
    lic::init(actor_critic, rng);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(actor_critic.actor, data_file.getGroup("actor"));
    lic::load(actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    using ReplayBufferSpec = lic::rl::components::replay_buffer::Spec<DTYPE, 3, 1, 100>;
    using ReplayBufferType = lic::rl::components::ReplayBuffer<lic::devices::Generic, ReplayBufferSpec>;
    ReplayBufferType replay_buffer;
    load_dataset(data_file.getGroup("batch"), replay_buffer);
    static_assert(TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE == TD3Parameters<DTYPE>::CRITIC_BATCH_SIZE, "ACTOR_BATCH_SIZE must be CRITIC_BATCH_SIZE");
    replay_buffer.position = TD3Parameters<DTYPE>::ACTOR_BATCH_SIZE;

    auto pre_actor = actor_critic.actor;
    lic::reset_optimizer_state(actor_critic.actor);
    DTYPE mean_ratio = 0;
    DTYPE mean_ratio_grad = 0;
    DTYPE mean_ratio_adam = 0;
    int num_updates = data_file.getGroup("actor_training").getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        auto post_actor = actor_critic.actor;
        std::stringstream ss;
        ss << "actor_training/" << training_step_i;
        lic::load(post_actor, data_file.getGroup(ss.str()));

        DTYPE actor_1_loss = lic::train_actor<ActorCriticType::SPEC, ReplayBufferType::DEVICE, ReplayBufferType::CAPACITY, typeof(rng), true>(actor_critic, replay_buffer, rng);

        DTYPE pre_post_diff_per_weight = abs_diff(pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_per_weight = abs_diff(post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

        DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_grad_per_weight = abs_diff_grad(post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

        DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_adam_per_weight = abs_diff_adam(post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
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

        mean_ratio += diff_ratio;
        mean_ratio_grad += diff_ratio_grad;
        mean_ratio_adam += diff_ratio_adam;

//        ASSERT_LT(diff_target_per_weight, 1e-7);
//        actor_critic.critic_1 = post_critic;

    }
    mean_ratio /= num_updates;
    mean_ratio_grad /= num_updates;
    mean_ratio_adam /= num_updates;
    std::cout << "mean_ratio: " << mean_ratio << std::endl;
    std::cout << "mean_ratio_grad: " << mean_ratio_grad << std::endl;
    std::cout << "mean_ratio_adam: " << mean_ratio_adam << std::endl;
    ASSERT_GT(mean_ratio, 1e-15); // TANH introduces a lot of inaccuracy
    ASSERT_GT(mean_ratio_grad, 1e-15);
    ASSERT_GT(mean_ratio_adam, 1e-15);
}
