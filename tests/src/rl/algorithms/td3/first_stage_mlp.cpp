#include <layer_in_c/operations/cpu.h>

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
    std::cout << "Using data file: " << DATA_FILE_PATH << std::endl;
    return DATA_FILE_PATH;
}
#define DTYPE double
using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
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

template <typename DEVICE, typename RB>
void load_dataset(DEVICE& device, HighFive::Group g, RB& rb){
    lic::load(device, rb.observations, g, "states");
    lic::load(device, rb.actions, g, "actions");
    lic::load(device, rb.next_observations, g, "next_states");
    auto rT = lic::view_transpose(device, rb.rewards);
    lic::load(device, rT, g, "rewards");
    std::vector<std::vector<typename RB::T>> terminated_matrix;
    g.getDataSet("terminated").read(terminated_matrix);
    assert(terminated_matrix.size() == 1);
    auto terminated = terminated_matrix[0];
    for(int i = 0; i < terminated.size(); i++){
        lic::set(rb.terminated, i, 0, terminated[i] == 1);
    }
    std::vector<std::vector<typename RB::T>> truncated_matrix;
    g.getDataSet("truncated").read(truncated_matrix);
    assert(truncated_matrix.size() == 1);
    auto truncated = truncated_matrix[0];
    for(int i = 0; i < truncated.size(); i++){
        lic::set(rb.truncated, i, 0, truncated[i] == 1);
    }
    rb.position = terminated.size();
//    g.getDataSet("states").read(rb.observations.data);
//    g.getDataSet("actions").read(rb.actions.data);
//    g.getDataSet("next_states").read(rb.next_observations.data);
//    g.getDataSet("rewards").read(rb.rewards.data);
//    g.getDataSet("terminated").read(terminated);
//    g.getDataSet("truncated").read(truncated);
}

template <typename SPEC>
typename SPEC::T assign(lic::nn::layers::dense::Layer<SPEC>& layer, const HighFive::Group g){
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

using AC_DEVICE = lic::devices::DefaultCPU;
template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T, AC_DEVICE::index_t>{
    constexpr static typename AC_DEVICE::index_t CRITIC_BATCH_SIZE = 32;
    constexpr static typename AC_DEVICE::index_t ACTOR_BATCH_SIZE = 32;
};

namespace first_stage_first_stage{
    using TD3_PARAMETERS = TD3PendulumParameters<DTYPE>;

    using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, 1>;
    using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, 1>;

    using NN_DEVICE = lic::devices::DefaultCPU;
    using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
    using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
    using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;


    using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
    using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;
}

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("0/weight").read(weights);
    acc += abs_diff_matrix<T, NT::SPEC::LAYER_1::OUTPUT_DIM, NT::SPEC::LAYER_1::INPUT_DIM>(network.layer_1.weights, weights);
    return acc;
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_CRITIC_FORWARD) {
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device(logger);
    first_stage_first_stage::NN_DEVICE nn_device(logger);
    first_stage_first_stage::ActorCriticType actor_critic;
    lic::malloc(device, actor_critic);

    std::mt19937 rng(0);
    lic::init(device, actor_critic, rng);
    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(device, actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(device, actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));

    std::vector<std::vector<DTYPE>> outputs;
    data_file.getDataSet("batch_output").read(outputs);

    for(int batch_sample_i = 0; batch_sample_i < batch.states.size(); batch_sample_i++){
        lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, first_stage_first_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM>> input;
        lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1>> output;
        lic::malloc(device, input);
        lic::malloc(device, output);
        for (int i = 0; i < batch.states[batch_sample_i].size(); i++) {
            lic::set(input, 0, i, batch.states[batch_sample_i][i]);
        }
        for (int i = 0; i < batch.actions[batch_sample_i].size(); i++) {
            lic::set(input, 0, batch.states[batch_sample_i].size() + i, batch.actions[batch_sample_i][i]);
        }

        lic::evaluate(device, actor_critic.critic_1, input, output);
        std::cout << "output: " << lic::get(output, 0, 0) << std::endl;
        ASSERT_LT(abs(lic::get(output, 0, 0) - outputs[batch_sample_i][0]), 1e-15);

        lic::evaluate(device, actor_critic.critic_target_1, input, output);
        std::cout << "output: " << lic::get(output, 0, 0) << std::endl;
        ASSERT_LT(abs(lic::get(output, 0, 0) - outputs[batch_sample_i][0]), 1e-15);
        lic::free(device, input);
        lic::free(device, output);
    }

}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_CRITIC_BACKWARD) {
//    using ActorCriticSpec = lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3_PARAMETERS>;
//    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device(logger);
    first_stage_first_stage::NN_DEVICE nn_device(logger);
    first_stage_first_stage::ActorCriticType actor_critic;
    typename first_stage_first_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::BuffersForwardBackward<> critic_buffers;
    typename first_stage_first_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::Buffers<> actor_buffers;

    lic::malloc(device, actor_critic);
    lic::malloc(device, critic_buffers);
    lic::malloc(device, actor_buffers);


    std::mt19937 rng(0);
    lic::init(device, actor_critic, rng);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(device, actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(device, actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));

    Dataset<DTYPE> batch(data_file.getGroup("batch"));
    assert(batch.states.size() == 32);

    DTYPE loss = 0;
    lic::zero_gradient(device, actor_critic.critic_1);
    for(int batch_sample_i = 0; batch_sample_i < batch.states.size(); batch_sample_i++){
//        DTYPE input[first_stage_first_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM];
        lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, first_stage_first_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM>> input;
        lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1>> output;
        lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1>> target;
        lic::malloc(device, input);
        lic::malloc(device, output);
        lic::malloc(device, target);
        for (int i = 0; i < batch.states[batch_sample_i].size(); i++) {
            lic::set(input, 0, i, batch.states[batch_sample_i][i]);
        }
        for (int i = 0; i < batch.actions[batch_sample_i].size(); i++) {
            lic::set(input, 0, batch.states[batch_sample_i].size() + i, batch.actions[batch_sample_i][i]);
        }
        lic::set(target, 0, 0, 1);
        lic::evaluate(device, actor_critic.critic_1, input, output);
        loss += lic::nn::loss_functions::mse(device, output, target);

        lic::forward_backward_mse(device, actor_critic.critic_1, input, target, critic_buffers, DTYPE(1)/32);
        std::cout << "output: " << lic::get(actor_critic.critic_1.output_layer.output, 0, 0) << std::endl;
        lic::free(device, input);
        lic::free(device, output);
        lic::free(device, target);
    }

    decltype(actor_critic.critic_1) critic_1_after_backward;
    lic::malloc(device, critic_1_after_backward);
    lic::load(device, critic_1_after_backward, data_file.getGroup("critic_1_backward"));
    lic::reset_forward_state(device, actor_critic.critic_1);
    lic::reset_forward_state(device, critic_1_after_backward);
    DTYPE diff_grad_per_weight = abs_diff_grad(device, actor_critic.critic_1, critic_1_after_backward)/first_stage_first_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
    ASSERT_LT(diff_grad_per_weight, 1e-17);

    std::cout << "diff_grad_per_weight: " << diff_grad_per_weight << std::endl;
}
namespace first_stage_second_stage{
    using TD3_PARAMETERS = TD3PendulumParameters<DTYPE>;

    using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;

    using NN_DEVICE = lic::devices::DefaultCPU;
    using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
    using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
    using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;


    using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
    using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_CRITIC_TRAINING) {
    constexpr bool verbose = true;
//    typedef lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3_PARAMETERS> ActorCriticSpec;
//    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device(logger);
    first_stage_second_stage::NN_DEVICE nn_device(logger);
    first_stage_second_stage::ActorCriticType actor_critic;
    lic::malloc(device, actor_critic);

    std::mt19937 rng(0);
    lic::init(device, actor_critic, rng);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(device, actor_critic.actor, data_file.getGroup("actor"));
    lic::load(device, actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(device, actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(device, actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(device, actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(device, actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    using DEVICE = lic::devices::DefaultCPU;
    using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, 1, 32, 100, layer_in_c::rl::components::off_policy_runner::DefaultParameters<DTYPE>>;
    using OFF_POLICY_RUNNER_TYPE = lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
    using DEVICE = lic::devices::DefaultCPU;
    using ReplayBufferType = OFF_POLICY_RUNNER_TYPE::REPLAY_BUFFER_TYPE;
    using ReplayBufferSpec = OFF_POLICY_RUNNER_TYPE::REPLAY_BUFFER_SPEC;
//    using ReplayBufferSpec = lic::rl::components::replay_buffer::Specification<DTYPE, AC_DEVICE::index_t, 3, 1, 32>;
//    using ReplayBufferType = lic::rl::components::ReplayBuffer<ReplayBufferSpec>;
    OFF_POLICY_RUNNER_TYPE off_policy_runner;
    lic::malloc(device, off_policy_runner);
    auto& replay_buffer = off_policy_runner.replay_buffers[0];
    load_dataset(device, data_file.getGroup("batch"), replay_buffer);
    if(lic::is_nan(device, replay_buffer.observations) ||lic::is_nan(device, replay_buffer.actions) ||lic::is_nan(device, replay_buffer.next_observations) ||lic::is_nan(device, replay_buffer.rewards)){
        assert(false);
    }
    static_assert(first_stage_second_stage::TD3_PARAMETERS::ACTOR_BATCH_SIZE == first_stage_second_stage::TD3_PARAMETERS::CRITIC_BATCH_SIZE, "ACTOR_BATCH_SIZE must be CRITIC_BATCH_SIZE");
    replay_buffer.position = first_stage_second_stage::TD3_PARAMETERS::ACTOR_BATCH_SIZE;

    lic::rl::components::off_policy_runner::Batch<OFF_POLICY_RUNNER_SPEC, first_stage_second_stage::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_batch;
    lic::rl::algorithms::td3::CriticTrainingBuffers<first_stage_second_stage::ActorCriticType::SPEC> critic_training_buffers;
    lic::rl::algorithms::td3::CriticTrainingBuffers<first_stage_second_stage::ActorCriticType::SPEC> critic_training_buffers_target;
    lic::malloc(device, critic_batch);
    lic::malloc(device, critic_training_buffers);
    lic::malloc(device, critic_training_buffers_target);

    first_stage_second_stage::CRITIC_NETWORK_TYPE::BuffersForwardBackward<> critic_buffers[2];
    lic::malloc(device, critic_buffers[0]);
    lic::malloc(device, critic_buffers[1]);

    first_stage_second_stage::ACTOR_NETWORK_TYPE::Buffers<> actor_buffers[2];
    lic::malloc(device, actor_buffers[0]);
    lic::malloc(device, actor_buffers[1]);

    decltype(actor_critic.critic_1) pre_critic_1;
    lic::malloc(device, pre_critic_1);
    lic::reset_optimizer_state(device, actor_critic.critic_1);
    lic::copy(device, device, pre_critic_1, actor_critic.critic_1);
    DTYPE mean_ratio = 0;
    DTYPE mean_ratio_grad = 0;
    DTYPE mean_ratio_adam = 0;
    auto critic_training_group = data_file.getGroup("critic_training");
    int num_updates = critic_training_group.getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        auto step_group = critic_training_group.getGroup(std::to_string(training_step_i));
        lic::load(device, critic_training_buffers_target.next_actions, step_group.getGroup("train_critics"), "target_next_actions_clipped");
        lic::load(device, critic_training_buffers_target.next_state_action_value_input, step_group.getGroup("train_critics"), "next_state_action_value_input");
        lic::load(device, critic_training_buffers_target.next_state_action_value_critic_1, step_group.getGroup("train_critics"), "next_state_action_values_critic_1");
        lic::load(device, critic_training_buffers_target.next_state_action_value_critic_2, step_group.getGroup("train_critics"), "next_state_action_values_critic_2");
        lic::load(device, critic_training_buffers_target.target_action_value, step_group.getGroup("train_critics"), "target_action_values");

        first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE post_critic_1;
        lic::malloc(device, post_critic_1);
        lic::load(device, post_critic_1, step_group.getGroup("critic"));

        std::vector<std::vector<DTYPE>> target_next_action_noise_vector;
        step_group.getDataSet("target_next_action_noise").read(target_next_action_noise_vector);


        for(int i = 0; i < first_stage_second_stage::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE; i++){
            for(int j = 0; j < first_stage_second_stage::ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM; j++){
                lic::set(critic_training_buffers.target_next_action_noise, i, j, target_next_action_noise_vector[i][j]);
            }
        }

        lic::gather_batch<DEVICE, OFF_POLICY_RUNNER_SPEC, first_stage_second_stage::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, decltype(rng), true>(device, off_policy_runner, critic_batch, rng);
        if(
            lic::is_nan(device, critic_batch.observations) ||
            lic::is_nan(device, critic_batch.actions) ||
            lic::is_nan(device, critic_batch.next_observations) ||
            lic::is_nan(device, critic_batch.rewards)
        ){
            assert(false);
        }
//        assert(!lic::is_nan(device, actor_critic));

        lic::train_critic(device, actor_critic, actor_critic.critic_1, critic_batch, actor_buffers[0], critic_buffers[0], critic_training_buffers);

        auto target_next_action_diff = lic::abs_diff(device, critic_training_buffers_target.next_actions, critic_training_buffers.next_actions);
        auto next_state_action_value_input_diff = lic::abs_diff(device, critic_training_buffers_target.next_state_action_value_input, critic_training_buffers.next_state_action_value_input);
        auto next_state_action_value_critic_1_diff = lic::abs_diff(device, critic_training_buffers_target.next_state_action_value_critic_1, critic_training_buffers.next_state_action_value_critic_1);
        auto next_state_action_value_critic_2_diff = lic::abs_diff(device, critic_training_buffers_target.next_state_action_value_critic_2, critic_training_buffers.next_state_action_value_critic_2);
        auto target_action_value_diff = lic::abs_diff(device, critic_training_buffers_target.target_action_value, critic_training_buffers.target_action_value);
        ASSERT_LT(target_next_action_diff, 1e-14);
        ASSERT_LT(next_state_action_value_input_diff, 1e-14);
        ASSERT_LT(next_state_action_value_critic_1_diff, 1e-14);
        ASSERT_LT(next_state_action_value_critic_2_diff, 1e-14);
        ASSERT_LT(target_action_value_diff, 1e-14);

        lic::reset_forward_state(device, pre_critic_1);
        lic::reset_forward_state(device, post_critic_1);
        lic::reset_forward_state(device, actor_critic.critic_1);

        DTYPE pre_post_diff_per_weight = abs_diff(device, pre_critic_1, post_critic_1)/first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_per_weight = abs_diff(device, post_critic_1, actor_critic.critic_1)/first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

        DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(device, pre_critic_1, post_critic_1)/first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_grad_per_weight = abs_diff_grad(device, post_critic_1, actor_critic.critic_1)/first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

        DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(device, pre_critic_1, post_critic_1)/first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_adam_per_weight = abs_diff_adam(device, post_critic_1, actor_critic.critic_1)/first_stage_second_stage::ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
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
    lic::free(device, replay_buffer);
}


TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_FIRST_STAGE, TEST_ACTOR_TRAINING) {
    constexpr bool verbose = true;
//    typedef lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3_PARAMETERS> ActorCriticSpec;
//    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, ActorCriticSpec> ActorCriticType;
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device(logger);
    first_stage_second_stage::NN_DEVICE nn_device(logger);
    first_stage_second_stage::ActorCriticType actor_critic;
    lic::malloc(device, actor_critic);

    std::mt19937 rng(0);
    lic::init(device, actor_critic, rng);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    lic::load(device, actor_critic.actor, data_file.getGroup("actor"));
    lic::load(device, actor_critic.actor_target, data_file.getGroup("actor_target"));
    lic::load(device, actor_critic.critic_1, data_file.getGroup("critic_1"));
    lic::load(device, actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    lic::load(device, actor_critic.critic_2, data_file.getGroup("critic_2"));
    lic::load(device, actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    using DEVICE = lic::devices::DefaultCPU;
//    using ReplayBufferSpec = lic::rl::components::replay_buffer::Specification<DTYPE, AC_DEVICE::index_t, 3, 1, 32>;
//    using ReplayBufferType = lic::rl::components::ReplayBuffer<ReplayBufferSpec>;
    using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, 1, 32, 100, layer_in_c::rl::components::off_policy_runner::DefaultParameters<DTYPE>>;
    using OFF_POLICY_RUNNER_TYPE = lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
    OFF_POLICY_RUNNER_TYPE off_policy_runner;
    lic::malloc(device, off_policy_runner);
    auto& replay_buffer = off_policy_runner.replay_buffers[0];
    load_dataset(device, data_file.getGroup("batch"), replay_buffer);
    static_assert(first_stage_second_stage::TD3_PARAMETERS::ACTOR_BATCH_SIZE == first_stage_second_stage::TD3_PARAMETERS::CRITIC_BATCH_SIZE, "ACTOR_BATCH_SIZE must be CRITIC_BATCH_SIZE");
    replay_buffer.position = first_stage_second_stage::TD3_PARAMETERS::ACTOR_BATCH_SIZE;

    lic::rl::components::off_policy_runner::Batch<OFF_POLICY_RUNNER_SPEC, first_stage_second_stage::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_batch;
    lic::rl::algorithms::td3::ActorTrainingBuffers<first_stage_second_stage::ActorCriticType::SPEC> actor_training_buffers;
    lic::malloc(device, actor_batch);
    lic::malloc(device, actor_training_buffers);

    first_stage_second_stage::CRITIC_NETWORK_TYPE::Buffers<> critic_buffers[2];
    lic::malloc(device, critic_buffers[0]);
    lic::malloc(device, critic_buffers[1]);

    first_stage_second_stage::ACTOR_NETWORK_TYPE::Buffers<> actor_buffers[2];
    lic::malloc(device, actor_buffers[0]);
    lic::malloc(device, actor_buffers[1]);


    decltype(actor_critic.actor) pre_actor;
    lic::malloc(device, pre_actor);
    lic::copy(device, device, pre_actor, actor_critic.actor);
    lic::reset_optimizer_state(device, actor_critic.actor);
    DTYPE mean_ratio = 0;
    DTYPE mean_ratio_grad = 0;
    DTYPE mean_ratio_adam = 0;
    int num_updates = data_file.getGroup("actor_training").getNumberObjects();
    for(int training_step_i = 0; training_step_i < num_updates; training_step_i++){
        decltype(actor_critic.actor) post_actor;
        lic::malloc(device, post_actor);
        std::stringstream ss;
        ss << "actor_training/" << training_step_i;
        lic::load(device, post_actor, data_file.getGroup(ss.str()));

//        DTYPE actor_1_loss = lic::train_actor<AC_DEVICE, ActorCriticType::SPEC, ReplayBufferType::CAPACITY, typeof(rng), true>(device, actor_critic, replay_buffer, rng);
        lic::gather_batch<DEVICE, OFF_POLICY_RUNNER_SPEC , first_stage_second_stage::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE, decltype(rng), true>(device, off_policy_runner, actor_batch, rng);
        DTYPE actor_1_loss = lic::train_actor(device, actor_critic, actor_batch, actor_buffers[0], critic_buffers[0], actor_training_buffers);

        lic::reset_forward_state(device, pre_actor);
        lic::reset_forward_state(device, post_actor);
        lic::reset_forward_state(device, actor_critic.actor);

        DTYPE pre_post_diff_per_weight = abs_diff(device, pre_actor, post_actor)/first_stage_second_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_per_weight = abs_diff(device, post_actor, actor_critic.actor)/first_stage_second_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

        DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(device, pre_actor, post_actor)/first_stage_second_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_grad_per_weight = abs_diff_grad(device, post_actor, actor_critic.actor)/first_stage_second_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

        DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(device, pre_actor, post_actor)/first_stage_second_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
        DTYPE diff_target_adam_per_weight = abs_diff_adam(device, post_actor, actor_critic.actor)/first_stage_second_stage::ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
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
    lic::free(device, replay_buffer);
}
