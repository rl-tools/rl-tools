#include <rl_tools/operations/cpu.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <rl_tools/rl/environments/operations_cpu.h>
#include <rl_tools/rl/algorithms/td3/operations_cpu.h>

#include <rl_tools/nn_models/persist.h>
#include <rl_tools/rl/utils/evaluation.h>
#include <rl_tools/utils/generic/memcpy.h>

#include "../../../utils/utils.h"
#include "../../../utils/nn_comparison_mlp.h"

#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
#include <rl_tools/rl/environments/pendulum/ui.h>
#include <rl_tools/rl/utils/evaluation_visual.h>
#endif

#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


std::string get_data_file_path(){
    std::string DATA_FILE_NAME = "model_second_stage.hdf5";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TESTS_DATA_PATH);
    std::string DATA_FILE_PATH = std::string(data_path_stub) + "/" + DATA_FILE_NAME;
    return DATA_FILE_PATH;
}
#define DTYPE double
using DEVICE = rlt::devices::DefaultCPU;
typedef rlt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, rlt::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
typedef rlt::rl::environments::pendulum::UI<DTYPE> UI;
#endif
ENVIRONMENT env;

using AC_DEVICE = rlt::devices::DefaultCPU;

struct TD3ParametersCopyTraining: public rlt::rl::algorithms::td3::DefaultParameters<DTYPE, AC_DEVICE::index_t>{
    constexpr static typename AC_DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename AC_DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using ActorStructureSpec = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH, TD3ParametersCopyTraining::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY, TD3ParametersCopyTraining::CRITIC_BATCH_SIZE>;

using NN_DEVICE = rlt::devices::DefaultCPU;
using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<DTYPE, typename DEVICE::index_t>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using ACTOR_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
using ACTOR_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
using CRITIC_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;


using TD3_SPEC = rlt::rl::algorithms::td3::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3ParametersCopyTraining>;
using ActorCriticType = rlt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


TEST(RL_TOOLS_RL_ALGORITHMS_TD3_MLP_SECOND_STAGE, TEST_LOADING_TRAINED_ACTOR) {
    constexpr bool verbose = false;
    AC_DEVICE device;
    NN_DEVICE nn_device;
    ActorCriticType actor_critic;
    actor_critic.actor_optimizer.parameters = rlt::nn::optimizers::adam::default_parameters_torch<DTYPE>;
    actor_critic.critic_optimizers[0].parameters = rlt::nn::optimizers::adam::default_parameters_torch<DTYPE>;
    actor_critic.critic_optimizers[1].parameters = rlt::nn::optimizers::adam::default_parameters_torch<DTYPE>;

    ActorCriticType::SPEC::ACTOR_TYPE::Buffer<1> eval_buffers;
    rlt::malloc(device, actor_critic);
    rlt::malloc(device, eval_buffers);

    std::mt19937 rng(0);

    rlt::rl::environments::DummyUI ui;

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    int step = data_file.getGroup("full_training").getGroup("steps").getNumberObjects()-1;
    assert(step >= 0);
    auto step_group = data_file.getGroup("full_training").getGroup("steps").getGroup(std::to_string(step));
    rlt::load(device, actor_critic.actor, step_group.getGroup("actor"));
    auto result = rlt::evaluate(device, env, ui, actor_critic.actor, rlt::rl::utils::evaluation::Specification<100, 200>(), eval_buffers, rng, true);
    std::cout << "mean return: " << result.returns_mean << std::endl;
}

//using ReplayBufferSpecCopyTraining = rlt::rl::components::replay_buffer::Specification<DTYPE, AC_DEVICE::index_t, 3, 1, 1000>;
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, 1, false, 1000, 100>;
using OFF_POLICY_RUNNER_TYPE = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
using DEVICE = rlt::devices::DefaultCPU;
typedef OFF_POLICY_RUNNER_TYPE::REPLAY_BUFFER_TYPE ReplayBufferTypeCopyTraining;
constexpr int BATCH_DIM = ENVIRONMENT::OBSERVATION_DIM * 2 + ENVIRONMENT::ACTION_DIM + 2;
template <typename DEVICE, typename T>
void load(DEVICE& device, ReplayBufferTypeCopyTraining& rb, std::vector<std::vector<T>> batch){
    for(int i = 0; i < batch.size(); i++){
//        rlt::utils::memcpy(&rb.     rlt::get(observations, i, 0), &batch[i][0], ENVIRONMENT::OBSERVATION_DIM);
        rlt::assign(device, &batch[i][0], rb.observations, i, 0, 1, ENVIRONMENT::OBSERVATION_DIM);
//        rlt::utils::memcpy(&rb.          rlt::get(actions, i, 0), &batch[i][ENVIRONMENT::OBSERVATION_DIM], ENVIRONMENT::ACTION_DIM);
        rlt::assign(device, &batch[i][ENVIRONMENT::OBSERVATION_DIM], rb.actions, i, 0, 1, ENVIRONMENT::ACTION_DIM);
//        rlt::utils::memcpy(&rlt::get(rb.next_observations, i, 0), &batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM], ENVIRONMENT::OBSERVATION_DIM);
        rlt::assign(device, &batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM], rb.next_observations, i, 0, 1, ENVIRONMENT::OBSERVATION_DIM);
        rlt::set(rb.rewards, i, 0, batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM]);
        rlt::set(rb.terminated, i, 0, batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM + 1] == 1);
        rlt::set(rb.truncated, i, 0, batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM + 2] == 1);
    }
    rb.position = batch.size();
}
TEST(RL_TOOLS_RL_ALGORITHMS_TD3_MLP_SECOND_STAGE, FP_ACC) {
    for(int i = 0; i < 1000; i++){
        std::normal_distribution<float> dist;
        auto rng = std::mt19937(0);
        float a = dist(rng) * 5e-3;
        float b = dist(rng) / 10;
        float aa = dist(rng);

        float c = a * b;
        c += aa;
        float d = c / b;
        d -= aa / b;
        float e = a - d;

//        std::cout << e << std::endl;
    }
    for(int i = 0; i < 1000; i++){
        std::normal_distribution<double> dist;
        auto rng = std::mt19937(0);
        double a = dist(rng) * 5e-3;
        double b = dist(rng) / 10;
        double aa = dist(rng);

        double c = a * b;
        c += aa;
        double d = c / b;
        d -= aa / b;
        double e = a - d;

//        std::cout << e << std::endl;
    }
}
TEST(RL_TOOLS_RL_ALGORITHMS_TD3_MLP_SECOND_STAGE, TEST_COPY_TRAINING) {
#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
    UI ui;
#endif
    constexpr bool verbose = true;
    AC_DEVICE device;
    NN_DEVICE nn_device;
    ActorCriticType actor_critic;
    actor_critic.actor_optimizer.parameters = rlt::nn::optimizers::adam::default_parameters_torch<DTYPE>;
    actor_critic.critic_optimizers[0].parameters = rlt::nn::optimizers::adam::default_parameters_torch<DTYPE>;
    actor_critic.critic_optimizers[1].parameters = rlt::nn::optimizers::adam::default_parameters_torch<DTYPE>;
    ActorCriticType::SPEC::ACTOR_TYPE::Buffer<1> actor_eval_buffers;
    rlt::malloc(device, actor_critic);
    rlt::malloc(device, actor_eval_buffers);

    std::mt19937 rng(0);
    rlt::init(device, actor_critic,rng);


    rlt::rl::environments::DummyUI ui;



    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    rlt::load(device, actor_critic.actor, data_file.getGroup("actor"));
    rlt::load(device, actor_critic.actor_target, data_file.getGroup("actor_target"));
    rlt::load(device, actor_critic.critic_1, data_file.getGroup("critic_1"));
    rlt::load(device, actor_critic.critic_target_1, data_file.getGroup("critic_target_1"));
    rlt::load(device, actor_critic.critic_2, data_file.getGroup("critic_2"));
    rlt::load(device, actor_critic.critic_target_2, data_file.getGroup("critic_target_2"));

    OFF_POLICY_RUNNER_TYPE off_policy_runner;
    rlt::malloc(device, off_policy_runner);

    rlt::reset_optimizer_state(device, actor_critic.actor_optimizer     , actor_critic.actor   );
    rlt::reset_optimizer_state(device, actor_critic.critic_optimizers[0], actor_critic.critic_1);
    rlt::reset_optimizer_state(device, actor_critic.critic_optimizers[1], actor_critic.critic_2);
    DTYPE mean_ratio_critic = 0;
    DTYPE mean_ratio_critic_grad = 0;
    DTYPE mean_ratio_critic_adam = 0;
    DTYPE mean_ratio_actor = 0;
    DTYPE mean_ratio_actor_grad = 0;
    DTYPE mean_ratio_actor_adam = 0;
    DTYPE mean_ratio_critic_target = 0;
    auto full_training_group = data_file.getGroup("full_training");
    auto steps_group = full_training_group.getGroup("steps");
    int num_steps = std::min(steps_group.getNumberObjects(), (typename DEVICE::index_t)1000);
    decltype(actor_critic.critic_1) pre_critic_1;
    rlt::malloc(device, pre_critic_1);
    rlt::copy(device, device, actor_critic.critic_1, pre_critic_1);
    decltype(actor_critic.actor) pre_actor;
    rlt::malloc(device, pre_actor);
    rlt::copy(device, device, actor_critic.actor, pre_actor);
    decltype(actor_critic.critic_target_1) pre_critic_1_target;
    rlt::malloc(device, pre_critic_1_target);
    rlt::copy(device, device, actor_critic.critic_target_1, pre_critic_1_target);

    using CRITIC_BATCH_SPEC = rlt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
    rlt::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batch;
    rlt::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    CRITIC_TYPE::Buffer<> critic_buffers[2];
    rlt::malloc(device, critic_batch);
    rlt::malloc(device, critic_training_buffers);
    rlt::malloc(device, critic_buffers[0]);
    rlt::malloc(device, critic_buffers[1]);

    using ACTOR_BATCH_SPEC = rlt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
    rlt::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
    rlt::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    ACTOR_TYPE::Buffer<> actor_buffers[2];
    rlt::malloc(device, actor_batch);
    rlt::malloc(device, actor_training_buffers);
    rlt::malloc(device, actor_buffers[0]);
    rlt::malloc(device, actor_buffers[1]);

    for(int step_i = 0; step_i < num_steps; step_i++){
        if(verbose){
            std::cout << "step_i: " << step_i << std::endl;
        }
        auto step_group = steps_group.getGroup(std::to_string(step_i));
        if(step_group.exist("critics_batch")){
            std::vector<std::vector<DTYPE>> batch;
            step_group.getDataSet("critics_batch").read(batch);
            assert(batch.size() == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

//            step_group.getDataSet("target_next_action_noise").read(critic_training_buffers.target_next_action_noise.data);
            rlt::load(device, critic_training_buffers.target_next_action_noise, step_group, "target_next_action_noise");

            load(device, off_policy_runner.replay_buffers[0], batch);
//            if (step_i == 0 && step_group.exist("pre_critic1")){
//                decltype(actor_critic.critic_1) pre_critic_1_step;
//                rlt::malloc(device, pre_critic_1_step);
//                rlt::load(device, pre_critic_1_step, step_group.getGroup("pre_critic1"));
//                rlt::reset_forward_state(device, pre_critic_1_step);
//                rlt::reset_forward_state(device, actor_critic.critic_1);
//                DTYPE pre_current_diff = abs_diff(device, pre_critic_1_step, actor_critic.critic_1);
//                ASSERT_EQ(pre_current_diff, 0);
//                rlt::free(device, pre_critic_1_step);
//            }

            decltype(actor_critic.critic_1) post_critic_1;// = actor_critic.critic_1;
            rlt::malloc(device, post_critic_1);
            rlt::load(device, post_critic_1, step_group.getGroup("critic1"));


            rlt::gather_batch<DEVICE, OFF_POLICY_RUNNER_SPEC, CRITIC_BATCH_SPEC, decltype(rng), true>(device, off_policy_runner, critic_batch, rng);
            rlt::train_critic(device, actor_critic, actor_critic.critic_1, critic_batch, actor_critic.critic_optimizers[0], actor_buffers[0], critic_buffers[0], critic_training_buffers);


            rlt::reset_forward_state(device, pre_critic_1);
            rlt::reset_forward_state(device, post_critic_1);
            rlt::reset_forward_state(device, actor_critic.critic_1);
            DTYPE pre_post_diff_per_weight = abs_diff(device, pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_per_weight = abs_diff(device, post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

            DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(device, pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_grad_per_weight = abs_diff_grad(device, post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

            DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(device, pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_adam_per_weight = abs_diff_adam(device, post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
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
            if(diff_ratio < 1e10){
//                std::cout << "something wrong here" << std::endl;
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
            mean_ratio_critic_grad += diff_ratio_grad;
            mean_ratio_critic_adam += diff_ratio_adam;

            {
//                step_group.getDataSet("target_next_action_noise").read(critic_training_buffers.target_next_action_noise.data);
                rlt::load(device, critic_training_buffers.target_next_action_noise, step_group, "target_next_action_noise");

                rlt::gather_batch<DEVICE, OFF_POLICY_RUNNER_SPEC, CRITIC_BATCH_SPEC, decltype(rng), true>(device, off_policy_runner, critic_batch, rng);
                rlt::train_critic(device, actor_critic, actor_critic.critic_2, critic_batch, actor_critic.critic_optimizers[1], actor_buffers[0], critic_buffers[0], critic_training_buffers);
            }
            rlt::copy(device, device, actor_critic.critic_1, pre_critic_1);

//            if(false){//(step_i % 100 == 0){
//                DTYPE diff = 0;
//                for(int batch_sample_i = 0; batch_sample_i < ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
//                    DTYPE input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
//                    rlt::utils::memcpy(input, &rlt::get(replay_buffer.observations, batch_sample_i, 0), ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM);
//                    rlt::utils::memcpy(&input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM], &rlt::get(replay_buffer.actions, batch_sample_i, 0), ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM);
//                    using input_layout = rlt::matrix::layouts::RowMajorAlignment<DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM, 1>;
//                    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM, layout>> input_matrix = {input};
//                    DTYPE current_value;
//                    using current_value_layout = rlt::matrix::layouts::RowMajorAlignment<DEVICE::index_t, 1, 1, 1>;
//                    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1, current_value_layout>> current_value_matrix = {&current_value};
//                    rlt::evaluate(device, actor_critic.critic_1, input_matrix, current_value_matrix);
////                    DTYPE desired_value;
////                    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1>> desired_value_matrix = {&desired_value};
////                    rlt::evaluate(device, post_critic_1, input_matrix, desired_value_matrix);
////                    diff += (current_value - desired_value) * (current_value - desired_value) / ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
//                }
////                std::cout << "value mse: " << diff << std::endl;
//            }
            rlt::free(device, post_critic_1);
        }

        if(step_group.exist("actor_batch")){
            std::vector<std::vector<DTYPE>> batch;
            step_group.getDataSet("actor_batch").read(batch);
            assert(batch.size() == ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
            load(device, off_policy_runner.replay_buffers[0], batch);

            decltype(actor_critic.actor) post_actor;
            rlt::malloc(device, post_actor);
            rlt::load(device, post_actor, step_group.getGroup("actor"));

            decltype(actor_critic.actor) pre_actor_loaded;
            rlt::malloc(device, pre_actor_loaded);
            rlt::load(device, pre_actor_loaded, step_group.getGroup("pre_actor"));
            rlt::reset_forward_state(device, pre_actor_loaded);
            rlt::reset_forward_state(device, actor_critic.actor);
            DTYPE pre_current_diff = abs_diff(device, pre_actor_loaded, actor_critic.actor);
            if(step_i == 0){
                ASSERT_EQ(pre_current_diff, 0);
            }


            {
                rlt::gather_batch<DEVICE, OFF_POLICY_RUNNER_SPEC, ACTOR_BATCH_SPEC, decltype(rng), true>(device, off_policy_runner, actor_batch, rng);
                rlt::train_actor(device, actor_critic, actor_batch, actor_critic.actor_optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
            }
//            DTYPE actor_loss = rlt::train_actor<AC_DEVICE, ActorCriticType::SPEC, decltype(replay_buffer)::CAPACITY, typeof(rng), true>(device, actor_critic, replay_buffer, rng);

//            if(true){//(step_i % 100 == 1){
//                DTYPE diff = 0;
//                for(int batch_sample_i = 0; batch_sample_i < ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE; batch_sample_i++){
//                    DTYPE current_action[ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
//                    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> current_action_matrix = {current_action};
//                    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM>> observation_matrix = {&replay_buffer.observations.data[batch_sample_i]};
//                    rlt::evaluate(device, actor_critic.actor, observation_matrix, current_action_matrix);
//                    DTYPE desired_action[ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
//                    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> desired_action_matrix = {desired_action};
//                    rlt::evaluate(device, post_actor, observation_matrix, desired_action_matrix);
//                    diff += rlt::nn::loss_functions::mse::evaluate(device, current_action_matrix, desired_action_matrix, DTYPE(1)/ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
//                }
////                std::cout << "action mse: " << diff << std::endl;
//            }

            rlt::reset_forward_state(device, pre_actor);
            rlt::reset_forward_state(device, post_actor);
            rlt::reset_forward_state(device, actor_critic.actor);

            DTYPE pre_post_diff_per_weight = abs_diff(device, pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_per_weight = abs_diff(device, post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

            DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(device, pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_grad_per_weight = abs_diff_grad(device, post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

            DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(device, pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_adam_per_weight = abs_diff_adam(device, post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_TYPE::NUM_WEIGHTS;
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
            mean_ratio_actor_grad += diff_ratio_grad;
            mean_ratio_actor_adam += diff_ratio_adam;

            rlt::copy(device, device, actor_critic.actor, pre_actor);
            rlt::free(device, post_actor);
            rlt::free(device, pre_actor_loaded);
        }
        if(step_group.exist("critic1_target")){
            if(verbose){
                std:: cout << "    target update" << std::endl;
            }
            if (step_i == 0){
                decltype(actor_critic.critic_target_1) pre_critic_1_target_step;
                rlt::malloc(device, pre_critic_1_target_step);
                rlt::load(device, pre_critic_1_target_step, step_group.getGroup("pre_critic1_target"));
                DTYPE pre_current_diff = abs_diff(device, pre_critic_1_target_step, actor_critic.critic_target_1);
                ASSERT_EQ(pre_current_diff, 0);
                rlt::free(device, pre_critic_1_target_step);
            }
            else{
                if (step_i >= ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE){

                    decltype(actor_critic.critic_target_1) post_critic_1_target;
                    rlt::malloc(device, post_critic_1_target);
                    rlt::load(device, post_critic_1_target, step_group.getGroup("critic1_target"));

                    rlt::update_critic_targets(device, actor_critic);
                    rlt::update_actor_target(device, actor_critic);

                    DTYPE pre_post_diff_per_weight = abs_diff(device, pre_critic_1_target, post_critic_1_target)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
                    DTYPE diff_target_per_weight = abs_diff(device, post_critic_1_target, actor_critic.critic_target_1)/ActorCriticType::SPEC::CRITIC_TYPE::NUM_WEIGHTS;
                    DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

                    if(verbose){
                        std::cout << "    critic target update" << std::endl;
//                        std::cout << "        pre_post_diff_per_weight: " << pre_post_diff_per_weight << std::endl;
//                        std::cout << "        diff_target_per_weight: " << diff_target_per_weight << std::endl;
                        std::cout << "        update ratio     : " << diff_ratio << std::endl;
                    }

                    switch(step_i){
                        case 0: {
                            ASSERT_GT(diff_ratio, 1e6);
                        }
                            break;
                    }

                    mean_ratio_critic_target += diff_ratio;

                    rlt::copy(device, device, actor_critic.critic_target_1, pre_critic_1_target);

//                    if(true){//(step_i % 100 == 0){
//                        DTYPE diff = 0;
//                        for(int batch_sample_i = 0; batch_sample_i < ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
//                            DTYPE input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
//                            rlt::utils::memcpy(input, &replay_buffer.observations.data[batch_sample_i*ENVIRONMENT::OBSERVATION_DIM], ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM);
//                            rlt::utils::memcpy(&input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM], &replay_buffer.actions.data[batch_sample_i*ENVIRONMENT::ACTION_DIM], ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM);
//                            rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> input_matrix = {input};
//                            DTYPE current_value;
//                            rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1>> current_value_matrix = {&current_value};
//                            rlt::evaluate(device, actor_critic.critic_target_1, input_matrix, current_value_matrix);
//                            DTYPE desired_value;
//                            rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, 1>> desired_value_matrix = {&desired_value};
//                            rlt::evaluate(device, post_critic_1_target, input_matrix, desired_value_matrix);
//                            diff += (current_value - desired_value) * (current_value - desired_value) / ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
//                        }
////                        std::cout << "value mse: " << diff << std::endl;
//                    }
                    rlt::free(device, post_critic_1_target);
                }
            }

        }
        if(step_i % 100 == 0){
            if(!verbose){
                std::cout << "step_i: " << step_i << std::endl;
            }
            auto result = rlt::evaluate(device, env, ui, actor_critic.actor, rlt::rl::utils::evaluation::Specification<100, 200>(), actor_eval_buffers, rng, true);
#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_TYPE, ActorCriticType::CRITIC_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("second_stage"), step_i);
#endif
#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
            if(mean_return > -400){
                while(true){
                    ENVIRONMENT::State initial_state;
                    rlt::sample_initial_state(env, initial_state, rng);
                    rlt::evaluate_visual<ENVIRONMENT, UI, decltype(actor_critic.actor), 100, 3>(env, ui, actor_critic.actor, initial_state);
                }
            }
#endif
        }
    }
    mean_ratio_critic /= num_steps;
    mean_ratio_critic_grad /= num_steps;
    mean_ratio_critic_adam /= num_steps;
    mean_ratio_actor /= num_steps;
    mean_ratio_actor_grad /= num_steps;
    mean_ratio_actor_adam /= num_steps;
    mean_ratio_critic_target /= num_steps;
    std::cout << "mean_ratio_critic: " << mean_ratio_critic << std::endl;
    std::cout << "mean_ratio_critic_grad: " << mean_ratio_critic_grad << std::endl;
    std::cout << "mean_ratio_critic_adam: " << mean_ratio_critic_adam << std::endl;
    std::cout << "mean_ratio_actor: " << mean_ratio_actor << std::endl;
    std::cout << "mean_ratio_actor_grad: " << mean_ratio_actor_grad << std::endl;
    std::cout << "mean_ratio_actor_adam: " << mean_ratio_actor_adam << std::endl;
    std::cout << "mean_ratio_critic_target: " << mean_ratio_critic_target << std::endl;
    ASSERT_GT(mean_ratio_critic, 1e12);
    ASSERT_GT(mean_ratio_critic_grad, 1e13);
    ASSERT_GT(mean_ratio_critic_adam, 1e12);
    ASSERT_GT(mean_ratio_actor, 1e12);
    ASSERT_GT(mean_ratio_actor_grad, 1e12);
    ASSERT_GT(mean_ratio_actor_adam, 1e12);
    ASSERT_GT(mean_ratio_critic_target, 1e11);

    rlt::free(device, critic_batch);
    rlt::free(device, critic_training_buffers);
    rlt::free(device, actor_batch);
    rlt::free(device, actor_training_buffers);
    rlt::free(device, off_policy_runner);
}
