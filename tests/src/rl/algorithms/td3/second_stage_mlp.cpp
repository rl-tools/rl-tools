#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/rl/environments/operations_cpu.h>
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>

#include <layer_in_c/nn_models/persist.h>
#include <layer_in_c/rl/utils/evaluation.h>
#include <layer_in_c/utils/generic/memcpy.h>

#include "../../../utils/utils.h"
#include "../../../utils/nn_comparison_mlp.h"

#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
#include <layer_in_c/rl/environments/pendulum/ui.h>
#include <layer_in_c/rl/utils/evaluation_visual.h>
#endif

#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif

namespace lic = layer_in_c;
std::string get_data_file_path(){
    std::string DATA_FILE_PATH = "../multirotor-torch/model_second_stage.hdf5";
    const char* data_file_path = std::getenv("LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_DATA_FILE");
    if (data_file_path != NULL){
        DATA_FILE_PATH = std::string(data_file_path);
//            std::runtime_error("Environment variable LAYER_IN_C_TEST_DATA_DIR not set. Skipping test.");
    }
    return DATA_FILE_PATH;
}
#define DTYPE double
using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::environments::pendulum::Specification<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<DEVICE, PENDULUM_SPEC> ENVIRONMENT;
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
typedef lic::rl::environments::pendulum::UI<DTYPE> UI;
#endif
ENVIRONMENT env;

using AC_DEVICE = lic::devices::DefaultCPU;

struct TD3ParametersCopyTraining: public lic::rl::algorithms::td3::DefaultParameters<DTYPE, AC_DEVICE::index_t>{
    constexpr static typename AC_DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename AC_DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3ParametersCopyTraining::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3ParametersCopyTraining::CRITIC_BATCH_SIZE>;

using NN_DEVICE = lic::devices::DefaultCPU;
using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;


using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3ParametersCopyTraining>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_SECOND_STAGE, TEST_LOADING_TRAINED_ACTOR) {
    constexpr bool verbose = false;
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device(logger);
    NN_DEVICE nn_device(logger);
    ActorCriticType actor_critic;
    lic::malloc(device, actor_critic);

    std::mt19937 rng(0);

    auto data_file = HighFive::File(get_data_file_path(), HighFive::File::ReadOnly);
    int step = data_file.getGroup("full_training").getGroup("steps").getNumberObjects()-1;
    assert(step >= 0);
    auto step_group = data_file.getGroup("full_training").getGroup("steps").getGroup(std::to_string(step));
    lic::load(device, actor_critic.actor, step_group.getGroup("actor"));
    DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), 200, true>(device, env, actor_critic.actor, 100, rng);
    std::cout << "mean return: " << mean_return << std::endl;
}

using ReplayBufferSpecCopyTraining = lic::rl::components::replay_buffer::Specification<DTYPE, AC_DEVICE::index_t, 3, 1, 1000>;
using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::components::ReplayBuffer<ReplayBufferSpecCopyTraining> ReplayBufferTypeCopyTraining;
constexpr int BATCH_DIM = ENVIRONMENT::OBSERVATION_DIM * 2 + ENVIRONMENT::ACTION_DIM + 2;
template <typename T, typename REPLAY_BUFFER_TYPE>
void load(ReplayBufferTypeCopyTraining& rb, std::vector<std::vector<T>> batch){
    for(int i = 0; i < batch.size(); i++){
        lic::utils::memcpy( rb.     observations[i], &batch[i][0], ENVIRONMENT::OBSERVATION_DIM);
        lic::utils::memcpy( rb.          actions[i], &batch[i][ENVIRONMENT::OBSERVATION_DIM], ENVIRONMENT::ACTION_DIM);
        lic::utils::memcpy( rb.next_observations[i], &batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM], ENVIRONMENT::OBSERVATION_DIM);
        rb.   rewards[i] = batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM    ];
        rb.terminated[i] = batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM + 1] == 1;
        rb. truncated[i] = batch[i][ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM + ENVIRONMENT::OBSERVATION_DIM + 2] == 1;
    }
    rb.position = batch.size();
}
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_SECOND_STAGE, FP_ACC) {
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
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_MLP_SECOND_STAGE, TEST_COPY_TRAINING) {
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
    UI ui;
#endif
    constexpr bool verbose = true;
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device(logger);
    NN_DEVICE nn_device(logger);
    ActorCriticType actor_critic;
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

    ReplayBufferTypeCopyTraining replay_buffer;

    lic::reset_optimizer_state(device, actor_critic.actor);
    lic::reset_optimizer_state(device, actor_critic.critic_1);
    lic::reset_optimizer_state(device, actor_critic.critic_2);
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
    lic::malloc(device, pre_critic_1);
    lic::copy(device, pre_critic_1, actor_critic.critic_1);
    decltype(actor_critic.actor) pre_actor;
    lic::malloc(device, pre_actor);
    lic::copy(device, pre_actor, actor_critic.actor);
    decltype(actor_critic.critic_target_1) pre_critic_1_target;
    lic::malloc(device, pre_critic_1_target);
    lic::copy(device, pre_critic_1_target, actor_critic.critic_target_1);

    for(int step_i = 0; step_i < num_steps; step_i++){
        if(verbose){
            std::cout << "step_i: " << step_i << std::endl;
        }
        auto step_group = steps_group.getGroup(std::to_string(step_i));
        if(step_group.exist("critics_batch")){
            std::vector<std::vector<DTYPE>> batch;
            step_group.getDataSet("critics_batch").read(batch);
            assert(batch.size() == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

            DTYPE target_next_action_noise[ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE][ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
            lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> target_next_action_noise_matrix = {(DTYPE*)target_next_action_noise};
            step_group.getDataSet("target_next_action_noise").read(target_next_action_noise);

            load<DTYPE, ReplayBufferTypeCopyTraining>(replay_buffer, batch);
            if (step_i == 0 && step_group.exist("pre_critic1")){
                decltype(actor_critic.critic_1) pre_critic_1_step;
                lic::malloc(device, pre_critic_1_step);
                lic::load(device, pre_critic_1_step, step_group.getGroup("pre_critic1"));
                lic::reset_forward_state(device, pre_critic_1_step);
                lic::reset_forward_state(device, actor_critic.critic_1);
                DTYPE pre_current_diff = abs_diff(device, pre_critic_1_step, actor_critic.critic_1);
                ASSERT_EQ(pre_current_diff, 0);
                lic::free(device, pre_critic_1_step);
            }

            decltype(actor_critic.critic_1) post_critic_1;// = actor_critic.critic_1;
            lic::malloc(device, post_critic_1);
            lic::load(device, post_critic_1, step_group.getGroup("critic1"));

            lic::rl::components::replay_buffer::Batch<ReplayBufferSpecCopyTraining, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> batch_struct;
            lic::malloc(device, batch_struct);

            lic::gather_batch<DEVICE, ReplayBufferSpecCopyTraining, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, decltype(rng), true>(device, replay_buffer, batch_struct, rng);

            lic::train_critic(device, actor_critic, actor_critic.critic_1, batch_struct, target_next_action_noise_matrix);
            lic::free(device, batch_struct);


//            DTYPE critic_1_loss = lic::train_critic<
//                    AC_DEVICE,
//                    decltype(actor_critic)::SPEC,
//                    decltype(actor_critic.critic_1),
//                    decltype(replay_buffer)::CAPACITY,
//                    decltype(rng),
//                    true
//            >(device, actor_critic, actor_critic.critic_1, replay_buffer, target_next_action_noise, rng);



            lic::reset_forward_state(device, pre_critic_1);
            lic::reset_forward_state(device, post_critic_1);
            lic::reset_forward_state(device, actor_critic.critic_1);
            DTYPE pre_post_diff_per_weight = abs_diff(device, pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_per_weight = abs_diff(device, post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

            DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(device, pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_grad_per_weight = abs_diff_grad(device, post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

            DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(device, pre_critic_1, post_critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_adam_per_weight = abs_diff_adam(device, post_critic_1, actor_critic.critic_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
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
                lic::rl::components::replay_buffer::Batch<ReplayBufferSpecCopyTraining, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> batch_struct;
                lic::malloc(device, batch_struct);

                lic::gather_batch<DEVICE, ReplayBufferSpecCopyTraining, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, decltype(rng), true>(device, replay_buffer, batch_struct, rng);

                lic::train_critic(device, actor_critic, actor_critic.critic_2, batch_struct, target_next_action_noise_matrix);
                lic::free(device, batch_struct);
            }
//            DTYPE critic_2_loss = lic::train_critic<
//                AC_DEVICE,
//                decltype(actor_critic)::SPEC,
//                decltype(actor_critic.critic_2),
//                decltype(replay_buffer)::CAPACITY,
//                decltype(rng),
//                true
//            >(device, actor_critic, actor_critic.critic_2, replay_buffer, target_next_action_noise, rng);
            lic::copy(device, pre_critic_1, actor_critic.critic_1);

            if(true){//(step_i % 100 == 0){
                DTYPE diff = 0;
                for(int batch_sample_i = 0; batch_sample_i < ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
                    DTYPE input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
                    lic::utils::memcpy(input, replay_buffer.observations[batch_sample_i], ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM);
                    lic::utils::memcpy(&input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM], replay_buffer.actions[batch_sample_i], ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM);
                    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> input_matrix = {input};
                    DTYPE current_value;
                    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, 1>> current_value_matrix = {&current_value};
                    lic::evaluate(device, actor_critic.critic_1, input_matrix, current_value_matrix);
                    DTYPE desired_value;
                    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, 1>> desired_value_matrix = {&desired_value};
                    lic::evaluate(device, post_critic_1, input_matrix, desired_value_matrix);
                    diff += (current_value - desired_value) * (current_value - desired_value) / ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
                }
//                std::cout << "value mse: " << diff << std::endl;
            }
            lic::free(device, post_critic_1);
        }

        if(step_group.exist("actor_batch")){
            std::vector<std::vector<DTYPE>> batch;
            step_group.getDataSet("actor_batch").read(batch);
            assert(batch.size() == ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
            load<DTYPE, ReplayBufferTypeCopyTraining>(replay_buffer, batch);

            decltype(actor_critic.actor) post_actor;
            lic::malloc(device, post_actor);
            lic::load(device, post_actor, step_group.getGroup("actor"));

            decltype(actor_critic.actor) pre_actor_loaded;
            lic::malloc(device, pre_actor_loaded);
            lic::load(device, pre_actor_loaded, step_group.getGroup("pre_actor"));
            lic::reset_forward_state(device, pre_actor_loaded);
            lic::reset_forward_state(device, actor_critic.actor);
            DTYPE pre_current_diff = abs_diff(device, pre_actor_loaded, actor_critic.actor);
            if(step_i == 0){
                ASSERT_EQ(pre_current_diff, 0);
            }


            {
                lic::rl::components::replay_buffer::Batch<ReplayBufferSpecCopyTraining, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> batch_struct;
                lic::malloc(device, batch_struct);

                lic::gather_batch<DEVICE, ReplayBufferSpecCopyTraining, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE, decltype(rng), true>(device, replay_buffer, batch_struct, rng);

                lic::train_actor(device, actor_critic, batch_struct);
                lic::free(device, batch_struct);
            }
//            DTYPE actor_loss = lic::train_actor<AC_DEVICE, ActorCriticType::SPEC, decltype(replay_buffer)::CAPACITY, typeof(rng), true>(device, actor_critic, replay_buffer, rng);

            if(true){//(step_i % 100 == 1){
                DTYPE diff = 0;
                for(int batch_sample_i = 0; batch_sample_i < ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE; batch_sample_i++){
                    DTYPE current_action[ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
                    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> current_action_matrix = {current_action};
                    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM>> observation_matrix = {replay_buffer.observations[batch_sample_i]};
                    lic::evaluate(device, actor_critic.actor, observation_matrix, current_action_matrix);
                    DTYPE desired_action[ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
                    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> desired_action_matrix = {desired_action};
                    lic::evaluate(device, post_actor, observation_matrix, desired_action_matrix);
                    diff += lic::nn::loss_functions::mse(device, current_action_matrix, desired_action_matrix, DTYPE(1)/ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
                }
//                std::cout << "action mse: " << diff << std::endl;
            }

            lic::reset_forward_state(device, pre_actor);
            lic::reset_forward_state(device, post_actor);
            lic::reset_forward_state(device, actor_critic.actor);

            DTYPE pre_post_diff_per_weight = abs_diff(device, pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_per_weight = abs_diff(device, post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio = pre_post_diff_per_weight/diff_target_per_weight;

            DTYPE pre_post_diff_grad_per_weight = abs_diff_grad(device, pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_grad_per_weight = abs_diff_grad(device, post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_ratio_grad = pre_post_diff_grad_per_weight/diff_target_grad_per_weight;

            DTYPE pre_post_diff_adam_per_weight = abs_diff_adam(device, pre_actor, post_actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
            DTYPE diff_target_adam_per_weight = abs_diff_adam(device, post_actor, actor_critic.actor)/ActorCriticType::SPEC::ACTOR_NETWORK_TYPE::NUM_WEIGHTS;
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

            lic::copy(device, pre_actor, actor_critic.actor);
            lic::free(device, post_actor);
            lic::free(device, pre_actor_loaded);
        }
        if(step_group.exist("critic1_target")){
            if(verbose){
                std:: cout << "    target update" << std::endl;
            }
            if (step_i == 0){
                decltype(actor_critic.critic_target_1) pre_critic_1_target_step;
                lic::malloc(device, pre_critic_1_target_step);
                lic::load(device, pre_critic_1_target_step, step_group.getGroup("pre_critic1_target"));
                DTYPE pre_current_diff = abs_diff(device, pre_critic_1_target_step, actor_critic.critic_target_1);
                ASSERT_EQ(pre_current_diff, 0);
                lic::free(device, pre_critic_1_target_step);
            }
            else{
                if (step_i >= ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE){

                    decltype(actor_critic.critic_target_1) post_critic_1_target;
                    lic::malloc(device, post_critic_1_target);
                    lic::load(device, post_critic_1_target, step_group.getGroup("critic1_target"));

                    lic::update_targets(device, actor_critic);

                    DTYPE pre_post_diff_per_weight = abs_diff(device, pre_critic_1_target, post_critic_1_target)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
                    DTYPE diff_target_per_weight = abs_diff(device, post_critic_1_target, actor_critic.critic_target_1)/ActorCriticType::SPEC::CRITIC_NETWORK_TYPE::NUM_WEIGHTS;
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

                    lic::copy(device, pre_critic_1_target, actor_critic.critic_target_1);

                    if(true){//(step_i % 100 == 0){
                        DTYPE diff = 0;
                        for(int batch_sample_i = 0; batch_sample_i < ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
                            DTYPE input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM];
                            lic::utils::memcpy(input, replay_buffer.observations[batch_sample_i], ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM);
                            lic::utils::memcpy(&input[ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM], replay_buffer.actions[batch_sample_i], ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM);
                            lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, ActorCriticType::SPEC::ENVIRONMENT::OBSERVATION_DIM + ActorCriticType::SPEC::ENVIRONMENT::ACTION_DIM>> input_matrix = {input};
                            DTYPE current_value;
                            lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, 1>> current_value_matrix = {&current_value};
                            lic::evaluate(device, actor_critic.critic_target_1, input_matrix, current_value_matrix);
                            DTYPE desired_value;
                            lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, 1>> desired_value_matrix = {&desired_value};
                            lic::evaluate(device, post_critic_1_target, input_matrix, desired_value_matrix);
                            diff += (current_value - desired_value) * (current_value - desired_value) / ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
                        }
//                        std::cout << "value mse: " << diff << std::endl;
                    }
                    lic::free(device, post_critic_1_target);
                }
            }

        }
        if(step_i % 100 == 0){
            if(!verbose){
                std::cout << "step_i: " << step_i << std::endl;
            }
            DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), 200, true>(device, env, actor_critic.actor, 100, rng);
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, ActorCriticType::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("second_stage"), step_i);
#endif
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY
            if(mean_return > -400){
                while(true){
                    ENVIRONMENT::State initial_state;
                    lic::sample_initial_state(env, initial_state, rng);
                    lic::evaluate_visual<ENVIRONMENT, UI, decltype(actor_critic.actor), 100, 3>(env, ui, actor_critic.actor, initial_state);
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
}
