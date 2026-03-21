#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_CUDA_H

#include "ppo.h"
#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::ppo::cuda{
        template <typename DEVICE, typename DATASET_SPEC, typename PPO_PARAMETERS>
        __global__
        void estimate_generalized_advantages_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, PPO_PARAMETERS){
            using OPR_SPEC = typename DATASET_SPEC::SPEC;
            using T = typename OPR_SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            constexpr TI STEPS_PER_ENV = DATASET_SPEC::STEPS_PER_ENV;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < OPR_SPEC::N_ENVIRONMENTS){
                T previous_value = get(dataset.all_values, STEPS_PER_ENV * OPR_SPEC::N_ENVIRONMENTS + env_i, 0);
                T previous_advantage = 0;
                for(TI step_forward_i = 0; step_forward_i < STEPS_PER_ENV; step_forward_i++){
                    TI step_backward_i = (STEPS_PER_ENV - 1 - step_forward_i);
                    TI pos = step_backward_i * OPR_SPEC::N_ENVIRONMENTS + env_i;
                    bool terminated_flag = get(dataset.terminated, pos, 0);
                    bool truncated_flag = get(dataset.truncated, pos, 0);
                    T current_step_value = get(dataset.values, pos, 0);
                    bool terminated_actual = terminated_flag && !PPO_PARAMETERS::IGNORE_TERMINATION;
                    T next_step_value = terminated_actual ? 0 : previous_value;
                    T td_error = get(dataset.rewards, pos, 0) + PPO_PARAMETERS::GAMMA * next_step_value - current_step_value;
                    if(truncated_flag){
                        if(!terminated_flag){
                            td_error = 0;
                        }
                        previous_advantage = 0;
                    }
                    T advantage = PPO_PARAMETERS::LAMBDA * PPO_PARAMETERS::GAMMA * previous_advantage + td_error;
                    set(dataset.advantages, pos, 0, advantage);
                    set(dataset.target_values, pos, 0, advantage + current_step_value);
                    previous_advantage = advantage;
                    previous_value = current_step_value;
                }
            }
        }
        template <typename DEVICE, typename SPEC, typename T>
        __global__
        void reduce_mean_std_kernel(DEVICE device, Matrix<SPEC> data, T* out_mean, T* out_sq_mean){
            using TI = typename DEVICE::index_t;
            constexpr TI N = SPEC::ROWS;
            extern __shared__ char shared_mem[];
            T* shared_sum = reinterpret_cast<T*>(shared_mem);
            T* shared_sq = shared_sum + blockDim.x;
            TI tid = threadIdx.x;
            T local_sum = 0, local_sq = 0;
            for(TI i = tid; i < N; i += blockDim.x){
                T val = get(data, i, 0);
                local_sum += val;
                local_sq += val * val;
            }
            shared_sum[tid] = local_sum;
            shared_sq[tid] = local_sq;
            __syncthreads();
            for(TI s = blockDim.x / 2; s > 0; s >>= 1){
                if(tid < s){
                    shared_sum[tid] += shared_sum[tid + s];
                    shared_sq[tid] += shared_sq[tid + s];
                }
                __syncthreads();
            }
            if(tid == 0){
                *out_mean = shared_sum[0] / N;
                *out_sq_mean = shared_sq[0] / N;
            }
        }
        template <typename DEVICE, typename PPO_SPEC, typename DATASET_SPEC, typename BUFFERS_SPEC, typename BATCH_ADVANTAGES_SPEC, typename BATCH_ACTIONS_SPEC, typename BATCH_ACTION_LOG_PROBS_SPEC, typename LOG_STD_SPEC>
        __global__
        void ppo_per_sample_kernel(DEVICE device, rl::algorithms::ppo::Buffers<BUFFERS_SPEC> ppo_buffers, Matrix<BATCH_ADVANTAGES_SPEC> batch_advantages, Matrix<BATCH_ACTIONS_SPEC> batch_actions, Matrix<BATCH_ACTION_LOG_PROBS_SPEC> batch_action_log_probs, Matrix<LOG_STD_SPEC> log_std_params, Matrix<LOG_STD_SPEC> log_std_gradient, typename PPO_SPEC::TYPE_POLICY::DEFAULT advantage_mean, typename PPO_SPEC::TYPE_POLICY::DEFAULT advantage_std){
            using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
            constexpr TI ACTION_DIM = PPO_SPEC::ENVIRONMENT::ACTION_DIM;
            constexpr TI N_AGENTS = PPO_SPEC::ENVIRONMENT::N_AGENTS;
            constexpr TI PER_AGENT_ACTION_DIM = ACTION_DIM / N_AGENTS;
            TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_step_i < BATCH_SIZE){
                T action_log_prob = 0;
                for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                    T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                    T rollout_action = get(batch_actions, batch_step_i, action_i);
                    T current_action_log_std = get(log_std_params, 0, action_i % PER_AGENT_ACTION_DIM);
                    T current_action_std = math::exp(device.math, current_action_log_std);
                    action_log_prob += random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                    set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                    if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                        T d_entropy_loss_d_current_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                        atomicAdd(&get(log_std_gradient, 0, action_i % PER_AGENT_ACTION_DIM), d_entropy_loss_d_current_action_log_std);
                        T d_action_log_prob_d_current_action_log_std = random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action);
                        set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, d_action_log_prob_d_current_action_log_std);
                    }
                }
                T rollout_action_log_prob = get(batch_action_log_probs, batch_step_i, 0);
                T advantage = get(batch_advantages, batch_step_i, 0);
                if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                    advantage = (advantage - advantage_mean) / (advantage_std + PPO_SPEC::PARAMETERS::ADVANTAGE_EPSILON);
                }
                T log_ratio = action_log_prob - rollout_action_log_prob;
                T ratio = math::exp(device.math, log_ratio);
                T clipped_ratio = math::clamp(device.math, ratio, (T)1 - PPO_SPEC::PARAMETERS::EPSILON_CLIP, (T)1 + PPO_SPEC::PARAMETERS::EPSILON_CLIP);
                bool clipped = ratio != clipped_ratio;
                T normal_advantage = ratio * advantage;
                T clipped_advantage = clipped_ratio * advantage;
                bool ratio_min_switch = normal_advantage - clipped_advantage <= (T)0;
                T d_loss_d_pessimistic_surrogate = -(T)1/BATCH_SIZE;
                T d_pessimistic_surrogate_d_normal_advantage = ratio_min_switch ? 1 : 0;
                T d_pessimistic_surrogate_d_clipped_advantage = ratio_min_switch ? 0 : 1;
                T d_normal_advantage_d_ratio = advantage;
                T d_clipped_advantage_d_clipped_ratio = advantage;
                T d_clipped_ratio_d_ratio = clipped ? 0 : 1;
                T d_pessimistic_surrogate_d_ratio = d_pessimistic_surrogate_d_normal_advantage * d_normal_advantage_d_ratio + d_pessimistic_surrogate_d_clipped_advantage * d_clipped_advantage_d_clipped_ratio * d_clipped_ratio_d_ratio;
                T d_loss_d_ratio = d_loss_d_pessimistic_surrogate * d_pessimistic_surrogate_d_ratio;
                T d_ratio_d_action_log_prob = ratio;
                T d_loss_d_action_log_prob = d_loss_d_ratio * d_ratio_d_action_log_prob;
                for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                    multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                    if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                        T current_d = get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                        atomicAdd(&get(log_std_gradient, 0, action_i % PER_AGENT_ACTION_DIM), d_loss_d_action_log_prob * current_d);
                    }
                }
            }
        }
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename PPO_PARAMETERS>
    void estimate_generalized_advantages(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, PPO_PARAMETERS ppo_parameters_tag){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using OPR_SPEC = typename DATASET_SPEC::SPEC;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OPR_SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::algorithms::ppo::cuda::estimate_generalized_advantages_kernel<<<grid, block, 0, device.stream>>>(tag_device, dataset, ppo_parameters_tag);
        check_status(device);
    }
    template <typename DEV_SPEC, typename PPO_SPEC, typename DATASET_SPEC, typename ACTOR_OPTIMIZER, typename CRITIC_OPTIMIZER, typename BUFFERS_SPEC, typename ACTOR_BUFFER, typename CRITIC_BUFFER, typename RNG>
    void train(devices::CUDA<DEV_SPEC>& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, ACTOR_OPTIMIZER& actor_optimizer, CRITIC_OPTIMIZER& critic_optimizer, rl::algorithms::ppo::Buffers<BUFFERS_SPEC>& ppo_buffers, ACTOR_BUFFER& actor_buffers, CRITIC_BUFFER& critic_buffers, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename PPO_SPEC::TI;
        using ENVIRONMENT = typename PPO_SPEC::ENVIRONMENT;
        using DATASET = rl::components::on_policy_runner::Dataset<DATASET_SPEC>;
        constexpr TI N_EPOCHS = PPO_SPEC::PARAMETERS::N_EPOCHS;
        constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
        constexpr TI N_BATCHES = DATASET::STEPS_TOTAL/BATCH_SIZE;
        constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        using OBS_SHAPE = typename DATASET::OBS_SHAPE;
        using OBS_PRIV_SHAPE = typename DATASET::OBS_PRIV_SHAPE;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        T* d_adv_mean;
        T* d_adv_sq_mean;
        cudaMalloc(&d_adv_mean, sizeof(T));
        cudaMalloc(&d_adv_sq_mean, sizeof(T));
        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
                zero_gradient(device, ppo.critic);
                zero_gradient(device, ppo.actor);
                auto batch_offset = batch_i * BATCH_SIZE;
                auto batch_observations            = view_range(device, dataset.all_observations              , batch_offset, tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_observations_privileged = view_range(device, dataset.all_observations_privileged   , batch_offset, tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_actions                 = view(device, dataset.actions                    , matrix::ViewSpec<BATCH_SIZE, ACTION_DIM>(), batch_offset, 0);
                auto batch_action_log_probs        = view(device, dataset.action_log_probs           , matrix::ViewSpec<BATCH_SIZE, 1         >(), batch_offset, 0);
                auto batch_advantages              = view(device, dataset.advantages                 , matrix::ViewSpec<BATCH_SIZE, 1         >(), batch_offset, 0);
                auto batch_target_values           = view(device, dataset.target_values              , matrix::ViewSpec<BATCH_SIZE, 1         >(), batch_offset, 0);
                auto batch_reset                   = view(device, dataset.reset                      , matrix::ViewSpec<BATCH_SIZE, 1         >(), batch_offset, 0);
                T advantage_mean = 0, advantage_std = 0;
                if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                    constexpr TI REDUCE_BLOCKSIZE = 256;
                    rl::algorithms::ppo::cuda::reduce_mean_std_kernel<<<1, REDUCE_BLOCKSIZE, 2 * REDUCE_BLOCKSIZE * sizeof(T), device.stream>>>(tag_device, batch_advantages, d_adv_mean, d_adv_sq_mean);
                    check_status(device);
                    T h_mean, h_sq_mean;
                    cudaMemcpyAsync(&h_mean, d_adv_mean, sizeof(T), cudaMemcpyDeviceToHost, device.stream);
                    cudaMemcpyAsync(&h_sq_mean, d_adv_sq_mean, sizeof(T), cudaMemcpyDeviceToHost, device.stream);
                    cudaStreamSynchronize(device.stream);
                    advantage_mean = h_mean;
                    advantage_std = math::sqrt(device.math, math::max(device.math, (T)0, h_sq_mean - h_mean * h_mean));
                }
                static constexpr TI STEPS = PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? DATASET_SPEC::STEPS_PER_ENV : 1;
                static constexpr TI FORWARD_BATCH_SIZE = PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? DATASET_SPEC::SPEC::N_ENVIRONMENTS : BATCH_SIZE;
                using ACTOR_INPUT_SHAPE = tensor::Prepend<tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
                auto batch_observations_reshaped = reshape_row_major(device, batch_observations, ACTOR_INPUT_SHAPE{});
                auto current_batch_actions_tensor = to_tensor(device, ppo_buffers.current_batch_actions);
                auto current_batch_actions_tensor_reshaped = reshape_row_major(device, current_batch_actions_tensor, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, ACTION_DIM>{});
                auto batch_reset_tensor_flat = to_tensor(device, batch_reset);
                auto batch_reset_tensor = reshape_row_major(device, batch_reset_tensor_flat, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, 1>{});
                Mode<nn::layers::gru::ResetMode<mode::Rollout<>, nn::layers::gru::ResetModeSpecification<TI, decltype(batch_reset_tensor)>>> mode;
                mode.reset_container = batch_reset_tensor;
                forward(device, ppo.actor, batch_observations_reshaped, current_batch_actions_tensor_reshaped, actor_buffers, rng, mode);
                auto& last_layer = get_last_layer(ppo.actor);
                auto log_std = matrix_view(device, last_layer.log_std.parameters);
                auto log_std_grad = matrix_view(device, last_layer.log_std.gradient);
                {
                    constexpr TI KERNEL_BLOCKSIZE = 32;
                    constexpr TI KERNEL_N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, KERNEL_BLOCKSIZE);
                    rl::algorithms::ppo::cuda::ppo_per_sample_kernel<decltype(tag_device), PPO_SPEC, DATASET_SPEC, BUFFERS_SPEC><<<KERNEL_N_BLOCKS, KERNEL_BLOCKSIZE, 0, device.stream>>>(tag_device, ppo_buffers, batch_advantages, batch_actions, batch_action_log_probs, log_std, log_std_grad, advantage_mean, advantage_std);
                    check_status(device);
                }
                auto d_action_d_log_prob_action_tensor = to_tensor(device, ppo_buffers.d_action_log_prob_d_action);
                auto d_action_d_log_prob_action_tensor_reshaped = reshape_row_major(device, d_action_d_log_prob_action_tensor, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, ACTION_DIM>{});
                backward(device, ppo.actor, batch_observations_reshaped, d_action_d_log_prob_action_tensor_reshaped, actor_buffers, mode);
                using CRITIC_INPUT_SHAPE = tensor::Prepend<tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
                auto batch_observations_privileged_reshaped = reshape_row_major(device, batch_observations_privileged, CRITIC_INPUT_SHAPE{});
                {
                    forward(device, ppo.critic, batch_observations_privileged_reshaped, critic_buffers, rng, mode);
                    auto output_tensor = output(device, ppo.critic);
                    auto output_matrix_view = matrix_view(device, output_tensor);
                    nn::loss_functions::mse::gradient(device, output_matrix_view, batch_target_values, ppo_buffers.d_critic_output, 0.5);
                    auto d_critic_output_tensor = to_tensor(device, ppo_buffers.d_critic_output);
                    auto d_critic_output_tensor_reshaped = reshape_row_major(device, d_critic_output_tensor, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, 1>{});
                    backward(device, ppo.critic, batch_observations_privileged_reshaped, d_critic_output_tensor_reshaped, critic_buffers, mode);
                }
                step(device, actor_optimizer, ppo.actor);
                step(device, critic_optimizer, ppo.critic);
            }
        }
        cudaFree(d_adv_mean);
        cudaFree(d_adv_sq_mean);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
