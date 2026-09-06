#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_CUDA_H

#include "ppo.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::ppo::cuda{
        template <typename DEVICE, typename DATASET_SPEC, typename BOOTSTRAP_SPEC, typename PPO_PARAMETERS>
        __global__
        void estimate_generalized_advantages_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, const Matrix<BOOTSTRAP_SPEC> bootstrap_values, PPO_PARAMETERS){
            using OPR_SPEC = typename DATASET_SPEC::SPEC;
            using T = typename OPR_SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            constexpr TI STEPS_PER_ENV = DATASET_SPEC::STEPS_PER_ENV;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < OPR_SPEC::N_ENVIRONMENTS){
                T previous_advantage = 0;
                for(TI step_forward_i = 0; step_forward_i < STEPS_PER_ENV; step_forward_i++){
                    TI step_backward_i = (STEPS_PER_ENV - 1 - step_forward_i);
                    TI pos = step_backward_i * OPR_SPEC::N_ENVIRONMENTS + env_i;
                    bool terminated_flag = get(dataset.terminated, pos, 0);
                    bool truncated_flag = get(dataset.truncated, pos, 0);
                    T current_step_value = get(dataset.values, pos, 0);
                    bool stop_bootstrap = terminated_flag ? !PPO_PARAMETERS::IGNORE_TERMINATION : (truncated_flag && !PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS);
                    T next_step_value = stop_bootstrap ? 0 : get(bootstrap_values, pos, 0);
                    T td_error = get(dataset.rewards, pos, 0) + PPO_PARAMETERS::GAMMA * next_step_value - current_step_value;
                    if(truncated_flag){
                        previous_advantage = 0;
                    }
                    T advantage = PPO_PARAMETERS::LAMBDA * PPO_PARAMETERS::GAMMA * previous_advantage + td_error;
                    set(dataset.advantages, pos, 0, advantage);
                    set(dataset.target_values, pos, 0, advantage + current_step_value);
                    previous_advantage = advantage;
                }
            }
        }
        template <typename DEVICE, typename PPO_SPEC, typename DATASET_SPEC, typename BUFFERS_SPEC, typename BATCH_ADVANTAGES_SPEC, typename BATCH_ACTIONS_SPEC, typename BATCH_ACTION_LOG_PROBS_SPEC, typename LOG_STD_SPEC>
        __global__
        void ppo_per_sample_kernel(DEVICE device, rl::algorithms::ppo::Buffers<BUFFERS_SPEC> ppo_buffers, Matrix<BATCH_ADVANTAGES_SPEC> batch_advantages, Matrix<BATCH_ACTIONS_SPEC> batch_actions, Matrix<BATCH_ACTION_LOG_PROBS_SPEC> batch_action_log_probs, Matrix<LOG_STD_SPEC> log_std_params){
            using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
            constexpr TI ACTION_DIM = PPO_SPEC::ENVIRONMENT::ACTION_DIM;
            constexpr TI N_AGENTS = PPO_SPEC::ENVIRONMENT::N_AGENTS;
            constexpr TI PER_AGENT_ACTION_DIM = ACTION_DIM / N_AGENTS;
            constexpr TI KERNEL_BLOCKSIZE = BATCH_SIZE < 1024 ? (BATCH_SIZE <= 32 ? 32 : (BATCH_SIZE <= 64 ? 64 : (BATCH_SIZE <= 128 ? 128 : (BATCH_SIZE <= 256 ? 256 : (BATCH_SIZE <= 512 ? 512 : 1024))))) : 1024;
            static_assert(2 * KERNEL_BLOCKSIZE * sizeof(T) <= 48 * 1024, "ppo_per_sample_kernel shared memory usage exceeds 48KB limit");
            // Fused advantage normalization via shared memory reduction
            extern __shared__ char shared_mem_raw[];
            T* shared_sum = reinterpret_cast<T*>(shared_mem_raw);
            T* shared_sq = shared_sum + blockDim.x;
            T advantage_mean = 0;
            T advantage_std = 0;
            if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                T local_sum = 0, local_sq = 0;
                for(TI i = threadIdx.x; i < BATCH_SIZE; i += blockDim.x){
                    T val = get(batch_advantages, i, 0);
                    local_sum += val;
                    local_sq += val * val;
                }
                shared_sum[threadIdx.x] = local_sum;
                shared_sq[threadIdx.x] = local_sq;
                __syncthreads();
                for(TI s = blockDim.x / 2; s > 0; s >>= 1){
                    if(threadIdx.x < s){
                        shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
                        shared_sq[threadIdx.x] += shared_sq[threadIdx.x + s];
                    }
                    __syncthreads();
                }
                advantage_mean = shared_sum[0] / BATCH_SIZE;
                T sq_mean = shared_sq[0] / BATCH_SIZE;
                T variance = sq_mean - advantage_mean * advantage_mean;
                advantage_std = variance > 0 ? math::sqrt(device.math, variance) : 0;
            }
            for(TI batch_step_i = threadIdx.x; batch_step_i < BATCH_SIZE; batch_step_i += blockDim.x){
                T action_log_prob = 0;
                for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                    T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                    T rollout_action = get(batch_actions, batch_step_i, action_i);
                    T current_action_log_std = get(log_std_params, 0, action_i % PER_AGENT_ACTION_DIM);
                    action_log_prob += random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                    set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                    if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
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
                        multiply(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, d_loss_d_action_log_prob);
                    }
                }
            }
        }
        // Deterministic reduction: sum columns of per-sample log_std gradients into log_std_gradient
        template <typename DEVICE, typename LOG_STD_SPEC, typename D_LOG_STD_SPEC>
        __global__
        void reduce_log_std_gradient_kernel(DEVICE device, Matrix<D_LOG_STD_SPEC> d_action_log_prob_d_action_log_std, Matrix<LOG_STD_SPEC> log_std_gradient, typename LOG_STD_SPEC::TI batch_size, typename LOG_STD_SPEC::TI action_dim, typename LOG_STD_SPEC::TI per_agent_action_dim, typename LOG_STD_SPEC::T entropy_grad){
            using T = typename LOG_STD_SPEC::T;
            using TI = typename DEVICE::index_t;
            TI action_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(action_i < per_agent_action_dim){
                T sum = 0;
                TI n_agents = action_dim / per_agent_action_dim;
                for(TI agent_i = 0; agent_i < n_agents; agent_i++){
                    TI col = agent_i * per_agent_action_dim + action_i;
                    for(TI batch_i = 0; batch_i < batch_size; batch_i++){
                        sum += get(d_action_log_prob_d_action_log_std, batch_i, col);
                    }
                }
                sum += entropy_grad;
                set(log_std_gradient, 0, action_i, get(log_std_gradient, 0, action_i) + sum);
            }
        }
    }
    // CUDA overload: GAE
    template <typename DEV_SPEC, typename DATASET_SPEC, typename BOOTSTRAP_SPEC, typename PPO_PARAMETERS>
    void estimate_generalized_advantages(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Matrix<BOOTSTRAP_SPEC>& bootstrap_values, PPO_PARAMETERS ppo_parameters_tag){
        static_assert(DATASET_SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS || !(PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS || PPO_PARAMETERS::IGNORE_TERMINATION), "PPO bootstrapping requires pre-reset observation capture");
        static_assert(BOOTSTRAP_SPEC::ROWS == DATASET_SPEC::STEPS_TOTAL && BOOTSTRAP_SPEC::COLS == 1, "GAE requires one bootstrap value per transition");
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using OPR_SPEC = typename DATASET_SPEC::SPEC;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OPR_SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::algorithms::ppo::cuda::estimate_generalized_advantages_kernel<<<grid, block, 0, device.stream>>>(tag_device, dataset, bootstrap_values, ppo_parameters_tag);
        check_status(device);
    }
    // CUDA overload: per-sample PPO loss gradient (with fused advantage normalization)
    template <typename DEV_SPEC, typename PPO_SPEC, typename BUFFERS_SPEC, typename BATCH_ACTIONS_SPEC, typename BATCH_ACTIONS_MEAN_SPEC, typename BATCH_ACTION_LOG_PROBS_SPEC, typename BATCH_ADVANTAGES_SPEC, typename RNG>
    void ppo_compute_actor_loss_gradient(devices::CUDA<DEV_SPEC>& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::algorithms::ppo::Buffers<BUFFERS_SPEC>& ppo_buffers, Matrix<BATCH_ACTIONS_SPEC>& batch_actions, Matrix<BATCH_ACTIONS_MEAN_SPEC>& batch_actions_mean, Matrix<BATCH_ACTION_LOG_PROBS_SPEC>& batch_action_log_probs, Matrix<BATCH_ADVANTAGES_SPEC>& batch_advantages, typename PPO_SPEC::TYPE_POLICY::DEFAULT& policy_kl_divergence, typename PPO_SPEC::TYPE_POLICY::DEFAULT& batch_policy_kl_divergence, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
        constexpr TI ACTION_DIM = PPO_SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI N_AGENTS = PPO_SPEC::ENVIRONMENT::N_AGENTS;
        constexpr TI PER_AGENT_ACTION_DIM = ACTION_DIM / N_AGENTS;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto& last_layer = get_last_layer(ppo.actor);
        auto log_std = matrix_view(device, last_layer.log_std.parameters);
        auto log_std_grad = matrix_view(device, last_layer.log_std.gradient);
        // Single block: shared memory reduction computes mean/std, then all threads process their samples
        // Each thread handles ceil(BATCH_SIZE/BLOCKSIZE) samples for both reduction and per-sample computation
        constexpr TI KERNEL_BLOCKSIZE = BATCH_SIZE < 1024 ? (BATCH_SIZE <= 32 ? 32 : (BATCH_SIZE <= 64 ? 64 : (BATCH_SIZE <= 128 ? 128 : (BATCH_SIZE <= 256 ? 256 : (BATCH_SIZE <= 512 ? 512 : 1024))))) : 1024;
        constexpr TI SHARED_MEM_SIZE = 2 * KERNEL_BLOCKSIZE * sizeof(T);
        static_assert(SHARED_MEM_SIZE <= 48 * 1024, "ppo_per_sample_kernel shared memory exceeds 48KB limit");
        rl::algorithms::ppo::cuda::ppo_per_sample_kernel<decltype(tag_device), PPO_SPEC, BUFFERS_SPEC, BUFFERS_SPEC><<<1, KERNEL_BLOCKSIZE, SHARED_MEM_SIZE, device.stream>>>(tag_device, ppo_buffers, batch_advantages, batch_actions, batch_action_log_probs, log_std);
        check_status(device);
        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
            T entropy_grad = -(T)PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT * (T)N_AGENTS;
            constexpr TI REDUCE_BLOCKSIZE = 32;
            constexpr TI REDUCE_N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(PER_AGENT_ACTION_DIM, REDUCE_BLOCKSIZE);
            rl::algorithms::ppo::cuda::reduce_log_std_gradient_kernel<<<REDUCE_N_BLOCKS, REDUCE_BLOCKSIZE, 0, device.stream>>>(tag_device, ppo_buffers.d_action_log_prob_d_action_log_std, log_std_grad, (TI)BATCH_SIZE, (TI)ACTION_DIM, (TI)PER_AGENT_ACTION_DIM, entropy_grad);
            check_status(device);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
