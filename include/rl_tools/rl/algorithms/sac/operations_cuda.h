

#include "../../../utils/polyak/operations_cuda.h"
#include "../../../rl/algorithms/sac/sac.h"
#include "operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename SPEC, typename ALPHA_PARAMETER>
    __global__
    void target_actions_kernel(devices::CUDA<DEV_SPEC> device, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>> batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers, ALPHA_PARAMETER log_alpha) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, get(log_alpha.parameters, 0, 0));
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(batch_step_i < BATCH_SIZE){
            target_actions_per_sample(device, batch, training_buffers, alpha, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename SPEC, typename ALPHA_PARAMETER>
    void target_actions(devices::CUDA<DEV_SPEC>& device, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>> batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers, ALPHA_PARAMETER& log_alpha) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        target_actions_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, batch, training_buffers, log_alpha);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename ACTION_NOISE_SPEC>
    __global__
    void sample_actions_critic_kernel(devices::CUDA<DEV_SPEC> device, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers, Matrix<ACTION_NOISE_SPEC> action_noise) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
//        curandState rng_state;
//        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            sample_actions_critic_per_sample(device, training_buffers, action_noise, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename ACTION_NOISE_SPEC>
    void sample_actions_critic(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::sac::ActorTrainingBuffers<SPEC>::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        sample_actions_critic_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, training_buffers, action_noise);
        check_status(device);
    }

    template <typename DEV_SPEC, typename OUTPUT_SPEC, typename SPEC, typename ACTION_NOISE_SPEC>
    __global__
    void sample_actions_actor_kernel(devices::CUDA<DEV_SPEC> device, Matrix<OUTPUT_SPEC> output, rl::algorithms::sac::ActorTrainingBuffers<SPEC> training_buffers, Matrix<ACTION_NOISE_SPEC> action_noise) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
//        curandState rng_state;
//        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            sample_actions_actor_per_sample(device, output, training_buffers, action_noise, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename ACTION_NOISE_SPEC>
    void sample_actions_actor(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::sac::ActorTrainingBuffers<SPEC>::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        auto actions_full = output(actor_critic.actor);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        sample_actions_actor_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, actions_full, training_buffers, action_noise);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename ACTION_SPEC, typename CRITIC_1_OUTPUT_SPEC, typename CRITIC_2_OUTPUT_SPEC, typename T_ALPHA>
    __global__
    void d_action_d_action_distribution_kernel(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::ActorTrainingBuffers<SPEC> training_buffers, Matrix<ACTION_SPEC> actions, Matrix<CRITIC_1_OUTPUT_SPEC> critic_1_output, Matrix<CRITIC_2_OUTPUT_SPEC> critic_2_output, T_ALPHA log_alpha){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, get(log_alpha.parameters, 0, 0));

        if(batch_step_i < BATCH_SIZE){
            T d_alpha_incremental = d_action_d_action_distribution_per_sample(device, training_buffers, actions, critic_1_output, critic_2_output, alpha, batch_step_i);
            atomicAdd(log_alpha.gradient._data, d_alpha_incremental);
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void d_action_d_action_distribution(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::sac::ActorTrainingBuffers<SPEC>::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        auto actions_full = output(actor_critic.actor);
//        typename decltype(actor_critic.log_alpha)::T zero = 0;
//        cudaMemcpy(actor_critic.log_alpha.gradient._data, &zero, sizeof(T), cudaMemcpyHostToDevice);
        d_action_d_action_distribution_kernel<<<bias_grid, bias_block, 0, device.stream>>>(device, training_buffers, actions_full, output(actor_critic.critic_1), output(actor_critic.critic_2), actor_critic.log_alpha);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
