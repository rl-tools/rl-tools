

#include "../../../utils/polyak/operations_cuda.h"
#include "../../../rl/algorithms/sac/sac.h"
#include "operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename SPEC, typename OUTPUT_SPEC, typename RNG>
    __global__
    void target_action_noise_kernel(devices::CUDA<DEV_SPEC>& device, const rl::algorithms::sac::ActorCritic<SPEC> actor_critic, Matrix<OUTPUT_SPEC> target_action_noise, RNG rng ) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        curandState rng_state;
        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            for(TI action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                set(target_action_noise, batch_step_i, action_i, math::clamp(device.math,
                    random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD, rng_state),
//                        curand_normal(&rng_state),
                        -(T)SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP,
                         (T)SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP
                ));
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename OUTPUT_SPEC, typename RNG>
    void target_action_noise(devices::CUDA<DEV_SPEC>& device, const rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, Matrix<OUTPUT_SPEC>& target_action_noise, RNG& rng ) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(OUTPUT_SPEC::ROWS == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(OUTPUT_SPEC::COLS == SPEC::ENVIRONMENT::ACTION_DIM);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        target_action_noise_kernel<DEV_SPEC, SPEC, OUTPUT_SPEC, RNG><<<bias_grid, bias_block>>>(device, actor_critic, target_action_noise, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC>
    __global__
    void noisy_next_actions_kernel(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(batch_step_i < BATCH_SIZE){
            for(TI action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                T noisy_next_action = get(training_buffers.next_actions, batch_step_i, action_i) + get(training_buffers.target_next_action_noise, batch_step_i, action_i);
                noisy_next_action = math::clamp<T>(device.math, noisy_next_action, -1, 1);
                set(training_buffers.next_actions, batch_step_i, action_i, noisy_next_action);
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void noisy_next_actions(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        noisy_next_actions_kernel<DEV_SPEC, SPEC><<<bias_grid, bias_block>>>(device, training_buffers);
        check_status(device);
    }

    template <typename DEV_SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename SPEC, typename ALPHA_PARAMETER>
    __global__
    void target_actions_kernel(devices::CUDA<DEV_SPEC>& device, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>> batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers, ALPHA_PARAMETER log_alpha) {
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
        target_actions_kernel<<<bias_grid, bias_block>>>(device, batch, training_buffers, log_alpha);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename RNG>
    __global__
    void sample_actions_critic_kernel(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers, RNG rng) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        curandState rng_state;
        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            sample_actions_critic_per_sample(device, training_buffers, rng_state, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename RNG>
    void sample_actions_critic(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, RNG& rng) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::sac::ActorTrainingBuffers<SPEC>::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        sample_actions_critic_kernel<<<bias_grid, bias_block>>>(device, training_buffers, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename OUTPUT_SPEC, typename SPEC, typename RNG>
    __global__
    void sample_actions_actor_kernel(devices::CUDA<DEV_SPEC>& device, Matrix<OUTPUT_SPEC> output, rl::algorithms::sac::ActorTrainingBuffers<SPEC> training_buffers, RNG rng) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        curandState rng_state;
        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            sample_actions_actor_per_sample(device, output, training_buffers, rng_state, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename RNG>
    void sample_actions_actor(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers, RNG& rng) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::sac::ActorTrainingBuffers<SPEC>::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        auto actions_full = output(actor_critic.actor);
        sample_actions_actor_kernel<<<bias_grid, bias_block>>>(device, actions_full, training_buffers, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename ACTION_SPEC, typename CRITIC_1_OUTPUT_SPEC, typename CRITIC_2_OUTPUT_SPEC, typename T_ALPHA>
    __global__
    void d_action_d_action_distribution_kernel(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::ActorTrainingBuffers<SPEC> training_buffers, Matrix<ACTION_SPEC> actions, Matrix<CRITIC_1_OUTPUT_SPEC> critic_1_output, Matrix<CRITIC_2_OUTPUT_SPEC> critic_2_output, T_ALPHA log_alpha){
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
        d_action_d_action_distribution_kernel<<<bias_grid, bias_block>>>(device, training_buffers, actions_full, output(actor_critic.critic_1), output(actor_critic.critic_2), actor_critic.log_alpha);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
