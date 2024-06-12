

#include "../../../utils/polyak/operations_cuda.h"
#include "../../../rl/algorithms/sac/sac.h"
#include "../../../rl/components/off_policy_runner/off_policy_runner.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC,typename ALPHA_PARAMETER>
    __global__
    void target_actions_kernel(devices::CUDA<DEV_SPEC> device, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>> batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC> training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC> next_action_log_probs, ALPHA_PARAMETER log_alpha) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, get(log_alpha.parameters, 0, 0));
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(batch_step_i < BATCH_SIZE){
            target_actions_per_sample(device, batch, training_buffers, next_action_log_probs, alpha, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename BATCH_SPEC, typename SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename ALPHA_PARAMETER>
    void target_actions(devices::CUDA<DEV_SPEC>& device, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, ALPHA_PARAMETER& log_alpha) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        target_actions_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, batch, training_buffers, next_action_log_probs, log_alpha);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC>
    __global__
    void min_value_d_output_kernel(devices::CUDA<DEV_SPEC> device, rl::algorithms::sac::ActorCritic<SPEC> actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC> training_buffers){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
//        curandState rng_state;
//        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            min_value_d_output_per_sample(device, actor_critic, training_buffers, batch_step_i);
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void min_value_d_output(devices::CUDA<DEV_SPEC>& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::sac::ActorTrainingBuffers<SPEC>::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        min_value_d_output_kernel<<<bias_grid, bias_block, 0, device.stream>>>(device, actor_critic, training_buffers);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "operations_generic.h"
