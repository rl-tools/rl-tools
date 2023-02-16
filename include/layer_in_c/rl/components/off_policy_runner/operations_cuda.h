#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CUDA_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include <layer_in_c/devices/dummy.h>

namespace layer_in_c{
    namespace rl::components::off_policy_runner{
        template <typename DEVICE, typename RUNNER_SPEC, typename BATCH_SPEC, typename RNG, bool DETERMINISTIC = false>
        __global__
        void gather_batch_kernel(const rl::components::OffPolicyRunner<RUNNER_SPEC>* runner, rl::components::off_policy_runner::Batch<BATCH_SPEC>* batch, RNG rng) {
            using BATCH = rl::components::off_policy_runner::Batch<BATCH_SPEC>;
            using T = typename RUNNER_SPEC::T;
            using TI = typename RUNNER_SPEC::TI;
            static_assert(decltype(batch->observations)::COL_PITCH == 1);
            static_assert(decltype(batch->actions)::COL_PITCH == 1);
            static_assert(decltype(batch->next_observations)::COL_PITCH == 1);
            // rng
            constexpr auto rand_max = 4294967296ULL;
            static_assert(rand_max / RUNNER_SPEC::N_ENVIRONMENTS > 100);
            static_assert(rand_max / RUNNER_SPEC::REPLAY_BUFFER_CAPACITY > 100); // so that the distribution is not skewed too much towards the remainder values

            curandState rng_state;

            TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
            curand_init(rng, batch_step_i, 0, &rng_state);

            typename DEVICE::index_t env_i = DETERMINISTIC ? 0 : curand(&rng_state) % RUNNER_SPEC::N_ENVIRONMENTS;
            auto& replay_buffer = runner->replay_buffers[env_i];
            static_assert(decltype(replay_buffer.observations)::COL_PITCH == 1);
            static_assert(decltype(replay_buffer.actions)::COL_PITCH == 1);
            static_assert(decltype(replay_buffer.next_observations)::COL_PITCH == 1);

            if(batch_step_i < BATCH_SPEC::BATCH_SIZE){
                set(batch->observations, batch_step_i, 0, get(replay_buffer.observations, batch_step_i, 0));
                typename DEVICE::index_t sample_index_max = (replay_buffer.full ? RUNNER_SPEC::REPLAY_BUFFER_CAPACITY : replay_buffer.position);
#ifdef LAYER_IN_C_DEBUG_DEVICE_CUDA_CHECK_BOUNDS
                if(sample_index_max < 1){
                    printf("sample_index_max: %d\n", sample_index_max);
                    assert(sample_index_max > 0);
                }
#endif
                typename DEVICE::index_t sample_index = DETERMINISTIC ? batch_step_i : curand(&rng_state) % sample_index_max;

                // todo: replace with smarter, coalesced copy
                for(typename DEVICE::index_t i = 0; i < BATCH::OBSERVATION_DIM; i++){
                    set(batch->observations, batch_step_i, i, get(replay_buffer.observations, sample_index, i));
                    set(batch->next_observations, batch_step_i, i, get(replay_buffer.next_observations, sample_index, i));
                }
                for(typename DEVICE::index_t i = 0; i < BATCH::ACTION_DIM; i++){
                    set(batch->actions, batch_step_i, i, get(replay_buffer.actions, sample_index, i));
                }

                set(batch->rewards, 0, batch_step_i, get(replay_buffer.rewards, sample_index, 0));
                set(batch->terminated, 0, batch_step_i, get(replay_buffer.terminated, sample_index, 0));
                set(batch->truncated, 0, batch_step_i, get(replay_buffer.truncated,  sample_index, 0));
            }
        }

        template<typename DEVICE, typename SPEC, typename RNG>
        __global__
        void prologue_kernel(DEVICE device, rl::components::OffPolicyRunner<SPEC>* runner, RNG rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            curandState rng_state;
            curand_init(rng, env_i, 0, &rng_state);
            if(env_i < SPEC::N_ENVIRONMENTS){
                prologue_per_env(device, runner, rng_state, env_i);
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename RNG>
        void prologue(devices::CUDA<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>* runner, RNG &rng) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            constexpr TI BLOCKSIZE_COLS = 32;
            constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE_COLS);
            dim3 grid(N_BLOCKS_COLS);
            dim3 block(BLOCKSIZE_COLS);
            prologue_kernel<<<grid, block>>>(device, runner, rng);
            check_status(device);
        }
        template<typename DEV_SPEC, typename SPEC, typename POLICY>
        void interlude(devices::CUDA<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, POLICY &policy, typename POLICY::template Buffers<SPEC::N_ENVIRONMENTS>& policy_eval_buffers) {
            // runner struct should be on the CPU while its buffers should be on the GPU
            evaluate(device, policy, runner.buffers.observations, runner.buffers.actions, policy_eval_buffers);
        }

        template<typename DEVICE, typename SPEC, typename RNG>
        __global__
        void epilogue_kernel(DEVICE device, rl::components::OffPolicyRunner<SPEC>* runner, RNG rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;

            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            curandState rng_state;
            curand_init(rng, env_i, 0, &rng_state);
            if(env_i < SPEC::N_ENVIRONMENTS){
                epilogue_per_env(device, runner, rng_state, env_i);
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename RNG>
        void epilogue(devices::CUDA<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>* runner, RNG& rng) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            constexpr TI BLOCKSIZE_COLS = 32;
            constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE_COLS);
            dim3 grid(N_BLOCKS_COLS);
            dim3 block(BLOCKSIZE_COLS);
            epilogue_kernel<<<grid, block>>>(device, runner, rng);
            check_status(device);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG, bool DETERMINISTIC = false>
    void gather_batch(devices::CUDA<DEV_SPEC>& device, const rl::components::OffPolicyRunner<SPEC>* runner, rl::components::off_policy_runner::Batch<BATCH_SPEC>* batch, RNG rng){
        static_assert(utils::typing::is_same_v<RNG, random::cuda::RNG>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(utils::typing::is_same_v<SPEC, typename BATCH_SPEC::SPEC>);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr typename DEVICE::index_t BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        rl::components::off_policy_runner::gather_batch_kernel<DEVICE, SPEC, BATCH_SPEC, RNG, DETERMINISTIC><<<bias_grid, bias_block>>>(runner, batch, rng);
        check_status(device);
    }
}


#endif
