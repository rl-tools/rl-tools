#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_RANDOM_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_RANDOM_OPERATIONS_CUDA_H


#include "../utils/generic/typing.h"
#include "operations_generic.h"

#include <curand_kernel.h>
#include "../containers/matrix/matrix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::random{
    namespace cuda{
        template <typename T_TI, T_TI T_NUM_RNGS, typename CURAND_TYPE=curandState>
        struct RNG{
            using TI = T_TI;
            static constexpr TI NUM_RNGS = T_NUM_RNGS;
            Matrix<matrix::Specification<CURAND_TYPE, TI, 1, NUM_RNGS, true>> states;
        };
        template <typename DEVICE, typename T_TI, T_TI T_NUM_RNGS, typename CURAND_TYPE>
        __global__
        void init_rng_kernel(DEVICE device, RNG<T_TI, T_NUM_RNGS, CURAND_TYPE> rng, typename DEVICE::index_t seed){
            auto i = threadIdx.x + blockIdx.x * blockDim.x;
            if(i < T_NUM_RNGS){
                curand_init(seed, i, 0, &get(rng.states, 0, i));
            }
        }

    }
    template <typename DEV_SPEC, typename T_TI = typename devices::CUDA<DEV_SPEC>::index_t, T_TI NUM_RNGS = 1024>
    auto default_engine(devices::CUDA<DEV_SPEC>& device, typename devices::CUDA<DEV_SPEC>::index_t seed = 1){
        cuda::RNG<T_TI, NUM_RNGS> rng;
        malloc(device, rng.states);
        constexpr T_TI BLOCKSIZE_COLS = 32;
        constexpr T_TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(NUM_RNGS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        cuda::init_rng_kernel<<<grid, block, 0, device.stream>>>(device, rng, seed);
        return rng;
    };
    template <typename DEV_SPEC, typename T_TI, T_TI T_NUM_RNGS, typename CURAND_TYPE=curandState>
    void free(devices::CUDA<DEV_SPEC>& device, cuda::RNG<T_TI, T_NUM_RNGS, CURAND_TYPE>& rng){
        free(device, rng.states);
    }

    template<typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT T uniform_real_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>, "Only float and double are supported");
        if constexpr(utils::typing::is_same_v<T, float>){
            return curand_uniform(&rng) * (high - low) + low;
        }
        else{
            if constexpr(utils::typing::is_same_v<T, double>){
                return curand_uniform_double(&rng) * (high - low) + low;
            }
        }
        return 0;
    }
    template<typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT T uniform_int_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        auto r = uniform_real_distribution(dev, (float)low, (float)high, rng);
        return (T)r;
    }
    namespace normal_distribution{
        template<typename T, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT T sample(const devices::random::CUDA& dev, T mean, T std, RNG& rng){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
            if constexpr(utils::typing::is_same_v<T, float>){
                return curand_normal(&rng) * std + mean;
            }
            else{
                if constexpr(utils::typing::is_same_v<T, double>){
                    return curand_normal_double(&rng) * std + mean;
                }
                else{
                    return 0;
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
