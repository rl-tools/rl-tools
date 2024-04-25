#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_RANDOM_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_RANDOM_OPERATIONS_CPU_H


#include "../utils/generic/typing.h"
#include "operations_generic.h"

#include <random>
#include <limits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::random{
    auto default_engine(const devices::random::CPU& dev, devices::random::CPU::index_t seed = 0){
        return std::default_random_engine(seed+1);
    };

    template<typename T, typename RNG>
    T uniform_int_distribution(const devices::random::CPU& dev, T low, T high, RNG& rng){
        return std::uniform_int_distribution<T>(low, high)(rng);
    }
    template <typename TI, typename RNG>
    auto split(const devices::random::CPU& dev, TI split_id, RNG& rng){
        // this operation should not alter the state of rng
        RNG rng_copy = rng;
        TI new_seed = random::uniform_int_distribution(dev, std::numeric_limits<TI>::min(), std::numeric_limits<TI>::max(), rng_copy);
        return std::default_random_engine(new_seed + split_id);
    }

    template<typename T, typename RNG>
    T uniform_real_distribution(const devices::random::CPU& dev, T low, T high, RNG& rng){
        static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
        return std::uniform_real_distribution<T>(low, high)(rng);
    }
//    template<typename T, typename RNG>
//    const std::normal_distribution<T> standard_normal_distribution(0, 1);
    namespace normal_distribution{
        template<typename T, typename RNG>
        T sample(const devices::random::CPU& dev, T mean, T std, RNG& rng){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
            return std::normal_distribution<T>(mean, std)(rng);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
