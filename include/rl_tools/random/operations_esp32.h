#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_RANDOM_OPERATIONS_ESP32_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_RANDOM_OPERATIONS_ESP32_H


#include "operations_generic.h"
#include "../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::random{
   devices::random::ESP32::index_t default_engine(const devices::random::ESP32& dev, devices::random::ESP32::index_t seed = 1){
       return 0b10101010101010101010101010101010 + seed;
   };
   template<typename RNG>
   void next(const devices::random::ESP32& dev, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ESP32::index_t>);
       rng ^= (rng << 13);
       rng ^= (rng >> 17);
       rng ^= (rng << 5);
   }

   template<typename T, typename RNG>
   T uniform_int_distribution(const devices::random::ESP32& dev, T low, T high, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ESP32::index_t>);
       using TI = devices::random::ESP32::index_t;
       TI range = static_cast<devices::random::ESP32::index_t>(high - low) + 1;
       next(dev, rng);
       TI r = rng % range;
       return static_cast<T>(r) + low;
   }
   template<typename T, typename RNG>
   T uniform_real_distribution(const devices::random::ESP32& dev, T low, T high, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ESP32::index_t>);
       static_assert(utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, float>);
       next(dev, rng);
       return (rng / static_cast<T>(std::numeric_limits<RNG>::max())) * (high - low) + low;
   }
   template<typename T, typename RNG>
   T normal_distribution(const devices::random::ESP32& dev, T mean, T std, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ESP32::index_t>);
       static_assert(utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, float>);
       next(dev, rng);
       T u1 = rng / static_cast<T>(std::numeric_limits<RNG>::max());
       next(dev, rng);
       T u2 = rng / static_cast<T>(std::numeric_limits<RNG>::max());
       T x = math::sqrt(devices::math::ESP32(), -2.0 * math::log(devices::math::ESP32(), u1));
       T y = 2.0 * math::PI<T> * u2;
       T z = x * math::cos(devices::math::ESP32(), y);
       return z * std + mean;
   }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
