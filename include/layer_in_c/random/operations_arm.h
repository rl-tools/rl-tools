#ifndef LAYER_IN_C_UTILS_RANDOM_OPERATIONS_ARM_H
#define LAYER_IN_C_UTILS_RANDOM_OPERATIONS_ARM_H


#include <layer_in_c/utils/generic/typing.h>

namespace layer_in_c::random{
   devices::random::ARM::index_t default_engine(const devices::random::ARM& dev, devices::random::ARM::index_t seed = 1){
       return 0b10101010101010101010101010101010 + seed;
   };
   constexpr devices::random::ARM::index_t next_max(const devices::random::ARM& dev){
       return devices::random::ARM::MAX_INDEX;
   }
   template<typename RNG>
   void next(const devices::random::ARM& dev, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       rng ^= (rng << 13);
       rng ^= (rng >> 17);
       rng ^= (rng << 5);
   }

   template<typename T, typename RNG>
   T uniform_int_distribution(const devices::random::ARM& dev, T low, T high, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       using TI = devices::random::ARM::index_t;
       TI range = static_cast<devices::random::ARM::index_t>(high - low) + 1;
       next(dev, rng);
       TI r = rng % range;
       return static_cast<T>(r) + low;
   }
   template<typename T, typename RNG>
   T uniform_real_distribution(const devices::random::ARM& dev, T low, T high, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       static_assert(utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, float>);
       next(dev, rng);
       return (rng / static_cast<T>(next_max(dev))) * (high - low) + low;
   }
   template<typename T, typename RNG>
   T normal_distribution(const devices::random::ARM& dev, T mean, T std, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       static_assert(utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, float>);
       next(dev, rng);
       T u1 = rng / static_cast<T>(next_max(dev));
       next(dev, rng);
       T u2 = rng / static_cast<T>(next_max(dev));
       T x = math::sqrt(devices::math::ARM(), -2.0 * math::log(devices::math::ARM(), u1));
       T y = 2.0 * math::PI<T> * u2;
       T z = x * math::cos(devices::math::ARM(), y);
       return z * std + mean;
   }
}

#endif
