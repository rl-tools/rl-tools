#ifndef LAYER_IN_C_DEVICES_ARM_H
#define LAYER_IN_C_DEVICES_ARM_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include <cstddef>
namespace layer_in_c::devices{
    namespace arm{
        template <typename T_MATH, typename T_RANDOM, typename T_LOGGING>
        struct Specification{
            using MATH = T_MATH;
            using RANDOM = T_RANDOM;
            using LOGGING = T_LOGGING;
        };
        struct Base{
            static constexpr DeviceId DEVICE_ID = DeviceId::ARM;
            using index_t = size_t;
        };
    }
    namespace math{
        struct ARM: arm::Base{
            static constexpr Type TYPE = Type::math;
        };
    }
    namespace random{
        struct ARM: arm::Base{
            static constexpr Type TYPE = Type::random;
        };
    }
    namespace logging{
        struct ARM: arm::Base{
            static constexpr Type TYPE = Type::logging;
        };
    }
    template <typename T_SPEC>
    struct ARM: Device<T_SPEC>, arm::Base{
        template <typename OTHER_DEVICE>
        static constexpr bool compatible = OTHER_DEVICE::DEVICE == DeviceId::ARM;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING* logger = nullptr;
    };

    using DefaultARMSpecification = arm::Specification<math::ARM, random::ARM, logging::ARM>;
    using DefaultARM = ARM<DefaultARMSpecification>;
}

#endif
