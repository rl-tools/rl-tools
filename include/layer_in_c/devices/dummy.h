#ifndef LAYER_IN_C_DEVICES_DUMMY_H
#define LAYER_IN_C_DEVICES_DUMMY_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"
namespace layer_in_c::devices{
    namespace math{
        struct Dummy{
            static constexpr Device DEVICE = Device::Dummy;
            static constexpr Domain DOMAIN = Domain::math;
        };
    }
    namespace random{
        struct Dummy{
            static constexpr Device DEVICE = Device::Dummy;
            static constexpr Domain DOMAIN = Domain::random;
        };
    }
    namespace logging{
        struct Dummy{
            static constexpr Device DEVICE = Device::Dummy;
            static constexpr Domain DOMAIN = Domain::logging;
        };
        struct Dummy_WANDB{
            static constexpr Device DEVICE = Device::Dummy;
            static constexpr Domain DOMAIN = Domain::logging;
            int counter = 0;
        };
    }
    template <typename T_SPEC>
    struct Dummy{
        using SPEC = T_SPEC;
        typename SPEC::LOGGING& logger;
        explicit Dummy(typename SPEC::LOGGING& logger) : logger(logger) {}
    };
    struct DefaultDummySpecification{
        using MATH = math::Dummy;
        using RANDOM = random::Dummy;
        using LOGGING = logging::Dummy;
    };
    using DefaultDummy = Dummy<DefaultDummySpecification>;
}

#endif
