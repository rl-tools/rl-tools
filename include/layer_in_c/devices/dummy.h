#ifndef LAYER_IN_C_DEVICES_DUMMY_H
#define LAYER_IN_C_DEVICES_DUMMY_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"
namespace layer_in_c::devices{
    namespace dummy{
        struct Base{
            static constexpr Device DEVICE = Device::Dummy;
            using index_t = unsigned;
        };
    }
    namespace math{
        struct Dummy: dummy::Base{
            static constexpr Domain DOMAIN = Domain::math;
        };
    }
    namespace random{
        struct Dummy: dummy::Base{
            static constexpr Domain DOMAIN = Domain::random;
            using State = unsigned;
        };
    }
    namespace logging{
        struct Dummy: dummy::Base{
            static constexpr Domain DOMAIN = Domain::logging;
        };
    }
    template <typename T_SPEC>
    struct Dummy: dummy::Base{
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
