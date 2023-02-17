#ifndef LAYER_IN_C_DEVICES_DUMMY_H
#define LAYER_IN_C_DEVICES_DUMMY_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"
namespace layer_in_c::devices{
    namespace dummy{
        struct Base{
            static constexpr DeviceId DEVICE_ID = DeviceId::Dummy;
            using index_t = unsigned;
        };
    }
    namespace math{
        struct Dummy: dummy::Base{
            static constexpr Type TYPE = Type::math;
        };
    }
    namespace random{
        struct Dummy: dummy::Base{
            static constexpr Type TYPE = Type::random;
            using State = unsigned;
        };
    }
    namespace logging{
        struct Dummy: dummy::Base{
            static constexpr Type TYPE = Type::logging;
        };
    }
    template <typename T_SPEC>
    struct Dummy: dummy::Base{
        template <typename OTHER_DEVICE>
        static constexpr bool compatible = utils::typing::is_same_v<OTHER_DEVICE, Dummy<T_SPEC>>;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING* logger = nullptr;
    };
    struct DefaultDummySpecification{
        using MATH = math::Dummy;
        using RANDOM = random::Dummy;
        using LOGGING = logging::Dummy;
    };
    using DefaultDummy = Dummy<DefaultDummySpecification>;
}

#endif
