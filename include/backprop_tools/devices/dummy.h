#include "../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_DEVICES_DUMMY_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_DEVICES_DUMMY_H

#include "../utils/generic/typing.h"
#include "devices.h"
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::devices{
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
    struct Dummy: Device<T_SPEC>, dummy::Base{
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
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif
