#ifndef LAYER_IN_C_DEVICES_ESP32_H
#define LAYER_IN_C_DEVICES_ESP32_H
#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include <cstddef>
namespace layer_in_c::devices{
    namespace esp32{
        enum class Hardware{
            DEFAULT,
            ORIG,
            C3
        };
        template <typename T_MATH, typename T_RANDOM, typename T_LOGGING, Hardware T_HARDWARE>
        struct Specification{
            using MATH = T_MATH;
            using RANDOM = T_RANDOM;
            using LOGGING = T_LOGGING;
            static constexpr Hardware HARDWARE = T_HARDWARE;
        };
        struct Base{
            static constexpr DeviceId DEVICE_ID = DeviceId::ESP32;
            using index_t = size_t;
        };
    }
    namespace math{
        struct ESP32: esp32::Base{
            static constexpr Type TYPE = Type::math;
        };
    }
    namespace random{
        struct ESP32: esp32::Base{
            static constexpr Type TYPE = Type::random;
        };
    }
    namespace logging{
        struct ESP32: esp32::Base{
            static constexpr Type TYPE = Type::logging;
        };
    }
    template <typename T_SPEC>
    struct ESP32: Device<T_SPEC>, esp32::Base{
        template <typename OTHER_DEVICE>
        static constexpr bool compatible = OTHER_DEVICE::DEVICE == DeviceId::ESP32;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING* logger = nullptr;
#ifdef LAYER_IN_C_DEBUG_CONTAINER_COUNT_MALLOC
        index_t malloc_counter = 0;
#endif
    };
    namespace esp32{
        template <typename T_SPEC>
        struct Generic: ESP32<T_SPEC>{};
        template <typename T_SPEC>
        struct DSP: ESP32<T_SPEC>{};
        template <typename T_SPEC>
        struct OPT: ESP32<T_SPEC>{};
    }

    template <esp32::Hardware T_HARDWARE = esp32::Hardware::DEFAULT>
    using DefaultESP32Specification = esp32::Specification<math::ESP32, random::ESP32, logging::ESP32, T_HARDWARE>;
    using DefaultESP32 = esp32::OPT<DefaultESP32Specification<>>;
}

namespace layer_in_c{
#ifdef LAYER_IN_C_DEBUG_CONTAINER_COUNT_MALLOC
    template <typename DEV_SPEC, typename TI>
    void count_malloc(devices::ESP32<DEV_SPEC>& device, TI size){
        device.malloc_counter += size;
    }
#endif
}


#endif