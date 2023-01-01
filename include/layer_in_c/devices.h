#ifndef LAYER_IN_C_DEVICES_H
#define LAYER_IN_C_DEVICES_H

namespace layer_in_c {
    namespace devices {
        enum class Device {
            Generic,
            CPU
        };
        struct Generic{
            static constexpr Device DEVICE = Device::Generic;
        };
        struct CPU{
            static constexpr Device DEVICE = Device::CPU;
        };
    }
}

#endif