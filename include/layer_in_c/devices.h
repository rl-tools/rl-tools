#ifndef LAYER_IN_C_DEVICES_H
#define LAYER_IN_C_DEVICES_H

namespace layer_in_c {
    namespace devices {
        enum class Device {
            Generic,
            Dummy,
            CPU
        };
        struct Generic{
            static constexpr Device DEVICE = Device::Generic;
        };
        struct CPU{
            static constexpr Device DEVICE = Device::CPU;
        };
        struct Dummy{ // for testing the dependencylessness
            static constexpr Device DEVICE = Device::Dummy;
        };
    }
}

#endif