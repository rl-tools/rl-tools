#ifndef LAYER_IN_C_DEVICES_DEVICES_H
#define LAYER_IN_C_DEVICES_DEVICES_H

#include <layer_in_c/utils/generic/typing.h>

namespace layer_in_c {
    namespace devices {
        enum class Device {
            Generic,
            Dummy,
            CPU
        };
        enum class Domain {
            math,
            random,
            logging
        };
    }
}

#endif