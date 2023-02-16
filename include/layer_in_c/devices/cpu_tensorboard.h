#ifndef LAYER_IN_C_DEVICES_CPU_TENSORBOARD_H
#define LAYER_IN_C_DEVICES_CPU_TENSORBOARD_H

#include "devices.h"
#include "cpu.h"

#include <tensorboard_logger.h>
#include <mutex>

namespace layer_in_c::devices{
    namespace logging{
        struct CPU_TENSORBOARD: logging::CPU{
            static constexpr Device DEVICE = Device::CPU_TENSORBOARD;
            static constexpr Type TYPE = Type::logging;
            index_t step = 0;
            TensorBoardLogger* tb = nullptr;
            std::mutex mutex;
        };
    }
}

#endif